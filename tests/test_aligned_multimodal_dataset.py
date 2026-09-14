"""Tests for the CPU-only injectable aligned multimodal Dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

import emotion_model.data.dataset as dataset_module
from emotion_model.common import LabelProtocol, derive_quadrant_labels
from emotion_model.data import (
    AlignedMultimodalDataset,
    AlignedMultimodalSample,
    EmotionScores,
    IndexSpan,
    MultimodalWindowRecord,
    PhysioArtifactPolicy,
    PhysioChannelSourceRef,
    SourceAdapterError,
    SpeechSourceData,
    TimedSourceRef,
    TimeInterval,
)
from emotion_model.physiology import (
    ChannelwiseZScoreNormalizer,
    NormalizationFitScope,
    NormalizationKey,
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)


class MemorySpeechAdapter:
    """Return in-memory speech sources while recording exact calls."""

    def __init__(self, sources: dict[str, object]) -> None:
        self.sources = sources
        self.calls: list[TimedSourceRef] = []

    def load_speech(self, source: TimedSourceRef) -> object:
        """Return or raise the configured source object."""
        self.calls.append(source)
        result = self.sources[source.source_id]
        if isinstance(result, Exception):
            raise result
        return result


class MemoryPhysioAdapter:
    """Return in-memory channel windows while recording source/spec calls."""

    def __init__(self, sources: dict[str, object]) -> None:
        self.sources = sources
        self.calls: list[tuple[PhysioChannelSourceRef, PhysioChannelSpec]] = []

    def load_physio(
        self,
        source: PhysioChannelSourceRef,
        spec: PhysioChannelSpec,
    ) -> object:
        """Return or raise the configured channel object."""
        self.calls.append((source, spec))
        result = self.sources[source.source.source_id]
        if isinstance(result, Exception):
            raise result
        return result


def _spec(
    name: str = "eda",
    *,
    rate: float = 1.0,
    kind: PhysioSignalKind = PhysioSignalKind.EDA,
) -> PhysioChannelSpec:
    """Create one explicit channel specification."""
    unit = "uS" if kind is PhysioSignalKind.EDA else "degC"
    return PhysioChannelSpec(name, kind, rate, unit)


def _speech_ref(
    source_id: str = "speech-key",
    *,
    start: float = 0.0,
    end: float = 10.0,
) -> TimedSourceRef:
    """Create an abstract speech reference."""
    return TimedSourceRef(source_id, TimeInterval(start, end))


def _physio_ref(
    name: str = "eda",
    source_id: str = "physio-key",
    *,
    start: float = 0.0,
    end: float = 10.0,
) -> PhysioChannelSourceRef:
    """Create an abstract physiology reference."""
    return PhysioChannelSourceRef(
        name,
        TimedSourceRef(source_id, TimeInterval(start, end)),
    )


def _record(
    sample_id: str = "sample-1",
    *,
    participant_id: str = "p1",
    session_id: str = "s1",
    start: float = 1.0,
    end: float = 3.0,
    scores: EmotionScores | None = None,
    speech: bool = True,
    physio_sources: tuple[PhysioChannelSourceRef, ...] = (),
) -> MultimodalWindowRecord:
    """Create a valid record with caller-selected modality availability."""
    return MultimodalWindowRecord(
        sample_id,
        participant_id,
        session_id,
        TimeInterval(start, end),
        EmotionScores(2, 4) if scores is None else scores,
        (
            _speech_ref(start=min(0.0, start), end=max(10.0, end))
            if speech
            else None
        ),
        physio_sources,
    )


def _speech_data(
    *,
    rate: int = 2,
    start: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> SpeechSourceData:
    """Create a regular waveform whose values reveal extracted indices."""
    return SpeechSourceData(
        torch.arange(12, dtype=dtype),
        rate,
        start,
    )


def _physio_window(
    spec: PhysioChannelSpec,
    values: Tensor,
    timestamps: Tensor,
    mask: Tensor | None = None,
) -> PhysioChannelWindow:
    """Create an explicit timestamp channel window."""
    if mask is None:
        mask = torch.ones(values.shape, dtype=torch.bool)
    return PhysioChannelWindow(spec, values, mask, timestamps)


def _speech_only_dataset(
    record: MultimodalWindowRecord | None = None,
    *,
    protocol: LabelProtocol = LabelProtocol.OFFICIAL_MID_HIGH,
    data: SpeechSourceData | object | None = None,
    output_dtype: torch.dtype = torch.float32,
    label_ignore_index: int = -100,
) -> tuple[AlignedMultimodalDataset, MemorySpeechAdapter]:
    """Create a simple speech-only Dataset and its recording adapter."""
    selected_record = _record() if record is None else record
    adapter = MemorySpeechAdapter(
        {"speech-key": _speech_data() if data is None else data}
    )
    dataset = AlignedMultimodalDataset(
        (selected_record,),
        (),
        label_protocol=protocol,
        speech_adapter=adapter,
        required_speech_sample_rate_hz=2,
        output_dtype=output_dtype,
        label_ignore_index=label_ignore_index,
    )
    return dataset, adapter


def test_dataset_construction_is_lazy_and_preserves_record_order() -> None:
    """Construct mixed-modality data and query length without adapter calls."""
    spec = _spec()
    speech_record = _record("speech", speech=True)
    physio_record = _record(
        "physio",
        speech=False,
        physio_sources=(_physio_ref(),),
    )
    speech_adapter = MemorySpeechAdapter({"speech-key": _speech_data()})
    physio_adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                spec,
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
            )
        }
    )

    dataset = AlignedMultimodalDataset(
        [speech_record, physio_record],
        [spec],
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=speech_adapter,
        physio_adapter=physio_adapter,
        physio_target_sample_rate_hz=1.0,
    )

    assert len(dataset) == 2
    assert dataset.records == (speech_record, physio_record)
    assert dataset.channel_specs == (spec,)
    assert speech_adapter.calls == []
    assert physio_adapter.calls == []


def test_dataset_can_disable_each_unselected_modality_without_loading_it() -> None:
    """Expose one branch only and never call the other source adapter."""

    spec = _spec()
    record = _record(physio_sources=(_physio_ref(),))
    speech_adapter = MemorySpeechAdapter({"speech-key": _speech_data()})
    physiology_adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                spec,
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
            )
        }
    )
    speech_only = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        speech_adapter=speech_adapter,
        required_speech_sample_rate_hz=2,
        enable_speech=True,
        enable_physiology=False,
    )[0]
    assert speech_only.speech_available
    assert not speech_only.physiology_available
    assert len(speech_adapter.calls) == 1
    assert physiology_adapter.calls == []

    physiology_only = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        physio_adapter=physiology_adapter,
        physio_target_sample_rate_hz=1.0,
        enable_speech=False,
        enable_physiology=True,
    )[0]
    assert not physiology_only.speech_available
    assert physiology_only.physiology_available
    assert len(speech_adapter.calls) == 1
    assert len(physiology_adapter.calls) == 1


def test_dataset_rejects_disabling_both_modalities() -> None:
    """Require at least one selected input branch for every Dataset."""

    with pytest.raises(ValueError, match="at least one modality"):
        AlignedMultimodalDataset(
            (_record(),),
            (),
            label_protocol=LabelProtocol.OFFICIAL_MID_HIGH,
            enable_speech=False,
            enable_physiology=False,
        )


@pytest.mark.parametrize(
    ("records", "exception"),
    [
        ((), ValueError),
        ("sample", TypeError),
        ((object(),), TypeError),
    ],
)
def test_dataset_rejects_invalid_record_sequences(
    records: object,
    exception: type[Exception],
) -> None:
    """Require a non-string, non-empty sequence of manifest records."""
    with pytest.raises(exception):
        AlignedMultimodalDataset(  # type: ignore[arg-type]
            records,
            (),
            label_protocol=LabelProtocol.MID_LOW,
        )


def test_dataset_rejects_duplicate_samples_specs_and_unknown_channels() -> None:
    """Validate unique samples/specs and every referenced channel name."""
    record = _record()
    adapter = MemorySpeechAdapter({"speech-key": _speech_data()})
    with pytest.raises(ValueError, match="sample_id"):
        AlignedMultimodalDataset(
            (record, record),
            (),
            label_protocol=LabelProtocol.MID_LOW,
            speech_adapter=adapter,
        )
    spec = _spec()
    with pytest.raises(ValueError, match="unique"):
        AlignedMultimodalDataset(
            (record,),
            (spec, spec),
            label_protocol=LabelProtocol.MID_LOW,
            speech_adapter=adapter,
            physio_target_sample_rate_hz=1.0,
        )
    physio_record = _record(
        speech=False,
        physio_sources=(_physio_ref("unknown"),),
    )
    with pytest.raises(ValueError, match="unknown"):
        AlignedMultimodalDataset(
            (physio_record,),
            (spec,),
            label_protocol=LabelProtocol.MID_LOW,
            physio_adapter=MemoryPhysioAdapter({}),
            physio_target_sample_rate_hz=1.0,
        )


def test_dataset_requires_adapters_only_for_declared_sources() -> None:
    """Require each adapter when a record declares its modality source."""
    with pytest.raises(ValueError, match="speech_adapter"):
        AlignedMultimodalDataset(
            (_record(),),
            (),
            label_protocol=LabelProtocol.MID_LOW,
        )
    spec = _spec()
    with pytest.raises(ValueError, match="physio_adapter"):
        AlignedMultimodalDataset(
            (_record(speech=False, physio_sources=(_physio_ref(),)),),
            (spec,),
            label_protocol=LabelProtocol.MID_LOW,
            physio_target_sample_rate_hz=1.0,
        )
    speech_dataset, _ = _speech_only_dataset()
    assert len(speech_dataset) == 1
    physio_free_record = _record()
    dataset = AlignedMultimodalDataset(
        (physio_free_record,),
        (_spec(),),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=MemorySpeechAdapter({"speech-key": _speech_data()}),
        physio_target_sample_rate_hz=1.0,
    )
    assert len(dataset) == 1


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"label_protocol": "mid_low"}, TypeError),
        ({"required_speech_sample_rate_hz": True}, TypeError),
        ({"required_speech_sample_rate_hz": 0}, ValueError),
        ({"output_dtype": torch.float16}, ValueError),
        ({"label_ignore_index": True}, TypeError),
        ({"label_ignore_index": 0}, ValueError),
        ({"speech_adapter": object()}, TypeError),
    ],
)
def test_dataset_rejects_invalid_common_configuration(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    """Reject implicit label protocols, bad rates/dtypes/labels, and adapters."""
    arguments: dict[str, object] = {
        "label_protocol": LabelProtocol.MID_LOW,
        "speech_adapter": MemorySpeechAdapter({"speech-key": _speech_data()}),
        "required_speech_sample_rate_hz": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(exception):
        AlignedMultimodalDataset(
            (_record(),),
            (),
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("rate", "exception"),
    [
        (None, TypeError),
        (True, TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_dataset_requires_valid_physio_target_rate(
    rate: object,
    exception: type[Exception],
) -> None:
    """Require a finite positive common rate when channel specs exist."""
    with pytest.raises(exception):
        AlignedMultimodalDataset(
            (_record(),),
            (_spec(),),
            label_protocol=LabelProtocol.MID_LOW,
            speech_adapter=MemorySpeechAdapter({"speech-key": _speech_data()}),
            physio_target_sample_rate_hz=rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"physio_target_sample_rate_hz": 1.0},
        {"physio_adapter": MemoryPhysioAdapter({})},
        {"physio_filters": {}},
        {"artifact_policy": PhysioArtifactPolicy()},
        {"physio_normalizer": ChannelwiseZScoreNormalizer()},
    ],
)
def test_empty_channel_specs_reject_physiology_configuration(
    extra: dict[str, object],
) -> None:
    """Do not accept unused physiology behavior without channel semantics."""
    with pytest.raises(ValueError, match="channel_specs is empty"):
        AlignedMultimodalDataset(
            (_record(),),
            (),
            label_protocol=LabelProtocol.MID_LOW,
            speech_adapter=MemorySpeechAdapter({"speech-key": _speech_data()}),
            **extra,  # type: ignore[arg-type]
        )


def test_dataset_validates_filters_and_normalizer_resolver_pairing() -> None:
    """Reject unknown/non-callable filters and a resolver without a normalizer."""
    spec = _spec()
    common: dict[str, object] = {
        "label_protocol": LabelProtocol.MID_LOW,
        "speech_adapter": MemorySpeechAdapter({"speech-key": _speech_data()}),
        "physio_target_sample_rate_hz": 1.0,
    }
    with pytest.raises(ValueError, match="unknown"):
        AlignedMultimodalDataset(
            (_record(),),
            (spec,),
            physio_filters={"other": lambda values, mask, rate: values},
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="callable"):
        AlignedMultimodalDataset(
            (_record(),),
            (spec,),
            physio_filters={"eda": object()},  # type: ignore[dict-item]
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires"):
        AlignedMultimodalDataset(
            (_record(),),
            (spec,),
            normalization_key_resolver=lambda record, channel: NormalizationKey(
                channel.name
            ),
            **common,  # type: ignore[arg-type]
        )


def test_dataset_indexing_supports_negative_and_rejects_non_integer() -> None:
    """Follow integer sequence indexing without slice or bool support."""
    first = _record("first", start=0, end=1)
    last = _record("last", start=1, end=2)
    adapter = MemorySpeechAdapter(
        {"speech-key": SpeechSourceData(torch.arange(4.0), 2, 0)}
    )
    dataset = AlignedMultimodalDataset(
        (first, last),
        (),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=adapter,
        required_speech_sample_rate_hz=2,
    )

    assert dataset[0].record is first
    assert dataset[-1].record is last
    with pytest.raises(TypeError):
        dataset[True]  # type: ignore[index]
    with pytest.raises(TypeError):
        dataset[:]  # type: ignore[index]
    with pytest.raises(IndexError):
        dataset[2]
    with pytest.raises(IndexError):
        dataset[-3]


def test_getitem_loads_only_target_and_reloads_without_cache() -> None:
    """Call one target source per access and permit repeated adapter calls."""
    first = _record("first", start=0, end=1)
    second = _record("second", start=1, end=2)
    adapter = MemorySpeechAdapter(
        {"speech-key": SpeechSourceData(torch.arange(4.0), 2, 0)}
    )
    dataset = AlignedMultimodalDataset(
        (first, second),
        (),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=adapter,
        required_speech_sample_rate_hz=2,
    )

    dataset[1]
    assert len(adapter.calls) == 1
    dataset[1]
    assert len(adapter.calls) == 2


@pytest.mark.parametrize(
    ("protocol", "score", "expected"),
    [
        (LabelProtocol.OFFICIAL_MID_HIGH, 3.0, 1),
        (LabelProtocol.STRICT_DROP_MID, 3.0, -7),
        (LabelProtocol.MID_LOW, 3.0, 0),
    ],
)
def test_dataset_applies_explicit_public_label_protocol(
    protocol: LabelProtocol,
    score: float,
    expected: int,
) -> None:
    """Apply each public protocol while preserving the raw score."""
    record = _record(scores=EmotionScores(score, 4))
    adapter = MemorySpeechAdapter({"speech-key": _speech_data()})
    dataset = AlignedMultimodalDataset(
        (record,),
        (),
        label_protocol=protocol,
        speech_adapter=adapter,
        required_speech_sample_rate_hz=2,
        label_ignore_index=-7,
    )

    sample = dataset[0]

    assert sample.raw_arousal == score
    assert sample.arousal_label == expected
    assert sample.valence_label == 1
    expected_quadrant = derive_quadrant_labels(
        torch.tensor(expected),
        torch.tensor(1),
        ignore_index=-7,
    )
    assert sample.quadrant_label == int(expected_quadrant)


@pytest.mark.parametrize(
    ("scores", "expected_arousal", "expected_valence"),
    [
        (EmotionScores(3, 4), -9, 1),
        (EmotionScores(4, 3), 1, -9),
        (EmotionScores(3, 3), -9, -9),
    ],
)
def test_strict_mid_ignore_propagates_to_quadrant_without_dropping_sample(
    scores: EmotionScores,
    expected_arousal: int,
    expected_valence: int,
) -> None:
    """Return ignored labels and the original record instead of filtering it."""
    record = _record(scores=scores)
    dataset, _ = _speech_only_dataset(
        record,
        protocol=LabelProtocol.STRICT_DROP_MID,
        label_ignore_index=-9,
    )

    sample = dataset[0]

    assert sample.record is record
    assert sample.arousal_label == expected_arousal
    assert sample.valence_label == expected_valence
    assert sample.quadrant_label == -9


def test_speech_extraction_is_exact_half_open_new_contiguous_output() -> None:
    """Select manual sample times start<=t<end without amplitude changes."""
    data = _speech_data(dtype=torch.float64)
    record = _record(start=1.0, end=3.0)
    dataset, adapter = _speech_only_dataset(
        record,
        data=data,
        output_dtype=torch.float32,
    )
    original = data.waveform.clone()

    sample = dataset[0]

    sample_times = [index / 2 for index in range(data.waveform.numel())]
    expected_indices = [
        index for index, time in enumerate(sample_times) if 1.0 <= time < 3.0
    ]
    assert torch.equal(
        sample.speech_waveform,
        torch.tensor(expected_indices, dtype=torch.float32),
    )
    assert sample.speech_waveform is not None
    assert sample.speech_waveform.is_contiguous()
    assert sample.speech_waveform.data_ptr() != data.waveform.data_ptr()
    assert sample.speech_attention_mask is not None
    assert sample.speech_attention_mask.dtype == torch.bool
    assert bool(sample.speech_attention_mask.all())
    assert sample.speech_sample_rate_hz == 2
    assert sample.speech_available
    assert torch.equal(data.waveform, original)
    assert adapter.calls == [record.speech_source]


def test_speech_extraction_supports_nonzero_timeline_origin() -> None:
    """Map absolute or session-relative windows using the adapter origin."""
    record = _record(start=10.5, end=11.5)
    data = SpeechSourceData(torch.arange(6.0), 2, 10.0)
    dataset, _ = _speech_only_dataset(record, data=data)
    assert torch.equal(dataset[0].speech_waveform, torch.tensor([1.0, 2.0]))


def test_speech_window_with_no_sample_hit_is_rejected() -> None:
    """Reject a legal source interval that maps to an empty regular span."""
    record = MultimodalWindowRecord(
        "empty-speech-span",
        "p1",
        "s1",
        TimeInterval(0.1, 0.2),
        EmotionScores(2, 4),
        TimedSourceRef("speech-key", TimeInterval(0, 3)),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=MemorySpeechAdapter(
            {"speech-key": SpeechSourceData(torch.arange(3.0), 1, 0)}
        ),
        required_speech_sample_rate_hz=1,
    )
    with pytest.raises(SourceAdapterError, match="empty speech sample span"):
        dataset[0]


def test_dataset_never_interprets_path_like_speech_source_id() -> None:
    """Resolve a nonexistent path-like ID solely through the injected adapter."""
    source_id = r"Z:\adapter-owned\nonexistent\audio.wav"
    record = MultimodalWindowRecord(
        "path-like",
        "p1",
        "s1",
        TimeInterval(0, 1),
        EmotionScores(2, 4),
        TimedSourceRef(source_id, TimeInterval(0, 2)),
    )
    adapter = MemorySpeechAdapter(
        {source_id: SpeechSourceData(torch.tensor([3.0, 4.0]), 2, 0)}
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=adapter,
        required_speech_sample_rate_hz=2,
    )
    assert torch.equal(dataset[0].speech_waveform, torch.tensor([3.0, 4.0]))
    assert adapter.calls[0].source_id == source_id


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (SpeechSourceData(torch.arange(12.0), 4, 0), "sample rate mismatch"),
        (SpeechSourceData(torch.arange(2.0), 2, 0), "source time range"),
        (object(), "SpeechSourceData"),
        (RuntimeError("adapter boom"), "adapter boom"),
    ],
)
def test_speech_failures_are_contextual_source_adapter_errors(
    data: object,
    message: str,
) -> None:
    """Never convert a declared but failed speech source into missing speech."""
    dataset, _ = _speech_only_dataset(data=data)
    with pytest.raises(SourceAdapterError) as error_info:
        dataset[0]
    message_text = str(error_info.value)
    assert "sample-1" in message_text
    assert "speech" in message_text
    assert "speech-key" in message_text
    assert "original_exception=" in message_text
    assert message in message_text


def test_missing_speech_returns_none_without_adapter_call() -> None:
    """Represent absent speech explicitly for a physiology-only record."""
    spec = _spec()
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    physio_adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                spec,
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
            )
        }
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=physio_adapter,
        physio_target_sample_rate_hz=1.0,
    )
    sample = dataset[0]
    assert sample.speech_waveform is None
    assert sample.speech_attention_mask is None
    assert sample.speech_sample_rate_hz is None
    assert not sample.speech_available


def test_label_processing_calls_both_existing_public_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove Dataset delegates protocol and quadrant mathematics."""
    original_binarize = dataset_module.binarize_emotion_scores
    original_quadrant = dataset_module.derive_quadrant_labels
    calls: list[str] = []

    def binarize_spy(
        scores: Tensor,
        protocol: LabelProtocol,
        *,
        ignore_index: int,
    ) -> tuple[Tensor, Tensor]:
        calls.append(f"binarize:{protocol.value}:{float(scores)}")
        return original_binarize(scores, protocol, ignore_index=ignore_index)

    def quadrant_spy(
        arousal: Tensor,
        valence: Tensor,
        *,
        ignore_index: int,
    ) -> Tensor:
        calls.append("quadrant")
        return original_quadrant(
            arousal,
            valence,
            ignore_index=ignore_index,
        )

    monkeypatch.setattr(dataset_module, "binarize_emotion_scores", binarize_spy)
    monkeypatch.setattr(dataset_module, "derive_quadrant_labels", quadrant_spy)
    dataset, _ = _speech_only_dataset()

    dataset[0]

    assert calls == [
        "binarize:official_mid_high:2.0",
        "binarize:official_mid_high:4.0",
        "quadrant",
    ]


@pytest.mark.parametrize(
    ("start", "end", "rate", "expected"),
    [
        (0.0, 1.0, 4.0, [0.0, 0.25, 0.5, 0.75]),
        (0.0, 0.1 + 0.2, 10.0, [0.0, 0.1, 0.2]),
        (2.0, 2.1, 1.0, [2.0]),
    ],
)
def test_physio_target_timeline_uses_integer_indices_without_end_point(
    start: float,
    end: float,
    rate: float,
    expected: list[float],
) -> None:
    """Include start, exclude end, avoid cumulative drift, and keep float64."""
    record = _record(start=start, end=end)
    dataset = AlignedMultimodalDataset(
        (record,),
        (_spec(),),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=MemorySpeechAdapter(
            {
                "speech-key": SpeechSourceData(
                    torch.arange(100.0),
                    20,
                    0,
                )
            }
        ),
        required_speech_sample_rate_hz=20,
        physio_target_sample_rate_hz=rate,
    )

    timestamps = dataset[0].physio_timestamps_seconds

    assert timestamps is not None
    assert timestamps.dtype == torch.float64
    torch.testing.assert_close(
        timestamps,
        torch.tensor(expected, dtype=torch.float64),
    )
    assert bool((timestamps >= start).all())
    assert bool((timestamps < end).all())


def test_target_timeline_rejects_unrepresentable_float64_spacing() -> None:
    """Fail clearly instead of emitting duplicate target timestamps."""
    start = 1.0e16
    end = start + 2.0
    spec = _spec()
    record = _record(
        speech=False,
        start=start,
        end=end,
        physio_sources=(
            _physio_ref("eda", start=start, end=end),
        ),
    )
    adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                spec,
                torch.tensor([1.0]),
                torch.tensor([start], dtype=torch.float64),
            )
        }
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=2.0,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        dataset[0]
    assert adapter.calls == []


def test_physio_alignment_crops_before_resampling_and_keeps_channel_order() -> None:
    """Use only overlap points and assemble columns in spec, not source, order."""
    eda = _spec("eda")
    temp = _spec("temp", kind=PhysioSignalKind.TEMPERATURE)
    record = _record(
        speech=False,
        start=1,
        end=3,
        physio_sources=(
            _physio_ref("temp", "temp-key", start=1, end=3),
            _physio_ref("eda", "eda-key", start=1, end=3),
        ),
    )
    eda_window = _physio_window(
        eda,
        torch.tensor([100.0, 1.0, 3.0, 200.0]),
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    temp_window = _physio_window(
        temp,
        torch.tensor([300.0, 10.0, 14.0, 400.0]),
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    adapter = MemoryPhysioAdapter(
        {"eda-key": eda_window, "temp-key": temp_window}
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (eda, temp),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=2.0,
    )
    eda_original = eda_window.values.clone()

    sample = dataset[0]

    assert sample.channel_names == ("eda", "temp")
    assert sample.physio_input is not None
    torch.testing.assert_close(
        sample.physio_input,
        torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 12.0],
                [3.0, 14.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert sample.physio_valid_mask is not None
    assert torch.equal(
        sample.physio_valid_mask,
        torch.tensor(
            [
                [True, True],
                [True, True],
                [True, True],
                [False, False],
            ]
        ),
    )
    assert [call[0].channel_name for call in adapter.calls] == ["eda", "temp"]
    assert adapter.calls[0][0] is record.physio_sources[1]
    assert adapter.calls[1][0] is record.physio_sources[0]
    assert torch.equal(eda_window.values, eda_original)


def test_all_channels_receive_the_same_public_resampling_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove public resampling is called with one shared target tensor."""
    original_resample = dataset_module.resample_channel_to_timeline
    targets: list[Tensor] = []

    def resample_spy(
        window: PhysioChannelWindow,
        target: Tensor,
    ) -> PhysioChannelWindow:
        targets.append(target)
        return original_resample(window, target)

    monkeypatch.setattr(
        dataset_module,
        "resample_channel_to_timeline",
        resample_spy,
    )
    eda = _spec("eda")
    temp = _spec("temp", kind=PhysioSignalKind.TEMPERATURE)
    record = _record(
        speech=False,
        start=0,
        end=2,
        physio_sources=(
            _physio_ref("eda", "eda-key", start=0, end=2),
            _physio_ref("temp", "temp-key", start=0, end=2),
        ),
    )
    adapter = MemoryPhysioAdapter(
        {
            "eda-key": _physio_window(
                eda,
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.0, 1.0]),
            ),
            "temp-key": _physio_window(
                temp,
                torch.tensor([3.0, 4.0]),
                torch.tensor([0.0, 1.0]),
            ),
        }
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (eda, temp),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=1.0,
    )

    sample = dataset[0]

    assert len(targets) == 2
    assert targets[0] is targets[1]
    assert sample.physio_timestamps_seconds is targets[0]


def test_invalid_native_point_is_not_bridged_and_placeholder_does_not_pollute() -> None:
    """Preserve missing brackets and safely accept invalid NaN placeholders."""
    spec = _spec()
    record = _record(
        speech=False,
        start=0,
        end=3,
        physio_sources=(_physio_ref(start=0, end=3),),
    )
    window = _physio_window(
        spec,
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([True, False, True]),
    )
    adapter = MemoryPhysioAdapter({"physio-key": window})
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=2.0,
    )

    sample = dataset[0]

    assert sample.physio_input is not None
    assert sample.physio_valid_mask is not None
    assert torch.equal(
        sample.physio_valid_mask[:, 0],
        torch.tensor([True, False, False, False, True, False]),
    )
    assert torch.isfinite(sample.physio_input).all()
    assert torch.equal(window.valid_mask, torch.tensor([True, False, True]))
    assert torch.isnan(window.values[1])


def test_missing_channels_and_all_missing_physio_have_fixed_shapes_and_quality() -> None:
    """Return full configured matrices without calling a physiology adapter."""
    record = _record()
    speech_adapter = MemorySpeechAdapter({"speech-key": _speech_data()})
    specs = (_spec("eda"), _spec("temp", kind=PhysioSignalKind.TEMPERATURE))
    dataset = AlignedMultimodalDataset(
        (record,),
        specs,
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=speech_adapter,
        required_speech_sample_rate_hz=2,
        physio_target_sample_rate_hz=2.0,
        output_dtype=torch.float64,
    )

    sample = dataset[0]

    assert sample.physio_input is not None
    assert sample.physio_input.shape == (4, 2)
    assert sample.physio_input.dtype == torch.float64
    assert not bool(sample.physio_input.any())
    assert sample.physio_valid_mask is not None
    assert not bool(sample.physio_valid_mask.any())
    assert sample.physio_time_mask is not None
    assert not bool(sample.physio_time_mask.any())
    assert sample.physio_channel_mask is not None
    assert not bool(sample.physio_channel_mask.any())
    assert sample.physio_channel_quality is not None
    expected = torch.tensor(
        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        dtype=torch.float64,
    )
    assert torch.equal(sample.physio_channel_quality, expected)
    assert torch.equal(sample.physio_quality_features, expected.reshape(-1))
    assert not sample.physiology_available


def test_no_channel_specs_return_none_physio_fields() -> None:
    """Omit physiology tensors only when no top-level channel semantics exist."""
    sample = _speech_only_dataset()[0][0]
    assert sample.channel_names == ()
    assert sample.physio_input is None
    assert sample.physio_valid_mask is None
    assert sample.physio_time_mask is None
    assert sample.physio_channel_mask is None
    assert sample.physio_timestamps_seconds is None
    assert sample.physio_channel_quality is None
    assert sample.physio_quality_features is None
    assert not sample.physiology_available


def test_empty_native_overlap_selection_is_missing_and_skips_filter() -> None:
    """Treat no timestamp hit as observable absence, not adapter failure."""
    spec = _spec()
    record = _record(
        speech=False,
        start=1,
        end=2,
        physio_sources=(_physio_ref(start=1, end=2),),
    )
    window = _physio_window(
        spec,
        torch.tensor([10.0, 20.0]),
        torch.tensor([0.0, 3.0]),
    )
    adapter = MemoryPhysioAdapter({"physio-key": window})
    filter_calls: list[Tensor] = []

    def filter_fn(values: Tensor, mask: Tensor, rate: float) -> Tensor:
        del mask, rate
        filter_calls.append(values)
        return values

    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=2.0,
        physio_filters={"eda": filter_fn},
    )
    sample = dataset[0]

    assert len(adapter.calls) == 1
    assert filter_calls == []
    assert sample.physio_valid_mask is not None
    assert not bool(sample.physio_valid_mask.any())


def test_filter_receives_only_cropped_channel_and_is_channel_specific() -> None:
    """Apply a configured filter after native cropping and before resampling."""
    eda = _spec("eda")
    temp = _spec("temp", kind=PhysioSignalKind.TEMPERATURE)
    record = _record(
        speech=False,
        start=1,
        end=3,
        physio_sources=(
            _physio_ref("eda", "eda-key", start=1, end=3),
            _physio_ref("temp", "temp-key", start=1, end=3),
        ),
    )
    adapter = MemoryPhysioAdapter(
        {
            "eda-key": _physio_window(
                eda,
                torch.tensor([100.0, 1.0, 2.0, 200.0]),
                torch.tensor([0.0, 1.0, 2.0, 3.0]),
            ),
            "temp-key": _physio_window(
                temp,
                torch.tensor([100.0, 3.0, 4.0, 200.0]),
                torch.tensor([0.0, 1.0, 2.0, 3.0]),
            ),
        }
    )
    captured: list[Tensor] = []

    def filter_fn(values: Tensor, mask: Tensor, rate: float) -> Tensor:
        assert bool(mask.all())
        assert rate == eda.native_sample_rate_hz
        captured.append(values.clone())
        return values + 10

    dataset = AlignedMultimodalDataset(
        (record,),
        (eda, temp),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=1.0,
        physio_filters={"eda": filter_fn},
    )
    sample = dataset[0]

    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([1.0, 2.0]))
    assert sample.physio_input is not None
    assert torch.equal(sample.physio_input[:, 0], torch.tensor([11.0, 12.0]))
    assert torch.equal(sample.physio_input[:, 1], torch.tensor([3.0, 4.0]))


def test_filter_exception_is_wrapped_with_physio_context() -> None:
    """Wrap filter failures with sample, channel, source, and exception type."""
    spec = _spec()
    record = _record(
        speech=False,
        physio_sources=(_physio_ref(),),
    )
    adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                spec,
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
            )
        }
    )

    def fail(values: Tensor, mask: Tensor, rate: float) -> Tensor:
        del values, mask, rate
        raise LookupError("filter boom")

    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=1.0,
        physio_filters={"eda": fail},
    )
    with pytest.raises(SourceAdapterError) as error_info:
        dataset[0]
    message = str(error_info.value)
    assert "sample-1" in message
    assert "physiology" in message
    assert "eda" in message
    assert "physio-key" in message
    assert "LookupError" in message


@pytest.mark.parametrize("bad_result", [object(), RuntimeError("load boom")])
def test_physio_adapter_failures_are_contextual(bad_result: object) -> None:
    """Reject wrong return types and raised errors without hiding the modality."""
    spec = _spec()
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": bad_result}),
        physio_target_sample_rate_hz=1.0,
    )
    with pytest.raises(SourceAdapterError) as error_info:
        dataset[0]
    message = str(error_info.value)
    assert all(
        item in message
        for item in ("sample-1", "physiology", "physio-key", "eda", "original_exception")
    )


def test_physio_adapter_spec_mismatch_is_rejected() -> None:
    """Require complete equality with configured top-level channel semantics."""
    expected = _spec()
    wrong = _spec("eda", rate=2.0)
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    loaded = _physio_window(
        wrong,
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (expected,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": loaded}),
        physio_target_sample_rate_hz=1.0,
    )
    with pytest.raises(SourceAdapterError, match="configured spec"):
        dataset[0]


def test_artifact_policy_none_never_calls_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not run or exclude artifact rules without explicit policy."""
    spec = _spec()
    record = _record(speech=False, start=0, end=3, physio_sources=(_physio_ref(),))
    window = _physio_window(
        spec,
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.0, 1.0, 2.0]),
    )

    def forbidden(*args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise AssertionError("detector must not run")

    monkeypatch.setattr(dataset_module, "detect_flatline_mask", forbidden)
    monkeypatch.setattr(dataset_module, "detect_robust_outlier_mask", forbidden)
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
    )
    sample = dataset[0]
    assert sample.physio_valid_mask is not None
    assert bool(sample.physio_valid_mask.all())
    assert sample.physio_channel_quality is not None
    assert torch.equal(
        sample.physio_channel_quality[0],
        torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32),
    )


@pytest.mark.parametrize("exclude", [True, False])
def test_artifact_masks_quality_and_exclusion_have_distinct_semantics(
    exclude: bool,
) -> None:
    """Compute quality from base validity before optionally excluding artifacts."""
    spec = _spec()
    record = _record(speech=False, start=0, end=3, physio_sources=(_physio_ref(),))
    window = _physio_window(
        spec,
        torch.tensor([0.0, 0.0, 10.0]),
        torch.tensor([0.0, 1.0, 2.0]),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
        artifact_policy=PhysioArtifactPolicy(
            flatline_atol=0.0,
            flatline_min_run_length=2,
            outlier_threshold=6.0,
            exclude_detected_artifacts=exclude,
        ),
    )
    sample = dataset[0]

    assert sample.physio_channel_quality is not None
    torch.testing.assert_close(
        sample.physio_channel_quality[0],
        torch.tensor([1.0, 0.0, 2 / 3, 1 / 3, 1.0, 1.0]),
    )
    assert sample.physio_valid_mask is not None
    assert bool(sample.physio_valid_mask.any()) is (not exclude)
    assert sample.physiology_available is (not exclude)
    assert sample.physio_input is not None
    if exclude:
        assert not bool(sample.physio_input.any())
    else:
        assert torch.equal(sample.physio_input[:, 0], window.values)


@pytest.mark.parametrize(
    ("detect_flatline", "detect_outliers", "expected"),
    [
        (True, False, [2 / 3, 0.0, 2 / 3]),
        (False, True, [0.0, 1 / 3, 1 / 3]),
        (False, False, [0.0, 0.0, 0.0]),
    ],
)
def test_artifact_detector_enable_flags_control_quality_ratios(
    detect_flatline: bool,
    detect_outliers: bool,
    expected: list[float],
) -> None:
    """Treat disabled detector masks as all False in fixed quality positions."""
    spec = _spec()
    record = _record(speech=False, start=0, end=3, physio_sources=(_physio_ref(),))
    window = _physio_window(
        spec,
        torch.tensor([0.0, 0.0, 10.0]),
        torch.tensor([0.0, 1.0, 2.0]),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
        artifact_policy=PhysioArtifactPolicy(
            detect_flatline=detect_flatline,
            flatline_atol=0.0,
            flatline_min_run_length=2,
            detect_outliers=detect_outliers,
            exclude_detected_artifacts=False,
        ),
    )
    quality = dataset[0].physio_channel_quality
    assert quality is not None
    torch.testing.assert_close(quality[0, 2:5], torch.tensor(expected))


def _fitted_normalizer(
    spec: PhysioChannelSpec,
    key: NormalizationKey,
) -> ChannelwiseZScoreNormalizer:
    """Fit a tiny authorized normalizer outside the Dataset."""
    fit_window = _physio_window(
        spec,
        torch.tensor([1.0, 3.0]),
        torch.tensor([0.0, 1.0]),
    )
    return ChannelwiseZScoreNormalizer().fit(
        {key: (fit_window,)},
        scope=NormalizationFitScope.TRAIN,
    )


def test_normalization_uses_default_global_key_without_changing_quality_or_state() -> None:
    """Transform final valid values while preserving masks, quality, and state."""
    spec = _spec()
    key = NormalizationKey("eda", None)
    normalizer = _fitted_normalizer(spec, key)
    before = normalizer.state_dict()
    record = _record(
        speech=False,
        start=1,
        end=3,
        physio_sources=(_physio_ref(start=1, end=3),),
    )
    window = _physio_window(
        spec,
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 2.0]),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
        physio_normalizer=normalizer,
    )

    sample = dataset[0]

    assert sample.physio_input is not None
    assert torch.equal(sample.physio_input[:, 0], torch.tensor([-1.0, 1.0]))
    assert sample.physio_valid_mask is not None
    assert bool(sample.physio_valid_mask.all())
    assert sample.physio_channel_quality is not None
    assert torch.equal(
        sample.physio_channel_quality[0],
        torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32),
    )
    assert normalizer.state_dict() == before


@pytest.mark.parametrize("group_source", ["participant", "session"])
def test_custom_normalization_key_resolver_supports_explicit_groups(
    group_source: str,
) -> None:
    """Resolve participant- or session-specific keys without inference."""
    spec = _spec()
    record = _record(
        participant_id="participant-A",
        session_id="session-B",
        speech=False,
        physio_sources=(_physio_ref(),),
    )
    group_id = (
        record.participant_id if group_source == "participant" else record.session_id
    )
    key = NormalizationKey("eda", group_id)
    normalizer = _fitted_normalizer(spec, key)
    captured: list[tuple[MultimodalWindowRecord, PhysioChannelSpec]] = []

    def resolver(
        current_record: MultimodalWindowRecord,
        current_spec: PhysioChannelSpec,
    ) -> NormalizationKey:
        captured.append((current_record, current_spec))
        return NormalizationKey(current_spec.name, group_id)

    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter(
            {
                "physio-key": _physio_window(
                    spec,
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([1.0, 2.0]),
                )
            }
        ),
        physio_target_sample_rate_hz=1.0,
        physio_normalizer=normalizer,
        normalization_key_resolver=resolver,
    )
    dataset[0]
    assert captured == [(record, spec)]


@pytest.mark.parametrize(
    ("normalizer", "resolver", "message"),
    [
        (ChannelwiseZScoreNormalizer(), None, "must be fitted"),
        (
            _fitted_normalizer(_spec(), NormalizationKey("eda")),
            lambda record, spec: object(),
            "NormalizationKey",
        ),
        (
            _fitted_normalizer(_spec(), NormalizationKey("eda")),
            lambda record, spec: NormalizationKey("other"),
            "channel_name",
        ),
        (
            _fitted_normalizer(_spec(), NormalizationKey("eda", "known")),
            lambda record, spec: NormalizationKey("eda", "unknown"),
            "Unknown normalization key",
        ),
    ],
)
def test_normalization_failures_are_contextual_and_do_not_fallback(
    normalizer: ChannelwiseZScoreNormalizer,
    resolver: Callable[
        [MultimodalWindowRecord, PhysioChannelSpec],
        object,
    ]
    | None,
    message: str,
) -> None:
    """Reject unfitted, wrong, mismatched, and unknown normalization keys."""
    spec = _spec()
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter(
            {
                "physio-key": _physio_window(
                    spec,
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([1.0, 2.0]),
                )
            }
        ),
        physio_target_sample_rate_hz=1.0,
        physio_normalizer=normalizer,
        normalization_key_resolver=resolver,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError) as error_info:
        dataset[0]
    error = str(error_info.value)
    assert "sample-1" in error
    assert "eda" in error
    assert message in error


def test_artifact_exclusion_precedes_normalization() -> None:
    """Normalize only the final artifact-filtered validity mask."""
    spec = _spec()
    normalizer = _fitted_normalizer(spec, NormalizationKey("eda"))
    record = _record(
        speech=False,
        start=0,
        end=2,
        physio_sources=(_physio_ref(start=0, end=2),),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter(
            {
                "physio-key": _physio_window(
                    spec,
                    torch.tensor([2.0, 2.0]),
                    torch.tensor([0.0, 1.0]),
                )
            }
        ),
        physio_target_sample_rate_hz=1.0,
        artifact_policy=PhysioArtifactPolicy(
            flatline_atol=0,
            flatline_min_run_length=2,
            detect_outliers=False,
        ),
        physio_normalizer=normalizer,
    )
    sample = dataset[0]
    assert sample.physio_valid_mask is not None
    assert not bool(sample.physio_valid_mask.any())
    assert sample.physio_input is not None
    assert not bool(sample.physio_input.any())
    assert sample.physio_channel_quality is not None
    assert sample.physio_channel_quality[0, 0] == 1
    assert sample.physio_channel_quality[0, 2] == 1


def test_aligned_sample_is_frozen_uses_original_record_and_cpu_outputs() -> None:
    """Return the original immutable record with fresh finite CPU tensors."""
    record = _record()
    dataset, _ = _speech_only_dataset(record, output_dtype=torch.float64)
    sample = dataset[0]

    assert isinstance(sample, AlignedMultimodalSample)
    assert sample.record is record
    assert sample.speech_waveform is not None
    assert sample.speech_waveform.dtype == torch.float64
    assert sample.speech_waveform.device.type == "cpu"
    with pytest.raises(FrozenInstanceError):
        sample.speech_available = False  # type: ignore[misc]


def test_final_physio_masks_availability_quality_and_cpu_contract() -> None:
    """Derive time/channel masks and channel-major quality from final validity."""
    eda = _spec("eda")
    temp = _spec("temp", kind=PhysioSignalKind.TEMPERATURE)
    record = _record(
        speech=False,
        start=0,
        end=2,
        physio_sources=(_physio_ref("eda", start=0, end=2),),
    )
    window = _physio_window(
        eda,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.tensor([True, False]),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (eda, temp),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
        output_dtype=torch.float64,
    )

    sample = dataset[0]

    assert sample.physio_valid_mask is not None
    assert torch.equal(
        sample.physio_valid_mask,
        torch.tensor([[True, False], [False, False]]),
    )
    assert sample.physio_time_mask is not None
    assert torch.equal(sample.physio_time_mask, torch.tensor([True, False]))
    assert sample.physio_channel_mask is not None
    assert torch.equal(sample.physio_channel_mask, torch.tensor([True, False]))
    assert sample.physiology_available
    assert sample.physio_channel_quality is not None
    expected_quality = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(sample.physio_channel_quality, expected_quality)
    assert sample.physio_quality_features is not None
    torch.testing.assert_close(
        sample.physio_quality_features,
        torch.tensor(
            [
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float64,
        ),
    )
    for tensor in (
        sample.physio_input,
        sample.physio_valid_mask,
        sample.physio_time_mask,
        sample.physio_channel_mask,
        sample.physio_timestamps_seconds,
        sample.physio_channel_quality,
        sample.physio_quality_features,
    ):
        assert tensor is not None
        assert tensor.device.type == "cpu"
    assert sample.physio_timestamps_seconds is not None
    assert sample.physio_timestamps_seconds.dtype == torch.float64


def test_dataset_does_not_modify_record_sources_or_adapter_buffers() -> None:
    """Snapshot every caller-owned object across one multimodal access."""
    spec = _spec()
    record = _record(
        start=1,
        end=3,
        physio_sources=(_physio_ref(start=1, end=3),),
    )
    speech_data = _speech_data()
    physio_window = _physio_window(
        spec,
        torch.tensor([1.0, float("nan"), 3.0]),
        torch.tensor([1.0, 1.5, 2.0]),
        torch.tensor([True, False, True]),
    )
    speech_before = speech_data.waveform.clone()
    physio_before = physio_window.values.clone()
    mask_before = physio_window.valid_mask.clone()
    sources_before = record.physio_sources
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        speech_adapter=MemorySpeechAdapter({"speech-key": speech_data}),
        physio_adapter=MemoryPhysioAdapter({"physio-key": physio_window}),
        required_speech_sample_rate_hz=2,
        physio_target_sample_rate_hz=2.0,
    )

    dataset[0]

    assert torch.equal(speech_data.waveform, speech_before)
    torch.testing.assert_close(physio_window.values, physio_before, equal_nan=True)
    assert torch.equal(physio_window.valid_mask, mask_before)
    assert record.physio_sources is sources_before


def test_dataset_calls_public_regular_sample_mapping_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove speech slicing delegates its exact timeline contract publicly."""
    record = _record(start=1.0, end=3.0)
    dataset, _ = _speech_only_dataset(record)
    original = dataset_module.regular_sample_span
    calls: list[tuple[TimeInterval, float, float, int]] = []

    def spy(
        interval: TimeInterval,
        *,
        timeline_start_seconds: float,
        sample_rate_hz: float,
        num_samples: int,
    ) -> IndexSpan:
        calls.append(
            (
                interval,
                timeline_start_seconds,
                sample_rate_hz,
                num_samples,
            )
        )
        return original(
            interval,
            timeline_start_seconds=timeline_start_seconds,
            sample_rate_hz=sample_rate_hz,
            num_samples=num_samples,
        )

    monkeypatch.setattr(dataset_module, "regular_sample_span", spy)

    sample = dataset[0]

    assert sample.speech_waveform is not None
    assert torch.equal(sample.speech_waveform, torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert calls == [(record.window, 0.0, 2, 12)]


def test_dataset_calls_public_timestamp_mapping_on_exact_source_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove native cropping uses only record/source intersection boundaries."""
    spec = _spec()
    source = _physio_ref(start=1.5, end=2.5)
    record = _record(
        speech=False,
        start=1.0,
        end=3.0,
        physio_sources=(source,),
    )
    window = _physio_window(
        spec,
        torch.arange(6, dtype=torch.float32),
        torch.tensor([0.0, 1.0, 1.5, 2.0, 2.5, 3.0]),
    )
    adapter = MemoryPhysioAdapter({"physio-key": window})
    original = dataset_module.timestamp_index_span
    calls: list[tuple[Tensor, TimeInterval]] = []

    def spy(timestamps: Tensor, interval: TimeInterval) -> IndexSpan:
        calls.append((timestamps, interval))
        return original(timestamps, interval)

    monkeypatch.setattr(dataset_module, "timestamp_index_span", spy)
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=2.0,
    )

    dataset[0]

    assert len(calls) == 1
    assert calls[0][0] is window.timestamps_seconds
    assert calls[0][1] == TimeInterval(1.5, 2.5)


def test_dataset_never_calls_normalizer_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a pre-fitted normalizer while making any Dataset-side fit fatal."""
    spec = _spec()
    normalizer = _fitted_normalizer(spec, NormalizationKey("eda"))

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Dataset must never fit a normalizer.")

    monkeypatch.setattr(ChannelwiseZScoreNormalizer, "fit", forbidden_fit)
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter(
            {
                "physio-key": _physio_window(
                    spec,
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([1.0, 2.0]),
                )
            }
        ),
        physio_target_sample_rate_hz=1.0,
        physio_normalizer=normalizer,
    )

    sample = dataset[0]

    assert sample.physio_input is not None
    assert torch.equal(sample.physio_input[:, 0], torch.tensor([-1.0, 1.0]))


def test_missing_declared_channel_skips_adapter_filter_and_normalizer() -> None:
    """Keep a missing channel fixed without touching any processing dependency."""
    eda = _spec("eda")
    temp = _spec("temp", kind=PhysioSignalKind.TEMPERATURE)
    record = _record(
        speech=False,
        physio_sources=(_physio_ref("eda"),),
    )
    adapter = MemoryPhysioAdapter(
        {
            "physio-key": _physio_window(
                eda,
                torch.tensor([1.0, 3.0]),
                torch.tensor([1.0, 2.0]),
            )
        }
    )
    normalizer = _fitted_normalizer(eda, NormalizationKey("eda"))

    def forbidden_filter(values: Tensor, mask: Tensor, rate: float) -> Tensor:
        del values, mask, rate
        raise AssertionError("missing channel filter must not run")

    dataset = AlignedMultimodalDataset(
        (record,),
        (eda, temp),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=adapter,
        physio_target_sample_rate_hz=1.0,
        physio_filters={"temp": forbidden_filter},
        physio_normalizer=normalizer,
    )

    sample = dataset[0]

    assert len(adapter.calls) == 1
    assert adapter.calls[0][0].channel_name == "eda"
    assert sample.physio_valid_mask is not None
    assert not bool(sample.physio_valid_mask[:, 1].any())
    assert sample.physio_input is not None
    assert not bool(sample.physio_input[:, 1].any())


def test_filter_cannot_modify_adapter_working_buffers() -> None:
    """Pass cloned safe values and mask to a mutating injected filter."""
    spec = _spec()
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    window = _physio_window(
        spec,
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
    )
    original_values = window.values.clone()
    original_mask = window.valid_mask.clone()

    def mutating_filter(values: Tensor, mask: Tensor, rate: float) -> Tensor:
        del rate
        values.add_(5.0)
        mask.fill_(False)
        return values

    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
        physio_filters={"eda": mutating_filter},
    )

    sample = dataset[0]

    assert torch.equal(window.values, original_values)
    assert torch.equal(window.valid_mask, original_mask)
    assert sample.physio_input is not None
    assert torch.equal(sample.physio_input[:, 0], torch.tensor([6.0, 7.0]))


def test_invalid_loaded_physio_is_wrapped_with_source_context() -> None:
    """Retain exception chaining when an adapter returns corrupted effective data."""
    spec = _spec()
    record = _record(speech=False, physio_sources=(_physio_ref(),))
    window = _physio_window(
        spec,
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
    )
    object.__setattr__(window, "values", torch.tensor([float("nan"), 2.0]))
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter({"physio-key": window}),
        physio_target_sample_rate_hz=1.0,
    )

    with pytest.raises(SourceAdapterError) as error_info:
        dataset[0]

    message = str(error_info.value)
    assert "sample-1" in message
    assert "physiology" in message
    assert "physio-key" in message
    assert "eda" in message
    assert isinstance(error_info.value.__cause__, ValueError)


def test_sample_rejects_nonfinite_single_point_physio_timeline() -> None:
    """Reject NaN even when a one-point timeline has no adjacent comparison."""
    spec = _spec()
    record = _record(
        speech=False,
        start=0.0,
        end=0.5,
        physio_sources=(_physio_ref(start=0.0, end=0.5),),
    )
    dataset = AlignedMultimodalDataset(
        (record,),
        (spec,),
        label_protocol=LabelProtocol.MID_LOW,
        physio_adapter=MemoryPhysioAdapter(
            {
                "physio-key": _physio_window(
                    spec,
                    torch.tensor([1.0]),
                    torch.tensor([0.0]),
                )
            }
        ),
        physio_target_sample_rate_hz=1.0,
    )
    sample = dataset[0]

    with pytest.raises(ValueError, match="must be finite"):
        replace(
            sample,
            physio_timestamps_seconds=torch.tensor(
                [float("nan")],
                dtype=torch.float64,
            ),
        )


def test_dataset_scope_contains_no_batch_model_split_or_training_logic() -> None:
    """Keep stage 11B restricted to single-sample CPU data preparation."""
    source = (
        Path(dataset_module.__file__).read_text(encoding="utf-8")
        if dataset_module.__file__ is not None
        else ""
    )
    forbidden = (
        "def collate",
        "DataLoader",
        "pad_sequence",
        "WavLM",
        "PhysiologicalEmotionBranch",
        "partition_manifest_by_participant",
        ".fit(",
        "optimizer",
        "backward(",
    )
    assert not any(term in source for term in forbidden)
