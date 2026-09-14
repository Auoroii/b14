"""Tests for compact CPU-only multimodal collation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

import emotion_model.data.collate as collate_module
import emotion_model.physiology.preprocessing as preprocessing_module
from emotion_model.data import (
    AlignedMultimodalBatch,
    AlignedMultimodalDataset,
    AlignedMultimodalSample,
    EmotionScores,
    MultimodalWindowRecord,
    PhysioSubBatch,
    SpeechSubBatch,
    TimedSourceRef,
    TimeInterval,
    apply_modality_keep_masks,
    collate_aligned_multimodal_samples,
    select_aligned_multimodal_batch,
)
from emotion_model.data.source_adapters import PhysioSourceAdapter, SpeechSourceAdapter
from emotion_model.experiments import (
    ModalityAblationMode,
    iter_both_modality_ablation_batches,
)
from emotion_model.physiology import ChannelwiseZScoreNormalizer


def _record(
    sample_id: str,
    *,
    raw_arousal: float = 2.0,
    raw_valence: float = 4.0,
) -> MultimodalWindowRecord:
    """Create one immutable source-free logical record."""
    return MultimodalWindowRecord(
        sample_id,
        f"participant-{sample_id}",
        f"session-{sample_id}",
        TimeInterval(0.0, 5.0),
        EmotionScores(raw_arousal, raw_valence),
        TimedSourceRef("logical-source", TimeInterval(0.0, 5.0)),
        (),
    )


def _quality(
    valid_mask: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Create identifiable finite quality rows without deriving batch output."""
    time_count, channel_count = valid_mask.shape
    rows: list[list[float]] = []
    for channel in range(channel_count):
        valid_ratio = float(valid_mask[:, channel].sum().item()) / time_count
        rows.append(
            [
                valid_ratio,
                1.0 - valid_ratio,
                0.05 * channel,
                0.1 * channel,
                0.15 * channel,
                float(bool(valid_mask[:, channel].any())),
            ]
        )
    return torch.tensor(rows, dtype=dtype)


def _sample(
    sample_id: str,
    *,
    speech_values: list[float] | None = None,
    physio_valid_mask: Tensor | None = None,
    physio_values: Tensor | None = None,
    channel_names: tuple[str, ...] = ("eda", "temp"),
    physio_length: int = 3,
    dtype: torch.dtype = torch.float32,
    raw_arousal: float = 2.0,
    raw_valence: float = 4.0,
    arousal_label: int = 0,
    valence_label: int = 1,
    quadrant_label: int | None = None,
    ignore_index: int = -100,
    sample_rate_hz: int = 16000,
    timestamp_origin: float = 0.0,
) -> AlignedMultimodalSample:
    """Create a small valid sample with explicit missing-modality tensors."""
    if quadrant_label is None:
        quadrant_label = (
            ignore_index
            if arousal_label == ignore_index or valence_label == ignore_index
            else arousal_label + 2 * valence_label
        )
    if speech_values is None:
        waveform = None
        speech_mask = None
        speech_rate = None
    else:
        waveform = torch.tensor(speech_values, dtype=dtype)
        speech_mask = torch.ones(len(speech_values), dtype=torch.bool)
        speech_rate = sample_rate_hz

    if not channel_names:
        physio_input = None
        valid_mask = None
        time_mask = None
        channel_mask = None
        timestamps = None
        channel_quality = None
        quality_features = None
        physiology_available = False
    else:
        channel_count = len(channel_names)
        if physio_valid_mask is not None:
            physio_length = physio_valid_mask.shape[0]
        valid_mask = (
            torch.zeros((physio_length, channel_count), dtype=torch.bool)
            if physio_valid_mask is None
            else physio_valid_mask.clone()
        )
        if physio_values is None:
            base = torch.arange(
                physio_length * channel_count,
                dtype=dtype,
            ).reshape(physio_length, channel_count)
            physio_input = torch.where(
                valid_mask,
                base + 1,
                torch.zeros_like(base),
            )
        else:
            physio_input = physio_values.clone().to(dtype=dtype)
        time_mask = valid_mask.any(dim=1)
        channel_mask = valid_mask.any(dim=0)
        timestamps = (
            torch.arange(physio_length, dtype=torch.float64) + timestamp_origin
        )
        channel_quality = _quality(valid_mask, dtype=dtype)
        quality_features = channel_quality.reshape(-1).clone()
        physiology_available = bool(valid_mask.any())

    return AlignedMultimodalSample(
        record=_record(
            sample_id,
            raw_arousal=raw_arousal,
            raw_valence=raw_valence,
        ),
        raw_arousal=raw_arousal,
        raw_valence=raw_valence,
        arousal_label=arousal_label,
        valence_label=valence_label,
        quadrant_label=quadrant_label,
        label_ignore_index=ignore_index,
        speech_waveform=waveform,
        speech_attention_mask=speech_mask,
        speech_sample_rate_hz=speech_rate,
        speech_available=waveform is not None,
        physio_input=physio_input,
        physio_valid_mask=valid_mask,
        physio_time_mask=time_mask,
        physio_channel_mask=channel_mask,
        physio_timestamps_seconds=timestamps,
        physio_channel_quality=channel_quality,
        physio_quality_features=quality_features,
        physiology_available=physiology_available,
        channel_names=channel_names,
    )


def _speech_subbatch() -> SpeechSubBatch:
    """Create a legal two-row right-padded speech subbatch."""
    return SpeechSubBatch(
        waveform=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        attention_mask=torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        sequence_lengths=torch.tensor([2, 3], dtype=torch.long),
        sample_rate_hz=16000,
        batch_indices=torch.tensor([0, 2], dtype=torch.long),
    )


def _physio_subbatch() -> PhysioSubBatch:
    """Create a legal subbatch containing internal gaps and right padding."""
    valid_mask = torch.zeros((2, 5, 2), dtype=torch.bool)
    valid_mask[0, 0, 0] = True
    valid_mask[0, 1, 1] = True
    valid_mask[0, 4, 0] = True
    valid_mask[1, 0, 1] = True
    valid_mask[1, 2, 0] = True
    values = torch.zeros((2, 5, 2))
    values[valid_mask] = torch.arange(1, 6, dtype=torch.float32)
    quality = torch.tensor(
        [
            [[0.6, 0.4, 0.0, 0.0, 0.0, 1.0], [0.2, 0.8, 0.0, 0.0, 0.0, 1.0]],
            [[0.3, 0.7, 0.0, 0.0, 0.0, 1.0], [0.4, 0.6, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    return PhysioSubBatch(
        physio_input=values,
        physio_valid_mask=valid_mask,
        physio_time_mask=valid_mask.any(dim=2),
        physio_channel_mask=valid_mask.any(dim=1),
        physio_timestamps_seconds=torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        physio_timeline_mask=torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
        timeline_lengths=torch.tensor([5, 3], dtype=torch.long),
        physio_channel_quality=quality,
        physio_quality_features=quality.reshape(2, 12).clone(),
        channel_names=("eda", "temp"),
        batch_indices=torch.tensor([0, 2], dtype=torch.long),
    )


def test_public_subbatch_dataclasses_are_valid_and_frozen() -> None:
    """Accept valid contracts and prevent field reassignment."""
    speech = _speech_subbatch()
    physiology = _physio_subbatch()

    assert speech.waveform.shape == (2, 3)
    assert physiology.physio_input.shape == (2, 5, 2)
    with pytest.raises(FrozenInstanceError):
        speech.sample_rate_hz = 8000  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        physiology.channel_names = ("temp", "eda")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("change", "error_type"),
    [
        (lambda value: replace(value, waveform=torch.ones(3)), ValueError),
        (
            lambda value: replace(
                value,
                waveform=torch.ones((2, 3), dtype=torch.long),
            ),
            TypeError,
        ),
        (
            lambda value: replace(
                value,
                attention_mask=value.attention_mask.to(torch.int64),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                attention_mask=torch.tensor(
                    [[True, False, True], [True, True, True]]
                ),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                attention_mask=torch.tensor(
                    [[False, False, False], [True, True, True]]
                ),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                waveform=torch.tensor(
                    [[1.0, 2.0, 9.0], [3.0, 4.0, 5.0]]
                ),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                sequence_lengths=torch.tensor([1, 3]),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                batch_indices=torch.tensor([0, 2], dtype=torch.int32),
            ),
            ValueError,
        ),
        (
            lambda value: replace(
                value,
                batch_indices=torch.tensor([2, 1]),
            ),
            ValueError,
        ),
        (lambda value: replace(value, sample_rate_hz=True), TypeError),
        (lambda value: replace(value, sample_rate_hz=0), ValueError),
    ],
)
def test_speech_subbatch_rejects_contract_violations(
    change: Callable[[SpeechSubBatch], SpeechSubBatch],
    error_type: type[Exception],
) -> None:
    """Reject invalid shapes, masks, padding, lengths, indices, and rates."""
    with pytest.raises(error_type):
        change(_speech_subbatch())


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, physio_input=torch.ones((2, 5))),
        lambda value: replace(
            value,
            physio_valid_mask=value.physio_valid_mask.to(torch.int64),
        ),
        lambda value: replace(
            value,
            physio_time_mask=torch.ones((2, 5), dtype=torch.bool),
        ),
        lambda value: replace(
            value,
            physio_channel_mask=torch.zeros((2, 2), dtype=torch.bool),
        ),
        lambda value: replace(
            value,
            physio_timeline_mask=torch.tensor(
                [
                    [True, False, True, True, True],
                    [True, True, True, False, False],
                ]
            ),
        ),
        lambda value: replace(
            value,
            timeline_lengths=torch.tensor([4, 3]),
        ),
        lambda value: replace(
            value,
            physio_timestamps_seconds=value.physio_timestamps_seconds.float(),
        ),
        lambda value: replace(
            value,
            physio_timestamps_seconds=torch.tensor(
                [
                    [0.0, float("nan"), 2.0, 3.0, 4.0],
                    [10.0, 11.0, 12.0, 0.0, 0.0],
                ],
                dtype=torch.float64,
            ),
        ),
        lambda value: replace(
            value,
            physio_timestamps_seconds=torch.tensor(
                [
                    [0.0, 1.0, 1.0, 3.0, 4.0],
                    [10.0, 11.0, 12.0, 0.0, 0.0],
                ],
                dtype=torch.float64,
            ),
        ),
        lambda value: replace(
            value,
            physio_timestamps_seconds=torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [10.0, 11.0, 12.0, 9.0, 0.0],
                ],
                dtype=torch.float64,
            ),
        ),
        lambda value: replace(
            value,
            physio_input=value.physio_input
            + (~value.physio_valid_mask).to(value.physio_input.dtype),
        ),
        lambda value: replace(
            value,
            physio_channel_quality=torch.zeros((2, 2, 5)),
        ),
        lambda value: replace(
            value,
            physio_quality_features=torch.ones((2, 12)),
        ),
        lambda value: replace(
            value,
            physio_channel_quality=torch.full((2, 2, 6), 1.1),
            physio_quality_features=torch.full((2, 12), 1.1),
        ),
        lambda value: replace(value, channel_names=("eda", "eda")),
        lambda value: replace(
            value,
            batch_indices=torch.tensor([2, 1]),
        ),
    ],
)
def test_physio_subbatch_rejects_contract_violations(
    change: Callable[[PhysioSubBatch], PhysioSubBatch],
) -> None:
    """Reject incompatible shapes, masks, timelines, quality, and indices."""
    with pytest.raises((TypeError, ValueError)):
        change(_physio_subbatch())


def test_physio_subbatch_rejects_validity_outside_timeline() -> None:
    """Do not allow valid physiology points in right-padding positions."""
    value = _physio_subbatch()
    timeline_mask = value.physio_timeline_mask.clone()
    timeline_mask[0, 4] = False
    lengths = torch.tensor([4, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="timeline"):
        replace(
            value,
            physio_timeline_mask=timeline_mask,
            timeline_lengths=lengths,
        )


def _full_batch() -> AlignedMultimodalBatch:
    """Create a valid full batch through the public collate API."""
    valid = torch.tensor(
        [[True, False], [False, False], [False, True]]
    )
    samples = (
        _sample("a", speech_values=[1.0, 2.0], physio_valid_mask=valid),
        _sample("b", speech_values=[3.0], physio_length=2),
        _sample("c", physio_valid_mask=valid),
    )
    return collate_aligned_multimodal_samples(samples)


def test_aligned_batch_contract_is_valid_and_frozen() -> None:
    """Validate full shapes, exact modality mappings, and immutability."""
    batch = _full_batch()

    assert batch.raw_arousal.shape == (3,)
    assert batch.speech is not None
    assert torch.equal(batch.speech.batch_indices, torch.tensor([0, 1]))
    assert batch.physiology is not None
    assert torch.equal(batch.physiology.batch_indices, torch.tensor([0, 2]))
    with pytest.raises(FrozenInstanceError):
        batch.label_ignore_index = -1  # type: ignore[misc]


def test_modality_keep_masks_filter_compact_rows_without_mutation() -> None:
    """Drop existing modalities using logical ``[B]`` masks only."""

    batch = _full_batch()
    transformed = apply_modality_keep_masks(
        batch,
        speech_keep_mask=torch.tensor([True, False, True]),
        physiology_keep_mask=torch.tensor([False, True, True]),
    )

    assert torch.equal(
        transformed.speech_available,
        torch.tensor([True, False, False]),
    )
    assert transformed.speech is not None
    assert torch.equal(
        transformed.speech.batch_indices,
        torch.tensor([0]),
    )
    assert torch.equal(
        transformed.physiology_available,
        torch.tensor([False, False, True]),
    )
    assert transformed.physiology is not None
    assert torch.equal(
        transformed.physiology.batch_indices,
        torch.tensor([2]),
    )
    assert torch.equal(
        batch.speech_available,
        torch.tensor([True, True, False]),
    )
    assert torch.equal(
        batch.physiology_available,
        torch.tensor([True, False, True]),
    )


def test_logical_row_selection_reindexes_both_compact_modalities() -> None:
    """Select logical rows and remap compact indices to the new batch."""

    selected = select_aligned_multimodal_batch(
        _full_batch(),
        torch.tensor([False, True, True]),
    )

    assert tuple(record.sample_id for record in selected.records) == ("b", "c")
    assert selected.speech is not None
    assert torch.equal(selected.speech.batch_indices, torch.tensor([0]))
    assert selected.physiology is not None
    assert torch.equal(selected.physiology.batch_indices, torch.tensor([1]))
    assert torch.equal(selected.arousal_labels, torch.tensor([0, 0]))


def test_controlled_ablation_uses_identical_originally_both_rows() -> None:
    """Keep sample order fixed while retaining full, speech, or physiology."""

    outputs = {
        mode: tuple(iter_both_modality_ablation_batches((_full_batch(),), mode))
        for mode in ModalityAblationMode
    }
    for batches in outputs.values():
        assert len(batches) == 1
        assert tuple(record.sample_id for record in batches[0].records) == ("a",)
    full = outputs[ModalityAblationMode.FULL][0]
    speech = outputs[ModalityAblationMode.SPEECH_ONLY][0]
    physiology = outputs[ModalityAblationMode.PHYSIOLOGY_ONLY][0]
    assert full.speech is not None and full.physiology is not None
    assert speech.speech is not None and speech.physiology is None
    assert physiology.speech is None and physiology.physiology is not None


@pytest.mark.parametrize(
    ("mask", "error_type"),
    [
        (torch.tensor([False, False, False]), ValueError),
        (torch.tensor([1, 0, 0]), ValueError),
        (torch.tensor([True, False]), ValueError),
    ],
)
def test_logical_row_selection_rejects_invalid_masks(
    mask: Tensor,
    error_type: type[Exception],
) -> None:
    """Reject empty, non-boolean, and incorrectly shaped row masks."""

    with pytest.raises(error_type):
        select_aligned_multimodal_batch(_full_batch(), mask)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, records=value.records[:2]),
        lambda value: replace(value, raw_arousal=torch.ones(2, dtype=torch.float64)),
        lambda value: replace(
            value,
            raw_arousal=torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64),
        ),
        lambda value: replace(value, label_ignore_index=2),
        lambda value: replace(
            value,
            arousal_labels=torch.tensor([0, 8, 0]),
        ),
        lambda value: replace(
            value,
            quadrant_labels=torch.tensor([0, 2, 2]),
        ),
        lambda value: replace(
            value,
            speech_available=torch.tensor([False, False, False]),
        ),
        lambda value: replace(value, speech=None),
        lambda value: replace(
            value,
            physiology_available=torch.tensor([False, False, False]),
        ),
        lambda value: replace(value, physiology=None),
    ],
)
def test_aligned_batch_rejects_contract_violations(
    change: Callable[[AlignedMultimodalBatch], AlignedMultimodalBatch],
) -> None:
    """Reject invalid full tensors and availability/subbatch contradictions."""
    with pytest.raises((TypeError, ValueError)):
        change(_full_batch())


@pytest.mark.parametrize(
    "samples",
    [
        (),
        [],
    ],
)
def test_collate_rejects_empty_samples(
    samples: list[AlignedMultimodalSample] | tuple[()],
) -> None:
    """Reject an empty logical batch."""
    with pytest.raises(ValueError, match="non-empty"):
        collate_aligned_multimodal_samples(samples)


@pytest.mark.parametrize("samples", ["abc", iter(())])
def test_collate_rejects_non_sequence(samples: object) -> None:
    """Reject strings and one-shot iterators."""
    with pytest.raises(TypeError, match="Sequence"):
        collate_aligned_multimodal_samples(samples)  # type: ignore[arg-type]


def test_collate_rejects_wrong_exact_sample_type() -> None:
    """Require every sequence item to be the public sample dataclass."""
    with pytest.raises(TypeError, match="exactly"):
        collate_aligned_multimodal_samples([object()])  # type: ignore[list-item]


def test_collate_preserves_full_order_labels_and_unavailable_rows() -> None:
    """Keep logical order independently of length, labels, and modality pattern."""
    valid_short = torch.tensor([[True, False], [False, True]])
    valid_long = torch.tensor(
        [[True, False], [False, False], [False, True], [True, False]]
    )
    samples = [
        _sample(
            "short-both",
            speech_values=[10.0],
            physio_valid_mask=valid_short,
            raw_arousal=5,
            raw_valence=1,
            arousal_label=1,
            valence_label=0,
        ),
        _sample(
            "long-speech",
            speech_values=[20.0, 21.0, 22.0, 23.0],
            physio_length=3,
            raw_arousal=1,
            raw_valence=5,
            arousal_label=0,
            valence_label=1,
        ),
        _sample(
            "long-physio",
            physio_valid_mask=valid_long,
            raw_arousal=4,
            raw_valence=4,
            arousal_label=1,
            valence_label=1,
        ),
        _sample(
            "ignored-none",
            physio_length=2,
            raw_arousal=3,
            raw_valence=3,
            arousal_label=-100,
            valence_label=-100,
        ),
    ]

    batch = collate_aligned_multimodal_samples(samples)

    assert tuple(record.sample_id for record in batch.records) == (
        "short-both",
        "long-speech",
        "long-physio",
        "ignored-none",
    )
    assert all(batch.records[index] is samples[index].record for index in range(4))
    assert torch.equal(
        batch.raw_arousal,
        torch.tensor([5.0, 1.0, 4.0, 3.0], dtype=torch.float64),
    )
    assert torch.equal(
        batch.raw_valence,
        torch.tensor([1.0, 5.0, 4.0, 3.0], dtype=torch.float64),
    )
    assert torch.equal(batch.arousal_labels, torch.tensor([1, 0, 1, -100]))
    assert torch.equal(batch.valence_labels, torch.tensor([0, 1, 1, -100]))
    assert torch.equal(batch.quadrant_labels, torch.tensor([1, 2, 3, -100]))
    assert torch.equal(
        batch.speech_available,
        torch.tensor([True, True, False, False]),
    )
    assert torch.equal(
        batch.physiology_available,
        torch.tensor([True, False, True, False]),
    )
    assert batch.speech is not None
    assert torch.equal(batch.speech.batch_indices, torch.tensor([0, 1]))
    assert torch.equal(batch.speech.sequence_lengths, torch.tensor([1, 4]))
    assert batch.physiology is not None
    assert torch.equal(batch.physiology.batch_indices, torch.tensor([0, 2]))
    assert torch.equal(batch.physiology.timeline_lengths, torch.tensor([2, 4]))
    selected_labels = batch.arousal_labels.index_select(
        0,
        batch.physiology.batch_indices,
    )
    assert torch.equal(selected_labels, torch.tensor([1, 1]))


def test_speech_right_padding_values_masks_and_storage_are_independent() -> None:
    """Right-pad without truncating, reordering, or sharing input storage."""
    first = _sample("first", speech_values=[1.0, 2.0])
    missing = _sample("missing")
    second = _sample("second", speech_values=[3.0, 4.0, 5.0, 6.0])
    first_before = first.speech_waveform.clone()  # type: ignore[union-attr]
    second_before = second.speech_waveform.clone()  # type: ignore[union-attr]

    batch = collate_aligned_multimodal_samples((first, missing, second))

    assert batch.speech is not None
    speech = batch.speech
    assert torch.equal(
        speech.waveform,
        torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 6.0]]),
    )
    assert torch.equal(
        speech.attention_mask,
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )
    assert torch.equal(speech.sequence_lengths, torch.tensor([2, 4]))
    assert torch.equal(speech.batch_indices, torch.tensor([0, 2]))
    assert bool(speech.attention_mask.any(dim=1).all())
    assert speech.waveform.is_contiguous()
    assert speech.attention_mask.is_contiguous()
    assert first.speech_waveform is not None
    assert first.speech_attention_mask is not None
    assert second.speech_waveform is not None
    assert speech.waveform.data_ptr() != first.speech_waveform.data_ptr()
    assert speech.waveform.data_ptr() != second.speech_waveform.data_ptr()
    assert (
        speech.attention_mask.data_ptr()
        != first.speech_attention_mask.data_ptr()
    )
    speech.waveform[0, 0] = 99
    assert torch.equal(first.speech_waveform, first_before)
    first.speech_waveform[0] = -1
    assert torch.equal(speech.waveform[0, 1:], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(second.speech_waveform, second_before)


def test_collate_runtime_boundary_calls_no_external_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make every out-of-scope data or execution boundary fail if invoked."""
    sample = _sample(
        "ready",
        speech_values=[1.0, 2.0],
        physio_valid_mask=torch.tensor(
            [[True, False], [False, True]]
        ),
    )
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("called")
        raise AssertionError("collate crossed an external execution boundary")

    for owner, attribute in (
        (AlignedMultimodalDataset, "__getitem__"),
        (SpeechSourceAdapter, "load_speech"),
        (PhysioSourceAdapter, "load_physio"),
        (preprocessing_module, "apply_channel_filter"),
        (preprocessing_module, "detect_flatline_mask"),
        (preprocessing_module, "detect_robust_outlier_mask"),
        (preprocessing_module, "resample_channel_to_timeline"),
        (ChannelwiseZScoreNormalizer, "transform"),
        (torch.nn.Module, "__call__"),
    ):
        monkeypatch.setattr(owner, attribute, forbidden)

    batch = collate_aligned_multimodal_samples((sample,))

    assert calls == []
    assert batch.speech is not None
    assert batch.physiology is not None


def test_physiology_padding_preserves_internal_gaps_and_timeline_extent() -> None:
    """Distinguish internal all-channel gaps from batch right padding."""
    mask_a = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
            [False, False],
            [True, False],
        ]
    )
    mask_b = torch.tensor(
        [[True, False], [False, False], [False, True]]
    )
    sample_a = _sample(
        "a",
        physio_valid_mask=mask_a,
        timestamp_origin=0.0,
    )
    sample_b = _sample(
        "b",
        physio_valid_mask=mask_b,
        timestamp_origin=10.0,
    )

    batch = collate_aligned_multimodal_samples((sample_a, sample_b))

    assert batch.physiology is not None
    physiology = batch.physiology
    assert torch.equal(
        physiology.physio_timeline_mask,
        torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        ),
    )
    assert torch.equal(
        physiology.physio_time_mask,
        torch.tensor(
            [
                [True, True, False, False, True],
                [True, False, True, False, False],
            ]
        ),
    )
    assert torch.equal(physiology.timeline_lengths, torch.tensor([5, 3]))
    assert torch.equal(
        physiology.physio_timestamps_seconds,
        torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        physiology.physio_input,
        torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 4.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [9.0, 0.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 6.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ]
        ),
    )
    assert not bool(physiology.physio_valid_mask[1, 3:].any())
    assert not bool(physiology.physio_input[1, 3:].any())
    assert torch.equal(
        physiology.physio_channel_mask,
        torch.stack(
            [sample_a.physio_channel_mask, sample_b.physio_channel_mask]
        ),
    )
    assert torch.equal(
        physiology.physio_channel_quality,
        torch.stack(
            [
                sample_a.physio_channel_quality,
                sample_b.physio_channel_quality,
            ]
        ),
    )
    assert torch.equal(
        physiology.physio_quality_features[0],
        torch.tensor(
            [
                0.4,
                0.6,
                0.0,
                0.0,
                0.0,
                1.0,
                0.2,
                0.8,
                0.05,
                0.1,
                0.15,
                1.0,
            ]
        ),
    )


def test_physiology_outputs_have_independent_storage() -> None:
    """Ensure every padded physiology tensor is newly allocated."""
    mask = torch.tensor([[True, False], [False, True]])
    sample = _sample("physio", physio_valid_mask=mask)
    input_before = sample.physio_input.clone()  # type: ignore[union-attr]
    valid_before = sample.physio_valid_mask.clone()  # type: ignore[union-attr]
    timestamps_before = sample.physio_timestamps_seconds.clone()  # type: ignore[union-attr]
    quality_before = sample.physio_channel_quality.clone()  # type: ignore[union-attr]
    features_before = sample.physio_quality_features.clone()  # type: ignore[union-attr]

    batch = collate_aligned_multimodal_samples((sample,))
    assert batch.physiology is not None
    physiology = batch.physiology
    input_tensors = (
        sample.physio_input,
        sample.physio_valid_mask,
        sample.physio_time_mask,
        sample.physio_channel_mask,
        sample.physio_timestamps_seconds,
        sample.physio_channel_quality,
        sample.physio_quality_features,
    )
    output_tensors = (
        physiology.physio_input,
        physiology.physio_valid_mask,
        physiology.physio_time_mask,
        physiology.physio_channel_mask,
        physiology.physio_timestamps_seconds,
        physiology.physio_channel_quality,
        physiology.physio_quality_features,
    )
    assert all(tensor is not None for tensor in input_tensors)
    for source, output in zip(input_tensors, output_tensors, strict=True):
        assert source is not None
        assert source.data_ptr() != output.data_ptr()
    physiology.physio_input[0, 0, 0] = 99
    physiology.physio_valid_mask[0, 0, 0] = False
    physiology.physio_timestamps_seconds[0, 0] = 99
    physiology.physio_channel_quality[0, 0, 0] = 0
    physiology.physio_quality_features[0, 0] = 0
    assert torch.equal(sample.physio_input, input_before)
    assert torch.equal(sample.physio_valid_mask, valid_before)
    assert torch.equal(sample.physio_timestamps_seconds, timestamps_before)
    assert torch.equal(sample.physio_channel_quality, quality_before)
    assert torch.equal(sample.physio_quality_features, features_before)
    assert sample.physio_input is not None
    assert sample.physio_valid_mask is not None
    sample.physio_input[1, 1] = 77
    sample.physio_valid_mask[1, 1] = False
    assert physiology.physio_input[0, 1, 1] != 77
    assert bool(physiology.physio_valid_mask[0, 1, 1])


def test_all_missing_modalities_return_none_subbatches_but_keep_labels() -> None:
    """Keep logical samples even when neither compact modality has a row."""
    first = _sample("first", physio_length=2)
    second = _sample(
        "ignored",
        physio_length=4,
        raw_arousal=3,
        raw_valence=3,
        arousal_label=-100,
        valence_label=-100,
    )

    batch = collate_aligned_multimodal_samples((first, second))

    assert batch.speech is None
    assert batch.physiology is None
    assert len(batch.records) == 2
    assert not bool(batch.speech_available.any())
    assert not bool(batch.physiology_available.any())
    assert torch.equal(batch.arousal_labels, torch.tensor([0, -100]))
    assert torch.equal(batch.quadrant_labels, torch.tensor([2, -100]))


def test_empty_channel_contract_requires_none_physiology_fields() -> None:
    """Support a shared empty channel contract without fabricating tensors."""
    samples = (
        _sample("a", speech_values=[1.0], channel_names=()),
        _sample("b", channel_names=()),
    )
    batch = collate_aligned_multimodal_samples(samples)
    assert batch.physiology is None
    assert batch.speech is not None


def test_collate_rejects_mixed_channel_contracts_and_order() -> None:
    """Require identical channel counts, names, and order across the batch."""
    standard = _sample("a")
    reversed_names = _sample("b", channel_names=("temp", "eda"))
    empty = _sample("c", channel_names=())
    with pytest.raises(ValueError, match="same channel_names"):
        collate_aligned_multimodal_samples((standard, reversed_names))
    with pytest.raises(ValueError, match="same channel_names"):
        collate_aligned_multimodal_samples((standard, empty))


def test_collate_rejects_mixed_ignore_indices() -> None:
    """Require one unambiguous ignore value for every full-batch label."""
    first = _sample("a")
    second = _sample("b")
    object.__setattr__(second, "label_ignore_index", -1)
    with pytest.raises(ValueError, match="share label_ignore_index"):
        collate_aligned_multimodal_samples((first, second))


def test_collate_rejects_inconsistent_quadrant_via_public_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensively validate labels through the shared quadrant API."""
    sample = _sample("a")
    calls: list[tuple[Tensor, Tensor, int]] = []
    original = collate_module.derive_quadrant_labels

    def spy(
        arousal: Tensor,
        valence: Tensor,
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        calls.append((arousal, valence, ignore_index))
        return original(arousal, valence, ignore_index=ignore_index)

    monkeypatch.setattr(collate_module, "derive_quadrant_labels", spy)
    collate_aligned_multimodal_samples((sample,))
    assert len(calls) == 1

    object.__setattr__(sample, "quadrant_label", 0)
    with pytest.raises(ValueError, match="public"):
        collate_aligned_multimodal_samples((sample,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collate_preserves_modality_dtypes_and_cpu_contract(
    dtype: torch.dtype,
) -> None:
    """Preserve float32/float64 without promotion or device movement."""
    mask = torch.tensor([[True, False], [False, True]])
    samples = (
        _sample(
            "a",
            speech_values=[1.0, 2.0],
            physio_valid_mask=mask,
            dtype=dtype,
        ),
        _sample(
            "b",
            speech_values=[3.0],
            physio_valid_mask=mask,
            dtype=dtype,
        ),
    )

    batch = collate_aligned_multimodal_samples(samples)

    assert batch.speech is not None
    assert batch.speech.waveform.dtype == dtype
    assert batch.physiology is not None
    assert batch.physiology.physio_input.dtype == dtype
    assert batch.physiology.physio_channel_quality.dtype == dtype
    assert batch.physiology.physio_timestamps_seconds.dtype == torch.float64
    for tensor in (
        batch.raw_arousal,
        batch.raw_valence,
        batch.arousal_labels,
        batch.valence_labels,
        batch.quadrant_labels,
        batch.speech_available,
        batch.physiology_available,
        batch.speech.waveform,
        batch.speech.attention_mask,
        batch.speech.sequence_lengths,
        batch.speech.batch_indices,
        batch.physiology.physio_input,
        batch.physiology.physio_valid_mask,
        batch.physiology.physio_time_mask,
        batch.physiology.physio_channel_mask,
        batch.physiology.physio_timestamps_seconds,
        batch.physiology.physio_timeline_mask,
        batch.physiology.timeline_lengths,
        batch.physiology.physio_channel_quality,
        batch.physiology.physio_quality_features,
        batch.physiology.batch_indices,
    ):
        assert tensor.device.type == "cpu"
        assert not tensor.is_pinned()


def test_collate_rejects_mixed_speech_dtype_rate_and_physio_dtype() -> None:
    """Reject mixed float precision and sampling rate without conversion."""
    with pytest.raises(TypeError, match="speech waveforms"):
        collate_aligned_multimodal_samples(
            (
                _sample(
                    "a",
                    speech_values=[1.0],
                    channel_names=(),
                    dtype=torch.float32,
                ),
                _sample(
                    "b",
                    speech_values=[2.0],
                    channel_names=(),
                    dtype=torch.float64,
                ),
            )
        )
    with pytest.raises(ValueError, match="sample rate"):
        collate_aligned_multimodal_samples(
            (
                _sample(
                    "a",
                    speech_values=[1.0],
                    channel_names=(),
                    sample_rate_hz=8000,
                ),
                _sample(
                    "b",
                    speech_values=[2.0],
                    channel_names=(),
                    sample_rate_hz=16000,
                ),
            )
        )
    with pytest.raises(TypeError, match="physiology-contract"):
        collate_aligned_multimodal_samples(
            (
                _sample("a", dtype=torch.float32),
                _sample("b", dtype=torch.float64),
            )
        )


def test_collate_defensively_rechecks_corrupted_sample_contracts() -> None:
    """Reject mutable-tensor corruption even after frozen sample construction."""
    invalid_speech = _sample("speech", speech_values=[1.0, 2.0])
    assert invalid_speech.speech_attention_mask is not None
    invalid_speech.speech_attention_mask[1] = False
    with pytest.raises(ValueError, match=r"samples\[0\]"):
        collate_aligned_multimodal_samples((invalid_speech,))

    invalid_physio = _sample(
        "physio",
        physio_valid_mask=torch.tensor(
            [[True, False], [False, True]]
        ),
    )
    assert invalid_physio.physio_input is not None
    invalid_physio.physio_input[0, 1] = 5
    with pytest.raises(ValueError, match=r"samples\[0\]"):
        collate_aligned_multimodal_samples((invalid_physio,))


def test_collate_does_not_modify_input_sequence_or_sample_scalars() -> None:
    """Preserve input container order, availability, labels, and records."""
    samples = [
        _sample(
            "a",
            speech_values=[1.0],
            physio_valid_mask=torch.tensor(
                [[True, False], [False, True]]
            ),
        ),
        _sample("b", physio_length=4),
    ]
    sequence_snapshot = tuple(samples)
    scalar_snapshot = tuple(
        (
            sample.record,
            sample.speech_available,
            sample.physiology_available,
            sample.arousal_label,
            sample.valence_label,
            sample.quadrant_label,
        )
        for sample in samples
    )

    collate_aligned_multimodal_samples(samples)

    assert tuple(samples) == sequence_snapshot
    assert scalar_snapshot == tuple(
        (
            sample.record,
            sample.speech_available,
            sample.physiology_available,
            sample.arousal_label,
            sample.valence_label,
            sample.quadrant_label,
        )
        for sample in samples
    )


def test_direct_routing_shapes_match_existing_classifier_inputs() -> None:
    """Verify routable tensor shapes without importing or executing a model."""
    mask = torch.tensor([[True, False], [False, True], [False, False]])
    batch = collate_aligned_multimodal_samples(
        (
            _sample(
                "a",
                speech_values=[1.0, 2.0],
                physio_valid_mask=mask,
            ),
            _sample("b", speech_values=[3.0], physio_length=2),
        )
    )

    assert batch.speech is not None
    assert batch.speech.waveform.ndim == 2
    assert batch.speech.attention_mask.shape == batch.speech.waveform.shape
    assert bool(batch.speech.attention_mask.any(dim=1).all())
    assert isinstance(batch.speech.sample_rate_hz, int)
    assert batch.physiology is not None
    assert batch.physiology.physio_input.ndim == 3
    assert (
        batch.physiology.physio_valid_mask.shape
        == batch.physiology.physio_input.shape
    )
    assert (
        batch.physiology.physio_time_mask.shape
        == batch.physiology.physio_input.shape[:2]
    )
    assert (
        batch.physiology.physio_channel_mask.shape
        == (
            batch.physiology.physio_input.shape[0],
            batch.physiology.physio_input.shape[2],
        )
    )
    assert bool(batch.physiology.physio_time_mask.any(dim=1).all())


def test_collate_scope_has_no_out_of_stage_execution_logic() -> None:
    """Keep stage 11C limited to validation, selection, stacking, and padding."""
    source = (
        Path(collate_module.__file__).read_text(encoding="utf-8")
        if collate_module.__file__ is not None
        else ""
    )
    forbidden = (
        "DataLoader",
        "torch.utils.data",
        "AlignedMultimodalDataset(",
        "load_speech(",
        "load_physio(",
        ".transform(",
        ".fit(",
        "SpeechEmotionClassifier",
        "PhysioEmotionClassifier",
        "optimizer",
        "backward(",
        "partition_manifest_by_participant",
        "resample_channel_to_timeline",
        "clean_speech",
        "reconstruct_waveform",
        "denoise_output",
        "noise_subtraction",
        "speech_enhancement",
    )
    assert not any(term in source for term in forbidden)
