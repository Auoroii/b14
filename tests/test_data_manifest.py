"""Tests for strict multimodal manifest data contracts and serialization."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from emotion_model.data import (
    EmotionScores,
    IndexSpan,
    MultimodalManifest,
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    TimedSourceRef,
    TimeInterval,
    manifest_from_dict,
    manifest_to_dict,
    read_manifest_json,
    write_manifest_json,
)
from emotion_model.physiology import PhysioChannelSpec, PhysioSignalKind


def _spec(
    name: str = "eda_left",
    kind: PhysioSignalKind = PhysioSignalKind.EDA,
) -> PhysioChannelSpec:
    """Create one explicit, dataset-independent channel specification."""
    unit = "uS" if kind is PhysioSignalKind.EDA else "mV"
    return PhysioChannelSpec(name, kind, 4.0, unit)


def _timed_source(
    source_id: str = "source:α",
    start: float = 0.0,
    end: float = 10.0,
) -> TimedSourceRef:
    """Create an abstract source reference."""
    return TimedSourceRef(source_id, TimeInterval(start, end))


def _physio_source(
    name: str = "eda_left",
    *,
    start: float = 0.0,
    end: float = 10.0,
) -> PhysioChannelSourceRef:
    """Create an explicitly named physiology source."""
    return PhysioChannelSourceRef(
        name,
        _timed_source(f"physio:{name}", start, end),
    )


def _record(
    sample_id: str = "sample-1",
    participant_id: str = "p1",
    session_id: str = "s1",
    *,
    window: TimeInterval | None = None,
    speech: bool = True,
    physio_names: tuple[str, ...] = ("eda_left",),
) -> MultimodalWindowRecord:
    """Create a valid synthetic manifest record."""
    logical_window = TimeInterval(2.0, 4.0) if window is None else window
    return MultimodalWindowRecord(
        sample_id=sample_id,
        participant_id=participant_id,
        session_id=session_id,
        window=logical_window,
        emotion_scores=EmotionScores(2.5, 4.0),
        speech_source=_timed_source("speech:session", 0.0, 8.0) if speech else None,
        physio_sources=tuple(
            _physio_source(
                name,
                start=logical_window.start_seconds - 1.0,
                end=logical_window.start_seconds + 1.0,
            )
            for name in physio_names
        ),
    )


def _manifest() -> MultimodalManifest:
    """Create a valid ordered manifest with two explicit channel kinds."""
    return MultimodalManifest(
        channel_specs=(
            _spec("eda_left", PhysioSignalKind.EDA),
            _spec("ecg_wave", PhysioSignalKind.ECG_WAVEFORM),
        ),
        records=(
            _record("样本一", physio_names=("eda_left",)),
            _record(
                "sample-2",
                "p2",
                "s2",
                window=TimeInterval(4.0, 6.0),
                physio_names=("ecg_wave",),
            ),
        ),
    )


def test_time_interval_half_open_operations_and_duration() -> None:
    """Use exact half-open containment, overlap, and intersection semantics."""
    outer = TimeInterval(0, 2)
    same = TimeInterval(0.0, 2.0)
    inside = TimeInterval(0.5, 1.5)
    adjacent = TimeInterval(2.0, 3.0)
    crossing = TimeInterval(1.5, 2.5)

    assert outer.duration_seconds == 2.0
    assert outer.contains(same)
    assert outer.contains(inside)
    assert not outer.overlaps(adjacent)
    assert outer.intersection(adjacent) is None
    assert outer.intersection(crossing) == TimeInterval(1.5, 2.0)


@pytest.mark.parametrize(
    ("start", "end", "exception"),
    [
        (0.0, 0.0, ValueError),
        (1.0, 0.0, ValueError),
        (float("nan"), 1.0, ValueError),
        (0.0, float("inf"), ValueError),
        (True, 1.0, TypeError),
        (0.0, False, TypeError),
        ("0", 1.0, TypeError),
    ],
)
def test_time_interval_rejects_invalid_boundaries(
    start: object,
    end: object,
    exception: type[Exception],
) -> None:
    """Reject non-finite, non-real, boolean, and non-positive intervals."""
    with pytest.raises(exception):
        TimeInterval(start, end)  # type: ignore[arg-type]


def test_time_interval_and_index_span_are_frozen() -> None:
    """Keep logical time and index ranges immutable."""
    interval = TimeInterval(0.0, 1.0)
    span = IndexSpan(1, 3)

    with pytest.raises(FrozenInstanceError):
        interval.start_seconds = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        span.end_index = 4  # type: ignore[misc]


def test_index_span_accepts_empty_and_reports_length() -> None:
    """Allow empty half-open spans without automatic clamping."""
    assert IndexSpan(2, 2).length == 0
    assert IndexSpan(2, 5).length == 3


@pytest.mark.parametrize(
    ("start", "end", "exception"),
    [
        (-1, 0, ValueError),
        (2, 1, ValueError),
        (True, 1, TypeError),
        (0, False, TypeError),
        (0.0, 1, TypeError),
    ],
)
def test_index_span_rejects_invalid_indices(
    start: object,
    end: object,
    exception: type[Exception],
) -> None:
    """Reject negative, reversed, boolean, and non-integer indices."""
    with pytest.raises(exception):
        IndexSpan(start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize(("arousal", "valence"), [(1, 5), (2.5, 3.75)])
def test_emotion_scores_preserve_raw_valid_ratings(
    arousal: float,
    valence: float,
) -> None:
    """Keep integer or averaged raw scores without deriving labels."""
    scores = EmotionScores(arousal, valence)

    assert scores.arousal == float(arousal)
    assert scores.valence == float(valence)
    assert not hasattr(scores, "quadrant")
    assert not hasattr(scores, "arousal_label")


@pytest.mark.parametrize(
    ("arousal", "valence", "exception"),
    [
        (0, 3, ValueError),
        (6, 3, ValueError),
        (3, float("nan"), ValueError),
        (float("inf"), 3, ValueError),
        (True, 3, TypeError),
        ("3", 3, TypeError),
    ],
)
def test_emotion_scores_reject_invalid_values(
    arousal: object,
    valence: object,
    exception: type[Exception],
) -> None:
    """Reject values outside [1,5], non-finite numbers, bool, and strings."""
    with pytest.raises(exception):
        EmotionScores(arousal, valence)  # type: ignore[arg-type]


def test_source_references_preserve_ids_and_are_frozen() -> None:
    """Preserve an abstract ID verbatim without path interpretation."""
    source = TimedSourceRef("  adapter-owned:id  ", TimeInterval(0, 1))
    channel = PhysioChannelSourceRef("rr_interval", source)

    assert source.source_id == "  adapter-owned:id  "
    assert channel.channel_name == "rr_interval"
    with pytest.raises(FrozenInstanceError):
        source.source_id = "changed"  # type: ignore[misc]


def test_source_id_is_not_interpreted_as_a_local_path() -> None:
    """Accept and preserve a nonexistent path-like ID without filesystem access."""
    source_id = r"Z:\adapter-owned\nonexistent\signal.bin"

    source = TimedSourceRef(source_id, TimeInterval(0, 1))

    assert source.source_id == source_id


@pytest.mark.parametrize("value", ["", "   ", 3, None])
def test_source_and_channel_references_reject_invalid_names(value: object) -> None:
    """Require non-empty string identifiers without guessing signal semantics."""
    with pytest.raises((TypeError, ValueError)):
        TimedSourceRef(value, TimeInterval(0, 1))  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        PhysioChannelSourceRef(value, _timed_source())  # type: ignore[arg-type]


def test_window_record_supports_each_modality_combination() -> None:
    """Accept speech-only, physiology-only, and multimodal records."""
    speech_only = _record("speech-only", physio_names=())
    physio_only = _record("physio-only", speech=False)
    multimodal = _record("both")

    assert speech_only.speech_available and not speech_only.physiology_available
    assert not physio_only.speech_available and physio_only.physiology_available
    assert multimodal.speech_available and multimodal.physiology_available


def test_window_record_rejects_no_modality_and_non_tuple_sources() -> None:
    """Reject absent modalities and mutable physiology source collections."""
    with pytest.raises(ValueError, match="speech or physiology"):
        _record("none", speech=False, physio_names=())
    with pytest.raises(TypeError, match="tuple"):
        MultimodalWindowRecord(
            "x",
            "p",
            "s",
            TimeInterval(0, 1),
            EmotionScores(3, 3),
            None,
            [_physio_source()],  # type: ignore[arg-type]
        )


def test_window_record_enforces_speech_full_coverage() -> None:
    """Reject speech sources that cover only part of a logical window."""
    with pytest.raises(ValueError, match="fully contain"):
        MultimodalWindowRecord(
            "x",
            "p",
            "s",
            TimeInterval(2, 4),
            EmotionScores(3, 3),
            _timed_source("speech", 2, 3),
        )


def test_window_record_allows_partial_physio_overlap_but_not_adjacency() -> None:
    """Allow positive partial physiology overlap and reject adjacent intervals."""
    record = _record("partial", speech=False)
    assert record.physio_sources[0].source.available_interval == TimeInterval(1, 3)

    with pytest.raises(ValueError, match="positive-length overlap"):
        MultimodalWindowRecord(
            "adjacent",
            "p",
            "s",
            TimeInterval(2, 4),
            EmotionScores(3, 3),
            None,
            (_physio_source(start=0, end=2),),
        )


def test_window_record_rejects_duplicate_physio_channel() -> None:
    """Reject two sources for the same named channel in one record."""
    with pytest.raises(ValueError, match="unique"):
        MultimodalWindowRecord(
            "duplicate-channel",
            "p",
            "s",
            TimeInterval(2, 4),
            EmotionScores(3, 3),
            None,
            (_physio_source(), _physio_source()),
        )


def test_manifest_preserves_record_and_channel_order() -> None:
    """Retain caller order without sorting by participant, label, or channel."""
    manifest = _manifest()

    assert tuple(spec.name for spec in manifest.channel_specs) == (
        "eda_left",
        "ecg_wave",
    )
    assert tuple(record.sample_id for record in manifest.records) == (
        "样本一",
        "sample-2",
    )


def test_manifest_allows_speech_only_without_channel_specs() -> None:
    """Permit empty channel semantics only when no physiology is referenced."""
    manifest = MultimodalManifest((), (_record("speech", physio_names=()),))
    assert manifest.channel_specs == ()


@pytest.mark.parametrize("failure", ["empty", "duplicate_spec", "unknown", "sample", "window"])
def test_manifest_rejects_invalid_global_contracts(failure: str) -> None:
    """Reject empty records, duplicate/unknown channels, and duplicate records."""
    if failure == "empty":
        with pytest.raises(ValueError, match="at least one"):
            MultimodalManifest((), ())
    elif failure == "duplicate_spec":
        with pytest.raises(ValueError, match="unique"):
            MultimodalManifest((_spec(), _spec()), (_record(),))
    elif failure == "unknown":
        with pytest.raises(ValueError, match="unknown"):
            MultimodalManifest((_spec("ecg_wave", PhysioSignalKind.ECG_WAVEFORM),), (_record(),))
    elif failure == "sample":
        with pytest.raises(ValueError, match="sample_id"):
            MultimodalManifest((_spec(),), (_record(), _record()))
    else:
        duplicate_window = _record("different-id")
        with pytest.raises(ValueError, match="duplicate"):
            MultimodalManifest((_spec(),), (_record(), duplicate_window))


def test_manifest_allows_missing_channels_and_repeated_participant_sessions() -> None:
    """Allow optional channels plus multiple sessions and windows per participant."""
    records = (
        _record("a", "p1", "s1", physio_names=("eda_left",)),
        _record(
            "b",
            "p1",
            "s2",
            window=TimeInterval(4, 6),
            physio_names=(),
        ),
        _record(
            "c",
            "p1",
            "s2",
            window=TimeInterval(6, 7),
            physio_names=(),
        ),
    )
    manifest = MultimodalManifest((_spec(),), records)
    assert manifest.records == records


def test_manifest_requires_tuples_and_supported_schema() -> None:
    """Reject mutable top-level collections, booleans, and unsupported versions."""
    with pytest.raises(TypeError, match="tuple"):
        MultimodalManifest([], (_record("speech", physio_names=()),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        MultimodalManifest((), [_record("speech", physio_names=())])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        MultimodalManifest((), (_record("speech", physio_names=()),), True)
    with pytest.raises(ValueError, match="Unsupported"):
        MultimodalManifest((), (_record("speech", physio_names=()),), 2)


def test_record_and_manifest_are_frozen_and_preserve_nested_objects() -> None:
    """Keep immutable records/manifests and reuse caller interval/source objects."""
    window = TimeInterval(2, 4)
    speech_source = _timed_source("speech", 0, 5)
    record = MultimodalWindowRecord(
        "sample",
        "participant",
        "session",
        window,
        EmotionScores(3, 4),
        speech_source,
    )
    manifest = MultimodalManifest((), (record,))

    assert record.window is window
    assert record.speech_source is speech_source
    assert manifest.records[0] is record
    with pytest.raises(FrozenInstanceError):
        record.sample_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.schema_version = 2  # type: ignore[misc]


def test_manifest_dict_round_trip_is_complete_and_json_compatible() -> None:
    """Round-trip complete raw data without enum or class representations."""
    manifest = _manifest()

    serialized = manifest_to_dict(manifest)
    restored = manifest_from_dict(serialized)
    encoded = json.dumps(serialized, ensure_ascii=False)

    assert restored == manifest
    assert serialized["schema_version"] == 1
    specs = serialized["channel_specs"]
    assert isinstance(specs, list)
    assert specs[0] == {
        "name": "eda_left",
        "signal_kind": "eda",
        "native_sample_rate_hz": 4.0,
        "unit": "uS",
        "description": None,
    }
    assert "样本一" in encoded
    assert "PhysioSignalKind" not in encoded
    assert "TimeInterval(" not in encoded
    assert "Tensor" not in encoded


def test_manifest_json_round_trip_uses_utf8_and_does_not_store_output_path() -> None:
    """Write/read UTF-8 JSON without embedding the destination path."""
    path = Path("tests") / "_stage11a_manifest_清单.json"
    manifest = _manifest()

    try:
        write_manifest_json(manifest, path)
        raw = path.read_text(encoding="utf-8")
        restored = read_manifest_json(path)
    finally:
        path.unlink(missing_ok=True)

    assert restored == manifest
    assert "样本一" in raw
    assert str(path) not in raw


@pytest.mark.parametrize(
    ("mutation", "exception"),
    [
        ("root_missing", ValueError),
        ("root_unknown", ValueError),
        ("nested_missing", ValueError),
        ("nested_unknown", ValueError),
        ("record_unknown", ValueError),
        ("source_unknown", ValueError),
        ("interval_unknown", ValueError),
        ("bad_version", ValueError),
        ("bool_version", TypeError),
        ("coerced_rate", TypeError),
        ("array_as_tuple", TypeError),
    ],
)
def test_manifest_parser_rejects_non_strict_schema(
    mutation: str,
    exception: type[Exception],
) -> None:
    """Reject missing, unknown, unsupported, and illegally typed JSON fields."""
    value = manifest_to_dict(_manifest())
    if mutation == "root_missing":
        del value["records"]
    elif mutation == "root_unknown":
        value["extra"] = 1
    elif mutation == "nested_missing":
        records = value["records"]
        assert isinstance(records, list)
        del records[0]["session_id"]
    elif mutation == "nested_unknown":
        specs = value["channel_specs"]
        assert isinstance(specs, list)
        specs[0]["dataset_column"] = "x"
    elif mutation == "record_unknown":
        records = value["records"]
        assert isinstance(records, list)
        records[0]["unexpected"] = "x"
    elif mutation == "source_unknown":
        records = value["records"]
        assert isinstance(records, list)
        records[0]["speech_source"]["unexpected"] = "x"
    elif mutation == "interval_unknown":
        records = value["records"]
        assert isinstance(records, list)
        records[0]["speech_source"]["available_interval"]["unexpected"] = "x"
    elif mutation == "bad_version":
        value["schema_version"] = 99
    elif mutation == "bool_version":
        value["schema_version"] = True
    elif mutation == "coerced_rate":
        specs = value["channel_specs"]
        assert isinstance(specs, list)
        specs[0]["native_sample_rate_hz"] = "4.0"
    else:
        value["records"] = tuple(value["records"])  # type: ignore[arg-type]

    with pytest.raises(exception):
        manifest_from_dict(value)


def test_deserialization_reexecutes_alignment_and_manifest_business_rules() -> None:
    """Reject invalid source coverage and duplicate sample IDs after parsing."""
    partial_speech = manifest_to_dict(_manifest())
    records = partial_speech["records"]
    assert isinstance(records, list)
    speech_source = records[0]["speech_source"]
    assert isinstance(speech_source, dict)
    speech_source["available_interval"] = {
        "start_seconds": 2.0,
        "end_seconds": 3.0,
    }
    with pytest.raises(ValueError, match="fully contain"):
        manifest_from_dict(partial_speech)

    duplicate = manifest_to_dict(_manifest())
    duplicate_records = duplicate["records"]
    assert isinstance(duplicate_records, list)
    duplicate_records[1]["sample_id"] = duplicate_records[0]["sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        manifest_from_dict(duplicate)


def test_manifest_parser_rejects_wrong_nested_source_and_score_types() -> None:
    """Do not coerce source IDs or raw scores from strings."""
    value = manifest_to_dict(_manifest())
    records = value["records"]
    assert isinstance(records, list)
    records[0]["emotion_scores"]["arousal"] = "2.5"
    with pytest.raises(TypeError, match="real number"):
        manifest_from_dict(value)

    value = manifest_to_dict(_manifest())
    records = value["records"]
    assert isinstance(records, list)
    records[0]["speech_source"]["source_id"] = 42
    with pytest.raises(TypeError, match="string"):
        manifest_from_dict(value)
