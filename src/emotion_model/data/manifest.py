"""Strict, dataset-independent multimodal manifest contracts.

The manifest records logical half-open windows and abstract source identifiers.
It does not open files, interpret source identifiers as paths, apply label
protocols, or derive quadrant labels.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from emotion_model.physiology.channel_metadata import (
    PhysioChannelSpec,
    PhysioSignalKind,
)

_SCHEMA_VERSION = 1


def _validated_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty.")
    return value


def _validated_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; received {value!r}.")
    return result


@dataclass(frozen=True, order=True)
class TimeInterval:
    """Represent an immutable half-open interval ``[start_seconds, end_seconds)``.

    Args:
        start_seconds: Finite real-valued start time in seconds.
        end_seconds: Finite real-valued exclusive end time in seconds.

    Raises:
        TypeError: If either boundary is not a real number or is boolean.
        ValueError: If either boundary is non-finite or ``end <= start``.
    """

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start = _validated_real(self.start_seconds, name="start_seconds")
        end = _validated_real(self.end_seconds, name="end_seconds")
        if end <= start:
            raise ValueError(
                "end_seconds must be strictly greater than start_seconds; "
                f"received [{start}, {end})."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        """Return the positive interval duration in seconds."""
        return self.end_seconds - self.start_seconds

    def contains(self, other: TimeInterval) -> bool:
        """Return whether this interval fully contains ``other``.

        Args:
            other: Another valid half-open interval.

        Returns:
            ``True`` when both boundaries of ``other`` lie within this interval.

        Raises:
            TypeError: If ``other`` is not a :class:`TimeInterval`.
        """
        if not isinstance(other, TimeInterval):
            raise TypeError("other must be a TimeInterval.")
        return (
            self.start_seconds <= other.start_seconds
            and other.end_seconds <= self.end_seconds
        )

    def overlaps(self, other: TimeInterval) -> bool:
        """Return whether ``other`` has a positive-length overlap.

        Adjacent half-open intervals do not overlap.
        """
        if not isinstance(other, TimeInterval):
            raise TypeError("other must be a TimeInterval.")
        return (
            max(self.start_seconds, other.start_seconds)
            < min(self.end_seconds, other.end_seconds)
        )

    def intersection(self, other: TimeInterval) -> TimeInterval | None:
        """Return the positive-length intersection, or ``None`` when disjoint."""
        if not isinstance(other, TimeInterval):
            raise TypeError("other must be a TimeInterval.")
        start = max(self.start_seconds, other.start_seconds)
        end = min(self.end_seconds, other.end_seconds)
        if start >= end:
            return None
        return TimeInterval(start, end)


@dataclass(frozen=True, order=True)
class IndexSpan:
    """Represent an immutable half-open integer span ``[start_index, end_index)``.

    Empty spans are valid when both indices are equal.
    """

    start_index: int
    end_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("start_index", self.start_index),
            ("end_index", self.end_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool.")
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0.")
        if self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index.")

    @property
    def length(self) -> int:
        """Return ``end_index - start_index``."""
        return self.end_index - self.start_index


@dataclass(frozen=True)
class EmotionScores:
    """Store raw arousal and valence scores without label conversion.

    Args:
        arousal: Finite real score in the inclusive range ``[1, 5]``.
        valence: Finite real score in the inclusive range ``[1, 5]``.

    Label binarization and derived quadrant logic are intentionally deferred
    until after participant partitioning and require an explicit label protocol.
    """

    arousal: float
    valence: float

    def __post_init__(self) -> None:
        arousal = _validated_real(self.arousal, name="arousal")
        valence = _validated_real(self.valence, name="valence")
        for name, value in (("arousal", arousal), ("valence", valence)):
            if not 1.0 <= value <= 5.0:
                raise ValueError(f"{name} must lie in the inclusive range [1, 5].")
        object.__setattr__(self, "arousal", arousal)
        object.__setattr__(self, "valence", valence)


@dataclass(frozen=True)
class TimedSourceRef:
    """Reference an abstract source and its available logical time interval.

    ``source_id`` is preserved exactly and is not interpreted as a file path,
    opened, normalized, or joined to any data root.
    """

    source_id: str
    available_interval: TimeInterval

    def __post_init__(self) -> None:
        _validated_non_empty_string(self.source_id, name="source_id")
        if not isinstance(self.available_interval, TimeInterval):
            raise TypeError("available_interval must be a TimeInterval.")


@dataclass(frozen=True)
class PhysioChannelSourceRef:
    """Associate an explicit channel name with an abstract timed source."""

    channel_name: str
    source: TimedSourceRef

    def __post_init__(self) -> None:
        _validated_non_empty_string(self.channel_name, name="channel_name")
        if not isinstance(self.source, TimedSourceRef):
            raise TypeError("source must be a TimedSourceRef.")


@dataclass(frozen=True)
class MultimodalWindowRecord:
    """Describe one logical multimodal emotion window without loading data.

    Args:
        sample_id: Stable non-empty sample identifier, preserved exactly.
        participant_id: Stable non-empty participant identifier.
        session_id: Stable non-empty session identifier.
        window: Shared logical half-open time window.
        emotion_scores: Raw arousal and valence scores.
        speech_source: Optional speech source that must fully cover ``window``.
        physio_sources: Tuple of uniquely named physiology sources. Each source
            needs only a positive-length overlap with ``window``.

    At least one modality must be present. Partial physiology coverage is
    represented later by a validity mask; partial speech coverage is rejected.
    """

    sample_id: str
    participant_id: str
    session_id: str
    window: TimeInterval
    emotion_scores: EmotionScores
    speech_source: TimedSourceRef | None = None
    physio_sources: tuple[PhysioChannelSourceRef, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_id", self.sample_id),
            ("participant_id", self.participant_id),
            ("session_id", self.session_id),
        ):
            _validated_non_empty_string(value, name=name)
        if not isinstance(self.window, TimeInterval):
            raise TypeError("window must be a TimeInterval.")
        if not isinstance(self.emotion_scores, EmotionScores):
            raise TypeError("emotion_scores must be EmotionScores.")
        if self.speech_source is not None and not isinstance(
            self.speech_source,
            TimedSourceRef,
        ):
            raise TypeError("speech_source must be TimedSourceRef or None.")
        if not isinstance(self.physio_sources, tuple):
            raise TypeError("physio_sources must be a tuple.")
        for source in self.physio_sources:
            if not isinstance(source, PhysioChannelSourceRef):
                raise TypeError(
                    "physio_sources must contain PhysioChannelSourceRef objects."
                )
        channel_names = tuple(source.channel_name for source in self.physio_sources)
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("physio_sources channel names must be unique per record.")
        if self.speech_source is None and not self.physio_sources:
            raise ValueError("a record must contain speech or physiology.")
        if (
            self.speech_source is not None
            and not self.speech_source.available_interval.contains(self.window)
        ):
            raise ValueError(
                "speech source available_interval must fully contain record window."
            )
        for source in self.physio_sources:
            if not source.source.available_interval.overlaps(self.window):
                raise ValueError(
                    "physiology source must have a positive-length overlap with "
                    f"record window; channel={source.channel_name!r}."
                )

    @property
    def speech_available(self) -> bool:
        """Return whether a speech source is present."""
        return self.speech_source is not None

    @property
    def physiology_available(self) -> bool:
        """Return whether at least one physiology source is present."""
        return bool(self.physio_sources)


@dataclass(frozen=True)
class MultimodalManifest:
    """Hold validated channel semantics and ordered multimodal window records."""

    channel_specs: tuple[PhysioChannelSpec, ...]
    records: tuple[MultimodalWindowRecord, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.channel_specs, tuple):
            raise TypeError("channel_specs must be a tuple.")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple.")
        if not self.records:
            raise ValueError("manifest must contain at least one record.")
        for spec in self.channel_specs:
            if not isinstance(spec, PhysioChannelSpec):
                raise TypeError("channel_specs must contain PhysioChannelSpec objects.")
        for record in self.records:
            if not isinstance(record, MultimodalWindowRecord):
                raise TypeError(
                    "records must contain MultimodalWindowRecord objects."
                )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
        ):
            raise TypeError("schema_version must be an integer, not bool.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version {self.schema_version!r}; "
                f"supported version is {_SCHEMA_VERSION}."
            )

        spec_names = tuple(spec.name for spec in self.channel_specs)
        if len(set(spec_names)) != len(spec_names):
            raise ValueError("channel_specs names must be unique.")
        known_channels = set(spec_names)
        for record in self.records:
            for source in record.physio_sources:
                if source.channel_name not in known_channels:
                    raise ValueError(
                        f"record {record.sample_id!r} references unknown physiology "
                        f"channel {source.channel_name!r}."
                    )

        sample_ids = tuple(record.sample_id for record in self.records)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_id values must be globally unique.")
        logical_windows = tuple(
            (record.participant_id, record.session_id, record.window)
            for record in self.records
        )
        if len(set(logical_windows)) != len(logical_windows):
            raise ValueError(
                "duplicate (participant_id, session_id, window) records are not allowed."
            )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
    return value


def _require_exact_fields(
    value: object,
    *,
    name: str,
    fields: set[str],
) -> Mapping[str, object]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise ValueError(
            f"{name} fields are invalid; missing={missing}, unknown={unknown}."
        )
    return mapping


def _require_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _interval_to_dict(interval: TimeInterval) -> dict[str, object]:
    return {
        "start_seconds": interval.start_seconds,
        "end_seconds": interval.end_seconds,
    }


def _interval_from_dict(value: object, *, name: str) -> TimeInterval:
    mapping = _require_exact_fields(
        value,
        name=name,
        fields={"start_seconds", "end_seconds"},
    )
    return TimeInterval(
        _validated_real(mapping["start_seconds"], name=f"{name}.start_seconds"),
        _validated_real(mapping["end_seconds"], name=f"{name}.end_seconds"),
    )


def _source_to_dict(source: TimedSourceRef) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "available_interval": _interval_to_dict(source.available_interval),
    }


def _source_from_dict(value: object, *, name: str) -> TimedSourceRef:
    mapping = _require_exact_fields(
        value,
        name=name,
        fields={"source_id", "available_interval"},
    )
    source_id = mapping["source_id"]
    source_id = _validated_non_empty_string(source_id, name=f"{name}.source_id")
    return TimedSourceRef(
        source_id=source_id,
        available_interval=_interval_from_dict(
            mapping["available_interval"],
            name=f"{name}.available_interval",
        ),
    )


def manifest_to_dict(manifest: MultimodalManifest) -> dict[str, object]:
    """Convert a manifest to a strict JSON-compatible dictionary.

    Args:
        manifest: Valid manifest containing no tensors or loaded signal data.

    Returns:
        A fresh dictionary containing schema version, complete channel specs,
        raw scores, logical windows, and abstract source references.

    Raises:
        TypeError: If ``manifest`` has the wrong type.
    """
    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be a MultimodalManifest.")

    channel_specs: list[object] = []
    for spec in manifest.channel_specs:
        channel_specs.append(
            {
                "name": spec.name,
                "signal_kind": spec.signal_kind.value,
                "native_sample_rate_hz": spec.native_sample_rate_hz,
                "unit": spec.unit,
                "description": spec.description,
            }
        )

    records: list[object] = []
    for record in manifest.records:
        physio_sources: list[object] = []
        for channel_source in record.physio_sources:
            physio_sources.append(
                {
                    "channel_name": channel_source.channel_name,
                    "source": _source_to_dict(channel_source.source),
                }
            )
        records.append(
            {
                "sample_id": record.sample_id,
                "participant_id": record.participant_id,
                "session_id": record.session_id,
                "window": _interval_to_dict(record.window),
                "emotion_scores": {
                    "arousal": record.emotion_scores.arousal,
                    "valence": record.emotion_scores.valence,
                },
                "speech_source": (
                    None
                    if record.speech_source is None
                    else _source_to_dict(record.speech_source)
                ),
                "physio_sources": physio_sources,
            }
        )
    return {
        "schema_version": manifest.schema_version,
        "channel_specs": channel_specs,
        "records": records,
    }


def manifest_from_dict(value: Mapping[str, object]) -> MultimodalManifest:
    """Parse and fully validate a strict JSON-compatible manifest mapping.

    Args:
        value: Mapping with exactly the fields emitted by
            :func:`manifest_to_dict`.

    Returns:
        An immutable :class:`MultimodalManifest`.

    Raises:
        TypeError: If a field has an illegal JSON-level type.
        ValueError: If fields are missing/unknown, the schema is unsupported,
            or any nested business contract is invalid.

    No illegal value is silently coerced, and all dataclass validation is run.
    """
    root = _require_exact_fields(
        value,
        name="manifest",
        fields={"schema_version", "channel_specs", "records"},
    )
    version = root["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("manifest.schema_version must be an integer, not bool.")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {version!r}; "
            f"supported version is {_SCHEMA_VERSION}."
        )

    channel_specs: list[PhysioChannelSpec] = []
    for index, raw_spec in enumerate(
        _require_list(root["channel_specs"], name="manifest.channel_specs")
    ):
        name = f"manifest.channel_specs[{index}]"
        mapping = _require_exact_fields(
            raw_spec,
            name=name,
            fields={
                "name",
                "signal_kind",
                "native_sample_rate_hz",
                "unit",
                "description",
            },
        )
        channel_name = mapping["name"]
        signal_kind = mapping["signal_kind"]
        sample_rate = mapping["native_sample_rate_hz"]
        unit = mapping["unit"]
        description = mapping["description"]
        channel_name = _validated_non_empty_string(
            channel_name,
            name=f"{name}.name",
        )
        if not isinstance(signal_kind, str):
            raise TypeError(f"{name}.signal_kind must be a string.")
        try:
            resolved_kind = PhysioSignalKind(signal_kind)
        except ValueError as error:
            raise ValueError(
                f"{name}.signal_kind has unsupported value {signal_kind!r}."
            ) from error
        unit = _validated_non_empty_string(unit, name=f"{name}.unit")
        if description is not None and not isinstance(description, str):
            raise TypeError(f"{name}.description must be a string or None.")
        channel_specs.append(
            PhysioChannelSpec(
                name=channel_name,
                signal_kind=resolved_kind,
                native_sample_rate_hz=_validated_real(
                    sample_rate,
                    name=f"{name}.native_sample_rate_hz",
                ),
                unit=unit,
                description=description,
            )
        )

    records: list[MultimodalWindowRecord] = []
    for record_index, raw_record in enumerate(
        _require_list(root["records"], name="manifest.records")
    ):
        name = f"manifest.records[{record_index}]"
        mapping = _require_exact_fields(
            raw_record,
            name=name,
            fields={
                "sample_id",
                "participant_id",
                "session_id",
                "window",
                "emotion_scores",
                "speech_source",
                "physio_sources",
            },
        )
        identifiers: dict[str, str] = {}
        for field in ("sample_id", "participant_id", "session_id"):
            identifier = mapping[field]
            identifiers[field] = _validated_non_empty_string(
                identifier,
                name=f"{name}.{field}",
            )

        raw_scores = _require_exact_fields(
            mapping["emotion_scores"],
            name=f"{name}.emotion_scores",
            fields={"arousal", "valence"},
        )
        raw_speech = mapping["speech_source"]
        speech_source = (
            None
            if raw_speech is None
            else _source_from_dict(raw_speech, name=f"{name}.speech_source")
        )
        physio_sources: list[PhysioChannelSourceRef] = []
        for channel_index, raw_channel in enumerate(
            _require_list(
                mapping["physio_sources"],
                name=f"{name}.physio_sources",
            )
        ):
            channel_name = f"{name}.physio_sources[{channel_index}]"
            channel_mapping = _require_exact_fields(
                raw_channel,
                name=channel_name,
                fields={"channel_name", "source"},
            )
            raw_channel_name = channel_mapping["channel_name"]
            raw_channel_name = _validated_non_empty_string(
                raw_channel_name,
                name=f"{channel_name}.channel_name",
            )
            physio_sources.append(
                PhysioChannelSourceRef(
                    channel_name=raw_channel_name,
                    source=_source_from_dict(
                        channel_mapping["source"],
                        name=f"{channel_name}.source",
                    ),
                )
            )

        records.append(
            MultimodalWindowRecord(
                sample_id=identifiers["sample_id"],
                participant_id=identifiers["participant_id"],
                session_id=identifiers["session_id"],
                window=_interval_from_dict(
                    mapping["window"],
                    name=f"{name}.window",
                ),
                emotion_scores=EmotionScores(
                    arousal=_validated_real(
                        raw_scores["arousal"],
                        name=f"{name}.emotion_scores.arousal",
                    ),
                    valence=_validated_real(
                        raw_scores["valence"],
                        name=f"{name}.emotion_scores.valence",
                    ),
                ),
                speech_source=speech_source,
                physio_sources=tuple(physio_sources),
            )
        )

    return MultimodalManifest(
        channel_specs=tuple(channel_specs),
        records=tuple(records),
        schema_version=version,
    )


def write_manifest_json(
    manifest: MultimodalManifest,
    path: str | os.PathLike[str],
) -> None:
    """Write a manifest dictionary as UTF-8 JSON.

    Args:
        manifest: Valid manifest to serialize.
        path: Output JSON path. The path is not stored in manifest content.

    Returns:
        ``None``.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            manifest_to_dict(manifest),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def read_manifest_json(path: str | os.PathLike[str]) -> MultimodalManifest:
    """Read UTF-8 JSON and return a fully validated manifest.

    Args:
        path: Input JSON path. It is never retained in the result.

    Returns:
        Parsed immutable :class:`MultimodalManifest`.

    Raises:
        TypeError: If the JSON root or a nested field has an illegal type.
        ValueError: If strict fields, schema, or business rules are invalid.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as file:
        value: object = json.load(file)
    mapping = _require_mapping(value, name="manifest")
    return manifest_from_dict(mapping)
