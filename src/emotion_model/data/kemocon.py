"""K-EmoCon adapters, manifest construction, and dyad-aware splits.

All identifiers stored in a manifest are POSIX-style paths relative to the
configured K-EmoCon data root.  Filesystem resolution happens only inside the
adapters, so manifests remain portable between the local workstation and a
server with the same dataset layout.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath

import torch
from torch import Tensor

from emotion_model.common import LabelProtocol, binarize_emotion_scores
from emotion_model.data.manifest import (
    EmotionScores,
    MultimodalManifest,
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    TimedSourceRef,
    TimeInterval,
)
from emotion_model.data.source_adapters import (
    PhysioSourceAdapter,
    SpeechSourceAdapter,
    SpeechSourceData,
)
from emotion_model.data.splits import ParticipantSplit
from emotion_model.physiology.channel_metadata import (
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)

_PARTICIPANT_AUDIO_PATTERN = re.compile(r"^P(?P<pid>[1-9]\d*)\.wav$", re.IGNORECASE)
_DEBATE_AUDIO_PATTERN = re.compile(
    r"^p(?P<left>[1-9]\d*)\.p(?P<right>[1-9]\d*)\.wav$",
    re.IGNORECASE,
)
_SUPPORTED_PERSPECTIVES = frozenset({"self", "partner"})
_CHANNEL_FILES = {
    "bvp": ("e4_data", "E4_BVP", ".csv"),
    "eda": ("e4_data", "E4_EDA", ".csv"),
    "temperature": ("e4_data", "E4_TEMP", ".csv"),
    # ``ecg`` corresponds to the Polar H7 heart-rate signal stored in
    # ``Polar_HR.csv``.  It is a low-frequency HR sequence, not raw ECG.
    "ecg": ("neurosky_polar_data", "Polar_HR", ".csv"),
}
_DEFAULT_CHANNEL_NAMES = ("bvp", "eda", "temperature")


@dataclass(frozen=True)
class KEmoConManifestBuildReport:
    """Return a built manifest together with transparent preparation counts.

    Attributes:
        manifest: Portable multimodal manifest containing 5-second records.
        participant_count: Number of participants represented by at least one
            record.
        session_count: Number of complete dyadic debate session identifiers.
        record_count: Number of retained labeled windows.
        skipped_annotation_rows: Rows rejected for malformed scores/times.
        skipped_tail_windows: Otherwise valid rows extending beyond the
            participant-specific 16 kHz audio.
    """

    manifest: MultimodalManifest
    participant_count: int
    session_count: int
    record_count: int
    skipped_annotation_rows: int
    skipped_tail_windows: int


@dataclass(frozen=True)
class _SubjectTimes:
    participant_id: str
    debate_start_ms: float
    debate_end_ms: float


@dataclass(frozen=True)
class _WaveMetadata:
    sample_rate_hz: int
    frame_count: int
    channel_count: int
    sample_width_bytes: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate_hz


def kemocon_channel_specs(
    channel_names: Sequence[str] = _DEFAULT_CHANNEL_NAMES,
) -> tuple[PhysioChannelSpec, ...]:
    """Build explicit K-EmoCon physiology channel semantics.

    Args:
        channel_names: Ordered non-empty subset of ``bvp``, ``eda``,
            ``temperature``, and ``ecg``.

    Returns:
        Ordered immutable channel specifications. BVP uses arbitrary sensor
        units; ``ecg`` is declared as heart rate rather than raw ECG.
    """

    if isinstance(channel_names, (str, bytes)) or not isinstance(
        channel_names,
        Sequence,
    ):
        raise TypeError("channel_names must be a non-string sequence.")
    names = tuple(channel_names)
    if not names:
        raise ValueError("channel_names must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique.")
    unknown = sorted(set(names) - set(_CHANNEL_FILES))
    if unknown:
        raise ValueError(f"unsupported K-EmoCon physiology channels: {unknown}.")
    available = {
        "bvp": PhysioChannelSpec(
            "bvp",
            PhysioSignalKind.BVP,
            64.0,
            "a.u.",
        ),
        "eda": PhysioChannelSpec(
            "eda",
            PhysioSignalKind.EDA,
            4.0,
            "uS",
        ),
        "temperature": PhysioChannelSpec(
            "temperature",
            PhysioSignalKind.TEMPERATURE,
            4.0,
            "degC",
        ),
        "ecg": PhysioChannelSpec(
            "ecg",
            PhysioSignalKind.HEART_RATE,
            1.0,
            "bpm",
            "Polar H7 heart-rate signal stored in Polar_HR.csv; not raw ECG waveform.",
        ),
    }
    return tuple(available[name] for name in names)


def _safe_relative_path(root: Path, source_id: str) -> Path:
    pure = PurePosixPath(source_id)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("source_id must be a safe path relative to the data root.")
    candidate = root.joinpath(*pure.parts)
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("source_id resolves outside the configured data root.") from error
    return resolved_candidate


def _relative_source_id(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _wave_metadata(path: Path) -> _WaveMetadata:
    with wave.open(str(path), "rb") as stream:
        return _WaveMetadata(
            sample_rate_hz=stream.getframerate(),
            frame_count=stream.getnframes(),
            channel_count=stream.getnchannels(),
            sample_width_bytes=stream.getsampwidth(),
        )


def _read_subject_times(data_root: Path) -> dict[str, _SubjectTimes]:
    path = data_root / "metadata" / "subjects.csv"
    if not path.is_file():
        raise FileNotFoundError(f"K-EmoCon metadata is missing: {path}.")
    subjects: dict[str, _SubjectTimes] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                pid = int(row["pid"])
                start = float(row["startTime"])
                end = float(row["endTime"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid subjects.csv row: {row!r}.") from error
            if pid <= 0 or not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(f"invalid subjects.csv values: {row!r}.")
            if end <= start:
                raise ValueError(f"subject {pid} debate end must follow start.")
            participant_id = f"P{pid}"
            if participant_id in subjects:
                raise ValueError(f"duplicate subject metadata for {participant_id}.")
            subjects[participant_id] = _SubjectTimes(
                participant_id=participant_id,
                debate_start_ms=start,
                debate_end_ms=end,
            )
    if not subjects:
        raise ValueError("subjects.csv contains no participants.")
    return subjects


def _discover_sessions(data_root: Path) -> dict[str, str]:
    audio_dir = data_root / "debate_audios"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"K-EmoCon debate audio directory is missing: {audio_dir}.")
    participant_sessions: dict[str, str] = {}
    for path in sorted(audio_dir.glob("*.wav")):
        match = _DEBATE_AUDIO_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        left = f"P{int(match.group('left'))}"
        right = f"P{int(match.group('right'))}"
        session_id = f"{left}-{right}"
        for participant_id in (left, right):
            if participant_id in participant_sessions:
                raise ValueError(
                    f"participant {participant_id} appears in multiple debate audios."
                )
            participant_sessions[participant_id] = session_id
    if not participant_sessions:
        raise ValueError("no p<X>.p<Y>.wav debate sessions were discovered.")
    return participant_sessions


def _annotation_path(
    data_root: Path,
    participant_id: str,
    perspective: str,
) -> Path:
    suffix = "self" if perspective == "self" else "partner"
    return (
        data_root
        / "emotion_annotations"
        / f"{perspective}_annotations"
        / f"{participant_id}.{suffix}.csv"
    )


def _physio_path(data_root: Path, participant_id: str, channel_name: str) -> Path | None:
    participant_number = str(int(participant_id[1:]))
    directory_name, prefix, suffix = _CHANNEL_FILES[channel_name]
    directory = data_root / directory_name / participant_number
    if not directory.is_dir():
        return None
    exact = directory / f"{prefix}{suffix}"
    if exact.is_file():
        return exact
    candidates = sorted(directory.glob(f"{prefix}_*{suffix}"))
    if len(candidates) > 1:
        raise ValueError(
            f"multiple candidate files for {participant_id}/{channel_name}: "
            f"{[path.name for path in candidates]}."
        )
    return candidates[0] if candidates else None


def _validated_score(value: str, *, name: str) -> float:
    score = float(value)
    if not math.isfinite(score) or score != round(score) or not 1.0 <= score <= 5.0:
        raise ValueError(f"{name} must be an integer-valued score in [1, 5].")
    return score


def build_kemocon_manifest(
    data_root: str | Path,
    *,
    annotation_perspective: str = "self",
    channel_names: Sequence[str] = _DEFAULT_CHANNEL_NAMES,
    window_seconds: float = 5.0,
) -> KEmoConManifestBuildReport:
    """Build a portable manifest from an extracted K-EmoCon directory.

    Args:
        data_root: Extracted K-EmoCon root containing ``metadata``,
            ``pre_audio``, ``e4_data``, and ``emotion_annotations``.
        annotation_perspective: ``"self"`` or ``"partner"``. Perspectives
            are never mixed inside one manifest.
        channel_names: Ordered physiology channel selection.
        window_seconds: Positive annotation-window duration. Official
            annotations use five seconds.

    Returns:
        :class:`KEmoConManifestBuildReport` containing records whose logical
        windows are relative to each participant's debate audio start.

    K-EmoCon annotation rows whose end time exceeds the participant-specific
    16 kHz audio duration are explicitly dropped. The unrelated CASE files
    sometimes placed under ``annotations/`` are never read.
    """

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"K-EmoCon data root does not exist: {root}.")
    if annotation_perspective not in _SUPPORTED_PERSPECTIVES:
        raise ValueError(
            "annotation_perspective must be one of "
            f"{sorted(_SUPPORTED_PERSPECTIVES)}."
        )
    if isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float)):
        raise TypeError("window_seconds must be a real number, not bool.")
    duration = float(window_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("window_seconds must be finite and positive.")

    specs = kemocon_channel_specs(channel_names)
    subjects = _read_subject_times(root)
    sessions = _discover_sessions(root)
    records: list[MultimodalWindowRecord] = []
    represented_participants: set[str] = set()
    represented_sessions: set[str] = set()
    skipped_annotation_rows = 0
    skipped_tail_windows = 0

    for participant_id in sorted(subjects, key=lambda value: int(value[1:])):
        session_id = sessions.get(participant_id)
        if session_id is None:
            continue
        audio_path = root / "pre_audio" / f"{participant_id}.wav"
        annotation_path = _annotation_path(
            root,
            participant_id,
            annotation_perspective,
        )
        if not audio_path.is_file() or not annotation_path.is_file():
            continue
        audio = _wave_metadata(audio_path)
        if (
            audio.sample_rate_hz != 16000
            or audio.channel_count != 1
            or audio.sample_width_bytes != 2
        ):
            raise ValueError(
                f"{audio_path} must be mono 16-bit PCM at 16 kHz; "
                f"received channels={audio.channel_count}, "
                f"width={audio.sample_width_bytes}, rate={audio.sample_rate_hz}."
            )
        speech_source_id = _relative_source_id(root, audio_path)
        channel_paths = {
            spec.name: _physio_path(root, participant_id, spec.name)
            for spec in specs
        }
        with annotation_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            for row_index, row in enumerate(csv.DictReader(stream)):
                try:
                    end_seconds = float(row["seconds"])
                    arousal = _validated_score(row["arousal"], name="arousal")
                    valence = _validated_score(row["valence"], name="valence")
                    start_seconds = end_seconds - duration
                    if (
                        not math.isfinite(end_seconds)
                        or start_seconds < 0.0
                        or end_seconds <= start_seconds
                    ):
                        raise ValueError("invalid annotation interval.")
                except (KeyError, TypeError, ValueError):
                    skipped_annotation_rows += 1
                    continue
                tolerance = 0.5 / audio.sample_rate_hz
                if end_seconds > audio.duration_seconds + tolerance:
                    skipped_tail_windows += 1
                    continue
                interval = TimeInterval(start_seconds, end_seconds)
                speech = TimedSourceRef(speech_source_id, interval)
                physiology = tuple(
                    PhysioChannelSourceRef(
                        spec.name,
                        TimedSourceRef(
                            _relative_source_id(root, path),
                            interval,
                        ),
                    )
                    for spec in specs
                    if (path := channel_paths[spec.name]) is not None
                )
                sample_id = (
                    f"{participant_id}-{annotation_perspective}-"
                    f"{row_index:04d}"
                )
                records.append(
                    MultimodalWindowRecord(
                        sample_id=sample_id,
                        participant_id=participant_id,
                        session_id=session_id,
                        window=interval,
                        emotion_scores=EmotionScores(arousal, valence),
                        speech_source=speech,
                        physio_sources=physiology,
                    )
                )
                represented_participants.add(participant_id)
                represented_sessions.add(session_id)

    if not records:
        raise ValueError("no valid K-EmoCon records were produced.")
    manifest = MultimodalManifest(
        channel_specs=specs,
        records=tuple(records),
    )
    return KEmoConManifestBuildReport(
        manifest=manifest,
        participant_count=len(represented_participants),
        session_count=len(represented_sessions),
        record_count=len(records),
        skipped_annotation_rows=skipped_annotation_rows,
        skipped_tail_windows=skipped_tail_windows,
    )


class KEmoConSpeechAdapter(SpeechSourceAdapter):
    """Read only the requested window from participant-specific 16 kHz PCM.

    Args:
        data_root: K-EmoCon root used to resolve relative manifest source IDs.

    The returned waveform has shape ``[L]`` on CPU. No resampling, amplitude
    normalization, denoising, waveform reconstruction, or source separation is
    performed.
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"K-EmoCon data root does not exist: {self.data_root}."
            )

    def load_speech(self, source: TimedSourceRef) -> SpeechSourceData:
        """Load one declared interval as a mono floating waveform ``[L]``."""

        if not isinstance(source, TimedSourceRef):
            raise TypeError("source must be TimedSourceRef.")
        path = _safe_relative_path(self.data_root, source.source_id)
        interval = source.available_interval
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels() != 1
                or stream.getsampwidth() != 2
                or stream.getframerate() != 16000
            ):
                raise ValueError(
                    "K-EmoCon pre_audio must be mono 16-bit PCM at 16 kHz."
                )
            rate = stream.getframerate()
            start_frame = int(round(interval.start_seconds * rate))
            end_frame = int(round(interval.end_seconds * rate))
            if (
                start_frame < 0
                or end_frame <= start_frame
                or end_frame > stream.getnframes()
            ):
                raise ValueError(
                    f"speech interval {interval} falls outside {path.name}."
                )
            stream.setpos(start_frame)
            frame_bytes = stream.readframes(end_frame - start_frame)
        waveform = torch.frombuffer(
            bytearray(frame_bytes),
            dtype=torch.int16,
        ).to(dtype=torch.float32)
        waveform = (waveform / 32768.0).contiguous()
        return SpeechSourceData(
            waveform=waveform,
            sample_rate_hz=rate,
            timeline_start_seconds=start_frame / rate,
        )


class KEmoConPhysioAdapter(PhysioSourceAdapter):
    """Load K-EmoCon physiology CSV channels with participant-relative time.

    Args:
        data_root: K-EmoCon root used to resolve relative source IDs.

    Returned values, validity, and timestamps have shape ``[T]`` on CPU.
    CSV files are cached per adapter instance. Finite BVP zeros remain valid
    because zero crossings are expected; zeros in EDA, temperature, and heart
    rate are treated as missing according to the dataset quality convention.
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"K-EmoCon data root does not exist: {self.data_root}."
            )
        self._subject_times = _read_subject_times(self.data_root)
        self._cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    @staticmethod
    def _recommended_serial(participant_id: str) -> str | None:
        recommendations = {
            "P29": "A013E1",
            "P30": "A01A3A",
            "P31": "A013E1",
            "P32": "A01A3A",
        }
        return recommendations.get(participant_id)

    def _load_csv(
        self,
        source_id: str,
        spec: PhysioChannelSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cached = self._cache.get(source_id)
        if cached is not None:
            return cached
        path = _safe_relative_path(self.data_root, source_id)
        timestamps: list[float] = []
        values: list[float] = []
        participant_id: str | None = None
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(row)
                if participant_id is None:
                    try:
                        participant_id = f"P{int(row['pid'])}"
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"{path} has an invalid pid column.") from error
        if participant_id is None or participant_id not in self._subject_times:
            raise ValueError(f"{path} contains no participant recognized by subjects.csv.")
        recommended_serial = self._recommended_serial(participant_id)
        for row in rows:
            if (
                recommended_serial is not None
                and row.get("device_serial")
                and row["device_serial"] != recommended_serial
            ):
                continue
            try:
                timestamp_ms = float(row["timestamp"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(timestamp_ms):
                continue
            relative_seconds = (
                timestamp_ms
                - self._subject_times[participant_id].debate_start_ms
            ) / 1000.0
            timestamps.append(relative_seconds)
            values.append(value)
        if not timestamps:
            raise ValueError(f"{path} contains no usable timestamped rows.")

        order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        unique_timestamps: list[float] = []
        unique_values: list[float] = []
        duplicate_values: list[float] = []
        previous: float | None = None
        for index in order:
            timestamp = timestamps[index]
            if previous is not None and timestamp == previous:
                duplicate_values.append(values[index])
                continue
            if duplicate_values:
                finite_duplicates = [value for value in duplicate_values if math.isfinite(value)]
                unique_values.append(
                    sum(finite_duplicates) / len(finite_duplicates)
                    if finite_duplicates
                    else float("nan")
                )
            unique_timestamps.append(timestamp)
            duplicate_values = [values[index]]
            previous = timestamp
        if duplicate_values:
            finite_duplicates = [value for value in duplicate_values if math.isfinite(value)]
            unique_values.append(
                sum(finite_duplicates) / len(finite_duplicates)
                if finite_duplicates
                else float("nan")
            )
        timestamp_tensor = torch.tensor(unique_timestamps, dtype=torch.float64)
        value_tensor = torch.tensor(unique_values, dtype=torch.float32)
        valid = torch.isfinite(value_tensor)
        if spec.signal_kind in {
            PhysioSignalKind.EDA,
            PhysioSignalKind.TEMPERATURE,
            PhysioSignalKind.HEART_RATE,
        }:
            valid &= value_tensor != 0
        safe_values = torch.where(
            valid,
            value_tensor,
            torch.zeros_like(value_tensor),
        )
        result = (
            safe_values.contiguous(),
            valid.contiguous(),
            timestamp_tensor.contiguous(),
        )
        self._cache[source_id] = result
        return result

    def load_physio(
        self,
        source: PhysioChannelSourceRef,
        spec: PhysioChannelSpec,
    ) -> PhysioChannelWindow:
        """Load one explicit physiology interval as CPU tensors ``[T]``."""

        if not isinstance(source, PhysioChannelSourceRef):
            raise TypeError("source must be PhysioChannelSourceRef.")
        if not isinstance(spec, PhysioChannelSpec):
            raise TypeError("spec must be PhysioChannelSpec.")
        if source.channel_name != spec.name:
            raise ValueError("source channel_name must match spec.name.")
        values, valid, timestamps = self._load_csv(source.source.source_id, spec)
        interval = source.source.available_interval
        start = int(
            torch.searchsorted(
                timestamps,
                torch.tensor(interval.start_seconds, dtype=torch.float64),
                right=False,
            ).item()
        )
        end = int(
            torch.searchsorted(
                timestamps,
                torch.tensor(interval.end_seconds, dtype=torch.float64),
                right=False,
            ).item()
        )
        if end <= start:
            placeholder_time = torch.tensor(
                [interval.start_seconds],
                dtype=torch.float64,
            )
            return PhysioChannelWindow(
                spec=spec,
                values=torch.zeros(1, dtype=torch.float32),
                valid_mask=torch.zeros(1, dtype=torch.bool),
                timestamps_seconds=placeholder_time,
            )
        return PhysioChannelWindow(
            spec=spec,
            values=values[start:end].clone(),
            valid_mask=valid[start:end].clone(),
            timestamps_seconds=timestamps[start:end].clone(),
        )


def _stable_session_digest(session_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{session_id}".encode()).digest()


def build_kemocon_dyad_splits(
    manifest: MultimodalManifest,
    *,
    num_folds: int = 5,
    seed: int = 42,
) -> tuple[ParticipantSplit, ...]:
    """Build rotating splits that never divide a debate dyad between sets.

    Args:
        manifest: K-EmoCon manifest with one stable ``session_id`` per dyad.
        num_folds: Fold count from three through the number of sessions.
        seed: Deterministic integer ordering seed.

    Returns:
        Rotating participant splits. Fold ``i`` is test, fold ``i+1`` is
        validation, and remaining dyads train. Every participant belonging to
        a session follows that complete session into exactly one partition.
    """

    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be MultimodalManifest.")
    if isinstance(num_folds, bool) or not isinstance(num_folds, int):
        raise TypeError("num_folds must be an integer, not bool.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool.")
    session_participants: dict[str, set[str]] = {}
    participant_session: dict[str, str] = {}
    for record in manifest.records:
        previous = participant_session.setdefault(
            record.participant_id,
            record.session_id,
        )
        if previous != record.session_id:
            raise ValueError(
                f"participant {record.participant_id} occurs in multiple sessions."
            )
        session_participants.setdefault(record.session_id, set()).add(
            record.participant_id
        )
    sessions = sorted(
        session_participants,
        key=lambda value: (_stable_session_digest(value, seed=seed), value),
    )
    if num_folds < 3 or num_folds > len(sessions):
        raise ValueError(
            f"num_folds must lie in [3, {len(sessions)}] for this manifest."
        )
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for index, session_id in enumerate(sessions):
        folds[index % num_folds].append(session_id)

    def participants(session_ids: Sequence[str]) -> tuple[str, ...]:
        values = {
            participant_id
            for session_id in session_ids
            for participant_id in session_participants[session_id]
        }
        return tuple(sorted(values, key=lambda value: int(value[1:])))

    splits: list[ParticipantSplit] = []
    for test_index in range(num_folds):
        validation_index = (test_index + 1) % num_folds
        train_sessions = tuple(
            session_id
            for fold_index, fold in enumerate(folds)
            if fold_index not in {test_index, validation_index}
            for session_id in fold
        )
        splits.append(
            ParticipantSplit(
                train_participant_ids=participants(train_sessions),
                validation_participant_ids=participants(folds[validation_index]),
                test_participant_ids=participants(folds[test_index]),
            )
        )
    return tuple(splits)


def _label_balance_vector(
    arousal_label: int,
    valence_label: int,
    *,
    arousal_valid: bool,
    valence_valid: bool,
) -> tuple[int, ...]:
    """Return record/arousal/valence/quadrant count increments ``[9]``."""

    quadrant_valid = arousal_valid and valence_valid
    quadrant = arousal_label + 2 * valence_label
    return (
        1,
        int(arousal_valid and arousal_label == 0),
        int(arousal_valid and arousal_label == 1),
        int(valence_valid and valence_label == 0),
        int(valence_valid and valence_label == 1),
        *(int(quadrant_valid and quadrant == value) for value in range(4)),
    )


def _add_balance_vectors(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Add two internal label count vectors ``[9]`` elementwise."""

    return tuple(a + b for a, b in zip(left, right, strict=True))


def _label_stratification_score(
    folds: Sequence[Sequence[str]],
    session_vectors: dict[str, tuple[int, ...]],
    totals: tuple[int, ...],
    *,
    session_count: int,
) -> float:
    """Measure normalized record and label-count deviation across folds."""

    score = 0.0
    for fold in folds:
        fold_vector = (0,) * len(totals)
        for session_id in fold:
            fold_vector = _add_balance_vectors(
                fold_vector,
                session_vectors[session_id],
            )
        target_fraction = len(fold) / session_count
        for index, (actual, total) in enumerate(
            zip(fold_vector, totals, strict=True)
        ):
            expected = total * target_fraction
            deviation = (actual - expected) / max(expected, 1.0)
            weight = 2.0 if index >= 5 else 1.0
            score += weight * deviation * deviation
    return score


def _folds_have_complete_label_coverage(
    folds: Sequence[Sequence[str]],
    session_vectors: dict[str, tuple[int, ...]],
) -> bool:
    """Return whether every fold contains both task classes and four quadrants."""

    for fold in folds:
        vector = (0,) * 9
        for session_id in fold:
            vector = _add_balance_vectors(vector, session_vectors[session_id])
        if any(value == 0 for value in vector[1:]):
            return False
    return True


def build_kemocon_label_stratified_dyad_splits(
    manifest: MultimodalManifest,
    *,
    label_protocol: LabelProtocol,
    num_folds: int = 5,
    seed: int = 42,
) -> tuple[ParticipantSplit, ...]:
    """Build label-balanced rotating folds without dividing debate dyads.

    Args:
        manifest: K-EmoCon manifest with one stable ``session_id`` per dyad.
        label_protocol: Named mapping applied to the arousal/valence scores.
        num_folds: Fold count from three through the number of sessions.
        seed: Deterministic integer candidate-order seed.

    Returns:
        Rotating participant splits. Fold ``i`` is test, fold ``i+1`` is
        validation, and the remaining complete dyads train. Every individual
        fold contains both arousal classes, both valence classes, and all four
        quadrants under ``label_protocol``.

    The search balances record count and task/quadrant counts. It rejects a
    dataset when grouped coverage is impossible, and never silently returns a
    validation or test fold with a missing class or quadrant.
    """

    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be MultimodalManifest.")
    if not isinstance(label_protocol, LabelProtocol):
        raise TypeError("label_protocol must be LabelProtocol.")
    if isinstance(num_folds, bool) or not isinstance(num_folds, int):
        raise TypeError("num_folds must be an integer, not bool.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool.")

    session_participants: dict[str, set[str]] = {}
    participant_session: dict[str, str] = {}
    session_records: dict[str, list[MultimodalWindowRecord]] = {}
    for record in manifest.records:
        previous = participant_session.setdefault(
            record.participant_id,
            record.session_id,
        )
        if previous != record.session_id:
            raise ValueError(
                f"participant {record.participant_id} occurs in multiple sessions."
            )
        session_participants.setdefault(record.session_id, set()).add(
            record.participant_id
        )
        session_records.setdefault(record.session_id, []).append(record)

    sessions = tuple(sorted(session_records))
    if num_folds < 3 or num_folds > len(sessions):
        raise ValueError(
            f"num_folds must lie in [3, {len(sessions)}] for this manifest."
        )

    session_vectors: dict[str, tuple[int, ...]] = {}
    for session_id in sessions:
        records = session_records[session_id]
        scores = torch.tensor(
            [
                [record.emotion_scores.arousal, record.emotion_scores.valence]
                for record in records
            ],
            dtype=torch.float32,
        )
        labels, valid = binarize_emotion_scores(scores, label_protocol)
        vector = (0,) * 9
        for label_pair, valid_pair in zip(labels, valid, strict=True):
            vector = _add_balance_vectors(
                vector,
                _label_balance_vector(
                    int(label_pair[0].item()),
                    int(label_pair[1].item()),
                    arousal_valid=bool(valid_pair[0].item()),
                    valence_valid=bool(valid_pair[1].item()),
                ),
            )
        session_vectors[session_id] = vector

    totals = (0,) * 9
    for vector in session_vectors.values():
        totals = _add_balance_vectors(totals, vector)
    missing_global = [
        index
        for index, value in enumerate(totals[1:], start=1)
        if value == 0
    ]
    if missing_global:
        raise ValueError(
            "label-stratified folds require both task classes and all four "
            f"quadrants globally; missing balance-vector indices {missing_global}."
        )
    insufficient_session_coverage = [
        index
        for index in range(1, 9)
        if sum(session_vectors[session_id][index] > 0 for session_id in sessions)
        < num_folds
    ]
    if insufficient_session_coverage:
        raise ValueError(
            "dyad grouping cannot place every class and quadrant in every fold; "
            "insufficient session coverage at balance-vector indices "
            f"{insufficient_session_coverage}."
        )

    base_size, remainder = divmod(len(sessions), num_folds)
    fold_order = sorted(
        range(num_folds),
        key=lambda index: (
            _stable_session_digest(f"fold-{index}", seed=seed),
            index,
        ),
    )
    capacities = [base_size] * num_folds
    for index in fold_order[:remainder]:
        capacities[index] += 1

    best_score = math.inf
    best_folds: tuple[tuple[str, ...], ...] | None = None
    for trial in range(8192):
        ordered = sorted(
            sessions,
            key=lambda session_id: (
                hashlib.sha256(
                    f"{seed}\0{trial}\0{session_id}".encode()
                ).digest(),
                session_id,
            ),
        )
        candidate: list[tuple[str, ...]] = []
        offset = 0
        for capacity in capacities:
            candidate.append(tuple(sorted(ordered[offset : offset + capacity])))
            offset += capacity
        candidate_folds = tuple(candidate)
        if not _folds_have_complete_label_coverage(
            candidate_folds,
            session_vectors,
        ):
            continue
        score = _label_stratification_score(
            candidate_folds,
            session_vectors,
            totals,
            session_count=len(sessions),
        )
        if (score, candidate_folds) < (best_score, best_folds or candidate_folds):
            best_score = score
            best_folds = candidate_folds

    if best_folds is None:
        raise RuntimeError(
            "failed to construct dyad-safe folds with complete label coverage; "
            "choose fewer folds or inspect per-dyad label distributions."
        )

    def participants(session_ids: Sequence[str]) -> tuple[str, ...]:
        values = {
            participant_id
            for session_id in session_ids
            for participant_id in session_participants[session_id]
        }
        return tuple(sorted(values, key=lambda value: int(value[1:])))

    splits: list[ParticipantSplit] = []
    for test_index in range(num_folds):
        validation_index = (test_index + 1) % num_folds
        train_sessions = tuple(
            session_id
            for index, fold in enumerate(best_folds)
            if index not in {test_index, validation_index}
            for session_id in fold
        )
        splits.append(
            ParticipantSplit(
                train_participant_ids=participants(train_sessions),
                validation_participant_ids=participants(
                    best_folds[validation_index]
                ),
                test_participant_ids=participants(best_folds[test_index]),
            )
        )
    return tuple(splits)


def build_kemocon_dyad_ratio_split(
    manifest: MultimodalManifest,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> ParticipantSplit:
    """Build one deterministic ratio split without dividing debate dyads.

    Args:
        manifest: K-EmoCon manifest with one stable ``session_id`` per dyad.
        train_fraction: Requested positive fraction assigned to training.
        validation_fraction: Requested positive fraction assigned to validation.
        test_fraction: Requested positive fraction assigned to testing.
        seed: Deterministic integer ordering seed.

    Returns:
        One participant split whose positive integer session counts minimize
        squared deviation from the requested quotas. Fractions therefore
        remain targets when complete dyads cannot represent them exactly.

    Session counts depend only on the requested fractions. For the small
    K-EmoCon session set, assignment ranks record-count and declared
    modality-pattern balance while requiring validation/test coverage for
    patterns present in at least three sessions. The seed selects one of the
    32 best balanced assignments, so repeated seeds produce meaningful fresh
    dyad partitions without using labels.
    """

    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be MultimodalManifest.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool.")
    fractions: tuple[float, float, float] = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    for name, value in zip(
        ("train_fraction", "validation_fraction", "test_fraction"),
        fractions,
        strict=True,
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, not bool.")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    normalized_fractions = tuple(float(value) for value in fractions)
    if not math.isclose(
        sum(normalized_fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("train/validation/test fractions must sum to 1.")

    session_participants: dict[str, set[str]] = {}
    session_record_counts: dict[str, int] = {}
    session_pattern_counts: dict[str, list[int]] = {}
    participant_session: dict[str, str] = {}
    for record in manifest.records:
        previous = participant_session.setdefault(
            record.participant_id,
            record.session_id,
        )
        if previous != record.session_id:
            raise ValueError(
                f"participant {record.participant_id} occurs in multiple sessions."
            )
        session_participants.setdefault(record.session_id, set()).add(
            record.participant_id
        )
        session_record_counts[record.session_id] = (
            session_record_counts.get(record.session_id, 0) + 1
        )
        pattern_counts = session_pattern_counts.setdefault(
            record.session_id,
            [0, 0, 0],
        )
        if record.speech_available and record.physiology_available:
            pattern_counts[0] += 1
        elif record.speech_available:
            pattern_counts[1] += 1
        elif record.physiology_available:
            pattern_counts[2] += 1
    sessions = sorted(
        session_participants,
        key=lambda value: (_stable_session_digest(value, seed=seed), value),
    )
    if len(sessions) < 3:
        raise ValueError("ratio splitting requires at least three sessions.")

    quotas = tuple(len(sessions) * fraction for fraction in normalized_fractions)
    candidates = (
        (
            train_count,
            validation_count,
            len(sessions) - train_count - validation_count,
        )
        for train_count in range(1, len(sessions) - 1)
        for validation_count in range(1, len(sessions) - train_count)
    )
    counts = min(
        candidates,
        key=lambda values: (
            sum(
                (count - quota) ** 2
                for count, quota in zip(values, quotas, strict=True)
            ),
            max(
                abs(count - quota)
                for count, quota in zip(values, quotas, strict=True)
            ),
            -values[0],
            -values[1],
        ),
    )

    train_sessions: Sequence[str]
    validation_sessions: Sequence[str]
    test_sessions: Sequence[str]
    if len(sessions) <= 20:
        indices = tuple(range(len(sessions)))
        total_records = sum(session_record_counts.values())
        total_patterns = tuple(
            sum(values[index] for values in session_pattern_counts.values())
            for index in range(3)
        )
        sessions_with_pattern = tuple(
            sum(
                session_pattern_counts[session_id][index] > 0
                for session_id in sessions
            )
            for index in range(3)
        )
        ranked_assignments: list[
            tuple[
                float,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        for validation_indices in combinations(indices, counts[1]):
            validation_set = set(validation_indices)
            remaining_indices = tuple(
                index for index in indices if index not in validation_set
            )
            for test_indices in combinations(remaining_indices, counts[2]):
                test_set = set(test_indices)
                train_indices = tuple(
                    index
                    for index in remaining_indices
                    if index not in test_set
                )
                partitions = (
                    train_indices,
                    validation_indices,
                    test_indices,
                )
                score = 0.0
                partition_records = tuple(
                    sum(
                        session_record_counts[sessions[index]]
                        for index in partition
                    )
                    for partition in partitions
                )
                score += sum(
                    (
                        record_count / total_records
                        - normalized_fractions[index]
                    )
                    ** 2
                    for index, record_count in enumerate(partition_records)
                )
                for pattern_index, pattern_total in enumerate(total_patterns):
                    if pattern_total == 0:
                        continue
                    partition_patterns = tuple(
                        sum(
                            session_pattern_counts[sessions[index]][
                                pattern_index
                            ]
                            for index in partition
                        )
                        for partition in partitions
                    )
                    score += 0.5 * sum(
                        (
                            pattern_count / pattern_total
                            - normalized_fractions[index]
                        )
                        ** 2
                        for index, pattern_count in enumerate(
                            partition_patterns
                        )
                    )
                    if sessions_with_pattern[pattern_index] >= 3:
                        score += 1000.0 * sum(
                            pattern_count == 0
                            for pattern_count in partition_patterns
                        )
                ranked_assignments.append(
                    (
                        score,
                        tuple(sessions[index] for index in train_indices),
                        tuple(
                            sessions[index] for index in validation_indices
                        ),
                        tuple(sessions[index] for index in test_indices),
                    )
                )
        if not ranked_assignments:
            raise RuntimeError("failed to assign ratio split sessions.")
        ranked_assignments.sort(
            key=lambda value: (
                value[0],
                tuple(sorted(value[2])),
                tuple(sorted(value[3])),
            )
        )
        shortlist = ranked_assignments[: min(32, len(ranked_assignments))]

        def seeded_assignment_digest(
            assignment: tuple[
                float,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ],
        ) -> str:
            payload = "|".join(
                (
                    str(seed),
                    ",".join(sorted(assignment[1])),
                    ",".join(sorted(assignment[2])),
                    ",".join(sorted(assignment[3])),
                )
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        selected = min(shortlist, key=seeded_assignment_digest)
        train_sessions = selected[1]
        validation_sessions = selected[2]
        test_sessions = selected[3]
    else:
        train_end = counts[0]
        validation_end = train_end + counts[1]
        train_sessions = sessions[:train_end]
        validation_sessions = sessions[train_end:validation_end]
        test_sessions = sessions[validation_end:]

    def participants(session_ids: Sequence[str]) -> tuple[str, ...]:
        values = {
            participant_id
            for session_id in session_ids
            for participant_id in session_participants[session_id]
        }
        return tuple(sorted(values, key=lambda value: int(value[1:])))

    return ParticipantSplit(
        train_participant_ids=participants(train_sessions),
        validation_participant_ids=participants(validation_sessions),
        test_participant_ids=participants(test_sessions),
    )


__all__ = [
    "KEmoConManifestBuildReport",
    "KEmoConPhysioAdapter",
    "KEmoConSpeechAdapter",
    "build_kemocon_dyad_ratio_split",
    "build_kemocon_dyad_splits",
    "build_kemocon_manifest",
    "kemocon_channel_specs",
]
