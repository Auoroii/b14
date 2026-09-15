"""Structured, portable experiment artifacts for K-EmoCon runs."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import transformers
import yaml  # type: ignore[import-untyped]

from emotion_model.common import LabelProtocol, binarize_emotion_scores
from emotion_model.data import (
    AlignedMultimodalBatch,
    AlignedMultimodalSample,
    MultimodalWindowRecord,
    PartitionedManifest,
)
from emotion_model.evaluation import (
    ClassificationMetrics,
    EmotionTaskMetrics,
    EvaluationPredictions,
    ParticipantIndependentEvaluationOutput,
    SpeechActivityBin,
    compute_speech_activity_strata,
    speech_activity_bin_masks,
)


@dataclass
class FullWindowSpeechAvailabilityStatistics:
    """Accumulate reporting-only full-window speech availability counts.

    ``observe`` accepts one natural collated batch with logical masks ``[B]``
    and, optionally, the corresponding batch after explicit modality dropout.
    Activity ratios are counted only for records whose speech source is
    declared present. No tensor or model input is modified.
    """

    total_records: int = 0
    speech_source_present: int = 0
    speech_activity_ratio_observed: int = 0
    speech_activity_ratio_eq_0: int = 0
    speech_activity_ratio_0_to_0_1: int = 0
    speech_activity_ratio_0_1_to_0_25: int = 0
    speech_activity_ratio_ge_0_25: int = 0
    speech_source_unavailable: int = 0
    artificial_speech_modality_dropout: int = 0

    def observe(
        self,
        natural_batch: AlignedMultimodalBatch,
        effective_batch: AlignedMultimodalBatch | None = None,
    ) -> None:
        """Add counts from natural and optional post-dropout masks ``[B]``.

        Args:
            natural_batch: Collated batch before artificial modality dropout.
            effective_batch: Same logical records after artificial dropout, or
                ``None`` when no dropout transform was applied.

        Returns:
            ``None``. Only Python integer counters on this object are updated.
        """

        if not isinstance(natural_batch, AlignedMultimodalBatch):
            raise TypeError("natural_batch must be AlignedMultimodalBatch.")
        effective = natural_batch if effective_batch is None else effective_batch
        if not isinstance(effective, AlignedMultimodalBatch):
            raise TypeError("effective_batch must be AlignedMultimodalBatch or None.")
        if effective.records != natural_batch.records:
            raise ValueError(
                "effective_batch must contain the same ordered logical records."
            )
        if bool((effective.speech_available & ~natural_batch.speech_available).any()):
            raise ValueError("effective_batch cannot create speech availability.")

        source_present = torch.tensor(
            [record.speech_source is not None for record in natural_batch.records],
            dtype=torch.bool,
        )
        self.total_records += len(natural_batch.records)
        self.speech_source_present += int(source_present.sum().item())
        self.speech_source_unavailable += int((~source_present).sum().item())
        self.artificial_speech_modality_dropout += int(
            (
                natural_batch.speech_available
                & ~effective.speech_available
            ).sum().item()
        )

        ratios = natural_batch.speech_activity_ratios
        if ratios is None:
            return
        observed = natural_batch.speech_activity_observed
        if observed is None:
            raise RuntimeError(
                "activity ratios require an aligned observation mask."
            )
        bin_masks = speech_activity_bin_masks(
            ratios,
            source_present & observed,
        )
        self.speech_activity_ratio_observed += int(
            sum(mask.sum().item() for mask in bin_masks.values())
        )
        self.speech_activity_ratio_eq_0 += int(
            bin_masks[SpeechActivityBin.RATIO_EQ_0].sum().item()
        )
        self.speech_activity_ratio_0_to_0_1 += int(
            bin_masks[SpeechActivityBin.RATIO_0_TO_0_1].sum().item()
        )
        self.speech_activity_ratio_0_1_to_0_25 += int(
            bin_masks[SpeechActivityBin.RATIO_0_1_TO_0_25].sum().item()
        )
        self.speech_activity_ratio_ge_0_25 += int(
            (
                bin_masks[SpeechActivityBin.RATIO_0_25_TO_0_5].sum()
                + bin_masks[SpeechActivityBin.RATIO_GE_0_5].sum()
            ).item()
        )

    def to_summary(self) -> dict[str, int]:
        """Return a fresh JSON-compatible mapping of all scalar counts."""

        return {
            "total_records": self.total_records,
            "speech_source_present": self.speech_source_present,
            "speech_activity_ratio_observed": (
                self.speech_activity_ratio_observed
            ),
            "speech_activity_ratio_eq_0": self.speech_activity_ratio_eq_0,
            "speech_activity_ratio_0_to_0_1": (
                self.speech_activity_ratio_0_to_0_1
            ),
            "speech_activity_ratio_0_1_to_0_25": (
                self.speech_activity_ratio_0_1_to_0_25
            ),
            "speech_activity_ratio_ge_0_25": (
                self.speech_activity_ratio_ge_0_25
            ),
            "speech_source_unavailable": self.speech_source_unavailable,
            "artificial_speech_modality_dropout": (
                self.artificial_speech_modality_dropout
            ),
        }


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one UTF-8 JSON object to ``path``.

    Args:
        path: Output file whose parent directory already exists.
        payload: JSON-serializable string-key mapping without tensors.

    Raises:
        TypeError: If public inputs have invalid types.
        FileNotFoundError: If the output parent directory does not exist.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"artifact parent does not exist: {path.parent}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Append and flush one compact JSON object to a UTF-8 JSONL artifact."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"artifact parent does not exist: {path.parent}")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_effective_config_snapshot(
    source_path: Path,
    destination_path: Path,
    *,
    fold_index: int,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Write effective YAML configuration with runtime overrides applied.

    Args:
        source_path: Existing source YAML configuration.
        destination_path: Output YAML path whose parent already exists.
        fold_index: Non-negative effective fold index.
        overrides: Optional section-to-field mappings applied after the fold
            override. Values must be YAML-serializable scalars or containers.

    Returns:
        ``None``. The output is written atomically and contains no tensors.
    """

    if not isinstance(source_path, Path) or not source_path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {source_path}")
    if not isinstance(destination_path, Path):
        raise TypeError("destination_path must be a Path.")
    if not isinstance(fold_index, int) or isinstance(fold_index, bool):
        raise TypeError("fold_index must be an integer.")
    if fold_index < 0:
        raise ValueError("fold_index must be non-negative.")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping or None.")
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("split"), dict):
        raise ValueError("configuration must contain a split mapping.")
    raw["split"]["fold_index"] = fold_index
    if overrides is not None:
        for section_name, section_overrides in overrides.items():
            if not isinstance(section_name, str):
                raise TypeError("override section names must be strings.")
            if not isinstance(section_overrides, Mapping):
                raise TypeError("each override section must be a mapping.")
            section = raw.get(section_name)
            if not isinstance(section, dict):
                raise ValueError(
                    f"configuration override references missing section "
                    f"{section_name!r}."
                )
            for field_name, value in section_overrides.items():
                if not isinstance(field_name, str):
                    raise TypeError("override field names must be strings.")
                if field_name not in section:
                    raise ValueError(
                        f"configuration override references missing field "
                        f"{section_name}.{field_name}."
                    )
                section[field_name] = value
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                raw,
                stream,
                allow_unicode=True,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_environment_summary(device: torch.device) -> dict[str, object]:
    """Return JSON-compatible Python, dependency, CUDA, and GPU information."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device.")
    cuda_available = torch.cuda.is_available()
    gpus: list[dict[str, object]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )
    return {
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_device": str(device),
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


_QUADRANT_NAMES = ("LALV", "HALV", "LAHV", "HAHV")
_DECLARED_MODALITY_PATTERNS = (
    "both",
    "speech_only",
    "physiology_only",
    "neither",
)


def _label_distribution(
    records: Sequence[MultimodalWindowRecord],
    label_protocol: LabelProtocol,
) -> dict[str, object]:
    """Count binary tasks and four quadrants under one named protocol."""

    arousal_scores = torch.tensor(
        [record.emotion_scores.arousal for record in records],
        dtype=torch.float64,
    )
    valence_scores = torch.tensor(
        [record.emotion_scores.valence for record in records],
        dtype=torch.float64,
    )
    arousal_labels, arousal_valid = binarize_emotion_scores(
        arousal_scores,
        label_protocol,
    )
    valence_labels, valence_valid = binarize_emotion_scores(
        valence_scores,
        label_protocol,
    )
    joint_valid = arousal_valid & valence_valid
    quadrant_labels = arousal_labels + 2 * valence_labels

    def binary_counts(labels: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
        return {
            "low": int(((labels == 0) & valid).sum().item()),
            "high": int(((labels == 1) & valid).sum().item()),
            "ignored": int((~valid).sum().item()),
        }

    return {
        "record_count": len(records),
        "arousal": binary_counts(arousal_labels, arousal_valid),
        "valence": binary_counts(valence_labels, valence_valid),
        "quadrant": {
            **{
                name: int(((quadrant_labels == index) & joint_valid).sum().item())
                for index, name in enumerate(_QUADRANT_NAMES)
            },
            "ignored": int((~joint_valid).sum().item()),
        },
    }


def _declared_modality_pattern(record: MultimodalWindowRecord) -> str:
    if record.speech_available and record.physiology_available:
        return "both"
    if record.speech_available:
        return "speech_only"
    if record.physiology_available:
        return "physiology_only"
    return "neither"


def _record_partition_summary(
    records: Sequence[MultimodalWindowRecord],
    participant_ids: Sequence[str],
    label_protocol: LabelProtocol,
) -> dict[str, object]:
    session_ids = sorted({record.session_id for record in records})
    speech_count = sum(record.speech_available for record in records)
    physiology_count = sum(record.physiology_available for record in records)
    both_count = sum(
        record.speech_available and record.physiology_available
        for record in records
    )
    speech_only_count = sum(
        record.speech_available and not record.physiology_available
        for record in records
    )
    physiology_only_count = sum(
        not record.speech_available and record.physiology_available
        for record in records
    )
    neither_count = (
        len(records) - both_count - speech_only_count - physiology_only_count
    )
    participant_records = {
        participant_id: tuple(
            record
            for record in records
            if record.participant_id == participant_id
        )
        for participant_id in participant_ids
    }
    modality_records = {
        pattern: tuple(
            record
            for record in records
            if _declared_modality_pattern(record) == pattern
        )
        for pattern in _DECLARED_MODALITY_PATTERNS
    }
    return {
        "record_count": len(records),
        "participant_count": len(participant_ids),
        "participant_ids": list(participant_ids),
        "session_count": len(session_ids),
        "session_ids": session_ids,
        "speech_declared_count": speech_count,
        "physiology_declared_count": physiology_count,
        "modality_pattern_counts": {
            "both": both_count,
            "speech_only": speech_only_count,
            "physiology_only": physiology_only_count,
            "neither": neither_count,
        },
        "label_distribution": _label_distribution(records, label_protocol),
        "participant_label_distributions": {
            participant_id: _label_distribution(
                participant_records[participant_id],
                label_protocol,
            )
            for participant_id in participant_ids
        },
        "declared_modality_label_distributions": {
            pattern: _label_distribution(
                modality_records[pattern],
                label_protocol,
            )
            for pattern in _DECLARED_MODALITY_PATTERNS
        },
    }


def _availability_pattern(speech: bool, physiology: bool) -> str:
    if speech and physiology:
        return "both"
    if speech:
        return "speech_only"
    if physiology:
        return "physiology_only"
    return "neither"


def _empty_runtime_label_distribution() -> dict[str, object]:
    return {
        "record_count": 0,
        "arousal": {"low": 0, "high": 0, "ignored": 0},
        "valence": {"low": 0, "high": 0, "ignored": 0},
        "quadrant": {
            "LALV": 0,
            "HALV": 0,
            "LAHV": 0,
            "HAHV": 0,
            "ignored": 0,
        },
    }


def _increment_runtime_label_distribution(
    distribution: dict[str, object],
    sample: AlignedMultimodalSample,
) -> None:
    distribution["record_count"] = int(distribution["record_count"]) + 1
    for task, label in (
        ("arousal", sample.arousal_label),
        ("valence", sample.valence_label),
    ):
        counts = cast(dict[str, int], distribution[task])
        name = {0: "low", 1: "high"}.get(label, "ignored")
        counts[name] += 1
    quadrant = cast(dict[str, int], distribution["quadrant"])
    quadrant_name = {
        0: "LALV",
        1: "HALV",
        2: "LAHV",
        3: "HAHV",
    }.get(sample.quadrant_label, "ignored")
    quadrant[quadrant_name] += 1


def build_runtime_availability_audit(
    samples: Iterable[AlignedMultimodalSample],
) -> dict[str, object]:
    """Compare manifest declarations with loaded runtime availability.

    Args:
        samples: One complete pass of aligned samples. Speech tensors are
            ``[L]`` and physiology tensors/masks are ``[T,C]``; this function
            reads only metadata and boolean availability fields.

    Returns:
        JSON-compatible declaration-to-runtime modality transitions, mismatch
        sample IDs, per-participant counts, and per-channel physiology counts.
        Input samples and tensors are not modified.
    """

    transition_counts = {
        declared: {runtime: 0 for runtime in _DECLARED_MODALITY_PATTERNS}
        for declared in _DECLARED_MODALITY_PATTERNS
    }
    modality_counts = {
        name: {
            "declared_count": 0,
            "runtime_available_count": 0,
            "declared_but_runtime_unavailable_count": 0,
            "runtime_available_without_declaration_count": 0,
        }
        for name in ("speech", "physiology")
    }
    participant_counts: dict[str, dict[str, object]] = {}
    declared_channel_counts: dict[str, int] = {}
    runtime_channel_counts: dict[str, int] = {}
    mismatch_sample_ids: list[str] = []
    runtime_label_distributions = {
        pattern: _empty_runtime_label_distribution()
        for pattern in _DECLARED_MODALITY_PATTERNS
    }
    mismatch_label_distribution = _empty_runtime_label_distribution()
    seen_sample_ids: set[str] = set()
    record_count = 0

    for sample in samples:
        if not isinstance(sample, AlignedMultimodalSample):
            raise TypeError("samples must contain AlignedMultimodalSample objects.")
        sample_id = sample.record.sample_id
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate runtime audit sample_id: {sample_id!r}.")
        seen_sample_ids.add(sample_id)
        record_count += 1

        declared_speech = sample.record.speech_source is not None
        declared_physiology = bool(sample.record.physio_sources)
        runtime_speech = sample.speech_available
        runtime_physiology = sample.physiology_available
        declared_pattern = _availability_pattern(
            declared_speech,
            declared_physiology,
        )
        runtime_pattern = _availability_pattern(runtime_speech, runtime_physiology)
        transition_counts[declared_pattern][runtime_pattern] += 1
        _increment_runtime_label_distribution(
            runtime_label_distributions[runtime_pattern],
            sample,
        )
        if declared_pattern != runtime_pattern:
            mismatch_sample_ids.append(sample_id)
            _increment_runtime_label_distribution(
                mismatch_label_distribution,
                sample,
            )

        participant = participant_counts.setdefault(
            sample.record.participant_id,
            {
                "record_count": 0,
                "declared_physiology_count": 0,
                "runtime_physiology_available_count": 0,
                "physiology_mismatch_count": 0,
            },
        )
        participant["record_count"] = int(participant["record_count"]) + 1

        for name, declared, runtime in (
            ("speech", declared_speech, runtime_speech),
            ("physiology", declared_physiology, runtime_physiology),
        ):
            counts = modality_counts[name]
            counts["declared_count"] += int(declared)
            counts["runtime_available_count"] += int(runtime)
            counts["declared_but_runtime_unavailable_count"] += int(
                declared and not runtime
            )
            counts["runtime_available_without_declaration_count"] += int(
                runtime and not declared
            )
        participant["declared_physiology_count"] = int(
            participant["declared_physiology_count"]
        ) + int(declared_physiology)
        participant["runtime_physiology_available_count"] = int(
            participant["runtime_physiology_available_count"]
        ) + int(runtime_physiology)
        participant["physiology_mismatch_count"] = int(
            participant["physiology_mismatch_count"]
        ) + int(declared_physiology != runtime_physiology)

        declared_channels = {
            source.channel_name for source in sample.record.physio_sources
        }
        for channel_name in declared_channels:
            declared_channel_counts[channel_name] = (
                declared_channel_counts.get(channel_name, 0) + 1
            )
        if sample.physio_channel_mask is not None:
            for channel_name, available in zip(
                sample.channel_names,
                sample.physio_channel_mask.tolist(),
                strict=True,
            ):
                runtime_channel_counts[channel_name] = (
                    runtime_channel_counts.get(channel_name, 0) + int(available)
                )

    channel_names = sorted(set(declared_channel_counts) | set(runtime_channel_counts))
    return {
        "record_count": record_count,
        "modality_counts": modality_counts,
        "declared_to_runtime_pattern_counts": transition_counts,
        "mismatch_count": len(mismatch_sample_ids),
        "mismatch_sample_ids": mismatch_sample_ids,
        "runtime_modality_label_distributions": runtime_label_distributions,
        "mismatch_label_distribution": mismatch_label_distribution,
        "participant_counts": {
            participant_id: participant_counts[participant_id]
            for participant_id in sorted(participant_counts)
        },
        "physiology_channel_counts": {
            channel_name: {
                "declared_count": declared_channel_counts.get(channel_name, 0),
                "runtime_available_count": runtime_channel_counts.get(
                    channel_name,
                    0,
                ),
            }
            for channel_name in channel_names
        },
    }


def build_cross_fold_evaluation_summary(
    fold_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate completed participant-independent fold metrics.

    Args:
        fold_summaries: Per-fold evaluation mappings containing scalar overall
            arousal/valence metrics and disjoint ``participant_ids`` lists.

    Returns:
        JSON-compatible per-fold values plus population mean and standard
        deviation. Duplicate fold indices or test participants are rejected.
    """

    if not fold_summaries:
        raise ValueError("fold_summaries must be non-empty.")
    metric_paths = tuple(
        (task, metric)
        for task in ("arousal", "valence")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy")
    )
    values: dict[str, list[float]] = {
        f"overall.{task}.{metric}": [] for task, metric in metric_paths
    }
    for task in ("arousal", "valence"):
        for label in ("low", "high"):
            for metric in ("recall", "f1", "support"):
                values[f"overall.{task}.per_class.{label}.{metric}"] = []

    fold_indices: list[int] = []
    test_participants: list[str] = []
    fold_records: list[dict[str, object]] = []
    for summary in fold_summaries:
        fold_index = summary.get("fold_index")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise TypeError("every fold summary must contain an integer fold_index.")
        if fold_index in fold_indices:
            raise ValueError(f"duplicate fold_index: {fold_index}.")
        fold_indices.append(fold_index)
        raw_participants = summary.get("participant_ids")
        if not isinstance(raw_participants, list) or not all(
            isinstance(value, str) for value in raw_participants
        ):
            raise TypeError("every fold summary must contain string participant_ids.")
        participants = cast(list[str], raw_participants)
        duplicate_participants = set(test_participants) & set(participants)
        if duplicate_participants:
            raise ValueError(
                "test participants occur in multiple folds: "
                f"{sorted(duplicate_participants)}."
            )
        test_participants.extend(participants)
        overall = summary.get("overall")
        if not isinstance(overall, Mapping):
            raise TypeError("every fold summary must contain an overall mapping.")
        per_fold_metrics: dict[str, float] = {}
        for path in values:
            current: object = overall
            for component in path.split(".")[1:]:
                if not isinstance(current, Mapping) or component not in current:
                    raise ValueError(f"fold {fold_index} is missing metric {path}.")
                current = current[component]
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                raise TypeError(f"fold {fold_index} metric {path} must be numeric.")
            number = float(current)
            if not math.isfinite(number):
                raise ValueError(f"fold {fold_index} metric {path} must be finite.")
            values[path].append(number)
            per_fold_metrics[path] = number
        fold_records.append(
            {
                "fold_index": fold_index,
                "record_count": summary.get("record_count"),
                "participant_ids": participants,
                "participants": summary.get("participants", []),
                "metrics": per_fold_metrics,
            }
        )

    return {
        "fold_count": len(fold_summaries),
        "fold_indices": fold_indices,
        "test_participant_count": len(test_participants),
        "test_participant_ids": test_participants,
        "aggregation": "unweighted_fold_mean_population_std",
        "metrics": {
            path: {
                "mean": statistics.fmean(numbers),
                "std": statistics.pstdev(numbers),
                "fold_values": numbers,
            }
            for path, numbers in values.items()
        },
        "folds": fold_records,
    }


def build_split_summary(
    partitioned: PartitionedManifest,
    *,
    fold_index: int,
    split_seed: int,
    label_protocol: LabelProtocol,
) -> dict[str, object]:
    """Return label and declared-data counts for each participant partition.

    Args:
        partitioned: Participant-disjoint manifest partitions.
        fold_index: Non-negative split index recorded in the artifact.
        split_seed: Integer split seed recorded in the artifact.
        label_protocol: Named binary label mapping used for all counts.

    Returns:
        JSON-compatible partition, participant, declared-modality, binary-task,
        and four-quadrant counts. Counts use raw manifest declarations, not
        runtime signal-validity or speech-activity masks.
    """

    if not isinstance(partitioned, PartitionedManifest):
        raise TypeError("partitioned must be PartitionedManifest.")
    for name, value in (("fold_index", fold_index), ("split_seed", split_seed)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
    if not isinstance(label_protocol, LabelProtocol):
        raise TypeError("label_protocol must be LabelProtocol.")
    split = partitioned.split
    return {
        "fold_index": fold_index,
        "split_seed": split_seed,
        "label_protocol": label_protocol.value,
        "partitions": {
            "train": _record_partition_summary(
                partitioned.train_records,
                split.train_participant_ids,
                label_protocol,
            ),
            "validation": _record_partition_summary(
                partitioned.validation_records,
                split.validation_participant_ids,
                label_protocol,
            ),
            "test": _record_partition_summary(
                partitioned.test_records,
                split.test_participant_ids,
                label_protocol,
            ),
        },
    }


def classification_metrics_summary(
    metrics: ClassificationMetrics,
) -> dict[str, object]:
    """Convert one CPU ``[K,K]`` confusion-matrix result to JSON values.

    Binary classes use the K-EmoCon ``low``, ``high`` order. Quadrant classes
    use ``LALV``, ``HALV``, ``LAHV``, ``HAHV``; other class counts receive
    stable zero-based names. Every per-class entry contains Python numeric
    precision, recall, F1, and support values suitable for JSON encoding.
    """

    if not isinstance(metrics, ClassificationMetrics):
        raise TypeError("metrics must be ClassificationMetrics.")
    class_names = {
        2: ("low", "high"),
        4: ("LALV", "HALV", "LAHV", "HAHV"),
    }.get(
        metrics.num_classes,
        tuple(f"class_{index}" for index in range(metrics.num_classes)),
    )
    return {
        "evaluated_count": metrics.evaluated_count,
        "correct_count": metrics.correct_count,
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
        "class_support": metrics.class_support.tolist(),
        "per_class": {
            name: {
                "precision": float(metrics.class_precision[index].item()),
                "recall": float(metrics.class_recall[index].item()),
                "f1": float(metrics.class_f1[index].item()),
                "support": int(metrics.class_support[index].item()),
            }
            for index, name in enumerate(class_names)
        },
    }


def emotion_metrics_summary(metrics: EmotionTaskMetrics) -> dict[str, object]:
    """Convert arousal ``[2]``, valence ``[2]``, and quadrant ``[4]`` metrics."""

    if not isinstance(metrics, EmotionTaskMetrics):
        raise TypeError("metrics must be EmotionTaskMetrics.")
    return {
        "arousal": classification_metrics_summary(metrics.arousal),
        "valence": classification_metrics_summary(metrics.valence),
        "quadrant": classification_metrics_summary(metrics.quadrant),
    }


def speech_activity_stratified_summary(
    predictions: EvaluationPredictions,
) -> dict[str, object]:
    """Summarize five reporting-only activity strata from CPU rows ``[N]``."""

    return {
        stratum.activity_bin.value: {
            "record_count": stratum.record_count,
            "participant_count": stratum.participant_count,
            "arousal": classification_metrics_summary(stratum.arousal),
            "valence": classification_metrics_summary(stratum.valence),
            "fusion": {
                "availability_patterns": dict(stratum.availability_patterns),
                "all_valid": asdict(stratum.all_valid_fusion),
                "both_available": asdict(stratum.both_available_fusion),
            },
        }
        for stratum in compute_speech_activity_strata(predictions)
    }


def build_evaluation_summary(
    result: ParticipantIndependentEvaluationOutput,
    *,
    fold_index: int,
    checkpoint: Path,
) -> dict[str, object]:
    """Convert a detached test evaluation into a complete JSON report."""

    if not isinstance(result, ParticipantIndependentEvaluationOutput):
        raise TypeError(
            "result must be ParticipantIndependentEvaluationOutput."
        )
    if not isinstance(fold_index, int) or isinstance(fold_index, bool):
        raise TypeError("fold_index must be an integer.")
    if not isinstance(checkpoint, Path) or checkpoint.is_absolute():
        raise ValueError("checkpoint must be a project-relative Path.")
    return {
        "fold_index": fold_index,
        "checkpoint": checkpoint.as_posix(),
        "partition": result.partition.value,
        "batch_count": result.batch_count,
        "record_count": result.record_count,
        "participant_count": result.participant_count,
        "participant_ids": list(result.participant_ids),
        "modality_pattern_counts": {
            pattern.value: count
            for pattern, count in result.modality_pattern_counts.items()
        },
        "loss": asdict(result.loss_epoch.loss_averages),
        "overall": emotion_metrics_summary(result.overall_metrics),
        "speech_activity_stratified": speech_activity_stratified_summary(
            result.predictions
        ),
        "participant_macro": asdict(result.participant_macro_metrics),
        "binary_decision_thresholds": (
            result.binary_decision_thresholds.to_metadata()
        ),
        "fusion_diagnostics": dict(result.fusion_diagnostics),
        "participants": [
            {
                "participant_id": item.participant_id,
                "record_count": item.record_count,
                "modality_pattern_counts": {
                    pattern.value: count
                    for pattern, count in item.modality_pattern_counts.items()
                },
                "metrics": emotion_metrics_summary(item.task_metrics),
            }
            for item in result.participant_metrics
        ],
        "modality_strata": [
            {
                "pattern": item.pattern.value,
                "record_count": item.record_count,
                "metrics": emotion_metrics_summary(item.task_metrics),
            }
            for item in result.modality_stratum_metrics
        ],
    }


def build_ablation_evaluation_summary(
    results: Mapping[str, ParticipantIndependentEvaluationOutput],
    *,
    fold_index: int,
    checkpoint: Path,
) -> dict[str, object]:
    """Build full/speech/physiology reports over identical both-valid rows."""

    expected_modes = ("full", "speech_only", "physiology_only")
    if tuple(results) != expected_modes:
        raise ValueError(
            "results must follow full, speech_only, physiology_only order."
        )
    sample_ids = tuple(
        results[mode].predictions.sample_ids for mode in expected_modes
    )
    if not sample_ids[0] or any(
        values != sample_ids[0] for values in sample_ids[1:]
    ):
        raise ValueError(
            "all ablation modes must evaluate identical ordered sample IDs."
        )
    return {
        "fold_index": fold_index,
        "checkpoint": checkpoint.as_posix(),
        "sample_scope": "originally_both_modalities_valid",
        "record_count": len(sample_ids[0]),
        "modes": {
            mode: build_evaluation_summary(
                results[mode],
                fold_index=fold_index,
                checkpoint=checkpoint,
            )
            for mode in expected_modes
        },
    }


__all__ = [
    "FullWindowSpeechAvailabilityStatistics",
    "append_jsonl_artifact",
    "build_ablation_evaluation_summary",
    "build_environment_summary",
    "build_evaluation_summary",
    "build_cross_fold_evaluation_summary",
    "build_runtime_availability_audit",
    "build_split_summary",
    "classification_metrics_summary",
    "emotion_metrics_summary",
    "speech_activity_stratified_summary",
    "write_effective_config_snapshot",
    "write_json_artifact",
]
