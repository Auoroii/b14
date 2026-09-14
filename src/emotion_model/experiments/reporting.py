"""Structured, portable experiment artifacts for K-EmoCon runs."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
import yaml  # type: ignore[import-untyped]

from emotion_model.common import LabelProtocol, binarize_emotion_scores
from emotion_model.data import (
    AlignedMultimodalBatch,
    MultimodalWindowRecord,
    PartitionedManifest,
)
from emotion_model.evaluation import (
    ClassificationMetrics,
    EmotionTaskMetrics,
    ParticipantIndependentEvaluationOutput,
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
        observed = ratios[source_present]
        self.speech_activity_ratio_observed += observed.numel()
        self.speech_activity_ratio_eq_0 += int((observed == 0.0).sum().item())
        self.speech_activity_ratio_0_to_0_1 += int(
            ((observed > 0.0) & (observed < 0.1)).sum().item()
        )
        self.speech_activity_ratio_0_1_to_0_25 += int(
            ((observed >= 0.1) & (observed < 0.25)).sum().item()
        )
        self.speech_activity_ratio_ge_0_25 += int(
            (observed >= 0.25).sum().item()
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
    """Convert one CPU ``[K,K]`` confusion-matrix result to JSON values."""

    if not isinstance(metrics, ClassificationMetrics):
        raise TypeError("metrics must be ClassificationMetrics.")
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
    "build_split_summary",
    "classification_metrics_summary",
    "emotion_metrics_summary",
    "write_effective_config_snapshot",
    "write_json_artifact",
]
