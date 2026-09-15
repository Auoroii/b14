"""Participant-independent, single-forward multimodal evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import torch
from torch import Tensor

from emotion_model.data import (
    AlignedMultimodalBatch,
    DatasetPartition,
    ParticipantSplit,
)
from emotion_model.evaluation.activity import (
    SpeechActivityBin,
    speech_activity_bin_masks,
)
from emotion_model.evaluation.calibration import (
    BinaryDecisionThresholds,
    calibrate_binary_decision_thresholds,
)
from emotion_model.evaluation.metrics import (
    ClassificationMetrics,
    EmotionTaskMetrics,
    compute_emotion_task_metrics,
)
from emotion_model.multimodal import (
    MultimodalEmotionClassifier,
    MultimodalEmotionClassifierOutput,
)
from emotion_model.training.epochs import (
    EpochPhase,
    MultimodalEpochOutput,
    _EpochAccumulator,
)
from emotion_model.training.objectives import (
    EmotionTaskClassWeights,
    MultimodalTrainingObjective,
)
from emotion_model.training.steps import (
    MultimodalValidationStepOutput,
    validate_multimodal_batch,
)


class ModalityPattern(StrEnum):
    """Stable full-batch modality availability patterns."""

    BOTH = "both"
    SPEECH_ONLY = "speech_only"
    PHYSIOLOGY_ONLY = "physiology_only"
    NEITHER = "neither"


def _validate_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_probability_float(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a Python float.")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")


def _partition_ids(
    split: ParticipantSplit,
    partition: DatasetPartition,
) -> tuple[str, ...]:
    if partition is DatasetPartition.TRAIN:
        return split.train_participant_ids
    if partition is DatasetPartition.VALIDATION:
        return split.validation_participant_ids
    return split.test_participant_ids


@dataclass(frozen=True)
class ParticipantEvaluationScope:
    """Declare one exact participant partition eligible for evaluation.

    ``expected_participant_ids`` preserves the tuple order configured in
    :class:`ParticipantSplit`. No labels, sessions, record counts, or model
    outputs influence membership.
    """

    split: ParticipantSplit
    partition: DatasetPartition
    require_all_partition_participants: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.split, ParticipantSplit):
            raise TypeError("split must be ParticipantSplit.")
        if not isinstance(self.partition, DatasetPartition):
            raise TypeError("partition must be DatasetPartition.")
        if not isinstance(self.require_all_partition_participants, bool):
            raise TypeError(
                "require_all_partition_participants must be bool."
            )
        if not self.expected_participant_ids:
            raise ValueError(
                "the selected evaluation partition must contain participants."
            )

    @property
    def expected_participant_ids(self) -> tuple[str, ...]:
        """Return exact eligible IDs in the split's configured order."""
        return _partition_ids(self.split, self.partition)


def _validate_ids(
    values: tuple[str, ...],
    *,
    name: str,
    expected_length: int | None = None,
    unique: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple.")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}; received {len(values)}."
        )
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} entries must be strings.")
        if not value.strip():
            raise ValueError(f"{name} entries must be non-empty.")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be globally unique.")


def _validate_prediction_tensor(
    value: Tensor,
    *,
    name: str,
    length: int,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}.")
    if tuple(value.shape) != (length,):
        raise ValueError(
            f"{name} must have shape [{length}]; received {tuple(value.shape)}."
        )
    return value


@dataclass(frozen=True)
class EvaluationPredictions:
    """Compact detached CPU predictions for ``N`` evaluated records.

    Availability, source-presence, activity-observation, and ``sample_valid``
    tensors are boolean CPU tensors ``[N]``. Activity ratios are a float32 CPU
    diagnostic tensor ``[N]`` and have no model role. Optional retained shared
    modality weights are a detached floating CPU tensor ``[N,2]`` in
    ``[speech, physiology]`` order. Targets and predictions are long CPU
    tensors ``[N]``. Original protocol targets are retained; predictions are
    class indices only on model-valid rows and equal ``ignore_index`` on
    invalid rows. Construction defensively clones every tensor, so no returned
    field shares caller storage.
    """

    sample_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    speech_available: Tensor
    physiology_available: Tensor
    sample_valid: Tensor
    arousal_targets: Tensor
    arousal_predictions: Tensor
    valence_targets: Tensor
    valence_predictions: Tensor
    quadrant_targets: Tensor
    quadrant_predictions: Tensor
    ignore_index: int
    speech_source_present: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    speech_activity_observed: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    speech_activity_ratios: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    modality_weights: Tensor | None = None

    def __post_init__(self) -> None:
        _validate_ids(self.sample_ids, name="sample_ids", unique=True)
        length = len(self.sample_ids)
        _validate_ids(
            self.participant_ids,
            name="participant_ids",
            expected_length=length,
        )
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index,
            int,
        ):
            raise TypeError("ignore_index must be an integer, not bool.")
        if 0 <= self.ignore_index < 4:
            raise ValueError("ignore_index must lie outside [0, 4).")

        bool_names = (
            "speech_available",
            "physiology_available",
            "sample_valid",
        )
        long_names = (
            "arousal_targets",
            "arousal_predictions",
            "valence_targets",
            "valence_predictions",
            "quadrant_targets",
            "quadrant_predictions",
        )
        for name in bool_names:
            value = _validate_prediction_tensor(
                getattr(self, name),
                name=name,
                length=length,
                dtype=torch.bool,
            )
            object.__setattr__(self, name, value.detach().clone())
        for name in long_names:
            value = _validate_prediction_tensor(
                getattr(self, name),
                name=name,
                length=length,
                dtype=torch.long,
            )
            object.__setattr__(self, name, value.detach().clone())

        diagnostic_specs = (
            ("speech_source_present", torch.bool),
            ("speech_activity_observed", torch.bool),
            ("speech_activity_ratios", torch.float32),
        )
        for name, dtype in diagnostic_specs:
            value = getattr(self, name)
            if (
                isinstance(value, Tensor)
                and tuple(value.shape) == (0,)
                and length > 0
            ):
                value = torch.zeros(length, dtype=dtype)
            value = _validate_prediction_tensor(
                value,
                name=name,
                length=length,
                dtype=dtype,
            )
            if name == "speech_activity_ratios" and not bool(
                torch.isfinite(value).all()
                and (value >= 0.0).all()
                and (value <= 1.0).all()
            ):
                raise ValueError(
                    "speech_activity_ratios must be finite in [0, 1]."
                )
            object.__setattr__(self, name, value.detach().clone())
        if bool(
            (
                self.speech_activity_observed
                & ~self.speech_source_present
            ).any()
        ):
            raise ValueError(
                "speech activity can be observed only for a real source."
            )

        weights = self.modality_weights
        if weights is not None:
            if not isinstance(weights, Tensor) or not weights.is_floating_point():
                raise TypeError("modality_weights must be a floating Tensor or None.")
            if weights.device.type != "cpu":
                raise ValueError("modality_weights must be on CPU.")
            if tuple(weights.shape) != (length, 2):
                raise ValueError(
                    "modality_weights must have shape "
                    f"[{length}, 2]; received {tuple(weights.shape)}."
                )
            if not bool(
                torch.isfinite(weights).all()
                and (weights >= 0.0).all()
                and (weights <= 1.0).all()
            ):
                raise ValueError("modality_weights must be finite in [0, 1].")
            both = self.speech_available & self.physiology_available
            speech_only = self.speech_available & ~self.physiology_available
            physiology_only = ~self.speech_available & self.physiology_available
            neither = ~self.sample_valid
            if not torch.allclose(
                weights[both].sum(dim=1),
                torch.ones_like(weights[both, 0]),
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "both-available modality weights must sum to one."
                )
            if not torch.equal(
                weights[speech_only],
                torch.tensor(
                    [1.0, 0.0],
                    dtype=weights.dtype,
                ).expand(int(speech_only.sum().item()), -1),
            ):
                raise ValueError("speech-only modality weights must equal [1, 0].")
            if not torch.equal(
                weights[physiology_only],
                torch.tensor(
                    [0.0, 1.0],
                    dtype=weights.dtype,
                ).expand(int(physiology_only.sum().item()), -1),
            ):
                raise ValueError(
                    "physiology-only modality weights must equal [0, 1]."
                )
            if not torch.equal(weights[neither], torch.zeros_like(weights[neither])):
                raise ValueError("invalid-row modality weights must be exact zero.")
            object.__setattr__(self, "modality_weights", weights.detach().clone())

        expected_valid = self.speech_available | self.physiology_available
        if not torch.equal(self.sample_valid, expected_valid):
            raise ValueError(
                "sample_valid must equal speech_available OR "
                "physiology_available."
            )
        invalid = ~self.sample_valid
        for name in (
            "arousal_predictions",
            "valence_predictions",
            "quadrant_predictions",
        ):
            if not bool((getattr(self, name)[invalid] == self.ignore_index).all()):
                raise ValueError(
                    f"{name} must equal ignore_index on invalid model rows."
                )
        for name, class_count in (
            ("arousal_targets", 2),
            ("valence_targets", 2),
            ("quadrant_targets", 4),
        ):
            values = getattr(self, name)
            legal = (
                (values == self.ignore_index)
                | ((values >= 0) & (values < class_count))
            )
            if not bool(legal.all()):
                raise ValueError(
                    f"{name} must contain classes [0, {class_count}) or "
                    "ignore_index."
                )
        for name, class_count in (
            ("arousal_predictions", 2),
            ("valence_predictions", 2),
            ("quadrant_predictions", 4),
        ):
            values = getattr(self, name)[self.sample_valid]
            if bool(((values < 0) | (values >= class_count)).any()):
                raise ValueError(
                    f"{name} valid rows must lie in [0, {class_count})."
                )


def _readonly_pattern_counts(
    counts: Mapping[ModalityPattern, int],
    *,
    expected_total: int,
) -> Mapping[ModalityPattern, int]:
    if not isinstance(counts, Mapping):
        raise TypeError("modality_pattern_counts must be a mapping.")
    if not all(isinstance(key, ModalityPattern) for key in counts):
        raise TypeError(
            "modality_pattern_counts keys must be ModalityPattern values."
        )
    if set(counts) != set(ModalityPattern):
        raise ValueError(
            "modality_pattern_counts must contain exactly all four patterns."
        )
    copied: dict[ModalityPattern, int] = {}
    for pattern in ModalityPattern:
        value = counts[pattern]
        _validate_nonnegative_int(
            value,
            name=f"modality_pattern_counts[{pattern.value}]",
        )
        copied[pattern] = value
    if sum(copied.values()) != expected_total:
        raise ValueError(
            "modality pattern counts must sum to the record count."
        )
    return MappingProxyType(copied)


@dataclass(frozen=True)
class ParticipantEvaluationMetrics:
    """Window metrics for one participant.

    ``record_count`` includes ignored and unavailable windows. The four-entry
    modality mapping is a defensive read-only copy.
    """

    participant_id: str
    record_count: int
    modality_pattern_counts: Mapping[ModalityPattern, int]
    task_metrics: EmotionTaskMetrics

    def __post_init__(self) -> None:
        _validate_ids((self.participant_id,), name="participant_id")
        _validate_nonnegative_int(self.record_count, name="record_count")
        if self.record_count <= 0:
            raise ValueError("record_count must be > 0.")
        if not isinstance(self.task_metrics, EmotionTaskMetrics):
            raise TypeError("task_metrics must be EmotionTaskMetrics.")
        for task_name in ("arousal", "valence", "quadrant"):
            if (
                getattr(self.task_metrics, task_name).evaluated_count
                > self.record_count
            ):
                raise ValueError(
                    f"{task_name} evaluated_count cannot exceed record_count."
                )
        object.__setattr__(
            self,
            "modality_pattern_counts",
            _readonly_pattern_counts(
                self.modality_pattern_counts,
                expected_total=self.record_count,
            ),
        )


@dataclass(frozen=True)
class ParticipantMacroTaskMetrics:
    """Equal-participant macro summary for one classification task.

    Only participants with at least one evaluated window are included. All
    five metric means weight those participants equally, never by window
    count. With no eligible participant, counts and means are zero.
    """

    participant_count: int
    evaluated_window_count: int
    mean_accuracy: float
    mean_macro_precision: float
    mean_macro_recall: float
    mean_macro_f1: float
    mean_balanced_accuracy: float

    def __post_init__(self) -> None:
        _validate_nonnegative_int(
            self.participant_count,
            name="participant_count",
        )
        _validate_nonnegative_int(
            self.evaluated_window_count,
            name="evaluated_window_count",
        )
        if (
            self.participant_count == 0
            and self.evaluated_window_count != 0
        ) or self.evaluated_window_count < self.participant_count:
            raise ValueError(
                "evaluated_window_count must be zero with no participant and "
                "otherwise at least participant_count."
            )
        for name in (
            "mean_accuracy",
            "mean_macro_precision",
            "mean_macro_recall",
            "mean_macro_f1",
            "mean_balanced_accuracy",
        ):
            _validate_probability_float(getattr(self, name), name=name)
        if self.participant_count == 0 and any(
            getattr(self, name) != 0.0
            for name in (
                "mean_accuracy",
                "mean_macro_precision",
                "mean_macro_recall",
                "mean_macro_f1",
                "mean_balanced_accuracy",
            )
        ):
            raise ValueError(
                "participant macro means must be zero when no participant "
                "is included."
            )


@dataclass(frozen=True)
class ParticipantMacroEmotionMetrics:
    """Equal-participant macro summaries for all three emotion tasks."""

    arousal: ParticipantMacroTaskMetrics
    valence: ParticipantMacroTaskMetrics
    quadrant: ParticipantMacroTaskMetrics

    def __post_init__(self) -> None:
        for name in ("arousal", "valence", "quadrant"):
            if not isinstance(
                getattr(self, name),
                ParticipantMacroTaskMetrics,
            ):
                raise TypeError(
                    f"{name} must be ParticipantMacroTaskMetrics."
                )


@dataclass(frozen=True)
class ModalityStratumMetrics:
    """Metrics for one explicit availability pattern over ``record_count`` rows."""

    pattern: ModalityPattern
    record_count: int
    task_metrics: EmotionTaskMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, ModalityPattern):
            raise TypeError("pattern must be ModalityPattern.")
        _validate_nonnegative_int(self.record_count, name="record_count")
        if not isinstance(self.task_metrics, EmotionTaskMetrics):
            raise TypeError("task_metrics must be EmotionTaskMetrics.")
        for task_name in ("arousal", "valence", "quadrant"):
            if (
                getattr(self.task_metrics, task_name).evaluated_count
                > self.record_count
            ):
                raise ValueError(
                    f"{task_name} evaluated_count cannot exceed record_count."
                )


@dataclass(frozen=True)
class ParticipantIndependentEvaluationOutput:
    """Detached result of one participant-independent evaluation pass.

    Counts are Python integers; metric scalars are Python floats; retained
    prediction and confusion tensors are detached CPU ``[N]``/``[K,K]``
    tensors. No logits, probabilities, embeddings, batches, or graphs are
    retained.
    """

    partition: DatasetPartition
    batch_count: int
    record_count: int
    participant_count: int
    participant_ids: tuple[str, ...]
    modality_pattern_counts: Mapping[ModalityPattern, int]
    loss_epoch: MultimodalEpochOutput
    overall_metrics: EmotionTaskMetrics
    participant_metrics: tuple[ParticipantEvaluationMetrics, ...]
    participant_macro_metrics: ParticipantMacroEmotionMetrics
    modality_stratum_metrics: tuple[ModalityStratumMetrics, ...]
    predictions: EvaluationPredictions
    binary_decision_thresholds: BinaryDecisionThresholds
    fusion_diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.partition, DatasetPartition):
            raise TypeError("partition must be DatasetPartition.")
        for name in ("batch_count", "record_count", "participant_count"):
            _validate_nonnegative_int(getattr(self, name), name=name)
        if self.batch_count <= 0 or self.record_count <= 0:
            raise ValueError("batch_count and record_count must be > 0.")
        _validate_ids(self.participant_ids, name="participant_ids", unique=True)
        if len(self.participant_ids) != self.participant_count:
            raise ValueError(
                "participant_count must equal len(participant_ids)."
            )
        if not isinstance(self.loss_epoch, MultimodalEpochOutput):
            raise TypeError("loss_epoch must be MultimodalEpochOutput.")
        if self.loss_epoch.phase is not EpochPhase.VALIDATION:
            raise ValueError("loss_epoch must have validation phase.")
        if self.loss_epoch.batch_count != self.batch_count:
            raise ValueError("loss_epoch batch count must match batch_count.")
        if not isinstance(self.overall_metrics, EmotionTaskMetrics):
            raise TypeError("overall_metrics must be EmotionTaskMetrics.")
        if not isinstance(
            self.participant_macro_metrics,
            ParticipantMacroEmotionMetrics,
        ):
            raise TypeError(
                "participant_macro_metrics must be "
                "ParticipantMacroEmotionMetrics."
            )
        if not isinstance(self.predictions, EvaluationPredictions):
            raise TypeError("predictions must be EvaluationPredictions.")
        if not isinstance(
            self.binary_decision_thresholds,
            BinaryDecisionThresholds,
        ):
            raise TypeError(
                "binary_decision_thresholds must be "
                "BinaryDecisionThresholds."
            )
        if not isinstance(self.fusion_diagnostics, Mapping):
            raise TypeError("fusion_diagnostics must be a mapping.")
        for name, value in self.fusion_diagnostics.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "fusion_diagnostics must map non-empty names to finite "
                    "real values."
                )
        if len(self.predictions.sample_ids) != self.record_count:
            raise ValueError("predictions length must equal record_count.")
        if not isinstance(self.participant_metrics, tuple) or not all(
            isinstance(item, ParticipantEvaluationMetrics)
            for item in self.participant_metrics
        ):
            raise TypeError(
                "participant_metrics must be a tuple of "
                "ParticipantEvaluationMetrics."
            )
        if tuple(item.participant_id for item in self.participant_metrics) != (
            self.participant_ids
        ):
            raise ValueError(
                "participant_metrics must match participant_ids order."
            )
        if len(self.participant_metrics) != self.participant_count or sum(
            item.record_count for item in self.participant_metrics
        ) != self.record_count:
            raise ValueError(
                "participant metric counts must cover every retained record."
            )
        if set(self.predictions.participant_ids) != set(self.participant_ids):
            raise ValueError(
                "participant_ids must exactly cover prediction participants."
            )
        if not isinstance(self.modality_stratum_metrics, tuple) or not all(
            isinstance(item, ModalityStratumMetrics)
            for item in self.modality_stratum_metrics
        ):
            raise TypeError(
                "modality_stratum_metrics must be a tuple of "
                "ModalityStratumMetrics."
            )
        if tuple(item.pattern for item in self.modality_stratum_metrics) != tuple(
            ModalityPattern
        ):
            raise ValueError(
                "modality_stratum_metrics must follow enum order."
            )
        readonly_counts = _readonly_pattern_counts(
            self.modality_pattern_counts,
            expected_total=self.record_count,
        )
        if any(
            item.record_count != readonly_counts[item.pattern]
            for item in self.modality_stratum_metrics
        ):
            raise ValueError(
                "stratum record counts must match modality_pattern_counts."
            )
        predicted_pattern_counts = {
            pattern: int(mask.sum().item())
            for pattern, mask in _pattern_masks(self.predictions).items()
        }
        if any(
            readonly_counts[pattern] != predicted_pattern_counts[pattern]
            for pattern in ModalityPattern
        ):
            raise ValueError(
                "modality_pattern_counts must match retained availability."
            )
        object.__setattr__(self, "modality_pattern_counts", readonly_counts)


def _pattern_masks(predictions: EvaluationPredictions) -> dict[ModalityPattern, Tensor]:
    speech = predictions.speech_available
    physiology = predictions.physiology_available
    return {
        ModalityPattern.BOTH: speech & physiology,
        ModalityPattern.SPEECH_ONLY: speech & ~physiology,
        ModalityPattern.PHYSIOLOGY_ONLY: ~speech & physiology,
        ModalityPattern.NEITHER: ~speech & ~physiology,
    }


def _working_targets(
    predictions: EvaluationPredictions,
) -> tuple[Tensor, Tensor, Tensor]:
    targets = (
        predictions.arousal_targets.clone(),
        predictions.valence_targets.clone(),
        predictions.quadrant_targets.clone(),
    )
    invalid = ~predictions.sample_valid
    for value in targets:
        value[invalid] = predictions.ignore_index
    return targets


def _metrics_for_rows(
    predictions: EvaluationPredictions,
    working_targets: tuple[Tensor, Tensor, Tensor],
    rows: Tensor,
) -> EmotionTaskMetrics:
    return compute_emotion_task_metrics(
        arousal_targets=working_targets[0][rows],
        arousal_predictions=predictions.arousal_predictions[rows],
        valence_targets=working_targets[1][rows],
        valence_predictions=predictions.valence_predictions[rows],
        quadrant_targets=working_targets[2][rows],
        quadrant_predictions=predictions.quadrant_predictions[rows],
        ignore_index=predictions.ignore_index,
    )


def _participant_macro_task(
    participant_metrics: tuple[ParticipantEvaluationMetrics, ...],
    *,
    task_name: str,
) -> ParticipantMacroTaskMetrics:
    included: list[ClassificationMetrics] = []
    for participant in participant_metrics:
        task = getattr(participant.task_metrics, task_name)
        if task.evaluated_count > 0:
            included.append(task)
    if not included:
        return ParticipantMacroTaskMetrics(
            participant_count=0,
            evaluated_window_count=0,
            mean_accuracy=0.0,
            mean_macro_precision=0.0,
            mean_macro_recall=0.0,
            mean_macro_f1=0.0,
            mean_balanced_accuracy=0.0,
        )
    count = len(included)
    return ParticipantMacroTaskMetrics(
        participant_count=count,
        evaluated_window_count=sum(item.evaluated_count for item in included),
        mean_accuracy=sum(item.accuracy for item in included) / count,
        mean_macro_precision=(
            sum(item.macro_precision for item in included) / count
        ),
        mean_macro_recall=sum(item.macro_recall for item in included) / count,
        mean_macro_f1=sum(item.macro_f1 for item in included) / count,
        mean_balanced_accuracy=(
            sum(item.balanced_accuracy for item in included) / count
        ),
    )


def _validate_probability_output(
    output: MultimodalEmotionClassifierOutput,
    batch: AlignedMultimodalBatch,
) -> None:
    if not isinstance(output, MultimodalEmotionClassifierOutput):
        raise RuntimeError(
            "validation model must return MultimodalEmotionClassifierOutput."
        )
    scheduled = output.fusion_output.scheduled_outputs
    if scheduled.batch is not batch:
        raise RuntimeError(
            "model_output scheduled batch must be the exact evaluated batch."
        )
    batch_size = len(batch.records)
    valid = output.sample_valid
    if valid.dtype != torch.bool or tuple(valid.shape) != (batch_size,):
        raise RuntimeError(
            f"model sample_valid must be bool [{batch_size}]."
        )
    expected_valid = batch.speech_available | batch.physiology_available
    if valid.device != expected_valid.device or not torch.equal(
        valid,
        expected_valid,
    ):
        raise RuntimeError(
            "model sample_valid must equal batch modality availability."
        )
    availability = scheduled.availability
    if not torch.equal(
        availability.speech_available,
        batch.speech_available,
    ) or not torch.equal(
        availability.physiology_available,
        batch.physiology_available,
    ):
        raise RuntimeError(
            "scheduled availability must exactly match the evaluated batch."
        )
    local_valid = valid.to(device=output.arousal_probabilities.device)
    for name, probabilities, class_count in (
        ("arousal_probabilities", output.arousal_probabilities, 2),
        ("valence_probabilities", output.valence_probabilities, 2),
        ("quadrant_probabilities", output.quadrant_probabilities, 4),
    ):
        if (
            not isinstance(probabilities, Tensor)
            or not probabilities.is_floating_point()
            or tuple(probabilities.shape) != (batch_size, class_count)
        ):
            raise RuntimeError(
                f"{name} must be floating [{batch_size}, {class_count}]."
            )
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError(f"{name} must be finite.")
        valid_rows = probabilities[local_valid]
        if valid_rows.numel() and (
            not bool((valid_rows >= 0).all() and (valid_rows <= 1).all())
            or not torch.allclose(
                valid_rows.sum(dim=1),
                torch.ones_like(valid_rows[:, 0]),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        ):
            raise RuntimeError(f"{name} valid rows must sum to one.")
        invalid_rows = probabilities[~local_valid]
        if not torch.equal(invalid_rows, torch.zeros_like(invalid_rows)):
            raise RuntimeError(f"{name} invalid rows must be exact zero.")


def _predictions_from_thresholds(
    arousal_high_probabilities: Tensor,
    valence_high_probabilities: Tensor,
    sample_valid: Tensor,
    *,
    thresholds: BinaryDecisionThresholds,
    ignore_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(thresholds, BinaryDecisionThresholds):
        raise TypeError("thresholds must be BinaryDecisionThresholds.")
    arousal = (
        arousal_high_probabilities > thresholds.arousal_high
    ).to(dtype=torch.long)
    valence = (
        valence_high_probabilities > thresholds.valence_high
    ).to(dtype=torch.long)
    quadrant = arousal + 2 * valence
    invalid = ~sample_valid
    for prediction in (arousal, valence, quadrant):
        prediction[invalid] = ignore_index
    return arousal, valence, quadrant


def _validate_runner_inputs(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batches: Iterable[AlignedMultimodalBatch],
    scope: ParticipantEvaluationScope,
) -> None:
    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    if not isinstance(objective, MultimodalTrainingObjective):
        raise TypeError("objective must be MultimodalTrainingObjective.")
    if not isinstance(scope, ParticipantEvaluationScope):
        raise TypeError("scope must be ParticipantEvaluationScope.")
    if isinstance(
        batches,
        (str, bytes, Tensor, AlignedMultimodalBatch),
    ) or not isinstance(batches, Iterable):
        raise TypeError(
            "batches must be a non-string iterable of AlignedMultimodalBatch."
        )


def evaluate_participant_independent(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batches: Iterable[AlignedMultimodalBatch],
    *,
    scope: ParticipantEvaluationScope,
    class_weights: EmotionTaskClassWeights | None = None,
    binary_thresholds: BinaryDecisionThresholds | None = None,
    calibrate_thresholds: bool = False,
) -> ParticipantIndependentEvaluationOutput:
    """Evaluate one participant partition with one forward per batch.

    Args:
        model: Final multimodal classifier producing probabilities
            ``[B,2]``, ``[B,2]``, and ``[B,4]`` plus validity ``[B]``.
        objective: Existing stage-13A objective used once per batch.
        batches: Non-empty, one-pass iterable of logical batches. It is never
            sized, reordered, cached, or replayed.
        scope: Exact participant split and partition membership contract.
        class_weights: Optional pre-existing task weights; evaluation never
            fits or changes them.
        binary_thresholds: Optional preselected arousal/valence thresholds.
        calibrate_thresholds: Whether to select thresholds from this validation
            partition using equal-participant macro-F1.

    Returns:
        :class:`ParticipantIndependentEvaluationOutput` containing validation
        loss aggregation, window metrics, participant metrics, equal-person
        macro metrics, four availability strata, and detached CPU prediction
        vectors ``[N]``.

    Raises:
        TypeError: If public inputs or an iterated batch have wrong types.
        ValueError: If the iterable is empty, participant coverage is wrong,
            sample IDs repeat, or ignore indices differ.
        RuntimeError: If a validation/model output violates its batch,
            probability, availability, or validity contract.

    The model is switched to eval exactly once and remains there. The existing
    no-gradient validation step and stage-13B private loss accumulator are
    reused; no training, backward, optimizer, checkpoint, or unimodal
    prediction fallback occurs. Threshold fitting is allowed only when
    explicitly requested for a validation scope.
    """
    _validate_runner_inputs(model, objective, batches, scope)
    if not isinstance(calibrate_thresholds, bool):
        raise TypeError("calibrate_thresholds must be bool.")
    if binary_thresholds is not None and not isinstance(
        binary_thresholds,
        BinaryDecisionThresholds,
    ):
        raise TypeError("binary_thresholds must be BinaryDecisionThresholds.")
    if calibrate_thresholds and binary_thresholds is not None:
        raise ValueError(
            "calibrate_thresholds and binary_thresholds are mutually exclusive."
        )
    if (
        calibrate_thresholds
        and scope.partition is not DatasetPartition.VALIDATION
    ):
        raise ValueError(
            "threshold calibration is allowed only on validation partitions."
        )
    expected_ids = scope.expected_participant_ids
    expected_set = set(expected_ids)
    observed_participants: set[str] = set()
    observed_sample_ids: set[str] = set()
    sample_ids: list[str] = []
    participant_ids: list[str] = []
    speech_available: list[Tensor] = []
    physiology_available: list[Tensor] = []
    speech_source_present: list[Tensor] = []
    speech_activity_observed: list[Tensor] = []
    aligned_speech_activity_ratios: list[Tensor] = []
    observed_speech_activity_ratios: list[Tensor] = []
    modality_weights_rows: list[Tensor] = []
    speech_source_unavailable_count = 0
    sample_valid: list[Tensor] = []
    arousal_targets: list[Tensor] = []
    valence_targets: list[Tensor] = []
    quadrant_targets: list[Tensor] = []
    arousal_high_probabilities: list[Tensor] = []
    valence_high_probabilities: list[Tensor] = []
    shared_gate_sums = {"speech": 0.0, "physiology": 0.0}
    shared_gate_both_count = 0
    activity_gate_sums: dict[str, dict[str, float]] = {
        "low": {"count": 0.0, "speech": 0.0},
        "medium": {"count": 0.0, "speech": 0.0},
        "high": {"count": 0.0, "speech": 0.0},
    }
    ignore_index: int | None = None

    model.eval()
    loss_accumulator = _EpochAccumulator()
    for batch in batches:
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError(
                "every evaluation item must be AlignedMultimodalBatch; "
                f"received {type(batch).__name__}."
            )
        if ignore_index is None:
            ignore_index = batch.label_ignore_index
        elif batch.label_ignore_index != ignore_index:
            raise ValueError(
                "all evaluation batches must share label_ignore_index."
            )
        for record in batch.records:
            if record.participant_id not in expected_set:
                raise ValueError(
                    f"sample {record.sample_id!r} participant "
                    f"{record.participant_id!r} is outside "
                    f"{scope.partition.value!r} evaluation scope."
                )
            if record.sample_id in observed_sample_ids:
                raise ValueError(
                    f"sample {record.sample_id!r} is evaluated more than once."
                )
            observed_sample_ids.add(record.sample_id)
            observed_participants.add(record.participant_id)
            sample_ids.append(record.sample_id)
            participant_ids.append(record.participant_id)

        records_before = batch.records
        record_field_snapshots = tuple(
            (
                record.sample_id,
                record.participant_id,
                record.session_id,
                record.window,
                record.emotion_scores,
                record.speech_source,
                record.physio_sources,
            )
            for record in records_before
        )
        label_snapshots = (
            batch.arousal_labels.clone(),
            batch.valence_labels.clone(),
            batch.quadrant_labels.clone(),
        )
        availability_snapshots = (
            batch.speech_available.clone(),
            batch.physiology_available.clone(),
        )
        step_output = validate_multimodal_batch(
            model,
            objective,
            batch,
            class_weights=class_weights,
        )
        if not isinstance(step_output, MultimodalValidationStepOutput):
            raise RuntimeError(
                "validate_multimodal_batch must return "
                "MultimodalValidationStepOutput."
            )
        if batch.records is not records_before or any(
            current is not original
            for current, original in zip(
                batch.records,
                records_before,
                strict=True,
            )
        ):
            raise RuntimeError(
                "validation must not replace or reorder batch records."
            )
        current_record_fields = tuple(
            (
                record.sample_id,
                record.participant_id,
                record.session_id,
                record.window,
                record.emotion_scores,
                record.speech_source,
                record.physio_sources,
            )
            for record in batch.records
        )
        if current_record_fields != record_field_snapshots:
            raise RuntimeError(
                "validation must not modify batch record fields."
            )
        if not all(
            torch.equal(current, snapshot)
            for current, snapshot in zip(
                (
                    batch.arousal_labels,
                    batch.valence_labels,
                    batch.quadrant_labels,
                ),
                label_snapshots,
                strict=True,
            )
        ):
            raise RuntimeError("validation must not modify batch labels.")
        if not all(
            torch.equal(current, snapshot)
            for current, snapshot in zip(
                (
                    batch.speech_available,
                    batch.physiology_available,
                ),
                availability_snapshots,
                strict=True,
            )
        ):
            raise RuntimeError(
                "validation must not modify batch modality availability."
            )
        output = step_output.model_output
        _validate_probability_output(output, batch)
        fusion_output = output.fusion_output
        modality_weights = fusion_output.modality_weights
        modality_weights_rows.append(
            modality_weights.detach().clone().cpu()
        )
        both = (batch.speech_available & batch.physiology_available).to(
            device=fusion_output.fused_embedding.device
        )
        both_count = int(both.sum().item())
        if both_count > 0:
            summed_weights = modality_weights[both].detach().sum(dim=0).cpu()
            shared_gate_sums["speech"] += float(summed_weights[0].item())
            shared_gate_sums["physiology"] += float(summed_weights[1].item())
            shared_gate_both_count += both_count
            if batch.speech_activity_ratios is not None:
                ratio = batch.speech_activity_ratios.to(
                    device=fusion_output.fused_embedding.device
                )
                if batch.speech_activity_observed is None:
                    raise RuntimeError(
                        "activity ratios require an observation mask."
                    )
                activity_observed = batch.speech_activity_observed.to(
                    device=fusion_output.fused_embedding.device
                )
                activity_masks = speech_activity_bin_masks(
                    ratio,
                    both & activity_observed,
                )
                bucket_masks = {
                    "low": activity_masks[
                        SpeechActivityBin.RATIO_0_1_TO_0_25
                    ],
                    "medium": activity_masks[
                        SpeechActivityBin.RATIO_0_25_TO_0_5
                    ],
                    "high": activity_masks[
                        SpeechActivityBin.RATIO_GE_0_5
                    ],
                }
                for bucket_name, bucket_mask in bucket_masks.items():
                    bucket_count = int(bucket_mask.sum().item())
                    if bucket_count == 0:
                        continue
                    activity_gate_sums[bucket_name]["count"] += bucket_count
                    activity_gate_sums[bucket_name]["speech"] += float(
                        modality_weights[bucket_mask, 0].detach().sum().item()
                    )
        loss_accumulator.add(
            step_output.loss_output,
            optimizer_step_performed=False,
            gradient_norm=0.0,
        )

        speech_available.append(batch.speech_available.detach().clone().cpu())
        source_present = torch.tensor(
            [record.speech_source is not None for record in batch.records],
            dtype=torch.bool,
        )
        speech_source_present.append(source_present)
        speech_source_unavailable_count += int((~source_present).sum().item())
        physiology_available.append(
            batch.physiology_available.detach().clone().cpu()
        )
        batch_activity_observed = torch.zeros_like(source_present)
        batch_activity_ratios = torch.zeros(
            len(batch.records),
            dtype=torch.float32,
        )
        if batch.speech_activity_ratios is not None:
            if batch.speech_activity_observed is None:
                raise RuntimeError(
                    "activity ratios require an observation mask."
                )
            batch_activity_ratios = (
                batch.speech_activity_ratios.detach().clone().cpu()
            )
            batch_activity_observed = (
                batch.speech_activity_observed
                & source_present
            ).detach().clone().cpu()
        speech_activity_observed.append(batch_activity_observed)
        aligned_speech_activity_ratios.append(batch_activity_ratios)
        if bool(batch_activity_observed.any()):
            observed_speech_activity_ratios.append(
                batch_activity_ratios[batch_activity_observed]
                .detach()
                .clone()
                .cpu()
            )
        sample_valid.append(output.sample_valid.detach().clone().cpu())
        arousal_targets.append(batch.arousal_labels.detach().clone().cpu())
        valence_targets.append(batch.valence_labels.detach().clone().cpu())
        quadrant_targets.append(batch.quadrant_labels.detach().clone().cpu())
        arousal_high_probabilities.append(
            output.arousal_probabilities[:, 1].detach().clone().cpu()
        )
        valence_high_probabilities.append(
            output.valence_probabilities[:, 1].detach().clone().cpu()
        )

    loss_epoch = loss_accumulator.output(EpochPhase.VALIDATION)
    assert ignore_index is not None
    if (
        scope.require_all_partition_participants
        and observed_participants != expected_set
    ):
        missing = tuple(
            participant
            for participant in expected_ids
            if participant not in observed_participants
        )
        raise ValueError(
            "evaluation did not observe every required partition participant; "
            f"missing={missing}."
        )

    combined_sample_valid = torch.cat(sample_valid)
    combined_arousal_targets = torch.cat(arousal_targets)
    combined_valence_targets = torch.cat(valence_targets)
    combined_arousal_high = torch.cat(arousal_high_probabilities)
    combined_valence_high = torch.cat(valence_high_probabilities)
    if calibrate_thresholds:
        resolved_thresholds = calibrate_binary_decision_thresholds(
            arousal_high_probabilities=combined_arousal_high,
            valence_high_probabilities=combined_valence_high,
            arousal_targets=combined_arousal_targets,
            valence_targets=combined_valence_targets,
            participant_ids=tuple(participant_ids),
            sample_valid=combined_sample_valid,
            ignore_index=ignore_index,
        )
    else:
        resolved_thresholds = (
            binary_thresholds
            if binary_thresholds is not None
            else BinaryDecisionThresholds()
        )
    calibrated_predictions = _predictions_from_thresholds(
        combined_arousal_high,
        combined_valence_high,
        combined_sample_valid,
        thresholds=resolved_thresholds,
        ignore_index=ignore_index,
    )
    evaluation_predictions = EvaluationPredictions(
        sample_ids=tuple(sample_ids),
        participant_ids=tuple(participant_ids),
        speech_available=torch.cat(speech_available),
        physiology_available=torch.cat(physiology_available),
        sample_valid=combined_sample_valid,
        arousal_targets=combined_arousal_targets,
        arousal_predictions=calibrated_predictions[0],
        valence_targets=combined_valence_targets,
        valence_predictions=calibrated_predictions[1],
        quadrant_targets=torch.cat(quadrant_targets),
        quadrant_predictions=calibrated_predictions[2],
        ignore_index=ignore_index,
        speech_source_present=torch.cat(speech_source_present),
        speech_activity_observed=torch.cat(speech_activity_observed),
        speech_activity_ratios=torch.cat(aligned_speech_activity_ratios),
        modality_weights=torch.cat(modality_weights_rows),
    )
    working_targets = _working_targets(evaluation_predictions)
    all_rows = torch.ones(len(sample_ids), dtype=torch.bool)
    overall_metrics = _metrics_for_rows(
        evaluation_predictions,
        working_targets,
        all_rows,
    )
    pattern_masks = _pattern_masks(evaluation_predictions)
    pattern_counts = {
        pattern: int(mask.sum().item())
        for pattern, mask in pattern_masks.items()
    }

    ordered_participant_ids = tuple(
        participant
        for participant in expected_ids
        if participant in observed_participants
    )
    participant_metrics_list: list[ParticipantEvaluationMetrics] = []
    for participant_id in ordered_participant_ids:
        rows = torch.tensor(
            [
                value == participant_id
                for value in evaluation_predictions.participant_ids
            ],
            dtype=torch.bool,
        )
        participant_counts = {
            pattern: int((rows & pattern_mask).sum().item())
            for pattern, pattern_mask in pattern_masks.items()
        }
        participant_metrics_list.append(
            ParticipantEvaluationMetrics(
                participant_id=participant_id,
                record_count=int(rows.sum().item()),
                modality_pattern_counts=participant_counts,
                task_metrics=_metrics_for_rows(
                    evaluation_predictions,
                    working_targets,
                    rows,
                ),
            )
        )
    participant_metrics = tuple(participant_metrics_list)
    participant_macro = ParticipantMacroEmotionMetrics(
        arousal=_participant_macro_task(
            participant_metrics,
            task_name="arousal",
        ),
        valence=_participant_macro_task(
            participant_metrics,
            task_name="valence",
        ),
        quadrant=_participant_macro_task(
            participant_metrics,
            task_name="quadrant",
        ),
    )
    stratum_metrics = tuple(
        ModalityStratumMetrics(
            pattern=pattern,
            record_count=pattern_counts[pattern],
            task_metrics=_metrics_for_rows(
                evaluation_predictions,
                working_targets,
                pattern_masks[pattern],
            ),
        )
        for pattern in ModalityPattern
    )
    valid_prediction_count = int(
        evaluation_predictions.sample_valid.sum().item()
    )

    def predicted_low_fraction(predictions: Tensor) -> float:
        if valid_prediction_count == 0:
            return 0.0
        return float(
            (
                predictions[evaluation_predictions.sample_valid] == 0
            )
            .to(dtype=torch.float64)
            .mean()
            .item()
        )

    fusion_diagnostics: dict[str, float] = {
        "arousal_predicted_low_fraction": predicted_low_fraction(
            evaluation_predictions.arousal_predictions
        ),
        "valence_predicted_low_fraction": predicted_low_fraction(
            evaluation_predictions.valence_predictions
        ),
        "speech_source_unavailable_count": float(
            speech_source_unavailable_count
        ),
    }
    if shared_gate_both_count > 0:
        fusion_diagnostics.update(
            {
                f"shared_{name}_weight_both": total / shared_gate_both_count
                for name, total in shared_gate_sums.items()
            }
        )
        fusion_diagnostics["shared_gate_both_count"] = float(
            shared_gate_both_count
        )
    if observed_speech_activity_ratios:
        ratios = torch.cat(observed_speech_activity_ratios).to(
            dtype=torch.float64
        )
        fusion_diagnostics.update(
            {
                "speech_available_count": float(
                    evaluation_predictions.speech_available.sum().item()
                ),
                "mean_speech_activity_ratio": float(ratios.mean().item()),
                "median_speech_activity_ratio": float(
                    torch.quantile(ratios, 0.5).item()
                ),
                "p10_speech_activity_ratio": float(
                    torch.quantile(ratios, 0.1).item()
                ),
                "p90_speech_activity_ratio": float(
                    torch.quantile(ratios, 0.9).item()
                ),
            }
        )
        for bucket_name, bucket_values in activity_gate_sums.items():
            bucket_count = int(bucket_values["count"])
            fusion_diagnostics[
                f"activity_{bucket_name}_both_count"
            ] = float(bucket_count)
            if bucket_count > 0:
                fusion_diagnostics[
                    f"activity_{bucket_name}_mean_shared_speech_weight"
                ] = float(bucket_values["speech"]) / bucket_count
    return ParticipantIndependentEvaluationOutput(
        partition=scope.partition,
        batch_count=loss_epoch.batch_count,
        record_count=len(sample_ids),
        participant_count=len(ordered_participant_ids),
        participant_ids=ordered_participant_ids,
        modality_pattern_counts=pattern_counts,
        loss_epoch=loss_epoch,
        overall_metrics=overall_metrics,
        participant_metrics=participant_metrics,
        participant_macro_metrics=participant_macro,
        modality_stratum_metrics=stratum_metrics,
        predictions=evaluation_predictions,
        binary_decision_thresholds=resolved_thresholds,
        fusion_diagnostics=MappingProxyType(fusion_diagnostics),
    )


__all__ = [
    "EvaluationPredictions",
    "ModalityPattern",
    "ModalityStratumMetrics",
    "ParticipantEvaluationMetrics",
    "ParticipantEvaluationScope",
    "ParticipantIndependentEvaluationOutput",
    "ParticipantMacroEmotionMetrics",
    "ParticipantMacroTaskMetrics",
    "evaluate_participant_independent",
]
