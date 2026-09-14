"""Epoch-level orchestration for multimodal training and validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import torch

from emotion_model.data import AlignedMultimodalBatch
from emotion_model.multimodal import MultimodalEmotionClassifier
from emotion_model.training.objectives import (
    EmotionTaskClassWeights,
    MultimodalLossOutput,
    MultimodalTrainingObjective,
)
from emotion_model.training.steps import (
    MultimodalTrainStepOutput,
    MultimodalValidationStepOutput,
    train_multimodal_batch,
    validate_multimodal_batch,
)

_LOSS_FIELD_NAMES = (
    "total_loss",
    "fused_loss",
    "fused_arousal_loss",
    "fused_valence_loss",
    "fused_quadrant_loss",
    "speech_auxiliary_loss",
    "speech_arousal_loss",
    "speech_valence_loss",
    "speech_quadrant_loss",
    "physiology_auxiliary_loss",
    "physiology_arousal_loss",
    "physiology_valence_loss",
    "physiology_quadrant_loss",
)


class EpochPhase(StrEnum):
    """Supported epoch runner phases."""

    TRAIN = "train"
    VALIDATION = "validation"


def _validate_finite_nonnegative_float(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a Python float, not bool.")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class MultimodalEpochLossAverages:
    """Detached arithmetic means over supervised batches in one epoch.

    Each field is a finite non-negative Python ``float``. The denominator is
    the number of batches whose ``active_target_count`` is greater than zero;
    empty-supervision batches are excluded. These are batch-wise arithmetic
    means, not sample-weighted, participant-weighted, or evaluation metrics.
    If no batch has supervision, every field is ``0.0``.
    """

    total_loss: float
    fused_loss: float
    fused_arousal_loss: float
    fused_valence_loss: float
    fused_quadrant_loss: float
    speech_auxiliary_loss: float
    speech_arousal_loss: float
    speech_valence_loss: float
    speech_quadrant_loss: float
    physiology_auxiliary_loss: float
    physiology_arousal_loss: float
    physiology_valence_loss: float
    physiology_quadrant_loss: float

    def __post_init__(self) -> None:
        for name in _LOSS_FIELD_NAMES:
            _validate_finite_nonnegative_float(
                getattr(self, name),
                name=name,
            )


def _validate_nonnegative_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class MultimodalEpochOutput:
    """Detached aggregate diagnostics for one non-empty epoch.

    ``loss_averages`` contains Python floats only. Counts are Python integers,
    and ``max_gradient_norm`` is the maximum pre-clipping norm from actual
    optimizer steps. No model output, loss tensor, batch, gradient, optimizer
    state, or computation graph is retained.
    """

    phase: EpochPhase
    batch_count: int
    supervised_batch_count: int
    empty_supervision_batch_count: int
    optimizer_step_count: int
    active_target_count: int
    loss_averages: MultimodalEpochLossAverages
    max_gradient_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.phase, EpochPhase):
            raise TypeError("phase must be EpochPhase.")
        for name in (
            "batch_count",
            "supervised_batch_count",
            "empty_supervision_batch_count",
            "optimizer_step_count",
            "active_target_count",
        ):
            _validate_nonnegative_integer(getattr(self, name), name=name)
        if self.batch_count <= 0:
            raise ValueError("batch_count must be > 0.")
        if (
            self.supervised_batch_count
            + self.empty_supervision_batch_count
            != self.batch_count
        ):
            raise ValueError(
                "supervised and empty-supervision counts must sum to batch_count."
            )
        if (
            self.supervised_batch_count == 0
            and self.active_target_count != 0
        ) or self.active_target_count < self.supervised_batch_count:
            raise ValueError(
                "active_target_count must be zero exactly when no batch is "
                "supervised, and otherwise at least supervised_batch_count."
            )
        if self.phase is EpochPhase.TRAIN:
            if self.optimizer_step_count != self.supervised_batch_count:
                raise ValueError(
                    "train optimizer_step_count must equal supervised_batch_count."
                )
        elif self.optimizer_step_count != 0:
            raise ValueError(
                "validation optimizer_step_count must be zero."
            )
        if not isinstance(self.loss_averages, MultimodalEpochLossAverages):
            raise TypeError(
                "loss_averages must be MultimodalEpochLossAverages."
            )
        _validate_finite_nonnegative_float(
            self.max_gradient_norm,
            name="max_gradient_norm",
        )
        if (
            self.phase is EpochPhase.VALIDATION
            or self.optimizer_step_count == 0
        ) and self.max_gradient_norm != 0.0:
            raise ValueError(
                "validation or no-step epoch max_gradient_norm must be 0.0."
            )


@dataclass(frozen=True)
class MultimodalTrainingState:
    """Serializable progress across completed multimodal training epochs."""

    completed_epochs: int = 0
    global_optimizer_steps: int = 0
    best_validation_loss: float | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_integer(
            self.completed_epochs,
            name="completed_epochs",
        )
        _validate_nonnegative_integer(
            self.global_optimizer_steps,
            name="global_optimizer_steps",
        )
        if self.best_validation_loss is not None:
            _validate_finite_nonnegative_float(
                self.best_validation_loss,
                name="best_validation_loss",
            )


def advance_training_state(
    state: MultimodalTrainingState,
    training_epoch: MultimodalEpochOutput,
    validation_epoch: MultimodalEpochOutput | None = None,
) -> MultimodalTrainingState:
    """Advance immutable progress by one completed training epoch.

    Args:
        state: Existing non-negative epoch/optimizer-step counters.
        training_epoch: One aggregate with ``phase=TRAIN``.
        validation_epoch: Optional aggregate with ``phase=VALIDATION``.

    Returns:
        A new :class:`MultimodalTrainingState`. Epochs increase by one and
        global steps increase by the training epoch's actual step count.
        Validation total loss updates the best value only when that epoch has
        at least one supervised batch.

    Raises:
        TypeError: If inputs have wrong public types.
        ValueError: If an epoch has the wrong phase.
    """
    if not isinstance(state, MultimodalTrainingState):
        raise TypeError("state must be MultimodalTrainingState.")
    if not isinstance(training_epoch, MultimodalEpochOutput):
        raise TypeError("training_epoch must be MultimodalEpochOutput.")
    if training_epoch.phase is not EpochPhase.TRAIN:
        raise ValueError("training_epoch must have phase TRAIN.")
    if validation_epoch is not None:
        if not isinstance(validation_epoch, MultimodalEpochOutput):
            raise TypeError(
                "validation_epoch must be MultimodalEpochOutput or None."
            )
        if validation_epoch.phase is not EpochPhase.VALIDATION:
            raise ValueError("validation_epoch must have phase VALIDATION.")

    best = state.best_validation_loss
    if (
        validation_epoch is not None
        and validation_epoch.supervised_batch_count > 0
    ):
        candidate = validation_epoch.loss_averages.total_loss
        best = candidate if best is None else min(best, candidate)
    return MultimodalTrainingState(
        completed_epochs=state.completed_epochs + 1,
        global_optimizer_steps=(
            state.global_optimizer_steps
            + training_epoch.optimizer_step_count
        ),
        best_validation_loss=best,
    )


class _EpochAccumulator:
    """Private detached scalar accumulator."""

    def __init__(self) -> None:
        self.batch_count = 0
        self.supervised_batch_count = 0
        self.empty_supervision_batch_count = 0
        self.optimizer_step_count = 0
        self.active_target_count = 0
        self.max_gradient_norm = 0.0
        self.loss_sums = {name: 0.0 for name in _LOSS_FIELD_NAMES}

    def add(
        self,
        loss_output: MultimodalLossOutput,
        *,
        optimizer_step_performed: bool,
        gradient_norm: float,
    ) -> None:
        self.batch_count += 1
        self.active_target_count += loss_output.active_target_count
        if loss_output.active_target_count == 0:
            self.empty_supervision_batch_count += 1
            return
        self.supervised_batch_count += 1
        if optimizer_step_performed:
            self.optimizer_step_count += 1
            self.max_gradient_norm = max(
                self.max_gradient_norm,
                gradient_norm,
            )
        for name in _LOSS_FIELD_NAMES:
            value = float(getattr(loss_output, name).detach().item())
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    f"epoch batch {name} must be finite and non-negative."
                )
            self.loss_sums[name] += value

    def output(self, phase: EpochPhase) -> MultimodalEpochOutput:
        if self.batch_count == 0:
            raise ValueError("epoch batches iterable must not be empty.")
        denominator = self.supervised_batch_count
        averages = {
            name: (
                self.loss_sums[name] / denominator
                if denominator > 0
                else 0.0
            )
            for name in _LOSS_FIELD_NAMES
        }
        loss_averages = MultimodalEpochLossAverages(
            total_loss=averages["total_loss"],
            fused_loss=averages["fused_loss"],
            fused_arousal_loss=averages["fused_arousal_loss"],
            fused_valence_loss=averages["fused_valence_loss"],
            fused_quadrant_loss=averages["fused_quadrant_loss"],
            speech_auxiliary_loss=averages["speech_auxiliary_loss"],
            speech_arousal_loss=averages["speech_arousal_loss"],
            speech_valence_loss=averages["speech_valence_loss"],
            speech_quadrant_loss=averages["speech_quadrant_loss"],
            physiology_auxiliary_loss=averages[
                "physiology_auxiliary_loss"
            ],
            physiology_arousal_loss=averages[
                "physiology_arousal_loss"
            ],
            physiology_valence_loss=averages[
                "physiology_valence_loss"
            ],
            physiology_quadrant_loss=averages[
                "physiology_quadrant_loss"
            ],
        )
        return MultimodalEpochOutput(
            phase=phase,
            batch_count=self.batch_count,
            supervised_batch_count=self.supervised_batch_count,
            empty_supervision_batch_count=self.empty_supervision_batch_count,
            optimizer_step_count=self.optimizer_step_count,
            active_target_count=self.active_target_count,
            loss_averages=loss_averages,
            max_gradient_norm=(
                self.max_gradient_norm if phase is EpochPhase.TRAIN else 0.0
            ),
        )


def _validate_runner_inputs(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batches: Iterable[AlignedMultimodalBatch],
) -> None:
    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    if not isinstance(objective, MultimodalTrainingObjective):
        raise TypeError("objective must be MultimodalTrainingObjective.")
    if isinstance(
        batches,
        (str, bytes, torch.Tensor, AlignedMultimodalBatch),
    ) or not isinstance(batches, Iterable):
        raise TypeError(
            "batches must be a non-string iterable of AlignedMultimodalBatch."
        )


def run_multimodal_training_epoch(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batches: Iterable[AlignedMultimodalBatch],
    optimizer: torch.optim.Optimizer,
    *,
    class_weights: EmotionTaskClassWeights | None = None,
    max_gradient_norm: float | None = None,
) -> MultimodalEpochOutput:
    """Run one ordered, non-empty iterable of training batches.

    Args:
        model: Final multimodal classifier, switched to train mode once.
        objective: Parameter-free per-batch training objective.
        batches: One-pass iterable of logical batches; it is not sized,
            reordered, cached, or retained.
        optimizer: Existing PyTorch optimizer passed to every train step.
        class_weights: Optional task weights ``[2]``, ``[2]``, ``[4]``.
        max_gradient_norm: Optional positive per-step clipping bound.

    Returns:
        Detached :class:`MultimodalEpochOutput` with Python counts/floats.
        Losses are arithmetic means over supervised batches only, and the
        maximum norm is taken over actual optimizer steps.

    Raises:
        TypeError: If public inputs or an iterated batch have wrong types.
        ValueError: If the iterable is empty.

    The model remains in train mode. Each batch calls the stage-13A training
    step exactly once. No validation, scheduling, accumulation, metric, or
    prediction retention is performed.
    """
    _validate_runner_inputs(model, objective, batches)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    model.train()
    accumulator = _EpochAccumulator()
    for batch in batches:
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError(
                "every epoch item must be AlignedMultimodalBatch; "
                f"received {type(batch).__name__}."
            )
        step_output = train_multimodal_batch(
            model,
            objective,
            batch,
            optimizer,
            class_weights=class_weights,
            max_gradient_norm=max_gradient_norm,
        )
        if not isinstance(step_output, MultimodalTrainStepOutput):
            raise RuntimeError(
                "train_multimodal_batch must return MultimodalTrainStepOutput."
            )
        if step_output.optimizer_step_performed != (
            step_output.loss_output.active_target_count > 0
        ):
            raise RuntimeError(
                "train step flag must match active supervision."
            )
        accumulator.add(
            step_output.loss_output,
            optimizer_step_performed=step_output.optimizer_step_performed,
            gradient_norm=float(step_output.gradient_norm.detach().item()),
        )
    return accumulator.output(EpochPhase.TRAIN)


def run_multimodal_validation_epoch(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batches: Iterable[AlignedMultimodalBatch],
    *,
    class_weights: EmotionTaskClassWeights | None = None,
) -> MultimodalEpochOutput:
    """Run one ordered, non-empty iterable of validation batches.

    Args:
        model: Final multimodal classifier, switched to eval mode once.
        objective: Parameter-free per-batch objective.
        batches: One-pass iterable of logical batches; it is not sized,
            reordered, cached, or retained.
        class_weights: Optional task weights ``[2]``, ``[2]``, ``[4]``.

    Returns:
        Detached :class:`MultimodalEpochOutput` with validation phase, zero
        optimizer steps/norm, and batch-wise supervised loss averages.

    Raises:
        TypeError: If public inputs or an iterated batch have wrong types.
        ValueError: If the iterable is empty.

    The model remains in eval mode. Each batch calls the stage-13A no-gradient
    validation step exactly once. Parameters, predictions, labels, metrics,
    and cross-batch state are not retained.
    """
    _validate_runner_inputs(model, objective, batches)
    model.eval()
    accumulator = _EpochAccumulator()
    for batch in batches:
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError(
                "every epoch item must be AlignedMultimodalBatch; "
                f"received {type(batch).__name__}."
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
        accumulator.add(
            step_output.loss_output,
            optimizer_step_performed=False,
            gradient_norm=0.0,
        )
    return accumulator.output(EpochPhase.VALIDATION)


__all__ = [
    "EpochPhase",
    "MultimodalEpochLossAverages",
    "MultimodalEpochOutput",
    "MultimodalTrainingState",
    "advance_training_state",
    "run_multimodal_training_epoch",
    "run_multimodal_validation_epoch",
]
