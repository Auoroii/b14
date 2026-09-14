"""Single-batch multimodal training and validation steps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from emotion_model.data import AlignedMultimodalBatch
from emotion_model.multimodal import (
    MultimodalEmotionClassifier,
    MultimodalEmotionClassifierOutput,
)
from emotion_model.training.objectives import (
    EmotionTaskClassWeights,
    MultimodalLossOutput,
    MultimodalTrainingObjective,
)


@dataclass(frozen=True)
class MultimodalTrainStepOutput:
    """Result of one supervised or safely skipped training batch.

    Attributes:
        model_output: Final model tensors with logits ``[B, 2]``, quadrant
            probabilities ``[B, 4]``, fused embeddings ``[B, F]``, and
            validity ``[B]``.
        loss_output: Scalar objective diagnostics for the same batch.
        optimizer_step_performed: Whether backward and one optimizer step ran.
        gradient_norm: Detached finite non-negative scalar tensor ``[]`` on
            the model parameter device. It is the pre-clipping global L2 norm,
            or exact zero when the batch has no active supervision.
    """

    model_output: MultimodalEmotionClassifierOutput
    loss_output: MultimodalLossOutput
    optimizer_step_performed: bool
    gradient_norm: Tensor

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_output,
            MultimodalEmotionClassifierOutput,
        ):
            raise TypeError(
                "model_output must be MultimodalEmotionClassifierOutput."
            )
        if not isinstance(self.loss_output, MultimodalLossOutput):
            raise TypeError("loss_output must be MultimodalLossOutput.")
        if not isinstance(self.optimizer_step_performed, bool):
            raise TypeError("optimizer_step_performed must be bool.")
        if (
            not isinstance(self.gradient_norm, Tensor)
            or not self.gradient_norm.is_floating_point()
            or self.gradient_norm.ndim != 0
            or not bool(torch.isfinite(self.gradient_norm))
            or bool(self.gradient_norm < 0)
        ):
            raise ValueError(
                "gradient_norm must be a finite non-negative floating scalar."
            )


@dataclass(frozen=True)
class MultimodalValidationStepOutput:
    """No-gradient model and scalar loss outputs for one validation batch.

    ``model_output`` contains logits ``[B, 2]``, quadrant probabilities
    ``[B, 4]``, fused embeddings ``[B, F]``, and validity ``[B]``.
    ``loss_output`` contains scalar tensors ``[]``.
    """

    model_output: MultimodalEmotionClassifierOutput
    loss_output: MultimodalLossOutput

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_output,
            MultimodalEmotionClassifierOutput,
        ):
            raise TypeError(
                "model_output must be MultimodalEmotionClassifierOutput."
            )
        if not isinstance(self.loss_output, MultimodalLossOutput):
            raise TypeError("loss_output must be MultimodalLossOutput.")


def _validate_step_inputs(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batch: AlignedMultimodalBatch,
) -> None:
    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    if not isinstance(objective, MultimodalTrainingObjective):
        raise TypeError("objective must be MultimodalTrainingObjective.")
    if not isinstance(batch, AlignedMultimodalBatch):
        raise TypeError("batch must be AlignedMultimodalBatch.")


def _parameter_reference(model: MultimodalEmotionClassifier) -> Tensor:
    try:
        return cast(Tensor, next(model.parameters()))
    except StopIteration as error:
        raise RuntimeError("model must contain at least one parameter.") from error


def _validate_max_gradient_norm(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_gradient_norm must be a real number or None.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("max_gradient_norm must be finite and > 0.")
    return normalized


def _gradient_parameters(
    model: MultimodalEmotionClassifier,
) -> list[Tensor]:
    parameters: list[Tensor] = []
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("model gradient contains NaN or Inf.")
            parameters.append(parameter)
    return parameters


def _global_gradient_norm(
    parameters: list[Tensor],
    *,
    reference: Tensor,
) -> Tensor:
    if not parameters:
        return reference.new_zeros(())
    per_parameter = torch.stack(
        [
            torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
            for parameter in parameters
            if parameter.grad is not None
        ]
    )
    return cast(Tensor, torch.linalg.vector_norm(per_parameter, ord=2))


def train_multimodal_batch(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batch: AlignedMultimodalBatch,
    optimizer: torch.optim.Optimizer,
    *,
    class_weights: EmotionTaskClassWeights | None = None,
    max_gradient_norm: float | None = None,
) -> MultimodalTrainStepOutput:
    """Run exactly one multimodal training batch.

    Args:
        model: Train-mode final classifier receiving a logical batch ``B``.
        objective: Parameter-free objective returning scalar losses ``[]``.
        batch: :class:`AlignedMultimodalBatch` with full labels ``[B]``.
        optimizer: Existing PyTorch optimizer; it is not replaced or reconfigured.
        class_weights: Optional task class weights ``[2]``, ``[2]``, ``[4]``.
        max_gradient_norm: Optional finite positive global L2 clipping bound.

    Returns:
        :class:`MultimodalTrainStepOutput`. ``gradient_norm`` is the detached
        pre-clipping global norm. If no enabled valid labels exist, model and
        objective still run once, gradients are cleared, backward/step are
        skipped, and the norm is exact zero.

    Raises:
        TypeError: If public input types are wrong.
        ValueError: If the clipping bound is invalid.
        RuntimeError: If the model is not in train mode, a gradient is
            non-finite, or an updated parameter becomes non-finite.

    This function performs no mode switch, accumulation, AMP, scheduler
    operation, epoch management, metric computation, or state persistence.
    """
    _validate_step_inputs(model, objective, batch)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if not model.training:
        raise RuntimeError("train_multimodal_batch requires model.train() mode.")
    clipping_bound = _validate_max_gradient_norm(max_gradient_norm)
    reference = _parameter_reference(model)
    optimizer.zero_grad(set_to_none=True)
    model_output = model(batch)
    loss_output = objective(
        model_output,
        batch,
        class_weights=class_weights,
    )
    if loss_output.active_target_count == 0:
        return MultimodalTrainStepOutput(
            model_output=model_output,
            loss_output=loss_output,
            optimizer_step_performed=False,
            gradient_norm=reference.new_zeros(()),
        )

    loss_output.total_loss.backward()
    gradient_parameters = _gradient_parameters(model)
    if clipping_bound is None:
        gradient_norm = _global_gradient_norm(
            gradient_parameters,
            reference=reference,
        )
    else:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gradient_parameters,
            clipping_bound,
            error_if_nonfinite=True,
        )
        for parameter in gradient_parameters:
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError("gradient clipping produced NaN or Inf.")
    gradient_norm = gradient_norm.detach().to(
        device=reference.device,
        dtype=reference.dtype,
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("global gradient norm contains NaN or Inf.")
    optimizer.step()
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("optimizer step produced a non-finite parameter.")
    return MultimodalTrainStepOutput(
        model_output=model_output,
        loss_output=loss_output,
        optimizer_step_performed=True,
        gradient_norm=gradient_norm,
    )


def validate_multimodal_batch(
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    batch: AlignedMultimodalBatch,
    *,
    class_weights: EmotionTaskClassWeights | None = None,
) -> MultimodalValidationStepOutput:
    """Run exactly one no-gradient multimodal validation batch.

    Args:
        model: Eval-mode final classifier receiving a logical batch ``B``.
        objective: Parameter-free objective returning scalar losses ``[]``.
        batch: :class:`AlignedMultimodalBatch` with full labels ``[B]``.
        class_weights: Optional task class weights ``[2]``, ``[2]``, ``[4]``.

    Returns:
        :class:`MultimodalValidationStepOutput` containing model tensors
        ``[B, ...]`` and scalar loss tensors ``[]``, all without autograd
        requirements. Empty supervision still returns complete zero losses.

    Raises:
        TypeError: If public input types are wrong.
        RuntimeError: If the model is not in eval mode.

    The function does not alter model mode, parameters, optimizer state, or
    cross-batch state, and computes no metrics.
    """
    _validate_step_inputs(model, objective, batch)
    if model.training:
        raise RuntimeError(
            "validate_multimodal_batch requires model.eval() mode."
        )
    with torch.no_grad():
        model_output = model(batch)
        loss_output = objective(
            model_output,
            batch,
            class_weights=class_weights,
        )
    return MultimodalValidationStepOutput(
        model_output=model_output,
        loss_output=loss_output,
    )


__all__ = [
    "MultimodalTrainStepOutput",
    "MultimodalValidationStepOutput",
    "train_multimodal_batch",
    "validate_multimodal_batch",
]
