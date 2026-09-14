"""Numerically safe classification losses for multimodal tasks."""

import math
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor

_Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            f"reduction must be 'none', 'mean', or 'sum'; received {reduction!r}."
        )


def _validate_classification_inputs(logits: Tensor, target: Tensor, ignore_index: int) -> None:
    if logits.ndim < 2:
        raise ValueError(
            "logits must have shape [N, C, ...] with at least two dimensions; "
            f"received {tuple(logits.shape)}."
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point; received {logits.dtype}.")
    if target.dtype != torch.long:
        raise TypeError(f"target must have dtype torch.long; received {target.dtype}.")
    if logits.device != target.device:
        raise ValueError(
            "logits and target must be on the same device; "
            f"received {logits.device} and {target.device}."
        )
    expected_target_shape = (logits.shape[0], *logits.shape[2:])
    if tuple(target.shape) != expected_target_shape:
        raise ValueError(
            "target shape must exactly match logits without the class dimension; "
            f"expected {expected_target_shape}, received {tuple(target.shape)}."
        )
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError(f"ignore_index must be an integer; received {ignore_index!r}.")

    class_count = logits.shape[1]
    valid_target = target != ignore_index
    invalid_target = valid_target & ((target < 0) | (target >= class_count))
    if bool(invalid_target.any()):
        raise ValueError(
            f"target values must be in [0, {class_count - 1}] or equal "
            f"ignore_index={ignore_index}."
        )


def _validate_class_weights(logits: Tensor, class_weights: Tensor | None) -> None:
    if class_weights is None:
        return
    if not class_weights.is_floating_point():
        raise TypeError(
            f"class_weights must be floating point; received {class_weights.dtype}."
        )
    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError(
            "class_weights must have shape [C] matching logits class dimension; "
            f"expected ({logits.shape[1]},), received {tuple(class_weights.shape)}."
        )
    if class_weights.dtype != logits.dtype:
        raise TypeError(
            "class_weights and logits must have the same dtype; "
            f"received {class_weights.dtype} and {logits.dtype}."
        )
    if class_weights.device != logits.device:
        raise ValueError(
            "class_weights and logits must be on the same device; "
            f"received {class_weights.device} and {logits.device}."
        )
    if not bool(torch.isfinite(class_weights).all()):
        raise ValueError("class_weights must contain only finite values.")
    if not bool((class_weights >= 0).all()):
        raise ValueError("class_weights must be non-negative.")


def _reduce_classification_losses(
    losses: Tensor,
    target: Tensor,
    *,
    class_weights: Tensor | None,
    ignore_index: int,
    reduction: _Reduction,
) -> Tensor:
    if reduction == "none":
        return losses

    numerator = losses.sum()
    if reduction == "sum":
        return numerator

    valid_target = target != ignore_index
    if class_weights is None:
        denominator = valid_target.sum().to(dtype=losses.dtype)
    else:
        safe_target = torch.where(valid_target, target, torch.zeros_like(target))
        selected_weights = class_weights[safe_target]
        denominator = torch.where(
            valid_target,
            selected_weights,
            torch.zeros_like(selected_weights),
        ).sum()
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    return numerator / safe_denominator


def class_weighted_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    class_weights: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Compute class-weighted multiclass cross entropy safely.

    Args:
        logits: Floating-point tensor with shape ``[N, C]`` or the standard
            PyTorch cross-entropy shape ``[N, C, d1, ..., dK]``.
        target: ``torch.long`` tensor with shape ``[N]`` or
            ``[N, d1, ..., dK]`` containing class indices or ``ignore_index``.
        class_weights: Optional finite, non-negative floating tensor with shape
            ``[C]`` and the same dtype/device as ``logits``.
        ignore_index: Integer target value excluded from the loss.
        reduction: One of ``"none"``, ``"mean"``, or ``"sum"``.

    Returns:
        For ``reduction="none"``, a tensor with the same shape as ``target``;
        ignored entries are exactly zero. Otherwise returns a scalar. Mean
        reduction matches PyTorch cross entropy for non-empty effective
        targets. If every target is ignored, or all selected class weights are
        zero, mean reduction returns a finite differentiable zero scalar.

    Raises:
        TypeError: If logits, target, weights, or ``ignore_index`` have invalid
            dtypes.
        ValueError: If shapes/devices mismatch, targets are outside the class
            range, weights are non-finite/negative, or reduction is invalid.

    Inputs are not modified. Class weights are supplied, not estimated here.
    """
    _validate_reduction(reduction)
    _validate_classification_inputs(logits, target, ignore_index)
    _validate_class_weights(logits, class_weights)

    losses = functional.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    return _reduce_classification_losses(
        losses,
        target,
        class_weights=class_weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def multiclass_focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    gamma: float = 2.0,
    class_weights: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Compute multiclass focal loss from target-class log probabilities.

    Args:
        logits: Floating-point tensor with shape ``[N, C]`` or
            ``[N, C, d1, ..., dK]``.
        target: ``torch.long`` tensor with shape ``[N]`` or
            ``[N, d1, ..., dK]`` containing class indices or ``ignore_index``.
        gamma: Finite focusing exponent greater than or equal to zero.
        class_weights: Optional finite, non-negative class alpha/weight tensor
            with shape ``[C]`` and the same dtype/device as ``logits``.
        ignore_index: Integer target value excluded from the loss.
        reduction: One of ``"none"``, ``"mean"``, or ``"sum"``.

    Returns:
        Per-target losses for ``"none"`` or a scalar for ``"mean"``/``"sum"``.
        Ignored targets are exactly zero. A fully ignored target returns a
        finite differentiable zero for scalar reductions. With ``gamma=0``,
        the result equals :func:`class_weighted_cross_entropy` for identical
        weights, ignore index, and reduction.

    Raises:
        TypeError: If logits, target, weights, ``ignore_index``, or ``gamma``
            have invalid types/dtypes.
        ValueError: If shapes/devices mismatch, gamma is negative/non-finite,
            targets are invalid, weights are invalid, or reduction is invalid.

    The implementation uses ``log_softmax`` and does not modify its inputs.
    """
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError(f"gamma must be a real number; received {gamma!r}.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0:
        raise ValueError(f"gamma must be finite and >= 0; received {gamma!r}.")

    _validate_reduction(reduction)
    _validate_classification_inputs(logits, target, ignore_index)
    _validate_class_weights(logits, class_weights)

    valid_target = target != ignore_index
    safe_target = torch.where(valid_target, target, torch.zeros_like(target))
    log_probabilities = functional.log_softmax(logits, dim=1)
    target_log_probability = log_probabilities.gather(
        dim=1,
        index=safe_target.unsqueeze(1),
    ).squeeze(1)
    target_probability_complement = -torch.expm1(target_log_probability)
    losses = target_probability_complement.pow(gamma_value) * (-target_log_probability)

    if class_weights is not None:
        losses = losses * class_weights[safe_target]
    losses = torch.where(valid_target, losses, torch.zeros_like(losses))
    return _reduce_classification_losses(
        losses,
        target,
        class_weights=class_weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )
