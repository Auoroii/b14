"""Tests for safe weighted cross entropy and multiclass focal loss."""

from collections.abc import Callable
from typing import Literal

import pytest
import torch
import torch.nn.functional as functional

from emotion_model.multimodal import (
    class_weighted_cross_entropy,
    multiclass_focal_loss,
)

LossFunction = Callable[..., torch.Tensor]


def test_cross_entropy_default_matches_pytorch() -> None:
    """Match PyTorch's default unweighted mean cross entropy."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [0.2, 0.7, -0.3]])
    target = torch.tensor([0, 2])

    actual = class_weighted_cross_entropy(logits, target)
    expected = functional.cross_entropy(logits, target)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_weighted_cross_entropy_matches_pytorch_standard_shapes(
    reduction: Literal["none", "mean", "sum"],
) -> None:
    """Match PyTorch for spatial targets, class weights, ignore, and reductions."""
    logits = torch.tensor(
        [
            [[2.0, 0.5], [0.0, 1.0], [-1.0, 2.0]],
            [[0.5, 1.5], [1.0, -0.5], [0.0, 0.25]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[0, 2], [1, -100]])
    class_weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    actual = class_weighted_cross_entropy(
        logits,
        target,
        class_weights=class_weights,
        reduction=reduction,
    )
    expected = functional.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=-100,
        reduction=reduction,
    )

    torch.testing.assert_close(actual, expected)
    if reduction == "none":
        assert actual[1, 1].item() == 0.0


def test_cross_entropy_all_ignored_returns_differentiable_zero() -> None:
    """Return finite zero with weights and zero gradients for all ignored."""
    logits = torch.randn((3, 2), requires_grad=True)
    target = torch.full((3,), -100, dtype=torch.long)
    class_weights = torch.tensor([1.0, 2.0])

    loss = class_weighted_cross_entropy(
        logits,
        target,
        class_weights=class_weights,
    )
    loss.backward()

    assert loss.ndim == 0
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_focal_gamma_zero_equals_cross_entropy(
    reduction: Literal["none", "mean", "sum"],
) -> None:
    """Reduce focal loss to weighted cross entropy when gamma is zero."""
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.5, -0.5]], dtype=torch.float64)
    target = torch.tensor([0, -9])
    class_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    focal = multiclass_focal_loss(
        logits,
        target,
        gamma=0.0,
        class_weights=class_weights,
        ignore_index=-9,
        reduction=reduction,
    )
    cross_entropy = class_weighted_cross_entropy(
        logits,
        target,
        class_weights=class_weights,
        ignore_index=-9,
        reduction=reduction,
    )

    torch.testing.assert_close(focal, cross_entropy)


def test_focal_positive_gamma_matches_manual_weighted_reference() -> None:
    """Match a manual target-log-probability focal calculation."""
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    target = torch.tensor([0, 2])
    class_weights = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float64)
    gamma = 2.0

    actual = multiclass_focal_loss(
        logits,
        target,
        gamma=gamma,
        class_weights=class_weights,
    )

    log_probabilities = functional.log_softmax(logits, dim=1)
    target_log_probability = log_probabilities[torch.arange(2), target]
    target_probability = target_log_probability.exp()
    per_sample = (
        -(1.0 - target_probability).pow(gamma)
        * target_log_probability
        * class_weights[target]
    )
    expected = per_sample.sum() / class_weights[target].sum()
    torch.testing.assert_close(actual, expected)


def test_focal_positive_gamma_supports_spatial_targets_and_ignore() -> None:
    """Apply target-only focal factors to spatial logits and zero ignored entries."""
    logits = torch.tensor(
        [[[2.0, 0.5], [0.0, 1.0], [-1.0, 2.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[0, -100]])
    class_weights = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float64)
    gamma = 1.5

    actual = multiclass_focal_loss(
        logits,
        target,
        gamma=gamma,
        class_weights=class_weights,
        reduction="none",
    )

    target_log_probability = functional.log_softmax(logits, dim=1)[0, 0, 0]
    expected_valid = (
        -(1.0 - target_log_probability.exp()).pow(gamma)
        * target_log_probability
        * class_weights[0]
    )
    expected = torch.stack((expected_valid, torch.zeros_like(expected_valid))).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_focal_all_ignored_returns_differentiable_zero() -> None:
    """Return finite zero and finite gradients with weights when all ignored."""
    logits = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
    target = torch.full((2,), -100, dtype=torch.long)
    class_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    loss = multiclass_focal_loss(
        logits,
        target,
        gamma=2.0,
        class_weights=class_weights,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("gamma", [-1.0, float("nan"), float("inf")])
def test_focal_rejects_invalid_gamma(gamma: float) -> None:
    """Reject negative and non-finite focusing exponents."""
    with pytest.raises(ValueError, match="gamma"):
        multiclass_focal_loss(
            torch.ones((1, 2)),
            torch.tensor([0]),
            gamma=gamma,
        )


@pytest.mark.parametrize(
    "class_weights",
    [
        torch.ones((2, 2)),
        torch.tensor([1, 2]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("nan")]),
    ],
)
def test_losses_reject_invalid_class_weights(class_weights: torch.Tensor) -> None:
    """Reject weights with invalid shape, dtype, sign, or finiteness."""
    with pytest.raises((TypeError, ValueError)):
        multiclass_focal_loss(
            torch.ones((2, 2)),
            torch.tensor([0, 1]),
            class_weights=class_weights,
        )


@pytest.mark.parametrize(
    ("target", "expected_exception"),
    [
        (torch.tensor([0.0]), TypeError),
        (torch.tensor([2]), ValueError),
        (torch.tensor([[0]]), ValueError),
    ],
)
def test_losses_reject_invalid_targets(
    target: torch.Tensor,
    expected_exception: type[Exception],
) -> None:
    """Reject target dtype, class range, and shape errors clearly."""
    for loss_function in (class_weighted_cross_entropy, multiclass_focal_loss):
        with pytest.raises(expected_exception):
            loss_function(torch.ones((1, 2)), target)


@pytest.mark.parametrize(
    "loss_function",
    [class_weighted_cross_entropy, multiclass_focal_loss],
)
def test_losses_reject_invalid_reduction(loss_function: LossFunction) -> None:
    """Reject reductions outside none, mean, and sum."""
    with pytest.raises(ValueError, match="reduction"):
        loss_function(
            torch.ones((1, 2)),
            torch.tensor([0]),
            reduction="median",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "loss_function",
    [class_weighted_cross_entropy, multiclass_focal_loss],
)
def test_losses_preserve_inputs_and_propagate_finite_gradients(
    dtype: torch.dtype,
    loss_function: LossFunction,
) -> None:
    """Preserve inputs and backpropagate finite gradients in both dtypes."""
    logits = torch.tensor(
        [[1.0, -0.5, 0.25], [-1.0, 2.0, 0.5]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor([0, 1])
    class_weights = torch.tensor([1.0, 2.0, 0.5], dtype=dtype)
    original_logits = logits.detach().clone()
    original_target = target.clone()
    original_weights = class_weights.clone()

    loss = loss_function(logits, target, class_weights=class_weights)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0
    torch.testing.assert_close(logits.detach(), original_logits)
    assert torch.equal(target, original_target)
    assert torch.equal(class_weights, original_weights)


def test_mean_losses_are_safe_when_selected_class_weights_are_zero() -> None:
    """Return differentiable zero when all observed classes have zero weight."""
    logits = torch.tensor([[1.0, 0.0], [0.5, -0.5]], requires_grad=True)
    target = torch.tensor([0, 0])
    class_weights = torch.tensor([0.0, 1.0])

    cross_entropy = class_weighted_cross_entropy(
        logits,
        target,
        class_weights=class_weights,
    )
    focal = multiclass_focal_loss(
        logits,
        target,
        class_weights=class_weights,
    )
    (cross_entropy + focal).backward()

    assert cross_entropy.item() == 0.0
    assert focal.item() == 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
