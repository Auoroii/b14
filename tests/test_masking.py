"""Tests for common mask validation and masking operations."""

from collections.abc import Callable

import pytest
import torch

from emotion_model.common import (
    apply_query_mask,
    safe_masked_softmax,
    valid_to_key_padding_mask,
    validate_sequence_mask,
)


def test_valid_to_key_padding_mask_inverts_polarity() -> None:
    """Convert True-valid masks to True-ignore masks exactly."""
    valid_mask = torch.tensor([[True, False, True], [False, False, True]])

    key_padding_mask = valid_to_key_padding_mask(valid_mask)

    expected = torch.tensor([[False, True, False], [True, True, False]])
    assert torch.equal(key_padding_mask, expected)
    assert torch.equal(valid_mask, torch.tensor([[True, False, True], [False, False, True]]))


def test_apply_query_mask_zeros_only_invalid_queries_without_mutation() -> None:
    """Keep valid queries, zero non-finite padding, and preserve both inputs."""
    sequence = torch.tensor(
        [
            [[1.0, 2.0], [float("nan"), float("inf")], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [11.0, 12.0]],
        ]
    )
    valid_mask = torch.tensor([[True, False, True], [False, False, False]])
    original_sequence = sequence.clone()
    original_mask = valid_mask.clone()

    masked = apply_query_mask(sequence, valid_mask)

    expected = torch.tensor(
        [
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    assert torch.equal(masked, expected)
    torch.testing.assert_close(sequence, original_sequence, equal_nan=True)
    assert torch.equal(valid_mask, original_mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_safe_masked_softmax_normalizes_valid_positions(dtype: torch.dtype) -> None:
    """Normalize valid positions and preserve the floating-point dtype."""
    logits = torch.tensor([[1.0, 3.0, -2.0], [2.0, -1.0, 4.0]], dtype=dtype)
    valid_mask = torch.tensor([[True, True, False], [False, True, True]])

    probabilities = safe_masked_softmax(logits, valid_mask)

    assert probabilities.dtype == dtype
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, dtype=dtype),
    )
    assert torch.equal(probabilities[~valid_mask], torch.zeros(2, dtype=dtype))


def test_safe_masked_softmax_ignores_nonfinite_masked_logits() -> None:
    """Ignore non-finite padding and return zeros for a fully masked row."""
    logits = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf"), 1.0],
            [1.0, float("nan"), 2.0, float("inf")],
        ]
    )
    valid_mask = torch.tensor(
        [[False, False, False, False], [True, False, True, False]]
    )
    original_logits = logits.clone()
    original_mask = valid_mask.clone()

    probabilities = safe_masked_softmax(logits, valid_mask)

    assert torch.equal(probabilities[0], torch.zeros(4))
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities[1].sum(), torch.tensor(1.0))
    assert probabilities[1, 1].item() == 0.0
    assert probabilities[1, 3].item() == 0.0
    torch.testing.assert_close(logits, original_logits, equal_nan=True)
    assert torch.equal(valid_mask, original_mask)


def test_safe_masked_softmax_supports_attention_mask_alignment() -> None:
    """Align [B, K] and explicit singleton masks to [B, H, Q, K]."""
    logits = torch.arange(48, dtype=torch.float64).reshape(2, 2, 3, 4)
    valid_mask = torch.tensor([[True, False, True, False], [False, False, False, False]])

    from_batch_time = safe_masked_softmax(logits, valid_mask, dim=-1)
    explicit_mask = valid_mask[:, None, None, :]
    from_explicit_mask = safe_masked_softmax(logits, explicit_mask, dim=-1)

    torch.testing.assert_close(from_batch_time, from_explicit_mask)
    torch.testing.assert_close(
        from_batch_time[0].sum(dim=-1),
        torch.ones((2, 3), dtype=torch.float64),
    )
    assert torch.equal(from_batch_time[1], torch.zeros((2, 3, 4), dtype=torch.float64))
    assert torch.equal(
        from_batch_time[..., 1],
        torch.zeros((2, 2, 3), dtype=torch.float64),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_safe_masked_softmax_propagates_finite_gradients(dtype: torch.dtype) -> None:
    """Backpropagate in both supported dtypes while blocking masked logits."""
    logits = torch.tensor([[0.5, -3.0, 1.5]], dtype=dtype, requires_grad=True)
    valid_mask = torch.tensor([[True, False, True]])
    weights = torch.tensor([[1.0, 2.0, -1.0]], dtype=dtype)

    probabilities = safe_masked_softmax(logits, valid_mask)
    (probabilities * weights).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[0, [0, 2]].abs().sum().item() > 0.0


@pytest.mark.parametrize(
    ("operation", "mask"),
    [
        (
            lambda sequence, valid_mask: validate_sequence_mask(sequence, valid_mask),
            torch.ones((2, 3), dtype=torch.float32),
        ),
        (
            lambda sequence, valid_mask: apply_query_mask(sequence, valid_mask),
            torch.ones((2, 3), dtype=torch.int64),
        ),
    ],
)
def test_sequence_mask_operations_reject_non_bool_masks(
    operation: Callable[[torch.Tensor, torch.Tensor], None | torch.Tensor],
    mask: torch.Tensor,
) -> None:
    """Reject float and integer masks instead of converting them implicitly."""
    sequence = torch.ones((2, 3, 4))

    with pytest.raises(TypeError, match="torch.bool"):
        operation(sequence, mask)


def test_safe_masked_softmax_rejects_non_bool_mask() -> None:
    """Reject a floating-point mask at the softmax boundary."""
    with pytest.raises(TypeError, match="torch.bool"):
        safe_masked_softmax(torch.ones((2, 3)), torch.ones((2, 3)))


@pytest.mark.parametrize(
    "valid_mask",
    [
        torch.ones((2, 3, 1), dtype=torch.bool),
        torch.ones((1, 3), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
    ],
)
def test_validate_sequence_mask_rejects_mismatched_shapes(valid_mask: torch.Tensor) -> None:
    """Reject extra dimensions and all batch/time size mismatches."""
    sequence = torch.ones((2, 3, 4))

    with pytest.raises(ValueError, match="shape"):
        validate_sequence_mask(sequence, valid_mask)


def test_safe_masked_softmax_rejects_ambiguous_mask_shape() -> None:
    """Reject a [B, T] mask that does not align to the selected dimension."""
    logits = torch.ones((2, 3, 4))
    invalid_mask = torch.ones((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="explicit"):
        safe_masked_softmax(logits, invalid_mask, dim=-1)


def test_safe_masked_softmax_rejects_batch_broadcast() -> None:
    """Reject a singleton mask batch that could hide missing sample masks."""
    logits = torch.ones((2, 3, 4, 5))
    batch_broadcast_mask = torch.ones((1, 1, 1, 5), dtype=torch.bool)

    with pytest.raises(ValueError, match="batch size"):
        safe_masked_softmax(logits, batch_broadcast_mask, dim=-1)


@pytest.mark.parametrize(
    "invalid_mask",
    [
        torch.ones((2, 3), dtype=torch.float32),
        torch.ones((2, 3, 1), dtype=torch.bool),
    ],
)
def test_valid_to_key_padding_mask_rejects_invalid_masks(
    invalid_mask: torch.Tensor,
) -> None:
    """Reject non-boolean and non-[B,T] key-padding source masks."""
    expected_exception = TypeError if invalid_mask.dtype != torch.bool else ValueError

    with pytest.raises(expected_exception):
        valid_to_key_padding_mask(invalid_mask)
