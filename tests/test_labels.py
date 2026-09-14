"""Tests for label protocols and derived quadrant utilities."""

import pytest
import torch

from emotion_model.common import (
    LabelProtocol,
    binarize_emotion_scores,
    derive_quadrant_labels,
    derive_quadrant_probabilities,
)


@pytest.mark.parametrize(
    ("protocol", "expected_labels", "expected_valid"),
    [
        (
            LabelProtocol.OFFICIAL_MID_HIGH,
            [0, 0, 1, 1, 1],
            [True, True, True, True, True],
        ),
        (
            LabelProtocol.STRICT_DROP_MID,
            [0, 0, -100, 1, 1],
            [True, True, False, True, True],
        ),
        (
            LabelProtocol.MID_LOW,
            [0, 0, 0, 1, 1],
            [True, True, True, True, True],
        ),
    ],
)
def test_binarize_emotion_scores_maps_all_protocols(
    protocol: LabelProtocol,
    expected_labels: list[int],
    expected_valid: list[bool],
) -> None:
    """Map every score from 1 through 5 under each named protocol."""
    scores = torch.arange(1, 6)

    labels, valid_mask = binarize_emotion_scores(scores, protocol)

    assert labels.dtype == torch.long
    assert valid_mask.dtype == torch.bool
    assert torch.equal(labels, torch.tensor(expected_labels))
    assert torch.equal(valid_mask, torch.tensor(expected_valid))


def test_binarize_supports_string_protocol_and_custom_ignore_index() -> None:
    """Resolve a string protocol and propagate a caller-selected ignore index."""
    labels, valid_mask = binarize_emotion_scores(
        torch.tensor([2.0, 3.0, 4.0]),
        "strict_drop_mid",
        ignore_index=-7,
    )

    assert torch.equal(labels, torch.tensor([0, -7, 1]))
    assert torch.equal(valid_mask, torch.tensor([True, False, True]))


@pytest.mark.parametrize("protocol", list(LabelProtocol))
def test_string_and_enum_protocols_are_equivalent(protocol: LabelProtocol) -> None:
    """Produce identical labels and validity for enum and string forms."""
    scores = torch.arange(1, 6, dtype=torch.float64)

    enum_result = binarize_emotion_scores(scores, protocol)
    string_result = binarize_emotion_scores(scores, protocol.value)

    assert torch.equal(enum_result[0], string_result[0])
    assert torch.equal(enum_result[1], string_result[1])


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([0]),
        torch.tensor([6]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([2.5]),
    ],
)
def test_binarize_rejects_invalid_scores(scores: torch.Tensor) -> None:
    """Reject out-of-range, non-finite, and fractional scores."""
    with pytest.raises(ValueError):
        binarize_emotion_scores(scores, LabelProtocol.OFFICIAL_MID_HIGH)


def test_binarize_does_not_modify_scores() -> None:
    """Leave the caller's score tensor unchanged."""
    scores = torch.tensor([[1.0, 3.0, 5.0]])
    original = scores.clone()

    binarize_emotion_scores(scores, LabelProtocol.STRICT_DROP_MID)

    assert torch.equal(scores, original)


def test_binarize_rejects_unknown_protocol() -> None:
    """Reject unknown protocol strings with a clear error."""
    with pytest.raises(ValueError, match="Unknown label protocol"):
        binarize_emotion_scores(torch.tensor([1, 2]), "unknown")


def test_derive_quadrant_labels_maps_all_four_classes() -> None:
    """Produce the fixed LALV, HALV, LAHV, HAHV class order."""
    arousal = torch.tensor([0, 1, 0, 1])
    valence = torch.tensor([0, 0, 1, 1])

    quadrant = derive_quadrant_labels(arousal, valence)

    assert quadrant.dtype == torch.long
    assert torch.equal(quadrant, torch.tensor([0, 1, 2, 3]))


def test_derive_quadrant_labels_returns_long_from_other_integer_dtypes() -> None:
    """Return torch.long even when valid input labels use smaller integers."""
    arousal = torch.tensor([0, 1], dtype=torch.int16)
    valence = torch.tensor([1, 0], dtype=torch.int32)

    quadrant = derive_quadrant_labels(arousal, valence)

    assert quadrant.dtype == torch.long
    assert torch.equal(quadrant, torch.tensor([2, 1]))


def test_derive_quadrant_labels_propagates_ignore_without_mutation() -> None:
    """Propagate ignore_index from either axis and preserve both inputs."""
    arousal = torch.tensor([0, -9, 1, 0])
    valence = torch.tensor([1, 1, -9, -9])
    original_arousal = arousal.clone()
    original_valence = valence.clone()

    quadrant = derive_quadrant_labels(arousal, valence, ignore_index=-9)

    assert torch.equal(quadrant, torch.tensor([2, -9, -9, -9]))
    assert torch.equal(arousal, original_arousal)
    assert torch.equal(valence, original_valence)


def test_derive_quadrant_labels_rejects_shape_mismatch() -> None:
    """Reject shapes that would otherwise broadcast."""
    with pytest.raises(ValueError, match="same shape"):
        derive_quadrant_labels(torch.tensor([0, 1]), torch.tensor([[0, 1]]))


@pytest.mark.parametrize(
    ("arousal", "valence"),
    [
        (torch.tensor([2]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([-1])),
    ],
)
def test_derive_quadrant_labels_rejects_nonbinary_values(
    arousal: torch.Tensor,
    valence: torch.Tensor,
) -> None:
    """Reject labels outside 0, 1, and the configured ignore index."""
    with pytest.raises(ValueError, match="only 0, 1"):
        derive_quadrant_labels(arousal, valence)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_derive_quadrant_probabilities_values_order_sum_and_gradient(
    dtype: torch.dtype,
) -> None:
    """Verify formula, order, normalization, dtype, gradients, and immutability."""
    arousal = torch.tensor([[0.75, 0.25]], dtype=dtype, requires_grad=True)
    valence = torch.tensor([[0.4, 0.6]], dtype=dtype, requires_grad=True)
    original_arousal = arousal.detach().clone()
    original_valence = valence.detach().clone()

    quadrant = derive_quadrant_probabilities(arousal, valence)

    expected = torch.tensor([[0.30, 0.10, 0.45, 0.15]], dtype=dtype)
    torch.testing.assert_close(quadrant, expected)
    torch.testing.assert_close(quadrant.sum(dim=-1), torch.ones(1, dtype=dtype))
    assert quadrant.dtype == dtype

    weights = torch.tensor([[1.0, -1.0, 2.0, -2.0]], dtype=dtype)
    (quadrant * weights).sum().backward()
    assert arousal.grad is not None
    assert valence.grad is not None
    assert torch.isfinite(arousal.grad).all()
    assert torch.isfinite(valence.grad).all()
    torch.testing.assert_close(arousal.detach(), original_arousal)
    torch.testing.assert_close(valence.detach(), original_valence)


@pytest.mark.parametrize(
    ("arousal", "valence", "message"),
    [
        (
            torch.tensor([[0.4, 0.5]]),
            torch.tensor([[0.5, 0.5]]),
            "sum to 1",
        ),
        (
            torch.tensor([[-0.1, 1.1]]),
            torch.tensor([[0.5, 0.5]]),
            "range",
        ),
        (
            torch.tensor([[float("nan"), float("nan")]]),
            torch.tensor([[0.5, 0.5]]),
            "finite",
        ),
        (
            torch.tensor([[float("inf"), 0.0]]),
            torch.tensor([[0.5, 0.5]]),
            "finite",
        ),
    ],
)
def test_derive_quadrant_probabilities_rejects_invalid_values(
    arousal: torch.Tensor,
    valence: torch.Tensor,
    message: str,
) -> None:
    """Reject unnormalized, negative, NaN, and infinite probabilities."""
    with pytest.raises(ValueError, match=message):
        derive_quadrant_probabilities(arousal, valence)


@pytest.mark.parametrize(
    ("arousal", "valence"),
    [
        (torch.ones((1, 2)), torch.ones((2, 2))),
        (torch.ones((1, 3)), torch.ones((1, 3))),
    ],
)
def test_derive_quadrant_probabilities_rejects_invalid_shapes(
    arousal: torch.Tensor,
    valence: torch.Tensor,
) -> None:
    """Reject shape mismatch and a final dimension other than two."""
    with pytest.raises(ValueError):
        derive_quadrant_probabilities(arousal, valence)
