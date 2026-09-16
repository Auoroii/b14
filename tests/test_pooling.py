"""Tests for common masked pooling operations."""

import pytest
import torch

from emotion_model.common import (
    MaskAwareAttentiveStatisticsPooling,
    masked_mean,
    masked_mean_std,
    masked_population_std,
    masked_population_variance,
)


def test_attentive_statistics_shapes_masks_and_fully_padded_rows() -> None:
    """Normalize only valid frames and zero every fully padded output."""

    torch.manual_seed(4)
    pooling = MaskAwareAttentiveStatisticsPooling(3, 5)
    features = torch.randn(3, 4, 3)
    valid_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, True, False, True],
            [False, False, False, False],
        ]
    )

    statistics, weights, sample_valid = pooling(features, valid_mask)

    assert statistics.shape == (3, 6)
    assert weights.shape == (3, 4)
    assert sample_valid.tolist() == [True, True, False]
    torch.testing.assert_close(weights[:2].sum(dim=1), torch.ones(2))
    assert torch.equal(weights.masked_select(~valid_mask), torch.zeros(8))
    assert torch.equal(weights[2], torch.zeros(4))
    assert torch.equal(statistics[2], torch.zeros(6))
    assert bool(torch.isfinite(statistics).all())
    assert bool(torch.isfinite(weights).all())


def test_attentive_statistics_match_weighted_definition() -> None:
    """Return concatenated weighted population mean and stabilized std."""

    pooling = MaskAwareAttentiveStatisticsPooling(2, 3, epsilon=1.0e-5)
    features = torch.tensor([[[1.0, 2.0], [3.0, 8.0], [20.0, 30.0]]])
    valid_mask = torch.tensor([[True, True, False]])

    statistics, weights, _ = pooling(features, valid_mask)
    safe = torch.where(valid_mask.unsqueeze(-1), features, torch.zeros_like(features))
    expected_mean = (weights.unsqueeze(-1) * safe).sum(dim=1)
    expected_variance = (
        weights.unsqueeze(-1) * (safe - expected_mean.unsqueeze(1)).square()
    ).sum(dim=1)
    expected = torch.cat(
        (expected_mean, torch.sqrt(expected_variance.clamp_min(1.0e-5))),
        dim=-1,
    )

    torch.testing.assert_close(statistics, expected)


def test_attentive_statistics_single_frame_and_padding_invariance() -> None:
    """Handle one valid frame and ignore arbitrary non-finite padding values."""

    pooling = MaskAwareAttentiveStatisticsPooling(2, 4)
    baseline = torch.tensor([[[2.0, -4.0], [0.0, 0.0], [0.0, 0.0]]])
    changed = baseline.clone()
    changed[0, 1] = torch.tensor([float("nan"), float("inf")])
    changed[0, 2] = torch.tensor([1.0e20, -1.0e20])
    changed.requires_grad_()
    valid_mask = torch.tensor([[True, False, False]])

    baseline_output = pooling(baseline, valid_mask)
    changed_output = pooling(changed, valid_mask)

    for baseline_value, changed_value in zip(baseline_output[:2], changed_output[:2]):
        torch.testing.assert_close(changed_value, baseline_value)
    torch.testing.assert_close(baseline_output[1], torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        baseline_output[0],
        torch.tensor([[2.0, -4.0, 1.0e-5**0.5, 1.0e-5**0.5]]),
    )
    changed_output[0].sum().backward()
    assert changed.grad is not None
    assert bool(torch.isfinite(changed.grad).all())
    assert torch.equal(changed.grad[:, 1:], torch.zeros(1, 2, 2))


def test_attentive_statistics_backpropagates_finite_score_gradients() -> None:
    """Train every score-network parameter through weighted statistics."""

    torch.manual_seed(8)
    pooling = MaskAwareAttentiveStatisticsPooling(4, 6)
    features = torch.randn(2, 5, 4, requires_grad=True)
    valid_mask = torch.tensor(
        [[True, True, True, False, False], [False, True, True, True, False]]
    )

    statistics, weights, _ = pooling(features, valid_mask)
    (statistics.square().sum() + weights.square().sum()).backward()

    gradients = [parameter.grad for parameter in pooling.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def test_attentive_statistics_validates_dimensions_and_parameter_count() -> None:
    """Reject malformed construction/input and expose the exact lightweight cost."""

    pooling = MaskAwareAttentiveStatisticsPooling(8, 5)
    assert sum(parameter.numel() for parameter in pooling.parameters()) == 67
    with pytest.raises(ValueError, match="width"):
        pooling(torch.ones(2, 3, 7), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="positive"):
        MaskAwareAttentiveStatisticsPooling(8, 0)


def test_masked_mean_uses_only_valid_positions() -> None:
    """Average each feature over valid time positions only."""
    sequence = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [100.0, -100.0]]])
    valid_mask = torch.tensor([[True, True, False]])

    mean, sample_valid = masked_mean(sequence, valid_mask)

    torch.testing.assert_close(mean, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(sample_valid, torch.tensor([True]))


def test_masked_population_variance_and_std_are_correct() -> None:
    """Use the population denominator for variance and standard deviation."""
    sequence = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [50.0, 50.0]]])
    valid_mask = torch.tensor([[True, True, False]])

    variance, variance_valid = masked_population_variance(sequence, valid_mask)
    std, std_valid = masked_population_std(sequence, valid_mask)

    torch.testing.assert_close(variance, torch.tensor([[1.0, 4.0]]))
    torch.testing.assert_close(std, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(variance_valid, torch.tensor([True]))
    assert torch.equal(std_valid, variance_valid)


def test_single_valid_position_has_zero_population_std() -> None:
    """Return exact zero variance and std for one valid time position."""
    sequence = torch.tensor([[[4.0, -2.0], [100.0, 100.0], [-100.0, -100.0]]])
    valid_mask = torch.tensor([[True, False, False]])

    variance, sample_valid = masked_population_variance(sequence, valid_mask)
    std, _ = masked_population_std(sequence, valid_mask)

    assert torch.equal(variance, torch.zeros((1, 2)))
    assert torch.equal(std, torch.zeros((1, 2)))
    assert torch.equal(sample_valid, torch.tensor([True]))


def test_fully_padded_samples_return_zero_and_invalid_state() -> None:
    """Return finite zero statistics without marking full padding as valid."""
    sequence = torch.tensor(
        [[[float("nan"), float("inf")], [float("-inf"), 1.0]]],
        dtype=torch.float64,
    )
    valid_mask = torch.tensor([[False, False]])

    mean, mean_valid = masked_mean(sequence, valid_mask)
    variance, variance_valid = masked_population_variance(sequence, valid_mask)
    std, std_valid = masked_population_std(sequence, valid_mask)
    statistics, statistics_valid = masked_mean_std(sequence, valid_mask)

    for output in (mean, variance, std, statistics):
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output).item() == 0
    for state in (mean_valid, variance_valid, std_valid, statistics_valid):
        assert torch.equal(state, torch.tensor([False]))
    assert statistics.shape == (1, 4)


def test_padding_values_do_not_change_masked_statistics() -> None:
    """Ignore extreme and non-finite values in padded positions."""
    baseline = torch.tensor([[[1.0, 5.0], [3.0, 9.0], [0.0, 0.0]]])
    changed = baseline.clone()
    changed[:, 2] = torch.tensor([1.0e30, -1.0e30])
    nonfinite = baseline.clone()
    nonfinite[:, 2] = torch.tensor([float("nan"), float("inf")])
    valid_mask = torch.tensor([[True, True, False]])
    original_nonfinite = nonfinite.clone()
    original_mask = valid_mask.clone()

    baseline_statistics, _ = masked_mean_std(baseline, valid_mask)
    changed_statistics, _ = masked_mean_std(changed, valid_mask)
    nonfinite_statistics, _ = masked_mean_std(nonfinite, valid_mask)

    torch.testing.assert_close(changed_statistics, baseline_statistics)
    torch.testing.assert_close(nonfinite_statistics, baseline_statistics)
    torch.testing.assert_close(nonfinite, original_nonfinite, equal_nan=True)
    assert torch.equal(valid_mask, original_mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_pooling_preserves_supported_dtype(dtype: torch.dtype) -> None:
    """Preserve float32 and float64 through all pooled statistics."""
    sequence = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=dtype)
    valid_mask = torch.tensor([[True, False, True]])

    mean, _ = masked_mean(sequence, valid_mask)
    variance, _ = masked_population_variance(sequence, valid_mask)
    std, _ = masked_population_std(sequence, valid_mask)
    statistics, _ = masked_mean_std(sequence, valid_mask)

    for output in (mean, variance, std, statistics):
        assert output.dtype == dtype
    torch.testing.assert_close(mean, torch.tensor([[2.5]], dtype=dtype))
    torch.testing.assert_close(variance, torch.tensor([[2.25]], dtype=dtype))
    torch.testing.assert_close(std, torch.tensor([[1.5]], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_mean_and_std_propagate_finite_gradients(dtype: torch.dtype) -> None:
    """Backpropagate in both supported dtypes only through valid positions."""
    sequence = torch.tensor(
        [[[1.0, 2.0], [1000.0, -1000.0], [4.0, 8.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    valid_mask = torch.tensor([[True, False, True]])

    mean, _ = masked_mean(sequence, valid_mask)
    std, _ = masked_population_std(sequence, valid_mask)
    (mean.sum() + std.sum()).backward()

    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()
    assert torch.equal(sequence.grad[:, 1], torch.zeros((1, 2), dtype=dtype))
    assert sequence.grad[:, [0, 2]].abs().sum().item() > 0.0


def test_single_point_std_has_finite_zero_gradient() -> None:
    """Avoid the undefined sqrt-at-zero gradient for a one-point std."""
    sequence = torch.tensor([[[2.0], [9.0]]], requires_grad=True)
    valid_mask = torch.tensor([[True, False]])

    std, _ = masked_population_std(sequence, valid_mask)
    std.sum().backward()

    assert torch.equal(std, torch.zeros((1, 1)))
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()
    assert torch.equal(sequence.grad, torch.zeros_like(sequence))


def test_masked_pooling_rejects_non_bool_and_mismatched_masks() -> None:
    """Reject invalid mask dtype and shape rather than broadcasting."""
    sequence = torch.ones((2, 3, 4))

    with pytest.raises(TypeError, match="torch.bool"):
        masked_mean(sequence, torch.ones((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        masked_mean(sequence, torch.ones((2, 1), dtype=torch.bool))
