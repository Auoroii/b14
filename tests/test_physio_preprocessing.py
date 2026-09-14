"""Tests for channel-agnostic mask-safe physiology preprocessing."""

import pytest
import torch
from torch import Tensor

from emotion_model.physiology import (
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
    apply_channel_filter,
    combine_invalid_masks,
    detect_flatline_mask,
    detect_robust_outlier_mask,
    resample_channel_to_timeline,
    zero_invalid_values,
)


def _window(
    values: Tensor,
    mask: Tensor | None = None,
    timestamps: Tensor | None = None,
    *,
    name: str = "eda",
) -> PhysioChannelWindow:
    """Create a small explicitly timed physiology channel."""
    length = values.shape[0]
    if mask is None:
        mask = torch.ones(length, dtype=torch.bool)
    if timestamps is None:
        timestamps = torch.arange(length, dtype=values.dtype)
    return PhysioChannelWindow(
        PhysioChannelSpec(name, PhysioSignalKind.EDA, 4.0, "uS"),
        values,
        mask,
        timestamps,
    )


def test_zero_invalid_values_preserves_effective_data_without_mutation() -> None:
    """Replace padded NaN/Inf with zero and preserve both inputs."""
    values = torch.tensor([1.0, float("nan"), 3.0, float("inf")])
    mask = torch.tensor([True, False, True, False])
    original = values.clone()

    safe = zero_invalid_values(values, mask)

    assert torch.equal(safe, torch.tensor([1.0, 0.0, 3.0, 0.0]))
    torch.testing.assert_close(values, original, equal_nan=True)
    assert torch.equal(mask, torch.tensor([True, False, True, False]))


@pytest.mark.parametrize(
    ("values", "mask", "exception"),
    [
        (torch.ones(3), torch.ones(3), TypeError),
        (torch.ones((1, 3)), torch.ones(3, dtype=torch.bool), ValueError),
        (torch.ones(3), torch.ones(2, dtype=torch.bool), ValueError),
        (torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.bool), TypeError),
    ],
)
def test_zero_invalid_values_rejects_bad_contracts(
    values: Tensor,
    mask: Tensor,
    exception: type[Exception],
) -> None:
    """Reject float masks, non-vectors, shape mismatch, and integer values."""
    with pytest.raises(exception):
        zero_invalid_values(values, mask)


def test_combine_invalid_masks_has_explicit_artifact_polarity() -> None:
    """Compute base-valid AND NOT artifact for every supplied artifact mask."""
    base = torch.tensor([True, True, False, True])
    flatline = torch.tensor([False, True, False, False])
    outlier = torch.tensor([False, False, True, True])

    combined = combine_invalid_masks(base, flatline, outlier)

    assert torch.equal(combined, torch.tensor([True, False, False, False]))
    assert torch.equal(combine_invalid_masks(base), base)
    assert combine_invalid_masks(base).data_ptr() != base.data_ptr()
    with pytest.raises(TypeError, match="torch.bool"):
        combine_invalid_masks(base, torch.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        combine_invalid_masks(base, torch.zeros(3, dtype=torch.bool))


def test_filter_receives_safe_input_and_preserves_window_contract() -> None:
    """Pass no padding NaN to a filter and re-zero its invalid output."""
    values = torch.tensor([1.0, float("nan"), 3.0])
    mask = torch.tensor([True, False, True])
    window = _window(values, mask)
    captured: list[tuple[Tensor, Tensor, float]] = []

    def filter_fn(safe_values: Tensor, valid_mask: Tensor, rate: float) -> Tensor:
        captured.append((safe_values.clone(), valid_mask, rate))
        return safe_values + 2.0

    output = apply_channel_filter(window, filter_fn)

    assert torch.equal(captured[0][0], torch.tensor([1.0, 0.0, 3.0]))
    assert torch.equal(captured[0][1], window.valid_mask)
    assert captured[0][1].data_ptr() != window.valid_mask.data_ptr()
    assert captured[0][2] == 4.0
    assert torch.equal(output.values, torch.tensor([3.0, 0.0, 5.0]))
    assert output.spec is window.spec
    assert output.valid_mask is window.valid_mask
    assert output.timestamps_seconds is window.timestamps_seconds
    torch.testing.assert_close(values, torch.tensor([1.0, float("nan"), 3.0]), equal_nan=True)


@pytest.mark.parametrize("failure", ["shape", "dtype", "nonfinite", "type"])
def test_filter_rejects_invalid_outputs(failure: str) -> None:
    """Reject filter outputs that violate shape, dtype, type, or finite data."""
    window = _window(torch.tensor([1.0, 2.0, 3.0]))

    def filter_fn(values: Tensor, mask: Tensor, rate: float) -> object:
        del mask, rate
        if failure == "shape":
            return values[:2]
        if failure == "dtype":
            return values.to(torch.float64)
        if failure == "nonfinite":
            result = values.clone()
            result[0] = float("nan")
            return result
        return [1.0, 2.0, 3.0]

    expected = TypeError if failure in {"dtype", "type"} else ValueError
    with pytest.raises(expected):
        apply_channel_filter(window, filter_fn)  # type: ignore[arg-type]


def test_filter_cannot_modify_caller_inputs_even_if_it_mutates_them() -> None:
    """Isolate in-place filter mutations from caller values and valid mask."""
    values = torch.tensor([1.0, float("nan"), 3.0])
    window = _window(values, torch.tensor([True, False, True]))
    original = values.clone()
    original_mask = window.valid_mask.clone()

    def mutating_filter(safe: Tensor, mask: Tensor, rate: float) -> Tensor:
        del rate
        safe.add_(1.0)
        mask.fill_(False)
        return safe

    output = apply_channel_filter(window, mutating_filter)

    torch.testing.assert_close(values, original, equal_nan=True)
    assert torch.equal(window.valid_mask, original_mask)
    assert torch.equal(output.values, torch.tensor([2.0, 0.0, 4.0]))


def test_flatline_detects_runs_and_invalid_points_split_them() -> None:
    """Mark qualifying adjacent runs without bridging a missing point."""
    values = torch.tensor([1.0, 1.01, 1.02, float("nan"), 1.0, 1.0, 2.0])
    mask = torch.tensor([True, True, True, False, True, True, True])

    artifact = detect_flatline_mask(
        values,
        mask,
        atol=0.02,
        min_run_length=3,
    )

    assert torch.equal(
        artifact,
        torch.tensor([True, True, True, False, False, False, False]),
    )


def test_flatline_constant_signal_and_short_run_boundaries() -> None:
    """Mark a constant effective channel but not a run below the minimum."""
    constant = torch.ones(4)
    mask = torch.ones(4, dtype=torch.bool)
    assert torch.equal(
        detect_flatline_mask(constant, mask, atol=0.0, min_run_length=4),
        mask,
    )
    assert not bool(
        detect_flatline_mask(constant[:3], mask[:3], atol=0.0, min_run_length=4).any()
    )


@pytest.mark.parametrize(
    ("values", "minimum", "expected"),
    [
        ([1.0, 1.0, 2.0], 2, [True, True, False]),
        ([1.0, 1.0, 1.0, 2.0], 3, [True, True, True, False]),
        ([1.0, 1.0, 1.0, 2.0], 4, [False, False, False, False]),
    ],
)
def test_flatline_minimum_run_length_has_no_off_by_one(
    values: list[float],
    minimum: int,
    expected: list[bool],
) -> None:
    """Compare exact boundary runs against manually enumerated artifact masks."""
    tensor = torch.tensor(values)
    artifact = detect_flatline_mask(
        tensor,
        torch.ones(len(values), dtype=torch.bool),
        atol=0.0,
        min_run_length=minimum,
    )
    assert torch.equal(artifact, torch.tensor(expected))


@pytest.mark.parametrize(
    ("atol", "run_length", "exception"),
    [
        (-1.0, 3, ValueError),
        (float("inf"), 3, ValueError),
        (0.0, 1, ValueError),
        (0.0, True, TypeError),
    ],
)
def test_flatline_rejects_invalid_parameters(
    atol: float,
    run_length: int,
    exception: type[Exception],
) -> None:
    """Reject invalid tolerances and minimum run lengths."""
    with pytest.raises(exception):
        detect_flatline_mask(
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            atol=atol,
            min_run_length=run_length,
        )


def test_robust_outlier_detects_spike_without_marking_constant_values() -> None:
    """Use median/MAD and epsilon to isolate a clear spike."""
    values = torch.tensor([1.0, 1.0, 1.0, 10.0, 1.0])
    mask = torch.ones(5, dtype=torch.bool)

    artifact = detect_robust_outlier_mask(values, mask, threshold=6.0)

    assert torch.equal(artifact, torch.tensor([False, False, False, True, False]))
    assert not bool(detect_robust_outlier_mask(torch.ones(5), mask).any())


def test_robust_outlier_uses_midpoint_median_for_even_samples() -> None:
    """Avoid lower-median bias that would mark half an even sample as outliers."""
    values = torch.tensor([0.0, 0.0, 10.0, 10.0])
    artifact = detect_robust_outlier_mask(
        values,
        torch.ones(4, dtype=torch.bool),
    )
    assert not bool(artifact.any())


def test_robust_outlier_all_invalid_single_value_and_padding_nan() -> None:
    """Keep degenerate cases finite and never mark invalid placeholders."""
    all_invalid_values = torch.tensor([float("nan"), float("inf")])
    all_invalid_mask = torch.zeros(2, dtype=torch.bool)
    assert not bool(
        detect_robust_outlier_mask(all_invalid_values, all_invalid_mask).any()
    )
    assert not bool(
        detect_robust_outlier_mask(
            torch.tensor([2.0]),
            torch.tensor([True]),
        ).any()
    )
    mixed = detect_robust_outlier_mask(
        torch.tensor([1.0, float("nan"), 1.0]),
        torch.tensor([True, False, True]),
    )
    assert not bool(mixed.any())


@pytest.mark.parametrize(
    ("threshold", "epsilon", "exception"),
    [
        (0.0, 1.0e-6, ValueError),
        (float("nan"), 1.0e-6, ValueError),
        (6.0, 0.0, ValueError),
        (True, 1.0e-6, TypeError),
    ],
)
def test_robust_outlier_rejects_invalid_parameters(
    threshold: float,
    epsilon: float,
    exception: type[Exception],
) -> None:
    """Reject non-positive, non-finite, and boolean detector parameters."""
    with pytest.raises(exception):
        detect_robust_outlier_mask(
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            threshold=threshold,
            mad_epsilon=epsilon,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_resample_exact_linear_outside_and_dtype(dtype: torch.dtype) -> None:
    """Copy exact points, linearly interpolate interiors, and avoid extrapolation."""
    source = _window(
        torch.tensor([0.0, 10.0, 30.0], dtype=dtype),
        timestamps=torch.tensor([0.0, 1.0, 3.0], dtype=dtype),
    )
    target = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
    original_target = target.clone()

    output = resample_channel_to_timeline(source, target)

    assert output.values.dtype == dtype
    assert output.timestamps_seconds is target
    assert torch.equal(
        output.valid_mask,
        torch.tensor([False, True, True, True, True, True, False]),
    )
    torch.testing.assert_close(
        output.values,
        torch.tensor([0.0, 0.0, 5.0, 10.0, 20.0, 30.0, 0.0], dtype=dtype),
    )
    assert torch.equal(target, original_target)


def test_resample_does_not_bridge_invalid_source_or_excessive_gap() -> None:
    """Invalidate interpolation when a neighbor is missing or gap exceeds limit."""
    source = _window(
        torch.tensor([0.0, float("nan"), 20.0, 40.0]),
        torch.tensor([True, False, True, True]),
        torch.tensor([0.0, 1.0, 2.0, 5.0]),
    )
    target = torch.tensor([0.5, 1.0, 1.5, 3.0, 5.0])

    output = resample_channel_to_timeline(
        source,
        target,
        max_interpolation_gap_seconds=2.0,
    )

    assert torch.equal(
        output.valid_mask,
        torch.tensor([False, False, False, False, True]),
    )
    assert torch.equal(output.values, torch.tensor([0.0, 0.0, 0.0, 0.0, 40.0]))


def test_resample_explicit_three_point_missing_case_never_bridges_gap() -> None:
    """Keep 0.5, exact missing 1.0, and 1.5 invalid around one missing source."""
    source = _window(
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([True, False, True]),
        torch.tensor([0.0, 1.0, 2.0]),
    )
    target = torch.tensor([0.5, 1.0, 1.5])

    output = resample_channel_to_timeline(source, target)

    assert torch.equal(output.valid_mask, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(output.values, torch.zeros(3))
@pytest.mark.parametrize(
    ("target", "gap", "exception"),
    [
        (torch.tensor([0.0, 0.0]), None, ValueError),
        (torch.tensor([1.0, 0.0]), None, ValueError),
        (torch.tensor([0.0, float("nan")]), None, ValueError),
        (torch.tensor([0, 1]), None, TypeError),
        (torch.tensor([]), None, ValueError),
        (torch.tensor([0.0, 1.0]), 0.0, ValueError),
        (torch.tensor([0.0, 1.0]), float("inf"), ValueError),
    ],
)
def test_resample_rejects_invalid_target_and_gap(
    target: Tensor,
    gap: float | None,
    exception: type[Exception],
) -> None:
    """Reject invalid target timelines and interpolation gap constraints."""
    with pytest.raises(exception):
        resample_channel_to_timeline(
            _window(torch.tensor([0.0, 1.0])),
            target,
            max_interpolation_gap_seconds=gap,
        )
