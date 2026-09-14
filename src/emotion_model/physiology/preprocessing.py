"""Mask-safe, channel-agnostic physiology preprocessing hooks."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from emotion_model.physiology.channel_metadata import PhysioChannelWindow


class PhysioFilter(Protocol):
    """Callable protocol for filtering one safe channel tensor.

    Implementations receive ``values: [T]``, ``valid_mask: [T]`` with
    ``True=valid``, and a positive native sample rate. They must return a
    floating tensor with the same shape, dtype, and device. This protocol does
    not prescribe channel-specific filter frequencies.
    """

    def __call__(
        self,
        values: Tensor,
        valid_mask: Tensor,
        sample_rate_hz: float,
    ) -> Tensor:
        """Filter safe values ``[T]`` and return a tensor with shape ``[T]``."""
        ...


def _validate_channel_values_and_mask(values: Tensor, valid_mask: Tensor) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("values must be a Tensor.")
    if values.ndim != 1:
        raise ValueError(
            f"values must have exact shape [T]; received {tuple(values.shape)}."
        )
    if not values.is_floating_point():
        raise TypeError(f"values must be floating point; received {values.dtype}.")
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a Tensor.")
    if valid_mask.ndim != 1:
        raise ValueError(
            "valid_mask must have exact shape [T]; "
            f"received {tuple(valid_mask.shape)}."
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError(
            "valid_mask must have dtype torch.bool with True meaning valid; "
            f"received {valid_mask.dtype}."
        )
    if tuple(valid_mask.shape) != tuple(values.shape):
        raise ValueError(
            "valid_mask shape must exactly match values; "
            f"received {tuple(valid_mask.shape)} and {tuple(values.shape)}."
        )
    if valid_mask.device != values.device:
        raise ValueError("values and valid_mask must be on the same device.")
    if not bool(torch.isfinite(values[valid_mask]).all()):
        raise ValueError("values at valid positions must contain only finite values.")


def _validate_artifact_mask(
    artifact_mask: Tensor,
    *,
    reference: Tensor,
    name: str,
) -> None:
    if not isinstance(artifact_mask, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if artifact_mask.dtype != torch.bool:
        raise TypeError(
            f"{name} must have dtype torch.bool with True meaning artifact."
        )
    if artifact_mask.ndim != 1 or tuple(artifact_mask.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} must have exact shape {tuple(reference.shape)}; "
            f"received {tuple(artifact_mask.shape)}."
        )
    if artifact_mask.device != reference.device:
        raise ValueError(f"{name} and the base mask must be on the same device.")


def zero_invalid_values(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Safely zero invalid values in one physiology channel.

    Args:
        values: Floating tensor with shape ``[T]``. Effective positions must be
            finite; invalid positions may contain arbitrary placeholders.
        valid_mask: Boolean tensor with exact shape ``[T]`` using
            ``True=valid`` semantics.

    Returns:
        New tensor with shape/dtype/device matching ``values``. Effective
        values are preserved and invalid values are exact zeros.

    Raises:
        TypeError: If values are not floating or the mask is not boolean.
        ValueError: If shapes/devices mismatch or a valid value is non-finite.

    Inputs are not modified.
    """
    _validate_channel_values_and_mask(values, valid_mask)
    return torch.where(valid_mask, values, torch.zeros_like(values))


def combine_invalid_masks(
    base_valid_mask: Tensor,
    *artifact_masks: Tensor,
) -> Tensor:
    """Combine valid and artifact masks without confusing their polarities.

    Args:
        base_valid_mask: Boolean tensor ``[T]`` where ``True`` means valid.
        artifact_masks: Zero or more boolean tensors ``[T]`` where ``True``
            means an artifact that must make the final position invalid.

    Returns:
        New boolean tensor ``[T]`` with project semantics ``True=valid``,
        equivalent to ``base_valid_mask & ~artifact_mask_1 & ...``.

    Raises:
        TypeError: If any mask is not boolean.
        ValueError: If masks are not one-dimensional or differ in shape/device.

    Inputs are not modified.
    """
    if not isinstance(base_valid_mask, Tensor):
        raise TypeError("base_valid_mask must be a Tensor.")
    if base_valid_mask.dtype != torch.bool:
        raise TypeError("base_valid_mask must have dtype torch.bool.")
    if base_valid_mask.ndim != 1:
        raise ValueError(
            "base_valid_mask must have exact shape [T]; "
            f"received {tuple(base_valid_mask.shape)}."
        )
    combined = base_valid_mask.clone()
    for index, artifact_mask in enumerate(artifact_masks):
        _validate_artifact_mask(
            artifact_mask,
            reference=base_valid_mask,
            name=f"artifact_masks[{index}]",
        )
        combined = combined & ~artifact_mask
    return combined


def apply_channel_filter(
    window: PhysioChannelWindow,
    filter_fn: PhysioFilter,
) -> PhysioChannelWindow:
    """Apply a caller-supplied channel filter through a safe wrapper.

    Args:
        window: Source channel window with ``values/mask/timestamps: [T]``.
        filter_fn: Callable receiving safe-zeroed ``values [T]``, the unchanged
            boolean mask ``[T]``, and the native sample rate.

    Returns:
        New :class:`PhysioChannelWindow` with filtered values ``[T]``. Spec,
        timestamps, and mask objects are preserved; invalid output positions
        are reset to zero.

    Raises:
        TypeError: If the window/filter or filter output type is invalid.
        ValueError: If output shape/dtype/device changes or effective output
            contains NaN or Inf.

    No concrete ECG/BVP/EDA/temperature filter is implemented here. The caller's
    original values are never passed to the filter and are not modified.
    """
    if not isinstance(window, PhysioChannelWindow):
        raise TypeError("window must be a PhysioChannelWindow.")
    if not callable(filter_fn):
        raise TypeError("filter_fn must be callable.")
    safe_input = window.safe_values()
    filter_mask = window.valid_mask.clone()
    filtered = filter_fn(
        safe_input,
        filter_mask,
        window.spec.native_sample_rate_hz,
    )
    if not isinstance(filtered, Tensor):
        raise TypeError("filter_fn must return a Tensor.")
    if tuple(filtered.shape) != tuple(window.values.shape):
        raise ValueError(
            "filter_fn output shape must match input exactly; "
            f"expected {tuple(window.values.shape)}, received {tuple(filtered.shape)}."
        )
    if filtered.dtype != window.values.dtype:
        raise TypeError(
            "filter_fn output dtype must match input exactly; "
            f"expected {window.values.dtype}, received {filtered.dtype}."
        )
    if filtered.device != window.values.device:
        raise ValueError("filter_fn output device must match input.")
    safe_filtered = zero_invalid_values(filtered, window.valid_mask)
    return PhysioChannelWindow(
        spec=window.spec,
        values=safe_filtered,
        valid_mask=window.valid_mask,
        timestamps_seconds=window.timestamps_seconds,
    )


def detect_flatline_mask(
    values: Tensor,
    valid_mask: Tensor,
    *,
    atol: float,
    min_run_length: int,
) -> Tensor:
    """Detect generic approximately constant runs in a channel.

    Args:
        values: Floating tensor ``[T]`` with finite effective values.
        valid_mask: Boolean tensor ``[T]`` with ``True=valid``.
        atol: Finite non-negative maximum adjacent absolute change within a run.
        min_run_length: Minimum number of consecutive valid samples in a flat
            run; must be an integer of at least two.

    Returns:
        Boolean artifact mask ``[T]`` where ``True`` marks samples belonging to
        a qualifying flat run. Invalid positions are always ``False`` and split
        runs, even when their placeholder values match.

    Raises:
        TypeError: If parameter or tensor dtypes are invalid.
        ValueError: If shapes/devices, finite effective values, or parameter
            ranges are invalid.

    This is a generic statistical detector, not a medical-grade signal-quality
    algorithm. Inputs are not modified.
    """
    _validate_channel_values_and_mask(values, valid_mask)
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise TypeError("atol must be a real number.")
    atol_value = float(atol)
    if not math.isfinite(atol_value) or atol_value < 0.0:
        raise ValueError("atol must be finite and >= 0.")
    if isinstance(min_run_length, bool) or not isinstance(min_run_length, int):
        raise TypeError("min_run_length must be an integer.")
    if min_run_length < 2:
        raise ValueError("min_run_length must be >= 2.")

    result = torch.zeros_like(valid_mask)
    length = values.shape[0]
    index = 0
    while index < length:
        if not bool(valid_mask[index]):
            index += 1
            continue
        run_start = index
        index += 1
        while (
            index < length
            and bool(valid_mask[index])
            and bool(torch.abs(values[index] - values[index - 1]) <= atol_value)
        ):
            index += 1
        if index - run_start >= min_run_length:
            result[run_start:index] = True
    return result


def detect_robust_outlier_mask(
    values: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float = 6.0,
    mad_epsilon: float = 1.0e-6,
) -> Tensor:
    """Detect generic robust outliers using the median absolute deviation.

    Args:
        values: Floating tensor ``[T]`` with finite effective values.
        valid_mask: Boolean tensor ``[T]`` with ``True=valid``.
        threshold: Finite positive robust-z threshold.
        mad_epsilon: Finite positive lower bound for MAD.

    Returns:
        Boolean artifact mask ``[T]`` where ``True`` means robust outlier.
        Invalid positions are ``False``. Fully invalid, single-valid-value, and
        constant valid signals return all ``False``; when MAD is zero but an
        effective value differs from the median, epsilon keeps the score finite.

    Raises:
        TypeError: If parameters or tensor dtypes are invalid.
        ValueError: If shapes/devices, effective finiteness, or parameter ranges
            are invalid.

    Effective NaN/Inf values are rejected rather than silently repaired.
    Inputs are not modified.
    """
    _validate_channel_values_and_mask(values, valid_mask)
    for name, parameter in (
        ("threshold", threshold),
        ("mad_epsilon", mad_epsilon),
    ):
        if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        numeric = float(parameter)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
    result = torch.zeros_like(valid_mask)
    effective_values = values[valid_mask]
    if effective_values.numel() == 0:
        return result
    statistics_values = effective_values.to(dtype=torch.float64)
    median = torch.quantile(statistics_values, 0.5)
    absolute_deviation = torch.abs(statistics_values - median)
    mad = torch.quantile(absolute_deviation, 0.5)
    denominator = torch.clamp(
        mad,
        min=torch.as_tensor(
            float(mad_epsilon),
            dtype=statistics_values.dtype,
            device=values.device,
        ),
    )
    robust_z = 0.67448975 * absolute_deviation / denominator
    result[valid_mask] = robust_z > float(threshold)
    return result


def _validate_target_timestamps(
    target_timestamps_seconds: Tensor,
    *,
    source_device: torch.device,
) -> None:
    if not isinstance(target_timestamps_seconds, Tensor):
        raise TypeError("target_timestamps_seconds must be a Tensor.")
    if target_timestamps_seconds.ndim != 1:
        raise ValueError(
            "target_timestamps_seconds must have exact shape [T_target]; "
            f"received {tuple(target_timestamps_seconds.shape)}."
        )
    if target_timestamps_seconds.numel() == 0:
        raise ValueError("target_timestamps_seconds must be non-empty.")
    if not target_timestamps_seconds.is_floating_point():
        raise TypeError("target_timestamps_seconds must be floating point.")
    if target_timestamps_seconds.device != source_device:
        raise ValueError("target and source timestamps must be on the same device.")
    if not bool(torch.isfinite(target_timestamps_seconds).all()):
        raise ValueError("target_timestamps_seconds must contain only finite values.")
    if target_timestamps_seconds.numel() > 1 and not bool(
        (target_timestamps_seconds[1:] > target_timestamps_seconds[:-1]).all()
    ):
        raise ValueError("target_timestamps_seconds must be strictly increasing.")


def resample_channel_to_timeline(
    window: PhysioChannelWindow,
    target_timestamps_seconds: Tensor,
    *,
    max_interpolation_gap_seconds: float | None = None,
) -> PhysioChannelWindow:
    """Linearly project a channel onto an explicit target timeline.

    Args:
        window: Source window with values/mask/timestamps ``[T_source]``.
        target_timestamps_seconds: Finite strictly increasing floating tensor
            ``[T_target]`` on the source device.
        max_interpolation_gap_seconds: Optional finite positive upper bound on
            the time difference between interpolation neighbors.

    Returns:
        New window with source spec, exact target timestamp object, values
        ``[T_target]`` in the source value dtype, and a boolean valid mask
        ``[T_target]``. Invalid values are exact zeros.

    Raises:
        TypeError: If inputs or gap type are invalid.
        ValueError: If target timestamps are empty/non-finite/not increasing,
            devices differ, or the optional gap is non-positive/non-finite.

    Exact valid source timestamps are copied. Interior points are interpolated
    only when both adjacent source points are valid and their gap is allowed.
    The function never extrapolates, bridges an invalid source point, assumes a
    fixed sample-rate ratio, or modifies source/target tensors.
    """
    if not isinstance(window, PhysioChannelWindow):
        raise TypeError("window must be a PhysioChannelWindow.")
    _validate_target_timestamps(
        target_timestamps_seconds,
        source_device=window.timestamps_seconds.device,
    )
    if max_interpolation_gap_seconds is not None:
        if isinstance(max_interpolation_gap_seconds, bool) or not isinstance(
            max_interpolation_gap_seconds,
            (int, float),
        ):
            raise TypeError("max_interpolation_gap_seconds must be a real number or None.")
        gap_limit = float(max_interpolation_gap_seconds)
        if not math.isfinite(gap_limit) or gap_limit <= 0.0:
            raise ValueError(
                "max_interpolation_gap_seconds must be finite and > 0."
            )
    else:
        gap_limit = None

    source_times = window.timestamps_seconds
    safe_source_values = window.safe_values()
    target_length = target_timestamps_seconds.shape[0]
    output_values = torch.zeros(
        target_length,
        dtype=window.values.dtype,
        device=window.values.device,
    )
    output_valid = torch.zeros(
        target_length,
        dtype=torch.bool,
        device=window.values.device,
    )
    insertion_indices = torch.searchsorted(source_times, target_timestamps_seconds)
    source_length = source_times.shape[0]
    for target_index in range(target_length):
        insertion_index = int(insertion_indices[target_index].item())
        target_time = target_timestamps_seconds[target_index]
        if (
            insertion_index < source_length
            and bool(source_times[insertion_index] == target_time)
        ):
            if bool(window.valid_mask[insertion_index]):
                output_values[target_index] = safe_source_values[insertion_index]
                output_valid[target_index] = True
            continue
        if insertion_index == 0 or insertion_index == source_length:
            continue
        left_index = insertion_index - 1
        right_index = insertion_index
        if not (
            bool(window.valid_mask[left_index])
            and bool(window.valid_mask[right_index])
        ):
            continue
        source_gap = source_times[right_index] - source_times[left_index]
        if gap_limit is not None and bool(source_gap > gap_limit):
            continue
        fraction = (target_time - source_times[left_index]) / source_gap
        output_values[target_index] = (
            safe_source_values[left_index]
            + fraction.to(dtype=window.values.dtype)
            * (safe_source_values[right_index] - safe_source_values[left_index])
        )
        output_valid[target_index] = True

    return PhysioChannelWindow(
        spec=window.spec,
        values=output_values,
        valid_mask=output_valid,
        timestamps_seconds=target_timestamps_seconds,
    )
