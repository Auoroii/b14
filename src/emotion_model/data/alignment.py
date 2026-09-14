"""Half-open time alignment helpers that never read or resample signals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from emotion_model.data.manifest import (
    IndexSpan,
    MultimodalWindowRecord,
    TimedSourceRef,
    TimeInterval,
)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def regular_sample_span(
    interval: TimeInterval,
    *,
    timeline_start_seconds: float,
    sample_rate_hz: float,
    num_samples: int,
    tolerance: float = 1.0e-9,
) -> IndexSpan:
    """Map a half-open time interval to regularly sampled indices.

    Args:
        interval: Logical ``[start_seconds, end_seconds)`` interval.
        timeline_start_seconds: Finite timestamp of sample index zero.
        sample_rate_hz: Finite positive sample rate.
        num_samples: Non-negative number of source samples; booleans fail.
        tolerance: Finite non-negative tolerance in sample-index units, used
            only for floating-point boundary error.

    Returns:
        Half-open :class:`IndexSpan` containing exactly sample times ``t`` for
        which ``interval.start_seconds <= t < interval.end_seconds``. A legal
        interval may map to an empty span.

    Raises:
        TypeError: If argument types violate the contract.
        ValueError: If numeric parameters are invalid or ``interval`` extends
            beyond ``[origin, origin + num_samples / sample_rate_hz)`` by more
            than ``tolerance``.

    The indices use
    ``ceil((boundary - origin) * sample_rate_hz - tolerance)``. The function
    neither reads samples nor performs resampling or general-purpose clamping.
    """
    if not isinstance(interval, TimeInterval):
        raise TypeError("interval must be a TimeInterval.")
    origin = _finite_real(timeline_start_seconds, name="timeline_start_seconds")
    rate = _finite_real(sample_rate_hz, name="sample_rate_hz")
    if rate <= 0.0:
        raise ValueError("sample_rate_hz must be > 0.")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer, not bool.")
    if num_samples < 0:
        raise ValueError("num_samples must be >= 0.")
    tolerance_value = _finite_real(tolerance, name="tolerance")
    if tolerance_value < 0.0:
        raise ValueError("tolerance must be >= 0.")
    if num_samples == 0:
        raise ValueError(
            "a non-empty interval cannot lie within a regular source with "
            "num_samples=0."
        )

    start_position = (interval.start_seconds - origin) * rate
    end_position = (interval.end_seconds - origin) * rate
    if start_position < -tolerance_value or end_position > num_samples + tolerance_value:
        source_end = origin + num_samples / rate
        raise ValueError(
            "interval must lie fully within the regular source time range "
            f"[{origin}, {source_end}); received "
            f"[{interval.start_seconds}, {interval.end_seconds})."
        )

    start_index = math.ceil(start_position - tolerance_value)
    end_index = math.ceil(end_position - tolerance_value)
    if not (0 <= start_index <= end_index <= num_samples):
        raise ValueError(
            "floating-point tolerance could not resolve interval to valid source indices; "
            f"computed [{start_index}, {end_index}) for {num_samples} samples."
        )
    return IndexSpan(start_index, end_index)


def timestamp_index_span(
    timestamps_seconds: Tensor,
    interval: TimeInterval,
) -> IndexSpan:
    """Map an interval to indices in an explicit timestamp tensor.

    Args:
        timestamps_seconds: Non-empty, finite, strictly increasing floating
            tensor with shape ``[T]``.
        interval: Logical half-open interval ``[start_seconds, end_seconds)``.

    Returns:
        Half-open index span found with left insertion for both boundaries.
        Intervals may lie partly or wholly outside the timestamp range; no
        interpolation, extrapolation, or timestamp filling occurs.

    Raises:
        TypeError: If the timestamp input is not a floating tensor or interval
            has the wrong type.
        ValueError: If timestamps are not a non-empty one-dimensional finite,
            strictly increasing sequence.

    This is a CPU-oriented data-layer helper. Passing a GPU tensor is supported
    by PyTorch but scalar validation and result extraction synchronize the
    device. Stored timestamps are promoted to float64 for comparison so a
    higher-precision interval boundary is not rounded back to float32. The
    input tensor is never modified.
    """
    if not isinstance(timestamps_seconds, Tensor):
        raise TypeError("timestamps_seconds must be a Tensor.")
    if not isinstance(interval, TimeInterval):
        raise TypeError("interval must be a TimeInterval.")
    if timestamps_seconds.ndim != 1:
        raise ValueError(
            "timestamps_seconds must have exact shape [T]; "
            f"received {tuple(timestamps_seconds.shape)}."
        )
    if timestamps_seconds.numel() == 0:
        raise ValueError("timestamps_seconds must be non-empty.")
    if not timestamps_seconds.is_floating_point():
        raise TypeError("timestamps_seconds must be floating point.")
    if not bool(torch.isfinite(timestamps_seconds).all()):
        raise ValueError("timestamps_seconds must contain only finite values.")
    if timestamps_seconds.numel() > 1 and not bool(
        (timestamps_seconds[1:] > timestamps_seconds[:-1]).all()
    ):
        raise ValueError("timestamps_seconds must be strictly increasing.")

    comparison_timestamps = timestamps_seconds.to(dtype=torch.float64)
    boundaries = torch.tensor(
        [interval.start_seconds, interval.end_seconds],
        dtype=torch.float64,
        device=timestamps_seconds.device,
    )
    positions = torch.searchsorted(comparison_timestamps, boundaries, right=False)
    return IndexSpan(int(positions[0].item()), int(positions[1].item()))


def source_overlap(
    window: TimeInterval,
    source: TimedSourceRef,
) -> TimeInterval | None:
    """Return the positive-length overlap between a window and timed source."""
    if not isinstance(window, TimeInterval):
        raise TypeError("window must be a TimeInterval.")
    if not isinstance(source, TimedSourceRef):
        raise TypeError("source must be a TimedSourceRef.")
    return window.intersection(source.available_interval)


def validate_record_alignment(record: MultimodalWindowRecord) -> None:
    """Validate source coverage for one immutable manifest record.

    Speech, when present, must fully contain the record window. Every
    physiology source must have positive-length overlap; partial physiology
    coverage is legal. Adjacent intervals do not overlap.

    Args:
        record: Valid record to inspect without loading sources or making masks.

    Returns:
        ``None``.

    Raises:
        TypeError: If ``record`` has the wrong type.
        ValueError: If speech coverage or physiology overlap is invalid.
    """
    if not isinstance(record, MultimodalWindowRecord):
        raise TypeError("record must be a MultimodalWindowRecord.")
    if (
        record.speech_source is not None
        and not record.speech_source.available_interval.contains(record.window)
    ):
        raise ValueError("speech source must fully cover the record window.")
    for channel_source in record.physio_sources:
        if source_overlap(record.window, channel_source.source) is None:
            raise ValueError(
                "physiology source must overlap the record window; "
                f"channel={channel_source.channel_name!r}."
            )
