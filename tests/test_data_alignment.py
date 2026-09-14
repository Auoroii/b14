"""Tests for half-open regular and explicit-timestamp alignment helpers."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from emotion_model.data import (
    EmotionScores,
    IndexSpan,
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    TimedSourceRef,
    TimeInterval,
    regular_sample_span,
    source_overlap,
    timestamp_index_span,
    validate_record_alignment,
)


def test_regular_sample_span_uses_exact_half_open_boundaries() -> None:
    """Include start samples and exclude samples exactly at end."""
    span = regular_sample_span(
        TimeInterval(0.2, 0.5),
        timeline_start_seconds=0.0,
        sample_rate_hz=10.0,
        num_samples=10,
    )
    assert span == IndexSpan(2, 5)
    sample_times = [index / 10 for index in range(10)]
    manual = [index for index, time in enumerate(sample_times) if 0.2 <= time < 0.5]
    assert manual == list(range(span.start_index, span.end_index))


@pytest.mark.parametrize(
    ("interval", "origin", "rate", "count", "expected"),
    [
        (TimeInterval(2.0, 2.5), 2.0, 4.0, 8, IndexSpan(0, 2)),
        (TimeInterval(2.125, 2.375), 2.0, 4.0, 8, IndexSpan(1, 2)),
        (TimeInterval(0.1, 0.2), 0.0, 1.0, 3, IndexSpan(1, 1)),
        (TimeInterval(0.3, 0.6), 0.0, 10.0, 10, IndexSpan(3, 6)),
    ],
)
def test_regular_sample_span_handles_origin_fractional_boundaries_and_empty(
    interval: TimeInterval,
    origin: float,
    rate: float,
    count: int,
    expected: IndexSpan,
) -> None:
    """Map nonzero origins, fractional limits, roundoff, and empty selections."""
    assert regular_sample_span(
        interval,
        timeline_start_seconds=origin,
        sample_rate_hz=rate,
        num_samples=count,
    ) == expected


def test_regular_sample_span_tolerates_only_boundary_roundoff() -> None:
    """Resolve a tiny source-boundary error but reject a material overrun."""
    accepted = regular_sample_span(
        TimeInterval(-5.0e-11, 1.00000000005),
        timeline_start_seconds=0.0,
        sample_rate_hz=10.0,
        num_samples=10,
        tolerance=1.0e-9,
    )
    assert accepted == IndexSpan(0, 10)
    with pytest.raises(ValueError, match="source time range"):
        regular_sample_span(
            TimeInterval(-1.0e-4, 0.5),
            timeline_start_seconds=0.0,
            sample_rate_hz=10.0,
            num_samples=10,
        )
    with pytest.raises(ValueError, match="source time range"):
        regular_sample_span(
            TimeInterval(0.5, 1.001),
            timeline_start_seconds=0.0,
            sample_rate_hz=10.0,
            num_samples=10,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"sample_rate_hz": 0.0}, ValueError),
        ({"sample_rate_hz": float("inf")}, ValueError),
        ({"sample_rate_hz": True}, TypeError),
        ({"timeline_start_seconds": float("nan")}, ValueError),
        ({"num_samples": -1}, ValueError),
        ({"num_samples": True}, TypeError),
        ({"num_samples": 2.0}, TypeError),
        ({"tolerance": -1.0}, ValueError),
        ({"tolerance": float("inf")}, ValueError),
        ({"tolerance": False}, TypeError),
    ],
)
def test_regular_sample_span_rejects_invalid_parameters(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    """Validate rates, origins, counts, and tolerance without coercion."""
    arguments: dict[str, object] = {
        "timeline_start_seconds": 0.0,
        "sample_rate_hz": 10.0,
        "num_samples": 10,
        "tolerance": 1.0e-9,
    }
    arguments.update(kwargs)
    with pytest.raises(exception):
        regular_sample_span(TimeInterval(0.1, 0.2), **arguments)  # type: ignore[arg-type]


def test_regular_sample_span_rejects_interval_for_empty_source() -> None:
    """Reject every positive interval when the regular source has no samples."""
    with pytest.raises(ValueError, match="num_samples=0"):
        regular_sample_span(
            TimeInterval(0, 0.1),
            timeline_start_seconds=0.0,
            sample_rate_hz=10.0,
            num_samples=0,
        )
    with pytest.raises(ValueError, match="num_samples=0"):
        regular_sample_span(
            TimeInterval(-5.0e-11, 5.0e-11),
            timeline_start_seconds=0.0,
            sample_rate_hz=10.0,
            num_samples=0,
            tolerance=1.0e-9,
        )


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (TimeInterval(1.0, 3.0), IndexSpan(1, 3)),
        (TimeInterval(1.1, 2.9), IndexSpan(2, 3)),
        (TimeInterval(-2.0, 0.5), IndexSpan(0, 1)),
        (TimeInterval(3.0, 8.0), IndexSpan(3, 4)),
        (TimeInterval(-3.0, -2.0), IndexSpan(0, 0)),
        (TimeInterval(5.0, 6.0), IndexSpan(4, 4)),
        (TimeInterval(2.0, 2.1), IndexSpan(2, 3)),
    ],
)
def test_timestamp_index_span_matches_left_insertion_contract(
    interval: TimeInterval,
    expected: IndexSpan,
) -> None:
    """Handle exact, internal, outside, empty, and single-point intervals."""
    timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    original = timestamps.clone()

    span = timestamp_index_span(timestamps, interval)

    assert span == expected
    assert torch.equal(timestamps, original)
    manual = torch.nonzero(
        (timestamps >= interval.start_seconds)
        & (timestamps < interval.end_seconds),
        as_tuple=False,
    ).flatten()
    if manual.numel() == 0:
        assert span.length == 0
    else:
        assert span == IndexSpan(int(manual[0]), int(manual[-1]) + 1)


@pytest.mark.parametrize(
    ("timestamps", "exception"),
    [
        (torch.ones((1, 2)), ValueError),
        (torch.tensor([], dtype=torch.float32), ValueError),
        (torch.tensor([0, 1]), TypeError),
        (torch.tensor([0.0, float("nan")]), ValueError),
        (torch.tensor([0.0, float("inf")]), ValueError),
        (torch.tensor([0.0, 0.0]), ValueError),
        (torch.tensor([1.0, 0.0]), ValueError),
    ],
)
def test_timestamp_index_span_rejects_invalid_timestamps(
    timestamps: Tensor,
    exception: type[Exception],
) -> None:
    """Reject non-vector, empty, integer, non-finite, and unordered timestamps."""
    with pytest.raises(exception):
        timestamp_index_span(timestamps, TimeInterval(0, 1))


def test_timestamp_index_span_preserves_float32_boundary_behavior() -> None:
    """Use the timestamp dtype for search boundaries in float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=dtype)
        interval = TimeInterval(float(timestamps[1]), float(timestamps[3]))
        assert timestamp_index_span(timestamps, interval) == IndexSpan(1, 3)


def test_timestamp_span_does_not_round_interval_boundary_back_to_float32() -> None:
    """Exclude a timestamp strictly below start even within one float32 ULP."""
    timestamps = torch.tensor([0.1, 0.2], dtype=torch.float32)
    stored_first = float(timestamps[0].item())
    interval = TimeInterval(stored_first + 1.0e-10, 0.25)

    span = timestamp_index_span(timestamps, interval)

    assert timestamps[0].item() < interval.start_seconds
    assert span == IndexSpan(1, 2)


def _record_for_alignment() -> MultimodalWindowRecord:
    """Create one valid record with full speech and partial physiology coverage."""
    return MultimodalWindowRecord(
        "sample",
        "participant",
        "session",
        TimeInterval(2, 4),
        EmotionScores(3, 4),
        TimedSourceRef("speech", TimeInterval(0, 5)),
        (
            PhysioChannelSourceRef(
                "eda",
                TimedSourceRef("eda", TimeInterval(3, 6)),
            ),
        ),
    )


def test_source_overlap_and_record_alignment_use_positive_overlap() -> None:
    """Return exact intersection and accept full-speech/partial-physiology rules."""
    record = _record_for_alignment()
    overlap = source_overlap(record.window, record.physio_sources[0].source)

    assert overlap == TimeInterval(3, 4)
    assert validate_record_alignment(record) is None
    assert source_overlap(
        TimeInterval(0, 1),
        TimedSourceRef("adjacent", TimeInterval(1, 2)),
    ) is None


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (source_overlap, ("not-an-interval", TimedSourceRef("x", TimeInterval(0, 1)))),
        (source_overlap, (TimeInterval(0, 1), "not-a-source")),
        (validate_record_alignment, ("not-a-record",)),
        (regular_sample_span, ("not-an-interval",)),
        (timestamp_index_span, ([0.0, 1.0], TimeInterval(0, 1))),
    ],
)
def test_alignment_helpers_reject_wrong_public_types(
    function: object,
    args: tuple[object, ...],
) -> None:
    """Give clear type errors instead of silently coercing public inputs."""
    with pytest.raises(TypeError):
        if function is regular_sample_span:
            regular_sample_span(  # type: ignore[arg-type]
                args[0],
                timeline_start_seconds=0,
                sample_rate_hz=1,
                num_samples=2,
            )
        else:
            function(*args)  # type: ignore[operator]


def test_regular_formula_matches_manual_times_across_fractional_grid() -> None:
    """Independently compare the index formula with explicit sample filtering."""
    origin = -0.25
    rate = 8.0
    count = 20
    interval = TimeInterval(-0.01, 1.13)
    span = regular_sample_span(
        interval,
        timeline_start_seconds=origin,
        sample_rate_hz=rate,
        num_samples=count,
    )
    times = [origin + index / rate for index in range(count)]
    expected = [
        index
        for index, time in enumerate(times)
        if interval.start_seconds <= time < interval.end_seconds
    ]
    assert expected == list(range(span.start_index, span.end_index))
    assert span.start_index == math.ceil(
        (interval.start_seconds - origin) * rate - 1.0e-9
    )
