"""Tests for observable physiology channel quality statistics."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import Tensor

from emotion_model.physiology import (
    PhysioChannelQuality,
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
    compute_channel_quality,
)


def _window(mask: Tensor) -> PhysioChannelWindow:
    """Create a quality-test window with explicit validity."""
    values = torch.arange(mask.numel(), dtype=torch.float32)
    values[~mask] = float("nan")
    return PhysioChannelWindow(
        PhysioChannelSpec("bvp", PhysioSignalKind.BVP, 64.0, "a.u."),
        values,
        mask,
        torch.arange(mask.numel(), dtype=torch.float32) / 64.0,
    )


def test_quality_ratios_use_only_originally_valid_positions_and_fixed_order() -> None:
    """Verify all definitions against independent hand-counted ratios."""
    window = _window(torch.tensor([True, True, False, True, False]))
    flatline = torch.tensor([True, False, True, False, False])
    outlier = torch.tensor([False, True, False, False, True])
    artifact = torch.tensor([True, False, False, True, True])
    originals = tuple(mask.clone() for mask in (flatline, outlier, artifact))

    quality = compute_channel_quality(
        window,
        flatline_mask=flatline,
        outlier_mask=outlier,
        artifact_mask=artifact,
    )

    expected = torch.tensor([3 / 5, 2 / 5, 1 / 3, 1 / 3, 2 / 3, 1.0])
    torch.testing.assert_close(quality.as_tensor(), expected)
    assert all(bool((value >= 0.0) & (value <= 1.0)) for value in quality.as_tensor())
    for actual, original in zip((flatline, outlier, artifact), originals):
        assert torch.equal(actual, original)
    assert torch.equal(window.valid_mask, torch.tensor([True, True, False, True, False]))
    with pytest.raises(FrozenInstanceError):
        quality.valid_ratio = torch.tensor(0.0)


def test_quality_fully_invalid_contract() -> None:
    """Return [0,1,0,0,0,0] for a completely unavailable channel."""
    window = _window(torch.zeros(4, dtype=torch.bool))
    artifact = torch.ones(4, dtype=torch.bool)

    quality = compute_channel_quality(
        window,
        flatline_mask=artifact,
        outlier_mask=artifact,
        artifact_mask=artifact,
    )

    assert torch.equal(quality.as_tensor(), torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(4),
        torch.ones(3, dtype=torch.bool),
        torch.ones((1, 4), dtype=torch.bool),
    ],
)
def test_quality_rejects_invalid_artifact_masks(mask: Tensor) -> None:
    """Reject float, wrong-length, and non-vector artifact masks."""
    exception = TypeError if mask.dtype != torch.bool else ValueError
    with pytest.raises(exception):
        compute_channel_quality(_window(torch.ones(4, dtype=torch.bool)), flatline_mask=mask)


def test_quality_dataclass_rejects_non_scalar_or_out_of_range_values() -> None:
    """Keep public diagnostic fields finite floating scalars in [0,1]."""
    valid = {
        "valid_ratio": torch.tensor(0.5),
        "missing_ratio": torch.tensor(0.5),
        "flatline_ratio": torch.tensor(0.0),
        "outlier_ratio": torch.tensor(0.0),
        "artifact_ratio": torch.tensor(0.0),
        "channel_available": torch.tensor(1.0),
    }
    with pytest.raises(ValueError, match="floating scalars"):
        PhysioChannelQuality(**{**valid, "valid_ratio": torch.tensor([0.5])})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PhysioChannelQuality(**{**valid, "artifact_ratio": torch.tensor(1.1)})
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        PhysioChannelQuality(
            **{**valid, "channel_available": torch.tensor(0.5)}
        )
