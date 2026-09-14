"""Observable channel-level physiology quality statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from emotion_model.physiology.channel_metadata import PhysioChannelWindow


def _validate_optional_artifact_mask(
    artifact_mask: Tensor | None,
    *,
    window: PhysioChannelWindow,
    name: str,
) -> Tensor:
    if artifact_mask is None:
        return torch.zeros_like(window.valid_mask)
    if not isinstance(artifact_mask, Tensor):
        raise TypeError(f"{name} must be a Tensor or None.")
    if artifact_mask.dtype != torch.bool:
        raise TypeError(f"{name} must be bool with True meaning artifact.")
    if artifact_mask.ndim != 1 or tuple(artifact_mask.shape) != tuple(
        window.valid_mask.shape
    ):
        raise ValueError(
            f"{name} must have exact shape {tuple(window.valid_mask.shape)}; "
            f"received {tuple(artifact_mask.shape)}."
        )
    if artifact_mask.device != window.valid_mask.device:
        raise ValueError(f"{name} must be on the window device.")
    return artifact_mask


@dataclass(frozen=True)
class PhysioChannelQuality:
    """Fixed-order observable quality statistics for one channel window.

    Attributes:
        valid_ratio: Scalar tensor in ``[0, 1]`` equal to valid count / ``T``.
        missing_ratio: Scalar tensor in ``[0, 1]`` equal to one minus valid ratio.
        flatline_ratio: Scalar flatline-artifact count divided by valid count.
        outlier_ratio: Scalar robust-outlier count divided by valid count.
        artifact_ratio: Scalar caller-supplied artifact count divided by valid count.
        channel_available: Scalar ``1`` when at least one position is valid,
            otherwise ``0``.

    All ratios describe observable statistics, not medical-grade quality or
    calibrated reliability.
    """

    valid_ratio: Tensor
    missing_ratio: Tensor
    flatline_ratio: Tensor
    outlier_ratio: Tensor
    artifact_ratio: Tensor
    channel_available: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.valid_ratio,
            self.missing_ratio,
            self.flatline_ratio,
            self.outlier_ratio,
            self.artifact_ratio,
            self.channel_available,
        )
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("PhysioChannelQuality fields must be tensors.")
        reference = tensors[0]
        for value in tensors:
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError("PhysioChannelQuality fields must be floating scalars.")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    "PhysioChannelQuality fields must share dtype and device."
                )
            if not bool(torch.isfinite(value)) or not bool(
                (value >= 0.0) & (value <= 1.0)
            ):
                raise ValueError("PhysioChannelQuality fields must be finite in [0, 1].")
        if not bool(
            (self.channel_available == 0.0) | (self.channel_available == 1.0)
        ):
            raise ValueError("channel_available must be exactly 0 or 1.")

    def as_tensor(self) -> Tensor:
        """Return quality features with fixed shape ``[6]`` and order.

        Returns:
            Tensor ``[valid_ratio, missing_ratio, flatline_ratio,
            outlier_ratio, artifact_ratio, channel_available]`` on the same
            dtype/device as the scalar fields.
        """
        return torch.stack(
            (
                self.valid_ratio,
                self.missing_ratio,
                self.flatline_ratio,
                self.outlier_ratio,
                self.artifact_ratio,
                self.channel_available,
            )
        )


def compute_channel_quality(
    window: PhysioChannelWindow,
    *,
    flatline_mask: Tensor | None = None,
    outlier_mask: Tensor | None = None,
    artifact_mask: Tensor | None = None,
) -> PhysioChannelQuality:
    """Compute observable quality ratios for one channel window.

    Args:
        window: Channel window with ``values/mask/timestamps: [T]``.
        flatline_mask: Optional boolean artifact mask ``[T]``.
        outlier_mask: Optional boolean artifact mask ``[T]``.
        artifact_mask: Optional generic boolean artifact mask ``[T]``.

    Returns:
        Immutable scalar statistics. Artifact ratios count only positions that
        were originally valid and divide by ``max(valid_count, 1)``. A fully
        invalid channel returns ``[0, 1, 0, 0, 0, 0]``.

    Raises:
        TypeError: If the window or supplied masks have invalid types/dtypes.
        ValueError: If a supplied mask has the wrong shape or device.

    Artifact masks use ``True=artifact`` and are not merged into the window's
    ``True=valid`` mask. Inputs are not modified.
    """
    if not isinstance(window, PhysioChannelWindow):
        raise TypeError("window must be a PhysioChannelWindow.")
    flatline = _validate_optional_artifact_mask(
        flatline_mask,
        window=window,
        name="flatline_mask",
    )
    outlier = _validate_optional_artifact_mask(
        outlier_mask,
        window=window,
        name="outlier_mask",
    )
    artifact = _validate_optional_artifact_mask(
        artifact_mask,
        window=window,
        name="artifact_mask",
    )
    dtype = window.values.dtype
    valid_count = window.valid_mask.sum().to(dtype=dtype)
    total_count = torch.as_tensor(
        window.valid_mask.numel(),
        dtype=dtype,
        device=window.values.device,
    )
    denominator = torch.clamp(valid_count, min=1.0)
    valid_ratio = valid_count / total_count
    missing_ratio = 1.0 - valid_ratio

    def artifact_ratio(mask: Tensor) -> Tensor:
        count = (mask & window.valid_mask).sum().to(dtype=dtype)
        return count / denominator

    return PhysioChannelQuality(
        valid_ratio=valid_ratio,
        missing_ratio=missing_ratio,
        flatline_ratio=artifact_ratio(flatline),
        outlier_ratio=artifact_ratio(outlier),
        artifact_ratio=artifact_ratio(artifact),
        channel_available=(valid_count > 0).to(dtype=dtype),
    )
