"""Explicit physiology channel semantics and immutable window contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


class PhysioSignalKind(StrEnum):
    """Stable, dataset-independent physiology signal semantics."""

    ECG_WAVEFORM = "ecg_waveform"
    HEART_RATE = "heart_rate"
    RR_INTERVAL = "rr_interval"
    BVP = "bvp"
    EDA = "eda"
    TEMPERATURE = "temperature"
    OTHER = "other"


@dataclass(frozen=True)
class PhysioChannelSpec:
    """Describe one physiology channel without guessing from its field name.

    Args:
        name: Non-empty caller-defined channel name.
        signal_kind: Explicit :class:`PhysioSignalKind` or its stable string
            value. ECG waveform, heart rate, and RR interval are distinct.
        native_sample_rate_hz: Finite positive native sampling rate in hertz.
        unit: Non-empty physical unit string.
        description: Optional explanatory text. It is required and non-empty
            when ``signal_kind`` is :attr:`PhysioSignalKind.OTHER`.

    Raises:
        TypeError: If strings or numeric values have invalid types.
        ValueError: If text is empty, the rate is non-finite/non-positive, the
            signal kind is unknown, or ``OTHER`` lacks a description.

    No dataset-specific field mapping or filter frequency is inferred.
    """

    name: str
    signal_kind: PhysioSignalKind
    native_sample_rate_hz: float
    unit: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")
        if not self.name.strip():
            raise ValueError("name must be non-empty.")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string.")
        if not self.unit.strip():
            raise ValueError("unit must be non-empty.")
        if isinstance(self.native_sample_rate_hz, bool) or not isinstance(
            self.native_sample_rate_hz,
            (int, float),
        ):
            raise TypeError("native_sample_rate_hz must be a real number.")
        rate = float(self.native_sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                "native_sample_rate_hz must be finite and > 0; "
                f"received {self.native_sample_rate_hz!r}."
            )
        try:
            resolved_kind = PhysioSignalKind(self.signal_kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown physiology signal kind {self.signal_kind!r}."
            ) from error
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None.")
        if resolved_kind is PhysioSignalKind.OTHER and (
            self.description is None or not self.description.strip()
        ):
            raise ValueError("description must be non-empty for signal_kind='other'.")
        object.__setattr__(self, "signal_kind", resolved_kind)
        object.__setattr__(self, "native_sample_rate_hz", rate)


@dataclass(frozen=True)
class PhysioChannelWindow:
    """Hold one explicitly timed physiology channel window.

    Args:
        spec: Immutable channel semantics.
        values: Floating-point signal tensor with shape ``[T]``.
        valid_mask: Boolean tensor with shape ``[T]`` and project semantics
            ``True=valid``.
        timestamps_seconds: Floating-point tensor with shape ``[T]`` containing
            finite, strictly increasing timestamps in seconds.

    Raises:
        TypeError: If ``spec`` is invalid, values/timestamps are not floating
            tensors, or ``valid_mask`` is not boolean.
        ValueError: If tensors are not one-dimensional, are empty, differ in
            length/device, timestamps are non-finite/not strictly increasing,
            or an effective value contains NaN or Inf.

    Invalid values may contain arbitrary placeholders including NaN and Inf.
    Construction validates but never modifies caller tensors.
    """

    spec: PhysioChannelSpec
    values: Tensor
    valid_mask: Tensor
    timestamps_seconds: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PhysioChannelSpec):
            raise TypeError("spec must be a PhysioChannelSpec.")
        for name, tensor in (
            ("values", self.values),
            ("valid_mask", self.valid_mask),
            ("timestamps_seconds", self.timestamps_seconds),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a Tensor.")
            if tensor.ndim != 1:
                raise ValueError(
                    f"{name} must have exact shape [T]; received {tuple(tensor.shape)}."
                )
        if not self.values.is_floating_point():
            raise TypeError(
                f"values must be floating point; received {self.values.dtype}."
            )
        if self.valid_mask.dtype != torch.bool:
            raise TypeError(
                "valid_mask must have dtype torch.bool with True meaning valid; "
                f"received {self.valid_mask.dtype}."
            )
        if not self.timestamps_seconds.is_floating_point():
            raise TypeError(
                "timestamps_seconds must be floating point; "
                f"received {self.timestamps_seconds.dtype}."
            )
        length = self.values.shape[0]
        if length == 0:
            raise ValueError("PhysioChannelWindow requires a non-empty time axis.")
        if self.valid_mask.shape[0] != length or self.timestamps_seconds.shape[0] != length:
            raise ValueError(
                "values, valid_mask, and timestamps_seconds must have identical "
                f"lengths; received {length}, {self.valid_mask.shape[0]}, and "
                f"{self.timestamps_seconds.shape[0]}."
            )
        devices = {
            self.values.device,
            self.valid_mask.device,
            self.timestamps_seconds.device,
        }
        if len(devices) != 1:
            raise ValueError("window tensors must be on the same device.")
        if not bool(torch.isfinite(self.timestamps_seconds).all()):
            raise ValueError("timestamps_seconds must contain only finite values.")
        if length > 1 and not bool(
            (self.timestamps_seconds[1:] > self.timestamps_seconds[:-1]).all()
        ):
            raise ValueError("timestamps_seconds must be strictly increasing.")
        if not bool(torch.isfinite(self.values[self.valid_mask]).all()):
            raise ValueError("values at valid positions must contain only finite values.")

    def safe_values(self) -> Tensor:
        """Return signal values ``[T]`` with invalid positions replaced by zero.

        Returns:
            A new floating tensor with shape ``[T]``. Valid values are
            unchanged; ``False`` mask positions are finite exact zeros.

        The original ``values`` and ``valid_mask`` tensors are not modified.
        """
        return torch.where(
            self.valid_mask,
            self.values,
            torch.zeros_like(self.values),
        )
