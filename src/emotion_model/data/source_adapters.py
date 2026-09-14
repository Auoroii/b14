"""Injectable in-memory source adapter contracts for the CPU data layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from emotion_model.data.manifest import (
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    TimedSourceRef,
)
from emotion_model.physiology.channel_metadata import (
    PhysioChannelSpec,
    PhysioChannelWindow,
)
from emotion_model.physiology.normalization import NormalizationKey


class SourceAdapterError(RuntimeError):
    """Report source loading or adapter-contract failures with sample context."""


@dataclass(frozen=True)
class SpeechSourceData:
    """Hold one adapter-loaded, regularly sampled mono speech source.

    Args:
        waveform: Non-empty finite floating CPU tensor with shape ``[L]``.
        sample_rate_hz: Positive integer sample rate in hertz; booleans fail.
        timeline_start_seconds: Finite time corresponding to ``waveform[0]``.
        speech_activity_mask: Optional boolean participant-activity mask
            ``[L]``. Internal ``False`` regions are allowed.

    The source is already mono. No resampling or amplitude normalization is
    performed, and the caller's waveform tensor is validated without mutation.
    """

    waveform: Tensor
    sample_rate_hz: int
    timeline_start_seconds: float
    speech_activity_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor):
            raise TypeError("waveform must be a Tensor.")
        if self.waveform.ndim != 1:
            raise ValueError(
                "waveform must have exact shape [L]; "
                f"received {tuple(self.waveform.shape)}."
            )
        if self.waveform.numel() == 0:
            raise ValueError("waveform must be non-empty.")
        if not self.waveform.is_floating_point():
            raise TypeError("waveform must be floating point.")
        if self.waveform.device.type != "cpu":
            raise ValueError("waveform must be on CPU.")
        if not bool(torch.isfinite(self.waveform).all()):
            raise ValueError("waveform must contain only finite values.")
        if self.speech_activity_mask is not None:
            if not isinstance(self.speech_activity_mask, Tensor):
                raise TypeError("speech_activity_mask must be a Tensor or None.")
            if (
                self.speech_activity_mask.dtype != torch.bool
                or tuple(self.speech_activity_mask.shape)
                != tuple(self.waveform.shape)
            ):
                raise ValueError(
                    "speech_activity_mask must be bool with shape [L]."
                )
            if self.speech_activity_mask.device.type != "cpu":
                raise ValueError("speech_activity_mask must be on CPU.")
        if isinstance(self.sample_rate_hz, bool) or not isinstance(
            self.sample_rate_hz,
            int,
        ):
            raise TypeError("sample_rate_hz must be an integer, not bool.")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0.")
        if isinstance(self.timeline_start_seconds, bool) or not isinstance(
            self.timeline_start_seconds,
            (int, float),
        ):
            raise TypeError(
                "timeline_start_seconds must be a real number, not bool."
            )
        timeline_start = float(self.timeline_start_seconds)
        if not math.isfinite(timeline_start):
            raise ValueError("timeline_start_seconds must be finite.")
        object.__setattr__(self, "timeline_start_seconds", timeline_start)


class SpeechSourceAdapter(Protocol):
    """Structural protocol for resolving an abstract speech source."""

    def load_speech(self, source: TimedSourceRef) -> SpeechSourceData:
        """Load one source as mono waveform ``[L]`` on CPU."""
        ...


class PhysioSourceAdapter(Protocol):
    """Structural protocol for resolving one explicitly specified channel."""

    def load_physio(
        self,
        source: PhysioChannelSourceRef,
        spec: PhysioChannelSpec,
    ) -> PhysioChannelWindow:
        """Load values, valid mask, and timestamps with shape ``[T]`` on CPU."""
        ...


class NormalizationKeyResolver(Protocol):
    """Resolve an explicit fitted normalization key for one record and channel."""

    def __call__(
        self,
        record: MultimodalWindowRecord,
        spec: PhysioChannelSpec,
    ) -> NormalizationKey:
        """Return one exact normalization key without fitting statistics."""
        ...


@dataclass(frozen=True)
class PhysioArtifactPolicy:
    """Configure optional generic physiology artifact statistics.

    These detectors are generic observable rules, not medical-grade signal
    quality algorithms or diagnostic procedures. ``None`` at the Dataset level
    means no detector runs and no detected values are excluded.
    """

    detect_flatline: bool = True
    flatline_atol: float = 1.0e-6
    flatline_min_run_length: int = 5
    detect_outliers: bool = True
    outlier_threshold: float = 6.0
    outlier_mad_epsilon: float = 1.0e-6
    exclude_detected_artifacts: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("detect_flatline", self.detect_flatline),
            ("detect_outliers", self.detect_outliers),
            ("exclude_detected_artifacts", self.exclude_detected_artifacts),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")
        if isinstance(self.flatline_atol, bool) or not isinstance(
            self.flatline_atol,
            (int, float),
        ):
            raise TypeError("flatline_atol must be a real number, not bool.")
        flatline_atol = float(self.flatline_atol)
        if not math.isfinite(flatline_atol) or flatline_atol < 0.0:
            raise ValueError("flatline_atol must be finite and >= 0.")
        if isinstance(self.flatline_min_run_length, bool) or not isinstance(
            self.flatline_min_run_length,
            int,
        ):
            raise TypeError("flatline_min_run_length must be an integer, not bool.")
        if self.flatline_min_run_length < 2:
            raise ValueError("flatline_min_run_length must be >= 2.")
        for parameter_name, parameter_value in (
            ("outlier_threshold", self.outlier_threshold),
            ("outlier_mad_epsilon", self.outlier_mad_epsilon),
        ):
            if isinstance(parameter_value, bool) or not isinstance(
                parameter_value,
                (int, float),
            ):
                raise TypeError(
                    f"{parameter_name} must be a real number, not bool."
                )
            numeric = float(parameter_value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{parameter_name} must be finite and > 0.")
            object.__setattr__(self, parameter_name, numeric)
        object.__setattr__(self, "flatline_atol", flatline_atol)
