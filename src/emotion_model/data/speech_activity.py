"""Deterministic frame-RMS speaker activity for clean participant audio."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpeechActivityDetectorConfig:
    """Configure frame-RMS activity detection on a clean waveform ``[L]``.

    ``frame_ms`` and ``hop_ms`` define overlapping analysis frames and
    ``rms_threshold`` uses the strict comparison ``RMS > threshold``.
    ``availability_policy`` is fixed to ``"source_presence"``: activity is
    diagnostic-only and every successfully loaded source remains available.
    """

    enabled: bool = False
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    rms_threshold: float = 1.0e-4
    availability_policy: str = "source_presence"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool.")
        for name, value in (
            ("frame_ms", self.frame_ms),
            ("hop_ms", self.hop_ms),
            ("rms_threshold", self.rms_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.frame_ms <= 0.0 or self.hop_ms <= 0.0:
            raise ValueError("frame_ms and hop_ms must be positive.")
        if self.rms_threshold < 0.0:
            raise ValueError("rms_threshold must be non-negative.")
        if self.availability_policy != "source_presence":
            raise ValueError(
                "availability_policy must be 'source_presence'."
            )


def detect_frame_rms_speech_activity(
    waveform: Tensor,
    sample_rate_hz: int,
    config: SpeechActivityDetectorConfig,
) -> Tensor:
    """Detect activity in a clean mono waveform ``[L]``.

    Args:
        waveform: Non-empty finite floating tensor ``[L]``.
        sample_rate_hz: Positive integer sample rate in hertz.
        config: Frame, hop, and RMS threshold settings.

    Returns:
        Contiguous boolean mask ``[L]`` on the waveform device. Every sample
        covered by an active frame is ``True``. Frames at the right boundary
        use their actual, possibly shorter sample count.
    """

    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a Tensor.")
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must have non-empty shape [L].")
    if not waveform.is_floating_point():
        raise TypeError("waveform must be floating point.")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must contain only finite values.")
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int):
        raise TypeError("sample_rate_hz must be an integer, not bool.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if not isinstance(config, SpeechActivityDetectorConfig):
        raise TypeError("config must be SpeechActivityDetectorConfig.")

    frame_length = max(1, round(config.frame_ms * sample_rate_hz / 1000.0))
    hop_length = max(1, round(config.hop_ms * sample_rate_hz / 1000.0))
    sample_count = waveform.numel()
    starts = torch.arange(0, sample_count, hop_length, device=waveform.device)
    ends = torch.clamp(starts + frame_length, max=sample_count)
    squared = waveform.to(dtype=torch.float64).square()
    prefix = torch.cat((squared.new_zeros(1), squared.cumsum(dim=0)))
    energy = prefix.index_select(0, ends) - prefix.index_select(0, starts)
    counts = (ends - starts).to(dtype=energy.dtype)
    rms = torch.sqrt(energy / counts)
    active_frames = rms > config.rms_threshold

    difference = torch.zeros(
        sample_count + 1,
        dtype=torch.int32,
        device=waveform.device,
    )
    active_starts = starts[active_frames]
    active_ends = ends[active_frames]
    if active_starts.numel() > 0:
        ones = torch.ones_like(active_starts, dtype=difference.dtype)
        difference.index_add_(0, active_starts, ones)
        difference.index_add_(0, active_ends, -ones)
    return (difference[:-1].cumsum(dim=0) > 0).contiguous()


def summarize_speech_activity_ratios(
    ratios: Sequence[float],
) -> dict[str, float | int]:
    """Summarize finite per-window activity ratios in ``[0,1]``.

    Args:
        ratios: Non-empty sequence of scalar active/valid sample ratios.

    Returns:
        JSON-compatible distribution statistics and activity bin counts.
    """

    if isinstance(ratios, (str, bytes)) or not isinstance(ratios, Sequence):
        raise TypeError("ratios must be a non-string sequence.")
    if not ratios:
        raise ValueError("ratios must be non-empty.")
    values = torch.tensor(tuple(ratios), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("ratios must be finite.")
    if not bool(((values >= 0.0) & (values <= 1.0)).all()):
        raise ValueError("ratios must lie in [0, 1].")

    def quantile(value: float) -> float:
        return float(torch.quantile(values, value).item())

    return {
        "window_count": values.numel(),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": quantile(0.5),
        "p10": quantile(0.1),
        "p25": quantile(0.25),
        "p50": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "ratio_eq_0_count": int((values == 0.0).sum().item()),
        "ratio_0_to_0_1_count": int(
            ((values > 0.0) & (values < 0.1)).sum().item()
        ),
        "ratio_0_1_to_0_25_count": int(
            ((values >= 0.1) & (values < 0.25)).sum().item()
        ),
        "ratio_ge_0_25_count": int((values >= 0.25).sum().item()),
        "ratio_0_25_to_0_5_count": int(
            ((values >= 0.25) & (values < 0.5)).sum().item()
        ),
        "ratio_ge_0_5_count": int((values >= 0.5).sum().item()),
    }


__all__ = [
    "SpeechActivityDetectorConfig",
    "detect_frame_rms_speech_activity",
    "summarize_speech_activity_ratios",
]
