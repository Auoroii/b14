"""Independent tiny per-channel physiology classifier."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emotion_model.common import derive_quadrant_probabilities, masked_mean_std
from emotion_model.physiology.channel_metadata import PhysioChannelSpec

_RELIABILITY_SCORE_TYPE = "availability_indicator"
_STATE_VERSION = 2
PHYSIOLOGY_ENCODER_MODES = frozenset({"single_scale", "multiscale_dilated"})
ECG_ENCODER_TYPE = "mask_aware_statistics_mlp"


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _dropout(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1).")
    return result


class _IndependentChannelStem(nn.Module):
    """Map one sanitized channel ``[B, 1, T]`` to ``[B, 16, T]``."""

    def __init__(self, first_dim: int, second_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, first_dim, kernel_size=5, padding=2)
        self.activation1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            first_dim,
            second_dim,
            kernel_size=5,
            padding=2,
        )
        self.activation2 = nn.GELU()

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        """Encode values ``[B,1,T]`` using boolean mask ``[B,1,T]``."""
        hidden = self.conv1(values)
        hidden = torch.where(valid_mask, hidden, torch.zeros_like(hidden))
        hidden = self.dropout(self.activation1(hidden))
        hidden = torch.where(valid_mask, hidden, torch.zeros_like(hidden))
        hidden = self.conv2(hidden)
        hidden = torch.where(valid_mask, hidden, torch.zeros_like(hidden))
        hidden = self.activation2(hidden)
        return torch.where(valid_mask, hidden, torch.zeros_like(hidden))


class MultiScaleDilatedConv1dStem(nn.Module):
    """Encode one masked physiology channel at three temporal scales.

    Args:
        first_dim: Positive shared shallow feature width.
        second_dim: Positive output width for every dilation branch and the
            fused output.
        dilations: Exactly three distinct positive Conv1D dilation values.
        dropout: Shared shallow-feature dropout probability in ``[0, 1)``.

    Forward accepts floating ``values`` with shape ``[B, 1, T]`` and boolean
    ``valid_mask`` with shape ``[B, 1, T]``. It returns finite features with
    shape ``[B, second_dim, T]`` and exact zeros at invalid positions.
    """

    def __init__(
        self,
        first_dim: int,
        second_dim: int,
        dilations: Sequence[int] = (1, 2, 4),
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.first_dim = _positive_integer(first_dim, name="first_dim")
        self.second_dim = _positive_integer(second_dim, name="second_dim")
        dilation_values = tuple(dilations)
        if len(dilation_values) != 3:
            raise ValueError("dilations must contain exactly three values.")
        self.dilations = tuple(
            _positive_integer(value, name=f"dilations[{index}]")
            for index, value in enumerate(dilation_values)
        )
        if len(set(self.dilations)) != len(self.dilations):
            raise ValueError("dilations must contain three distinct values.")
        self.dropout_probability = _dropout(dropout, name="dropout")
        self.shared_convolution = nn.Conv1d(
            1,
            self.first_dim,
            kernel_size=3,
            dilation=1,
            padding=1,
        )
        self.shared_activation = nn.GELU()
        self.shared_dropout = nn.Dropout(self.dropout_probability)
        self.branch_convolutions = nn.ModuleList(
            nn.Conv1d(
                self.first_dim,
                self.second_dim,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            )
            for dilation in self.dilations
        )
        self.fusion_convolution = nn.Conv1d(
            len(self.dilations) * self.second_dim,
            self.second_dim,
            kernel_size=1,
        )
        self.fusion_activation = nn.GELU()

    @staticmethod
    def _validate_inputs(values: Tensor, valid_mask: Tensor) -> None:
        if not isinstance(values, Tensor) or values.ndim != 3:
            raise ValueError("values must have exact shape [B, 1, T].")
        if not values.is_floating_point():
            raise TypeError("values must be floating point.")
        if values.shape[0] <= 0 or values.shape[1] != 1 or values.shape[2] <= 0:
            raise ValueError("values must have non-empty exact shape [B, 1, T].")
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean Tensor.")
        if tuple(valid_mask.shape) != tuple(values.shape):
            raise ValueError("valid_mask must have the same [B, 1, T] shape.")
        if valid_mask.device != values.device:
            raise ValueError("values and valid_mask must be on the same device.")
        if not bool(torch.isfinite(values[valid_mask]).all()):
            raise ValueError("valid physiology values must be finite.")

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        """Map masked ``[B,1,T]`` values to finite ``[B,second_dim,T]``."""

        self._validate_inputs(values, valid_mask)
        safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
        hidden = self.shared_convolution(safe_values)
        hidden = torch.where(valid_mask, hidden, torch.zeros_like(hidden))
        hidden = self.shared_dropout(self.shared_activation(hidden))
        hidden = torch.where(valid_mask, hidden, torch.zeros_like(hidden))
        branches = []
        for convolution in self.branch_convolutions:
            branch = convolution(hidden)
            branch = torch.where(valid_mask, branch, torch.zeros_like(branch))
            branches.append(branch)
        concatenated = torch.cat(branches, dim=1)
        fused = self.fusion_convolution(concatenated)
        fused = torch.where(valid_mask, fused, torch.zeros_like(fused))
        fused = self.fusion_activation(fused)
        return torch.where(valid_mask, fused, torch.zeros_like(fused))


class LightweightEcgEncoder(nn.Module):
    """Encode low-frequency Polar HR values using five masked statistics.

    Forward consumes normalized ``ecg_values`` and boolean ``ecg_mask`` with
    shape ``[B, Te]`` plus an optional boolean timeline mask of the same shape.
    It returns ``(embedding, statistics, available)`` with shapes ``[B, De]``,
    ``[B, 5]``, and ``[B]``. The statistic order is mean, population standard
    deviation, last-minus-first delta, normalized-time least-squares slope,
    and valid ratio. Completely missing rows return exact zeros.
    """

    def __init__(self, embedding_dim: int = 16, *, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding_dim = _positive_integer(embedding_dim, name="embedding_dim")
        self.dropout = _dropout(dropout, name="dropout")
        self.network = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(16, self.embedding_dim),
        )

    def forward(
        self,
        ecg_values: Tensor,
        ecg_mask: Tensor,
        *,
        timeline_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ECG embedding ``[B,De]``, statistics ``[B,5]``, validity ``[B]``."""
        if not isinstance(ecg_values, Tensor) or ecg_values.ndim != 2:
            raise ValueError("ecg_values must have shape [B, Te].")
        if not ecg_values.is_floating_point():
            raise TypeError("ecg_values must be floating point.")
        if (
            not isinstance(ecg_mask, Tensor)
            or ecg_mask.dtype != torch.bool
            or tuple(ecg_mask.shape) != tuple(ecg_values.shape)
            or ecg_mask.device != ecg_values.device
        ):
            raise ValueError("ecg_mask must be bool [B, Te] on the input device.")
        if ecg_values.shape[0] <= 0 or ecg_values.shape[1] <= 0:
            raise ValueError("ecg_values dimensions must be non-empty.")
        if timeline_mask is None:
            timeline = torch.ones_like(ecg_mask)
        else:
            if (
                not isinstance(timeline_mask, Tensor)
                or timeline_mask.dtype != torch.bool
                or tuple(timeline_mask.shape) != tuple(ecg_values.shape)
                or timeline_mask.device != ecg_values.device
            ):
                raise ValueError("timeline_mask must be bool [B, Te].")
            timeline = timeline_mask
        if bool((ecg_mask & ~timeline).any()):
            raise ValueError("ecg_mask must be a subset of timeline_mask.")
        if not bool(torch.isfinite(ecg_values[ecg_mask]).all()):
            raise ValueError("valid ECG values must be finite.")
        safe = torch.where(ecg_mask, ecg_values, torch.zeros_like(ecg_values))
        counts = ecg_mask.sum(dim=1)
        available = counts > 0
        denominator = counts.clamp_min(1).to(dtype=ecg_values.dtype)
        mean = safe.sum(dim=1) / denominator
        centered = torch.where(
            ecg_mask,
            safe - mean.unsqueeze(1),
            torch.zeros_like(safe),
        )
        std = torch.sqrt((centered.square().sum(dim=1) / denominator).clamp_min(0.0))
        indices = torch.arange(ecg_values.shape[1], device=ecg_values.device)
        first_indices = torch.where(
            ecg_mask,
            indices.unsqueeze(0),
            ecg_values.shape[1],
        ).min(dim=1).values.clamp_max(ecg_values.shape[1] - 1)
        last_indices = (
            torch.where(ecg_mask, indices.unsqueeze(0), -1)
            .max(dim=1)
            .values.clamp_min(0)
        )
        first = safe.gather(1, first_indices.unsqueeze(1)).squeeze(1)
        last = safe.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        at_least_two = counts >= 2
        delta = torch.where(at_least_two, last - first, torch.zeros_like(mean))
        timeline_lengths = timeline.sum(dim=1).clamp_min(1)
        time = indices.to(dtype=ecg_values.dtype).unsqueeze(0) / (
            (timeline_lengths - 1).clamp_min(1).to(dtype=ecg_values.dtype).unsqueeze(1)
        )
        time_mean = (time * ecg_mask.to(dtype=ecg_values.dtype)).sum(dim=1) / denominator
        time_centered = torch.where(
            ecg_mask,
            time - time_mean.unsqueeze(1),
            torch.zeros_like(time),
        )
        slope_denominator = time_centered.square().sum(dim=1)
        slope = torch.where(
            at_least_two & (slope_denominator > 0),
            (time_centered * centered).sum(dim=1) / slope_denominator.clamp_min(1.0e-12),
            torch.zeros_like(mean),
        )
        valid_ratio = counts.to(dtype=ecg_values.dtype) / timeline_lengths.to(
            dtype=ecg_values.dtype
        )
        statistics = torch.stack((mean, std, delta, slope, valid_ratio), dim=1)
        statistics = torch.where(
            available.unsqueeze(1), statistics, torch.zeros_like(statistics)
        )
        embedding = self.network(statistics)
        embedding = torch.where(
            available.unsqueeze(1), embedding, torch.zeros_like(embedding)
        )
        if not bool(torch.isfinite(statistics).all() and torch.isfinite(embedding).all()):
            raise RuntimeError("ECG encoder output contains NaN or Inf.")
        return embedding, statistics, available


@dataclass(frozen=True)
class LightweightPhysioClassifierOutput:
    """Outputs of the lightweight physiology classifier.

    ``physio_embedding`` is ``[B, Dp]``; ``channel_features`` is
    ``[B, T, C, S2]``; ``channel_statistics`` is ``[B, C, 2*S2]``;
    ``channel_available`` is boolean ``[B, C]``; temporal weights are
    ``[B, T]``; logits/probabilities are ``[B, 2]`` and derived quadrant
    probabilities are ``[B, 4]``.
    """

    arousal_logits: Tensor
    valence_logits: Tensor
    arousal_probabilities: Tensor
    valence_probabilities: Tensor
    quadrant_probabilities: Tensor
    physio_embedding: Tensor
    channel_features: Tensor
    channel_statistics: Tensor
    channel_available: Tensor
    temporal_attention_weights: Tensor
    sample_valid: Tensor
    reliability: Tensor
    reliability_score_type: str
    quadrant_logits: Tensor | None = None
    ecg_embedding: Tensor | None = None
    ecg_statistics: Tensor | None = None
    ecg_available: Tensor | None = None


class LightweightPhysioEmotionClassifier(nn.Module):
    """Encode each named physiology channel with an independent tiny stem.

    Args:
        channel_specs: Ordered semantic channel specifications.
        stem_channels: Exactly two positive Conv1D widths.
        physiology_embedding_dim: Output embedding width.
        dropout: Statistics-projection dropout in ``[0, 1)``.
        stem_dropout: Per-channel stem dropout in ``[0, 1)``.
        physiology_encoder: ``"single_scale"`` for the established two-layer
            stem or ``"multiscale_dilated"`` for the P1 encoder.
        physiology_dilations: Exactly three distinct positive dilation values
            fingerprinted for both encoder modes.

    Forward consumes physiology ``[B,T,C]`` and boolean validity
    ``[B,T,C]`` plus optional boolean time ``[B,T]`` and channel ``[B,C]``
    masks, and returns :class:`LightweightPhysioClassifierOutput`.
    """

    def __init__(
        self,
        channel_specs: Sequence[PhysioChannelSpec],
        stem_channels: Sequence[int] = (8, 16),
        physiology_embedding_dim: int = 64,
        *,
        dropout: float = 0.3,
        stem_dropout: float = 0.1,
        physiology_encoder: str = "single_scale",
        physiology_dilations: Sequence[int] = (1, 2, 4),
        ecg_embedding_dim: int = 16,
        ecg_encoder_type: str = ECG_ENCODER_TYPE,
        ecg_sample_rate_hz: float = 1.0,
    ) -> None:
        super().__init__()
        specs = tuple(channel_specs)
        if not specs or not all(isinstance(spec, PhysioChannelSpec) for spec in specs):
            raise TypeError("channel_specs must contain PhysioChannelSpec values.")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("channel_specs names must be unique.")
        widths = tuple(stem_channels)
        if len(widths) != 2:
            raise ValueError("stem_channels must contain exactly two widths.")
        first_dim = _positive_integer(widths[0], name="stem_channels[0]")
        second_dim = _positive_integer(widths[1], name="stem_channels[1]")
        ecg_specs = tuple(spec for spec in specs if spec.name == "ecg")
        if len(ecg_specs) > 1:
            raise ValueError("at most one ecg channel specification is allowed.")
        dense_specs = tuple(spec for spec in specs if spec.name != "ecg")
        if not dense_specs:
            raise ValueError("at least one dense physiology channel is required.")
        self.channel_specs = specs
        self.dense_channel_specs = dense_specs
        self.channel_names = tuple(spec.name for spec in dense_specs)
        self.ecg_enabled = bool(ecg_specs)
        self.ecg_embedding_dim = _positive_integer(
            ecg_embedding_dim, name="ecg_embedding_dim"
        )
        if ecg_encoder_type != ECG_ENCODER_TYPE:
            raise ValueError(f"ecg_encoder_type must be {ECG_ENCODER_TYPE!r}.")
        self.ecg_encoder_type = ecg_encoder_type
        if (
            isinstance(ecg_sample_rate_hz, bool)
            or not isinstance(ecg_sample_rate_hz, (int, float))
            or not math.isfinite(float(ecg_sample_rate_hz))
            or float(ecg_sample_rate_hz) <= 0.0
        ):
            raise ValueError("ecg_sample_rate_hz must be finite and positive.")
        self.ecg_sample_rate_hz = float(ecg_sample_rate_hz)
        self.stem_channels = (first_dim, second_dim)
        self.physiology_embedding_dim = _positive_integer(
            physiology_embedding_dim,
            name="physiology_embedding_dim",
        )
        self.dropout = _dropout(dropout, name="dropout")
        self.stem_dropout = _dropout(stem_dropout, name="stem_dropout")
        if physiology_encoder not in PHYSIOLOGY_ENCODER_MODES:
            raise ValueError(
                "physiology_encoder must be 'single_scale' or "
                "'multiscale_dilated'."
            )
        self.physiology_encoder = physiology_encoder
        dilation_values = tuple(physiology_dilations)
        if len(dilation_values) != 3:
            raise ValueError(
                "physiology_dilations must contain exactly three values."
            )
        self.physiology_dilations = tuple(
            _positive_integer(value, name=f"physiology_dilations[{index}]")
            for index, value in enumerate(dilation_values)
        )
        if len(set(self.physiology_dilations)) != 3:
            raise ValueError(
                "physiology_dilations must contain three distinct values."
            )
        if self.physiology_encoder == "single_scale":
            self.channel_stems = nn.ModuleList(
                _IndependentChannelStem(first_dim, second_dim, self.stem_dropout)
                for _ in dense_specs
            )
        else:
            self.channel_stems = nn.ModuleList(
                MultiScaleDilatedConv1dStem(
                    first_dim,
                    second_dim,
                    self.physiology_dilations,
                    dropout=self.stem_dropout,
                )
                for _ in dense_specs
            )
        self.ecg_encoder = (
            LightweightEcgEncoder(self.ecg_embedding_dim, dropout=self.dropout)
            if self.ecg_enabled
            else None
        )
        statistics_dim = len(dense_specs) * (2 * second_dim) + len(dense_specs)
        if self.ecg_enabled:
            statistics_dim += self.ecg_embedding_dim + 1
        self.statistics_dim = statistics_dim
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Linear(statistics_dim, self.physiology_embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.arousal_head = nn.Linear(self.physiology_embedding_dim, 2)
        self.valence_head = nn.Linear(self.physiology_embedding_dim, 2)

    def get_extra_state(self) -> dict[str, object]:
        """Return all ordered semantics and architecture checkpoint fields."""
        state: dict[str, object] = {
            "state_version": _STATE_VERSION,
            "variant": "lightweight_shared_dynamic_relation_differential_full_window",
            "channel_specs": tuple(
                (
                    spec.name,
                    spec.signal_kind.value,
                    spec.native_sample_rate_hz,
                    spec.unit,
                    spec.description,
                )
                for spec in self.channel_specs
            ),
            "stem_channels": self.stem_channels,
            "physiology_embedding_dim": self.physiology_embedding_dim,
            "dropout": self.dropout,
            "stem_dropout": self.stem_dropout,
            "physiology_encoder": self.physiology_encoder,
            "physiology_dilations": self.physiology_dilations,
            "reliability_score_type": _RELIABILITY_SCORE_TYPE,
        }
        if self.ecg_enabled:
            state.update(
                {
                    "ecg_enabled": True,
                    "ecg_embedding_dim": self.ecg_embedding_dim,
                    "ecg_encoder_type": self.ecg_encoder_type,
                    "ecg_sample_rate_hz": self.ecg_sample_rate_hz,
                }
            )
        return state

    def set_extra_state(self, state: object) -> None:
        """Reject a checkpoint whose architecture or channel order differs."""
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "Lightweight physiology checkpoint configuration does not match."
            )

    def _validate_inputs(
        self,
        physio_input: Tensor,
        physio_valid_mask: Tensor,
        channel_names: Sequence[str],
        physio_time_mask: Tensor | None,
        physio_channel_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(physio_input, Tensor) or physio_input.ndim != 3:
            raise ValueError("physio_input must have shape [B, T, C].")
        if not physio_input.is_floating_point():
            raise TypeError("physio_input must be floating point.")
        if (
            not isinstance(physio_valid_mask, Tensor)
            or physio_valid_mask.dtype != torch.bool
            or tuple(physio_valid_mask.shape) != tuple(physio_input.shape)
            or physio_valid_mask.device != physio_input.device
        ):
            raise ValueError(
                "physio_valid_mask must be bool [B, T, C] on the input device."
            )
        batch_size, time_steps, channel_count = physio_input.shape
        if (
            batch_size <= 0
            or time_steps <= 0
            or channel_count != len(self.dense_channel_specs)
        ):
            raise ValueError("physio_input dimensions must be non-empty and match specs.")
        if isinstance(channel_names, (str, bytes)) or tuple(channel_names) != self.channel_names:
            raise ValueError("channel_names must exactly match configured spec order.")
        if physio_time_mask is None:
            time_mask = physio_valid_mask.any(dim=2)
        else:
            if (
                not isinstance(physio_time_mask, Tensor)
                or physio_time_mask.dtype != torch.bool
                or tuple(physio_time_mask.shape) != (batch_size, time_steps)
                or physio_time_mask.device != physio_input.device
            ):
                raise ValueError("physio_time_mask must be bool [B, T].")
            time_mask = physio_time_mask
            if not torch.equal(time_mask, physio_valid_mask.any(dim=2)):
                raise ValueError(
                    "physio_time_mask must equal physio_valid_mask.any(dim=2)."
                )
        if physio_channel_mask is None:
            channel_mask = physio_valid_mask.any(dim=1)
        else:
            if (
                not isinstance(physio_channel_mask, Tensor)
                or physio_channel_mask.dtype != torch.bool
                or tuple(physio_channel_mask.shape) != (batch_size, channel_count)
                or physio_channel_mask.device != physio_input.device
            ):
                raise ValueError("physio_channel_mask must be bool [B, C].")
            channel_mask = physio_channel_mask
            if not torch.equal(channel_mask, physio_valid_mask.any(dim=1)):
                raise ValueError(
                    "physio_channel_mask must equal "
                    "physio_valid_mask.any(dim=1)."
                )
        effective_mask = (
            physio_valid_mask
            & time_mask.unsqueeze(2)
            & channel_mask.unsqueeze(1)
        )
        if not bool(torch.isfinite(physio_input[effective_mask]).all()):
            raise ValueError("valid physiology positions must be finite.")
        return effective_mask, time_mask, channel_mask

    @staticmethod
    def _uniform_temporal_weights(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=dtype)
        return mask.to(dtype=dtype) / counts

    def forward(
        self,
        physio_input: Tensor,
        physio_valid_mask: Tensor,
        *,
        channel_names: Sequence[str],
        physio_time_mask: Tensor | None = None,
        physio_channel_mask: Tensor | None = None,
        physio_quality_features: Tensor | None = None,
        physio_quality_prior: Tensor | None = None,
        ecg_values: Tensor | None = None,
        ecg_valid_mask: Tensor | None = None,
        ecg_timeline_mask: Tensor | None = None,
    ) -> LightweightPhysioClassifierOutput:
        """Encode physiology ``[B,T,C]`` with ``True=valid`` masks.

        Observable quality tensors are accepted for scheduler compatibility
        but are not read by this lightweight availability-only model.
        """
        del physio_quality_features, physio_quality_prior
        effective_mask, _, _ = self._validate_inputs(
            physio_input,
            physio_valid_mask,
            channel_names,
            physio_time_mask,
            physio_channel_mask,
        )
        safe_input = torch.where(
            effective_mask,
            physio_input,
            torch.zeros_like(physio_input),
        )
        channel_features: list[Tensor] = []
        channel_statistics: list[Tensor] = []
        channel_available: list[Tensor] = []
        for channel_index, stem in enumerate(self.channel_stems):
            channel_mask = effective_mask[:, :, channel_index]
            stem_mask = channel_mask.unsqueeze(1)
            encoded = stem(
                safe_input[:, :, channel_index].unsqueeze(1),
                stem_mask,
            ).transpose(1, 2)
            statistics, available = masked_mean_std(encoded, channel_mask)
            channel_features.append(encoded)
            channel_statistics.append(statistics)
            channel_available.append(available)

        stacked_features = torch.stack(channel_features, dim=2)
        stacked_statistics = torch.stack(channel_statistics, dim=1)
        available_tensor = torch.stack(channel_available, dim=1)
        dense_valid = available_tensor.any(dim=1)
        concatenated_parts = [
            stacked_statistics.flatten(start_dim=1),
            available_tensor.to(dtype=physio_input.dtype),
        ]
        ecg_embedding: Tensor | None = None
        ecg_statistics: Tensor | None = None
        ecg_available: Tensor | None = None
        if self.ecg_enabled:
            if ecg_values is None or ecg_valid_mask is None:
                raise ValueError("enabled ECG branch requires values and valid mask.")
            assert self.ecg_encoder is not None
            ecg_embedding, ecg_statistics, ecg_available = self.ecg_encoder(
                ecg_values,
                ecg_valid_mask,
                timeline_mask=ecg_timeline_mask,
            )
            concatenated_parts.extend(
                (ecg_embedding, ecg_available.to(physio_input.dtype).unsqueeze(1))
            )
        elif any(value is not None for value in (ecg_values, ecg_valid_mask, ecg_timeline_mask)):
            raise ValueError("ECG tensors were provided to an ECG-disabled classifier.")
        sample_valid = dense_valid | (
            torch.zeros_like(dense_valid) if ecg_available is None else ecg_available
        )
        concatenated = torch.cat(concatenated_parts, dim=1)
        valid_rows = sample_valid.unsqueeze(1)
        physio_embedding = torch.where(
            valid_rows,
            self.embedding_projection(concatenated),
            torch.zeros(
                (physio_input.shape[0], self.physiology_embedding_dim),
                device=physio_input.device,
                dtype=physio_input.dtype,
            ),
        )
        arousal_logits = torch.where(
            valid_rows,
            self.arousal_head(physio_embedding),
            torch.zeros(
                (physio_input.shape[0], 2),
                device=physio_input.device,
                dtype=physio_input.dtype,
            ),
        )
        valence_logits = torch.where(
            valid_rows,
            self.valence_head(physio_embedding),
            torch.zeros_like(arousal_logits),
        )
        arousal_probabilities = torch.where(
            valid_rows,
            torch.softmax(arousal_logits, dim=-1),
            torch.zeros_like(arousal_logits),
        )
        valence_probabilities = torch.where(
            valid_rows,
            torch.softmax(valence_logits, dim=-1),
            torch.zeros_like(valence_logits),
        )
        valid_indices = sample_valid.nonzero(as_tuple=False).flatten()
        quadrant_probabilities = arousal_logits.new_zeros(
            (physio_input.shape[0], 4)
        ).index_copy(
            0,
            valid_indices,
            derive_quadrant_probabilities(
                arousal_probabilities.index_select(0, valid_indices),
                valence_probabilities.index_select(0, valid_indices),
            ),
        )
        effective_time_mask = effective_mask.any(dim=2)
        temporal_weights = self._uniform_temporal_weights(
            effective_time_mask,
            dtype=physio_input.dtype,
        )
        reliability = sample_valid.to(dtype=physio_input.dtype).unsqueeze(1)
        for value in (
            stacked_features,
            stacked_statistics,
            physio_embedding,
            arousal_logits,
            valence_logits,
            arousal_probabilities,
            valence_probabilities,
            quadrant_probabilities,
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError("lightweight physiology output contains NaN or Inf.")
        return LightweightPhysioClassifierOutput(
            arousal_logits=arousal_logits,
            valence_logits=valence_logits,
            arousal_probabilities=arousal_probabilities,
            valence_probabilities=valence_probabilities,
            quadrant_probabilities=quadrant_probabilities,
            physio_embedding=physio_embedding,
            channel_features=stacked_features,
            channel_statistics=stacked_statistics,
            channel_available=available_tensor,
            temporal_attention_weights=temporal_weights,
            sample_valid=sample_valid,
            reliability=reliability,
            reliability_score_type=_RELIABILITY_SCORE_TYPE,
            ecg_embedding=ecg_embedding,
            ecg_statistics=ecg_statistics,
            ecg_available=ecg_available,
        )


__all__ = [
    "MultiScaleDilatedConv1dStem",
    "PHYSIOLOGY_ENCODER_MODES",
    "LightweightPhysioClassifierOutput",
    "LightweightPhysioEmotionClassifier",
    "LightweightEcgEncoder",
    "ECG_ENCODER_TYPE",
]
