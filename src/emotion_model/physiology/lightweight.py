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
_STATE_VERSION = 1


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


class LightweightPhysioEmotionClassifier(nn.Module):
    """Encode each named physiology channel with an independent tiny stem.

    Args:
        channel_specs: Ordered semantic channel specifications.
        stem_channels: Exactly two positive Conv1D widths.
        physiology_embedding_dim: Output embedding width.
        dropout: Statistics-projection dropout in ``[0, 1)``.
        stem_dropout: Per-channel stem dropout in ``[0, 1)``.

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
        self.channel_specs = specs
        self.channel_names = tuple(spec.name for spec in specs)
        self.stem_channels = (first_dim, second_dim)
        self.physiology_embedding_dim = _positive_integer(
            physiology_embedding_dim,
            name="physiology_embedding_dim",
        )
        self.dropout = _dropout(dropout, name="dropout")
        self.stem_dropout = _dropout(stem_dropout, name="stem_dropout")
        self.channel_stems = nn.ModuleList(
            _IndependentChannelStem(first_dim, second_dim, self.stem_dropout)
            for _ in specs
        )
        statistics_dim = len(specs) * (2 * second_dim) + len(specs)
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
        return {
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
            "reliability_score_type": _RELIABILITY_SCORE_TYPE,
        }

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
        if batch_size <= 0 or time_steps <= 0 or channel_count != len(self.channel_specs):
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
        sample_valid = available_tensor.any(dim=1)
        concatenated = torch.cat(
            (
                stacked_statistics.flatten(start_dim=1),
                available_tensor.to(dtype=physio_input.dtype),
            ),
            dim=1,
        )
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
        )


__all__ = [
    "LightweightPhysioClassifierOutput",
    "LightweightPhysioEmotionClassifier",
]
