"""Shared learned dynamic fusion for full-window K-EmoCon variants."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from emotion_model.common import safe_masked_softmax
from emotion_model.multimodal.fusion_output import (
    MultimodalFusionOutput,
)
from emotion_model.multimodal.routing import ScheduledModalityOutputs

_FUSION_POLICY = "full_window_shared_embedding_dynamic_softmax"
_STATE_VERSION = 1


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _dropout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dropout must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("dropout must be finite and lie in [0, 1).")
    return result


def _projection(input_dim: int, fusion_dim: int, dropout: float) -> nn.Sequential:
    normalization: nn.Module = (
        nn.Identity() if input_dim == 1 else nn.LayerNorm(input_dim)
    )
    return nn.Sequential(
        normalization,
        nn.Linear(input_dim, fusion_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FullWindowDynamicMultimodalFusion(nn.Module):
    """Learn one availability-masked softmax over two projected embeddings.

    Args:
        speech_embedding_dim: Speech input width ``Ds``.
        physiology_embedding_dim: Physiology input width ``Dp``.
        fusion_dim: Shared fused width ``F``.
        gate_hidden_dim: Hidden width of the modality-logit MLP.
        dropout: Projection and gate dropout probability in ``[0, 1)``.

    Forward consumes :class:`ScheduledModalityOutputs` containing speech and
    physiology embeddings ``[B, Ds]`` and ``[B, Dp]``. It returns one shared
    fused embedding ``[B, F]`` and weights ``[B, 2]`` in
    ``[speech, physiology]`` order. The MLP sees only the concatenated projected
    embeddings. Availability is used only by masked softmax: an unavailable
    modality receives exact-zero weight, a sole modality receives exact-one
    weight, and a fully unavailable row remains finite, all-zero, and invalid.
    Participant speech activity diagnostics are not accepted by this API.
    """

    def __init__(
        self,
        speech_embedding_dim: int,
        physiology_embedding_dim: int,
        fusion_dim: int,
        *,
        gate_hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.speech_embedding_dim = _positive_integer(
            speech_embedding_dim,
            name="speech_embedding_dim",
        )
        self.physiology_embedding_dim = _positive_integer(
            physiology_embedding_dim,
            name="physiology_embedding_dim",
        )
        self.fusion_dim = _positive_integer(fusion_dim, name="fusion_dim")
        self.gate_hidden_dim = _positive_integer(
            gate_hidden_dim,
            name="gate_hidden_dim",
        )
        self.dropout = _dropout(dropout)
        self.fusion_policy = _FUSION_POLICY
        self.speech_projection = _projection(
            self.speech_embedding_dim,
            self.fusion_dim,
            self.dropout,
        )
        self.physiology_projection = _projection(
            self.physiology_embedding_dim,
            self.fusion_dim,
            self.dropout,
        )
        gate_input_dim = 2 * self.fusion_dim
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_hidden_dim, 2),
        )

    def get_extra_state(self) -> dict[str, object]:
        """Return the complete shared dynamic-fusion checkpoint fingerprint."""

        return {
            "state_version": _STATE_VERSION,
            "variant": "full_window_dynamic",
            "speech_embedding_dim": self.speech_embedding_dim,
            "physiology_embedding_dim": self.physiology_embedding_dim,
            "fusion_dim": self.fusion_dim,
            "gate_hidden_dim": self.gate_hidden_dim,
            "dropout": self.dropout,
            "fusion_policy": self.fusion_policy,
            "gate_input": "concatenated_projected_embeddings_only",
            "modality_order": ("speech", "physiology"),
            "task_fusion_policy": "one_shared_fused_embedding",
            "missing_policy": "availability_masked_softmax",
            "speech_activity_role": "not_an_input",
        }

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoints from a different fusion configuration."""

        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "Full-window dynamic fusion checkpoint configuration does not match."
            )

    def _reference(self) -> Tensor:
        linear = self.speech_projection[1]
        if not isinstance(linear, nn.Linear):
            raise RuntimeError("speech projection Linear is missing.")
        return linear.weight

    def forward(
        self,
        scheduled_outputs: ScheduledModalityOutputs,
    ) -> MultimodalFusionOutput:
        """Fuse scheduled ``[B,Ds]``/``[B,Dp]`` embeddings into one ``[B,F]``."""

        if not isinstance(scheduled_outputs, ScheduledModalityOutputs):
            raise TypeError("scheduled_outputs must be ScheduledModalityOutputs.")
        availability = scheduled_outputs.availability
        batch_size = len(scheduled_outputs.batch.records)
        reference = self._reference()
        speech_available = availability.speech_available.to(device=reference.device)
        physiology_available = availability.physiology_available.to(
            device=reference.device
        )
        modality_available = torch.stack(
            (speech_available, physiology_available),
            dim=1,
        )

        if scheduled_outputs.speech is None:
            speech_embedding = reference.new_zeros(
                (batch_size, self.speech_embedding_dim)
            )
        else:
            speech_embedding = scheduled_outputs.speech.embedding
        if scheduled_outputs.physiology is None:
            physiology_embedding = reference.new_zeros(
                (batch_size, self.physiology_embedding_dim)
            )
        else:
            physiology_embedding = scheduled_outputs.physiology.embedding
        if tuple(speech_embedding.shape) != (
            batch_size,
            self.speech_embedding_dim,
        ) or tuple(physiology_embedding.shape) != (
            batch_size,
            self.physiology_embedding_dim,
        ):
            raise ValueError("scheduled embedding dimensions do not match fusion.")
        if (
            speech_embedding.dtype != reference.dtype
            or speech_embedding.device != reference.device
            or physiology_embedding.dtype != reference.dtype
            or physiology_embedding.device != reference.device
        ):
            raise ValueError("scheduled embeddings must match fusion dtype/device.")
        if not bool(torch.isfinite(speech_embedding[speech_available]).all()):
            raise ValueError("available speech embeddings must be finite.")
        if not bool(torch.isfinite(physiology_embedding[physiology_available]).all()):
            raise ValueError("available physiology embeddings must be finite.")

        safe_speech = torch.where(
            speech_available.unsqueeze(1),
            speech_embedding,
            torch.zeros_like(speech_embedding),
        )
        safe_physiology = torch.where(
            physiology_available.unsqueeze(1),
            physiology_embedding,
            torch.zeros_like(physiology_embedding),
        )
        projected_speech = torch.where(
            speech_available.unsqueeze(1),
            self.speech_projection(safe_speech),
            reference.new_zeros((batch_size, self.fusion_dim)),
        )
        projected_physiology = torch.where(
            physiology_available.unsqueeze(1),
            self.physiology_projection(safe_physiology),
            reference.new_zeros((batch_size, self.fusion_dim)),
        )
        gate_input = torch.cat((projected_speech, projected_physiology), dim=1)
        modality_logits = self.modality_gate(gate_input)
        safe_logits = torch.where(
            modality_available,
            modality_logits,
            torch.zeros_like(modality_logits),
        )
        modality_weights = safe_masked_softmax(
            modality_logits,
            modality_available,
            dim=1,
        )
        fused = (
            modality_weights[:, 0:1] * projected_speech
            + modality_weights[:, 1:2] * projected_physiology
        )
        sample_valid = (
            availability.speech_available | availability.physiology_available
        )
        fused = torch.where(
            sample_valid.to(device=reference.device).unsqueeze(1),
            fused,
            torch.zeros_like(fused),
        )
        base_gate_scores = modality_available.to(dtype=reference.dtype)
        return MultimodalFusionOutput(
            fused_embedding=fused,
            projected_speech_embedding=projected_speech,
            projected_physiology_embedding=projected_physiology,
            base_gate_scores=base_gate_scores,
            learned_logit_corrections=safe_logits,
            modality_weights=modality_weights,
            sample_valid=sample_valid,
            availability=availability,
            scheduled_outputs=scheduled_outputs,
            fusion_policy=self.fusion_policy,
        )


__all__ = ["FullWindowDynamicMultimodalFusion"]
