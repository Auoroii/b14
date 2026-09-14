"""Validated output contract for V4.2 shared dynamic fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from emotion_model.multimodal.routing import (
    ModalityAvailabilityMasks,
    ScheduledModalityOutputs,
)


@dataclass(frozen=True)
class MultimodalFusionOutput:
    """Validated shared-fusion result for one logical batch.

    ``fused_embedding`` and both projected modality embeddings have shape
    ``[B, F]``. Gate diagnostics and ``modality_weights`` have shape ``[B, 2]``
    in ``[speech, physiology]`` order. ``sample_valid`` is boolean ``[B]`` and
    equals the union of modality availability. Unavailable modality entries
    and fully unavailable rows are exact-zero sentinels.
    """

    fused_embedding: Tensor
    projected_speech_embedding: Tensor
    projected_physiology_embedding: Tensor
    base_gate_scores: Tensor
    learned_logit_corrections: Tensor
    modality_weights: Tensor
    sample_valid: Tensor
    availability: ModalityAvailabilityMasks
    scheduled_outputs: ScheduledModalityOutputs
    fusion_policy: str = "full_window_shared_embedding_dynamic_softmax"

    def __post_init__(self) -> None:
        if not isinstance(self.availability, ModalityAvailabilityMasks):
            raise TypeError("availability must be ModalityAvailabilityMasks.")
        if not isinstance(self.scheduled_outputs, ScheduledModalityOutputs):
            raise TypeError("scheduled_outputs must be ScheduledModalityOutputs.")
        if self.availability is not self.scheduled_outputs.availability:
            raise ValueError(
                "availability must be the exact scheduled_outputs availability object."
            )
        if not isinstance(self.fusion_policy, str) or not self.fusion_policy.strip():
            raise ValueError("fusion_policy must be a non-empty string.")

        fused = self.fused_embedding
        if not isinstance(fused, Tensor) or not fused.is_floating_point():
            raise TypeError("fused_embedding must be a floating tensor [B, F].")
        if fused.ndim != 2 or fused.shape[0] <= 0 or fused.shape[1] <= 0:
            raise ValueError("fused_embedding must have non-empty shape [B, F].")
        batch_size, fusion_dim = fused.shape
        float_fields = (
            (
                "projected_speech_embedding",
                self.projected_speech_embedding,
                (batch_size, fusion_dim),
            ),
            (
                "projected_physiology_embedding",
                self.projected_physiology_embedding,
                (batch_size, fusion_dim),
            ),
            ("base_gate_scores", self.base_gate_scores, (batch_size, 2)),
            (
                "learned_logit_corrections",
                self.learned_logit_corrections,
                (batch_size, 2),
            ),
            ("modality_weights", self.modality_weights, (batch_size, 2)),
        )
        for name, value, expected_shape in float_fields:
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating tensor.")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"received {tuple(value.shape)}."
                )
            if value.dtype != fused.dtype or value.device != fused.device:
                raise ValueError(
                    "all fusion floating outputs must share dtype and device."
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")
        if not bool(torch.isfinite(fused).all()):
            raise ValueError("fused_embedding must contain only finite values.")

        sample_valid = self.sample_valid
        if (
            not isinstance(sample_valid, Tensor)
            or sample_valid.dtype != torch.bool
            or tuple(sample_valid.shape) != (batch_size,)
        ):
            raise ValueError("sample_valid must be bool with shape [B].")
        expected_valid = (
            self.availability.speech_available
            | self.availability.physiology_available
        )
        if sample_valid.device != expected_valid.device or not torch.equal(
            sample_valid, expected_valid
        ):
            raise ValueError(
                "sample_valid must equal speech OR physiology availability."
            )

        modality_available = torch.stack(
            (
                self.availability.speech_available,
                self.availability.physiology_available,
            ),
            dim=1,
        ).to(device=fused.device)
        if not bool(
            (self.base_gate_scores >= 0).all()
            and (self.base_gate_scores <= 1).all()
        ):
            raise ValueError("base_gate_scores must lie in [0, 1].")
        for name, values in (
            ("base_gate_scores", self.base_gate_scores),
            ("learned_logit_corrections", self.learned_logit_corrections),
            ("modality_weights", self.modality_weights),
        ):
            unavailable = values[~modality_available]
            if not torch.equal(unavailable, torch.zeros_like(unavailable)):
                raise ValueError(f"{name} unavailable entries must be exact zero.")
        for name, values, available in (
            (
                "projected_speech_embedding",
                self.projected_speech_embedding,
                modality_available[:, 0],
            ),
            (
                "projected_physiology_embedding",
                self.projected_physiology_embedding,
                modality_available[:, 1],
            ),
        ):
            unavailable = values[~available]
            if not torch.equal(unavailable, torch.zeros_like(unavailable)):
                raise ValueError(f"{name} unavailable rows must be exact zero.")
        if not bool(
            (self.modality_weights >= 0).all()
            and (self.modality_weights <= 1).all()
        ):
            raise ValueError("modality_weights must lie in [0, 1].")
        local_sample_valid = sample_valid.to(device=fused.device)
        valid_weight_sums = self.modality_weights[local_sample_valid].sum(dim=1)
        if not torch.allclose(
            valid_weight_sums,
            torch.ones_like(valid_weight_sums),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("valid modality_weights rows must sum to one.")
        expected_fused = (
            self.modality_weights[:, 0:1] * self.projected_speech_embedding
            + self.modality_weights[:, 1:2]
            * self.projected_physiology_embedding
        )
        expected_fused = torch.where(
            local_sample_valid.unsqueeze(1),
            expected_fused,
            torch.zeros_like(expected_fused),
        )
        if not torch.allclose(fused, expected_fused, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("fused_embedding must equal the modality weighted sum.")


__all__ = ["MultimodalFusionOutput"]
