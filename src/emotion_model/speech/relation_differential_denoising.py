"""Noise-conditioned relation-differential feature refinement for speech."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emotion_model.common import apply_query_mask, safe_masked_softmax, validate_sequence_mask

_STATE_VERSION = 1


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and lie in (0, 1).")
    return result


@dataclass(frozen=True)
class RelationDifferentialDenoisingOutput:
    """Feature-space relation refinement diagnostics.

    ``denoised_emotion_sequence`` and ``delta`` have shape ``[B,T,Dw]``;
    ``differential_response`` and ``differential_gate`` have shape
    ``[B,T,Dd]``; and ``differential_lambda`` is a scalar tensor. Invalid
    time positions are exact zero in every time-indexed output.
    """

    denoised_emotion_sequence: Tensor
    differential_response: Tensor
    differential_gate: Tensor
    delta: Tensor
    differential_lambda: Tensor


class NoiseConditionedRelationDifferentialDenoiser(nn.Module):
    """Refine an emotion sequence through gated low-dimensional relation change.

    Args:
        wavlm_hidden_dim: Input and residual output width ``Dw``.
        differential_dim: Low-dimensional relation width ``Dd``.
        num_heads: Number of attention heads; ``Dd`` must be divisible by it.
        lambda_init: Initial bounded subtraction coefficient in ``(0,1)``.
        residual_scale: Fixed residual multiplier in ``(0,1]``.
        condition_gate_on_noise: Whether local H1--H2 features condition the gate.

    Forward accepts raw H9--H12 emotion and H1--H2 acoustic-condition
    sequences ``[B,T,Dw]`` plus a boolean validity mask ``[B,T]`` and returns
    :class:`RelationDifferentialDenoisingOutput`. This module refines features;
    it never reconstructs or outputs a waveform.
    """

    def __init__(
        self,
        wavlm_hidden_dim: int,
        differential_dim: int = 32,
        num_heads: int = 4,
        *,
        lambda_init: float = 0.5,
        residual_scale: float = 0.1,
        condition_gate_on_noise: bool = True,
    ) -> None:
        super().__init__()
        self.wavlm_hidden_dim = _positive_integer(
            wavlm_hidden_dim,
            name="wavlm_hidden_dim",
        )
        self.differential_dim = _positive_integer(
            differential_dim,
            name="differential_dim",
        )
        self.num_heads = _positive_integer(num_heads, name="num_heads")
        if self.differential_dim % self.num_heads != 0:
            raise ValueError("differential_dim must be divisible by num_heads.")
        self.head_dim = self.differential_dim // self.num_heads
        self.lambda_init = _probability(lambda_init, name="lambda_init")
        if isinstance(residual_scale, bool) or not isinstance(
            residual_scale,
            (int, float),
        ):
            raise TypeError("residual_scale must be a real number, not bool.")
        self.residual_scale = float(residual_scale)
        if not math.isfinite(self.residual_scale) or not (
            0.0 < self.residual_scale <= 1.0
        ):
            raise ValueError("residual_scale must be finite and lie in (0, 1].")
        if not isinstance(condition_gate_on_noise, bool):
            raise TypeError("condition_gate_on_noise must be boolean.")
        self.condition_gate_on_noise = condition_gate_on_noise

        self.emotion_projection = nn.Sequential(
            nn.LayerNorm(self.wavlm_hidden_dim),
            nn.Linear(self.wavlm_hidden_dim, self.differential_dim),
        )
        self.noise_frame_projection: nn.Sequential | None = None
        if self.condition_gate_on_noise:
            self.noise_frame_projection = nn.Sequential(
                nn.LayerNorm(self.wavlm_hidden_dim),
                nn.Linear(self.wavlm_hidden_dim, self.differential_dim),
            )
        self.qkv_projection = nn.Linear(
            self.differential_dim,
            3 * self.differential_dim,
        )
        self.output_projection = nn.Linear(
            self.differential_dim,
            self.differential_dim,
        )
        gate_input_dim = self.differential_dim * (
            2 if self.condition_gate_on_noise else 1
        )
        self.response_norm = nn.LayerNorm(self.differential_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, self.differential_dim),
            nn.GELU(),
            nn.Linear(self.differential_dim, self.differential_dim),
        )
        self.up_projection = nn.Linear(
            self.differential_dim,
            self.wavlm_hidden_dim,
        )
        nn.init.normal_(self.up_projection.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.up_projection.bias)
        lambda_logit = math.log(self.lambda_init / (1.0 - self.lambda_init))
        self.lambda_logit = nn.Parameter(torch.tensor(lambda_logit))

    @property
    def differential_lambda(self) -> Tensor:
        """Return the learned scalar relation coefficient with shape ``[]``."""

        return torch.sigmoid(self.lambda_logit)

    def get_extra_state(self) -> dict[str, object]:
        """Return architecture metadata without tensor payloads."""

        return {
            "state_version": _STATE_VERSION,
            "wavlm_hidden_dim": self.wavlm_hidden_dim,
            "differential_dim": self.differential_dim,
            "num_heads": self.num_heads,
            "lambda_parameterization": "sigmoid_logit",
            "lambda_init": self.lambda_init,
            "residual_scale": self.residual_scale,
            "condition_gate_on_noise": self.condition_gate_on_noise,
        }

    def set_extra_state(self, state: object) -> None:
        """Reject a checkpoint whose relation architecture differs."""

        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "Relation-differential checkpoint configuration does not match."
            )

    @staticmethod
    def _shift_previous_valid(keys: Tensor, valid_mask: Tensor) -> Tensor:
        """Shift keys ``[B,H,T,Dh]`` along only the valid ``[B,T]`` sequence.

        The first valid key references itself; every later valid key references
        the preceding valid key. Invalid positions are exact zero and never
        become a reference for a later valid position.
        """

        if keys.ndim != 4:
            raise ValueError("keys must have shape [B,H,T,Dh].")
        if valid_mask.dtype != torch.bool or valid_mask.ndim != 2:
            raise ValueError("valid_mask must be bool with shape [B,T].")
        if (
            keys.shape[0] != valid_mask.shape[0]
            or keys.shape[2] != valid_mask.shape[1]
            or keys.device != valid_mask.device
        ):
            raise ValueError("keys and valid_mask batch/time/device must match.")

        previous = torch.zeros_like(keys[:, :, 0, :])
        seen = torch.zeros(
            valid_mask.shape[0],
            dtype=torch.bool,
            device=valid_mask.device,
        )
        references: list[Tensor] = []
        for index in range(keys.shape[2]):
            current = keys[:, :, index, :]
            is_valid = valid_mask[:, index]
            use_previous = (is_valid & seen).view(-1, 1, 1)
            reference = torch.where(use_previous, previous, current)
            reference = torch.where(
                is_valid.view(-1, 1, 1),
                reference,
                torch.zeros_like(reference),
            )
            references.append(reference)
            previous = torch.where(
                is_valid.view(-1, 1, 1),
                current,
                previous,
            )
            seen = seen | is_valid
        return torch.stack(references, dim=2)

    def forward(
        self,
        emotion_sequence: Tensor,
        noise_sequence: Tensor,
        valid_mask: Tensor,
    ) -> RelationDifferentialDenoisingOutput:
        """Refine sequences ``[B,T,Dw]`` under boolean mask ``[B,T]``.

        Returns feature-space tensors described by
        :class:`RelationDifferentialDenoisingOutput`. Rows with fewer than two
        valid frames bypass the differential branch exactly.
        """

        if not isinstance(emotion_sequence, Tensor) or not isinstance(
            noise_sequence,
            Tensor,
        ):
            raise TypeError("emotion_sequence and noise_sequence must be Tensors.")
        if emotion_sequence.ndim != 3 or emotion_sequence.shape[-1] != (
            self.wavlm_hidden_dim
        ):
            raise ValueError("emotion_sequence must have shape [B,T,Dw].")
        if tuple(noise_sequence.shape) != tuple(emotion_sequence.shape):
            raise ValueError("noise_sequence must match emotion_sequence shape.")
        if not emotion_sequence.is_floating_point() or not noise_sequence.is_floating_point():
            raise TypeError("emotion_sequence and noise_sequence must be floating point.")
        if (
            noise_sequence.device != emotion_sequence.device
            or noise_sequence.dtype != emotion_sequence.dtype
        ):
            raise ValueError("emotion_sequence and noise_sequence must share dtype/device.")
        validate_sequence_mask(emotion_sequence, valid_mask)

        frame_mask = valid_mask.unsqueeze(-1)
        clean_emotion = torch.where(
            frame_mask,
            emotion_sequence,
            torch.zeros_like(emotion_sequence),
        )
        clean_noise = torch.where(
            frame_mask,
            noise_sequence,
            torch.zeros_like(noise_sequence),
        )
        projected_emotion = apply_query_mask(
            self.emotion_projection(clean_emotion),
            valid_mask,
        )
        query, key, value = self.qkv_projection(projected_emotion).chunk(3, dim=-1)
        query = query.reshape(
            query.shape[0],
            query.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            key.shape[0],
            key.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        reference_key = self._shift_previous_valid(key, valid_mask)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        reference_logits = (
            torch.matmul(query.float(), reference_key.float().transpose(-2, -1))
            * scale
        )
        key_mask = valid_mask[:, None, None, :]
        attention = safe_masked_softmax(logits, key_mask, dim=-1).to(value.dtype)
        reference_attention = safe_masked_softmax(
            reference_logits,
            key_mask,
            dim=-1,
        ).to(value.dtype)
        head_response = torch.matmul(
            attention - self.differential_lambda.to(dtype=value.dtype) * reference_attention,
            value,
        )
        head_response = torch.where(
            valid_mask[:, None, :, None],
            head_response,
            torch.zeros_like(head_response),
        )
        response = head_response.transpose(1, 2).reshape(
            emotion_sequence.shape[0],
            emotion_sequence.shape[1],
            self.differential_dim,
        )
        response = apply_query_mask(self.output_projection(response), valid_mask)

        multi_frame = (valid_mask.sum(dim=1) > 1).view(-1, 1, 1)
        response = torch.where(multi_frame, response, torch.zeros_like(response))
        normalized_response = self.response_norm(response)
        if self.noise_frame_projection is None:
            gate_input = normalized_response
        else:
            projected_noise = apply_query_mask(
                self.noise_frame_projection(clean_noise),
                valid_mask,
            )
            gate_input = torch.cat((normalized_response, projected_noise), dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        gate = torch.where(frame_mask & multi_frame, gate, torch.zeros_like(gate))
        delta = apply_query_mask(self.up_projection(gate * response), valid_mask)
        delta = torch.where(multi_frame, delta, torch.zeros_like(delta))
        denoised = apply_query_mask(
            clean_emotion + self.residual_scale * delta,
            valid_mask,
        )

        for name, value_tensor in (
            ("denoised_emotion_sequence", denoised),
            ("differential_response", response),
            ("differential_gate", gate),
            ("delta", delta),
            ("differential_lambda", self.differential_lambda),
        ):
            if not bool(torch.isfinite(value_tensor).all()):
                raise RuntimeError(f"{name} contains NaN or Inf.")
        return RelationDifferentialDenoisingOutput(
            denoised_emotion_sequence=denoised,
            differential_response=response,
            differential_gate=gate,
            delta=delta,
            differential_lambda=self.differential_lambda,
        )


__all__ = [
    "NoiseConditionedRelationDifferentialDenoiser",
    "RelationDifferentialDenoisingOutput",
]
