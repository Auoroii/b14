"""Lightweight fixed-layer speech and acoustic-condition classifier."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emotion_model.common import derive_quadrant_probabilities, masked_mean_std
from emotion_model.speech.emotion_layer_aggregation import EmotionLayerAggregation
from emotion_model.speech.relation_differential_denoising import (
    NoiseConditionedRelationDifferentialDenoiser,
)
from emotion_model.speech.wavlm_encoder import (
    WavLMEncoder,
)

_EMOTION_LAYER_INDICES = (8, 9, 10, 11)
_NOISE_LAYER_INDICES = (0, 1)
_RELIABILITY_SCORE_TYPE = "availability_indicator"
_STATE_VERSION = 1
_MODEL_VARIANT = (
    "lightweight_shared_dynamic_relation_differential_full_window"
)


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


@dataclass(frozen=True)
class LightweightSpeechClassifierOutput:
    """Outputs of the lightweight speech classifier.

    All embeddings are floating tensors with leading shape ``[B, ...]``.
    Binary logits/probabilities are ``[B, 2]``, derived quadrant
    probabilities are ``[B, 4]``, layer weights are ``[B, 12]``, temporal
    diagnostics are ``[B, T_s]``, activity ratio is ``[B,1]``, and
    ``sample_valid`` is boolean ``[B]``. Feature attention remains the WavLM
    valid-prefix mask.
    ``noise_embedding`` represents only a continuous acoustic/noise condition.
    The required relation refinement returns a denoised sequence
    ``[B,T_s,D_w]``, response/gate tensors are ``[B,T_s,D_d]``, and lambda is
    scalar.
    """

    arousal_logits: Tensor
    valence_logits: Tensor
    arousal_probabilities: Tensor
    valence_probabilities: Tensor
    quadrant_probabilities: Tensor
    speech_embedding: Tensor
    base_speech_embedding: Tensor
    noise_embedding: Tensor
    emotion_sequence: Tensor
    noise_sequence: Tensor
    gamma_bounded: Tensor
    beta_bounded: Tensor
    emotion_layer_weights: Tensor
    noise_layer_weights: Tensor
    temporal_attention_weights: Tensor
    feature_attention_mask: Tensor
    speech_activity_ratio: Tensor
    reliability: Tensor
    sample_valid: Tensor
    reliability_score_type: str
    denoised_emotion_sequence: Tensor
    differential_response: Tensor
    differential_gate: Tensor
    differential_lambda: Tensor
    quadrant_logits: Tensor | None = None


class LightweightNoiseConditionedSpeechClassifier(nn.Module):
    """Pool fixed WavLM layers and apply bounded acoustic-condition FiLM.

    Args:
        wavlm_encoder: Fully frozen or last-N partially trainable 12-layer
            encoder. One forward produces all layer sequences
            ``[B, T_s, D_w]``.
        speech_embedding_dim: Output speech embedding width.
        noise_embedding_dim: Continuous acoustic/noise-condition width.
        relation_denoiser: Required V4.2 feature-space relation refinement.
        emotion_layer_aggregation: H9--H12 fixed or learned aggregation.
        film_scale: Positive finite bound in ``(0, 1]`` for both FiLM terms.
        dropout: Projection dropout probability in ``[0, 1)``.

    Forward accepts waveform ``[B, L]`` and boolean valid-prefix mask
    ``[B, L]`` and returns :class:`LightweightSpeechClassifierOutput`.
    """

    def __init__(
        self,
        wavlm_encoder: WavLMEncoder,
        speech_embedding_dim: int = 128,
        noise_embedding_dim: int = 16,
        *,
        relation_denoiser: NoiseConditionedRelationDifferentialDenoiser,
        emotion_layer_aggregation: EmotionLayerAggregation | None = None,
        film_scale: float = 0.1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not isinstance(wavlm_encoder, WavLMEncoder):
            raise TypeError("wavlm_encoder must be WavLMEncoder.")
        if wavlm_encoder.expected_num_hidden_layers != 12:
            raise ValueError("lightweight speech requires exactly 12 WavLM layers.")
        if emotion_layer_aggregation is None:
            emotion_layer_aggregation = EmotionLayerAggregation("fixed_mean")
        if not isinstance(emotion_layer_aggregation, EmotionLayerAggregation):
            raise TypeError(
                "emotion_layer_aggregation must be EmotionLayerAggregation."
            )
        if emotion_layer_aggregation.mode not in {
            "fixed_mean",
            "learnable_weighted",
        }:
            raise ValueError(
                "lightweight speech aggregation must be 'fixed_mean' or "
                "'learnable_weighted'."
            )
        self.wavlm_encoder = wavlm_encoder
        self.emotion_layer_aggregation = emotion_layer_aggregation
        self.wavlm_hidden_dim = int(wavlm_encoder.model.config.hidden_size)
        if not isinstance(
            relation_denoiser,
            NoiseConditionedRelationDifferentialDenoiser,
        ):
            raise TypeError(
                "relation_denoiser must be "
                "NoiseConditionedRelationDifferentialDenoiser."
            )
        if relation_denoiser.wavlm_hidden_dim != self.wavlm_hidden_dim:
            raise ValueError(
                "relation_denoiser wavlm_hidden_dim must match the WavLM encoder."
            )
        self.relation_denoiser = relation_denoiser
        self.speech_embedding_dim = _positive_integer(
            speech_embedding_dim,
            name="speech_embedding_dim",
        )
        self.noise_embedding_dim = _positive_integer(
            noise_embedding_dim,
            name="noise_embedding_dim",
        )
        if isinstance(film_scale, bool) or not isinstance(film_scale, (int, float)):
            raise TypeError("film_scale must be a real number, not bool.")
        self.film_scale = float(film_scale)
        if not math.isfinite(self.film_scale) or not 0.0 < self.film_scale <= 1.0:
            raise ValueError("film_scale must be finite and lie in (0, 1].")
        self.dropout = _dropout(dropout)
        self.variant = _MODEL_VARIANT
        self.full_window_activity_diagnostics_only = True

        statistics_dim = 2 * self.wavlm_hidden_dim
        self.speech_projection = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Linear(statistics_dim, self.speech_embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.noise_projection = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Linear(statistics_dim, self.noise_embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.film = nn.Linear(
            self.noise_embedding_dim,
            2 * self.speech_embedding_dim,
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.arousal_head = nn.Linear(self.speech_embedding_dim, 2)
        self.valence_head = nn.Linear(self.speech_embedding_dim, 2)

        emotion_weights = torch.zeros(12, dtype=torch.float32)
        emotion_weights[list(_EMOTION_LAYER_INDICES)] = 0.25
        noise_weights = torch.zeros(12, dtype=torch.float32)
        noise_weights[list(_NOISE_LAYER_INDICES)] = 0.5
        self.register_buffer("_emotion_layer_weights", emotion_weights)
        self.register_buffer("_noise_layer_weights", noise_weights)

    def get_extra_state(self) -> dict[str, object]:
        """Return all architecture and fixed-selection checkpoint fields."""
        state: dict[str, object] = {
            "state_version": _STATE_VERSION,
            "variant": self.variant,
            "wavlm_hidden_dim": self.wavlm_hidden_dim,
            "speech_embedding_dim": self.speech_embedding_dim,
            "noise_embedding_dim": self.noise_embedding_dim,
            "film_scale": self.film_scale,
            "dropout": self.dropout,
            "emotion_layer_indices": _EMOTION_LAYER_INDICES,
            "noise_layer_indices": _NOISE_LAYER_INDICES,
            "reliability_score_type": _RELIABILITY_SCORE_TYPE,
        }
        state["speech_temporal_mask_policy"] = "wavlm_feature_attention_only"
        state["speech_activity_role"] = "diagnostic_only"
        state["relation_differential"] = {
            "enabled": True,
            "differential_dim": self.relation_denoiser.differential_dim,
            "heads": self.relation_denoiser.num_heads,
            "lambda_init": self.relation_denoiser.lambda_init,
            "lambda_parameterization": "sigmoid_logit",
            "residual_scale": self.relation_denoiser.residual_scale,
            "condition_gate_on_noise": (
                self.relation_denoiser.condition_gate_on_noise
            ),
        }
        return state

    def set_extra_state(self, state: object) -> None:
        """Reject a checkpoint whose complete architecture differs."""
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "Lightweight speech checkpoint configuration does not match."
            )

    @staticmethod
    def _uniform_temporal_weights(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=dtype)
        return mask.to(dtype=dtype) / counts

    def forward(
        self,
        waveform: Tensor,
        speech_attention_mask: Tensor,
        *,
        speech_activity_mask: Tensor | None = None,
        sample_rate: int | None = None,
    ) -> LightweightSpeechClassifierOutput:
        """Classify waveform ``[B,L]`` with one WavLM forward.

        Args:
            waveform: Floating waveform tensor ``[B, L]``.
            speech_attention_mask: Boolean mask ``[B, L]``, ``True=valid``.
            speech_activity_mask: Optional boolean participant-activity mask
                ``[B,L]``. Internal holes are allowed. ``None`` falls back to
                ``speech_attention_mask`` for backward compatibility. In the
                full-window variant this mask affects only the returned activity
                ratio diagnostic, never temporal feature validity.
            sample_rate: Optional rate required to equal the encoder rate.

        Returns:
            Lightweight output with speech ``[B, Ds]``, acoustic condition
            ``[B, Dn]``, bounded FiLM terms ``[B, Ds]``, and predictions.
        """
        if speech_activity_mask is None:
            waveform_activity_mask = speech_attention_mask
        else:
            if not isinstance(speech_activity_mask, Tensor):
                raise TypeError("speech_activity_mask must be a Tensor or None.")
            if (
                speech_activity_mask.dtype != torch.bool
                or tuple(speech_activity_mask.shape) != tuple(waveform.shape)
                or speech_activity_mask.device != waveform.device
            ):
                raise ValueError(
                    "speech_activity_mask must be bool with waveform shape/device."
                )
            if bool((speech_activity_mask & ~speech_attention_mask).any()):
                raise ValueError(
                    "speech_activity_mask must be a subset of "
                    "speech_attention_mask."
                )
            waveform_activity_mask = speech_activity_mask
        encoded = self.wavlm_encoder(
            waveform,
            speech_attention_mask,
            sample_rate=sample_rate,
        )
        hidden_states = encoded.hidden_states
        if len(hidden_states) != 12:
            raise RuntimeError("WavLM must return exactly 12 Transformer layers.")
        feature_mask = encoded.feature_attention_mask
        if bool((~feature_mask.any(dim=1)).any()):
            raise RuntimeError(
                "speech row passed routing despite zero valid WavLM feature frames."
            )
        emotion_sequence = self.emotion_layer_aggregation(hidden_states)
        noise_sequence = torch.stack(
            tuple(hidden_states[index] for index in _NOISE_LAYER_INDICES),
            dim=0,
        ).mean(dim=0)
        emotion_sequence_mask = feature_mask.unsqueeze(-1)
        noise_feature_mask = feature_mask
        noise_sequence_mask = noise_feature_mask.unsqueeze(-1)
        emotion_sequence = torch.where(
            emotion_sequence_mask,
            emotion_sequence,
            torch.zeros_like(emotion_sequence),
        )
        noise_sequence = torch.where(
            noise_sequence_mask,
            noise_sequence,
            torch.zeros_like(noise_sequence),
        )
        relation_output = self.relation_denoiser(
            emotion_sequence,
            noise_sequence,
            feature_mask,
        )
        denoised_emotion_sequence = relation_output.denoised_emotion_sequence
        differential_response = relation_output.differential_response
        differential_gate = relation_output.differential_gate
        differential_lambda = relation_output.differential_lambda
        pooling_emotion_sequence = denoised_emotion_sequence
        speech_statistics, sample_valid = masked_mean_std(
            pooling_emotion_sequence,
            feature_mask,
        )
        noise_statistics, noise_valid = masked_mean_std(
            noise_sequence,
            noise_feature_mask,
        )
        if not torch.equal(sample_valid, noise_valid):
            raise RuntimeError("speech and acoustic-condition validity differ.")
        valid_rows = sample_valid.unsqueeze(1)
        base_speech_embedding = torch.where(
            valid_rows,
            self.speech_projection(speech_statistics),
            torch.zeros(
                (waveform.shape[0], self.speech_embedding_dim),
                device=waveform.device,
                dtype=waveform.dtype,
            ),
        )
        noise_embedding = torch.where(
            valid_rows,
            self.noise_projection(noise_statistics),
            torch.zeros(
                (waveform.shape[0], self.noise_embedding_dim),
                device=waveform.device,
                dtype=waveform.dtype,
            ),
        )
        gamma, beta = self.film(noise_embedding).chunk(2, dim=-1)
        gamma_bounded = self.film_scale * torch.tanh(gamma)
        beta_bounded = self.film_scale * torch.tanh(beta)
        speech_embedding = base_speech_embedding * (1.0 + gamma_bounded) + beta_bounded
        speech_embedding = torch.where(
            valid_rows,
            speech_embedding,
            torch.zeros_like(speech_embedding),
        )

        arousal_logits = torch.where(
            valid_rows,
            self.arousal_head(speech_embedding),
            torch.zeros(
                (waveform.shape[0], 2),
                device=waveform.device,
                dtype=waveform.dtype,
            ),
        )
        valence_logits = torch.where(
            valid_rows,
            self.valence_head(speech_embedding),
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
        quadrant_probabilities = arousal_logits.new_zeros((waveform.shape[0], 4))
        quadrant_probabilities = quadrant_probabilities.index_copy(
            0,
            valid_indices,
            derive_quadrant_probabilities(
                arousal_probabilities.index_select(0, valid_indices),
                valence_probabilities.index_select(0, valid_indices),
            ),
        )
        temporal_weights = self._uniform_temporal_weights(
            feature_mask,
            dtype=waveform.dtype,
        )
        activity_counts = waveform_activity_mask.sum(dim=1, keepdim=True)
        valid_counts = speech_attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        speech_activity_ratio = activity_counts.to(dtype=waveform.dtype) / (
            valid_counts.to(dtype=waveform.dtype)
        )
        diagnostic_valid = sample_valid.unsqueeze(1).to(dtype=waveform.dtype)
        current_emotion_weights = self.emotion_layer_aggregation.normalized_weights()
        full_emotion_weights = self._emotion_layer_weights.to(
            dtype=waveform.dtype
        ).clone()
        full_emotion_weights[list(_EMOTION_LAYER_INDICES)] = current_emotion_weights.to(
            device=waveform.device,
            dtype=waveform.dtype,
        )
        emotion_layer_weights = full_emotion_weights.unsqueeze(0) * diagnostic_valid
        noise_layer_weights = (
            self._noise_layer_weights.to(dtype=waveform.dtype).unsqueeze(0)
            * diagnostic_valid
        )
        reliability = sample_valid.to(dtype=waveform.dtype).unsqueeze(1)
        for value in (
            emotion_sequence,
            noise_sequence,
            speech_embedding,
            base_speech_embedding,
            noise_embedding,
            gamma_bounded,
            beta_bounded,
            arousal_logits,
            valence_logits,
            arousal_probabilities,
            valence_probabilities,
            quadrant_probabilities,
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError("lightweight speech output contains NaN or Inf.")
        for diagnostic in (
            denoised_emotion_sequence,
            differential_response,
            differential_gate,
            differential_lambda,
        ):
            if not bool(torch.isfinite(diagnostic).all()):
                raise RuntimeError("lightweight speech diagnostic contains NaN or Inf.")
        return LightweightSpeechClassifierOutput(
            arousal_logits=arousal_logits,
            valence_logits=valence_logits,
            arousal_probabilities=arousal_probabilities,
            valence_probabilities=valence_probabilities,
            quadrant_probabilities=quadrant_probabilities,
            speech_embedding=speech_embedding,
            base_speech_embedding=base_speech_embedding,
            noise_embedding=noise_embedding,
            emotion_sequence=emotion_sequence,
            noise_sequence=noise_sequence,
            gamma_bounded=gamma_bounded,
            beta_bounded=beta_bounded,
            emotion_layer_weights=emotion_layer_weights,
            noise_layer_weights=noise_layer_weights,
            temporal_attention_weights=temporal_weights,
            feature_attention_mask=feature_mask,
            speech_activity_ratio=speech_activity_ratio,
            reliability=reliability,
            sample_valid=sample_valid,
            reliability_score_type=_RELIABILITY_SCORE_TYPE,
            denoised_emotion_sequence=denoised_emotion_sequence,
            differential_response=differential_response,
            differential_gate=differential_gate,
            differential_lambda=differential_lambda,
        )


__all__ = [
    "LightweightNoiseConditionedSpeechClassifier",
    "LightweightSpeechClassifierOutput",
]
