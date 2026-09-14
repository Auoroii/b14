"""Dependency-injected WavLM hidden-state extraction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from transformers import WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from emotion_model.common import apply_query_mask, validate_sequence_mask


def _wavlm_feature_lengths(model: WavLMModel, input_lengths: Tensor) -> Tensor:
    """Call the installed Transformers feature-length helper in one boundary."""
    helper = getattr(model, "_get_feat_extract_output_lengths", None)
    if helper is None or not callable(helper):
        raise RuntimeError(
            "The installed WavLMModel does not provide "
            "_get_feat_extract_output_lengths; update the compatibility wrapper "
            "for this Transformers version."
        )
    output_lengths = helper(input_lengths)
    if not isinstance(output_lengths, Tensor):
        raise RuntimeError(
            "WavLM feature-length helper returned an unsupported value of type "
            f"{type(output_lengths).__name__}."
        )
    return output_lengths.to(device=input_lengths.device, dtype=torch.long)


@dataclass(frozen=True)
class WavLMEncoderOutput:
    """Strongly typed WavLM sequence features.

    Attributes:
        hidden_states: Transformer-layer outputs only. The tuple has length H
            and each tensor has shape ``[B, T_s, D_wavlm]``. The embedding or
            feature-projection output is excluded. Invalid feature frames are
            explicitly zero.
        feature_attention_mask: Boolean tensor with shape ``[B, T_s]`` using
            project semantics: ``True`` is a valid feature frame and ``False``
            is padding.
    """

    hidden_states: tuple[Tensor, ...]
    feature_attention_mask: Tensor

    @property
    def last_hidden_state(self) -> Tensor:
        """Return the final Transformer layer with shape ``[B, T_s, D_wavlm]``."""
        return self.hidden_states[-1]


class WavLMEncoder(nn.Module):
    """Extract every Transformer layer from an injected Hugging Face WavLM.

    Args:
        model: Locally constructed or otherwise injected :class:`WavLMModel`.
            Construction never calls ``from_pretrained`` and never downloads
            weights.
        sample_rate: Expected waveform sample rate in Hz. It must be positive.
        freeze_wavlm: ``True`` selects the fully frozen mode and requires
            ``unfreeze_last_n_layers=0``. ``False`` selects partial fine-tuning
            and requires a positive layer count.
        unfreeze_last_n_layers: Number of final Transformer layers to train.
            It must lie in ``[0, expected_num_hidden_layers]``. Convolutional
            feature extraction, feature projection, and all other WavLM
            parameters remain frozen in partial fine-tuning mode.
        expected_num_hidden_layers: Required number of WavLM Transformer
            layers. It must match ``model.config.num_hidden_layers`` exactly.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If numeric arguments are invalid or the configured WavLM
            layer count differs from ``expected_num_hidden_layers``.
    """

    def __init__(
        self,
        model: WavLMModel,
        *,
        sample_rate: int = 16000,
        freeze_wavlm: bool = True,
        unfreeze_last_n_layers: int = 0,
        expected_num_hidden_layers: int = 12,
    ) -> None:
        super().__init__()
        if not isinstance(model, WavLMModel):
            raise TypeError(f"model must be a WavLMModel; received {type(model).__name__}.")
        self._validate_sample_rate(sample_rate, name="sample_rate")
        if not isinstance(freeze_wavlm, bool):
            raise TypeError(f"freeze_wavlm must be bool; received {freeze_wavlm!r}.")
        if (
            isinstance(unfreeze_last_n_layers, bool)
            or not isinstance(unfreeze_last_n_layers, int)
        ):
            raise TypeError(
                "unfreeze_last_n_layers must be an integer, not bool; "
                f"received {unfreeze_last_n_layers!r}."
            )
        if (
            isinstance(expected_num_hidden_layers, bool)
            or not isinstance(expected_num_hidden_layers, int)
        ):
            raise TypeError(
                "expected_num_hidden_layers must be an integer; "
                f"received {expected_num_hidden_layers!r}."
            )
        if expected_num_hidden_layers <= 0:
            raise ValueError("expected_num_hidden_layers must be greater than zero.")
        if not 0 <= unfreeze_last_n_layers <= expected_num_hidden_layers:
            raise ValueError(
                "unfreeze_last_n_layers must lie in "
                f"[0, {expected_num_hidden_layers}]; received "
                f"{unfreeze_last_n_layers}."
            )
        if freeze_wavlm != (unfreeze_last_n_layers == 0):
            raise ValueError(
                "freeze_wavlm must be true exactly when "
                "unfreeze_last_n_layers is 0."
            )

        configured_layers = model.config.num_hidden_layers
        if configured_layers != expected_num_hidden_layers:
            raise ValueError(
                "WavLM configured layer count does not match the wrapper expectation: "
                f"model has {configured_layers}, expected {expected_num_hidden_layers}."
            )

        self.model = model
        self.sample_rate = sample_rate
        self.freeze_wavlm = freeze_wavlm
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.expected_num_hidden_layers = expected_num_hidden_layers
        first_trainable_layer = expected_num_hidden_layers - unfreeze_last_n_layers
        self.trainable_transformer_layer_indices = tuple(
            range(first_trainable_layer, expected_num_hidden_layers)
        )

        self.model.requires_grad_(False)
        for layer_index in self.trainable_transformer_layer_indices:
            self.model.encoder.layers[layer_index].requires_grad_(True)
        if self.fully_frozen:
            self.model.eval()
        else:
            self._disable_layerdrop_for_partial_fine_tuning()
            self.model.train(self.training)

    @property
    def fully_frozen(self) -> bool:
        """Return whether every WavLM parameter is frozen."""

        return self.unfreeze_last_n_layers == 0

    @staticmethod
    def _validate_sample_rate(sample_rate: int, *, name: str) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError(f"{name} must be an integer; received {sample_rate!r}.")
        if sample_rate <= 0:
            raise ValueError(f"{name} must be greater than zero; received {sample_rate}.")

    def _disable_layerdrop_for_partial_fine_tuning(self) -> None:
        """Disable block skipping so partial training still returns H1--H12."""

        self.model.config.layerdrop = 0.0
        encoder_config = getattr(self.model.encoder, "config", None)
        if encoder_config is None or not hasattr(encoder_config, "layerdrop"):
            raise RuntimeError("WavLM encoder config does not expose layerdrop.")
        encoder_config.layerdrop = 0.0

    def train(self, mode: bool = True) -> WavLMEncoder:
        """Set wrapper mode while keeping a frozen WavLM in eval mode.

        Args:
            mode: ``True`` for training mode and ``False`` for evaluation mode.

        Returns:
            This ``WavLMEncoder`` instance, following ``nn.Module.train``.

        Raises:
            ValueError: If ``mode`` is not boolean, as required by PyTorch.
        """
        super().train(mode)
        if self.fully_frozen:
            self.model.eval()
        else:
            self._disable_layerdrop_for_partial_fine_tuning()
            self.model.train(mode)
        return self

    def _validate_forward_inputs(
        self,
        waveform: Tensor,
        speech_attention_mask: Tensor,
        sample_rate: int | None,
    ) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have exact shape [B, L]; "
                f"received {tuple(waveform.shape)}."
            )
        if not waveform.is_floating_point():
            raise TypeError(f"waveform must be floating point; received {waveform.dtype}.")
        validate_sequence_mask(waveform, speech_attention_mask)
        if waveform.shape[0] == 0:
            raise ValueError("waveform batch dimension B must be greater than zero.")

        effective_sample_rate = self.sample_rate if sample_rate is None else sample_rate
        self._validate_sample_rate(effective_sample_rate, name="sample_rate")
        if effective_sample_rate != self.sample_rate:
            raise ValueError(
                f"sample_rate must equal the configured {self.sample_rate} Hz; "
                f"received {effective_sample_rate}."
            )

        if speech_attention_mask.shape[1] > 1:
            internal_holes = (
                ~speech_attention_mask[:, :-1] & speech_attention_mask[:, 1:]
            )
            if bool(internal_holes.any()):
                raise ValueError(
                    "speech_attention_mask must contain only a valid prefix followed "
                    "by right padding; internal False-to-True holes are not supported."
                )

        valid_lengths = speech_attention_mask.sum(dim=1, dtype=torch.long)
        if bool((valid_lengths == 0).any()):
            raise ValueError(
                "Every waveform sample must contain at least one valid point; "
                "fully padded samples are not supported by WavLMEncoder."
            )

        valid_waveform_values = waveform[speech_attention_mask]
        if not bool(torch.isfinite(valid_waveform_values).all()):
            raise ValueError("waveform values at valid mask positions must be finite.")

        feature_lengths = _wavlm_feature_lengths(self.model, valid_lengths)
        too_short = feature_lengths < 1
        if bool(too_short.any()):
            invalid_lengths = valid_lengths[too_short].tolist()
            raise ValueError(
                "Valid waveform length is too short for the WavLM convolutional "
                f"frontend to produce one feature frame: {invalid_lengths}."
            )
        return feature_lengths

    def forward(
        self,
        waveform: Tensor,
        speech_attention_mask: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> WavLMEncoderOutput:
        """Extract Transformer hidden states from padded waveform batches.

        Args:
            waveform: Floating-point tensor with shape ``[B, L]``.
            speech_attention_mask: Boolean tensor with exact shape ``[B, L]``.
                ``True`` marks a valid waveform prefix and ``False`` marks right
                padding. Internal holes and fully padded samples are rejected.
            sample_rate: Optional runtime sample rate in Hz. When supplied, it
                must equal the configured ``sample_rate``.

        Returns:
            :class:`WavLMEncoderOutput` containing exactly H Transformer layer
            tensors of shape ``[B, T_s, D_wavlm]`` and a boolean
            ``feature_attention_mask`` of shape ``[B, T_s]``. The embedding
            output is excluded. Invalid feature frames are explicitly zero in
            every returned layer.

        Raises:
            TypeError: If waveform, mask, or sample-rate dtypes/types are invalid.
            ValueError: If shapes/devices mismatch, masks are not right padded,
                a sample is fully padded or too short, valid waveform values are
                non-finite, or the runtime sample rate differs.
            RuntimeError: If the installed WavLM output contract, hidden-state
                count/shapes/devices, feature lengths, or output finiteness is
                inconsistent with this wrapper.

        Inputs are not modified. Padding waveform values are replaced by zero
        only in a new tensor before crossing the Hugging Face boundary.
        """
        feature_lengths = self._validate_forward_inputs(
            waveform,
            speech_attention_mask,
            sample_rate,
        )
        sanitized_waveform = torch.where(
            speech_attention_mask,
            waveform,
            torch.zeros_like(waveform),
        )

        gradient_context = torch.no_grad() if self.fully_frozen else nullcontext()
        with gradient_context:
            outputs = cast(
                Wav2Vec2BaseModelOutput,
                self.model(
                    input_values=sanitized_waveform,
                    attention_mask=speech_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ),
            )

        all_hidden_states = outputs.hidden_states
        expected_output_count = self.expected_num_hidden_layers + 1
        if all_hidden_states is None or len(all_hidden_states) != expected_output_count:
            actual_count = 0 if all_hidden_states is None else len(all_hidden_states)
            raise RuntimeError(
                "WavLM hidden-state count is inconsistent: expected embedding output "
                f"plus {self.expected_num_hidden_layers} Transformer layers "
                f"({expected_output_count} total), received {actual_count}."
            )

        transformer_hidden_states = tuple(all_hidden_states[1:])
        reference_shape = tuple(transformer_hidden_states[0].shape)
        if len(reference_shape) != 3:
            raise RuntimeError(
                "WavLM hidden states must have shape [B, T_s, D_wavlm]; "
                f"received {reference_shape}."
            )
        if reference_shape[0] != waveform.shape[0]:
            raise RuntimeError(
                "WavLM hidden-state batch dimension does not match waveform: "
                f"expected {waveform.shape[0]}, received {reference_shape[0]}."
            )
        configured_hidden_size = self.model.config.hidden_size
        if reference_shape[2] != configured_hidden_size:
            raise RuntimeError(
                "WavLM hidden-state feature dimension does not match model.config.hidden_size: "
                f"expected {configured_hidden_size}, received {reference_shape[2]}."
            )

        for layer_index, hidden_state in enumerate(transformer_hidden_states, start=1):
            if tuple(hidden_state.shape) != reference_shape:
                raise RuntimeError(
                    "All WavLM Transformer hidden states must have identical shapes; "
                    f"layer 1 has {reference_shape}, layer {layer_index} has "
                    f"{tuple(hidden_state.shape)}."
                )
            if hidden_state.device != waveform.device:
                raise RuntimeError(
                    f"WavLM layer {layer_index} is on {hidden_state.device}, "
                    f"expected {waveform.device}."
                )

        feature_time = reference_shape[1]
        if bool((feature_lengths > feature_time).any()):
            raise RuntimeError(
                "WavLM feature-length helper produced a length greater than the "
                f"hidden-state time dimension {feature_time}: {feature_lengths.tolist()}."
            )
        feature_positions = torch.arange(feature_time, device=waveform.device).unsqueeze(0)
        feature_attention_mask = feature_positions < feature_lengths.unsqueeze(1)

        masked_hidden_states = tuple(
            apply_query_mask(hidden_state, feature_attention_mask)
            for hidden_state in transformer_hidden_states
        )
        for layer_index, masked_hidden_state in enumerate(masked_hidden_states, start=1):
            if not bool(torch.isfinite(masked_hidden_state).all()):
                raise RuntimeError(
                    f"WavLM Transformer hidden state {layer_index} contains NaN or Inf."
                )

        if feature_attention_mask.dtype != torch.bool:
            raise RuntimeError("WavLM feature attention mask must have dtype torch.bool.")
        if tuple(feature_attention_mask.shape) != reference_shape[:2]:
            raise RuntimeError(
                "WavLM feature attention mask shape must match hidden-state batch/time "
                f"dimensions {reference_shape[:2]}; received "
                f"{tuple(feature_attention_mask.shape)}."
            )
        return WavLMEncoderOutput(
            hidden_states=masked_hidden_states,
            feature_attention_mask=feature_attention_mask,
        )
