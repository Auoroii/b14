"""Configurable aggregation of the upper WavLM emotion layers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

EMOTION_LAYER_INDICES = (8, 9, 10, 11)
EMOTION_LAYER_NAMES = ("H9", "H10", "H11", "H12")
EMOTION_LAYER_AGGREGATION_MODES = {
    "fixed_mean",
    "learnable_weighted",
}


class EmotionLayerAggregation(nn.Module):
    """Aggregate H9--H12 tensors ``[B,T,D]`` without changing time.

    Args:
        mode: ``fixed_mean`` computes the exact H9--H12 arithmetic mean, and
            ``learnable_weighted`` applies four softmax-normalized scalar
            logits initialized to zero.
        expected_num_layers: Number of WavLM Transformer states in the input.

    The learnable mode adds exactly four downstream parameters. Internal
    indices ``(8, 9, 10, 11)`` correspond to human-readable H9--H12.
    """

    def __init__(
        self,
        mode: str = "fixed_mean",
        *,
        expected_num_layers: int = 12,
    ) -> None:
        super().__init__()
        if mode not in EMOTION_LAYER_AGGREGATION_MODES:
            raise ValueError(
                "mode must be one of "
                f"{sorted(EMOTION_LAYER_AGGREGATION_MODES)}; received {mode!r}."
            )
        if expected_num_layers != 12:
            raise ValueError("expected_num_layers must be 12 for H1--H12.")
        self.mode = mode
        self.expected_num_layers = expected_num_layers
        self.emotion_layer_logits: nn.Parameter | None
        if mode == "learnable_weighted":
            self.emotion_layer_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("emotion_layer_logits", None)

    def normalized_weights(self) -> Tensor:
        """Return finite H9--H12 weights with shape ``[4]`` summing to one."""

        if self.mode == "fixed_mean":
            reference = next(self.parameters(), None)
            if reference is None:
                return torch.full((4,), 0.25)
            return reference.new_full((4,), 0.25)
        logits = self.emotion_layer_logits
        if logits is None:
            raise RuntimeError("learnable aggregation logits are unavailable.")
        return torch.softmax(logits, dim=0)

    def forward(self, hidden_states: Sequence[Tensor]) -> Tensor:
        """Return one emotion sequence ``[B,T,D]`` from WavLM H1--H12."""

        if isinstance(hidden_states, (str, bytes)) or not isinstance(
            hidden_states,
            Sequence,
        ):
            raise TypeError("hidden_states must be a sequence of [B,T,D] tensors.")
        if len(hidden_states) != self.expected_num_layers:
            raise ValueError(
                f"hidden_states must contain exactly {self.expected_num_layers} layers."
            )
        selected = tuple(hidden_states[index] for index in EMOTION_LAYER_INDICES)
        reference = selected[0]
        if not isinstance(reference, Tensor) or reference.ndim != 3:
            raise ValueError("every selected hidden state must be a [B,T,D] tensor.")
        for hidden_state in selected[1:]:
            if (
                not isinstance(hidden_state, Tensor)
                or hidden_state.ndim != 3
                or hidden_state.shape != reference.shape
                or hidden_state.dtype != reference.dtype
                or hidden_state.device != reference.device
            ):
                raise ValueError(
                    "H9--H12 must share [B,T,D] shape, dtype, and device."
                )
        stacked = torch.stack(selected, dim=0)
        weights = self.normalized_weights().to(
            device=reference.device,
            dtype=reference.dtype,
        )
        return torch.sum(weights[:, None, None, None] * stacked, dim=0)


__all__ = [
    "EMOTION_LAYER_AGGREGATION_MODES",
    "EMOTION_LAYER_INDICES",
    "EMOTION_LAYER_NAMES",
    "EmotionLayerAggregation",
]
