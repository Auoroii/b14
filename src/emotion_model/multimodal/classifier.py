"""Final shared-fusion multimodal emotion classifier."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emotion_model.common import derive_quadrant_probabilities
from emotion_model.data import AlignedMultimodalBatch
from emotion_model.multimodal.full_window_dynamic_fusion import (
    FullWindowDynamicMultimodalFusion,
)
from emotion_model.multimodal.fusion_output import (
    MultimodalFusionOutput,
)
from emotion_model.multimodal.routing import (
    ModalityAvailabilityMasks,
    MultimodalBatchScheduler,
    ScheduledModalityOutputs,
)

_CLASSIFICATION_POLICY = "fused_embedding_only"
_QUADRANT_POLICY = "derived_from_arousal_valence"
_INVALID_SAMPLE_OUTPUT_POLICY = "all_zero"
_STATE_VERSION = 1
_MODEL_VARIANT = (
    "lightweight_shared_dynamic_relation_differential_full_window"
)


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0; received {value}.")


def _validate_dropout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dropout must be a real number, not bool.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError(
            f"dropout must be finite and lie in [0, 1); received {value!r}."
        )
    return probability


def _classifier_trunk(
    fusion_dim: int,
    classifier_hidden_dim: int,
    dropout: float,
) -> nn.Sequential:
    normalization: nn.Module = (
        nn.Identity() if fusion_dim == 1 else nn.LayerNorm(fusion_dim)
    )
    return nn.Sequential(
        normalization,
        nn.Linear(fusion_dim, classifier_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


@dataclass(frozen=True)
class MultimodalEmotionClassifierOutput:
    """Final predictions and retained fusion diagnostics for one logical batch.

    Attributes:
        arousal_logits: Floating tensor ``[B, 2]``.
        valence_logits: Floating tensor ``[B, 2]``.
        arousal_probabilities: Floating tensor ``[B, 2]`` in
            ``[low, high]`` order.
        valence_probabilities: Floating tensor ``[B, 2]`` in
            ``[low, high]`` order.
        quadrant_probabilities: Floating tensor ``[B, 4]`` ordered as
            ``[LALV, HALV, LAHV, HAHV]`` and derived from the two binary
            probability tensors.
        fused_embedding: The exact shared ``[B, F]`` tensor object returned by
            ``fusion_output`` and consumed by both task heads.
        sample_valid: The exact boolean ``[B]`` tensor object returned by
            ``fusion_output``. ``True`` means at least one modality is valid.
        fusion_output: The exact :class:`MultimodalFusionOutput`, through
            which scheduler, compact, scattered, and fusion diagnostics remain
            available without duplication.
        quadrant_logits: Optional floating tensor ``[B, 4]`` from an explicitly
            enabled independent diagnostic head; otherwise ``None``.

    Rows where ``sample_valid=False`` are exact-zero unavailable sentinels in
    every public logit and probability tensor. Their probability rows
    intentionally do not sum to one and must not be interpreted as predictions.
    Direct construction validates shapes, dtype/device consistency, finite
    values, exact tensor identity for retained fusion fields, probability
    normalization on valid rows, and exact-zero invalid rows. Invalid inputs
    raise ``TypeError`` or ``ValueError``.
    """

    arousal_logits: Tensor
    valence_logits: Tensor
    arousal_probabilities: Tensor
    valence_probabilities: Tensor
    quadrant_probabilities: Tensor
    fused_embedding: Tensor
    sample_valid: Tensor
    fusion_output: MultimodalFusionOutput
    quadrant_logits: Tensor | None

    def __post_init__(self) -> None:
        if not isinstance(self.fusion_output, MultimodalFusionOutput):
            raise TypeError("fusion_output must be MultimodalFusionOutput.")
        if self.fused_embedding is not self.fusion_output.fused_embedding:
            raise ValueError(
                "fused_embedding must be the exact fusion_output tensor object."
            )
        if self.sample_valid is not self.fusion_output.sample_valid:
            raise ValueError(
                "sample_valid must be the exact fusion_output tensor object."
            )
        fused = self.fused_embedding
        valid = self.sample_valid
        if (
            not isinstance(fused, Tensor)
            or not fused.is_floating_point()
            or fused.ndim != 2
            or fused.shape[0] <= 0
            or fused.shape[1] <= 0
        ):
            raise ValueError("fused_embedding must be floating with shape [B, F].")
        batch_size = fused.shape[0]
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or tuple(valid.shape) != (batch_size,)
        ):
            raise ValueError("sample_valid must be bool with shape [B].")

        fields: list[tuple[str, Tensor, tuple[int, int]]] = [
            ("arousal_logits", self.arousal_logits, (batch_size, 2)),
            ("valence_logits", self.valence_logits, (batch_size, 2)),
            (
                "arousal_probabilities",
                self.arousal_probabilities,
                (batch_size, 2),
            ),
            (
                "valence_probabilities",
                self.valence_probabilities,
                (batch_size, 2),
            ),
            (
                "quadrant_probabilities",
                self.quadrant_probabilities,
                (batch_size, 4),
            ),
        ]
        if self.quadrant_logits is not None:
            fields.append(
                ("quadrant_logits", self.quadrant_logits, (batch_size, 4))
            )
        local_valid = valid.to(device=fused.device)
        for name, value, expected_shape in fields:
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating tensor.")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"received {tuple(value.shape)}."
                )
            if value.dtype != fused.dtype or value.device != fused.device:
                raise ValueError(
                    f"{name} must match fused_embedding dtype/device."
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")
            invalid_values = value[~local_valid]
            if not torch.equal(
                invalid_values,
                torch.zeros_like(invalid_values),
            ):
                raise ValueError(f"{name} invalid rows must be exact zero.")

        for name, probabilities in (
            ("arousal_probabilities", self.arousal_probabilities),
            ("valence_probabilities", self.valence_probabilities),
            ("quadrant_probabilities", self.quadrant_probabilities),
        ):
            available = probabilities[local_valid]
            if not bool((available >= 0).all() and (available <= 1).all()):
                raise ValueError(f"{name} valid values must lie in [0, 1].")
            if not torch.allclose(
                available.sum(dim=1),
                torch.ones_like(available[:, 0]),
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(f"{name} valid rows must sum to one.")


class MultimodalEmotionClassifier(nn.Module):
    """Classify the validated shared V4.2 fused embedding.

    Args:
        batch_scheduler: Registered scheduler mapping one
            :class:`AlignedMultimodalBatch` to full-batch unimodal outputs.
        multimodal_fusion: Registered full-window dynamic feature fusion.
        fusion_dim: Positive fused embedding width ``F``. It must match the
            fusion module's public ``fusion_dim``.
        classifier_hidden_dim: Positive shared classifier width. ``None`` uses
            ``fusion_dim``.
        dropout: Dropout probability in ``[0, 1)``.
        enable_independent_quadrant_head: Whether to create an optional
            independent four-class diagnostic head. Derived quadrant
            probabilities remain the primary output in either mode.
        model_variant: Exact V4.2 architecture fingerprint.

    Forward input is one :class:`AlignedMultimodalBatch` of logical size ``B``.
    The output contains arousal/valence logits and probabilities ``[B, 2]``,
    derived quadrant probabilities ``[B, 4]``, fused embeddings ``[B, F]``,
    and validity ``[B]``. Both tasks classify the same shared fused embedding;
    unimodal logits and probabilities are never fused.

    Fully unavailable rows remain in the logical batch and return exact-zero
    logits and probabilities. These zero probability rows are explicit
    unavailable sentinels, not probability distributions. The forward method
    receives no labels and computes no loss.
    """

    def __init__(
        self,
        batch_scheduler: MultimodalBatchScheduler,
        multimodal_fusion: FullWindowDynamicMultimodalFusion,
        fusion_dim: int,
        *,
        classifier_hidden_dim: int | None = None,
        dropout: float = 0.1,
        enable_independent_quadrant_head: bool = False,
        model_variant: str = _MODEL_VARIANT,
    ) -> None:
        super().__init__()
        if not isinstance(batch_scheduler, MultimodalBatchScheduler):
            raise TypeError(
                "batch_scheduler must be MultimodalBatchScheduler; "
                f"received {type(batch_scheduler).__name__}."
            )
        if not isinstance(multimodal_fusion, FullWindowDynamicMultimodalFusion):
            raise TypeError(
                "multimodal_fusion must be FullWindowDynamicMultimodalFusion; "
                f"received {type(multimodal_fusion).__name__}."
            )
        _validate_positive_integer(fusion_dim, name="fusion_dim")
        if classifier_hidden_dim is None:
            classifier_hidden_dim = fusion_dim
        _validate_positive_integer(
            classifier_hidden_dim,
            name="classifier_hidden_dim",
        )
        dropout_probability = _validate_dropout(dropout)
        if not isinstance(enable_independent_quadrant_head, bool):
            raise TypeError("enable_independent_quadrant_head must be bool.")
        if model_variant != _MODEL_VARIANT:
            raise ValueError(f"model_variant must be {_MODEL_VARIANT!r}.")
        if multimodal_fusion.fusion_dim != fusion_dim:
            raise ValueError(
                "fusion_dim must match multimodal_fusion.fusion_dim; "
                f"received {fusion_dim} and {multimodal_fusion.fusion_dim}."
            )

        self.batch_scheduler = batch_scheduler
        self.multimodal_fusion = multimodal_fusion
        self.fusion_dim = fusion_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.dropout = dropout_probability
        self.enable_independent_quadrant_head = (
            enable_independent_quadrant_head
        )
        self.model_variant = model_variant
        self.classification_policy = _CLASSIFICATION_POLICY
        self.quadrant_policy = _QUADRANT_POLICY
        self.invalid_sample_output_policy = _INVALID_SAMPLE_OUTPUT_POLICY

        self.classifier_trunk = _classifier_trunk(
            fusion_dim,
            classifier_hidden_dim,
            dropout_probability,
        )
        self.arousal_head = nn.Linear(classifier_hidden_dim, 2)
        self.valence_head = nn.Linear(classifier_hidden_dim, 2)
        self.quadrant_head = (
            nn.Linear(classifier_hidden_dim, 4)
            if enable_independent_quadrant_head
            else None
        )

    def get_extra_state(self) -> dict[str, object]:
        """Return the final-classifier architecture and policy fingerprint."""
        state: dict[str, object] = {
            "state_version": _STATE_VERSION,
            "fusion_dim": self.fusion_dim,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "dropout": self.dropout,
            "enable_independent_quadrant_head": (
                self.enable_independent_quadrant_head
            ),
            "classification_policy": self.classification_policy,
            "quadrant_policy": self.quadrant_policy,
            "invalid_sample_output_policy": self.invalid_sample_output_policy,
        }
        state["model_variant"] = self.model_variant
        return state

    def set_extra_state(self, state: object) -> None:
        """Validate checkpoint metadata without changing construction."""
        if not isinstance(state, Mapping):
            raise RuntimeError(
                "MultimodalEmotionClassifier extra state must be a mapping."
            )
        if dict(state) != self.get_extra_state():
            raise RuntimeError(
                "MultimodalEmotionClassifier checkpoint configuration does "
                "not match the constructed module."
            )

    def _parameter_reference(self) -> Tensor:
        linear = self.classifier_trunk[1]
        if not isinstance(linear, nn.Linear):
            raise RuntimeError("classifier trunk Linear is missing.")
        return linear.weight

    def _validate_parameter_consistency(self, reference: Tensor) -> None:
        for name, parameter in self.named_parameters():
            if name.startswith(("batch_scheduler.", "multimodal_fusion.")):
                continue
            if (
                parameter.dtype != reference.dtype
                or parameter.device != reference.device
            ):
                raise RuntimeError(
                    "all final classifier parameters must share dtype/device; "
                    f"{name} has {parameter.dtype}/{parameter.device}, expected "
                    f"{reference.dtype}/{reference.device}."
                )

    @staticmethod
    def _validate_scheduled_output(
        scheduled_output: ScheduledModalityOutputs,
        batch: AlignedMultimodalBatch,
        *,
        batch_size: int,
    ) -> None:
        if scheduled_output.batch is not batch:
            raise RuntimeError(
                "batch_scheduler output must retain the exact input batch object."
            )
        availability = scheduled_output.availability
        if not isinstance(availability, ModalityAvailabilityMasks):
            raise RuntimeError(
                "batch_scheduler availability must be ModalityAvailabilityMasks."
            )
        for name, value in (
            ("speech_available", availability.speech_available),
            ("physiology_available", availability.physiology_available),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or tuple(value.shape) != (batch_size,)
            ):
                actual_shape = tuple(value.shape) if isinstance(value, Tensor) else None
                raise RuntimeError(
                    f"batch_scheduler {name} must be bool [{batch_size}]; "
                    f"received {actual_shape}."
                )

    def _validate_fusion_output(
        self,
        fusion_output: MultimodalFusionOutput,
        scheduled_output: ScheduledModalityOutputs,
        *,
        batch_size: int,
        reference: Tensor,
    ) -> None:
        if fusion_output.scheduled_outputs is not scheduled_output:
            raise RuntimeError(
                "multimodal_fusion output must retain the exact scheduler output."
            )
        if fusion_output.availability is not scheduled_output.availability:
            raise RuntimeError(
                "multimodal_fusion output must retain the exact availability object."
            )
        fused = fusion_output.fused_embedding
        if (
            not isinstance(fused, Tensor)
            or not fused.is_floating_point()
            or tuple(fused.shape) != (batch_size, self.fusion_dim)
        ):
            actual_shape = tuple(fused.shape) if isinstance(fused, Tensor) else None
            raise RuntimeError(
                "multimodal_fusion fused_embedding must be floating with shape "
                f"[{batch_size}, {self.fusion_dim}]; received {actual_shape}."
            )
        if fused.dtype != reference.dtype or fused.device != reference.device:
            raise RuntimeError(
                "multimodal_fusion fused_embedding must match final classifier "
                f"dtype/device {reference.dtype}/{reference.device}; received "
                f"{fused.dtype}/{fused.device}."
            )
        valid = fusion_output.sample_valid
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or tuple(valid.shape) != (batch_size,)
        ):
            raise RuntimeError(
                f"multimodal_fusion sample_valid must be bool [{batch_size}]."
            )
        expected_valid = (
            scheduled_output.availability.speech_available
            | scheduled_output.availability.physiology_available
        )
        if valid.device != expected_valid.device or not torch.equal(
            valid,
            expected_valid,
        ):
            raise RuntimeError(
                "multimodal_fusion sample_valid must equal speech OR "
                "physiology availability."
            )
        local_valid = valid.to(device=fused.device)
        if not bool(torch.isfinite(fused[local_valid]).all()):
            raise RuntimeError(
                "multimodal_fusion valid fused_embedding rows must be finite."
            )
        invalid = fused[~local_valid]
        if not torch.equal(invalid, torch.zeros_like(invalid)):
            raise RuntimeError(
                "multimodal_fusion invalid fused_embedding rows must be exact zero."
            )

    def forward(
        self,
        batch: AlignedMultimodalBatch,
    ) -> MultimodalEmotionClassifierOutput:
        """Schedule, fuse, and classify one logical multimodal batch.

        Args:
            batch: Valid :class:`AlignedMultimodalBatch` with logical size
                ``B``. Labels remain untouched and are not read by this method.

        Returns:
            :class:`MultimodalEmotionClassifierOutput` with binary logits and
            probabilities ``[B, 2]``, derived quadrant probabilities ``[B, 4]``,
            fused embeddings ``[B, fusion_dim]``, validity ``[B]``, retained
            fusion diagnostics, and optional quadrant logits ``[B, 4]``.
            Fully unavailable rows are exact-zero sentinels in every public
            prediction tensor.

        Raises:
            TypeError: If ``batch`` has the wrong type.
            RuntimeError: If scheduler or fusion return the wrong type or
                violate identity, shape, validity, finite-value, zero-sentinel,
                dtype, or device contracts.

        The scheduler and fusion module are each called exactly once. No label,
        loss, scatter, modality-weight, logit-fusion, or probability-fusion
        logic is performed here, and no input is modified.
        """
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError(
                "batch must be AlignedMultimodalBatch; "
                f"received {type(batch).__name__}."
            )
        batch_size = len(batch.records)
        raw_scheduled_output = self.batch_scheduler(batch)
        if not isinstance(raw_scheduled_output, ScheduledModalityOutputs):
            raise RuntimeError(
                "batch_scheduler must return ScheduledModalityOutputs; "
                f"received {type(raw_scheduled_output).__name__}."
            )
        scheduled_output = raw_scheduled_output
        self._validate_scheduled_output(
            scheduled_output,
            batch,
            batch_size=batch_size,
        )

        raw_fusion_output = self.multimodal_fusion(scheduled_output)
        if not isinstance(raw_fusion_output, MultimodalFusionOutput):
            raise RuntimeError(
                "multimodal_fusion must return MultimodalFusionOutput; "
                f"received {type(raw_fusion_output).__name__}."
            )
        fusion_output = raw_fusion_output
        reference = self._parameter_reference()
        self._validate_parameter_consistency(reference)
        self._validate_fusion_output(
            fusion_output,
            scheduled_output,
            batch_size=batch_size,
            reference=reference,
        )

        fused_embedding = fusion_output.fused_embedding
        sample_valid = fusion_output.sample_valid
        local_valid = sample_valid.to(device=fused_embedding.device).unsqueeze(1)
        safe_fused_embedding = torch.where(
            local_valid,
            fused_embedding,
            torch.zeros_like(fused_embedding),
        )
        classifier_hidden = self.classifier_trunk(safe_fused_embedding)
        arousal_hidden = classifier_hidden
        valence_hidden = classifier_hidden
        raw_arousal_logits = self.arousal_head(arousal_hidden)
        raw_valence_logits = self.valence_head(valence_hidden)
        arousal_logits = torch.where(
            local_valid,
            raw_arousal_logits,
            torch.zeros_like(raw_arousal_logits),
        )
        valence_logits = torch.where(
            local_valid,
            raw_valence_logits,
            torch.zeros_like(raw_valence_logits),
        )

        arousal_distribution = torch.softmax(arousal_logits, dim=-1)
        valence_distribution = torch.softmax(valence_logits, dim=-1)
        derived_quadrant_distribution = derive_quadrant_probabilities(
            arousal_distribution,
            valence_distribution,
        )
        arousal_probabilities = torch.where(
            local_valid,
            arousal_distribution,
            torch.zeros_like(arousal_distribution),
        )
        valence_probabilities = torch.where(
            local_valid,
            valence_distribution,
            torch.zeros_like(valence_distribution),
        )
        quadrant_probabilities = torch.where(
            local_valid,
            derived_quadrant_distribution,
            torch.zeros_like(derived_quadrant_distribution),
        )

        quadrant_logits: Tensor | None = None
        if self.quadrant_head is not None:
            raw_quadrant_logits = self.quadrant_head(classifier_hidden)
            quadrant_logits = torch.where(
                local_valid,
                raw_quadrant_logits,
                torch.zeros_like(raw_quadrant_logits),
            )

        for name, value in (
            ("classifier_hidden", classifier_hidden),
            ("arousal_logits", arousal_logits),
            ("valence_logits", valence_logits),
            ("arousal_probabilities", arousal_probabilities),
            ("valence_probabilities", valence_probabilities),
            ("quadrant_probabilities", quadrant_probabilities),
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{name} contains NaN or Inf.")
        if quadrant_logits is not None and not bool(
            torch.isfinite(quadrant_logits).all()
        ):
            raise RuntimeError("quadrant_logits contains NaN or Inf.")

        return MultimodalEmotionClassifierOutput(
            arousal_logits=arousal_logits,
            valence_logits=valence_logits,
            arousal_probabilities=arousal_probabilities,
            valence_probabilities=valence_probabilities,
            quadrant_probabilities=quadrant_probabilities,
            fused_embedding=fused_embedding,
            sample_valid=sample_valid,
            fusion_output=fusion_output,
            quadrant_logits=quadrant_logits,
        )


__all__ = [
    "MultimodalEmotionClassifier",
    "MultimodalEmotionClassifierOutput",
]
