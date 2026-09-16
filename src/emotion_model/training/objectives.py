"""Multimodal classification objective composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn

from emotion_model.data import AlignedMultimodalBatch
from emotion_model.multimodal.classifier import (
    MultimodalEmotionClassifierOutput,
)
from emotion_model.multimodal.losses import (
    class_weighted_cross_entropy,
    multiclass_focal_loss,
)
from emotion_model.multimodal.fusion_output import (
    MultimodalFusionOutput,
)
from emotion_model.multimodal.routing import ScheduledModalityOutputs
from emotion_model.physiology import LightweightPhysioClassifierOutput
from emotion_model.speech import LightweightSpeechClassifierOutput


class ClassificationLossKind(StrEnum):
    """Supported public multiclass classification loss implementations."""

    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"
    FOCAL = "focal"


def _finite_nonnegative_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


def _optional_activity_ratio_threshold(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number or None, not bool.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1).")
    return normalized


@dataclass(frozen=True)
class MultimodalLossWeights:
    """Non-negative coefficients for multimodal objective components.

    These scalar values multiply already reduced loss tensors; they are
    objective coefficients, not probabilities, and are not normalized.
    """

    fused: float = 1.0
    speech_auxiliary: float = 0.0
    physiology_auxiliary: float = 0.0
    fused_quadrant: float = 0.0
    speech_quadrant: float = 0.0
    physiology_quadrant: float = 0.0

    def __post_init__(self) -> None:
        normalized: list[float] = []
        for name in (
            "fused",
            "speech_auxiliary",
            "physiology_auxiliary",
            "fused_quadrant",
            "speech_quadrant",
            "physiology_quadrant",
        ):
            value = _finite_nonnegative_real(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
            normalized.append(value)
        if not any(value > 0.0 for value in normalized):
            raise ValueError("at least one multimodal loss weight must be > 0.")


def _validate_class_weight_tensor(
    value: Tensor,
    *,
    name: str,
    class_count: int,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor or None.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point; received {value.dtype}.")
    if tuple(value.shape) != (class_count,):
        raise ValueError(
            f"{name} must have shape [{class_count}]; "
            f"received {tuple(value.shape)}."
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool((value > 0).all()):
        raise ValueError(f"{name} values must be strictly positive.")


@dataclass(frozen=True)
class EmotionTaskClassWeights:
    """Optional class weights shared by fused and unimodal task losses.

    Attributes:
        arousal: Floating tensor ``[2]`` or ``None``.
        valence: Floating tensor ``[2]`` or ``None``.
        quadrant: Floating tensor ``[4]`` or ``None``.

    Tensors remain on the caller-selected device and retain their dtype. They
    are neither normalized, copied, nor moved automatically.
    """

    arousal: Tensor | None = None
    valence: Tensor | None = None
    quadrant: Tensor | None = None

    def __post_init__(self) -> None:
        for name, value, class_count in (
            ("arousal", self.arousal, 2),
            ("valence", self.valence, 2),
            ("quadrant", self.quadrant, 4),
        ):
            if value is not None:
                _validate_class_weight_tensor(
                    value,
                    name=name,
                    class_count=class_count,
                )


@dataclass(frozen=True)
class MultimodalObjectiveConfig:
    """Immutable multimodal objective selection and scalar configuration.

    ``speech_aux_min_activity_ratio`` is ``None`` for unfiltered auxiliary
    supervision. Otherwise, compact speech rows are supervised only when the
    full-batch activity observation is present and its ratio ``[B]`` is
    strictly greater than this threshold.
    """

    loss_kind: ClassificationLossKind = (
        ClassificationLossKind.WEIGHTED_CROSS_ENTROPY
    )
    weights: MultimodalLossWeights = MultimodalLossWeights()
    focal_gamma: float = 2.0
    speech_aux_min_activity_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_kind, ClassificationLossKind):
            raise TypeError("loss_kind must be ClassificationLossKind.")
        if not isinstance(self.weights, MultimodalLossWeights):
            raise TypeError("weights must be MultimodalLossWeights.")
        gamma = _finite_nonnegative_real(self.focal_gamma, name="focal_gamma")
        object.__setattr__(self, "focal_gamma", gamma)
        threshold = _optional_activity_ratio_threshold(
            self.speech_aux_min_activity_ratio,
            name="speech_aux_min_activity_ratio",
        )
        object.__setattr__(self, "speech_aux_min_activity_ratio", threshold)


@dataclass(frozen=True)
class MultimodalLossOutput:
    """Scalar losses and active supervision count for one logical batch.

    Every loss field is a finite floating scalar tensor ``[]`` sharing
    dtype/device with the final model logits. Disabled, unavailable, or fully
    ignored terms are graph-safe zeros. ``active_target_count`` counts valid
    labels from enabled tasks exactly once per task.
    """

    total_loss: Tensor
    fused_loss: Tensor
    fused_arousal_loss: Tensor
    fused_valence_loss: Tensor
    fused_quadrant_loss: Tensor
    speech_auxiliary_loss: Tensor
    speech_arousal_loss: Tensor
    speech_valence_loss: Tensor
    speech_quadrant_loss: Tensor
    physiology_auxiliary_loss: Tensor
    physiology_arousal_loss: Tensor
    physiology_valence_loss: Tensor
    physiology_quadrant_loss: Tensor
    active_target_count: int

    def __post_init__(self) -> None:
        reference = self.total_loss
        if (
            not isinstance(reference, Tensor)
            or not reference.is_floating_point()
            or reference.ndim != 0
        ):
            raise ValueError("total_loss must be a floating scalar tensor.")
        for name in (
            "total_loss",
            "fused_loss",
            "fused_arousal_loss",
            "fused_valence_loss",
            "fused_quadrant_loss",
            "speech_auxiliary_loss",
            "speech_arousal_loss",
            "speech_valence_loss",
            "speech_quadrant_loss",
            "physiology_auxiliary_loss",
            "physiology_arousal_loss",
            "physiology_valence_loss",
            "physiology_quadrant_loss",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.ndim != 0
            ):
                raise ValueError(f"{name} must be a floating scalar tensor.")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    "all multimodal loss tensors must share dtype and device."
                )
            if not bool(torch.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
        if (
            isinstance(self.active_target_count, bool)
            or not isinstance(self.active_target_count, int)
            or self.active_target_count < 0
        ):
            raise ValueError("active_target_count must be a non-negative integer.")


def _graph_safe_zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _validate_task_inputs(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    *,
    class_count: int,
    ignore_index: int,
    name: str,
) -> None:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError(f"{name} logits must be a floating tensor.")
    if tuple(logits.shape) != (labels.shape[0], class_count):
        raise ValueError(
            f"{name} logits must have shape [N, {class_count}]; "
            f"received {tuple(logits.shape)} for N={labels.shape[0]}."
        )
    if not isinstance(labels, Tensor) or labels.dtype != torch.long:
        raise TypeError(f"{name} labels must be a torch.long tensor [N].")
    if labels.ndim != 1:
        raise ValueError(f"{name} labels must have shape [N].")
    if labels.device != logits.device:
        raise ValueError(f"{name} labels and logits must share device.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} logits must contain only finite values.")
    valid = labels != ignore_index
    if bool((valid & ((labels < 0) | (labels >= class_count))).any()):
        raise ValueError(
            f"{name} labels must lie in [0, {class_count - 1}] or equal "
            f"ignore_index={ignore_index}."
        )
    if class_weights is not None:
        _validate_class_weight_tensor(
            class_weights,
            name=f"{name} class_weights",
            class_count=class_count,
        )
        if class_weights.dtype != logits.dtype:
            raise TypeError(
                f"{name} class_weights must match logits dtype; "
                f"received {class_weights.dtype} and {logits.dtype}."
            )
        if class_weights.device != logits.device:
            raise ValueError(
                f"{name} class_weights must match logits device; "
                f"received {class_weights.device} and {logits.device}."
            )


def _single_task_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    *,
    class_count: int,
    ignore_index: int,
    config: MultimodalObjectiveConfig,
    name: str,
) -> Tensor:
    _validate_task_inputs(
        logits,
        labels,
        class_weights,
        class_count=class_count,
        ignore_index=ignore_index,
        name=name,
    )
    if config.loss_kind is ClassificationLossKind.WEIGHTED_CROSS_ENTROPY:
        return class_weighted_cross_entropy(
            logits,
            labels,
            class_weights=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
    return multiclass_focal_loss(
        logits,
        labels,
        gamma=config.focal_gamma,
        class_weights=class_weights,
        ignore_index=ignore_index,
        reduction="mean",
    )


def _count_active(labels: Tensor, *, ignore_index: int) -> int:
    return int((labels != ignore_index).sum().item())


@dataclass(frozen=True)
class _BinaryTaskLosses:
    arousal: Tensor
    valence: Tensor
    combined: Tensor
    count: int


def _binary_task_losses(
    arousal_logits: Tensor,
    valence_logits: Tensor,
    arousal_labels: Tensor,
    valence_labels: Tensor,
    class_weights: EmotionTaskClassWeights,
    *,
    ignore_index: int,
    config: MultimodalObjectiveConfig,
    enabled: bool,
    name: str,
) -> _BinaryTaskLosses:
    if not enabled:
        arousal_loss = _graph_safe_zero(arousal_logits)
        valence_loss = _graph_safe_zero(valence_logits)
        return _BinaryTaskLosses(
            arousal_loss,
            valence_loss,
            arousal_loss + valence_loss,
            0,
        )
    arousal_loss = _single_task_loss(
        arousal_logits,
        arousal_labels,
        class_weights.arousal,
        class_count=2,
        ignore_index=ignore_index,
        config=config,
        name=f"{name} arousal",
    )
    valence_loss = _single_task_loss(
        valence_logits,
        valence_labels,
        class_weights.valence,
        class_count=2,
        ignore_index=ignore_index,
        config=config,
        name=f"{name} valence",
    )
    return _BinaryTaskLosses(
        arousal_loss,
        valence_loss,
        arousal_loss + valence_loss,
        _count_active(arousal_labels, ignore_index=ignore_index)
        + _count_active(valence_labels, ignore_index=ignore_index),
    )


def _quadrant_task_loss(
    logits: Tensor | None,
    labels: Tensor,
    class_weights: EmotionTaskClassWeights,
    *,
    ignore_index: int,
    config: MultimodalObjectiveConfig,
    enabled: bool,
    name: str,
    zero_reference: Tensor,
) -> tuple[Tensor, int]:
    if not enabled:
        return _graph_safe_zero(zero_reference), 0
    if logits is None:
        raise RuntimeError(
            f"{name} quadrant loss is enabled but independent quadrant logits "
            "are unavailable."
        )
    loss = _single_task_loss(
        logits,
        labels,
        class_weights.quadrant,
        class_count=4,
        ignore_index=ignore_index,
        config=config,
        name=f"{name} quadrant",
    )
    return loss, _count_active(labels, ignore_index=ignore_index)


class MultimodalTrainingObjective(nn.Module):
    """Compose enabled classification losses for one multimodal model output.

    Args:
        config: Immutable loss kind, focal gamma, and six objective
            coefficients.

    Forward consumes a :class:`MultimodalEmotionClassifierOutput` and its exact
    :class:`AlignedMultimodalBatch`. Final logits have shapes ``[B, 2]`` and
    optional ``[B, 4]``. Auxiliary logits use compact shapes ``[Bm, 2]`` and
    optional ``[Bm, 4]``. The returned losses are scalar tensors ``[]`` and
    never modify labels, outputs, or class weights.

    The module has no trainable parameters, does not call the model, scheduler,
    or fusion module, and performs no backward or optimizer operation. Its
    immutable ``config`` must be stored by the external experiment
    configuration; no checkpoint manager is implemented here.
    """

    def __init__(self, config: MultimodalObjectiveConfig) -> None:
        super().__init__()
        if not isinstance(config, MultimodalObjectiveConfig):
            raise TypeError("config must be MultimodalObjectiveConfig.")
        self.config = config

    def _validate_inputs(
        self,
        model_output: MultimodalEmotionClassifierOutput,
        batch: AlignedMultimodalBatch,
    ) -> tuple[int, Tensor]:
        if not isinstance(model_output, MultimodalEmotionClassifierOutput):
            raise TypeError(
                "model_output must be MultimodalEmotionClassifierOutput."
            )
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError("batch must be AlignedMultimodalBatch.")
        try:
            batch.__post_init__()
        except (TypeError, ValueError) as error:
            raise type(error)(f"batch violates its contract: {error}") from error
        batch_size = len(batch.records)
        fusion_output = model_output.fusion_output
        if not isinstance(fusion_output, MultimodalFusionOutput):
            raise TypeError(
                "model_output.fusion_output must be "
                "MultimodalFusionOutput."
            )
        scheduled = fusion_output.scheduled_outputs
        if not isinstance(scheduled, ScheduledModalityOutputs):
            raise TypeError(
                "model_output scheduled output must be "
                "ScheduledModalityOutputs."
            )
        if scheduled.batch is not batch:
            raise ValueError(
                "model_output scheduled batch must be the exact objective batch."
            )
        sample_valid = model_output.sample_valid
        if (
            not isinstance(sample_valid, Tensor)
            or sample_valid.dtype != torch.bool
            or tuple(sample_valid.shape) != (batch_size,)
            or sample_valid.device != batch.speech_available.device
        ):
            raise ValueError(
                f"model_output sample_valid must be bool [{batch_size}] on "
                "the batch availability device."
            )
        expected_valid = batch.speech_available | batch.physiology_available
        if not torch.equal(sample_valid, expected_valid):
            raise ValueError(
                "model_output sample_valid must equal batch modality availability."
            )
        if (
            sample_valid is not fusion_output.sample_valid
            or model_output.fused_embedding is not fusion_output.fused_embedding
        ):
            raise ValueError(
                "model_output must retain exact fusion validity and embedding "
                "tensor objects."
            )
        reference = model_output.arousal_logits
        for name, logits, classes in (
            ("fused arousal", model_output.arousal_logits, 2),
            ("fused valence", model_output.valence_logits, 2),
        ):
            if (
                not isinstance(logits, Tensor)
                or not logits.is_floating_point()
                or tuple(logits.shape) != (batch_size, classes)
                or logits.dtype != reference.dtype
                or logits.device != reference.device
                or not bool(torch.isfinite(logits).all())
            ):
                raise ValueError(
                    f"{name} logits must be finite floating [{batch_size}, "
                    f"{classes}] with shared dtype/device."
                )
        return batch_size, reference

    @staticmethod
    def _fused_labels(
        batch: AlignedMultimodalBatch,
        sample_valid: Tensor,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_valid = sample_valid.to(device=batch.arousal_labels.device)
        labels: list[Tensor] = []
        for source in (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        ):
            cloned = source.clone()
            cloned = torch.where(
                local_valid,
                cloned,
                torch.full_like(cloned, batch.label_ignore_index),
            )
            labels.append(cloned.to(device=device))
        return labels[0], labels[1], labels[2]

    @staticmethod
    def _compact_labels(
        batch: AlignedMultimodalBatch,
        indices: Tensor,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_indices = indices.to(device=batch.arousal_labels.device)
        return (
            batch.arousal_labels.index_select(0, local_indices).to(device=device),
            batch.valence_labels.index_select(0, local_indices).to(device=device),
            batch.quadrant_labels.index_select(0, local_indices).to(device=device),
        )

    def _speech_compact_labels(
        self,
        batch: AlignedMultimodalBatch,
        indices: Tensor,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select and activity-mask speech targets with compact shape ``[Bs]``."""

        labels = self._compact_labels(batch, indices, device=device)
        threshold = self.config.speech_aux_min_activity_ratio
        if threshold is None:
            return labels
        ratios = batch.speech_activity_ratios
        observed = batch.speech_activity_observed
        if ratios is None or observed is None:
            raise RuntimeError(
                "speech auxiliary activity filtering requires observed "
                "speech activity diagnostics."
            )
        local_indices = indices.to(device=ratios.device)
        compact_ratios = ratios.index_select(0, local_indices)
        compact_observed = observed.index_select(0, local_indices)
        if not bool(compact_observed.all()):
            raise RuntimeError(
                "speech auxiliary activity filtering requires observed "
                "speech activity diagnostics."
            )
        supervision_valid = compact_observed & (compact_ratios > threshold)
        supervision_valid = supervision_valid.to(device=device)
        masked_labels = tuple(
            torch.where(
                supervision_valid,
                label,
                torch.full_like(label, batch.label_ignore_index),
            )
            for label in labels
        )
        return masked_labels[0], masked_labels[1], masked_labels[2]

    @staticmethod
    def _validate_compact(
        output: LightweightSpeechClassifierOutput | LightweightPhysioClassifierOutput,
        indices: Tensor,
        *,
        reference: Tensor,
        name: str,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        compact_size = indices.shape[0]
        sample_valid = output.sample_valid
        if (
            sample_valid.dtype != torch.bool
            or tuple(sample_valid.shape) != (compact_size,)
            or sample_valid.device != reference.device
            or not bool(sample_valid.all())
        ):
            raise ValueError(
                f"{name} compact sample_valid must be all-True bool "
                f"[{compact_size}] on the logits device."
            )
        for task_name, logits, class_count in (
            ("arousal", output.arousal_logits, 2),
            ("valence", output.valence_logits, 2),
        ):
            if (
                not logits.is_floating_point()
                or tuple(logits.shape) != (compact_size, class_count)
                or logits.dtype != reference.dtype
                or logits.device != reference.device
                or not bool(torch.isfinite(logits).all())
            ):
                raise ValueError(
                    f"{name} compact {task_name} logits must be finite floating "
                    f"[{compact_size}, {class_count}] with fused dtype/device."
                )
        quadrant_logits = output.quadrant_logits
        if quadrant_logits is not None and (
            not quadrant_logits.is_floating_point()
            or tuple(quadrant_logits.shape) != (compact_size, 4)
            or quadrant_logits.dtype != reference.dtype
            or quadrant_logits.device != reference.device
            or not bool(torch.isfinite(quadrant_logits).all())
        ):
            raise ValueError(
                f"{name} compact quadrant logits must be finite floating "
                f"[{compact_size}, 4] or None."
            )
        return output.arousal_logits, output.valence_logits, quadrant_logits

    def forward(
        self,
        model_output: MultimodalEmotionClassifierOutput,
        batch: AlignedMultimodalBatch,
        *,
        class_weights: EmotionTaskClassWeights | None = None,
    ) -> MultimodalLossOutput:
        """Aggregate enabled losses for one logical batch.

        Args:
            model_output: Final logits ``[B, 2]``, optional independent
                quadrant logits ``[B, 4]``, and retained compact predictions.
            batch: Exact logical batch of size ``B`` used for ``model_output``.
            class_weights: Optional immutable task weights ``[2]``, ``[2]``,
                and ``[4]``. Dtype/device must already match used logits.

        Returns:
            :class:`MultimodalLossOutput` containing thirteen scalar tensors
            ``[]`` plus the enabled valid-label count.

        Raises:
            TypeError: If public input types are wrong.
            ValueError: If shapes, labels, masks, dtypes, devices, finite
                values, or batch identity violate their contracts.
            RuntimeError: If an enabled quadrant task lacks its independent
                logits, or enabled compact output contradicts batch presence.

        Protocol ignores are retained. Fused unavailable rows are additionally
        ignored. Auxiliary targets are selected by compact batch indices; when
        configured, speech targets also require an observed activity ratio
        strictly above the threshold. Inputs are never modified.
        """
        _, reference = self._validate_inputs(model_output, batch)
        if class_weights is None:
            class_weights = EmotionTaskClassWeights()
        elif not isinstance(class_weights, EmotionTaskClassWeights):
            raise TypeError("class_weights must be EmotionTaskClassWeights or None.")
        for name, value, class_count in (
            ("arousal", class_weights.arousal, 2),
            ("valence", class_weights.valence, 2),
            ("quadrant", class_weights.quadrant, 4),
        ):
            if value is not None:
                _validate_class_weight_tensor(
                    value,
                    name=name,
                    class_count=class_count,
                )

        config = self.config
        weights = config.weights
        ignore_index = batch.label_ignore_index
        fused_arousal_labels, fused_valence_labels, fused_quadrant_labels = (
            self._fused_labels(
                batch,
                model_output.sample_valid,
                device=reference.device,
            )
        )
        fused = _binary_task_losses(
            model_output.arousal_logits,
            model_output.valence_logits,
            fused_arousal_labels,
            fused_valence_labels,
            class_weights,
            ignore_index=ignore_index,
            config=config,
            enabled=weights.fused > 0.0,
            name="fused",
        )
        fused_quadrant_loss, fused_quadrant_count = _quadrant_task_loss(
            model_output.quadrant_logits,
            fused_quadrant_labels,
            class_weights,
            ignore_index=ignore_index,
            config=config,
            enabled=weights.fused_quadrant > 0.0,
            name="fused",
            zero_reference=reference,
        )

        speech_arousal_loss = _graph_safe_zero(reference)
        speech_valence_loss = _graph_safe_zero(reference)
        speech_auxiliary_loss = speech_arousal_loss + speech_valence_loss
        speech_quadrant_loss = _graph_safe_zero(reference)
        speech_count = 0
        speech_quadrant_count = 0
        speech_enabled = (
            weights.speech_auxiliary > 0.0
            or weights.speech_quadrant > 0.0
        )
        scheduled = model_output.fusion_output.scheduled_outputs
        if speech_enabled and batch.speech is not None:
            compact = scheduled.speech_compact_output
            if not isinstance(
                compact,
                LightweightSpeechClassifierOutput,
            ):
                raise RuntimeError(
                    "enabled speech supervision requires "
                    "a supported speech classifier output."
                )
            indices = batch.speech.batch_indices
            speech_arousal, speech_valence, speech_quadrant = (
                self._validate_compact(
                    compact,
                    indices,
                    reference=reference,
                    name="speech",
                )
            )
            speech_labels = self._speech_compact_labels(
                batch,
                indices,
                device=reference.device,
            )
            speech_binary = _binary_task_losses(
                speech_arousal,
                speech_valence,
                speech_labels[0],
                speech_labels[1],
                class_weights,
                ignore_index=ignore_index,
                config=config,
                enabled=weights.speech_auxiliary > 0.0,
                name="speech",
            )
            speech_arousal_loss = speech_binary.arousal
            speech_valence_loss = speech_binary.valence
            speech_auxiliary_loss = speech_binary.combined
            speech_count = speech_binary.count
            (
                speech_quadrant_loss,
                speech_quadrant_count,
            ) = _quadrant_task_loss(
                speech_quadrant,
                speech_labels[2],
                class_weights,
                ignore_index=ignore_index,
                config=config,
                enabled=weights.speech_quadrant > 0.0,
                name="speech",
                zero_reference=reference,
            )
        elif speech_enabled and scheduled.speech_compact_output is not None:
            raise RuntimeError(
                "speech compact output exists while batch speech is absent."
            )

        physiology_arousal_loss = _graph_safe_zero(reference)
        physiology_valence_loss = _graph_safe_zero(reference)
        physiology_auxiliary_loss = (
            physiology_arousal_loss + physiology_valence_loss
        )
        physiology_quadrant_loss = _graph_safe_zero(reference)
        physiology_count = 0
        physiology_quadrant_count = 0
        physiology_enabled = (
            weights.physiology_auxiliary > 0.0
            or weights.physiology_quadrant > 0.0
        )
        if physiology_enabled and batch.physiology is not None:
            physiology_compact = scheduled.physiology_compact_output
            if not isinstance(
                physiology_compact,
                LightweightPhysioClassifierOutput,
            ):
                raise RuntimeError(
                    "enabled physiology supervision requires "
                    "a supported physiology classifier output."
                )
            indices = batch.physiology.batch_indices
            physiology_arousal, physiology_valence, physiology_quadrant = (
                self._validate_compact(
                    physiology_compact,
                    indices,
                    reference=reference,
                    name="physiology",
                )
            )
            physiology_labels = self._compact_labels(
                batch,
                indices,
                device=reference.device,
            )
            physiology_binary = _binary_task_losses(
                physiology_arousal,
                physiology_valence,
                physiology_labels[0],
                physiology_labels[1],
                class_weights,
                ignore_index=ignore_index,
                config=config,
                enabled=weights.physiology_auxiliary > 0.0,
                name="physiology",
            )
            physiology_arousal_loss = physiology_binary.arousal
            physiology_valence_loss = physiology_binary.valence
            physiology_auxiliary_loss = physiology_binary.combined
            physiology_count = physiology_binary.count
            (
                physiology_quadrant_loss,
                physiology_quadrant_count,
            ) = _quadrant_task_loss(
                physiology_quadrant,
                physiology_labels[2],
                class_weights,
                ignore_index=ignore_index,
                config=config,
                enabled=weights.physiology_quadrant > 0.0,
                name="physiology",
                zero_reference=reference,
            )
        elif physiology_enabled and scheduled.physiology_compact_output is not None:
            raise RuntimeError(
                "physiology compact output exists while batch physiology is absent."
            )

        total_loss = (
            weights.fused * fused.combined
            + weights.speech_auxiliary * speech_auxiliary_loss
            + weights.physiology_auxiliary * physiology_auxiliary_loss
            + weights.fused_quadrant * fused_quadrant_loss
            + weights.speech_quadrant * speech_quadrant_loss
            + weights.physiology_quadrant * physiology_quadrant_loss
        )
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError("multimodal total loss contains NaN or Inf.")
        return MultimodalLossOutput(
            total_loss=total_loss,
            fused_loss=fused.combined,
            fused_arousal_loss=fused.arousal,
            fused_valence_loss=fused.valence,
            fused_quadrant_loss=fused_quadrant_loss,
            speech_auxiliary_loss=speech_auxiliary_loss,
            speech_arousal_loss=speech_arousal_loss,
            speech_valence_loss=speech_valence_loss,
            speech_quadrant_loss=speech_quadrant_loss,
            physiology_auxiliary_loss=physiology_auxiliary_loss,
            physiology_arousal_loss=physiology_arousal_loss,
            physiology_valence_loss=physiology_valence_loss,
            physiology_quadrant_loss=physiology_quadrant_loss,
            active_target_count=(
                fused.count
                + fused_quadrant_count
                + speech_count
                + speech_quadrant_count
                + physiology_count
                + physiology_quadrant_count
            ),
        )


__all__ = [
    "ClassificationLossKind",
    "EmotionTaskClassWeights",
    "MultimodalLossOutput",
    "MultimodalLossWeights",
    "MultimodalObjectiveConfig",
    "MultimodalTrainingObjective",
]
