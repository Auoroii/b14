"""Compact modality scheduling and differentiable full-batch scatter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from emotion_model.data import AlignedMultimodalBatch
from emotion_model.physiology.lightweight import (
    LightweightPhysioClassifierOutput,
    LightweightPhysioEmotionClassifier,
)
from emotion_model.speech.lightweight_noise_conditioned import (
    LightweightNoiseConditionedSpeechClassifier,
    LightweightSpeechClassifierOutput,
)

SpeechClassifier = LightweightNoiseConditionedSpeechClassifier
PhysiologyClassifier = LightweightPhysioEmotionClassifier
SpeechClassifierOutput = LightweightSpeechClassifierOutput
PhysiologyClassifierOutput = LightweightPhysioClassifierOutput


def _require_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    return value


def _module_reference(module: nn.Module, *, name: str) -> Tensor:
    """Return one parameter defining a classifier's runtime device/dtype."""
    reference = next(module.parameters(), None)
    if reference is None:
        raise RuntimeError(f"{name} must contain at least one parameter.")
    return reference


def _runtime_tensor(
    tensor: Tensor,
    reference: Tensor,
    *,
    floating: bool,
) -> Tensor:
    """Move a compact CPU input only when its runtime placement differs."""
    target_dtype = reference.dtype if floating else tensor.dtype
    if tensor.device == reference.device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(
        device=reference.device,
        dtype=target_dtype,
        non_blocking=True,
    )


def _validate_indices(
    batch_indices: object,
    *,
    compact_size: int,
    batch_size: int | None,
    name: str,
) -> Tensor:
    indices = _require_tensor(batch_indices, name=name)
    if indices.ndim != 1:
        raise ValueError(
            f"{name} must have exact shape [Bm]; received {tuple(indices.shape)}."
        )
    if indices.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long; received {indices.dtype}.")
    if indices.shape[0] != compact_size:
        raise ValueError(
            f"{name} length must equal Bm={compact_size}; "
            f"received {indices.shape[0]}."
        )
    if compact_size > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError(f"{name} must be strictly increasing without duplicates.")
    if bool((indices < 0).any()):
        raise ValueError(f"{name} cannot contain negative indices.")
    if batch_size is not None and bool((indices >= batch_size).any()):
        raise ValueError(f"{name} values must lie in [0, {batch_size}).")
    return indices


def scatter_compact_rows(
    compact_values: Tensor,
    batch_indices: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    """Scatter compact floating rows into a zero-filled logical batch.

    Args:
        compact_values: Floating tensor ``[Bm, ...]`` with ``Bm > 0``.
        batch_indices: Strictly increasing ``torch.long`` tensor ``[Bm]``.
            It may remain on CPU while ``compact_values`` resides elsewhere.
        batch_size: Positive integer full-batch size ``B``; booleans fail.

    Returns:
        New tensor ``[B, ...]`` on the same device and with the same dtype as
        ``compact_values``. Rows selected by ``batch_indices`` equal the
        compact rows; all other rows are exactly zero. The operation preserves
        autograd connectivity to ``compact_values``.

    Raises:
        TypeError: If values/indices or their dtypes have invalid categories.
        ValueError: If shapes, sizes, ordering, uniqueness, or index bounds are
            invalid.

    Inputs are not modified, detached, or converted through CPU/NumPy.
    """
    values = _require_tensor(compact_values, name="compact_values")
    if values.ndim < 1:
        raise ValueError("compact_values must have shape [Bm, ...].")
    if not values.is_floating_point():
        raise TypeError("compact_values must be floating point.")
    compact_size = values.shape[0]
    if compact_size <= 0:
        raise ValueError("compact_values requires Bm > 0.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer, not bool.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    indices = _validate_indices(
        batch_indices,
        compact_size=compact_size,
        batch_size=batch_size,
        name="batch_indices",
    )
    local_indices = indices.to(device=values.device)
    output_shape = (batch_size, *values.shape[1:])
    return values.new_zeros(output_shape).index_copy(
        0,
        local_indices,
        values,
    )


@dataclass(frozen=True)
class ModalityAvailabilityMasks:
    """Partition full-batch modality availability into four boolean masks.

    Every field is a boolean tensor ``[B]`` on one device. ``speech_available``
    and ``physiology_available`` are the original full-batch declarations.
    ``both_available``, ``speech_only``, ``physiology_only``, and
    ``neither_available`` are the exact, disjoint, exhaustive logical
    combinations. Invalid types, shapes, devices, or formulas raise
    ``TypeError`` or ``ValueError``.
    """

    speech_available: Tensor
    physiology_available: Tensor
    both_available: Tensor
    speech_only: Tensor
    physiology_only: Tensor
    neither_available: Tensor

    def __post_init__(self) -> None:
        fields = (
            ("speech_available", self.speech_available),
            ("physiology_available", self.physiology_available),
            ("both_available", self.both_available),
            ("speech_only", self.speech_only),
            ("physiology_only", self.physiology_only),
            ("neither_available", self.neither_available),
        )
        reference_shape: tuple[int, ...] | None = None
        reference_device: torch.device | None = None
        for name, value in fields:
            tensor = _require_tensor(value, name=name)
            if tensor.ndim != 1 or tensor.shape[0] <= 0:
                raise ValueError(f"{name} must have non-empty shape [B].")
            if tensor.dtype != torch.bool:
                raise TypeError(f"{name} must be bool; received {tensor.dtype}.")
            if reference_shape is None:
                reference_shape = tuple(tensor.shape)
                reference_device = tensor.device
            elif (
                tuple(tensor.shape) != reference_shape
                or tensor.device != reference_device
            ):
                raise ValueError(
                    "all modality availability masks must share shape and device."
                )

        speech = self.speech_available
        physiology = self.physiology_available
        expected = (
            speech & physiology,
            speech & ~physiology,
            ~speech & physiology,
            ~speech & ~physiology,
        )
        actual = (
            self.both_available,
            self.speech_only,
            self.physiology_only,
            self.neither_available,
        )
        if any(not torch.equal(left, right) for left, right in zip(actual, expected)):
            raise ValueError(
                "modality combination masks must match the documented formulas."
            )
        combined_count = torch.stack(actual, dim=0).to(dtype=torch.int8).sum(dim=0)
        if not bool((combined_count == 1).all()):
            raise ValueError(
                "modality combination masks must be disjoint and exhaustive."
            )

    @classmethod
    def from_availability(
        cls,
        speech_available: Tensor,
        physiology_available: Tensor,
    ) -> ModalityAvailabilityMasks:
        """Create exact combination masks from boolean tensors ``[B]``.

        Args:
            speech_available: Boolean full-batch tensor ``[B]``.
            physiology_available: Boolean full-batch tensor ``[B]`` on the
                same device.

        Returns:
            A validated immutable availability partition. Every output tensor
            is newly allocated and retains ``True=available`` semantics.

        Raises:
            TypeError: If either input is not a boolean tensor.
            ValueError: If input shapes/devices differ or ``B`` is zero.
        """
        speech = _require_tensor(
            speech_available,
            name="speech_available",
        )
        physiology = _require_tensor(
            physiology_available,
            name="physiology_available",
        )
        if speech.dtype != torch.bool or physiology.dtype != torch.bool:
            raise TypeError("availability inputs must be boolean tensors.")
        if (
            speech.ndim != 1
            or speech.shape[0] <= 0
            or tuple(physiology.shape) != tuple(speech.shape)
            or physiology.device != speech.device
        ):
            raise ValueError(
                "availability inputs must share one non-empty shape [B] and device."
            )
        speech_copy = speech.clone()
        physiology_copy = physiology.clone()
        return cls(
            speech_available=speech_copy,
            physiology_available=physiology_copy,
            both_available=speech_copy & physiology_copy,
            speech_only=speech_copy & ~physiology_copy,
            physiology_only=~speech_copy & physiology_copy,
            neither_available=~speech_copy & ~physiology_copy,
        )


@dataclass(frozen=True)
class ScatteredModalityPrediction:
    """One modality's compact predictions scattered to a full batch.

    Attributes:
        embedding: Floating tensor ``[B, Dm]``.
        arousal_logits: Floating tensor ``[B, 2]``.
        valence_logits: Floating tensor ``[B, 2]``.
        arousal_probabilities: Floating tensor ``[B, 2]``.
        valence_probabilities: Floating tensor ``[B, 2]``.
        quadrant_probabilities: Floating tensor ``[B, 4]`` ordered as
            ``[LALV, HALV, LAHV, HAHV]``.
        quadrant_logits: Optional floating tensor ``[B, 4]``.
        reliability: Floating tensor ``[B, 1]``.
        sample_valid: Boolean CPU tensor ``[B]`` equal to this modality's full
            availability.
        batch_indices: Strictly increasing long CPU tensor ``[Bm]`` equal to
            ``sample_valid.nonzero()``.
        reliability_score_type: Non-empty public interpretation string.

    Available probability rows sum to one. Every floating unavailable row is
    an exact-zero sentinel. Invalid direct construction raises ``TypeError`` or
    ``ValueError``.
    """

    embedding: Tensor
    arousal_logits: Tensor
    valence_logits: Tensor
    arousal_probabilities: Tensor
    valence_probabilities: Tensor
    quadrant_probabilities: Tensor
    quadrant_logits: Tensor | None
    reliability: Tensor
    sample_valid: Tensor
    batch_indices: Tensor
    reliability_score_type: str

    def __post_init__(self) -> None:
        embedding = _require_tensor(self.embedding, name="embedding")
        if embedding.ndim != 2 or embedding.shape[0] <= 0 or embedding.shape[1] <= 0:
            raise ValueError("embedding must have non-empty shape [B, Dm].")
        if not embedding.is_floating_point():
            raise TypeError("embedding must be floating point.")
        batch_size = embedding.shape[0]
        float_fields: tuple[tuple[str, Tensor, tuple[int, ...]], ...] = (
            ("embedding", embedding, tuple(embedding.shape)),
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
            ("reliability", self.reliability, (batch_size, 1)),
        )
        for name, value, expected_shape in float_fields:
            tensor = _require_tensor(value, name=name)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"received {tuple(tensor.shape)}."
                )
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point.")
            if tensor.dtype != embedding.dtype or tensor.device != embedding.device:
                raise ValueError(
                    "all scattered floating outputs must share dtype and device."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values.")
        if self.quadrant_logits is not None:
            quadrant_logits = _require_tensor(
                self.quadrant_logits,
                name="quadrant_logits",
            )
            if tuple(quadrant_logits.shape) != (batch_size, 4):
                raise ValueError("quadrant_logits must have shape [B, 4] or be None.")
            if (
                not quadrant_logits.is_floating_point()
                or quadrant_logits.dtype != embedding.dtype
                or quadrant_logits.device != embedding.device
                or not bool(torch.isfinite(quadrant_logits).all())
            ):
                raise ValueError(
                    "quadrant_logits must be finite and share embedding "
                    "dtype/device."
                )

        sample_valid = _require_tensor(self.sample_valid, name="sample_valid")
        if sample_valid.dtype != torch.bool or tuple(sample_valid.shape) != (
            batch_size,
        ):
            raise ValueError("sample_valid must be bool with shape [B].")
        if sample_valid.device.type != "cpu":
            raise ValueError("sample_valid must remain on CPU.")
        valid_count = int(sample_valid.sum().item())
        if valid_count <= 0:
            raise ValueError(
                "ScatteredModalityPrediction requires at least one available row."
            )
        indices = _validate_indices(
            self.batch_indices,
            compact_size=valid_count,
            batch_size=batch_size,
            name="batch_indices",
        )
        if indices.device.type != "cpu":
            raise ValueError("batch_indices must remain on CPU.")
        expected_indices = torch.nonzero(sample_valid, as_tuple=False).flatten()
        if not torch.equal(indices, expected_indices):
            raise ValueError(
                "batch_indices must exactly equal sample_valid nonzero indices."
            )

        local_valid = sample_valid.to(device=embedding.device)
        self._validate_zero_unavailable(
            tuple((name, tensor) for name, tensor, _ in float_fields)
            + (
                ()
                if self.quadrant_logits is None
                else (("quadrant_logits", self.quadrant_logits),)
            ),
            local_valid=local_valid,
        )
        self._validate_probabilities(
            (
                ("arousal_probabilities", self.arousal_probabilities),
                ("valence_probabilities", self.valence_probabilities),
                ("quadrant_probabilities", self.quadrant_probabilities),
            ),
            local_valid=local_valid,
        )
        if not bool(
            (self.reliability >= 0).all() and (self.reliability <= 1).all()
        ):
            raise ValueError("reliability must lie in [0, 1].")
        if (
            not isinstance(self.reliability_score_type, str)
            or not self.reliability_score_type.strip()
        ):
            raise ValueError("reliability_score_type must be a non-empty string.")

    @staticmethod
    def _validate_zero_unavailable(
        fields: tuple[tuple[str, Tensor], ...],
        *,
        local_valid: Tensor,
    ) -> None:
        for name, tensor in fields:
            unavailable = tensor[~local_valid]
            if not torch.equal(unavailable, torch.zeros_like(unavailable)):
                raise ValueError(
                    f"{name} unavailable rows must be exact-zero sentinels."
                )

    @staticmethod
    def _validate_probabilities(
        fields: tuple[tuple[str, Tensor], ...],
        *,
        local_valid: Tensor,
    ) -> None:
        for name, tensor in fields:
            available = tensor[local_valid]
            if not bool((available >= 0).all() and (available <= 1).all()):
                raise ValueError(f"{name} available rows must lie in [0, 1].")
            expected = torch.ones(
                available.shape[0],
                dtype=available.dtype,
                device=available.device,
            )
            if not torch.allclose(
                available.sum(dim=-1),
                expected,
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(f"{name} available rows must sum to one.")


@dataclass(frozen=True)
class ScheduledModalityOutputs:
    """Classifier outputs for compact modalities and their full-batch views.

    ``batch`` is the original :class:`AlignedMultimodalBatch`. Compact output
    objects are the exact classifier return objects. ``speech`` and
    ``physiology`` contain new full-batch tensors or are ``None`` when the
    corresponding availability mask is all ``False``. No fused feature,
    fused logits, or cross-modal prediction is represented.
    """

    batch: AlignedMultimodalBatch
    availability: ModalityAvailabilityMasks
    speech_compact_output: SpeechClassifierOutput | None
    physiology_compact_output: PhysiologyClassifierOutput | None
    speech: ScatteredModalityPrediction | None
    physiology: ScatteredModalityPrediction | None

    def __post_init__(self) -> None:
        if not isinstance(self.batch, AlignedMultimodalBatch):
            raise TypeError("batch must be AlignedMultimodalBatch.")
        if not isinstance(self.availability, ModalityAvailabilityMasks):
            raise TypeError("availability must be ModalityAvailabilityMasks.")
        if not torch.equal(
            self.availability.speech_available,
            self.batch.speech_available,
        ) or not torch.equal(
            self.availability.physiology_available,
            self.batch.physiology_available,
        ):
            raise ValueError(
                "availability must directly match the input full batch."
            )
        self._validate_modality_pair(
            availability=self.batch.speech_available,
            compact=self.speech_compact_output,
            scattered=self.speech,
            compact_types=(LightweightSpeechClassifierOutput,),
            name="speech",
        )
        self._validate_modality_pair(
            availability=self.batch.physiology_available,
            compact=self.physiology_compact_output,
            scattered=self.physiology,
            compact_types=(LightweightPhysioClassifierOutput,),
            name="physiology",
        )

    @staticmethod
    def _validate_modality_pair(
        *,
        availability: Tensor,
        compact: object,
        scattered: ScatteredModalityPrediction | None,
        compact_types: tuple[type[object], ...],
        name: str,
    ) -> None:
        if not bool(availability.any()):
            if compact is not None or scattered is not None:
                raise ValueError(
                    f"{name} compact and scattered outputs must both be None "
                    "when unavailable."
                )
            return
        if not isinstance(compact, compact_types):
            expected_names = " or ".join(
                compact_type.__name__ for compact_type in compact_types
            )
            raise TypeError(
                f"{name}_compact_output must be {expected_names}."
            )
        if not isinstance(scattered, ScatteredModalityPrediction):
            raise TypeError(
                f"{name} scattered output must be ScatteredModalityPrediction."
            )
        if not torch.equal(scattered.sample_valid, availability):
            raise ValueError(
                f"{name}.sample_valid must equal full-batch availability."
            )


@dataclass(frozen=True)
class _CompactPredictionFields:
    embedding: Tensor
    arousal_logits: Tensor
    valence_logits: Tensor
    arousal_probabilities: Tensor
    valence_probabilities: Tensor
    quadrant_probabilities: Tensor
    quadrant_logits: Tensor | None
    reliability: Tensor
    sample_valid: Tensor
    reliability_score_type: str
    temporal_attention_weights: Tensor


def _validate_compact_prediction(
    fields: _CompactPredictionFields,
    *,
    compact_size: int,
    classifier_name: str,
) -> None:
    embedding = _require_tensor(
        fields.embedding,
        name=f"{classifier_name}.embedding",
    )
    if (
        embedding.ndim != 2
        or tuple(embedding.shape[:1]) != (compact_size,)
        or embedding.shape[1] <= 0
    ):
        raise RuntimeError(
            f"{classifier_name} embedding must have shape [Bm, Dm] with "
            f"Bm={compact_size}; received {tuple(embedding.shape)}."
        )
    if not embedding.is_floating_point():
        raise RuntimeError(f"{classifier_name} embedding must be floating point.")
    expected_fields: tuple[tuple[str, Tensor, tuple[int, ...]], ...] = (
        ("arousal_logits", fields.arousal_logits, (compact_size, 2)),
        ("valence_logits", fields.valence_logits, (compact_size, 2)),
        (
            "arousal_probabilities",
            fields.arousal_probabilities,
            (compact_size, 2),
        ),
        (
            "valence_probabilities",
            fields.valence_probabilities,
            (compact_size, 2),
        ),
        (
            "quadrant_probabilities",
            fields.quadrant_probabilities,
            (compact_size, 4),
        ),
        ("reliability", fields.reliability, (compact_size, 1)),
    )
    all_float_tensors: list[tuple[str, Tensor]] = [("embedding", embedding)]
    for name, value, shape in expected_fields:
        tensor = _require_tensor(value, name=f"{classifier_name}.{name}")
        if tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"{classifier_name} {name} must have shape {shape}; "
                f"received {tuple(tensor.shape)}."
            )
        all_float_tensors.append((name, tensor))
    if fields.quadrant_logits is not None:
        quadrant_logits = _require_tensor(
            fields.quadrant_logits,
            name=f"{classifier_name}.quadrant_logits",
        )
        if tuple(quadrant_logits.shape) != (compact_size, 4):
            raise RuntimeError(
                f"{classifier_name} quadrant_logits must have shape [Bm, 4] "
                "or be None."
            )
        all_float_tensors.append(("quadrant_logits", quadrant_logits))
    attention = _require_tensor(
        fields.temporal_attention_weights,
        name=f"{classifier_name}.temporal_attention_weights",
    )
    if (
        attention.ndim != 2
        or attention.shape[0] != compact_size
        or attention.shape[1] <= 0
    ):
        raise RuntimeError(
            f"{classifier_name} temporal_attention_weights must have shape [Bm, T]."
        )
    all_float_tensors.append(("temporal_attention_weights", attention))
    for name, tensor in all_float_tensors:
        if (
            not tensor.is_floating_point()
            or tensor.dtype != embedding.dtype
            or tensor.device != embedding.device
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(
                f"{classifier_name} {name} must be finite floating point and "
                "share embedding dtype/device."
            )

    sample_valid = _require_tensor(
        fields.sample_valid,
        name=f"{classifier_name}.sample_valid",
    )
    if sample_valid.dtype != torch.bool or tuple(sample_valid.shape) != (
        compact_size,
    ):
        raise RuntimeError(
            f"{classifier_name} sample_valid must be bool with shape [Bm]."
        )
    if sample_valid.device != embedding.device:
        raise RuntimeError(
            f"{classifier_name} sample_valid must share output device."
        )
    if not bool(sample_valid.all()):
        raise RuntimeError(
            f"{classifier_name} compact sample_valid must be all True."
        )
    for name, probabilities in (
        ("arousal_probabilities", fields.arousal_probabilities),
        ("valence_probabilities", fields.valence_probabilities),
        ("quadrant_probabilities", fields.quadrant_probabilities),
    ):
        if not bool(
            (probabilities >= 0).all() and (probabilities <= 1).all()
        ) or not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(
                compact_size,
                dtype=probabilities.dtype,
                device=probabilities.device,
            ),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                f"{classifier_name} {name} rows must be probabilities summing to one."
            )
    if not bool(
        (fields.reliability >= 0).all() and (fields.reliability <= 1).all()
    ):
        raise RuntimeError(f"{classifier_name} reliability must lie in [0, 1].")
    if (
        not isinstance(fields.reliability_score_type, str)
        or not fields.reliability_score_type.strip()
    ):
        raise RuntimeError(
            f"{classifier_name} reliability_score_type must be non-empty."
        )


def _speech_fields(
    output: SpeechClassifierOutput,
    *,
    compact_size: int,
) -> _CompactPredictionFields:
    fields = _CompactPredictionFields(
        embedding=output.speech_embedding,
        arousal_logits=output.arousal_logits,
        valence_logits=output.valence_logits,
        arousal_probabilities=output.arousal_probabilities,
        valence_probabilities=output.valence_probabilities,
        quadrant_probabilities=output.quadrant_probabilities,
        quadrant_logits=output.quadrant_logits,
        reliability=output.reliability,
        sample_valid=output.sample_valid,
        reliability_score_type=output.reliability_score_type,
        temporal_attention_weights=output.temporal_attention_weights,
    )
    try:
        _validate_compact_prediction(
            fields,
            compact_size=compact_size,
            classifier_name="speech classifier",
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"speech classifier returned a malformed compact output: {error}"
        ) from error
    return fields


def _physiology_fields(
    output: PhysiologyClassifierOutput,
    *,
    compact_size: int,
) -> _CompactPredictionFields:
    fields = _CompactPredictionFields(
        embedding=output.physio_embedding,
        arousal_logits=output.arousal_logits,
        valence_logits=output.valence_logits,
        arousal_probabilities=output.arousal_probabilities,
        valence_probabilities=output.valence_probabilities,
        quadrant_probabilities=output.quadrant_probabilities,
        quadrant_logits=output.quadrant_logits,
        reliability=output.reliability,
        sample_valid=output.sample_valid,
        reliability_score_type=output.reliability_score_type,
        temporal_attention_weights=output.temporal_attention_weights,
    )
    try:
        _validate_compact_prediction(
            fields,
            compact_size=compact_size,
            classifier_name="physiology classifier",
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"physiology classifier returned a malformed compact output: {error}"
        ) from error
    return fields


def _scatter_prediction(
    fields: _CompactPredictionFields,
    *,
    batch_indices: Tensor,
    batch_size: int,
    sample_valid: Tensor,
) -> ScatteredModalityPrediction:
    return ScatteredModalityPrediction(
        embedding=scatter_compact_rows(
            fields.embedding,
            batch_indices,
            batch_size=batch_size,
        ),
        arousal_logits=scatter_compact_rows(
            fields.arousal_logits,
            batch_indices,
            batch_size=batch_size,
        ),
        valence_logits=scatter_compact_rows(
            fields.valence_logits,
            batch_indices,
            batch_size=batch_size,
        ),
        arousal_probabilities=scatter_compact_rows(
            fields.arousal_probabilities,
            batch_indices,
            batch_size=batch_size,
        ),
        valence_probabilities=scatter_compact_rows(
            fields.valence_probabilities,
            batch_indices,
            batch_size=batch_size,
        ),
        quadrant_probabilities=scatter_compact_rows(
            fields.quadrant_probabilities,
            batch_indices,
            batch_size=batch_size,
        ),
        quadrant_logits=(
            None
            if fields.quadrant_logits is None
            else scatter_compact_rows(
                fields.quadrant_logits,
                batch_indices,
                batch_size=batch_size,
            )
        ),
        reliability=scatter_compact_rows(
            fields.reliability,
            batch_indices,
            batch_size=batch_size,
        ),
        sample_valid=sample_valid.clone(),
        batch_indices=batch_indices.clone(),
        reliability_score_type=fields.reliability_score_type,
    )


class MultimodalBatchScheduler(nn.Module):
    """Run compact unimodal classifiers and scatter outputs to ``[B, ...]``.

    Args:
        speech_classifier: Optional registered speech classifier.
        physiology_classifier: Optional registered physiology classifier.
            At least one classifier is required; a missing classifier supports
            a true single-branch scheduler and rejects matching batch inputs.

    The scheduler adds no trainable architecture, loss, modality fusion, or
    reliability weighting. A missing compact subbatch skips that classifier.
    Available compact rows execute exactly once and are scattered through
    differentiable zero-initialized full-batch tensors.
    """

    def __init__(
        self,
        speech_classifier: SpeechClassifier | None,
        physiology_classifier: PhysiologyClassifier | None,
    ) -> None:
        super().__init__()
        if speech_classifier is not None and not isinstance(
            speech_classifier,
            LightweightNoiseConditionedSpeechClassifier,
        ):
            raise TypeError(
                "speech_classifier must be a supported speech classifier; "
                f"received {type(speech_classifier).__name__}."
            )
        if physiology_classifier is not None and not isinstance(
            physiology_classifier,
            LightweightPhysioEmotionClassifier,
        ):
            raise TypeError(
                "physiology_classifier must be a supported physiology classifier; "
                f"received {type(physiology_classifier).__name__}."
            )
        if speech_classifier is None and physiology_classifier is None:
            raise ValueError("at least one modality classifier must be provided.")
        self.speech_classifier = speech_classifier
        self.physiology_classifier = physiology_classifier

    def forward(
        self,
        batch: AlignedMultimodalBatch,
    ) -> ScheduledModalityOutputs:
        """Route compact subbatches and return full-batch modality predictions.

        Args:
            batch: Valid :class:`AlignedMultimodalBatch` with logical size
                ``B`` and optional compact speech/physiology subbatches.

        Returns:
            :class:`ScheduledModalityOutputs` retaining the exact input batch
            and compact classifier output objects. Present modalities have
            newly scattered tensors ``[B, Dm]``, ``[B, 2]``, ``[B, 4]``, and
            ``[B, 1]``; unavailable rows are exact zeros.

        Raises:
            TypeError: If the batch or a classifier output has the wrong type.
            ValueError: If the input batch contract is invalid.
            RuntimeError: If a classifier returns a malformed compact output.

        No labels are passed and no objective, fusion, or parameter update is
        computed. The input batch and compact tensors are not modified.
        """
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError("batch must be AlignedMultimodalBatch.")
        try:
            batch.__post_init__()
        except (TypeError, ValueError) as error:
            raise type(error)(f"batch violates its contract: {error}") from error
        availability = ModalityAvailabilityMasks.from_availability(
            batch.speech_available,
            batch.physiology_available,
        )
        batch_size = len(batch.records)

        speech_compact_output: SpeechClassifierOutput | None = None
        speech_prediction: ScatteredModalityPrediction | None = None
        if batch.speech is not None:
            if self.speech_classifier is None:
                raise RuntimeError(
                    "batch contains speech but this scheduler has no speech classifier."
                )
            speech_compact = batch.speech
            speech_reference = _module_reference(
                self.speech_classifier,
                name="speech_classifier",
            )
            runtime_waveform = _runtime_tensor(
                speech_compact.waveform,
                speech_reference,
                floating=True,
            )
            runtime_attention_mask = _runtime_tensor(
                speech_compact.attention_mask,
                speech_reference,
                floating=False,
            )
            runtime_activity_mask = (
                None
                if speech_compact.activity_mask is None
                else _runtime_tensor(
                    speech_compact.activity_mask,
                    speech_reference,
                    floating=False,
                )
            )
            raw_speech_output = self.speech_classifier(
                waveform=runtime_waveform,
                speech_attention_mask=runtime_attention_mask,
                speech_activity_mask=runtime_activity_mask,
                sample_rate=speech_compact.sample_rate_hz,
            )
            if not isinstance(raw_speech_output, LightweightSpeechClassifierOutput):
                raise RuntimeError(
                    "speech classifier must return LightweightSpeechClassifierOutput; "
                    f"received {type(raw_speech_output).__name__}."
                )
            speech_compact_output = raw_speech_output
            speech_fields = _speech_fields(
                raw_speech_output,
                compact_size=speech_compact.batch_indices.shape[0],
            )
            speech_prediction = _scatter_prediction(
                speech_fields,
                batch_indices=speech_compact.batch_indices,
                batch_size=batch_size,
                sample_valid=batch.speech_available,
            )

        physiology_compact_output: PhysiologyClassifierOutput | None = None
        physiology_prediction: ScatteredModalityPrediction | None = None
        if batch.physiology is not None:
            if self.physiology_classifier is None:
                raise RuntimeError(
                    "batch contains physiology but this scheduler has no physiology "
                    "classifier."
                )
            physiology_compact = batch.physiology
            physiology_reference = _module_reference(
                self.physiology_classifier,
                name="physiology_classifier",
            )
            physiology_arguments = {
                "physio_input": _runtime_tensor(
                    physiology_compact.physio_input,
                    physiology_reference,
                    floating=True,
                ),
                "physio_valid_mask": _runtime_tensor(
                    physiology_compact.physio_valid_mask,
                    physiology_reference,
                    floating=False,
                ),
                "channel_names": physiology_compact.channel_names,
                "physio_time_mask": _runtime_tensor(
                    physiology_compact.physio_time_mask,
                    physiology_reference,
                    floating=False,
                ),
                "physio_channel_mask": _runtime_tensor(
                    physiology_compact.physio_channel_mask,
                    physiology_reference,
                    floating=False,
                ),
                "physio_quality_features": _runtime_tensor(
                    physiology_compact.physio_quality_features,
                    physiology_reference,
                    floating=True,
                ),
            }
            if physiology_compact.ecg_values is not None:
                assert physiology_compact.ecg_valid_mask is not None
                assert physiology_compact.ecg_timeline_mask is not None
                physiology_arguments.update(
                    {
                        "ecg_values": _runtime_tensor(
                            physiology_compact.ecg_values,
                            physiology_reference,
                            floating=True,
                        ),
                        "ecg_valid_mask": _runtime_tensor(
                            physiology_compact.ecg_valid_mask,
                            physiology_reference,
                            floating=False,
                        ),
                        "ecg_timeline_mask": _runtime_tensor(
                            physiology_compact.ecg_timeline_mask,
                            physiology_reference,
                            floating=False,
                        ),
                    }
                )
            raw_physiology_output = self.physiology_classifier(
                **physiology_arguments  # type: ignore[arg-type]
            )
            if not isinstance(
                raw_physiology_output,
                LightweightPhysioClassifierOutput,
            ):
                raise RuntimeError(
                    "physiology classifier must return "
                    "LightweightPhysioClassifierOutput; "
                    f"received {type(raw_physiology_output).__name__}."
                )
            physiology_compact_output = raw_physiology_output
            physiology_fields = _physiology_fields(
                raw_physiology_output,
                compact_size=physiology_compact.batch_indices.shape[0],
            )
            physiology_prediction = _scatter_prediction(
                physiology_fields,
                batch_indices=physiology_compact.batch_indices,
                batch_size=batch_size,
                sample_valid=batch.physiology_available,
            )

        return ScheduledModalityOutputs(
            batch=batch,
            availability=availability,
            speech_compact_output=speech_compact_output,
            physiology_compact_output=physiology_compact_output,
            speech=speech_prediction,
            physiology=physiology_prediction,
        )


__all__ = [
    "ModalityAvailabilityMasks",
    "MultimodalBatchScheduler",
    "ScatteredModalityPrediction",
    "ScheduledModalityOutputs",
    "scatter_compact_rows",
]
