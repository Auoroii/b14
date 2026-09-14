"""Tests for compact modality scheduling and differentiable scatter-back."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn

import emotion_model.multimodal.routing as routing_module
from emotion_model.common import derive_quadrant_probabilities
from emotion_model.data import (
    AlignedMultimodalBatch,
    EmotionScores,
    MultimodalWindowRecord,
    PhysioSubBatch,
    SpeechSubBatch,
    TimedSourceRef,
    TimeInterval,
)
from emotion_model.multimodal import (
    ModalityAvailabilityMasks,
    MultimodalBatchScheduler,
    ScatteredModalityPrediction,
    ScheduledModalityOutputs,
    class_weighted_cross_entropy,
    scatter_compact_rows,
)
from emotion_model.physiology import (
    LightweightPhysioClassifierOutput,
    LightweightPhysioEmotionClassifier,
)
from emotion_model.speech import (
    LightweightNoiseConditionedSpeechClassifier,
    LightweightSpeechClassifierOutput,
)


class TinySpeechClassifier(LightweightNoiseConditionedSpeechClassifier):
    """Small local classifier double preserving the public output contract."""

    def __init__(
        self,
        embedding_dim: int = 3,
        *,
        quadrant_head: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.input_projection = nn.Linear(1, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.arousal_head = nn.Linear(embedding_dim, 2)
        self.valence_head = nn.Linear(embedding_dim, 2)
        self.quadrant_head = (
            nn.Linear(embedding_dim, 4) if quadrant_head else None
        )
        self.reliability_head = nn.Linear(embedding_dim, 1)
        self.calls: list[dict[str, object]] = []
        self.last_output: LightweightSpeechClassifierOutput | None = None
        self.override_output: object | None = None

    def get_extra_state(self) -> dict[str, int]:
        """Return the test double's embedding-width checkpoint contract."""
        return {"embedding_dim": self.input_projection.out_features}

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoint state from an incompatible test double."""
        if state != self.get_extra_state():
            raise RuntimeError(
                "TinySpeechClassifier checkpoint configuration mismatch."
            )

    def forward(
        self,
        waveform: Tensor,
        speech_attention_mask: Tensor,
        *,
        speech_activity_mask: Tensor | None = None,
        sample_rate: int | None = None,
    ) -> LightweightSpeechClassifierOutput:
        """Return finite predictions for input shapes ``[B,L]``."""
        self.calls.append(
            {
                "waveform": waveform,
                "speech_attention_mask": speech_attention_mask,
                "speech_activity_mask": speech_activity_mask,
                "sample_rate": sample_rate,
            }
        )
        valid = speech_attention_mask.to(dtype=waveform.dtype)
        summary = (waveform * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1,
            keepdim=True,
        )
        embedding = self.dropout(torch.tanh(self.input_projection(summary)))
        arousal_logits = self.arousal_head(embedding)
        valence_logits = self.valence_head(embedding)
        arousal_probabilities = torch.softmax(arousal_logits, dim=-1)
        valence_probabilities = torch.softmax(valence_logits, dim=-1)
        quadrant_probabilities = derive_quadrant_probabilities(
            arousal_probabilities,
            valence_probabilities,
        )
        sample_valid = torch.ones(
            waveform.shape[0],
            dtype=torch.bool,
            device=waveform.device,
        )
        reliability = torch.sigmoid(self.reliability_head(embedding))
        layer_weights = torch.full(
                (waveform.shape[0], 12),
                1.0 / 12.0,
                dtype=waveform.dtype,
                device=waveform.device,
        )
        feature_mask = sample_valid.unsqueeze(1)
        result = LightweightSpeechClassifierOutput(
            arousal_logits=arousal_logits,
            valence_logits=valence_logits,
            arousal_probabilities=arousal_probabilities,
            valence_probabilities=valence_probabilities,
            quadrant_probabilities=quadrant_probabilities,
            speech_embedding=embedding,
            base_speech_embedding=embedding,
            noise_embedding=embedding,
            emotion_sequence=embedding.unsqueeze(1),
            noise_sequence=embedding.unsqueeze(1),
            gamma_bounded=torch.zeros_like(embedding),
            beta_bounded=torch.zeros_like(embedding),
            emotion_layer_weights=layer_weights,
            noise_layer_weights=layer_weights,
            temporal_attention_weights=torch.ones(
                (waveform.shape[0], 1),
                dtype=waveform.dtype,
                device=waveform.device,
            ),
            feature_attention_mask=feature_mask,
            speech_activity_ratio=torch.ones(
                (waveform.shape[0], 1),
                dtype=waveform.dtype,
                device=waveform.device,
            ),
            reliability=reliability,
            sample_valid=sample_valid,
            reliability_score_type="availability_indicator",
            denoised_emotion_sequence=embedding.unsqueeze(1),
            differential_response=embedding.unsqueeze(1),
            differential_gate=torch.zeros_like(embedding).unsqueeze(1),
            differential_lambda=waveform.new_tensor(0.5),
            quadrant_logits=(
                None
                if self.quadrant_head is None
                else self.quadrant_head(embedding)
            ),
        )
        self.last_output = result
        return (
            result
            if self.override_output is None
            else cast(LightweightSpeechClassifierOutput, self.override_output)
        )


class TinyPhysioClassifier(LightweightPhysioEmotionClassifier):
    """Small local physiology classifier double with registered parameters."""

    def __init__(
        self,
        embedding_dim: int = 4,
        *,
        quadrant_head: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.input_projection = nn.Linear(1, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.arousal_head = nn.Linear(embedding_dim, 2)
        self.valence_head = nn.Linear(embedding_dim, 2)
        self.quadrant_head = (
            nn.Linear(embedding_dim, 4) if quadrant_head else None
        )
        self.reliability_head = nn.Linear(embedding_dim, 1)
        self.calls: list[dict[str, object]] = []
        self.last_output: LightweightPhysioClassifierOutput | None = None
        self.override_output: object | None = None

    def get_extra_state(self) -> dict[str, int]:
        """Return the test double's embedding-width checkpoint contract."""
        return {"embedding_dim": self.input_projection.out_features}

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoint state from an incompatible test double."""
        if state != self.get_extra_state():
            raise RuntimeError(
                "TinyPhysioClassifier checkpoint configuration mismatch."
            )

    def forward(
        self,
        physio_input: Tensor,
        physio_valid_mask: Tensor,
        *,
        channel_names: tuple[str, ...],
        physio_time_mask: Tensor | None = None,
        physio_channel_mask: Tensor | None = None,
        physio_quality_features: Tensor | None = None,
        physio_quality_prior: Tensor | None = None,
    ) -> LightweightPhysioClassifierOutput:
        """Return finite predictions for input shapes ``[B,T,C]``."""
        self.calls.append(
            {
                "physio_input": physio_input,
                "physio_valid_mask": physio_valid_mask,
                "channel_names": channel_names,
                "physio_time_mask": physio_time_mask,
                "physio_channel_mask": physio_channel_mask,
                "physio_quality_features": physio_quality_features,
                "physio_quality_prior": physio_quality_prior,
            }
        )
        valid = physio_valid_mask.to(dtype=physio_input.dtype)
        summary = (physio_input * valid).sum(dim=(1, 2), keepdim=False)
        summary = summary / valid.sum(dim=(1, 2)).clamp_min(1)
        embedding = self.dropout(
            torch.tanh(self.input_projection(summary.unsqueeze(1)))
        )
        arousal_logits = self.arousal_head(embedding)
        valence_logits = self.valence_head(embedding)
        arousal_probabilities = torch.softmax(arousal_logits, dim=-1)
        valence_probabilities = torch.softmax(valence_logits, dim=-1)
        quadrant_probabilities = derive_quadrant_probabilities(
            arousal_probabilities,
            valence_probabilities,
        )
        sample_valid = torch.ones(
            physio_input.shape[0],
            dtype=torch.bool,
            device=physio_input.device,
        )
        reliability = torch.sigmoid(self.reliability_head(embedding))
        channel_count = physio_input.shape[2]
        result = LightweightPhysioClassifierOutput(
            arousal_logits=arousal_logits,
            valence_logits=valence_logits,
            arousal_probabilities=arousal_probabilities,
            valence_probabilities=valence_probabilities,
            quadrant_probabilities=quadrant_probabilities,
            physio_embedding=embedding,
            channel_features=embedding[:, None, None, :].expand(
                -1, 1, channel_count, -1
            ),
            channel_statistics=embedding[:, None, :].expand(
                -1, channel_count, -1
            ),
            channel_available=physio_valid_mask.any(dim=1),
            temporal_attention_weights=torch.ones(
                (physio_input.shape[0], 1),
                dtype=physio_input.dtype,
                device=physio_input.device,
            ),
            sample_valid=sample_valid,
            reliability=reliability,
            reliability_score_type="availability_indicator",
            quadrant_logits=(
                None
                if self.quadrant_head is None
                else self.quadrant_head(embedding)
            ),
        )
        self.last_output = result
        return (
            result
            if self.override_output is None
            else cast(LightweightPhysioClassifierOutput, self.override_output)
        )


def _record(index: int) -> MultimodalWindowRecord:
    """Create one immutable record for a logical batch row."""
    return MultimodalWindowRecord(
        f"sample-{index}",
        f"participant-{index}",
        f"session-{index}",
        TimeInterval(0.0, 1.0),
        EmotionScores(2.0, 4.0),
        TimedSourceRef("unused-source", TimeInterval(0.0, 1.0)),
        (),
    )


def _batch(
    patterns: tuple[tuple[bool, bool], ...] = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ),
    *,
    dtype: torch.dtype = torch.float32,
) -> AlignedMultimodalBatch:
    """Build a valid full batch directly, without loading or collation."""
    batch_size = len(patterns)
    speech_available = torch.tensor(
        [speech for speech, _ in patterns],
        dtype=torch.bool,
    )
    physiology_available = torch.tensor(
        [physiology for _, physiology in patterns],
        dtype=torch.bool,
    )
    speech_indices = torch.nonzero(speech_available, as_tuple=False).flatten()
    physiology_indices = torch.nonzero(
        physiology_available,
        as_tuple=False,
    ).flatten()
    speech: SpeechSubBatch | None
    if speech_indices.numel() == 0:
        speech = None
    else:
        speech_count = speech_indices.shape[0]
        speech = SpeechSubBatch(
            waveform=torch.arange(
                1,
                1 + speech_count * 3,
                dtype=dtype,
            ).reshape(speech_count, 3),
            attention_mask=torch.ones(
                (speech_count, 3),
                dtype=torch.bool,
            ),
            sequence_lengths=torch.full(
                (speech_count,),
                3,
                dtype=torch.long,
            ),
            sample_rate_hz=16000,
            batch_indices=speech_indices,
        )
    physiology: PhysioSubBatch | None
    if physiology_indices.numel() == 0:
        physiology = None
    else:
        physiology_count = physiology_indices.shape[0]
        physio_input = torch.arange(
            1,
            1 + physiology_count * 4,
            dtype=dtype,
        ).reshape(physiology_count, 2, 2)
        valid_mask = torch.ones_like(physio_input, dtype=torch.bool)
        quality = torch.tensor(
            [[[1.0, 0.0, 0.1, 0.2, 0.3, 1.0]] * 2],
            dtype=dtype,
        ).expand(physiology_count, -1, -1).clone()
        physiology = PhysioSubBatch(
            physio_input=physio_input,
            physio_valid_mask=valid_mask,
            physio_time_mask=valid_mask.any(dim=2),
            physio_channel_mask=valid_mask.any(dim=1),
            physio_timestamps_seconds=torch.tensor(
                [[0.0, 0.5]] * physiology_count,
                dtype=torch.float64,
            ),
            physio_timeline_mask=torch.ones(
                (physiology_count, 2),
                dtype=torch.bool,
            ),
            timeline_lengths=torch.full(
                (physiology_count,),
                2,
                dtype=torch.long,
            ),
            physio_channel_quality=quality,
            physio_quality_features=quality.reshape(
                physiology_count,
                12,
            ).clone(),
            channel_names=("eda", "temp"),
            batch_indices=physiology_indices,
        )
    return AlignedMultimodalBatch(
        records=tuple(_record(index) for index in range(batch_size)),
        raw_arousal=torch.full((batch_size,), 2.0, dtype=torch.float64),
        raw_valence=torch.full((batch_size,), 4.0, dtype=torch.float64),
        arousal_labels=torch.zeros(batch_size, dtype=torch.long),
        valence_labels=torch.ones(batch_size, dtype=torch.long),
        quadrant_labels=torch.full((batch_size,), 2, dtype=torch.long),
        label_ignore_index=-100,
        speech_available=speech_available,
        physiology_available=physiology_available,
        speech=speech,
        physiology=physiology,
    )


def _scheduler(
    *,
    speech_quadrant: bool = False,
    physiology_quadrant: bool = False,
    speech_dim: int = 3,
    physiology_dim: int = 4,
) -> MultimodalBatchScheduler:
    """Create a scheduler with small registered classifier doubles."""
    return MultimodalBatchScheduler(
        TinySpeechClassifier(speech_dim, quadrant_head=speech_quadrant),
        TinyPhysioClassifier(
            physiology_dim,
            quadrant_head=physiology_quadrant,
        ),
    )


def _scattered_prediction(
    *,
    quadrant_logits: bool = False,
) -> ScatteredModalityPrediction:
    """Create one valid scattered prediction with two available rows."""
    indices = torch.tensor([0, 2], dtype=torch.long)
    sample_valid = torch.tensor([True, False, True, False])
    embedding = scatter_compact_rows(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        indices,
        batch_size=4,
    )
    arousal_logits = scatter_compact_rows(
        torch.tensor([[0.2, 0.8], [0.7, 0.3]]),
        indices,
        batch_size=4,
    )
    valence_logits = scatter_compact_rows(
        torch.tensor([[0.4, 0.6], [0.1, 0.9]]),
        indices,
        batch_size=4,
    )
    arousal_probabilities = scatter_compact_rows(
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        indices,
        batch_size=4,
    )
    valence_probabilities = scatter_compact_rows(
        torch.tensor([[0.8, 0.2], [0.25, 0.75]]),
        indices,
        batch_size=4,
    )
    quadrant_probabilities = scatter_compact_rows(
        derive_quadrant_probabilities(
            torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
            torch.tensor([[0.8, 0.2], [0.25, 0.75]]),
        ),
        indices,
        batch_size=4,
    )
    return ScatteredModalityPrediction(
        embedding=embedding,
        arousal_logits=arousal_logits,
        valence_logits=valence_logits,
        arousal_probabilities=arousal_probabilities,
        valence_probabilities=valence_probabilities,
        quadrant_probabilities=quadrant_probabilities,
        quadrant_logits=(
            scatter_compact_rows(
                torch.ones((2, 4)),
                indices,
                batch_size=4,
            )
            if quadrant_logits
            else None
        ),
        reliability=scatter_compact_rows(
            torch.tensor([[0.2], [0.9]]),
            indices,
            batch_size=4,
        ),
        sample_valid=sample_valid,
        batch_indices=indices,
        reliability_score_type="learned_gate_score",
    )


def test_scatter_compact_rows_two_and_three_dimensional_reference() -> None:
    """Scatter arbitrary trailing dimensions into exact zero-filled rows."""
    values_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    indices = torch.tensor([0, 2])
    output_2d = scatter_compact_rows(values_2d, indices, batch_size=4)
    assert torch.equal(
        output_2d,
        torch.tensor(
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]
        ),
    )

    values_3d = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    output_3d = scatter_compact_rows(values_3d, indices, batch_size=3)
    assert output_3d.shape == (3, 2, 2)
    assert torch.equal(output_3d[0], values_3d[0])
    assert not bool(output_3d[1].any())
    assert torch.equal(output_3d[2], values_3d[1])
    assert output_3d.dtype == torch.float64
    assert output_3d.device == values_3d.device


@pytest.mark.parametrize(
    ("values", "indices", "batch_size", "error_type"),
    [
        (torch.tensor(1.0), torch.tensor([0]), 1, ValueError),
        (torch.ones((1, 2), dtype=torch.long), torch.tensor([0]), 1, TypeError),
        (torch.empty((0, 2)), torch.empty(0, dtype=torch.long), 1, ValueError),
        (torch.ones((2, 2)), torch.tensor([[0, 1]]), 2, ValueError),
        (torch.ones((2, 2)), torch.tensor([0, 1], dtype=torch.int32), 2, TypeError),
        (torch.ones((2, 2)), torch.tensor([0]), 2, ValueError),
        (torch.ones((2, 2)), torch.tensor([0, 0]), 2, ValueError),
        (torch.ones((2, 2)), torch.tensor([1, 0]), 2, ValueError),
        (torch.ones((1, 2)), torch.tensor([-1]), 2, ValueError),
        (torch.ones((1, 2)), torch.tensor([2]), 2, ValueError),
        (torch.ones((1, 2)), torch.tensor([0]), True, TypeError),
        (torch.ones((1, 2)), torch.tensor([0]), 0, ValueError),
    ],
)
def test_scatter_compact_rows_rejects_invalid_contracts(
    values: Tensor,
    indices: Tensor,
    batch_size: int,
    error_type: type[Exception],
) -> None:
    """Reject invalid values, indices, and full-batch sizes."""
    with pytest.raises(error_type):
        scatter_compact_rows(values, indices, batch_size=batch_size)


def test_scatter_preserves_storage_inputs_and_exact_autograd_mapping() -> None:
    """Backpropagate selected full rows to their corresponding compact rows."""
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    indices = torch.tensor([0, 2])
    indices_before = indices.clone()
    values_before = values.detach().clone()
    output = scatter_compact_rows(values, indices, batch_size=4)

    assert output.requires_grad
    assert output.data_ptr() != values.data_ptr()
    assert torch.equal(values.detach(), values_before)
    assert torch.equal(indices, indices_before)
    weights = torch.tensor([[2.0], [0.0], [3.0], [0.0]])
    (output * weights).sum().backward()
    assert torch.equal(values.grad, torch.tensor([[2.0, 2.0], [3.0, 3.0]]))


@pytest.mark.parametrize(
    ("speech", "physiology", "expected"),
    [
        ([True, True], [True, True], ("both_available",)),
        ([True, True], [False, False], ("speech_only",)),
        ([False, False], [True, True], ("physiology_only",)),
        ([False, False], [False, False], ("neither_available",)),
    ],
)
def test_availability_masks_homogeneous_cases(
    speech: list[bool],
    physiology: list[bool],
    expected: tuple[str],
) -> None:
    """Partition each homogeneous availability case exactly."""
    masks = ModalityAvailabilityMasks.from_availability(
        torch.tensor(speech),
        torch.tensor(physiology),
    )
    for name in (
        "both_available",
        "speech_only",
        "physiology_only",
        "neither_available",
    ):
        field = getattr(masks, name)
        assert bool(field.all()) == (name in expected)


def test_availability_masks_mixed_partition_and_frozen() -> None:
    """Represent all four combinations as disjoint and exhaustive masks."""
    speech = torch.tensor([True, True, False, False])
    physiology = torch.tensor([True, False, True, False])
    masks = ModalityAvailabilityMasks.from_availability(speech, physiology)

    assert torch.equal(masks.both_available, torch.tensor([True, False, False, False]))
    assert torch.equal(masks.speech_only, torch.tensor([False, True, False, False]))
    assert torch.equal(masks.physiology_only, torch.tensor([False, False, True, False]))
    assert torch.equal(
        masks.neither_available,
        torch.tensor([False, False, False, True]),
    )
    stacked = torch.stack(
        [
            masks.both_available,
            masks.speech_only,
            masks.physiology_only,
            masks.neither_available,
        ]
    )
    assert not bool((stacked.to(torch.int8).sum(dim=0) != 1).any())
    assert masks.speech_available.data_ptr() != speech.data_ptr()
    with pytest.raises(FrozenInstanceError):
        masks.speech_only = speech  # type: ignore[misc]


@pytest.mark.parametrize(
    ("speech", "physiology"),
    [
        (torch.tensor([1]), torch.tensor([False])),
        (torch.tensor([True, False]), torch.tensor([True])),
        (torch.tensor([], dtype=torch.bool), torch.tensor([], dtype=torch.bool)),
    ],
)
def test_availability_masks_reject_invalid_inputs(
    speech: Tensor,
    physiology: Tensor,
) -> None:
    """Reject non-bool, mismatched, and empty availability inputs."""
    with pytest.raises((TypeError, ValueError)):
        ModalityAvailabilityMasks.from_availability(speech, physiology)


def test_availability_dataclass_rejects_wrong_direct_formula() -> None:
    """Validate combination formulas independently of the factory."""
    masks = ModalityAvailabilityMasks.from_availability(
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    )
    with pytest.raises(ValueError, match="formulas"):
        replace(masks, both_available=torch.tensor([True, True]))


def test_scattered_prediction_accepts_optional_quadrant_and_is_frozen() -> None:
    """Accept both quadrant configurations and prevent field reassignment."""
    without = _scattered_prediction()
    with_logits = _scattered_prediction(quadrant_logits=True)
    assert without.quadrant_logits is None
    assert with_logits.quadrant_logits is not None
    with pytest.raises(FrozenInstanceError):
        without.reliability_score_type = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, embedding=torch.zeros((4, 0))),
        lambda value: replace(value, arousal_logits=torch.zeros((4, 3))),
        lambda value: replace(value, valence_logits=torch.zeros((3, 2))),
        lambda value: replace(
            value,
            arousal_probabilities=torch.ones((4, 2)),
        ),
        lambda value: replace(
            value,
            quadrant_probabilities=torch.zeros((4, 4)),
        ),
        lambda value: replace(value, reliability=torch.zeros((4, 2))),
        lambda value: replace(
            value,
            reliability=torch.tensor([[0.2], [0.0], [1.2], [0.0]]),
        ),
        lambda value: replace(
            value,
            sample_valid=torch.tensor([True, True, False, False]),
        ),
        lambda value: replace(
            value,
            batch_indices=torch.tensor([0, 0]),
        ),
        lambda value: replace(value, reliability_score_type=""),
    ],
)
def test_scattered_prediction_rejects_invalid_contracts(
    change: Callable[
        [ScatteredModalityPrediction],
        ScatteredModalityPrediction,
    ],
) -> None:
    """Reject invalid shapes, sentinels, probabilities, scores, and indices."""
    with pytest.raises((TypeError, ValueError)):
        change(_scattered_prediction())


def test_scattered_prediction_rejects_nonzero_unavailable_rows() -> None:
    """Require every unavailable floating row to be an exact-zero sentinel."""
    prediction = _scattered_prediction()
    logits = prediction.arousal_logits.clone()
    logits[1, 0] = 1
    with pytest.raises(ValueError, match="unavailable"):
        replace(prediction, arousal_logits=logits)


def test_scheduler_validates_constructor_and_registers_both_classifiers() -> None:
    """Register exactly the two injected classifier modules."""
    speech = TinySpeechClassifier()
    physiology = TinyPhysioClassifier()
    scheduler = MultimodalBatchScheduler(speech, physiology)
    assert scheduler.speech_classifier is speech
    assert scheduler.physiology_classifier is physiology
    assert dict(scheduler.named_children()) == {
        "speech_classifier": speech,
        "physiology_classifier": physiology,
    }
    keys = tuple(scheduler.state_dict())
    assert any(key.startswith("speech_classifier.") for key in keys)
    assert any(key.startswith("physiology_classifier.") for key in keys)
    with pytest.raises(TypeError, match="speech_classifier"):
        MultimodalBatchScheduler(nn.Linear(1, 1), physiology)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="physiology_classifier"):
        MultimodalBatchScheduler(speech, nn.Linear(1, 1))  # type: ignore[arg-type]


def test_scheduler_routes_exact_keyword_objects_once_and_scatters() -> None:
    """Call each classifier once with exact compact tensors and scatter fields."""
    scheduler = _scheduler(speech_quadrant=True, physiology_quadrant=True)
    batch = _batch()
    output = scheduler(batch)

    assert isinstance(output, ScheduledModalityOutputs)
    assert output.batch is batch
    speech_classifier = cast(TinySpeechClassifier, scheduler.speech_classifier)
    physio_classifier = cast(TinyPhysioClassifier, scheduler.physiology_classifier)
    assert len(speech_classifier.calls) == 1
    assert len(physio_classifier.calls) == 1
    assert batch.speech is not None
    assert batch.physiology is not None
    speech_call = speech_classifier.calls[0]
    assert speech_call["waveform"] is batch.speech.waveform
    assert speech_call["speech_attention_mask"] is batch.speech.attention_mask
    assert speech_call["sample_rate"] == batch.speech.sample_rate_hz
    assert speech_call["speech_activity_mask"] is batch.speech.activity_mask
    physio_call = physio_classifier.calls[0]
    assert physio_call["physio_input"] is batch.physiology.physio_input
    assert physio_call["physio_valid_mask"] is batch.physiology.physio_valid_mask
    assert physio_call["channel_names"] is batch.physiology.channel_names
    assert physio_call["physio_time_mask"] is batch.physiology.physio_time_mask
    assert (
        physio_call["physio_channel_mask"]
        is batch.physiology.physio_channel_mask
    )
    assert (
        physio_call["physio_quality_features"]
        is batch.physiology.physio_quality_features
    )
    assert "physio_timeline_mask" not in physio_call
    assert "labels" not in speech_call
    assert "labels" not in physio_call

    assert output.speech_compact_output is speech_classifier.last_output
    assert output.physiology_compact_output is physio_classifier.last_output
    assert output.speech is not None
    assert output.physiology is not None
    assert torch.equal(output.speech.batch_indices, torch.tensor([0, 1]))
    assert torch.equal(output.physiology.batch_indices, torch.tensor([0, 2]))
    for prediction, compact_embedding in (
        (output.speech, output.speech_compact_output.speech_embedding),
        (output.physiology, output.physiology_compact_output.physio_embedding),
    ):
        assert torch.equal(
            prediction.embedding.index_select(0, prediction.batch_indices),
            compact_embedding,
        )
        assert not bool(prediction.embedding[~prediction.sample_valid].any())
        assert not bool(prediction.arousal_logits[~prediction.sample_valid].any())
        assert not bool(
            prediction.arousal_probabilities[~prediction.sample_valid].any()
        )
        assert not bool(prediction.reliability[~prediction.sample_valid].any())
        assert prediction.quadrant_logits is not None


@pytest.mark.parametrize(
    ("patterns", "speech_calls", "physiology_calls"),
    [
        (((True, False), (True, False)), 1, 0),
        (((False, True), (False, True)), 0, 1),
        (((False, False), (False, False)), 0, 0),
    ],
)
def test_scheduler_skips_absent_compact_subbatches(
    patterns: tuple[tuple[bool, bool], ...],
    speech_calls: int,
    physiology_calls: int,
) -> None:
    """Skip unavailable classifiers without creating placeholder model rows."""
    scheduler = _scheduler()
    output = scheduler(_batch(patterns))
    speech = cast(TinySpeechClassifier, scheduler.speech_classifier)
    physiology = cast(TinyPhysioClassifier, scheduler.physiology_classifier)
    assert len(speech.calls) == speech_calls
    assert len(physiology.calls) == physiology_calls
    assert (output.speech is None) == (speech_calls == 0)
    assert (output.speech_compact_output is None) == (speech_calls == 0)
    assert (output.physiology is None) == (physiology_calls == 0)
    assert (output.physiology_compact_output is None) == (physiology_calls == 0)
    assert len(output.batch.records) == len(patterns)


def test_scheduler_four_availability_combinations_are_not_fused() -> None:
    """Keep dual, unimodal, and neither rows explicit in separate outputs."""
    batch = _batch()
    output = _scheduler()(batch)
    assert torch.equal(
        output.availability.both_available,
        torch.tensor([True, False, False, False]),
    )
    assert torch.equal(
        output.availability.speech_only,
        torch.tensor([False, True, False, False]),
    )
    assert torch.equal(
        output.availability.physiology_only,
        torch.tensor([False, False, True, False]),
    )
    assert torch.equal(
        output.availability.neither_available,
        torch.tensor([False, False, False, True]),
    )
    assert output.speech is not None
    assert output.physiology is not None
    assert len(output.batch.records) == 4
    assert output.batch.arousal_labels.shape == (4,)
    assert output.batch.valence_labels.shape == (4,)
    assert torch.equal(output.speech.batch_indices, torch.tensor([0, 1]))
    assert torch.equal(output.physiology.batch_indices, torch.tensor([0, 2]))
    assert torch.equal(
        output.speech.sample_valid,
        torch.tensor([True, True, False, False]),
    )
    assert torch.equal(
        output.physiology.sample_valid,
        torch.tensor([True, False, True, False]),
    )
    assert not bool(output.speech.embedding[3].any())
    assert not bool(output.physiology.embedding[3].any())
    assert output.speech.embedding.shape[0] == 4
    assert output.physiology.embedding.shape[0] == 4


def test_scheduler_rejects_wrong_classifier_output_types_with_context() -> None:
    """Raise contextual RuntimeError before accessing output attributes."""
    scheduler = _scheduler()
    speech = cast(TinySpeechClassifier, scheduler.speech_classifier)
    speech.override_output = object()
    with pytest.raises(
        RuntimeError,
        match="speech classifier.*LightweightSpeechClassifierOutput.*object",
    ):
        scheduler(_batch(((True, False),)))

    scheduler = _scheduler()
    physiology = cast(TinyPhysioClassifier, scheduler.physiology_classifier)
    physiology.override_output = object()
    with pytest.raises(
        RuntimeError,
        match="physiology classifier.*LightweightPhysioClassifierOutput.*object",
    ):
        scheduler(_batch(((False, True),)))


@pytest.mark.parametrize(
    ("modality", "mutate", "message"),
    [
        (
            "speech",
            lambda value: replace(value, speech_embedding=torch.ones((3, 2))),
            "embedding",
        ),
        (
            "speech",
            lambda value: replace(value, arousal_logits=torch.ones((2, 3))),
            "arousal_logits",
        ),
        (
            "speech",
            lambda value: replace(
                value,
                arousal_probabilities=torch.ones((2, 2)),
            ),
            "arousal_probabilities",
        ),
        (
            "speech",
            lambda value: replace(
                value,
                sample_valid=torch.tensor([True, False]),
            ),
            "sample_valid",
        ),
        (
            "speech",
            lambda value: replace(
                value,
                reliability=torch.ones((2, 2)),
            ),
            "reliability",
        ),
        (
            "physiology",
            lambda value: replace(value, physio_embedding=torch.ones((3, 2))),
            "embedding",
        ),
        (
            "physiology",
            lambda value: replace(value, valence_logits=torch.ones((2, 3))),
            "valence_logits",
        ),
        (
            "physiology",
            lambda value: replace(
                value,
                reliability=torch.tensor([[0.5], [1.5]]),
            ),
            "reliability",
        ),
    ],
)
def test_scheduler_rejects_malformed_compact_outputs(
    modality: str,
    mutate: Callable[[object], object],
    message: str,
) -> None:
    """Reject malformed classifier fields without silently repairing them."""
    scheduler = _scheduler()
    batch = _batch()
    scheduler.eval()
    first = scheduler(batch)
    if modality == "speech":
        classifier = cast(TinySpeechClassifier, scheduler.speech_classifier)
        assert first.speech_compact_output is not None
        classifier.override_output = mutate(first.speech_compact_output)
    else:
        classifier = cast(TinyPhysioClassifier, scheduler.physiology_classifier)
        assert first.physiology_compact_output is not None
        classifier.override_output = mutate(first.physiology_compact_output)
    with pytest.raises(RuntimeError, match=message):
        scheduler(batch)


def test_scheduler_rejects_nonfinite_compact_output() -> None:
    """Reject non-finite classifier values with classifier context."""
    scheduler = _scheduler()
    scheduler.eval()
    batch = _batch(((True, False),))
    first = scheduler(batch)
    classifier = cast(TinySpeechClassifier, scheduler.speech_classifier)
    assert first.speech_compact_output is not None
    logits = first.speech_compact_output.arousal_logits.clone()
    logits[0, 0] = float("nan")
    classifier.override_output = replace(
        first.speech_compact_output,
        arousal_logits=logits,
    )
    with pytest.raises(RuntimeError, match="speech classifier.*finite"):
        scheduler(batch)


def _joint_loss(logits_a: Tensor, logits_v: Tensor, labels: Tensor) -> Tensor:
    """Use the existing public CE for two binary tasks."""
    return class_weighted_cross_entropy(
        logits_a,
        labels,
    ) + class_weighted_cross_entropy(logits_v, labels)


@pytest.mark.parametrize("modality", ["speech", "physiology"])
def test_compact_and_scattered_losses_and_gradients_are_equivalent(
    modality: str,
) -> None:
    """Show caller-side ignored labels recover the exact compact objective."""
    scheduler = _scheduler()
    scheduler.eval()
    batch = _batch()
    classifier = (
        scheduler.speech_classifier
        if modality == "speech"
        else scheduler.physiology_classifier
    )

    first = scheduler(batch)
    prediction = first.speech if modality == "speech" else first.physiology
    compact = (
        first.speech_compact_output
        if modality == "speech"
        else first.physiology_compact_output
    )
    assert prediction is not None
    assert compact is not None
    compact_labels = batch.arousal_labels.index_select(
        0,
        prediction.batch_indices,
    )
    compact_loss = _joint_loss(
        compact.arousal_logits,
        compact.valence_logits,
        compact_labels,
    )
    scheduler.zero_grad(set_to_none=True)
    compact_loss.backward()
    compact_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in classifier.named_parameters()
        if parameter.grad is not None
    }

    scheduler.zero_grad(set_to_none=True)
    second = scheduler(batch)
    second_prediction = (
        second.speech if modality == "speech" else second.physiology
    )
    assert second_prediction is not None
    full_labels = batch.arousal_labels.clone()
    full_labels[~second_prediction.sample_valid] = batch.label_ignore_index
    scattered_loss = _joint_loss(
        second_prediction.arousal_logits,
        second_prediction.valence_logits,
        full_labels,
    )
    scattered_loss.backward()
    scattered_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in classifier.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(scattered_loss, compact_loss)
    assert compact_gradients.keys() == scattered_gradients.keys()
    for name in compact_gradients:
        torch.testing.assert_close(
            scattered_gradients[name],
            compact_gradients[name],
        )
    assert any(
        bool(torch.count_nonzero(gradient))
        for gradient in scattered_gradients.values()
    )


def test_modality_losses_keep_classifier_gradients_isolated_then_combined() -> None:
    """Train only the selected classifier path, then both paths together."""
    scheduler = _scheduler()
    scheduler.eval()
    batch = _batch()
    output = scheduler(batch)
    assert output.speech is not None
    assert output.physiology is not None
    speech_labels = batch.arousal_labels.clone()
    speech_labels[~output.speech.sample_valid] = batch.label_ignore_index
    speech_loss = _joint_loss(
        output.speech.arousal_logits,
        output.speech.valence_logits,
        speech_labels,
    )
    scheduler.zero_grad(set_to_none=True)
    speech_loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in scheduler.speech_classifier.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in scheduler.physiology_classifier.parameters()
    )

    scheduler.zero_grad(set_to_none=True)
    output = scheduler(batch)
    assert output.physiology is not None
    physiology_labels = batch.arousal_labels.clone()
    physiology_labels[~output.physiology.sample_valid] = batch.label_ignore_index
    physiology_loss = _joint_loss(
        output.physiology.arousal_logits,
        output.physiology.valence_logits,
        physiology_labels,
    )
    physiology_loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in scheduler.physiology_classifier.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in scheduler.speech_classifier.parameters()
    )

    scheduler.zero_grad(set_to_none=True)
    output = scheduler(batch)
    assert output.speech is not None
    assert output.physiology is not None
    speech_labels = batch.arousal_labels.clone()
    physiology_labels = batch.arousal_labels.clone()
    speech_labels[~output.speech.sample_valid] = batch.label_ignore_index
    physiology_labels[~output.physiology.sample_valid] = batch.label_ignore_index
    combined = _joint_loss(
        output.speech.arousal_logits,
        output.speech.valence_logits,
        speech_labels,
    ) + _joint_loss(
        output.physiology.arousal_logits,
        output.physiology.valence_logits,
        physiology_labels,
    )
    combined.backward()
    for classifier in (
        scheduler.speech_classifier,
        scheduler.physiology_classifier,
    ):
        gradients = [
            parameter.grad
            for parameter in classifier.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def test_scheduler_train_eval_propagation_and_eval_determinism() -> None:
    """Use normal Module mode propagation without changing routing metadata."""
    torch.manual_seed(4)
    scheduler = _scheduler()
    batch = _batch()
    scheduler.train()
    assert scheduler.training
    assert scheduler.speech_classifier.training
    assert scheduler.physiology_classifier.training
    assert cast(TinySpeechClassifier, scheduler.speech_classifier).dropout.training
    assert cast(TinyPhysioClassifier, scheduler.physiology_classifier).dropout.training
    train_outputs = [scheduler(batch) for _ in range(4)]
    train_speech = [result.speech for result in train_outputs]
    assert all(result is not None for result in train_speech)
    assert any(
        not torch.equal(
            cast(ScatteredModalityPrediction, train_speech[0]).embedding,
            cast(ScatteredModalityPrediction, result).embedding,
        )
        for result in train_speech[1:]
    )

    scheduler.eval()
    first = scheduler(batch)
    second = scheduler(batch)
    assert not scheduler.speech_classifier.training
    assert not scheduler.physiology_classifier.training
    assert first.speech is not None and second.speech is not None
    assert first.physiology is not None and second.physiology is not None
    torch.testing.assert_close(first.speech.embedding, second.speech.embedding)
    torch.testing.assert_close(
        first.physiology.embedding,
        second.physiology.embedding,
    )
    assert torch.equal(
        first.availability.speech_available,
        second.availability.speech_available,
    )
    assert torch.equal(first.speech.batch_indices, second.speech.batch_indices)


def test_scheduler_state_round_trip_and_nested_shape_incompatibility() -> None:
    """Save both classifiers and reject incompatible nested architectures."""
    torch.manual_seed(7)
    source = _scheduler()
    source.eval()
    batch = _batch()
    parameter_ids_before = tuple(id(parameter) for parameter in source.parameters())
    state_keys_before_forward = tuple(source.state_dict())
    expected = source(batch)
    assert tuple(id(parameter) for parameter in source.parameters()) == (
        parameter_ids_before
    )
    assert tuple(source.state_dict()) == state_keys_before_forward
    state = source.state_dict()
    keys_before = tuple(state)
    assert not any(
        token in key
        for key in state
        for token in ("batch", "indices", "output", "label")
    )

    restored = _scheduler()
    restored.load_state_dict(state, strict=True)
    restored.eval()
    actual = restored(batch)
    assert expected.speech is not None and actual.speech is not None
    assert expected.physiology is not None and actual.physiology is not None
    torch.testing.assert_close(expected.speech.embedding, actual.speech.embedding)
    torch.testing.assert_close(
        expected.physiology.embedding,
        actual.physiology.embedding,
    )
    assert tuple(source.state_dict()) == keys_before

    speech_incompatible = _scheduler(speech_dim=5)
    speech_fingerprint = (
        cast(
            TinySpeechClassifier,
            speech_incompatible.speech_classifier,
        ).input_projection.out_features
    )
    with pytest.raises(RuntimeError):
        speech_incompatible.load_state_dict(state, strict=True)
    assert (
        cast(
            TinySpeechClassifier,
            speech_incompatible.speech_classifier,
        ).input_projection.out_features
        == speech_fingerprint
    )

    physiology_incompatible = _scheduler(physiology_dim=5)
    physiology_fingerprint = cast(
        TinyPhysioClassifier,
        physiology_incompatible.physiology_classifier,
    ).get_extra_state()
    with pytest.raises(RuntimeError):
        physiology_incompatible.load_state_dict(state, strict=True)
    assert (
        cast(
            TinyPhysioClassifier,
            physiology_incompatible.physiology_classifier,
        ).get_extra_state()
        == physiology_fingerprint
    )


def test_scheduler_preserves_input_and_separates_compact_scattered_storage() -> None:
    """Keep batch tensors and compact outputs unchanged across scatter-back."""
    scheduler = _scheduler()
    scheduler.eval()
    batch = _batch()
    assert batch.speech is not None
    assert batch.physiology is not None
    waveform_before = batch.speech.waveform.clone()
    physio_before = batch.physiology.physio_input.clone()
    speech_indices_before = batch.speech.batch_indices.clone()
    availability_before = batch.speech_available.clone()

    output = scheduler(batch)

    assert output.batch is batch
    assert torch.equal(batch.speech.waveform, waveform_before)
    assert torch.equal(batch.physiology.physio_input, physio_before)
    assert torch.equal(batch.speech.batch_indices, speech_indices_before)
    assert torch.equal(batch.speech_available, availability_before)
    assert output.speech is not None
    assert output.physiology is not None
    assert output.speech_compact_output is not None
    assert output.physiology_compact_output is not None
    compact_speech = output.speech_compact_output.speech_embedding
    compact_physio = output.physiology_compact_output.physio_embedding
    assert output.speech.embedding.data_ptr() != compact_speech.data_ptr()
    assert output.physiology.embedding.data_ptr() != compact_physio.data_ptr()
    scattered_speech_before = output.speech.embedding.clone()
    scattered_physio_before = output.physiology.embedding.clone()
    output.speech.embedding[0, 0] = 99
    output.physiology.embedding[0, 0] = 99
    assert compact_speech[0, 0] != 99
    assert compact_physio[0, 0] != 99
    compact_speech[0, 1] = -99
    compact_physio[0, 1] = -99
    assert torch.equal(
        output.speech.embedding[:, 1:],
        scattered_speech_before[:, 1:],
    )
    assert torch.equal(
        output.physiology.embedding[:, 1:],
        scattered_physio_before[:, 1:],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scheduler_float32_and_float64_core_paths(dtype: torch.dtype) -> None:
    """Preserve classifier precision through compact validation and scatter."""
    scheduler = _scheduler().to(dtype=dtype)
    scheduler.eval()
    output = scheduler(_batch(dtype=dtype))
    assert output.speech is not None
    assert output.physiology is not None
    assert output.speech.embedding.dtype == dtype
    assert output.physiology.embedding.dtype == dtype
    assert output.speech.arousal_logits.dtype == dtype
    assert output.physiology.reliability.dtype == dtype


def test_scheduled_output_dataclass_rejects_inconsistent_pairs_and_is_frozen() -> None:
    """Validate direct construction independently from scheduler forward."""
    scheduler = _scheduler()
    scheduler.eval()
    output = scheduler(_batch())
    with pytest.raises(FrozenInstanceError):
        output.speech = None  # type: ignore[misc]
    with pytest.raises(ValueError, match="availability"):
        replace(
            output,
            availability=ModalityAvailabilityMasks.from_availability(
                torch.tensor([False, False, False, False]),
                output.batch.physiology_available,
            ),
        )
    with pytest.raises(TypeError, match="speech"):
        replace(output, speech_compact_output=None)


def test_routing_scope_contains_no_fusion_training_or_data_pipeline() -> None:
    """Keep stage 12A limited to classifier routing and differentiable scatter."""
    source = (
        Path(routing_module.__file__).read_text(encoding="utf-8")
        if routing_module.__file__ is not None
        else ""
    )
    forbidden = (
        "DataLoader",
        "Sampler",
        "AlignedMultimodalDataset",
        "collate_aligned_multimodal_samples",
        "optimizer",
        "backward(",
        "cross_entropy(",
        "fused_logits",
        "fusion_layer",
        "all_reduce",
        "DistributedDataParallel",
        "clean_speech",
        "reconstruct_waveform",
        "denoise_output",
        "noise_subtraction",
        "speech_enhancement",
    )
    assert not any(term in source for term in forbidden)
