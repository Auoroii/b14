"""Tests for participant-independent multimodal evaluation."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch import Tensor

import emotion_model.evaluation.runner as runner_module
from emotion_model.common import derive_quadrant_labels
from emotion_model.data import (
    AlignedMultimodalBatch,
    DatasetPartition,
    ParticipantSplit,
)
from emotion_model.evaluation import (
    BinaryDecisionThresholds,
    EvaluationPredictions,
    ModalityPattern,
    ParticipantEvaluationScope,
    evaluate_participant_independent,
)
from emotion_model.multimodal import (
    MultimodalEmotionClassifier,
    MultimodalEmotionClassifierOutput,
)
from emotion_model.training import (
    EpochPhase,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
    run_multimodal_validation_epoch,
)
from tests.test_multimodal_classifier import _model
from tests.test_multimodal_routing import _batch


def _split(
    *,
    test_ids: tuple[str, ...] = ("p1", "p2"),
    validation_ids: tuple[str, ...] = ("validation-person",),
) -> ParticipantSplit:
    return ParticipantSplit(
        train_participant_ids=("training-person",),
        validation_participant_ids=validation_ids,
        test_participant_ids=test_ids,
    )


def _scope(
    *,
    test_ids: tuple[str, ...] = ("p1", "p2"),
    require_all: bool = True,
) -> ParticipantEvaluationScope:
    return ParticipantEvaluationScope(
        _split(test_ids=test_ids),
        DatasetPartition.TEST,
        require_all,
    )


def _with_record_ids(
    batch: AlignedMultimodalBatch,
    participant_ids: tuple[str, ...],
    *,
    sample_prefix: str = "evaluated",
) -> AlignedMultimodalBatch:
    assert len(participant_ids) == len(batch.records)
    records = tuple(
        replace(
            record,
            sample_id=f"{sample_prefix}-{index}",
            participant_id=participant_id,
            session_id=f"session-{index % 2}",
        )
        for index, (record, participant_id) in enumerate(
            zip(batch.records, participant_ids, strict=True)
        )
    )
    return replace(batch, records=records)


def _with_labels(
    batch: AlignedMultimodalBatch,
    *,
    arousal: Tensor,
    valence: Tensor,
) -> AlignedMultimodalBatch:
    quadrant = derive_quadrant_labels(
        arousal,
        valence,
        ignore_index=batch.label_ignore_index,
    )
    return replace(
        batch,
        arousal_labels=arousal,
        valence_labels=valence,
        quadrant_labels=quadrant,
    )


def _objective() -> MultimodalTrainingObjective:
    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(weights=MultimodalLossWeights())
    )


def _constant_prediction_model(
    *,
    independent_quadrant: bool = False,
) -> MultimodalEmotionClassifier:
    model = _model(
        dropout=0.0,
        independent_quadrant=independent_quadrant,
    )
    with torch.no_grad():
        model.arousal_head.weight.zero_()
        model.arousal_head.bias.copy_(torch.tensor([0.0, 2.0]))
        model.valence_head.weight.zero_()
        model.valence_head.bias.copy_(torch.tensor([2.0, 0.0]))
        if model.quadrant_head is not None:
            model.quadrant_head.weight.zero_()
            model.quadrant_head.bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 9.0])
            )
    return model


class _CountingModel(MultimodalEmotionClassifier):
    """Tiny final classifier that records eval, forward, and grad-mode calls."""

    def __init__(self) -> None:
        base = _constant_prediction_model()
        super().__init__(
            base.batch_scheduler,
            base.multimodal_fusion,
            base.fusion_dim,
            classifier_hidden_dim=base.classifier_hidden_dim,
            dropout=0.0,
        )
        with torch.no_grad():
            self.arousal_head.weight.zero_()
            self.arousal_head.bias.copy_(torch.tensor([0.0, 2.0]))
            self.valence_head.weight.zero_()
            self.valence_head.bias.copy_(torch.tensor([2.0, 0.0]))
        self.eval_calls = 0
        self.forward_calls = 0
        self.grad_enabled: list[bool] = []

    def eval(self) -> _CountingModel:
        """Record the single expected mode switch."""
        self.eval_calls += 1
        return super().eval()

    def forward(
        self,
        batch: AlignedMultimodalBatch,
    ) -> MultimodalEmotionClassifierOutput:
        """Record one call and delegate the full public batch contract."""
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return super().forward(batch)


class _CountingObjective(MultimodalTrainingObjective):
    """Objective double retaining production loss behavior and a call count."""

    def __init__(self) -> None:
        super().__init__(
            MultimodalObjectiveConfig(weights=MultimodalLossWeights())
        )
        self.forward_calls = 0

    def forward(  # type: ignore[override]
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Count and delegate one objective invocation."""
        self.forward_calls += 1
        return super().forward(*args, **kwargs)  # type: ignore[arg-type]


class _RecordMutatingModel(_CountingModel):
    """Deliberately violate the immutable record contract after inference."""

    def forward(
        self,
        batch: AlignedMultimodalBatch,
    ) -> MultimodalEmotionClassifierOutput:
        """Mutate one record field so the runner must fail explicitly."""
        output = super().forward(batch)
        object.__setattr__(
            batch.records[0],
            "session_id",
            "illegally-mutated-session",
        )
        return output


def _prediction_record() -> EvaluationPredictions:
    return EvaluationPredictions(
        sample_ids=("s0", "s1"),
        participant_ids=("p1", "p1"),
        speech_available=torch.tensor([True, False]),
        physiology_available=torch.tensor([False, False]),
        sample_valid=torch.tensor([True, False]),
        arousal_targets=torch.tensor([1, -100]),
        arousal_predictions=torch.tensor([1, -100]),
        valence_targets=torch.tensor([0, -100]),
        valence_predictions=torch.tensor([0, -100]),
        quadrant_targets=torch.tensor([1, -100]),
        quadrant_predictions=torch.tensor([1, -100]),
        ignore_index=-100,
    )


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (DatasetPartition.TRAIN, ("training-person",)),
        (DatasetPartition.VALIDATION, ("validation-person",)),
        (DatasetPartition.TEST, ("p1", "p2")),
    ],
)
def test_scope_preserves_partition_order(
    partition: DatasetPartition,
    expected: tuple[str, ...],
) -> None:
    """Select exact split tuples without sorting, normalizing, or mutation."""
    scope = ParticipantEvaluationScope(_split(), partition)
    assert scope.expected_participant_ids == expected
    with pytest.raises(FrozenInstanceError):
        scope.partition = DatasetPartition.TRAIN  # type: ignore[misc]


def test_scope_rejects_empty_partition_and_non_bool_requirement() -> None:
    """Require a populated selected partition and a strict bool flag."""
    split = _split(validation_ids=())
    with pytest.raises(ValueError, match="contain participants"):
        ParticipantEvaluationScope(split, DatasetPartition.VALIDATION)
    with pytest.raises(TypeError, match="bool"):
        ParticipantEvaluationScope(
            _split(),
            DatasetPartition.TEST,
            1,  # type: ignore[arg-type]
        )


def test_prediction_contract_is_frozen_cpu_and_storage_independent() -> None:
    """Clone public vectors and reject field reassignment."""
    valid = torch.tensor([True, False])
    prediction = replace(_prediction_record(), sample_valid=valid)
    valid[0] = False
    assert torch.equal(prediction.sample_valid, torch.tensor([True, False]))
    assert prediction.sample_valid.data_ptr() != valid.data_ptr()
    with pytest.raises(FrozenInstanceError):
        prediction.ignore_index = -1  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        {"sample_ids": ("s0", "s0")},
        {"participant_ids": ("p1", "")},
        {"speech_available": torch.tensor([1, 0])},
        {"arousal_targets": torch.tensor([1], dtype=torch.long)},
        {"arousal_targets": torch.tensor([2, -100])},
        {"arousal_predictions": torch.tensor([2, -100])},
        {"arousal_predictions": torch.tensor([1, 0])},
        {"ignore_index": 0},
    ],
)
def test_prediction_contract_rejects_malformed_fields(
    change: dict[str, object],
) -> None:
    """Reject duplicate IDs, bad tensors, ranges, and invalid-row predictions."""
    with pytest.raises((TypeError, ValueError)):
        replace(_prediction_record(), **change)


def test_runner_calls_eval_and_each_step_once_without_grad_or_len() -> None:
    """Consume a generator once with one model/objective call per batch."""
    first = _with_record_ids(
        _batch(((True, True),)),
        ("p1",),
        sample_prefix="first",
    )
    second = _with_record_ids(
        _batch(((False, True),)),
        ("p2",),
        sample_prefix="second",
    )
    consumed = 0

    def batches() -> Iterator[AlignedMultimodalBatch]:
        nonlocal consumed
        consumed += 1
        yield first
        yield second

    model = _CountingModel()
    objective = _CountingObjective()
    parameters_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
    }
    state_keys_before = tuple(model.state_dict())
    objective_config_before = objective.config
    split = _split()
    result = evaluate_participant_independent(
        model,
        objective,
        batches(),
        scope=ParticipantEvaluationScope(split, DatasetPartition.TEST),
    )
    assert consumed == 1
    assert model.eval_calls == 1
    assert model.forward_calls == 2
    assert objective.forward_calls == 2
    assert model.grad_enabled == [False, False]
    assert not model.training
    assert result.batch_count == 2
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in parameters_before.items()
    )
    assert tuple(model.state_dict()) == state_keys_before
    assert objective.config is objective_config_before
    assert split == _split()


def test_runner_extracts_final_argmax_preserves_targets_and_all_strata() -> None:
    """Use final derived probabilities, ignore invalid rows, and retain labels."""
    batch = _with_record_ids(
        _batch(
            (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            )
        ),
        ("p1", "p1", "p2", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([1, 0, -100, 1]),
        valence=torch.tensor([0, 1, 0, 1]),
    )
    model = _constant_prediction_model(independent_quadrant=True)
    result = evaluate_participant_independent(
        model,
        _objective(),
        (batch,),
        scope=_scope(),
    )
    predictions = result.predictions
    assert result.binary_decision_thresholds == BinaryDecisionThresholds()
    assert predictions.sample_ids == tuple(
        record.sample_id for record in batch.records
    )
    assert torch.equal(
        predictions.arousal_predictions,
        torch.tensor([1, 1, 1, -100]),
    )
    assert torch.equal(
        predictions.valence_predictions,
        torch.tensor([0, 0, 0, -100]),
    )
    # Independent quadrant head predicts 3, while derived binary heads predict 1.
    assert torch.equal(
        predictions.quadrant_predictions,
        torch.tensor([1, 1, 1, -100]),
    )
    assert torch.equal(predictions.arousal_targets, batch.arousal_labels)
    assert torch.equal(predictions.valence_targets, batch.valence_labels)
    assert result.overall_metrics.arousal.evaluated_count == 2
    assert result.overall_metrics.valence.evaluated_count == 3
    assert tuple(item.pattern for item in result.modality_stratum_metrics) == tuple(
        ModalityPattern
    )
    assert dict(result.modality_pattern_counts) == {
        ModalityPattern.BOTH: 1,
        ModalityPattern.SPEECH_ONLY: 1,
        ModalityPattern.PHYSIOLOGY_ONLY: 1,
        ModalityPattern.NEITHER: 1,
    }
    neither = result.modality_stratum_metrics[-1]
    assert neither.record_count == 1
    assert neither.task_metrics.arousal.evaluated_count == 0


def test_runner_calibrates_only_on_validation_participants() -> None:
    """Fit thresholds on validation and reject calibration on a test scope."""

    batch = _with_record_ids(
        _batch(((True, False),) * 4),
        ("p1", "p1", "p2", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([0, 1, 0, 1]),
        valence=torch.tensor([0, 1, 0, 1]),
    )
    split = ParticipantSplit(
        train_participant_ids=("training-person",),
        validation_participant_ids=("p1", "p2"),
        test_participant_ids=("test-person",),
    )
    calibrated = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=ParticipantEvaluationScope(
            split,
            DatasetPartition.VALIDATION,
        ),
        calibrate_thresholds=True,
    )
    assert "validation_pooled_macro_f1" in (
        calibrated.binary_decision_thresholds.policy
    )
    with pytest.raises(ValueError, match="only on validation"):
        evaluate_participant_independent(
            _constant_prediction_model(),
            _objective(),
            (batch,),
            scope=ParticipantEvaluationScope(
                ParticipantSplit(
                    train_participant_ids=("training-person",),
                    validation_participant_ids=("validation-person",),
                    test_participant_ids=("p1", "p2"),
                ),
                DatasetPartition.TEST,
            ),
            calibrate_thresholds=True,
        )


def test_runner_applies_preselected_thresholds_without_refitting() -> None:
    """Use checkpoint thresholds directly on a held-out test partition."""

    batch = _with_record_ids(
        _batch(((True, False), (True, False))),
        ("p1", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([0, 1]),
        valence=torch.tensor([0, 1]),
    )
    thresholds = BinaryDecisionThresholds(
        arousal_high=0.95,
        valence_high=0.05,
        policy="checkpoint-validation-thresholds",
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
        binary_thresholds=thresholds,
    )

    assert result.binary_decision_thresholds is thresholds
    assert torch.equal(
        result.predictions.arousal_predictions,
        torch.tensor([0, 0]),
    )
    assert torch.equal(
        result.predictions.valence_predictions,
        torch.tensor([1, 1]),
    )
    assert torch.equal(
        result.predictions.quadrant_predictions,
        torch.tensor([2, 2]),
    )


def test_participant_macro_is_equal_person_not_window_weighted() -> None:
    """Distinguish equal-person accuracy from the pooled window accuracy."""
    batch = _with_record_ids(
        _batch(((True, False),) * 4),
        ("p1", "p2", "p2", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([1, 0, 0, 0]),
        valence=torch.tensor([0, 0, 0, 0]),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert result.overall_metrics.arousal.accuracy == pytest.approx(0.25)
    assert (
        result.participant_macro_metrics.arousal.mean_accuracy
        == pytest.approx(0.5)
    )
    assert result.participant_macro_metrics.arousal.participant_count == 2
    assert result.participant_macro_metrics.arousal.evaluated_window_count == 4
    assert [item.record_count for item in result.participant_metrics] == [1, 3]


def test_task_specific_ignored_participant_is_retained_but_macro_excluded() -> None:
    """Keep record diagnostics while filtering participant macro per task."""
    batch = _with_record_ids(
        _batch(((True, False), (True, False))),
        ("p1", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([-100, 1]),
        valence=torch.tensor([0, 0]),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert result.participant_ids == ("p1", "p2")
    assert result.participant_metrics[0].task_metrics.arousal.evaluated_count == 0
    assert result.participant_macro_metrics.arousal.participant_count == 1
    assert result.participant_macro_metrics.valence.participant_count == 2


def test_one_task_all_ignored_keeps_other_tasks_and_all_group_outputs() -> None:
    """Zero only arousal while valence, participants, and strata remain live."""
    batch = _with_record_ids(
        _batch(((True, True), (True, False), (False, True))),
        ("p1", "p2", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([-100, -100, -100]),
        valence=torch.tensor([0, 1, 0]),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    arousal = result.overall_metrics.arousal
    assert arousal.evaluated_count == 0
    assert not bool(arousal.confusion_matrix.any())
    assert arousal.accuracy == arousal.macro_f1 == 0.0
    assert result.overall_metrics.valence.evaluated_count == 3
    assert len(result.participant_metrics) == 2
    assert len(result.modality_stratum_metrics) == 4
    assert all(
        participant.task_metrics.arousal.evaluated_count == 0
        for participant in result.participant_metrics
    )
    assert result.participant_macro_metrics.arousal.participant_count == 0


def test_all_neither_and_all_ignored_are_native_zero_results() -> None:
    """Retain records/participants while all final predictions are ignored."""
    batch = _with_record_ids(
        _batch(((False, False), (False, False))),
        ("p1", "p2"),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert result.record_count == result.participant_count == 2
    assert bool(
        (result.predictions.arousal_predictions == -100).all()
        and (result.predictions.valence_predictions == -100).all()
        and (result.predictions.quadrant_predictions == -100).all()
    )
    assert result.overall_metrics.arousal.evaluated_count == 0
    assert result.overall_metrics.valence.evaluated_count == 0
    assert result.overall_metrics.quadrant.evaluated_count == 0
    assert result.loss_epoch.empty_supervision_batch_count == 1
    assert result.modality_pattern_counts[ModalityPattern.NEITHER] == 2
    assert len(result.participant_metrics) == 2
    assert all(
        participant.task_metrics.arousal.evaluated_count == 0
        for participant in result.participant_metrics
    )
    for task_name in ("arousal", "valence", "quadrant"):
        macro = getattr(result.participant_macro_metrics, task_name)
        assert macro.participant_count == 0
        assert macro.evaluated_window_count == 0
        assert macro.mean_accuracy == macro.mean_macro_f1 == 0.0


def test_loss_epoch_matches_existing_runner_without_second_forward() -> None:
    """Reuse stage-13B arithmetic exactly on independent model copies."""
    batch = _with_record_ids(
        _batch(((True, True), (False, False))),
        ("p1", "p2"),
    )
    model = _constant_prediction_model()
    reference_model = copy.deepcopy(model)
    objective = _objective()
    reference_objective = copy.deepcopy(objective)
    result = evaluate_participant_independent(
        model,
        objective,
        (batch,),
        scope=_scope(),
    )
    expected = run_multimodal_validation_epoch(
        reference_model,
        reference_objective,
        (batch,),
    )
    assert result.loss_epoch == expected
    assert result.loss_epoch.phase is EpochPhase.VALIDATION


@pytest.mark.parametrize(
    ("participant_ids", "require_all", "match"),
    [
        (("validation-person",), False, "outside"),
        (("training-person",), False, "outside"),
        (("p1",), True, "missing"),
    ],
)
def test_partition_audit_rejects_out_of_scope_or_missing_participants(
    participant_ids: tuple[str, ...],
    require_all: bool,
    match: str,
) -> None:
    """Audit exact IDs immediately and complete required coverage at end."""
    batch = _with_record_ids(
        _batch(tuple((True, False) for _ in participant_ids)),
        participant_ids,
    )
    with pytest.raises(ValueError, match=match):
        evaluate_participant_independent(
            _constant_prediction_model(),
            _objective(),
            (batch,),
            scope=_scope(require_all=require_all),
        )


def test_require_all_false_allows_expected_subset_and_keeps_split_order() -> None:
    """Allow a subset while ordering observed results by the split tuple."""
    batch = _with_record_ids(
        _batch(((True, False),)),
        ("p2",),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(require_all=False),
    )
    assert result.participant_ids == ("p2",)
    assert result.participant_count == 1


def test_duplicate_sample_across_batches_fails_and_empty_iterable_fails() -> None:
    """Reject repeated evaluation records and a non-evaluation empty pass."""
    first = _with_record_ids(_batch(((True, False),)), ("p1",))
    duplicate = _with_record_ids(_batch(((True, False),)), ("p2",))
    with pytest.raises(ValueError, match="more than once"):
        evaluate_participant_independent(
            _constant_prediction_model(),
            _objective(),
            (first, duplicate),
            scope=_scope(),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_participant_independent(
            _constant_prediction_model(),
            _objective(),
            iter(()),
            scope=_scope(),
        )


def test_runner_detects_record_field_mutation_by_a_bad_model() -> None:
    """Fail in a controlled way when inference mutates frozen record content."""
    batch = _with_record_ids(
        _batch(((True, False), (False, True))),
        ("p1", "p2"),
    )
    with pytest.raises(RuntimeError, match="record fields"):
        evaluate_participant_independent(
            _RecordMutatingModel(),
            _objective(),
            (batch,),
            scope=_scope(),
        )


def test_runner_uses_public_validation_step_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the runner composes the public no-gradient step exactly once."""
    batch = _with_record_ids(
        _batch(((True, False), (False, True))),
        ("p1", "p2"),
    )
    original = runner_module.validate_multimodal_batch
    calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module, "validate_multimodal_batch", spy)
    evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert calls == 1


def test_returned_prediction_and_metric_mutation_do_not_write_back() -> None:
    """Keep batch labels and independent aggregate matrices isolated."""
    batch = _with_record_ids(
        _batch(((True, False), (False, True))),
        ("p1", "p2"),
    )
    labels_before = batch.arousal_labels.clone()
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    result.predictions.arousal_targets[0] = 99
    result.overall_metrics.arousal.confusion_matrix.zero_()
    assert torch.equal(batch.arousal_labels, labels_before)
    assert result.participant_metrics[0].task_metrics.arousal.evaluated_count == 1
    assert bool(
        result.participant_metrics[0]
        .task_metrics.arousal.confusion_matrix.any()
    )
    assert bool(
        result.modality_stratum_metrics[1]
        .task_metrics.arousal.confusion_matrix.any()
    )
    with pytest.raises(TypeError):
        result.modality_pattern_counts[ModalityPattern.BOTH] = 99  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.record_count = 99  # type: ignore[misc]


def test_participant_metrics_have_independent_manual_confusions() -> None:
    """Compute each person's confusion directly from their own ordered windows."""
    batch = _with_record_ids(
        _batch(((True, False),) * 4),
        ("p1", "p1", "p2", "p2"),
    )
    batch = _with_labels(
        batch,
        arousal=torch.tensor([1, 0, 0, 0]),
        valence=torch.tensor([0, 0, 0, 0]),
    )
    result = evaluate_participant_independent(
        _constant_prediction_model(),
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert torch.equal(
        result.participant_metrics[0].task_metrics.arousal.confusion_matrix,
        torch.tensor([[0, 1], [0, 1]]),
    )
    assert torch.equal(
        result.participant_metrics[1].task_metrics.arousal.confusion_matrix,
        torch.tensor([[0, 2], [0, 0]]),
    )
