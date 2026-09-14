"""Tests for multimodal epoch runners and immutable progress state."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest
import torch

import emotion_model.training.epochs as epochs_module
from emotion_model.data import AlignedMultimodalBatch
from emotion_model.training import (
    EpochPhase,
    MultimodalEpochLossAverages,
    MultimodalEpochOutput,
    MultimodalLossOutput,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
    MultimodalTrainingState,
    MultimodalTrainStepOutput,
    MultimodalValidationStepOutput,
    advance_training_state,
    run_multimodal_training_epoch,
    run_multimodal_validation_epoch,
    validate_multimodal_batch,
)
from tests.test_multimodal_classifier import _batch, _model


def _objective(
    weights: MultimodalLossWeights | None = None,
) -> MultimodalTrainingObjective:
    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(weights=weights or MultimodalLossWeights())
    )


def _loss_averages(value: float = 0.0) -> MultimodalEpochLossAverages:
    return MultimodalEpochLossAverages(
        total_loss=value,
        fused_loss=value,
        fused_arousal_loss=value,
        fused_valence_loss=value,
        fused_quadrant_loss=value,
        speech_auxiliary_loss=value,
        speech_arousal_loss=value,
        speech_valence_loss=value,
        speech_quadrant_loss=value,
        physiology_auxiliary_loss=value,
        physiology_arousal_loss=value,
        physiology_valence_loss=value,
        physiology_quadrant_loss=value,
    )


def _epoch(
    phase: EpochPhase,
    *,
    supervised: int = 1,
    empty: int = 0,
    steps: int | None = None,
    active: int = 2,
    loss: float = 1.0,
    gradient_norm: float | None = None,
) -> MultimodalEpochOutput:
    if steps is None:
        steps = supervised if phase is EpochPhase.TRAIN else 0
    if gradient_norm is None:
        gradient_norm = 0.5 if steps else 0.0
    return MultimodalEpochOutput(
        phase=phase,
        batch_count=supervised + empty,
        supervised_batch_count=supervised,
        empty_supervision_batch_count=empty,
        optimizer_step_count=steps,
        active_target_count=active,
        loss_averages=_loss_averages(loss),
        max_gradient_norm=gradient_norm,
    )


def _ignored_batch() -> AlignedMultimodalBatch:
    batch = _batch()
    ignored = torch.full_like(batch.arousal_labels, batch.label_ignore_index)
    return replace(
        batch,
        arousal_labels=ignored,
        valence_labels=ignored.clone(),
        quadrant_labels=ignored.clone(),
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -0.1],
)
def test_loss_averages_reject_nonfinite_or_negative(value: float) -> None:
    """Require finite non-negative detached Python floats."""
    arguments = {
        field.name: 0.0
        for field in fields(MultimodalEpochLossAverages)
    }
    arguments["total_loss"] = value
    with pytest.raises(ValueError):
        MultimodalEpochLossAverages(**arguments)


@pytest.mark.parametrize("value", [True, 1, torch.tensor(1.0)])
def test_loss_averages_reject_non_python_float(value: object) -> None:
    """Do not retain bool, integer, or tensor loss values."""
    arguments = {
        field.name: 0.0
        for field in fields(MultimodalEpochLossAverages)
    }
    arguments["fused_loss"] = value
    with pytest.raises(TypeError):
        MultimodalEpochLossAverages(**arguments)


def test_epoch_dataclasses_are_frozen_and_store_scalars_only() -> None:
    """Expose no tensors or graph-carrying step outputs."""
    averages = _loss_averages(0.25)
    output = _epoch(EpochPhase.TRAIN, loss=0.25)
    with pytest.raises(FrozenInstanceError):
        averages.total_loss = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        output.batch_count = 2  # type: ignore[misc]
    assert all(
        isinstance(getattr(averages, field.name), float)
        for field in fields(averages)
    )
    assert not any(
        isinstance(getattr(output, field.name), torch.Tensor)
        for field in fields(output)
    )


@pytest.mark.parametrize(
    ("updates", "error_type"),
    [
        ({"batch_count": 0}, ValueError),
        ({"batch_count": True}, TypeError),
        ({"supervised_batch_count": -1}, ValueError),
        ({"empty_supervision_batch_count": 1}, ValueError),
        ({"optimizer_step_count": 0}, ValueError),
        ({"active_target_count": False}, TypeError),
        ({"active_target_count": 0}, ValueError),
        ({"max_gradient_norm": float("nan")}, ValueError),
        ({"max_gradient_norm": -1.0}, ValueError),
        ({"phase": "train"}, TypeError),
        ({"loss_averages": object()}, TypeError),
    ],
)
def test_epoch_output_rejects_invalid_contract(
    updates: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Reject inconsistent phase, count, loss, and norm state."""
    arguments: dict[str, object] = {
        "phase": EpochPhase.TRAIN,
        "batch_count": 1,
        "supervised_batch_count": 1,
        "empty_supervision_batch_count": 0,
        "optimizer_step_count": 1,
        "active_target_count": 2,
        "loss_averages": _loss_averages(),
        "max_gradient_norm": 0.5,
    }
    arguments.update(updates)
    with pytest.raises(error_type):
        MultimodalEpochOutput(**arguments)  # type: ignore[arg-type]


def test_validation_epoch_output_requires_zero_step_and_norm() -> None:
    """Keep validation free of optimizer and gradient diagnostics."""
    with pytest.raises(ValueError, match="optimizer_step_count"):
        _epoch(EpochPhase.VALIDATION, steps=1, gradient_norm=0.0)
    with pytest.raises(ValueError, match="max_gradient_norm"):
        _epoch(EpochPhase.VALIDATION, steps=0, gradient_norm=0.1)


def test_training_epoch_runs_real_batches_and_counts_empty_supervision() -> None:
    """Run model/objective per batch and exclude empty supervision from means."""
    torch.manual_seed(31)
    model = _model()
    objective = _objective()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    result = run_multimodal_training_epoch(
        model,
        objective,
        [_batch(), _ignored_batch()],
        optimizer,
        max_gradient_norm=1.0,
    )
    assert result.phase is EpochPhase.TRAIN
    assert result.batch_count == 2
    assert result.supervised_batch_count == 1
    assert result.empty_supervision_batch_count == 1
    assert result.optimizer_step_count == 1
    assert result.active_target_count == 6
    assert result.loss_averages.total_loss > 0.0
    assert result.max_gradient_norm > 0.0
    assert model.training


def test_validation_epoch_average_matches_independent_batch_steps() -> None:
    """Use an independent per-batch reference for arithmetic mean semantics."""
    model = _model()
    objective = _objective()
    batches = [_batch(), _batch(((True, False),))]
    model.eval()
    references = [
        validate_multimodal_batch(model, objective, batch).loss_output
        for batch in batches
    ]
    expected_total = sum(
        float(output.total_loss.item()) for output in references
    ) / 2.0
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    result = run_multimodal_validation_epoch(model, objective, batches)
    assert result.phase is EpochPhase.VALIDATION
    assert result.batch_count == 2
    assert result.supervised_batch_count == 2
    assert result.optimizer_step_count == 0
    assert result.max_gradient_norm == 0.0
    assert result.loss_averages.total_loss == pytest.approx(expected_total)
    assert not model.training
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[name])


def test_epoch_uses_unweighted_asymmetric_detached_batch_arithmetic_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independently distinguish batch arithmetic from target-weighted means."""
    model = _model()
    objective = _objective()
    batches = [_batch(), _batch(), _ignored_batch()]
    model.eval()
    reference = validate_multimodal_batch(
        model,
        objective,
        batches[0],
    )
    loss_names = tuple(
        field.name
        for field in fields(MultimodalLossOutput)
        if field.name != "active_target_count"
    )

    def loss_output(multiplier: float, active_count: int) -> MultimodalLossOutput:
        updates = {
            name: reference.loss_output.total_loss.new_tensor(
                multiplier * (index + 1)
            )
            for index, name in enumerate(loss_names)
        }
        return replace(
            reference.loss_output,
            **updates,
            active_target_count=active_count,
        )

    outputs = iter(
        (
            MultimodalValidationStepOutput(
                reference.model_output,
                loss_output(1.0, 1),
            ),
            MultimodalValidationStepOutput(
                reference.model_output,
                loss_output(3.0, 9),
            ),
            MultimodalValidationStepOutput(
                reference.model_output,
                loss_output(0.0, 0),
            ),
        )
    )

    def fake_step(*_args: object, **_kwargs: object) -> object:
        return next(outputs)

    monkeypatch.setattr(
        epochs_module,
        "validate_multimodal_batch",
        fake_step,
    )
    result = run_multimodal_validation_epoch(model, objective, batches)
    assert result.supervised_batch_count == 2
    assert result.empty_supervision_batch_count == 1
    assert result.active_target_count == 10
    for index, name in enumerate(loss_names):
        expected_batch_mean = 2.0 * (index + 1)
        target_weighted_mean = 2.8 * (index + 1)
        actual = getattr(result.loss_averages, name)
        assert actual == pytest.approx(expected_batch_mean)
        assert actual != pytest.approx(target_weighted_mean)
        assert isinstance(actual, float)


def test_all_empty_epochs_return_zero_loss_averages() -> None:
    """Represent an epoch with no supervised batch by finite scalar zeros."""
    training_model = _model()
    training = run_multimodal_training_epoch(
        training_model,
        _objective(),
        [_ignored_batch(), _batch(((False, False),))],
        torch.optim.AdamW(training_model.parameters()),
    )
    validation = run_multimodal_validation_epoch(
        _model(),
        _objective(),
        [_ignored_batch()],
    )
    for result in (training, validation):
        assert result.supervised_batch_count == 0
        assert result.active_target_count == 0
        assert result.optimizer_step_count == 0
        assert result.max_gradient_norm == 0.0
        assert all(
            getattr(result.loss_averages, field.name) == 0.0
            for field in fields(result.loss_averages)
        )


class _OnePassBatches:
    """Iterable that rejects sizing and a second consumption."""

    def __init__(self, batches: list[AlignedMultimodalBatch]) -> None:
        self._batches = batches
        self.iterations = 0

    def __len__(self) -> int:
        raise AssertionError("epoch runner must not call len")

    def __iter__(self) -> Iterator[AlignedMultimodalBatch]:
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("epoch runner consumed iterable twice")
        yield from self._batches


def test_epoch_runner_consumes_generator_once_without_len() -> None:
    """Accept a one-pass iterable while preserving its yielded order."""
    batches = _OnePassBatches([_batch(), _ignored_batch()])
    result = run_multimodal_validation_epoch(
        _model(),
        _objective(),
        batches,
    )
    assert result.batch_count == 2
    assert batches.iterations == 1


@pytest.mark.parametrize(
    "runner",
    [run_multimodal_training_epoch, run_multimodal_validation_epoch],
)
def test_epoch_runners_reject_invalid_or_empty_iterables(
    runner: object,
) -> None:
    """Reject strings, tensors, a single batch, bad items, and empty input."""
    model = _model()
    objective = _objective()
    optimizer = torch.optim.AdamW(model.parameters())
    for invalid in ("batches", torch.tensor([1]), _batch()):
        with pytest.raises(TypeError, match="iterable"):
            if runner is run_multimodal_training_epoch:
                run_multimodal_training_epoch(
                    model, objective, invalid, optimizer  # type: ignore[arg-type]
                )
            else:
                run_multimodal_validation_epoch(
                    model, objective, invalid  # type: ignore[arg-type]
                )
    with pytest.raises(TypeError, match="every epoch item"):
        if runner is run_multimodal_training_epoch:
            run_multimodal_training_epoch(
                model, objective, [object()], optimizer  # type: ignore[list-item]
            )
        else:
            run_multimodal_validation_epoch(
                model, objective, [object()]  # type: ignore[list-item]
            )
    with pytest.raises(ValueError, match="must not be empty"):
        if runner is run_multimodal_training_epoch:
            run_multimodal_training_epoch(model, objective, [], optimizer)
        else:
            run_multimodal_validation_epoch(model, objective, [])


def test_epoch_runner_call_boundaries_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call mode once and exactly one stage-13A step per ordered batch."""
    model = _model()
    objective = _objective()
    optimizer = torch.optim.AdamW(model.parameters())
    batches = [_batch(((True, False),)), _batch(((False, True),))]
    model.eval()
    reference = validate_multimodal_batch(
        model,
        objective,
        batches[0],
    )
    calls: list[AlignedMultimodalBatch] = []
    train_calls = 0
    original_train = model.train

    def tracked_train(mode: bool = True) -> object:
        nonlocal train_calls
        train_calls += 1
        return original_train(mode)

    def fake_step(
        step_model: object,
        step_objective: object,
        batch: AlignedMultimodalBatch,
        step_optimizer: object,
        **_kwargs: object,
    ) -> MultimodalTrainStepOutput:
        assert step_model is model
        assert step_objective is objective
        assert step_optimizer is optimizer
        calls.append(batch)
        return MultimodalTrainStepOutput(
            model_output=reference.model_output,
            loss_output=reference.loss_output,
            optimizer_step_performed=True,
            gradient_norm=torch.tensor(float(len(calls))),
        )

    monkeypatch.setattr(model, "train", tracked_train)
    monkeypatch.setattr(epochs_module, "train_multimodal_batch", fake_step)
    result = run_multimodal_training_epoch(
        model,
        objective,
        iter(batches),
        optimizer,
    )
    assert train_calls == 1
    assert calls == batches
    assert result.max_gradient_norm == 2.0


def test_validation_runner_calls_eval_and_each_step_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep validation orchestration at its exact public boundaries."""
    model = _model()
    objective = _objective()
    batches = [_batch(), _batch(((False, True),))]
    model.eval()
    reference = validate_multimodal_batch(model, objective, batches[0])
    eval_calls = 0
    calls: list[AlignedMultimodalBatch] = []
    original_eval = model.eval

    def tracked_eval() -> object:
        nonlocal eval_calls
        eval_calls += 1
        return original_eval()

    def fake_step(
        step_model: object,
        step_objective: object,
        batch: AlignedMultimodalBatch,
        **_kwargs: object,
    ) -> MultimodalValidationStepOutput:
        assert step_model is model
        assert step_objective is objective
        calls.append(batch)
        return reference

    monkeypatch.setattr(model, "eval", tracked_eval)
    monkeypatch.setattr(epochs_module, "validate_multimodal_batch", fake_step)
    result = run_multimodal_validation_epoch(model, objective, iter(batches))
    assert eval_calls == 1
    assert calls == batches
    assert result.optimizer_step_count == 0


def test_training_state_validation_and_immutability() -> None:
    """Validate immutable, non-negative, finite progress scalars."""
    state = MultimodalTrainingState()
    assert state == MultimodalTrainingState(0, 0, None)
    with pytest.raises(FrozenInstanceError):
        state.completed_epochs = 1  # type: ignore[misc]
    for value in (True, -1):
        with pytest.raises((TypeError, ValueError)):
            MultimodalTrainingState(completed_epochs=value)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            MultimodalTrainingState(global_optimizer_steps=value)  # type: ignore[arg-type]
    for value in (True, -1.0, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            MultimodalTrainingState(best_validation_loss=value)  # type: ignore[arg-type]


def test_advance_training_state_updates_steps_and_best_only_when_supervised() -> None:
    """Advance one epoch and retain the best supervised validation average."""
    original = MultimodalTrainingState(2, 7, 1.5)
    training = _epoch(EpochPhase.TRAIN, supervised=2, active=8)
    improved = _epoch(EpochPhase.VALIDATION, loss=1.2)
    result = advance_training_state(original, training, improved)
    assert result == MultimodalTrainingState(3, 9, 1.2)
    worse = advance_training_state(
        result,
        training,
        _epoch(EpochPhase.VALIDATION, loss=1.4),
    )
    assert worse.best_validation_loss == 1.2
    empty_validation = _epoch(
        EpochPhase.VALIDATION,
        supervised=0,
        empty=1,
        active=0,
        loss=0.0,
    )
    without_update = advance_training_state(worse, training, empty_validation)
    assert without_update.best_validation_loss == 1.2
    no_validation = advance_training_state(without_update, training)
    assert no_validation.best_validation_loss == 1.2
    assert original == MultimodalTrainingState(2, 7, 1.5)


def test_advance_training_state_rejects_wrong_phases_and_types() -> None:
    """Prevent validation aggregates from being counted as training progress."""
    state = MultimodalTrainingState()
    train_epoch = _epoch(EpochPhase.TRAIN)
    validation = _epoch(EpochPhase.VALIDATION)
    with pytest.raises(ValueError, match="training_epoch"):
        advance_training_state(state, validation)
    with pytest.raises(ValueError, match="validation_epoch"):
        advance_training_state(state, train_epoch, train_epoch)
    with pytest.raises(TypeError):
        advance_training_state(object(), train_epoch)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        advance_training_state(state, object())  # type: ignore[arg-type]


def test_epoch_source_scope_excludes_future_training_features() -> None:
    """Keep epoch orchestration free of later-stage mechanisms."""
    assert epochs_module.__file__ is not None
    source = Path(epochs_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "DataLoader",
        "sampler",
        "scheduler",
        "autocast",
        "GradScaler",
        "distributed",
        "early_stopping",
        "participant_id",
    )
    assert not any(token in source for token in forbidden)
