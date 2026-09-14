"""Tests for single-batch multimodal training and validation steps."""

from __future__ import annotations

import copy
import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

import emotion_model.training.steps as steps_module
from emotion_model.training import (
    EmotionTaskClassWeights,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
    MultimodalTrainStepOutput,
    MultimodalValidationStepOutput,
    train_multimodal_batch,
    validate_multimodal_batch,
)
from tests.test_multimodal_classifier import _batch, _model
from tests.test_multimodal_routing import (
    TinyPhysioClassifier,
    TinySpeechClassifier,
)


class _CountingAdamW(torch.optim.AdamW):
    """AdamW test double recording public zero/step calls."""

    def __init__(self, parameters: Any, *, lr: float = 1.0e-2) -> None:
        super().__init__(parameters, lr=lr, weight_decay=0.0)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Record and delegate gradient clearing."""
        self.zero_grad_calls += 1
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        """Record and delegate exactly one optimizer update."""
        self.step_calls += 1
        return super().step(closure=closure)


class _OrderedAdamW(torch.optim.AdamW):
    """AdamW test double recording ordering around backward."""

    def __init__(self, parameters: Any, events: list[str]) -> None:
        super().__init__(parameters, lr=1.0e-2, weight_decay=0.0)
        self.events = events

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Record gradient clearing before model execution."""
        self.events.append("zero_grad")
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        """Record the final update after backward."""
        self.events.append("optimizer_step")
        return super().step(closure=closure)


def _objective(
    weights: MultimodalLossWeights | None = None,
) -> MultimodalTrainingObjective:
    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(
            weights=weights or MultimodalLossWeights()
        )
    )


def _all_ignored_batch() -> Any:
    batch = _batch()
    ignored = torch.full_like(
        batch.arousal_labels,
        batch.label_ignore_index,
    )
    return replace(
        batch,
        arousal_labels=ignored,
        valence_labels=ignored.clone(),
        quadrant_labels=ignored.clone(),
    )


def _parameter_snapshots(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_train_step_calls_model_objective_zero_and_step_once() -> None:
    """Execute the authorized single-batch call sequence exactly once."""
    model = _model()
    model.train()
    objective = _objective()
    batch = _batch()
    optimizer = _CountingAdamW(model.parameters())
    model_calls: list[object] = []
    objective_calls: list[object] = []
    handles = (
        model.register_forward_hook(
            lambda _module, inputs, _output: model_calls.append(inputs[0])
        ),
        objective.register_forward_hook(
            lambda _module, inputs, _output: objective_calls.append(inputs[0])
        ),
    )
    result = train_multimodal_batch(model, objective, batch, optimizer)
    for handle in handles:
        handle.remove()
    assert isinstance(result, MultimodalTrainStepOutput)
    assert model_calls == [batch]
    assert objective_calls == [result.model_output]
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
    assert result.optimizer_step_performed
    assert result.gradient_norm.ndim == 0
    assert bool(torch.isfinite(result.gradient_norm))
    assert result.gradient_norm > 0


def test_train_step_exact_order_includes_one_backward() -> None:
    """Prove zero, model, objective, backward, and step ordering independently."""
    events: list[str] = []
    model = _model()
    model.train()
    objective = _objective()
    optimizer = _OrderedAdamW(model.parameters(), events)

    def model_hook(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        events.append("model_forward")

    def objective_hook(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        events.append("objective_forward")
        cast(Any, output).total_loss.register_hook(
            lambda gradient: events.append("backward") or gradient
        )

    handles = (
        model.register_forward_hook(model_hook),
        objective.register_forward_hook(objective_hook),
    )
    train_multimodal_batch(model, objective, _batch(), optimizer)
    for handle in handles:
        handle.remove()
    assert events == [
        "zero_grad",
        "model_forward",
        "objective_forward",
        "backward",
        "optimizer_step",
    ]


def test_train_step_updates_concrete_parameters_and_keeps_them_finite() -> None:
    """Update final, fusion, representation, and reliability parameters."""
    torch.manual_seed(7)
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.train()
    objective = _objective()
    optimizer = _CountingAdamW(model.parameters())
    speech = cast(
        TinySpeechClassifier,
        model.batch_scheduler.speech_classifier,
    )
    physiology = cast(
        TinyPhysioClassifier,
        model.batch_scheduler.physiology_classifier,
    )
    active = {
        "trunk": cast(nn.Linear, model.classifier_trunk[1]).weight,
        "arousal": model.arousal_head.weight,
        "valence": model.valence_head.weight,
        "speech_projection": cast(
            nn.Linear,
            model.multimodal_fusion.speech_projection[1],
        ).weight,
        "physiology_projection": cast(
            nn.Linear,
            model.multimodal_fusion.physiology_projection[1],
        ).weight,
        "dynamic_gate": cast(
            nn.Linear,
            model.multimodal_fusion.modality_gate[-1],
        ).weight,
        "speech_representation": speech.input_projection.weight,
        "physiology_representation": physiology.input_projection.weight,
    }
    inactive = {
        "speech_reliability": speech.reliability_head.weight,
        "physiology_reliability": physiology.reliability_head.weight,
        "speech_arousal": speech.arousal_head.weight,
        "speech_valence": speech.valence_head.weight,
        "physiology_arousal": physiology.arousal_head.weight,
        "physiology_valence": physiology.valence_head.weight,
        "fused_quadrant": cast(nn.Linear, model.quadrant_head).weight,
        "speech_quadrant": cast(nn.Linear, speech.quadrant_head).weight,
        "physiology_quadrant": cast(nn.Linear, physiology.quadrant_head).weight,
    }
    before = {
        name: parameter.detach().clone()
        for name, parameter in {**active, **inactive}.items()
    }
    result = train_multimodal_batch(
        model,
        objective,
        _batch(),
        optimizer,
    )
    assert result.optimizer_step_performed
    for name, parameter in active.items():
        assert not torch.equal(parameter, before[name]), name
        assert bool(torch.isfinite(parameter).all())
    for name, parameter in inactive.items():
        assert torch.equal(parameter, before[name]), name


def test_unclipped_gradient_norm_matches_independent_global_l2() -> None:
    """Return the pre-step global L2 norm without altering gradients."""
    model = _model()
    model.train()
    result = train_multimodal_batch(
        model,
        _objective(),
        _batch(),
        _CountingAdamW(model.parameters()),
    )
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    manual = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    assert torch.allclose(result.gradient_norm, manual)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_train_step_rejects_invalid_gradient_clip_bound(
    value: object,
    error_type: type[Exception],
) -> None:
    """Require a finite positive optional global norm bound."""
    model = _model()
    model.train()
    with pytest.raises(error_type):
        train_multimodal_batch(
            model,
            _objective(),
            _batch(),
            _CountingAdamW(model.parameters()),
            max_gradient_norm=cast(float, value),
        )


def test_gradient_clipping_calls_public_api_once_and_returns_preclip_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate clipping to PyTorch and preserve its pre-clipping norm."""
    model = _model()
    model.train()
    calls: list[tuple[float, bool, Tensor]] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(
        parameters: Any,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> Tensor:
        parameter_list = list(parameters)
        result = original(
            parameter_list,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
        calls.append((max_norm, error_if_nonfinite, result.detach().clone()))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    result = train_multimodal_batch(
        model,
        _objective(),
        _batch(),
        _CountingAdamW(model.parameters()),
        max_gradient_norm=0.05,
    )
    assert len(calls) == 1
    assert calls[0][0] == 0.05
    assert calls[0][1] is True
    assert torch.allclose(result.gradient_norm, calls[0][2])
    clipped_norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    clipped = torch.linalg.vector_norm(torch.stack(clipped_norms), ord=2)
    assert clipped <= 0.05001


def test_nonfinite_gradient_is_rejected_before_optimizer_step() -> None:
    """Detect invalid task gradients before clipping or parameter mutation."""
    model = _model()
    model.train()
    optimizer = _CountingAdamW(model.parameters())
    before = model.arousal_head.weight.detach().clone()
    handle = model.arousal_head.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    with pytest.raises(RuntimeError, match="gradient contains NaN or Inf"):
        train_multimodal_batch(model, _objective(), _batch(), optimizer)
    handle.remove()
    assert optimizer.step_calls == 0
    assert torch.equal(model.arousal_head.weight, before)


@pytest.mark.parametrize("empty_case", ["neither", "protocol"])
def test_empty_supervision_skips_backward_step_and_optimizer_state(
    empty_case: str,
) -> None:
    """Clear old gradients but leave parameters and optimizer state unchanged."""
    model = _model()
    model.train()
    optimizer = _CountingAdamW(model.parameters())
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    before_parameters = _parameter_snapshots(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    batch = (
        _batch(((False, False), (False, False)))
        if empty_case == "neither"
        else _all_ignored_batch()
    )
    labels_before = tuple(
        value.clone()
        for value in (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        )
    )
    result = train_multimodal_batch(
        model,
        _objective(),
        batch,
        optimizer,
    )
    assert not result.optimizer_step_performed
    assert torch.equal(
        result.gradient_norm,
        torch.zeros_like(result.gradient_norm),
    )
    assert result.loss_output.active_target_count == 0
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 0
    assert optimizer.state_dict() == before_optimizer
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before_parameters[name])
        assert parameter.grad is None
    for actual, snapshot in zip(
        (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        ),
        labels_before,
    ):
        assert torch.equal(actual, snapshot)


def test_only_missing_auxiliary_supervision_skips_step() -> None:
    """Skip when the sole enabled auxiliary modality is absent."""
    model = _model()
    model.train()
    optimizer = _CountingAdamW(model.parameters())
    objective = _objective(
        MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
    )
    result = train_multimodal_batch(
        model,
        objective,
        _batch(((False, True),)),
        optimizer,
    )
    assert result.loss_output.active_target_count == 0
    assert not result.optimizer_step_performed
    assert optimizer.step_calls == 0


def test_empty_supervision_does_not_call_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return before backward and the public clipping boundary."""
    model = _model()
    model.train()
    optimizer = _CountingAdamW(model.parameters())
    clip_calls = 0

    def forbidden_clip(*_args: object, **_kwargs: object) -> Tensor:
        nonlocal clip_calls
        clip_calls += 1
        raise AssertionError("clip_grad_norm_ must not run for empty supervision")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", forbidden_clip)
    result = train_multimodal_batch(
        model,
        _objective(),
        _batch(((False, False),)),
        optimizer,
        max_gradient_norm=1.0,
    )
    assert not result.optimizer_step_performed
    assert clip_calls == 0
    assert optimizer.step_calls == 0


def test_speech_auxiliary_step_does_not_update_physiology_heads() -> None:
    """Keep auxiliary optimizer updates isolated to the selected modality."""
    model = _model()
    model.train()
    speech = cast(
        TinySpeechClassifier,
        model.batch_scheduler.speech_classifier,
    )
    physiology = cast(
        TinyPhysioClassifier,
        model.batch_scheduler.physiology_classifier,
    )
    speech_before = speech.arousal_head.weight.detach().clone()
    physiology_before = physiology.arousal_head.weight.detach().clone()
    result = train_multimodal_batch(
        model,
        _objective(
            MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
        ),
        _batch(),
        _CountingAdamW(model.parameters()),
    )
    assert result.optimizer_step_performed
    assert not torch.equal(speech.arousal_head.weight, speech_before)
    assert torch.equal(physiology.arousal_head.weight, physiology_before)


def test_frozen_or_unused_parameters_without_gradients_are_legal() -> None:
    """Ignore absent gradients while measuring and updating active parameters."""
    model = _model(independent_quadrant=True)
    model.train()
    assert model.quadrant_head is not None
    for parameter in model.quadrant_head.parameters():
        parameter.requires_grad_(False)
    result = train_multimodal_batch(
        model,
        _objective(),
        _batch(),
        _CountingAdamW(model.parameters()),
    )
    assert result.optimizer_step_performed
    assert result.gradient_norm > 0
    for parameter in model.quadrant_head.parameters():
        assert parameter.grad is None


def test_train_step_requires_correct_types_and_train_mode() -> None:
    """Reject wrong ownership or silently incorrect dropout mode."""
    model = _model()
    objective = _objective()
    batch = _batch()
    optimizer = _CountingAdamW(model.parameters())
    model.eval()
    with pytest.raises(RuntimeError, match=r"model\.train"):
        train_multimodal_batch(model, objective, batch, optimizer)
    model.train()
    with pytest.raises(TypeError, match="model"):
        train_multimodal_batch(
            cast(Any, object()),
            objective,
            batch,
            optimizer,
        )
    with pytest.raises(TypeError, match="objective"):
        train_multimodal_batch(
            model,
            cast(Any, object()),
            batch,
            optimizer,
        )
    with pytest.raises(TypeError, match="batch"):
        train_multimodal_batch(
            model,
            objective,
            cast(Any, object()),
            optimizer,
        )
    with pytest.raises(TypeError, match="optimizer"):
        train_multimodal_batch(
            model,
            objective,
            batch,
            cast(Any, object()),
        )


def test_validation_runs_model_and_objective_once_under_no_grad() -> None:
    """Return one complete eval result without constructing an autograd graph."""
    model = _model()
    model.eval()
    objective = _objective()
    batch = _batch()
    calls = {"model": 0, "objective": 0}
    handles = (
        model.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "model",
                calls["model"] + 1,
            )
        ),
        objective.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "objective",
                calls["objective"] + 1,
            )
        ),
    )
    result = validate_multimodal_batch(model, objective, batch)
    for handle in handles:
        handle.remove()
    assert isinstance(result, MultimodalValidationStepOutput)
    assert calls == {"model": 1, "objective": 1}
    for tensor in (
        result.model_output.arousal_logits,
        result.model_output.valence_logits,
        result.model_output.fused_embedding,
        result.loss_output.total_loss,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_validation_preserves_parameters_and_accepts_class_weights() -> None:
    """Keep eval parameters fixed while forwarding caller-owned class weights."""
    model = _model()
    model.eval()
    before = _parameter_snapshots(model)
    weights_tensor = torch.tensor([2.0, 3.0])
    class_weights = EmotionTaskClassWeights(arousal=weights_tensor)
    weights_before = weights_tensor.clone()
    validate_multimodal_batch(
        model,
        _objective(),
        _batch(),
        class_weights=class_weights,
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    assert torch.equal(weights_tensor, weights_before)


def test_validation_returns_empty_supervision_zero_loss() -> None:
    """Retain full outputs for an empty validation batch."""
    model = _model()
    model.eval()
    result = validate_multimodal_batch(
        model,
        _objective(),
        _batch(((False, False), (False, False))),
    )
    assert result.loss_output.active_target_count == 0
    assert torch.equal(
        result.loss_output.total_loss,
        torch.zeros_like(result.loss_output.total_loss),
    )
    assert not result.loss_output.total_loss.requires_grad


def test_validation_requires_eval_mode_and_correct_types() -> None:
    """Refuse to change model mode implicitly."""
    model = _model()
    model.train()
    with pytest.raises(RuntimeError, match=r"model\.eval"):
        validate_multimodal_batch(model, _objective(), _batch())
    model.eval()
    with pytest.raises(TypeError, match="objective"):
        validate_multimodal_batch(
            model,
            cast(Any, object()),
            _batch(),
        )


def test_step_outputs_are_frozen() -> None:
    """Expose immutable per-batch diagnostics."""
    model = _model()
    model.train()
    train_output = train_multimodal_batch(
        model,
        _objective(),
        _batch(),
        _CountingAdamW(model.parameters()),
    )
    with pytest.raises(FrozenInstanceError):
        train_output.optimizer_step_performed = False  # type: ignore[misc]
    model.eval()
    validation_output = validate_multimodal_batch(
        model,
        _objective(),
        _batch(),
    )
    with pytest.raises(FrozenInstanceError):
        validation_output.model_output = train_output.model_output  # type: ignore[misc]


def test_steps_store_no_runtime_state_on_model_or_objective() -> None:
    """Avoid retaining batches, outputs, losses, or counters between calls."""
    model = _model()
    model_keys = tuple(model.state_dict())
    model.train()
    objective = _objective()
    objective_state = objective.state_dict().copy()
    train_multimodal_batch(
        model,
        objective,
        _batch(),
        _CountingAdamW(model.parameters()),
    )
    assert tuple(model.state_dict()) == model_keys
    assert objective.state_dict() == objective_state == {}
    assert not any(
        key in vars(objective)
        for key in ("batch", "model_output", "loss_output", "step")
    )


def test_scope_excludes_epoch_metrics_checkpoint_and_advanced_training() -> None:
    """Keep the module restricted to one explicit batch operation."""
    source = Path(steps_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "DataLoader",
        "autocast",
        "GradScaler",
        "DistributedDataParallel",
        "lr_scheduler",
        "checkpoint_manager",
        "early_stopping",
        "metric_state",
        "epoch_loop",
        "gradient_accumulation",
    ):
        assert forbidden not in source
    assert "optimizer =" not in source
    assert "\n    model.train()\n" not in source
    assert "\n    model.eval()\n" not in source
    train_source = inspect.getsource(train_multimodal_batch)
    validation_source = inspect.getsource(validate_multimodal_batch)
    assert train_source.count("optimizer.step()") == 1
    assert "torch.no_grad()" in validation_source
    for prohibited in (
        "clean_speech",
        "reconstruct_waveform",
        "denoise_output",
        "noise_subtraction",
        "speech_enhancement",
    ):
        assert prohibited not in source
