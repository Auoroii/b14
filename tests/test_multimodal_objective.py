"""Tests for multimodal loss configuration and objective composition."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn

import emotion_model.training.objectives as objective_module
from emotion_model.multimodal import class_weighted_cross_entropy
from emotion_model.training import (
    ClassificationLossKind,
    EmotionTaskClassWeights,
    MultimodalLossOutput,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
)
from tests.test_multimodal_classifier import (
    _assert_finite_nonzero_gradient,
    _assert_zero_or_none_gradient,
    _batch,
    _model,
    _unsafe_copy,
)
from tests.test_multimodal_routing import (
    TinyPhysioClassifier,
    TinySpeechClassifier,
)


def _objective(
    *,
    loss_kind: ClassificationLossKind = (
        ClassificationLossKind.WEIGHTED_CROSS_ENTROPY
    ),
    weights: MultimodalLossWeights | None = None,
    gamma: float = 2.0,
    speech_aux_min_activity_ratio: float | None = None,
) -> MultimodalTrainingObjective:
    """Create one parameter-free objective."""
    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(
            loss_kind=loss_kind,
            weights=weights or MultimodalLossWeights(),
            focal_gamma=gamma,
            speech_aux_min_activity_ratio=speech_aux_min_activity_ratio,
        )
    )


def _all_terms_weights() -> MultimodalLossWeights:
    return MultimodalLossWeights(
        fused=1.2,
        speech_auxiliary=0.3,
        physiology_auxiliary=0.4,
        fused_quadrant=0.5,
        speech_quadrant=0.6,
        physiology_quadrant=0.7,
    )


def _ignored_batch() -> object:
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


def test_default_configuration_is_fused_weighted_ce_only() -> None:
    """Expose the documented immutable default objective."""
    config = MultimodalObjectiveConfig()
    assert config.loss_kind is ClassificationLossKind.WEIGHTED_CROSS_ENTROPY
    assert config.weights == MultimodalLossWeights()
    assert config.weights.fused == 1.0
    assert config.weights.speech_auxiliary == 0.0
    assert config.weights.physiology_auxiliary == 0.0
    assert config.weights.fused_quadrant == 0.0
    assert config.focal_gamma == 2.0


def test_all_auxiliary_and_quadrant_weights_are_not_normalized() -> None:
    """Retain arbitrary non-negative objective coefficients verbatim."""
    weights = _all_terms_weights()
    assert sum(
        (
            weights.fused,
            weights.speech_auxiliary,
            weights.physiology_auxiliary,
            weights.fused_quadrant,
            weights.speech_quadrant,
            weights.physiology_quadrant,
        )
    ) == pytest.approx(3.7)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("fused", -0.1, ValueError),
        ("speech_auxiliary", float("nan"), ValueError),
        ("physiology_auxiliary", float("inf"), ValueError),
        ("fused_quadrant", True, TypeError),
        ("speech_quadrant", "1", TypeError),
    ],
)
def test_loss_weights_reject_invalid_values(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    """Reject negative, non-finite, boolean, and non-real coefficients."""
    arguments: dict[str, object] = {
        "fused": 1.0,
        "speech_auxiliary": 0.0,
        "physiology_auxiliary": 0.0,
        "fused_quadrant": 0.0,
        "speech_quadrant": 0.0,
        "physiology_quadrant": 0.0,
    }
    arguments[field] = value
    with pytest.raises(error_type):
        MultimodalLossWeights(**arguments)  # type: ignore[arg-type]


def test_loss_weights_reject_all_zero_and_are_frozen() -> None:
    """Require at least one enabled term and immutable configuration."""
    with pytest.raises(ValueError, match="at least one"):
        MultimodalLossWeights(fused=0.0)
    weights = MultimodalLossWeights()
    with pytest.raises(FrozenInstanceError):
        weights.fused = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("loss_kind", "gamma", "error_type"),
    [
        ("focal", 2.0, TypeError),
        (ClassificationLossKind.FOCAL, -0.1, ValueError),
        (ClassificationLossKind.FOCAL, float("nan"), ValueError),
        (ClassificationLossKind.FOCAL, True, TypeError),
    ],
)
def test_objective_config_rejects_invalid_kind_or_gamma(
    loss_kind: object,
    gamma: object,
    error_type: type[Exception],
) -> None:
    """Require the stable enum and a finite non-negative focal exponent."""
    with pytest.raises(error_type):
        MultimodalObjectiveConfig(
            loss_kind=cast(ClassificationLossKind, loss_kind),
            focal_gamma=cast(float, gamma),
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (-0.1, ValueError),
        (1.0, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
        ("0.0", TypeError),
    ],
)
def test_objective_config_rejects_invalid_speech_activity_threshold(
    value: object,
    error_type: type[Exception],
) -> None:
    """Require an optional finite speech auxiliary threshold in ``[0,1)``."""

    with pytest.raises(error_type):
        MultimodalObjectiveConfig(
            speech_aux_min_activity_ratio=cast(float, value)
        )


def test_speech_auxiliary_uses_only_observed_rows_strictly_above_threshold() -> None:
    """Match a manual loss over ratios ``0.1`` and ``0.8``, excluding zero."""

    base = _batch(((True, True), (True, True), (True, True)))
    batch = replace(
        base,
        arousal_labels=torch.tensor([0, 1, 0]),
        valence_labels=torch.tensor([1, 0, 1]),
        quadrant_labels=torch.tensor([2, 1, 2]),
        speech_activity_ratios=torch.tensor([0.0, 0.1, 0.8]),
        speech_activity_observed=torch.ones(3, dtype=torch.bool),
    )
    model = _model().eval()
    output = model(batch)
    weights = MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
    loss = _objective(
        weights=weights,
        speech_aux_min_activity_ratio=0.0,
    )(output, batch)
    compact = output.fusion_output.scheduled_outputs.speech_compact_output
    assert compact is not None
    manual = class_weighted_cross_entropy(
        compact.arousal_logits[1:],
        batch.arousal_labels[1:],
        ignore_index=batch.label_ignore_index,
    ) + class_weighted_cross_entropy(
        compact.valence_logits[1:],
        batch.valence_labels[1:],
        ignore_index=batch.label_ignore_index,
    )

    torch.testing.assert_close(loss.speech_auxiliary_loss, manual)
    assert loss.active_target_count == 4


def test_all_silent_speech_auxiliary_is_graph_safe_zero_but_fused_is_active() -> None:
    """Keep fused supervision while all compact speech targets are ignored."""

    base = _batch(((True, True), (True, True), (True, True)))
    batch = replace(
        base,
        speech_activity_ratios=torch.zeros(3),
        speech_activity_observed=torch.ones(3, dtype=torch.bool),
    )
    model = _model().eval()
    output = model(batch)
    loss = _objective(
        weights=MultimodalLossWeights(fused=1.0, speech_auxiliary=0.3),
        speech_aux_min_activity_ratio=0.0,
    )(output, batch)

    assert torch.equal(
        loss.speech_auxiliary_loss,
        torch.zeros_like(loss.speech_auxiliary_loss),
    )
    assert bool(torch.isfinite(loss.speech_auxiliary_loss))
    assert loss.speech_auxiliary_loss.grad_fn is not None
    assert loss.fused_loss > 0.0
    torch.testing.assert_close(loss.total_loss, loss.fused_loss)
    assert loss.active_target_count == 6
    loss.total_loss.backward()


@pytest.mark.parametrize("with_ratios", [False, True])
def test_enabled_speech_activity_filter_requires_observed_diagnostics(
    with_ratios: bool,
) -> None:
    """Fail loudly for missing tensors or an unobserved compact speech row."""

    batch = _batch(((True, True), (True, True), (True, True)))
    if with_ratios:
        batch = replace(
            batch,
            speech_activity_ratios=torch.tensor([0.0, 0.1, 0.8]),
            speech_activity_observed=torch.tensor([True, False, True]),
        )
    model = _model().eval()
    objective = _objective(
        weights=MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0),
        speech_aux_min_activity_ratio=0.0,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "speech auxiliary activity filtering requires observed speech "
            "activity diagnostics"
        ),
    ):
        objective(model(batch), batch)


def test_configuration_dataclasses_are_frozen() -> None:
    """Prevent runtime mutation of selected objective semantics."""
    config = MultimodalObjectiveConfig()
    with pytest.raises(FrozenInstanceError):
        config.focal_gamma = 3.0  # type: ignore[misc]
    weights = EmotionTaskClassWeights()
    with pytest.raises(FrozenInstanceError):
        weights.arousal = torch.ones(2)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("arousal", torch.ones(3), ValueError),
        ("valence", torch.ones(2, dtype=torch.long), TypeError),
        ("quadrant", torch.tensor([1.0, 1.0, float("nan"), 1.0]), ValueError),
        ("quadrant", torch.tensor([1.0, 1.0, float("inf"), 1.0]), ValueError),
        ("arousal", torch.tensor([1.0, 0.0]), ValueError),
        ("valence", torch.tensor([-1.0, 1.0]), ValueError),
    ],
)
def test_class_weights_reject_invalid_contract(
    field: str,
    value: Tensor,
    error_type: type[Exception],
) -> None:
    """Validate task-specific shape, floating dtype, finite, positive values."""
    arguments: dict[str, Tensor | None] = {
        "arousal": None,
        "valence": None,
        "quadrant": None,
    }
    arguments[field] = value
    with pytest.raises(error_type):
        EmotionTaskClassWeights(**arguments)


def test_class_weights_preserve_identity_values_and_do_not_normalize() -> None:
    """Keep caller tensors unchanged and on their existing dtype/device."""
    arousal = torch.tensor([2.0, 5.0], dtype=torch.float64)
    valence = torch.tensor([3.0, 7.0], dtype=torch.float64)
    quadrant = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    snapshots = tuple(value.clone() for value in (arousal, valence, quadrant))
    weights = EmotionTaskClassWeights(arousal, valence, quadrant)
    assert weights.arousal is arousal
    assert weights.valence is valence
    assert weights.quadrant is quadrant
    for actual, snapshot in zip((arousal, valence, quadrant), snapshots):
        assert torch.equal(actual, snapshot)


def test_objective_has_no_parameters_or_runtime_state() -> None:
    """Store only immutable external configuration, not runtime tensors."""
    objective = _objective()
    assert list(objective.parameters()) == []
    assert objective.state_dict() == {}
    assert vars(objective)["_modules"] == {}
    assert isinstance(objective.config, MultimodalObjectiveConfig)


def test_default_objective_matches_public_losses_and_active_count() -> None:
    """Sum final arousal and valence means over valid modality rows."""
    model = _model()
    model.eval()
    batch = _batch()
    output = model(batch)
    loss = _objective()(output, batch)
    arousal = batch.arousal_labels.clone()
    valence = batch.valence_labels.clone()
    arousal[~output.sample_valid] = batch.label_ignore_index
    valence[~output.sample_valid] = batch.label_ignore_index
    expected_arousal = class_weighted_cross_entropy(
        output.arousal_logits,
        arousal,
        ignore_index=batch.label_ignore_index,
    )
    expected_valence = class_weighted_cross_entropy(
        output.valence_logits,
        valence,
        ignore_index=batch.label_ignore_index,
    )
    assert isinstance(loss, MultimodalLossOutput)
    assert torch.equal(loss.fused_arousal_loss, expected_arousal)
    assert torch.equal(loss.fused_valence_loss, expected_valence)
    assert torch.equal(loss.fused_loss, expected_arousal + expected_valence)
    assert torch.equal(loss.total_loss, loss.fused_loss)
    assert loss.active_target_count == 6
    for name, value in vars(loss).items():
        if name != "active_target_count":
            assert isinstance(value, Tensor)
            assert value.ndim == 0
            assert bool(torch.isfinite(value))


def test_total_formula_weights_each_component_once_without_normalization() -> None:
    """Match the six-term weighted sum without count or modality rescaling."""
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.eval()
    batch = _batch()
    output = model(batch)
    weights = _all_terms_weights()
    loss = _objective(weights=weights)(output, batch)
    manual = (
        weights.fused * loss.fused_loss
        + weights.speech_auxiliary * loss.speech_auxiliary_loss
        + weights.physiology_auxiliary * loss.physiology_auxiliary_loss
        + weights.fused_quadrant * loss.fused_quadrant_loss
        + weights.speech_quadrant * loss.speech_quadrant_loss
        + weights.physiology_quadrant * loss.physiology_quadrant_loss
    )
    assert torch.equal(loss.total_loss, manual)
    assert loss.active_target_count == 21
    assert loss.total_loss.grad_fn is not None


def test_total_formula_matches_asymmetric_fixed_public_loss_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independently detect repeated, omitted, normalized, or divided terms."""
    fixed_values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))

    def fixed_loss(
        logits: Tensor,
        _target: Tensor,
        **_kwargs: object,
    ) -> Tensor:
        return logits.sum() * 0.0 + next(fixed_values)

    monkeypatch.setattr(
        objective_module,
        "class_weighted_cross_entropy",
        fixed_loss,
    )
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.eval()
    batch = _batch()
    weights = _all_terms_weights()
    loss = _objective(weights=weights)(model(batch), batch)
    assert loss.fused_arousal_loss == 1.0
    assert loss.fused_valence_loss == 2.0
    assert loss.fused_quadrant_loss == 3.0
    assert loss.speech_arousal_loss == 4.0
    assert loss.speech_valence_loss == 5.0
    assert loss.speech_quadrant_loss == 6.0
    assert loss.physiology_arousal_loss == 7.0
    assert loss.physiology_valence_loss == 8.0
    assert loss.physiology_quadrant_loss == 9.0
    expected = (
        1.2 * (1.0 + 2.0)
        + 0.3 * (4.0 + 5.0)
        + 0.4 * (7.0 + 8.0)
        + 0.5 * 3.0
        + 0.6 * 6.0
        + 0.7 * 9.0
    )
    assert torch.allclose(
        loss.total_loss,
        loss.total_loss.new_tensor(expected),
    )
    assert loss.active_target_count == 21


def test_public_weighted_ce_is_called_with_task_weights_and_ignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spy on the existing weighted CE API rather than duplicated formulas."""
    calls: list[dict[str, object]] = []
    original = objective_module.class_weighted_cross_entropy

    def spy(
        logits: Tensor,
        target: Tensor,
        **kwargs: object,
    ) -> Tensor:
        calls.append(
            {
                "logits": logits,
                "target": target,
                **kwargs,
            }
        )
        return original(logits, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(objective_module, "class_weighted_cross_entropy", spy)
    model = _model()
    model.eval()
    batch = _batch()
    output = model(batch)
    arousal_weights = torch.tensor([1.0, 2.0])
    valence_weights = torch.tensor([3.0, 4.0])
    _objective()(
        output,
        batch,
        class_weights=EmotionTaskClassWeights(
            arousal=arousal_weights,
            valence=valence_weights,
        ),
    )
    assert len(calls) == 2
    assert calls[0]["class_weights"] is arousal_weights
    assert calls[1]["class_weights"] is valence_weights
    assert calls[0]["ignore_index"] == batch.label_ignore_index
    assert calls[0]["reduction"] == "mean"


def test_public_focal_is_called_with_gamma_and_quadrant_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route enabled binary and quadrant tasks through public focal loss."""
    calls: list[dict[str, object]] = []
    original = objective_module.multiclass_focal_loss

    def spy(
        logits: Tensor,
        target: Tensor,
        **kwargs: object,
    ) -> Tensor:
        calls.append(
            {
                "logits": logits,
                "target": target,
                **kwargs,
            }
        )
        return original(logits, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(objective_module, "multiclass_focal_loss", spy)
    model = _model(independent_quadrant=True)
    model.eval()
    batch = _batch()
    quadrant_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _objective(
        loss_kind=ClassificationLossKind.FOCAL,
        weights=MultimodalLossWeights(fused=1.0, fused_quadrant=0.5),
        gamma=1.25,
    )(
        model(batch),
        batch,
        class_weights=EmotionTaskClassWeights(quadrant=quadrant_weights),
    )
    assert len(calls) == 3
    assert all(call["gamma"] == 1.25 for call in calls)
    assert calls[2]["class_weights"] is quadrant_weights


def test_class_weight_dtype_mismatch_is_rejected_without_conversion() -> None:
    """Require callers to place class weights on the logits dtype/device."""
    model = _model(dtype=torch.float32)
    model.eval()
    batch = _batch()
    output = model(batch)
    weights = EmotionTaskClassWeights(
        arousal=torch.ones(2, dtype=torch.float64)
    )
    with pytest.raises(TypeError, match="match logits dtype"):
        _objective()(output, batch, class_weights=weights)
    assert weights.arousal is not None
    assert weights.arousal.dtype == torch.float64


def test_protocol_ignore_and_invalid_modality_are_combined_on_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve protocol ignore and add modality ignore without label mutation."""
    batch = _batch()
    arousal = batch.arousal_labels.clone()
    arousal[1] = batch.label_ignore_index
    quadrant = batch.quadrant_labels.clone()
    quadrant[1] = batch.label_ignore_index
    batch = replace(
        batch,
        arousal_labels=arousal,
        quadrant_labels=quadrant,
    )
    before = tuple(
        value.clone()
        for value in (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        )
    )
    observed: list[Tensor] = []
    original = objective_module.class_weighted_cross_entropy

    def spy(logits: Tensor, target: Tensor, **kwargs: object) -> Tensor:
        observed.append(target.clone())
        return original(logits, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(objective_module, "class_weighted_cross_entropy", spy)
    model = _model()
    model.eval()
    output = model(batch)
    loss = _objective()(output, batch)
    assert observed[0].tolist() == [0, -100, 0, -100]
    assert observed[1].tolist() == [1, 1, 1, -100]
    assert loss.active_target_count == 5
    for actual, snapshot in zip(
        (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        ),
        before,
    ):
        assert torch.equal(actual, snapshot)


@pytest.mark.parametrize("modality", ["speech", "physiology"])
def test_auxiliary_uses_compact_indices_and_not_scattered_zero_rows(
    modality: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select full labels by the exact modality compact index mapping."""
    weights = (
        MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
        if modality == "speech"
        else MultimodalLossWeights(fused=0.0, physiology_auxiliary=1.0)
    )
    model = _model()
    model.eval()
    base_batch = _batch()
    batch = replace(
        base_batch,
        arousal_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        valence_labels=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        quadrant_labels=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    output = model(batch)
    expected_indices = (
        batch.speech.batch_indices
        if modality == "speech" and batch.speech is not None
        else cast(object, batch.physiology).batch_indices  # type: ignore[attr-defined]
    )
    observed: list[tuple[int, ...]] = []
    original = objective_module.class_weighted_cross_entropy

    def spy(logits: Tensor, target: Tensor, **kwargs: object) -> Tensor:
        observed.append(tuple(target.tolist()))
        assert logits.shape[0] == expected_indices.shape[0]
        return original(logits, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(objective_module, "class_weighted_cross_entropy", spy)
    loss = _objective(weights=weights)(output, batch)
    expected = (
        [(0, 1), (0, 0)]
        if modality == "speech"
        else [(0, 0), (0, 1)]
    )
    assert observed == expected
    assert loss.active_target_count == 4
    if modality == "speech":
        assert loss.speech_auxiliary_loss > 0
        assert torch.equal(
            loss.physiology_auxiliary_loss,
            torch.zeros_like(loss.physiology_auxiliary_loss),
        )
    else:
        assert loss.physiology_auxiliary_loss > 0
        assert torch.equal(
            loss.speech_auxiliary_loss,
            torch.zeros_like(loss.speech_auxiliary_loss),
        )


@pytest.mark.parametrize("modality", ["speech", "physiology"])
def test_missing_enabled_auxiliary_is_graph_safe_zero(
    modality: str,
) -> None:
    """Treat absent optional modality supervision as empty, not erroneous."""
    pattern = ((False, True),) if modality == "speech" else ((True, False),)
    batch = _batch(pattern)
    weights = (
        MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
        if modality == "speech"
        else MultimodalLossWeights(fused=0.0, physiology_auxiliary=1.0)
    )
    model = _model()
    model.eval()
    output = model(batch)
    loss = _objective(weights=weights)(output, batch)
    assert loss.active_target_count == 0
    assert torch.equal(loss.total_loss, torch.zeros_like(loss.total_loss))
    assert loss.total_loss.grad_fn is not None
    loss.total_loss.backward()


@pytest.mark.parametrize(
    ("target", "weights"),
    [
        (
            "fused",
            MultimodalLossWeights(fused=0.0, fused_quadrant=1.0),
        ),
        (
            "speech",
            MultimodalLossWeights(fused=0.0, speech_quadrant=1.0),
        ),
        (
            "physiology",
            MultimodalLossWeights(fused=0.0, physiology_quadrant=1.0),
        ),
    ],
)
def test_enabled_quadrant_requires_corresponding_independent_head(
    target: str,
    weights: MultimodalLossWeights,
) -> None:
    """Fail clearly only when a present enabled quadrant target lacks logits."""
    model = _model()
    model.eval()
    batch = _batch()
    with pytest.raises(RuntimeError, match=f"{target} quadrant"):
        _objective(weights=weights)(model(batch), batch)


@pytest.mark.parametrize("target", ["fused", "speech", "physiology"])
def test_enabled_quadrant_trains_only_corresponding_head(target: str) -> None:
    """Route quadrant CE to exactly the explicitly weighted independent head."""
    arguments = {
        "fused": {"fused_quadrant": 1.0},
        "speech": {"speech_quadrant": 1.0},
        "physiology": {"physiology_quadrant": 1.0},
    }[target]
    weights = MultimodalLossWeights(fused=0.0, **arguments)
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.eval()
    batch = _batch()
    output = model(batch)
    loss = _objective(weights=weights)(output, batch)
    loss.total_loss.backward()
    speech = cast(
        TinySpeechClassifier,
        model.batch_scheduler.speech_classifier,
    )
    physiology = cast(
        TinyPhysioClassifier,
        model.batch_scheduler.physiology_classifier,
    )
    heads = {
        "fused": cast(nn.Linear, model.quadrant_head),
        "speech": cast(nn.Linear, speech.quadrant_head),
        "physiology": cast(nn.Linear, physiology.quadrant_head),
    }
    for name, head in heads.items():
        if name == target:
            _assert_finite_nonzero_gradient(head.weight)
        else:
            _assert_zero_or_none_gradient(head.weight)


def test_default_gradient_boundary_matches_fused_only_classifier() -> None:
    """Train fused representations/gates but not unimodal prediction heads."""
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.eval()
    batch = _batch()
    output = model(batch)
    _objective()(output, batch).total_loss.backward()
    speech = cast(
        TinySpeechClassifier,
        model.batch_scheduler.speech_classifier,
    )
    physiology = cast(
        TinyPhysioClassifier,
        model.batch_scheduler.physiology_classifier,
    )
    for parameter in (
        cast(nn.Linear, model.classifier_trunk[1]).weight,
        model.arousal_head.weight,
        model.valence_head.weight,
        cast(nn.Linear, model.multimodal_fusion.speech_projection[1]).weight,
        cast(
            nn.Linear,
            model.multimodal_fusion.physiology_projection[1],
        ).weight,
        cast(
            nn.Linear,
            model.multimodal_fusion.modality_gate[-1],
        ).weight,
        speech.input_projection.weight,
        physiology.input_projection.weight,
    ):
        _assert_finite_nonzero_gradient(parameter)
    for parameter in (
        speech.reliability_head.weight,
        physiology.reliability_head.weight,
    ):
        _assert_zero_or_none_gradient(parameter)
    for head in (
        speech.arousal_head,
        speech.valence_head,
        speech.quadrant_head,
        physiology.arousal_head,
        physiology.valence_head,
        physiology.quadrant_head,
        model.quadrant_head,
    ):
        assert head is not None
        for parameter in head.parameters():
            _assert_zero_or_none_gradient(parameter)


@pytest.mark.parametrize("modality", ["speech", "physiology"])
def test_auxiliary_gradient_is_isolated_between_modalities(
    modality: str,
) -> None:
    """Train only the explicitly enabled unimodal binary heads."""
    weights = (
        MultimodalLossWeights(fused=0.0, speech_auxiliary=1.0)
        if modality == "speech"
        else MultimodalLossWeights(fused=0.0, physiology_auxiliary=1.0)
    )
    model = _model()
    model.eval()
    batch = _batch()
    loss = _objective(weights=weights)(model(batch), batch)
    loss.total_loss.backward()
    speech = cast(
        TinySpeechClassifier,
        model.batch_scheduler.speech_classifier,
    )
    physiology = cast(
        TinyPhysioClassifier,
        model.batch_scheduler.physiology_classifier,
    )
    selected = speech if modality == "speech" else physiology
    other = physiology if modality == "speech" else speech
    _assert_finite_nonzero_gradient(selected.arousal_head.weight)
    _assert_finite_nonzero_gradient(selected.valence_head.weight)
    _assert_zero_or_none_gradient(other.arousal_head.weight)
    _assert_zero_or_none_gradient(other.valence_head.weight)


@pytest.mark.parametrize("case", ["neither", "protocol", "missing_auxiliary"])
def test_empty_supervision_returns_graph_safe_zero(case: str) -> None:
    """Return zero with backward connectivity for each empty-supervision cause."""
    if case == "neither":
        batch = _batch(((False, False), (False, False)))
        objective = _objective()
    elif case == "protocol":
        batch = cast(object, _ignored_batch())
        objective = _objective()
    else:
        batch = _batch(((False, True),))
        objective = _objective(
            weights=MultimodalLossWeights(
                fused=0.0,
                speech_auxiliary=1.0,
            )
        )
    model = _model()
    model.eval()
    output = model(batch)  # type: ignore[arg-type]
    loss = objective(output, batch)  # type: ignore[arg-type]
    assert loss.active_target_count == 0
    assert torch.equal(loss.total_loss, torch.zeros_like(loss.total_loss))
    assert bool(torch.isfinite(loss.total_loss))
    assert loss.total_loss.grad_fn is not None
    loss.total_loss.backward()


def test_all_enabled_quadrant_labels_ignored_are_empty_supervision() -> None:
    """Count no targets when every enabled independent quadrant label is ignored."""
    model = _model(
        independent_quadrant=True,
        unimodal_quadrants=True,
    )
    model.eval()
    batch = _ignored_batch()
    output = model(batch)  # type: ignore[arg-type]
    logits_before = (
        output.quadrant_logits.detach().clone()
        if output.quadrant_logits is not None
        else None
    )
    loss = _objective(
        weights=MultimodalLossWeights(
            fused=0.0,
            fused_quadrant=1.0,
            speech_quadrant=1.0,
            physiology_quadrant=1.0,
        )
    )(output, batch)  # type: ignore[arg-type]
    assert loss.active_target_count == 0
    assert torch.equal(loss.total_loss, torch.zeros_like(loss.total_loss))
    assert loss.total_loss.grad_fn is not None
    loss.total_loss.backward()
    assert output.quadrant_logits is not None
    assert logits_before is not None
    assert torch.equal(output.quadrant_logits, logits_before)


def test_objective_rejects_wrong_types_batch_identity_and_sample_valid() -> None:
    """Validate public types and exact batch/mask relationships."""
    objective = _objective()
    model = _model()
    model.eval()
    batch = _batch()
    output = model(batch)
    with pytest.raises(TypeError, match="model_output"):
        objective(cast(object, object()), batch)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="batch"):
        objective(output, cast(object, object()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact objective batch"):
        objective(output, _batch())
    wrong_valid = output.sample_valid.clone()
    wrong_valid[0] = False
    malformed = _unsafe_copy(output, sample_valid=wrong_valid)
    with pytest.raises(ValueError, match="availability"):
        objective(malformed, batch)


def test_objective_rejects_malformed_scheduled_type_without_attribute_error() -> None:
    """Type-check retained scheduler output before using compact attributes."""
    objective = _objective()
    model = _model()
    model.eval()
    batch = _batch()
    output = model(batch)
    malformed_fusion = _unsafe_copy(
        output.fusion_output,
        scheduled_outputs=object(),
    )
    malformed_output = _unsafe_copy(output, fusion_output=malformed_fusion)
    with pytest.raises(TypeError, match="ScheduledModalityOutputs"):
        objective(malformed_output, batch)


def test_output_and_inputs_are_frozen_or_unchanged() -> None:
    """Keep labels, model diagnostics, class weights, and config untouched."""
    model = _model()
    model.eval()
    batch = _batch()
    output = model(batch)
    weights_tensor = torch.tensor([2.0, 3.0])
    class_weights = EmotionTaskClassWeights(arousal=weights_tensor)
    objective = _objective()
    labels_before = tuple(
        value.clone()
        for value in (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        )
    )
    logits_before = output.arousal_logits.detach().clone()
    weights_before = weights_tensor.clone()
    loss = objective(output, batch, class_weights=class_weights)
    with pytest.raises(FrozenInstanceError):
        loss.active_target_count = 0  # type: ignore[misc]
    assert torch.equal(output.arousal_logits, logits_before)
    assert torch.equal(weights_tensor, weights_before)
    assert objective.config == MultimodalObjectiveConfig()
    for actual, snapshot in zip(
        (
            batch.arousal_labels,
            batch.valence_labels,
            batch.quadrant_labels,
        ),
        labels_before,
    ):
        assert torch.equal(actual, snapshot)


def test_scope_contains_no_model_call_backward_or_new_loss_formula() -> None:
    """Keep objective composition within the authorized training boundary."""
    source = Path(objective_module.__file__).read_text(encoding="utf-8")
    forward_source = inspect.getsource(MultimodalTrainingObjective.forward)
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert ".step(" not in source
    assert "raw_arousal" not in forward_source
    assert "raw_valence" not in forward_source
    assert "participant" not in forward_source
    assert "log_softmax" not in source
    assert "cross_entropy(" not in source.replace(
        "class_weighted_cross_entropy(",
        "",
    )
    for prohibited in (
        "clean_speech",
        "reconstruct_waveform",
        "denoise_output",
        "noise_subtraction",
        "speech_enhancement",
    ):
        assert prohibited not in source
