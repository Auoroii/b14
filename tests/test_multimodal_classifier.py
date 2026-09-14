"""Shared helpers and V4.2 multimodal classifier contracts."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch
from torch import Tensor, nn

from emotion_model.multimodal import (
    FullWindowDynamicMultimodalFusion,
    MultimodalEmotionClassifier,
)
from tests.test_multimodal_routing import _batch, _scheduler

_MODEL_VARIANT = "lightweight_shared_dynamic_relation_differential_full_window"
_MIXED = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)


def _model(
    *,
    fusion_dim: int = 5,
    classifier_hidden_dim: int | None = None,
    dropout: float = 0.0,
    independent_quadrant: bool = False,
    unimodal_quadrants: bool = False,
    dtype: torch.dtype = torch.float32,
) -> MultimodalEmotionClassifier:
    """Create a fully registered tiny V4.2 model without external resources."""

    scheduler = _scheduler(
        speech_quadrant=unimodal_quadrants,
        physiology_quadrant=unimodal_quadrants,
    )
    fusion = FullWindowDynamicMultimodalFusion(
        3,
        4,
        fusion_dim,
        dropout=dropout,
    )
    model = MultimodalEmotionClassifier(
        scheduler,
        fusion,
        fusion_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=dropout,
        enable_independent_quadrant_head=independent_quadrant,
        model_variant=_MODEL_VARIANT,
    )
    return model.to(dtype=dtype)


def _unsafe_copy(instance: Any, **updates: object) -> Any:
    """Copy a frozen dataclass without validation for negative contract tests."""

    result = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(
            result,
            field.name,
            updates.get(field.name, getattr(instance, field.name)),
        )
    return result


def _assert_finite_nonzero_gradient(parameter: Tensor) -> None:
    """Require a mathematically active parameter to receive finite signal."""

    assert parameter.grad is not None
    assert bool(torch.isfinite(parameter.grad).all())
    assert bool(torch.count_nonzero(parameter.grad))


def _assert_zero_or_none_gradient(parameter: Tensor) -> None:
    """Require an inactive parameter to receive no effective signal."""

    if parameter.grad is not None:
        assert bool(torch.isfinite(parameter.grad).all())
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))


def test_constructor_registers_only_shared_v4_2_fusion_path() -> None:
    """Register one shared trunk and the strict V4.2 fingerprint."""

    model = _model()
    assert isinstance(model.multimodal_fusion, FullWindowDynamicMultimodalFusion)
    assert model.model_variant == _MODEL_VARIANT
    assert isinstance(model.classifier_trunk[0], nn.LayerNorm)
    assert model.get_extra_state()["model_variant"] == _MODEL_VARIANT


def test_mixed_availability_forward_is_finite_and_masked() -> None:
    """Retain every logical row and obey exact missing-modality weights."""

    output = _model()(_batch(_MIXED))
    weights = output.fusion_output.modality_weights
    assert weights.shape == (4, 2)
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))
    assert torch.equal(weights[1], torch.tensor([1.0, 0.0]))
    assert torch.equal(weights[2], torch.tensor([0.0, 1.0]))
    assert torch.equal(weights[3], torch.zeros(2))
    assert bool(torch.isfinite(output.arousal_logits).all())
    assert bool(torch.isfinite(output.valence_logits).all())
    assert torch.equal(output.sample_valid, torch.tensor([True, True, True, False]))


def test_classifier_rejects_non_v4_2_fingerprint() -> None:
    """Reject a model checkpoint namespace outside the only supported variant."""

    fusion = FullWindowDynamicMultimodalFusion(3, 4, 5, dropout=0.0)
    try:
        MultimodalEmotionClassifier(
            _scheduler(),
            fusion,
            5,
            model_variant="unsupported_historical_variant",
        )
    except ValueError as error:
        assert "model_variant must be" in str(error)
    else:
        raise AssertionError("unsupported model variant was accepted")
