"""Contracts for the P1 multi-scale dilated physiology encoder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emotion_model.data import kemocon_channel_specs
from emotion_model.experiments import load_kemocon_experiment_config
from emotion_model.multimodal import (
    FullWindowDynamicMultimodalFusion,
    MultimodalBatchScheduler,
)
from emotion_model.physiology import (
    LightweightPhysioClassifierOutput,
    LightweightPhysioEmotionClassifier,
    MultiScaleDilatedConv1dStem,
)
from tests.test_multimodal_routing import TinySpeechClassifier, _batch

_S1_CONFIG = Path("configs/kemocon_v4_2_full_window_relation_differential.yaml")
_P1_CONFIG = Path("configs/kemocon_v4_2_p1_multiscale_dilated_physio.yaml")


def _classifier(
    encoder: str = "multiscale_dilated",
) -> LightweightPhysioEmotionClassifier:
    """Build one CPU classifier consuming ``[B,T,3]`` synthetic tensors."""

    return LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(),
        stem_channels=(8, 16),
        physiology_embedding_dim=12,
        dropout=0.0,
        stem_dropout=0.0,
        physiology_encoder=encoder,
        physiology_dilations=(1, 2, 4),
    )


def _forward(
    classifier: LightweightPhysioEmotionClassifier,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> LightweightPhysioClassifierOutput:
    """Run a classifier with matching ``[B,T,3]`` mask metadata."""

    return classifier(
        values,
        valid_mask,
        channel_names=("bvp", "eda", "temperature"),
        physio_time_mask=valid_mask.any(dim=2),
        physio_channel_mask=valid_mask.any(dim=1),
    )


def test_multiscale_stem_preserves_shape_and_branch_lengths() -> None:
    """Keep all dilation branches and the fused output at length ``T``."""

    stem = MultiScaleDilatedConv1dStem(8, 16, (1, 2, 4), dropout=0.0)
    values = torch.randn(2, 1, 19)
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    branch_shapes: list[tuple[int, ...]] = []
    handles = [
        branch.register_forward_hook(
            lambda _module, _inputs, output: branch_shapes.append(
                tuple(output.shape)
            )
        )
        for branch in stem.branch_convolutions
    ]
    try:
        output = stem(values, valid_mask)
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (2, 16, 19)
    assert branch_shapes == [(2, 16, 19)] * 3
    assert tuple(branch.dilation[0] for branch in stem.branch_convolutions) == (
        1,
        2,
        4,
    )
    assert tuple(branch.padding[0] for branch in stem.branch_convolutions) == (
        1,
        2,
        4,
    )


def test_multiscale_stem_masks_partial_and_all_padding_safely() -> None:
    """Return finite zeros at invalid positions, including an all-padding row."""

    stem = MultiScaleDilatedConv1dStem(4, 6, (1, 2, 4), dropout=0.0)
    values = torch.randn(2, 1, 11)
    valid_mask = torch.tensor(
        [
            [[True, True, True, True, False, False, False, False, False, False, False]],
            [[False] * 11],
        ]
    )

    output = stem(values, valid_mask)

    assert bool(torch.isfinite(output).all())
    expanded_mask = valid_mask.expand(-1, output.shape[1], -1)
    assert torch.equal(
        output.masked_select(~expanded_mask),
        torch.zeros_like(output.masked_select(~expanded_mask)),
    )
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_multiscale_padding_values_do_not_change_features_or_embedding() -> None:
    """Ignore arbitrary invalid input values throughout convolution and pooling."""

    torch.manual_seed(33)
    classifier = _classifier().eval()
    baseline = torch.randn(2, 13, 3)
    valid_mask = torch.tensor(
        [
            [[True, True, True]] * 8 + [[False, False, False]] * 5,
            [[True, True, False]] * 6 + [[False, False, False]] * 7,
        ]
    )
    changed = baseline.clone()
    changed[~valid_mask] = torch.linspace(
        -1.0e6,
        1.0e6,
        int((~valid_mask).sum()),
    )

    baseline_output = _forward(classifier, baseline, valid_mask)
    changed_output = _forward(classifier, changed, valid_mask)

    torch.testing.assert_close(
        changed_output.channel_features,
        baseline_output.channel_features,
    )
    torch.testing.assert_close(
        changed_output.channel_statistics,
        baseline_output.channel_statistics,
    )
    torch.testing.assert_close(
        changed_output.physio_embedding,
        baseline_output.physio_embedding,
    )


def test_multiscale_branches_and_fusion_receive_finite_gradients() -> None:
    """Backpropagate through all three dilation branches and 1x1 fusion."""

    torch.manual_seed(41)
    stem = MultiScaleDilatedConv1dStem(5, 7, (1, 2, 4), dropout=0.0)
    values = torch.randn(2, 1, 17, requires_grad=True)
    valid_mask = torch.tensor(
        [
            [[True] * 13 + [False] * 4],
            [[True] * 9 + [False] * 8],
        ]
    )

    stem(values, valid_mask).square().sum().backward()

    modules = (*stem.branch_convolutions, stem.fusion_convolution)
    for module in modules:
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            assert bool(torch.count_nonzero(parameter.grad))
    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all())


def test_classifier_handles_all_padding_and_missing_channel() -> None:
    """Keep sample/channel availability and downstream statistics mask-safe."""

    classifier = _classifier().eval()
    values = torch.randn(2, 12, 3)
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    valid_mask[0] = False
    valid_mask[1, :, 2] = False

    output = _forward(classifier, values, valid_mask)

    assert output.channel_features.shape == (2, 12, 3, 16)
    assert output.channel_statistics.shape == (2, 3, 32)
    assert output.sample_valid.tolist() == [False, True]
    assert output.channel_available.tolist() == [
        [False, False, False],
        [True, True, False],
    ]
    assert torch.equal(
        output.channel_features[0],
        torch.zeros_like(output.channel_features[0]),
    )
    assert torch.equal(
        output.channel_statistics[0],
        torch.zeros_like(output.channel_statistics[0]),
    )
    assert torch.equal(
        output.physio_embedding[0],
        torch.zeros_like(output.physio_embedding[0]),
    )
    assert torch.equal(
        output.channel_features[1, :, 2],
        torch.zeros_like(output.channel_features[1, :, 2]),
    )
    assert torch.equal(
        output.channel_statistics[1, 2],
        torch.zeros_like(output.channel_statistics[1, 2]),
    )
    for value in (
        output.channel_features,
        output.channel_statistics,
        output.physio_embedding,
        output.arousal_logits,
        output.valence_logits,
    ):
        assert bool(torch.isfinite(value).all())


def test_missing_physiology_keeps_speech_only_fusion_weight() -> None:
    """Route a missing P1 physiology modality with exact weights ``[1,0]``."""

    scheduler = MultimodalBatchScheduler(TinySpeechClassifier(), _classifier())
    scheduler.eval()
    scheduled = scheduler(_batch(((True, False),)))
    fusion = FullWindowDynamicMultimodalFusion(3, 12, 5, dropout=0.0).eval()

    output = fusion(scheduled)

    torch.testing.assert_close(
        output.modality_weights,
        torch.tensor([[1.0, 0.0]]),
    )
    assert output.sample_valid.tolist() == [True]
    assert bool(torch.isfinite(output.fused_embedding).all())


def test_single_scale_default_is_an_exact_behavioral_regression() -> None:
    """Keep default and explicit S1 single-scale stems bitwise equivalent."""

    torch.manual_seed(52)
    baseline = LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(),
        stem_channels=(8, 16),
        physiology_embedding_dim=12,
        dropout=0.0,
        stem_dropout=0.0,
    ).eval()
    torch.manual_seed(52)
    explicit = _classifier("single_scale").eval()
    values = torch.randn(2, 15, 3)
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    valid_mask[:, 11:] = False

    baseline_output = _forward(baseline, values, valid_mask)
    explicit_output = _forward(explicit, values, valid_mask)

    for name, value in baseline.state_dict().items():
        other = explicit.state_dict()[name]
        if isinstance(value, torch.Tensor):
            assert isinstance(other, torch.Tensor)
            assert torch.equal(value, other)
    torch.testing.assert_close(
        explicit_output.channel_features,
        baseline_output.channel_features,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        explicit_output.physio_embedding,
        baseline_output.physio_embedding,
        rtol=0.0,
        atol=0.0,
    )


def test_physiology_fingerprint_rejects_encoder_and_dilation_mismatch() -> None:
    """Reject checkpoints across single/multi-scale and dilation contracts."""

    single_scale = _classifier("single_scale")
    multiscale = _classifier("multiscale_dilated")
    other_dilations = LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(),
        physiology_encoder="multiscale_dilated",
        physiology_dilations=(1, 3, 5),
    )

    with pytest.raises(RuntimeError, match="configuration does not match"):
        multiscale.set_extra_state(single_scale.get_extra_state())
    with pytest.raises(RuntimeError, match="configuration does not match"):
        other_dilations.set_extra_state(multiscale.get_extra_state())
    with pytest.raises(RuntimeError, match="configuration does not match"):
        other_dilations.load_state_dict(multiscale.state_dict())


def test_p1_parameter_growth_is_lightweight_and_exact() -> None:
    """Report the production-width S1/P1 physiology parameter delta."""

    s1 = LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(),
        physiology_encoder="single_scale",
    )
    p1 = LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(),
        physiology_encoder="multiscale_dilated",
        physiology_dilations=(1, 2, 4),
    )
    s1_count = sum(parameter.numel() for parameter in s1.parameters())
    p1_count = sum(parameter.numel() for parameter in p1.parameters())

    assert s1_count == 8_970
    assert p1_count == 12_906
    assert p1_count - s1_count == 3_936
    assert (p1_count - s1_count) / s1_count == pytest.approx(0.4387959866)
    assert len({id(stem) for stem in p1.channel_stems}) == 3


@pytest.mark.parametrize(
    "dilations",
    [(), (1, 2), (1, 2, 4, 8), (1, 0, 4), (1, 2, 2)],
)
def test_multiscale_rejects_invalid_dilation_contract(
    dilations: tuple[int, ...],
) -> None:
    """Require exactly three distinct positive dilation values."""

    with pytest.raises((TypeError, ValueError)):
        MultiScaleDilatedConv1dStem(4, 6, dilations)


def test_s1_and_p1_configs_differ_only_by_encoder_and_output_path() -> None:
    """Keep every model/training variable except the P1 encoder identical."""

    s1 = load_kemocon_experiment_config(_S1_CONFIG)
    p1 = load_kemocon_experiment_config(_P1_CONFIG)

    assert s1.model.speech_pooling == "attentive_stats"
    assert p1.model.speech_pooling == "attentive_stats"
    assert s1.model.physiology_encoder == "single_scale"
    assert p1.model.physiology_encoder == "multiscale_dilated"
    assert s1.model.physiology_dilations == p1.model.physiology_dilations == (
        1,
        2,
        4,
    )
    normalized_p1 = replace(
        p1,
        paths=s1.paths,
        model=replace(p1.model, physiology_encoder="single_scale"),
    )
    assert normalized_p1 == s1
