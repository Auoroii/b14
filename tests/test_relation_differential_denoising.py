"""Contracts for the V4.2 relation-differential speech refinement."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel

from emotion_model.common import masked_mean_std
from emotion_model.speech import (
    LightweightNoiseConditionedSpeechClassifier,
    NoiseConditionedRelationDifferentialDenoiser,
    RelationDifferentialDenoisingOutput,
    WavLMEncoder,
    WavLMEncoderOutput,
)


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """Provide a disposable directory under the writable workspace."""

    path = Path("tmp") / f"relation-differential-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _tiny_wavlm(*, hidden_size: int = 24) -> WavLMModel:
    return WavLMModel(
        WavLMConfig(
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=4,
            intermediate_size=48,
            conv_dim=(8, 8, 8),
            conv_stride=(5, 2, 2),
            conv_kernel=(10, 3, 3),
            num_conv_pos_embedding_groups=4,
        )
    )


def _denoiser() -> NoiseConditionedRelationDifferentialDenoiser:
    return NoiseConditionedRelationDifferentialDenoiser(
        12,
        differential_dim=8,
        num_heads=2,
        lambda_init=0.5,
        residual_scale=0.1,
        condition_gate_on_noise=True,
    )


def test_module_shapes_bounds_initialization_and_finite_outputs() -> None:
    """Expose bounded scalar/gate diagnostics at every documented shape."""

    module = _denoiser()
    emotion = torch.randn(3, 5, 12)
    noise = torch.randn(3, 5, 12)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, False, True, True],
            [True, False, False, False, False],
        ]
    )
    output = module(emotion, noise, mask)
    assert output.denoised_emotion_sequence.shape == (3, 5, 12)
    assert output.differential_response.shape == (3, 5, 8)
    assert output.differential_gate.shape == (3, 5, 8)
    assert output.delta.shape == (3, 5, 12)
    assert output.differential_lambda.shape == ()
    assert output.differential_lambda.item() == pytest.approx(0.5)
    assert bool((output.differential_gate >= 0.0).all())
    assert bool((output.differential_gate <= 1.0).all())
    assert 0.0 < output.differential_lambda.item() < 1.0
    assert all(
        bool(torch.isfinite(value).all())
        for value in (
            output.denoised_emotion_sequence,
            output.differential_response,
            output.differential_gate,
            output.delta,
            output.differential_lambda,
        )
    )
    for value in (
        output.denoised_emotion_sequence,
        output.differential_response,
        output.differential_gate,
        output.delta,
    ):
        assert torch.equal(value[~mask], torch.zeros_like(value[~mask]))


def test_reference_shift_skips_arbitrary_invalid_positions() -> None:
    """Reference each valid key from the previous valid key across holes."""

    keys = torch.tensor([[[[1.0], [99.0], [2.0], [88.0], [3.0]]]])
    mask = torch.tensor([[True, False, True, False, True]])
    shifted = NoiseConditionedRelationDifferentialDenoiser._shift_previous_valid(
        keys,
        mask,
    )
    assert torch.equal(
        shifted,
        torch.tensor([[[[1.0], [0.0], [1.0], [0.0], [2.0]]]]),
    )


def test_padding_values_including_nan_inf_do_not_change_valid_outputs() -> None:
    """Clean invalid frames before every projection and attention operation."""

    torch.manual_seed(10)
    module = _denoiser().eval()
    emotion = torch.randn(2, 5, 12)
    noise = torch.randn(2, 5, 12)
    mask = torch.tensor(
        [[True, False, True, False, True], [False, True, True, False, False]]
    )
    clean = module(emotion, noise, mask)
    corrupted_emotion = emotion.clone()
    corrupted_noise = noise.clone()
    corrupted_emotion[~mask] = float("nan")
    corrupted_noise[~mask] = float("inf")
    corrupted = module(corrupted_emotion, corrupted_noise, mask)
    for left, right in (
        (clean.denoised_emotion_sequence, corrupted.denoised_emotion_sequence),
        (clean.differential_response, corrupted.differential_response),
        (clean.differential_gate, corrupted.differential_gate),
        (clean.delta, corrupted.delta),
    ):
        assert torch.equal(left[mask], right[mask])
        assert torch.equal(right[~mask], torch.zeros_like(right[~mask]))


def test_all_padding_and_single_valid_frame_are_safe_exact_bypasses() -> None:
    """Return zero for all padding and preserve the sole valid emotion frame."""

    module = _denoiser()
    emotion = torch.randn(2, 4, 12)
    noise = torch.randn(2, 4, 12)
    emotion[0] = float("nan")
    noise[0] = float("inf")
    mask = torch.tensor(
        [[False, False, False, False], [False, False, True, False]]
    )
    output = module(emotion, noise, mask)
    assert torch.equal(
        output.denoised_emotion_sequence[0],
        torch.zeros_like(output.denoised_emotion_sequence[0]),
    )
    assert torch.equal(
        output.denoised_emotion_sequence[1, 2],
        emotion[1, 2],
    )
    assert torch.equal(output.delta, torch.zeros_like(output.delta))
    assert torch.equal(
        output.differential_response,
        torch.zeros_like(output.differential_response),
    )
    assert torch.equal(
        output.differential_gate,
        torch.zeros_like(output.differential_gate),
    )


def test_multi_frame_branch_parameters_receive_nonzero_finite_gradients() -> None:
    """Keep the differential branch trainable while WavLM may remain frozen."""

    torch.manual_seed(21)
    module = _denoiser()
    emotion = torch.randn(2, 5, 12)
    noise = torch.randn(2, 5, 12)
    mask = torch.tensor(
        [[True, True, True, False, False], [False, True, False, True, True]]
    )
    output = module(emotion, noise, mask)
    output.denoised_emotion_sequence.square().sum().backward()
    gradients = {
        name: parameter.grad
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    assert gradients
    assert all(value is not None for value in gradients.values())
    assert all(bool(torch.isfinite(value).all()) for value in gradients.values())
    assert any(bool(torch.count_nonzero(value)) for value in gradients.values())
    assert gradients["lambda_logit"] is not None
    assert bool(torch.count_nonzero(gradients["lambda_logit"]))


def test_classifier_uses_one_forward_raw_layers_denoised_pooling_and_global_film() -> None:
    """Insert refinement between raw layer means and the unchanged pooling/FiLM path."""

    encoder = WavLMEncoder(_tiny_wavlm(), freeze_wavlm=True)
    denoiser = NoiseConditionedRelationDifferentialDenoiser(
        24,
        differential_dim=8,
        num_heads=2,
    )
    classifier = LightweightNoiseConditionedSpeechClassifier(
        encoder,
        speech_embedding_dim=8,
        noise_embedding_dim=4,
        relation_denoiser=denoiser,
        dropout=0.0,
    ).eval()
    mask = torch.tensor([[True, True, False], [True, True, True]])
    layers = tuple(
        torch.full((2, 3, 24), float(index + 1)) for index in range(12)
    )
    encoded = WavLMEncoderOutput(layers, mask)
    chosen = torch.where(
        mask.unsqueeze(-1),
        torch.full((2, 3, 24), 7.0),
        torch.zeros(2, 3, 24),
    )
    relation_output = RelationDifferentialDenoisingOutput(
        denoised_emotion_sequence=chosen,
        differential_response=torch.zeros(2, 3, 8),
        differential_gate=torch.full((2, 3, 8), 0.5),
        delta=torch.zeros(2, 3, 24),
        differential_lambda=torch.tensor(0.5),
    )
    waveform = torch.ones(2, 64)
    waveform_mask = torch.ones_like(waveform, dtype=torch.bool)
    with (
        patch.object(encoder, "forward", return_value=encoded) as wavlm_forward,
        patch.object(denoiser, "forward", return_value=relation_output) as relation_forward,
    ):
        output = classifier(waveform, waveform_mask, sample_rate=16000)
    wavlm_forward.assert_called_once()
    relation_forward.assert_called_once()
    assert torch.equal(output.emotion_sequence[mask], torch.full((5, 24), 10.5))
    assert torch.equal(output.noise_sequence[mask], torch.full((5, 24), 1.5))
    assert output.denoised_emotion_sequence is chosen
    expected_statistics, _ = masked_mean_std(chosen, mask)
    expected_base = classifier.speech_projection(expected_statistics)
    assert torch.allclose(output.base_speech_embedding, expected_base)
    assert output.differential_response is relation_output.differential_response
    assert output.differential_gate is relation_output.differential_gate
    assert output.differential_lambda is relation_output.differential_lambda
    assert classifier.film is not None


def test_frozen_wavlm_classifier_backpropagates_into_relation_and_keeps_film() -> None:
    """Train refinement downstream of one frozen WavLM forward and global FiLM."""

    torch.manual_seed(91)
    encoder = WavLMEncoder(_tiny_wavlm(), freeze_wavlm=True)
    denoiser = NoiseConditionedRelationDifferentialDenoiser(
        24,
        differential_dim=8,
        num_heads=2,
    )
    classifier = LightweightNoiseConditionedSpeechClassifier(
        encoder,
        speech_embedding_dim=8,
        noise_embedding_dim=4,
        relation_denoiser=denoiser,
        dropout=0.0,
    )
    nn.init.constant_(classifier.film.bias, 0.2)
    mask = torch.tensor([[True, True, True, False]])
    hidden = tuple(torch.randn(1, 4, 24) for _ in range(12))
    with patch.object(
        encoder,
        "forward",
        return_value=WavLMEncoderOutput(hidden, mask),
    ) as wavlm_forward:
        output = classifier(
            torch.ones(1, 64),
            torch.ones(1, 64, dtype=torch.bool),
        )
    wavlm_forward.assert_called_once()
    assert output.denoised_emotion_sequence is not None
    assert bool(torch.count_nonzero(output.gamma_bounded))
    assert bool(torch.count_nonzero(output.beta_bounded))
    assert not torch.equal(output.speech_embedding, output.base_speech_embedding)
    output.speech_embedding.square().sum().backward()
    relation_gradients = tuple(
        parameter.grad
        for parameter in denoiser.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    assert relation_gradients
    assert any(bool(torch.count_nonzero(value)) for value in relation_gradients)
    assert all(bool(torch.isfinite(value).all()) for value in relation_gradients)
    assert all(parameter.grad is None for parameter in encoder.model.parameters())
