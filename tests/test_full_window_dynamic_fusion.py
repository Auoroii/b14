"""Offline contracts for shared full-window K-EmoCon dynamic fusion."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel

from emotion_model.data import AlignedMultimodalBatch, kemocon_channel_specs
from emotion_model.experiments import (
    KEmoConModalityMode,
    build_kemocon_model,
    build_kemocon_optimizer,
    load_kemocon_experiment_config,
)
from emotion_model.multimodal import (
    FullWindowDynamicMultimodalFusion,
    MultimodalEmotionClassifier,
    ScheduledModalityOutputs,
)
from emotion_model.physiology import LightweightPhysioEmotionClassifier
from emotion_model.speech import LightweightNoiseConditionedSpeechClassifier
import scripts.evaluate_kemocon as evaluate_script
import scripts.train_kemocon as train_script
from tests.test_multimodal_routing import _batch, _scheduler

_MODEL_VARIANT = "lightweight_shared_dynamic_relation_differential_full_window"


def _tiny_wavlm() -> WavLMModel:
    """Return one local random 12-layer WavLM for CPU builder tests."""

    return WavLMModel(
        WavLMConfig(
            hidden_size=4,
            num_hidden_layers=12,
            num_attention_heads=2,
            intermediate_size=8,
            conv_dim=(4, 4),
            conv_kernel=(3, 3),
            conv_stride=(2, 2),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
        )
    )


def _scheduled(
    patterns: tuple[tuple[bool, bool], ...],
) -> ScheduledModalityOutputs:
    """Return deterministic scheduled embeddings for availability rows ``[B]``."""

    scheduler = _scheduler()
    scheduler.eval()
    return scheduler(_batch(patterns))


def _build_configured_model(config_path: str) -> MultimodalEmotionClassifier:
    """Build one configured CPU model using only a local tiny WavLM."""

    config = load_kemocon_experiment_config(Path(config_path))
    with (
        patch(
            "emotion_model.experiments.kemocon.resolve_project_relative",
            return_value=Path("."),
        ),
        patch(
            "emotion_model.experiments.kemocon.WavLMModel.from_pretrained",
            return_value=_tiny_wavlm(),
        ),
    ):
        model = build_kemocon_model(
            config,
            kemocon_channel_specs(),
            project_root=Path(".").resolve(),
            device=torch.device("cpu"),
        )
    assert isinstance(model, MultimodalEmotionClassifier)
    return model


def _activity_batch(ratio: float) -> AlignedMultimodalBatch:
    """Return one both-available batch with activity diagnostics ``[1]``."""

    batch = _batch(((True, True),))
    assert batch.speech is not None
    activity_mask = torch.full_like(
        batch.speech.attention_mask,
        ratio > 0.0,
    )
    speech = replace(batch.speech, activity_mask=activity_mask)
    return replace(
        batch,
        speech=speech,
        speech_activity_ratios=torch.tensor([ratio], dtype=torch.float32),
    )


def test_dynamic_weights_follow_availability_and_remain_finite() -> None:
    """Cover both, speech-only, physiology-only, and neither rows ``[B,2]``."""

    scheduled = _scheduled(
        ((True, True), (True, False), (False, True), (False, False))
    )
    fusion = FullWindowDynamicMultimodalFusion(3, 4, 5, dropout=0.0).eval()
    output = fusion(scheduled)

    assert output.modality_weights.shape == (4, 2)
    torch.testing.assert_close(
        output.modality_weights[0].sum(),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        output.modality_weights[1],
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(
        output.modality_weights[2],
        torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(
        output.modality_weights[3],
        torch.tensor([0.0, 0.0]),
    )
    assert output.sample_valid.tolist() == [True, True, True, False]
    for value in (
        output.fused_embedding,
        output.projected_speech_embedding,
        output.projected_physiology_embedding,
        output.learned_logit_corrections,
        output.modality_weights,
    ):
        assert bool(torch.isfinite(value).all())


def test_all_silent_available_speech_still_participates_in_fusion() -> None:
    """Keep an all-zero speech waveform available in a two-modality row."""

    batch = _activity_batch(0.0)
    assert batch.speech is not None
    silent_speech = replace(
        batch.speech,
        waveform=torch.zeros_like(batch.speech.waveform),
    )
    batch = replace(batch, speech=silent_speech)
    scheduler = _scheduler()
    scheduler.eval()
    scheduled = scheduler(batch)
    fusion = FullWindowDynamicMultimodalFusion(3, 4, 5, dropout=0.0).eval()

    output = fusion(scheduled)

    assert scheduled.availability.speech_available.tolist() == [True]
    assert scheduled.availability.physiology_available.tolist() == [True]
    assert 0.0 < output.modality_weights[0, 0].item() < 1.0
    assert 0.0 < output.modality_weights[0, 1].item() < 1.0
    torch.testing.assert_close(
        output.modality_weights.sum(dim=1),
        torch.ones(1),
    )


def test_activity_ratio_is_not_a_fusion_input() -> None:
    """Changing only activity diagnostics leaves embeddings and fusion unchanged."""

    scheduler = _scheduler()
    scheduler.eval()
    low = scheduler(_activity_batch(0.0))
    high = scheduler(_activity_batch(1.0))
    assert low.speech is not None and high.speech is not None
    torch.testing.assert_close(low.speech.embedding, high.speech.embedding)
    fusion = FullWindowDynamicMultimodalFusion(3, 4, 5, dropout=0.0).eval()

    low_output = fusion(low)
    high_output = fusion(high)

    torch.testing.assert_close(low_output.modality_weights, high_output.modality_weights)
    torch.testing.assert_close(low_output.fused_embedding, high_output.fused_embedding)


def test_classifier_uses_one_shared_fused_embedding_for_both_tasks() -> None:
    """Run one shared trunk rather than task-specific fusion or classifier trunks."""

    scheduler = _scheduler()
    fusion = FullWindowDynamicMultimodalFusion(3, 4, 5, dropout=0.0)
    model = MultimodalEmotionClassifier(
        scheduler,
        fusion,
        5,
        classifier_hidden_dim=6,
        dropout=0.0,
        model_variant=_MODEL_VARIANT,
    ).eval()
    calls = 0

    def hook(_module: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.classifier_trunk.register_forward_hook(hook)
    try:
        output = model(_batch(((True, True),)))
    finally:
        handle.remove()

    assert calls == 1
    assert output.fused_embedding is output.fusion_output.fused_embedding
    assert bool(torch.isfinite(output.arousal_logits).all())
    assert bool(torch.isfinite(output.valence_logits).all())


def test_builder_selects_new_backend_without_changing_full_window_speech() -> None:
    """Build the new backend while reusing the established full-window speech path."""

    config = load_kemocon_experiment_config(
        Path("configs/kemocon_v4_2_full_window_relation_differential.yaml")
    )
    assert config.model.variant == _MODEL_VARIANT
    assert config.dataset.window_seconds == 5.0
    assert config.dataset.speech_activity.availability_policy == "source_presence"
    assert config.training.modality_mode is KEmoConModalityMode.MULTIMODAL
    assert config.training.speech_modality_dropout == pytest.approx(0.1)
    assert config.training.physiology_modality_dropout == pytest.approx(0.1)
    assert (
        config.training.sampling_policy
        == "bounded_multitask_participant_balanced"
    )
    assert config.training.arousal_low_sampling_mass == pytest.approx(0.35)
    assert config.training.valence_low_sampling_mass == pytest.approx(0.35)
    assert config.training.max_sampling_weight_ratio == pytest.approx(20.0)
    assert config.training.threshold_calibration_enabled
    assert config.loss.speech_aux_weight == pytest.approx(0.3)
    assert config.loss.physiology_aux_weight == pytest.approx(0.1)
    assert config.training.evaluate_ablation
    assert config.model.freeze_wavlm
    assert config.model.unfreeze_last_n_layers == 0
    assert config.model.emotion_layer_aggregation == "fixed_mean"
    assert config.training.wavlm_learning_rate == pytest.approx(1.0e-4)
    assert config.training.min_wavlm_learning_rate is None
    model = _build_configured_model(
        "configs/kemocon_v4_2_full_window_relation_differential.yaml"
    )

    assert isinstance(model.multimodal_fusion, FullWindowDynamicMultimodalFusion)
    speech = model.batch_scheduler.speech_classifier
    assert isinstance(speech, LightweightNoiseConditionedSpeechClassifier)
    assert speech.variant == _MODEL_VARIANT
    assert speech.full_window_activity_diagnostics_only
    assert speech.relation_denoiser is not None
    assert speech.wavlm_encoder.trainable_transformer_layer_indices == ()
    assert speech.emotion_layer_aggregation.mode == "fixed_mean"
    layer_logits = speech.emotion_layer_aggregation.emotion_layer_logits
    assert layer_logits is None
    optimizer = build_kemocon_optimizer(
        model,
        config,
    )
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "downstream"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.0e-4]
    )
    assert isinstance(
        model.batch_scheduler.physiology_classifier,
        LightweightPhysioEmotionClassifier,
    )


def test_train_and_evaluate_clis_accept_the_v4_2_config_directly() -> None:
    """Parse and load the same relative config through both public CLIs."""

    config_path = "configs/kemocon_v4_2_full_window_relation_differential.yaml"
    with (
        patch.object(sys, "argv", ["train_kemocon.py", "--config", config_path, "--fold", "99"]),
        pytest.raises(ValueError, match="fold index lies outside"),
    ):
        train_script.main()
    with (
        patch.object(
            sys,
            "argv",
            ["evaluate_kemocon.py", "--config", config_path, "--fold", "99"],
        ),
        pytest.raises(ValueError, match="fold index lies outside"),
    ):
        evaluate_script.main()


def test_config_parser_rejects_a_historical_model_variant() -> None:
    """Accept only the current V4.2 model fingerprint."""

    source = Path(
        "configs/kemocon_v4_2_full_window_relation_differential.yaml"
    ).read_text(encoding="utf-8")
    temporary_directory = Path("tmp")
    temporary_directory.mkdir(exist_ok=True)
    legacy = temporary_directory / f"legacy-{uuid.uuid4().hex}.yaml"
    try:
        legacy.write_text(
            source.replace(
                _MODEL_VARIANT,
                "unsupported_model_variant",
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="model.variant must be"):
            load_kemocon_experiment_config(legacy)
    finally:
        legacy.unlink(missing_ok=True)
