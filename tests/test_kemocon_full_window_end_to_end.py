"""End-to-end regression tests for 80,000-sample full-window K-EmoCon."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from transformers import WavLMConfig, WavLMModel

from emotion_model.data import (
    AlignedMultimodalBatch,
    AlignedMultimodalDataset,
    EmotionScores,
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    SpeechSourceData,
    TimedSourceRef,
    TimeInterval,
    apply_modality_keep_masks,
    collate_aligned_multimodal_samples,
    kemocon_channel_specs,
)
from emotion_model.experiments import (
    FullWindowSpeechAvailabilityStatistics,
    build_kemocon_model,
    build_kemocon_objective,
    load_kemocon_experiment_config,
)
from emotion_model.multimodal import (
    FullWindowDynamicMultimodalFusion,
    MultimodalEmotionClassifier,
    MultimodalEmotionClassifierOutput,
)
from emotion_model.physiology import (
    LightweightPhysioEmotionClassifier,
    PhysioChannelSpec,
    PhysioChannelWindow,
)
from emotion_model.speech import (
    LightweightNoiseConditionedSpeechClassifier,
    LightweightSpeechClassifierOutput,
)
from scripts.train_kemocon import (
    _ModalityDropoutStats,
    _apply_training_symmetric_modality_dropout,
)

_CONFIG_PATH = Path("configs/kemocon_v4_2_full_window_relation_differential.yaml")
_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 80_000
_CASE_IDS = ("case_1", "case_2", "case_3", "case_4", "case_5")
_ACTIVE_SAMPLES = (_WINDOW_SAMPLES, 16_000, 3_200, 960, 0)


class _MemorySpeechAdapter:
    """Return immutable in-memory speech sources with waveform shape ``[80000]``."""

    def __init__(self, sources: dict[str, SpeechSourceData]) -> None:
        self.sources = sources

    def load_speech(self, source: TimedSourceRef) -> SpeechSourceData:
        """Return the configured speech waveform ``[80000]`` for one source."""

        return self.sources[source.source_id]


class _SyntheticPhysioAdapter:
    """Generate finite five-second physiology channels at native rates."""

    def load_physio(
        self,
        source: PhysioChannelSourceRef,
        spec: PhysioChannelSpec,
    ) -> PhysioChannelWindow:
        """Return values, validity, and timestamps with native shape ``[T]``."""

        del source
        length = int(round(5.0 * spec.native_sample_rate_hz))
        timestamps = (
            torch.arange(length, dtype=torch.float64)
            / spec.native_sample_rate_hz
        )
        phase = {
            "bvp": 0.0,
            "eda": 0.3,
            "temperature": 0.6,
        }[spec.name]
        values = (
            0.1 * torch.sin(2.0 * torch.pi * 0.5 * timestamps + phase)
            + phase
        ).to(dtype=torch.float32)
        return PhysioChannelWindow(
            spec,
            values,
            torch.ones(length, dtype=torch.bool),
            timestamps,
        )


@dataclass(frozen=True)
class _EndToEndContext:
    """Retain one synthetic dataset, ``[6]`` batch, and configured CPU model."""

    dataset: AlignedMultimodalDataset
    batch: AlignedMultimodalBatch
    model: MultimodalEmotionClassifier


def _speech_source(active_samples: int) -> SpeechSourceData:
    """Create a five-second waveform and source activity mask ``[80000]``."""

    waveform = torch.zeros(_WINDOW_SAMPLES, dtype=torch.float32)
    activity_mask = torch.zeros(_WINDOW_SAMPLES, dtype=torch.bool)
    if active_samples > 0:
        time = torch.arange(active_samples, dtype=torch.float32) / _SAMPLE_RATE
        waveform[:active_samples] = 0.1 * torch.sin(
            2.0 * torch.pi * 220.0 * time
        )
        activity_mask[:active_samples] = True
    return SpeechSourceData(
        waveform,
        _SAMPLE_RATE,
        0.0,
        activity_mask,
    )


def _records(
    specs: tuple[PhysioChannelSpec, ...],
) -> tuple[MultimodalWindowRecord, ...]:
    """Create five present-speech rows and one true-missing control row."""

    interval = TimeInterval(0.0, 5.0)
    physiology_sources = tuple(
        PhysioChannelSourceRef(
            spec.name,
            TimedSourceRef(f"physiology-{spec.name}", interval),
        )
        for spec in specs
    )
    records: list[MultimodalWindowRecord] = []
    for index, case_id in enumerate((*_CASE_IDS, "speech_missing")):
        records.append(
            MultimodalWindowRecord(
                sample_id=case_id,
                participant_id=f"P{index + 1}",
                session_id="synthetic-session",
                window=interval,
                emotion_scores=EmotionScores(
                    2.0 if index % 2 == 0 else 4.0,
                    4.0 if index % 2 == 0 else 2.0,
                ),
                speech_source=(
                    None
                    if case_id == "speech_missing"
                    else TimedSourceRef(f"speech-{case_id}", interval)
                ),
                physio_sources=physiology_sources,
            )
        )
    return tuple(records)


def _tiny_wavlm() -> WavLMModel:
    """Return a local 12-layer WavLM reducing ``80000`` samples to 20 frames."""

    return WavLMModel(
        WavLMConfig(
            hidden_size=4,
            num_hidden_layers=12,
            num_attention_heads=2,
            intermediate_size=8,
            conv_dim=(4, 4),
            conv_kernel=(400, 10),
            conv_stride=(400, 10),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
        )
    )


@pytest.fixture(scope="module")
def end_to_end_context() -> _EndToEndContext:
    """Build one offline dataset, compact batch, and full-window CPU model."""

    torch.manual_seed(2026)
    config = load_kemocon_experiment_config(_CONFIG_PATH)
    specs = kemocon_channel_specs(config.dataset.channel_names)
    speech_sources = {
        f"speech-{case_id}": _speech_source(active_samples)
        for case_id, active_samples in zip(_CASE_IDS, _ACTIVE_SAMPLES)
    }
    dataset = AlignedMultimodalDataset(
        _records(specs),
        specs,
        label_protocol=config.dataset.label_protocol,
        speech_adapter=_MemorySpeechAdapter(speech_sources),
        physio_adapter=_SyntheticPhysioAdapter(),
        required_speech_sample_rate_hz=config.dataset.speech_sample_rate_hz,
        physio_target_sample_rate_hz=(
            config.dataset.physio_target_sample_rate_hz
        ),
        speech_activity_config=config.dataset.speech_activity,
    )
    samples = tuple(dataset[index] for index in range(len(dataset)))
    batch = collate_aligned_multimodal_samples(samples)
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
        built = build_kemocon_model(
            config,
            specs,
            project_root=Path(".").resolve(),
            device=torch.device("cpu"),
        )
    assert isinstance(built, MultimodalEmotionClassifier)
    return _EndToEndContext(dataset, batch, built.eval())


def _run_with_branch_counts(
    model: MultimodalEmotionClassifier,
    batch: AlignedMultimodalBatch,
) -> tuple[MultimodalEmotionClassifierOutput, list[int], list[int]]:
    """Run ``[B]`` once and record speech/physiology compact batch sizes."""

    speech_rows: list[int] = []
    physiology_rows: list[int] = []

    def speech_hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
        _output: object,
    ) -> None:
        assert inputs == ()
        waveform = kwargs["waveform"]
        assert isinstance(waveform, Tensor)
        speech_rows.append(waveform.shape[0])

    def physiology_hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
        _output: object,
    ) -> None:
        assert inputs == ()
        physiology = kwargs["physio_input"]
        assert isinstance(physiology, Tensor)
        physiology_rows.append(physiology.shape[0])

    speech_handle = model.batch_scheduler.speech_classifier.register_forward_hook(
        speech_hook,
        with_kwargs=True,
    )
    physiology_handle = (
        model.batch_scheduler.physiology_classifier.register_forward_hook(
            physiology_hook,
            with_kwargs=True,
        )
    )
    try:
        output = model(batch)
    finally:
        speech_handle.remove()
        physiology_handle.remove()
    return output, speech_rows, physiology_rows


def test_five_windows_and_true_missing_control_flow_end_to_end(
    end_to_end_context: _EndToEndContext,
) -> None:
    """Verify dataset through fused loss for five ``[80000]`` speech cases."""

    context = end_to_end_context
    samples = tuple(context.dataset[index] for index in range(len(context.dataset)))
    assert [sample.speech_available for sample in samples] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert all(sample.physiology_available for sample in samples)
    assert [sample.speech_activity_ratio for sample in samples[:5]] == pytest.approx(
        [1.0, 0.2, 0.04, 0.012, 0.0]
    )

    batch = context.batch
    assert batch.speech is not None
    assert batch.physiology is not None
    assert batch.speech_activity_observed is not None
    assert batch.speech_activity_observed.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert batch.speech.waveform.shape == (5, _WINDOW_SAMPLES)
    assert batch.speech.batch_indices.tolist() == [0, 1, 2, 3, 4]
    assert batch.physiology.batch_indices.tolist() == [0, 1, 2, 3, 4, 5]
    for row, active_samples in enumerate(_ACTIVE_SAMPLES):
        assert not bool(batch.speech.waveform[row, active_samples:].any())
    assert not bool(batch.speech.activity_mask[4].any())
    assert not bool(batch.speech.waveform[4].any())

    model = context.model
    assert isinstance(
        model.batch_scheduler.speech_classifier,
        LightweightNoiseConditionedSpeechClassifier,
    )
    assert model.model_variant == (
        "lightweight_shared_dynamic_relation_differential_full_window"
    )
    assert isinstance(
        model.batch_scheduler.physiology_classifier,
        LightweightPhysioEmotionClassifier,
    )
    assert isinstance(model.multimodal_fusion, FullWindowDynamicMultimodalFusion)
    assert model.batch_scheduler.speech_classifier.relation_denoiser is not None
    output, speech_rows, physiology_rows = _run_with_branch_counts(model, batch)
    assert speech_rows == [5]
    assert physiology_rows == [6]

    scheduled = output.fusion_output.scheduled_outputs
    assert scheduled.speech is not None
    assert scheduled.speech.batch_indices.tolist() == [0, 1, 2, 3, 4]
    speech_output = scheduled.speech_compact_output
    assert isinstance(speech_output, LightweightSpeechClassifierOutput)
    assert speech_output.sample_valid.tolist() == [True] * 5
    torch.testing.assert_close(
        speech_output.speech_activity_ratio.squeeze(1),
        torch.tensor([1.0, 0.2, 0.04, 0.012, 0.0]),
    )
    assert bool(torch.isfinite(speech_output.speech_embedding).all())

    weights = output.fusion_output.modality_weights
    assert output.fusion_output.availability.both_available.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    torch.testing.assert_close(weights[:5].sum(dim=1), torch.ones(5))
    assert 0.0 < weights[4, 0].item() < 1.0
    torch.testing.assert_close(weights[5], torch.tensor([0.0, 1.0]))
    assert output.sample_valid.tolist() == [True] * 6
    assert bool(torch.isfinite(output.arousal_logits).all())
    assert bool(torch.isfinite(output.valence_logits).all())

    config = load_kemocon_experiment_config(_CONFIG_PATH)
    loss_output = build_kemocon_objective(config.loss)(output, batch)
    assert bool(torch.isfinite(loss_output.total_loss))
    assert loss_output.active_target_count == 34
    arousal_gradient, valence_gradient = torch.autograd.grad(
        loss_output.fused_loss,
        (output.arousal_logits, output.valence_logits),
    )
    assert bool((arousal_gradient[:5].abs().sum(dim=1) > 0.0).all())
    assert bool((valence_gradient[:5].abs().sum(dim=1) > 0.0).all())


def test_activity_diagnostic_and_modality_dropout_controls(
    end_to_end_context: _EndToEndContext,
) -> None:
    """Ignore ratio-only changes while allowing explicit speech dropout."""

    context = end_to_end_context
    batch = context.batch
    assert batch.speech_activity_ratios is not None
    changed_ratios = torch.linspace(
        0.91,
        0.16,
        len(batch.records),
        dtype=torch.float32,
    )
    assert not torch.equal(batch.speech_activity_ratios, changed_ratios)
    diagnostic_only_change = replace(
        batch,
        speech_activity_ratios=changed_ratios,
    )
    assert diagnostic_only_change.speech is batch.speech
    assert diagnostic_only_change.physiology is batch.physiology
    original_output = context.model(batch)
    changed_output = context.model(diagnostic_only_change)
    torch.testing.assert_close(
        original_output.fusion_output.modality_weights,
        changed_output.fusion_output.modality_weights,
    )
    torch.testing.assert_close(
        original_output.arousal_logits,
        changed_output.arousal_logits,
    )
    torch.testing.assert_close(
        original_output.valence_logits,
        changed_output.valence_logits,
    )

    dropped = apply_modality_keep_masks(
        batch,
        speech_keep_mask=torch.zeros(len(batch.records), dtype=torch.bool),
        physiology_keep_mask=torch.ones(len(batch.records), dtype=torch.bool),
    )
    dropped_output, speech_rows, physiology_rows = _run_with_branch_counts(
        context.model,
        dropped,
    )
    assert dropped.speech_available.tolist() == [False] * 6
    assert dropped.speech is None
    assert dropped.records[0].speech_source is not None
    assert speech_rows == []
    assert physiology_rows == [6]
    torch.testing.assert_close(
        dropped_output.fusion_output.modality_weights,
        torch.tensor([[0.0, 1.0]]).expand(6, -1),
    )
    assert dropped_output.sample_valid.tolist() == [True] * 6
    assert bool(torch.isfinite(dropped_output.arousal_logits).all())
    assert bool(torch.isfinite(dropped_output.valence_logits).all())


def test_training_reporting_separates_natural_unavailability_and_dropout(
    end_to_end_context: _EndToEndContext,
) -> None:
    """Report activity bins without turning them into selection inputs."""

    statistics = FullWindowSpeechAvailabilityStatistics()
    dropout_statistics = _ModalityDropoutStats()
    effective_batches = tuple(
        _apply_training_symmetric_modality_dropout(
            (end_to_end_context.batch,),
            speech_probability=1.0,
            physiology_probability=0.0,
            stats=dropout_statistics,
            speech_statistics=statistics,
        )
    )

    assert len(effective_batches) == 1
    assert effective_batches[0].speech_available.tolist() == [False] * 6
    assert dropout_statistics.speech_dropped_count == 5
    assert statistics.to_summary() == {
        "total_records": 6,
        "speech_source_present": 5,
        "speech_activity_ratio_observed": 5,
        "speech_activity_ratio_eq_0": 1,
        "speech_activity_ratio_0_to_0_1": 2,
        "speech_activity_ratio_0_1_to_0_25": 1,
        "speech_activity_ratio_ge_0_25": 1,
        "speech_source_unavailable": 1,
        "artificial_speech_modality_dropout": 5,
    }
