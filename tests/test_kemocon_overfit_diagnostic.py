"""Offline contracts for the K-EmoCon small-sample overfit diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from emotion_model.data import (
    AlignedMultimodalSample,
    EmotionScores,
    MultimodalWindowRecord,
    TimedSourceRef,
    TimeInterval,
)
from emotion_model.experiments import (
    KEmoConModalityMode,
    load_kemocon_experiment_config,
)
from scripts.train_kemocon import (
    _configure_overfit_diagnostic,
    _diagnostic_parameter_group,
    _parameter_update_diagnostics,
    _select_balanced_overfit_samples,
    _snapshot_trainable_parameters,
)


def _sample(
    sample_id: str,
    *,
    arousal_label: int,
    valence_label: int,
    speech_available: bool = True,
    physiology_available: bool = True,
    speech_activity_ratio: float | None = None,
) -> AlignedMultimodalSample:
    """Create one tiny CPU sample with explicit ``[L]``/``[T,C]`` tensors."""

    interval = TimeInterval(0.0, 5.0)
    record = MultimodalWindowRecord(
        sample_id=sample_id,
        participant_id=f"participant-{sample_id}",
        session_id=f"session-{sample_id}",
        window=interval,
        emotion_scores=EmotionScores(
            1.0 if arousal_label == 0 else 4.0,
            1.0 if valence_label == 0 else 4.0,
        ),
        speech_source=TimedSourceRef("speech.wav", interval),
    )
    waveform = torch.ones(4) if speech_available else None
    speech_mask = torch.ones(4, dtype=torch.bool) if speech_available else None
    physio_input = torch.ones((2, 1)) if physiology_available else None
    physio_valid = (
        torch.ones((2, 1), dtype=torch.bool)
        if physiology_available
        else None
    )
    return AlignedMultimodalSample(
        record=record,
        raw_arousal=record.emotion_scores.arousal,
        raw_valence=record.emotion_scores.valence,
        arousal_label=arousal_label,
        valence_label=valence_label,
        quadrant_label=arousal_label + 2 * valence_label,
        label_ignore_index=-100,
        speech_waveform=waveform,
        speech_attention_mask=speech_mask,
        speech_sample_rate_hz=16_000 if speech_available else None,
        speech_available=speech_available,
        physio_input=physio_input,
        physio_valid_mask=physio_valid,
        physio_time_mask=(
            torch.ones(2, dtype=torch.bool)
            if physiology_available
            else None
        ),
        physio_channel_mask=(
            torch.ones(1, dtype=torch.bool)
            if physiology_available
            else None
        ),
        physio_timestamps_seconds=(
            torch.tensor([0.0, 1.0], dtype=torch.float64)
            if physiology_available
            else None
        ),
        physio_channel_quality=(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            if physiology_available
            else None
        ),
        physio_quality_features=(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            if physiology_available
            else None
        ),
        physiology_available=physiology_available,
        channel_names=("bvp",) if physiology_available else (),
        speech_activity_mask=(
            None
            if speech_activity_ratio is None or not speech_available
            else torch.full(
                (4,),
                speech_activity_ratio > 0.0,
                dtype=torch.bool,
            )
        ),
        speech_activity_ratio=speech_activity_ratio,
    )


class _SampleDataset(Dataset[AlignedMultimodalSample]):
    """Expose fixed tiny samples through the Dataset integer-index contract."""

    def __init__(self, samples: tuple[AlignedMultimodalSample, ...]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AlignedMultimodalSample:
        return self.samples[index]


def test_overfit_selector_is_balanced_deterministic_and_runtime_both() -> None:
    """Select equal quadrants while rejecting a runtime-missing candidate."""

    samples = tuple(
        _sample(
            f"q{quadrant}-{copy}",
            arousal_label=quadrant % 2,
            valence_label=quadrant // 2,
        )
        for quadrant in range(4)
        for copy in range(3)
    ) + (
        _sample(
            "missing-physiology",
            arousal_label=0,
            valence_label=0,
            physiology_available=False,
        ),
    )
    dataset = _SampleDataset(samples)

    selected, diagnostics = _select_balanced_overfit_samples(
        dataset,
        sample_count=8,
        seed=42,
        modality_mode=KEmoConModalityMode.MULTIMODAL,
    )
    repeated, repeated_diagnostics = _select_balanced_overfit_samples(
        dataset,
        sample_count=8,
        seed=42,
        modality_mode=KEmoConModalityMode.MULTIMODAL,
    )

    assert tuple(item.record.sample_id for item in selected) == tuple(
        item.record.sample_id for item in repeated
    )
    assert diagnostics == repeated_diagnostics
    assert diagnostics["quadrant_counts"] == {
        "LALV": 2,
        "HALV": 2,
        "LAHV": 2,
        "HAHV": 2,
    }
    assert diagnostics["modality_mode"] == "multimodal"
    assert all(item.speech_available for item in selected)
    assert all(item.physiology_available for item in selected)


def test_overfit_selector_retains_source_present_zero_activity_windows() -> None:
    """Treat natural silence as available in a full-window diagnostic subset."""

    samples = tuple(
        _sample(
            f"silent-{quadrant}",
            arousal_label=quadrant % 2,
            valence_label=quadrant // 2,
            speech_activity_ratio=0.0,
        )
        for quadrant in range(4)
    )

    selected, diagnostics = _select_balanced_overfit_samples(
        _SampleDataset(samples),
        sample_count=4,
        seed=42,
        modality_mode=KEmoConModalityMode.MULTIMODAL,
    )

    assert len(selected) == 4
    assert all(sample.speech_available for sample in selected)
    assert diagnostics["speech_availability"][
        "speech_activity_ratio_eq_0"
    ] == 4
    assert diagnostics["speech_availability"][
        "speech_source_unavailable"
    ] == 0


def test_overfit_selector_rejects_an_unfillable_quadrant() -> None:
    """Fail clearly instead of silently returning an imbalanced subset."""

    dataset = _SampleDataset(
        tuple(
            _sample(
                f"sample-{index}",
                arousal_label=index % 2,
                valence_label=0,
            )
            for index in range(8)
        )
    )
    with pytest.raises(RuntimeError, match="modality_mode=multimodal"):
        _select_balanced_overfit_samples(
            dataset,
            sample_count=4,
            seed=1,
            modality_mode=KEmoConModalityMode.MULTIMODAL,
        )


def test_overfit_configuration_is_isolated_and_disables_regularization() -> None:
    """Preserve the production config while enforcing diagnostic controls."""

    source = load_kemocon_experiment_config(
        Path("configs/kemocon_v4_2_full_window_relation_differential.yaml")
    )
    diagnostic = _configure_overfit_diagnostic(
        source,
        sample_count=64,
        epochs=123,
        modality_mode=KEmoConModalityMode.MULTIMODAL,
    )

    assert source.model.dropout == 0.3
    assert source.training.epochs == 100
    assert diagnostic.paths.output_dir.name == "overfit_64"
    assert diagnostic.training.modality_mode is KEmoConModalityMode.MULTIMODAL
    assert diagnostic.model.dropout == 0.0
    assert diagnostic.loss.class_weight_power == 0.0
    assert diagnostic.training.batch_size == 16
    assert diagnostic.training.num_workers == 0
    assert diagnostic.training.epochs == 123
    assert diagnostic.training.learning_rate == pytest.approx(1.0e-3)
    assert diagnostic.training.weight_decay == 0.0
    assert not diagnostic.training.lr_scheduler_enabled
    assert diagnostic.training.speech_modality_dropout == 0.0
    assert diagnostic.training.physiology_modality_dropout == 0.0
    assert diagnostic.training.sampling_policy == "uniform"
    assert not diagnostic.training.threshold_calibration_enabled


def test_relation_denoiser_has_its_own_parameter_diagnostic_group() -> None:
    """Keep relation-differential updates separate from generic speech updates."""

    name = (
        "batch_scheduler.speech_classifier.relation_denoiser."
        "condition_projection.weight"
    )
    assert _diagnostic_parameter_group(name) == "speech_relation_differential"


def test_parameter_diagnostics_report_updates_and_last_batch_gradients() -> None:
    """Report changed ``[...]`` parameters and finite module gradient norms."""

    model = nn.Linear(2, 2)
    initial = _snapshot_trainable_parameters(model)  # type: ignore[arg-type]
    output = model(torch.tensor([[1.0, -1.0]])).sum()
    output.backward()
    with torch.no_grad():
        model.weight.add_(0.25)

    diagnostics = _parameter_update_diagnostics(  # type: ignore[arg-type]
        model,
        initial,
    )
    global_diagnostics = diagnostics["global"]
    assert isinstance(global_diagnostics, dict)
    assert global_diagnostics["tensor_count"] == 2
    assert global_diagnostics["changed_tensor_count"] == 1
    assert global_diagnostics["gradient_tensor_count"] == 2
    assert float(global_diagnostics["delta_l2"]) > 0.0
    assert float(global_diagnostics["last_batch_gradient_l2"]) > 0.0
