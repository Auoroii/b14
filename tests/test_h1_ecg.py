"""Offline CPU coverage for the H1 S1+ plus ECG experiment."""

from __future__ import annotations

import csv
import math
import shutil
import uuid
import wave
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from emotion_model.common import LabelProtocol
from emotion_model.data import (
    AlignedMultimodalDataset,
    KEmoConPhysioAdapter,
    KEmoConSpeechAdapter,
    build_kemocon_manifest,
    collate_aligned_multimodal_samples,
    kemocon_channel_specs,
)
from emotion_model.experiments import (
    build_runtime_availability_audit,
    fit_kemocon_train_normalizer,
    load_kemocon_experiment_config,
)
from emotion_model.physiology import (
    LightweightEcgEncoder,
    LightweightPhysioEmotionClassifier,
)


@pytest.fixture
def ecg_tmp_path() -> Iterator[Path]:
    """Provide a writable workspace-local test directory."""

    path = Path("tmp") / f"h1-ecg-tests-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _write_pcm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000 * 10)


def _root(path: Path) -> Path:
    root = path / "K-EmoCon"
    _write_csv(
        root / "metadata" / "subjects.csv",
        ("pid", "initTime", "startTime", "endTime"),
        ((1, 0, 100_000, 110_000), (2, 0, 100_000, 110_000)),
    )
    _write_pcm(root / "debate_audios" / "p1.p2.wav")
    for participant in ("P1", "P2"):
        _write_pcm(root / "pre_audio" / f"{participant}.wav")
        _write_csv(
            root
            / "emotion_annotations"
            / "self_annotations"
            / f"{participant}.self.csv",
            ("seconds", "arousal", "valence"),
            ((5, 1, 5), (10, 3, 2)),
        )
        dense_rows = tuple(
            (100_000.0 + 1000.0 * index, int(participant[1:]), 1.0 + index, "A00001")
            for index in range(10)
        )
        for file_name in ("E4_BVP.csv", "E4_EDA.csv", "E4_TEMP.csv"):
            _write_csv(
                root / "e4_data" / participant[1:] / file_name,
                ("timestamp", "pid", "value", "device_serial"),
                dense_rows,
            )
    _write_csv(
        root / "neurosky_polar_data" / "1" / "Polar_HR.csv",
        ("timestamp", "pid", "value", "device_serial"),
        (
            (100_000.0, 1, 60.0, "POLAR"),
            (100_000.0, 1, 62.0, "POLAR"),
            (101_500.0, 1, 64.0, "POLAR"),
            (104_900.0, 1, 66.0, "POLAR"),
        ),
    )
    return root


def _dataset(root: Path) -> tuple[AlignedMultimodalDataset, object]:
    report = build_kemocon_manifest(
        root,
        channel_names=("bvp", "eda", "temperature", "ecg"),
    )
    dataset = AlignedMultimodalDataset(
        report.manifest.records,
        report.manifest.channel_specs,
        label_protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        speech_adapter=KEmoConSpeechAdapter(root),
        physio_adapter=KEmoConPhysioAdapter(root),
        physio_target_sample_rate_hz=64.0,
        ecg_sample_rate_hz=1.0,
    )
    return dataset, report


def test_polar_hr_is_mapped_to_ecg_and_missing_file_keeps_records(ecg_tmp_path: Path) -> None:
    """Map Polar_HR.csv to ECG without filtering participants or windows."""

    dataset, report = _dataset(_root(ecg_tmp_path))
    manifest = report.manifest
    assert tuple(spec.name for spec in manifest.channel_specs) == (
        "bvp",
        "eda",
        "temperature",
        "ecg",
    )
    p1 = next(record for record in manifest.records if record.participant_id == "P1")
    p2 = next(record for record in manifest.records if record.participant_id == "P2")
    ecg_source = next(source for source in p1.physio_sources if source.channel_name == "ecg")
    assert ecg_source.source.source_id == "neurosky_polar_data/1/Polar_HR.csv"
    assert all(source.channel_name != "ecg" for source in p2.physio_sources)
    assert len(dataset) == 4


def test_ecg_window_alignment_duplicate_mean_empty_window_and_missing_file(
    ecg_tmp_path: Path,
) -> None:
    """Use the exact half-open five-second window and a separate 1 Hz grid."""

    dataset, _ = _dataset(_root(ecg_tmp_path))
    first = dataset[0]
    second = dataset[1]
    p2_first = dataset[2]
    assert first.ecg_timestamps_seconds is not None
    assert first.ecg_values is not None
    assert first.ecg_valid_mask is not None
    assert first.ecg_timestamps_seconds.tolist() == pytest.approx([0, 1, 2, 3, 4])
    assert first.ecg_valid_mask.tolist() == [True, True, False, False, True]
    assert first.ecg_values.tolist() == pytest.approx([61, 64, 0, 0, 66])
    assert first.ecg_available
    assert second.ecg_valid_mask is not None and not bool(second.ecg_valid_mask.any())
    assert not second.ecg_available
    assert p2_first.ecg_valid_mask is not None and not bool(p2_first.ecg_valid_mask.any())
    assert not p2_first.ecg_available
    assert p2_first.speech_available
    assert p2_first.physiology_available
    batch = collate_aligned_multimodal_samples((first, second, p2_first))
    assert batch.ecg_available is not None
    assert batch.ecg_available.tolist() == [True, False, False]
    assert batch.physiology is not None
    assert batch.physiology.ecg_values is not None


def test_runtime_audit_reports_ecg_window_and_participant_coverage(
    ecg_tmp_path: Path,
) -> None:
    """Report declared/runtime ECG counts without creating modality mismatches."""

    dataset, _ = _dataset(_root(ecg_tmp_path))
    audit = build_runtime_availability_audit(dataset[index] for index in range(len(dataset)))
    assert audit["ecg"] == {
        "declared_count": 2,
        "runtime_available_count": 1,
        "mismatch_count": 1,
        "available_window_count": 1,
        "missing_window_count": 3,
        "available_ratio": 0.25,
        "participants_with_ecg": ["P1"],
        "participants_without_ecg": ["P2"],
    }
    channel_counts = audit["physiology_channel_counts"]
    assert channel_counts["ecg"] == {
        "declared_count": 2,
        "runtime_available_count": 1,
    }
    assert audit["mismatch_count"] == 0


def test_ecg_normalization_is_fit_from_training_records_only(ecg_tmp_path: Path) -> None:
    """Fit ECG mean/scale only from the explicitly supplied train partition."""

    root = _root(ecg_tmp_path)
    _, report = _dataset(root)
    train_records = tuple(
        record for record in report.manifest.records if record.participant_id == "P1"
    )
    normalizer = fit_kemocon_train_normalizer(
        train_records,
        report.manifest.channel_specs,
        KEmoConPhysioAdapter(root),
    )
    state = normalizer.state_dict()
    ecg = next(item for item in state["statistics"] if item["channel_name"] == "ecg")
    assert ecg["valid_count"] == 3
    assert ecg["mean"] == pytest.approx((61.0 + 64.0 + 66.0) / 3.0)
    assert state["fit_scope"] == "train"


def test_ecg_statistics_missingness_and_gradients_are_safe() -> None:
    """Compute all five statistics and force missing embeddings to exact zero."""

    encoder = LightweightEcgEncoder(16, dropout=0.0)
    values = torch.tensor([[60.0, 62.0, 64.0, 0.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [False, False, False, False]])
    timeline = torch.ones_like(mask)
    embedding, statistics, available = encoder(values, mask, timeline_mask=timeline)
    assert statistics[0].tolist() == pytest.approx(
        [62.0, math.sqrt(8.0 / 3.0), 4.0, 6.0, 0.75]
    )
    assert torch.equal(statistics[1], torch.zeros(5))
    assert torch.equal(embedding[1], torch.zeros(16))
    assert available.tolist() == [True, False]
    embedding[0].sum().backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in encoder.parameters()
    )


def test_single_valid_ecg_point_has_safe_zero_delta_slope() -> None:
    """Keep single-point and fully missing short sequences finite."""

    encoder = LightweightEcgEncoder(16, dropout=0.0)
    _, statistics, _ = encoder(
        torch.tensor([[0.0, 72.0, 0.0]]),
        torch.tensor([[False, True, False]]),
    )
    assert statistics[0, 0].item() == pytest.approx(72.0)
    assert statistics[0, 1:4].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert statistics[0, 4].item() == pytest.approx(1.0 / 3.0)
    assert bool(torch.isfinite(statistics).all())


def _classifier(*, ecg: bool) -> LightweightPhysioEmotionClassifier:
    names = ("bvp", "eda", "temperature", "ecg") if ecg else (
        "bvp",
        "eda",
        "temperature",
    )
    return LightweightPhysioEmotionClassifier(
        kemocon_channel_specs(names),
        physiology_embedding_dim=64,
        dropout=0.0,
        stem_dropout=0.0,
        ecg_embedding_dim=16,
    )


def test_h1_classifier_shape_missingness_parameter_delta_and_fingerprint() -> None:
    """Preserve the 64-D physiology output and fingerprint the H1-only branch."""

    baseline = _classifier(ecg=False)
    h1 = _classifier(ecg=True)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    h1_count = sum(parameter.numel() for parameter in h1.parameters())
    assert h1_count - baseline_count == 1500
    assert baseline.ecg_encoder is None
    assert h1.ecg_encoder is not None
    assert "ecg_enabled" not in baseline.get_extra_state()
    assert {
        "ecg_enabled": True,
        "ecg_embedding_dim": 16,
        "ecg_encoder_type": "mask_aware_statistics_mlp",
        "ecg_sample_rate_hz": 1.0,
    }.items() <= h1.get_extra_state().items()

    dense = torch.randn(2, 5, 3)
    dense_mask = torch.ones_like(dense, dtype=torch.bool)
    output = h1(
        dense,
        dense_mask,
        channel_names=("bvp", "eda", "temperature"),
        ecg_values=torch.tensor([[60.0, 61.0, 62.0, 63.0, 64.0], [0.0] * 5]),
        ecg_valid_mask=torch.tensor([[True] * 5, [False] * 5]),
        ecg_timeline_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert output.physio_embedding.shape == (2, 64)
    assert output.ecg_embedding is not None
    assert torch.equal(output.ecg_embedding[1], torch.zeros(16))
    assert output.sample_valid.tolist() == [True, True]


def test_ecg_can_supply_physio_when_dense_channels_are_missing() -> None:
    """Use OR availability while leaving dense-channel masks unchanged."""

    model = _classifier(ecg=True)
    dense = torch.zeros(1, 5, 3)
    output = model(
        dense,
        torch.zeros_like(dense, dtype=torch.bool),
        channel_names=("bvp", "eda", "temperature"),
        ecg_values=torch.tensor([[70.0, 71.0, 0.0, 0.0, 0.0]]),
        ecg_valid_mask=torch.tensor([[True, True, False, False, False]]),
        ecg_timeline_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    assert output.sample_valid.tolist() == [True]
    assert output.channel_available.tolist() == [[False, False, False]]
    assert bool(torch.isfinite(output.physio_embedding).all())


def test_s1_and_h1_checkpoints_reject_each_other() -> None:
    """Reject both checkpoint directions through architecture state and shapes."""

    baseline = _classifier(ecg=False)
    h1 = _classifier(ecg=True)
    with pytest.raises(RuntimeError):
        h1.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(h1.state_dict(), strict=True)


def test_h1_config_is_single_variable_extension_of_s1() -> None:
    """Keep split, labels, speech, losses, sampling, and optimization identical."""

    baseline = load_kemocon_experiment_config(
        "configs/kemocon_v4_2_full_window_relation_differential.yaml"
    )
    h1 = load_kemocon_experiment_config("configs/kemocon_v4_2_h1_ecg.yaml")
    assert baseline.split == h1.split
    assert baseline.loss == h1.loss
    assert baseline.training == h1.training
    baseline_dataset = asdict(baseline.dataset)
    h1_dataset = asdict(h1.dataset)
    assert h1_dataset.pop("channel_names") == (
        "bvp",
        "eda",
        "temperature",
        "ecg",
    )
    assert baseline_dataset.pop("channel_names") == ("bvp", "eda", "temperature")
    assert baseline_dataset == h1_dataset
    assert baseline.model == h1.model
    assert baseline.paths.data_root == h1.paths.data_root
    assert baseline.paths.wavlm_model == h1.paths.wavlm_model
    assert baseline.paths.manifest != h1.paths.manifest
    assert baseline.paths.output_dir != h1.paths.output_dir
