"""Offline tests for structured K-EmoCon experiment artifacts."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emotion_model.common import LabelProtocol
from emotion_model.data import (
    EmotionScores,
    MultimodalManifest,
    MultimodalWindowRecord,
    ParticipantSplit,
    PartitionedManifest,
    PhysioChannelSourceRef,
    TimedSourceRef,
    TimeInterval,
    kemocon_channel_specs,
    partition_manifest_by_participant,
)
from emotion_model.evaluation import compute_classification_metrics
from emotion_model.experiments import (
    append_jsonl_artifact,
    build_environment_summary,
    build_split_summary,
    classification_metrics_summary,
    write_effective_config_snapshot,
    write_json_artifact,
)


@pytest.fixture
def reporting_tmp_path() -> Iterator[Path]:
    """Provide a writable workspace-local directory on restricted Windows."""

    path = Path("tmp") / f"reporting-tests-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_json_artifacts_are_atomic_appendable_and_utf8(
    reporting_tmp_path: Path,
) -> None:
    """Write JSON object artifacts and ordered JSONL history records."""

    object_path = reporting_tmp_path / "summary.json"
    history_path = reporting_tmp_path / "history.jsonl"
    write_json_artifact(object_path, {"status": "完成", "epoch": 2})
    append_jsonl_artifact(history_path, {"epoch": 1, "loss": 1.2})
    append_jsonl_artifact(history_path, {"epoch": 2, "loss": 0.9})

    assert json.loads(object_path.read_text(encoding="utf-8")) == {
        "status": "完成",
        "epoch": 2,
    }
    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert history == [
        {"epoch": 1, "loss": 1.2},
        {"epoch": 2, "loss": 0.9},
    ]
    assert not (reporting_tmp_path / ".summary.json.tmp").exists()


def test_effective_config_snapshot_applies_fold_override(
    reporting_tmp_path: Path,
) -> None:
    """Preserve relative paths while recording the effective CLI fold."""

    destination = reporting_tmp_path / "run_config.yaml"
    write_effective_config_snapshot(
        Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
        destination,
        fold_index=3,
    )
    text = destination.read_text(encoding="utf-8")
    assert "fold_index: 3" in text
    assert "data_root: data/K-EmoCon" in text
    assert "wavlm_model: models/wavlm-base" in text


def test_effective_config_snapshot_records_runtime_overrides(
    reporting_tmp_path: Path,
) -> None:
    """Persist diagnostic overrides instead of a misleading source config."""

    destination = reporting_tmp_path / "run_config.yaml"
    write_effective_config_snapshot(
        Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
        destination,
        fold_index=0,
        overrides={
            "model": {"dropout": 0.0},
            "training": {
                "epochs": 200,
                "learning_rate": 0.001,
                "lr_scheduler_enabled": False,
            },
        },
    )
    text = destination.read_text(encoding="utf-8")
    assert "dropout: 0.0" in text
    assert "epochs: 200" in text
    assert "learning_rate: 0.001" in text
    assert "lr_scheduler_enabled: false" in text


def _partitioned_manifest() -> PartitionedManifest:
    records = []
    for participant, session in (("P1", "S1"), ("P2", "S2"), ("P3", "S3")):
        interval = TimeInterval(0.0, 5.0)
        physiology = (
            ()
            if participant == "P2"
            else (
                PhysioChannelSourceRef(
                    "bvp",
                    TimedSourceRef(
                        f"e4_data/{participant}/E4_BVP.csv",
                        interval,
                    ),
                ),
            )
        )
        records.append(
            MultimodalWindowRecord(
                sample_id=f"{participant}-sample",
                participant_id=participant,
                session_id=session,
                window=interval,
                emotion_scores=EmotionScores(1.0, 5.0),
                speech_source=TimedSourceRef(
                    f"pre_audio/{participant}.wav",
                    interval,
                ),
                physio_sources=physiology,
            )
        )
    manifest = MultimodalManifest(
        channel_specs=kemocon_channel_specs(),
        records=tuple(records),
    )
    split = ParticipantSplit(("P1",), ("P2",), ("P3",))
    return partition_manifest_by_participant(manifest, split)


def test_split_and_environment_summaries_are_json_compatible() -> None:
    """Record partition modality counts and finite CPU environment metadata."""

    summary = build_split_summary(
        _partitioned_manifest(),
        fold_index=0,
        split_seed=42,
        label_protocol=LabelProtocol.OFFICIAL_MID_HIGH,
    )
    partitions = summary["partitions"]
    assert isinstance(partitions, dict)
    assert partitions["train"]["modality_pattern_counts"]["both"] == 1
    assert partitions["validation"]["modality_pattern_counts"]["speech_only"] == 1
    assert partitions["test"]["record_count"] == 1
    assert summary["label_protocol"] == "official_mid_high"
    assert partitions["train"]["label_distribution"] == {
        "record_count": 1,
        "arousal": {"low": 1, "high": 0, "ignored": 0},
        "valence": {"low": 0, "high": 1, "ignored": 0},
        "quadrant": {
            "LALV": 0,
            "HALV": 0,
            "LAHV": 1,
            "HAHV": 0,
            "ignored": 0,
        },
    }
    assert partitions["train"]["participant_label_distributions"]["P1"] == (
        partitions["train"]["label_distribution"]
    )
    validation_by_modality = partitions["validation"][
        "declared_modality_label_distributions"
    ]
    assert validation_by_modality["speech_only"]["record_count"] == 1
    assert validation_by_modality["speech_only"]["quadrant"]["LAHV"] == 1
    assert validation_by_modality["both"]["record_count"] == 0
    json.dumps(summary)

    environment = build_environment_summary(torch.device("cpu"))
    assert environment["requested_device"] == "cpu"
    assert isinstance(environment["torch_version"], str)
    json.dumps(environment)


def test_split_summary_uses_the_named_strict_label_protocol() -> None:
    """Count a middle score as ignored instead of silently fixing its class."""

    original = _partitioned_manifest()
    middle_record = replace(
        original.train_records[0],
        emotion_scores=EmotionScores(3.0, 3.0),
    )
    partitioned = replace(original, train_records=(middle_record,))

    summary = build_split_summary(
        partitioned,
        fold_index=0,
        split_seed=42,
        label_protocol=LabelProtocol.STRICT_DROP_MID,
    )

    partitions = summary["partitions"]
    assert isinstance(partitions, dict)
    train = partitions["train"]
    assert isinstance(train, dict)
    assert train["label_distribution"]["arousal"] == {
        "low": 0,
        "high": 0,
        "ignored": 1,
    }
    assert train["label_distribution"]["valence"] == {
        "low": 0,
        "high": 0,
        "ignored": 1,
    }
    assert train["label_distribution"]["quadrant"]["ignored"] == 1


def test_classification_summary_preserves_confusion_matrix() -> None:
    """Serialize scalar metrics and CPU confusion matrix ``[2,2]``."""

    metrics = compute_classification_metrics(
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        torch.tensor([0, 1, 1, 1], dtype=torch.long),
        num_classes=2,
        ignore_index=-100,
    )
    summary = classification_metrics_summary(metrics)
    assert summary["evaluated_count"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["confusion_matrix"] == [[1, 1], [0, 2]]
    assert summary["class_support"] == [2, 2]


def test_training_entrypoint_uses_best_only_and_records_all_artifacts() -> None:
    """Keep progress and complete records without creating ``last.pt``."""

    source = Path("scripts/train_kemocon.py").read_text(encoding="utf-8")
    assert source.index("sys.path.insert(0, str(_LOCAL_SOURCE))") < source.index(
        "from emotion_model.data import"
    )
    assert "last.pt" not in source
    assert "from tqdm.auto import tqdm" in source
    for artifact in (
        "run_config.yaml",
        "environment.json",
        "split_summary.json",
        "normalizer.json",
        "train.log",
        "history.jsonl",
        "best.pt",
        "training_summary.json",
        "test_metrics.json",
        "test_ablation_metrics.json",
    ):
        assert artifact in source
    assert "mean_macro_f1" in source
    assert "participant_mean_macro_f1" in source
    assert "build_kemocon_lr_scheduler" in source
    assert "best_emotion_layer_weights" in source
    assert "trainable_wavlm_parameter_count" in source
    assert "_apply_training_physiology_dropout" in source
    assert "_apply_training_symmetric_modality_dropout" in source
    assert "WeightedRandomSampler" in source
    assert "validation_fusion_diagnostics" in source
    assert "binary_decision_thresholds" in source
    assert "full_window_speech_statistics" in source
    assert "speech_statistics=training_speech_statistics" in source

    evaluation_source = Path("scripts/evaluate_kemocon.py").read_text(
        encoding="utf-8"
    )
    assert "full_window_speech_statistics" in evaluation_source
    assert "_track_speech_availability" in evaluation_source


def test_all_kemocon_entrypoints_prefer_their_local_source_tree() -> None:
    """Prevent a server-side installed package from shadowing copied code."""

    for relative_path in (
        "scripts/prepare_kemocon.py",
        "scripts/train_kemocon.py",
        "scripts/validate_kemocon.py",
        "scripts/evaluate_kemocon.py",
    ):
        source = Path(relative_path).read_text(encoding="utf-8")
        bootstrap = source.index("sys.path.insert(0, str(_LOCAL_SOURCE))")
        first_project_import = source.index("from emotion_model.")
        assert bootstrap < first_project_import
