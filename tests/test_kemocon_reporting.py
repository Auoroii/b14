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
    AlignedMultimodalSample,
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
from emotion_model.evaluation import (
    compute_classification_metrics,
    compute_emotion_task_metrics,
)
from emotion_model.experiments import (
    append_jsonl_artifact,
    build_cross_fold_evaluation_summary,
    build_environment_summary,
    build_runtime_availability_audit,
    build_split_summary,
    classification_metrics_summary,
    emotion_metrics_summary,
    write_effective_config_snapshot,
    write_json_artifact,
)


def _runtime_sample(
    record: MultimodalWindowRecord,
    *,
    physiology_available: bool,
) -> AlignedMultimodalSample:
    """Build one fully validated synthetic sample for availability audits."""

    channel_names = ("bvp", "eda", "temperature")
    valid = torch.zeros((2, 3), dtype=torch.bool)
    if physiology_available:
        valid[:, 0] = True
    quality = torch.zeros((3, 6), dtype=torch.float32)
    return AlignedMultimodalSample(
        record=record,
        raw_arousal=record.emotion_scores.arousal,
        raw_valence=record.emotion_scores.valence,
        arousal_label=0,
        valence_label=1,
        quadrant_label=2,
        label_ignore_index=-100,
        speech_waveform=torch.ones(4),
        speech_attention_mask=torch.ones(4, dtype=torch.bool),
        speech_sample_rate_hz=16_000,
        speech_available=True,
        physio_input=valid.to(dtype=torch.float32),
        physio_valid_mask=valid,
        physio_time_mask=valid.any(dim=1),
        physio_channel_mask=valid.any(dim=0),
        physio_timestamps_seconds=torch.tensor([0.0, 1.0], dtype=torch.float64),
        physio_channel_quality=quality,
        physio_quality_features=quality.reshape(-1),
        physiology_available=physiology_available,
        channel_names=channel_names,
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


def test_runtime_availability_audit_exposes_physio_losses_by_participant() -> None:
    """Distinguish declared physiology from signal-valid runtime availability."""

    partitioned = _partitioned_manifest()
    samples = (
        _runtime_sample(partitioned.train_records[0], physiology_available=True),
        _runtime_sample(
            partitioned.validation_records[0],
            physiology_available=False,
        ),
        _runtime_sample(partitioned.test_records[0], physiology_available=False),
    )

    audit = build_runtime_availability_audit(samples)

    assert audit["record_count"] == 3
    assert audit["mismatch_count"] == 1
    assert audit["mismatch_sample_ids"] == ["P3-sample"]
    assert audit["modality_counts"]["physiology"] == {
        "declared_count": 2,
        "runtime_available_count": 1,
        "declared_but_runtime_unavailable_count": 1,
        "runtime_available_without_declaration_count": 0,
    }
    assert audit["declared_to_runtime_pattern_counts"]["both"]["speech_only"] == 1
    assert audit["participant_counts"]["P3"]["physiology_mismatch_count"] == 1
    assert audit["mismatch_label_distribution"]["arousal"]["low"] == 1
    assert audit["runtime_modality_label_distributions"]["speech_only"][
        "record_count"
    ] == 2
    assert audit["physiology_channel_counts"]["bvp"] == {
        "declared_count": 2,
        "runtime_available_count": 1,
    }
    json.dumps(audit, allow_nan=False)


def _fold_evaluation(
    fold_index: int,
    participant_id: str,
    offset: float,
) -> dict[str, object]:
    """Create the metric subset required by cross-fold aggregation."""

    task = {
        "accuracy": 0.7 + offset,
        "macro_f1": 0.5 + offset,
        "balanced_accuracy": 0.6 + offset,
        "per_class": {
            "low": {
                "recall": 0.4 + offset,
                "f1": 0.3 + offset,
                "support": 3,
            },
            "high": {
                "recall": 0.8 + offset,
                "f1": 0.7 + offset,
                "support": 7,
            },
        },
    }
    return {
        "fold_index": fold_index,
        "record_count": 10,
        "participant_ids": [participant_id],
        "overall": {"arousal": task, "valence": task},
    }


def test_cross_fold_summary_reports_mean_std_and_unique_test_people() -> None:
    """Aggregate repeated folds without silently double-counting participants."""

    summary = build_cross_fold_evaluation_summary(
        (_fold_evaluation(0, "P1", 0.0), _fold_evaluation(1, "P2", 0.2))
    )

    assert summary["fold_count"] == 2
    assert summary["test_participant_ids"] == ["P1", "P2"]
    metric = summary["metrics"]["overall.valence.per_class.low.recall"]
    assert metric["mean"] == pytest.approx(0.5)
    assert metric["std"] == pytest.approx(0.1)
    assert metric["fold_values"] == pytest.approx([0.4, 0.6])
    json.dumps(summary, allow_nan=False)


def test_cross_fold_summary_rejects_repeated_test_participant() -> None:
    """Fail aggregation when dyad folds do not provide disjoint test people."""

    with pytest.raises(ValueError, match="multiple folds"):
        build_cross_fold_evaluation_summary(
            (_fold_evaluation(0, "P1", 0.0), _fold_evaluation(1, "P1", 0.1))
        )


def test_classification_summary_preserves_confusion_matrix() -> None:
    """Serialize existing fields and named binary per-class metrics."""

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
    assert summary["per_class"] == {
        "low": {
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2 / 3),
            "support": 2,
        },
        "high": {
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
            "f1": 0.8,
            "support": 2,
        },
    }
    json.dumps(summary, allow_nan=False)


def test_emotion_summary_exposes_binary_low_high_recall_paths() -> None:
    """Expose the four required arousal/valence JSON recall paths."""
    metrics = compute_emotion_task_metrics(
        arousal_targets=torch.tensor([0, 0, 1, 1]),
        arousal_predictions=torch.tensor([0, 1, 1, 1]),
        valence_targets=torch.tensor([0, 1, 1, 0]),
        valence_predictions=torch.tensor([0, 0, 1, 0]),
        quadrant_targets=torch.tensor([0, 2, 3, 1]),
        quadrant_predictions=torch.tensor([0, 2, 1, 1]),
        ignore_index=-100,
    )

    overall = emotion_metrics_summary(metrics)

    assert overall["arousal"]["per_class"]["low"]["recall"] == 0.5
    assert overall["arousal"]["per_class"]["high"]["recall"] == 1.0
    assert overall["valence"]["per_class"]["low"]["recall"] == 1.0
    assert overall["valence"]["per_class"]["high"]["recall"] == 0.5
    json.dumps({"overall": overall}, allow_nan=False)


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
        "cross_fold_summary.json",
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
    assert "--all-folds" in source
    assert "runtime_modality_availability_audit" in source

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


def test_validation_entrypoint_imports_torch_for_source_statistics() -> None:
    """Prevent the validation source-presence summary from raising NameError."""
    source = Path("scripts/validate_kemocon.py").read_text(encoding="utf-8")
    assert "import torch" in source
    assert source.index("import torch") < source.index("torch.tensor(")
