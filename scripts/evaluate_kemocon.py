"""Evaluate one saved K-EmoCon fold with participant-level metrics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SOURCE = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_LOCAL_SOURCE))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from emotion_model.data import (
    AlignedMultimodalBatch,
    DatasetPartition,
    collate_aligned_multimodal_samples,
    partition_manifest_by_participant,
    read_manifest_json,
)
from emotion_model.evaluation import (
    BinaryDecisionThresholds,
    ParticipantEvaluationScope,
    ParticipantIndependentEvaluationOutput,
    evaluate_participant_independent,
)
from emotion_model.experiments import (
    FullWindowSpeechAvailabilityStatistics,
    ModalityAblationMode,
    build_ablation_evaluation_summary,
    build_configured_kemocon_split,
    build_evaluation_summary,
    build_kemocon_class_weights,
    build_kemocon_dataset,
    build_kemocon_model,
    build_kemocon_objective,
    build_kemocon_optimizer,
    iter_both_modality_ablation_batches,
    load_kemocon_experiment_config,
    resolve_project_relative,
    write_json_artifact,
)
from emotion_model.physiology import ChannelwiseZScoreNormalizer
from emotion_model.training import load_multimodal_checkpoint


def _relative_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "paths must be project-relative and cannot contain '..'."
        )
    return path


def _resolve_ablation_enabled(
    *,
    configured: bool,
    forced_by_cli: bool,
) -> bool:
    """Resolve optional same-window modality ablation.

    Args:
        configured: Whether the loaded experiment configuration enables it.
        forced_by_cli: Whether ``--evaluate-ablation`` was supplied.
    Returns:
        ``True`` when three-mode same-window ablation must be evaluated.

    Raises:
        TypeError: If inputs do not have their declared scalar types.
    """

    if not isinstance(configured, bool):
        raise TypeError("configured must be bool.")
    if not isinstance(forced_by_cli, bool):
        raise TypeError("forced_by_cli must be bool.")
    return configured or forced_by_cli


def _track_speech_availability(
    batches: Iterable[AlignedMultimodalBatch],
    statistics: FullWindowSpeechAvailabilityStatistics,
) -> Iterable[AlignedMultimodalBatch]:
    """Observe reporting metadata and yield every logical batch unchanged."""

    for batch in batches:
        statistics.observe(batch)
        yield batch


def main() -> None:
    """Load the best checkpoint and write test metrics as relative artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=_relative_cli_path,
        default=Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--checkpoint", type=_relative_cli_path)
    parser.add_argument(
        "--evaluate-ablation",
        action="store_true",
        help=(
            "force full/speech-only/physiology-only evaluation of a "
            "multimodal checkpoint on identical both-valid test rows"
        ),
    )
    arguments = parser.parse_args()
    config = load_kemocon_experiment_config(
        resolve_project_relative(_PROJECT_ROOT, arguments.config)
    )
    evaluate_ablation = _resolve_ablation_enabled(
        configured=config.training.evaluate_ablation,
        forced_by_cli=arguments.evaluate_ablation,
    )
    report_full_window_speech = (
        config.dataset.speech_activity.availability_policy == "source_presence"
    )
    fold_index = (
        config.split.fold_index if arguments.fold is None else arguments.fold
    )
    if not 0 <= fold_index < config.split.run_count:
        raise ValueError("fold index lies outside configured runs.")
    if config.training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    manifest = read_manifest_json(
        resolve_project_relative(_PROJECT_ROOT, config.paths.manifest)
    )
    split = build_configured_kemocon_split(
        manifest,
        config.split,
        fold_index=fold_index,
        label_protocol=config.dataset.label_protocol,
    )
    partitioned = partition_manifest_by_participant(manifest, split)
    data_root = resolve_project_relative(_PROJECT_ROOT, config.paths.data_root)
    fold_relative = config.paths.output_dir / f"fold_{fold_index}"
    fold_directory = resolve_project_relative(_PROJECT_ROOT, fold_relative)
    normalizer_path = fold_directory / "normalizer.json"
    with normalizer_path.open("r", encoding="utf-8") as stream:
        normalizer_state = json.load(stream)
    normalizer = ChannelwiseZScoreNormalizer()
    normalizer.load_state_dict(normalizer_state)
    test_dataset = build_kemocon_dataset(
        partitioned.test_records,
        manifest.channel_specs,
        data_root=data_root,
        config=config.dataset,
        normalizer=normalizer,
        modality_mode=config.training.modality_mode,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_aligned_multimodal_samples,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        persistent_workers=config.training.num_workers > 0,
    )
    model = build_kemocon_model(
        config,
        manifest.channel_specs,
        project_root=_PROJECT_ROOT,
        device=device,
    )
    objective = build_kemocon_objective(config.loss)
    optimizer = build_kemocon_optimizer(model, config)
    class_weights = build_kemocon_class_weights(
        partitioned.train_records,
        protocol=config.dataset.label_protocol,
        device=device,
        participant_balanced=(
            config.training.participant_balanced_sampling
        ),
        class_weight_power=config.loss.class_weight_power,
    )
    checkpoint_relative = (
        arguments.checkpoint
        if arguments.checkpoint is not None
        else fold_relative / "best.pt"
    )
    loaded = load_multimodal_checkpoint(
        resolve_project_relative(_PROJECT_ROOT, checkpoint_relative),
        model=model,
        optimizer=optimizer,
        objective=objective,
        restore_rng_state=False,
    )
    metadata_modality_mode = loaded.extra_metadata.get(
        "training_modality_mode"
    )
    if (
        metadata_modality_mode is not None
        and metadata_modality_mode != config.training.modality_mode.value
    ):
        raise ValueError(
            "checkpoint training modality mode does not match the current "
            "configuration."
        )
    threshold_metadata = loaded.extra_metadata.get(
        "binary_decision_thresholds"
    )
    if threshold_metadata is None:
        if config.training.threshold_calibration_enabled:
            raise ValueError(
                "checkpoint lacks calibrated binary decision thresholds."
            )
        binary_thresholds = BinaryDecisionThresholds()
    else:
        binary_thresholds = BinaryDecisionThresholds.from_metadata(
            threshold_metadata
        )
    test_batches: Iterable[AlignedMultimodalBatch] = tqdm(
        test_loader,
        total=len(test_loader),
        desc=f"Fold {fold_index} test",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.25,
    )
    speech_statistics = (
        FullWindowSpeechAvailabilityStatistics()
        if report_full_window_speech
        else None
    )
    if speech_statistics is not None:
        test_batches = _track_speech_availability(
            test_batches,
            speech_statistics,
        )
    result = evaluate_participant_independent(
        model,
        objective,
        test_batches,
        scope=ParticipantEvaluationScope(
            split,
            DatasetPartition.TEST,
        ),
        class_weights=class_weights,
        binary_thresholds=binary_thresholds,
    )
    summary = build_evaluation_summary(
        result,
        fold_index=fold_index,
        checkpoint=checkpoint_relative,
    )
    summary["training_modality_mode"] = config.training.modality_mode.value
    summary["ablation_evaluation_enabled"] = evaluate_ablation
    summary["ablation_forced_by_cli"] = arguments.evaluate_ablation
    if speech_statistics is not None:
        summary["full_window_speech_statistics"] = (
            speech_statistics.to_summary()
        )
    metrics_path = fold_directory / "test_metrics.json"
    write_json_artifact(metrics_path, summary)
    overall = summary["overall"]
    if not isinstance(overall, dict):
        raise RuntimeError("evaluation summary overall section is invalid.")
    arousal = overall["arousal"]
    valence = overall["valence"]
    if not isinstance(arousal, dict) or not isinstance(valence, dict):
        raise RuntimeError("evaluation task summaries are invalid.")
    ablation_lines: list[str] = []
    saved_paths = [str(metrics_path.relative_to(_PROJECT_ROOT))]
    if evaluate_ablation:
        ablation_results: dict[
            str,
            ParticipantIndependentEvaluationOutput,
        ] = {}
        for mode in ModalityAblationMode:
            ablation_results[mode.value] = evaluate_participant_independent(
                model,
                objective,
                iter_both_modality_ablation_batches(
                    tqdm(
                        test_loader,
                        total=len(test_loader),
                        desc=f"Fold {fold_index} ablation {mode.value}",
                        unit="batch",
                        dynamic_ncols=True,
                        leave=False,
                        mininterval=0.25,
                    ),
                    mode,
                ),
                scope=ParticipantEvaluationScope(
                    split,
                    DatasetPartition.TEST,
                    require_all_partition_participants=False,
                ),
                class_weights=class_weights,
                binary_thresholds=binary_thresholds,
            )
        ablation_summary = build_ablation_evaluation_summary(
            ablation_results,
            fold_index=fold_index,
            checkpoint=checkpoint_relative,
        )
        ablation_summary["forced_by_cli"] = arguments.evaluate_ablation
        ablation_path = fold_directory / "test_ablation_metrics.json"
        write_json_artifact(ablation_path, ablation_summary)
        saved_paths.append(str(ablation_path.relative_to(_PROJECT_ROOT)))
        modes = ablation_summary["modes"]
        if not isinstance(modes, dict):
            raise RuntimeError("ablation summary modes section is invalid.")
        for mode in ModalityAblationMode:
            mode_summary = modes[mode.value]
            if not isinstance(mode_summary, dict):
                raise RuntimeError(
                    f"ablation summary for {mode.value} is invalid."
                )
            mode_overall = mode_summary["overall"]
            if not isinstance(mode_overall, dict):
                raise RuntimeError("ablation overall section is invalid.")
            mode_arousal = mode_overall["arousal"]
            mode_valence = mode_overall["valence"]
            if not isinstance(mode_arousal, dict) or not isinstance(
                mode_valence,
                dict,
            ):
                raise RuntimeError("ablation task sections are invalid.")
            ablation_lines.append(
                f"Ablation {mode.value} | "
                f"arousal_f1={mode_arousal['macro_f1']:.4f} "
                f"valence_f1={mode_valence['macro_f1']:.4f}"
            )
    else:
        ablation_lines.append("Ablation evaluation disabled.")
    speech_statistics_line = (
        "Full-window speech | "
        + " ".join(
            f"{name}={value}"
            for name, value in speech_statistics.to_summary().items()
        )
        + "\n"
        if speech_statistics is not None
        else ""
    )
    print(
        f"Test complete | records={result.record_count} "
        f"participants={result.participant_count}\n"
        f"Arousal | accuracy={arousal['accuracy']:.4f} "
        f"macro_f1={arousal['macro_f1']:.4f}\n"
        f"Valence | accuracy={valence['accuracy']:.4f} "
        f"macro_f1={valence['macro_f1']:.4f}\n"
        f"Thresholds | arousal_high="
        f"{binary_thresholds.arousal_high:.2f} "
        f"valence_high={binary_thresholds.valence_high:.2f}\n"
        f"{speech_statistics_line}"
        f"{chr(10).join(ablation_lines)}\n"
        f"Saved | {', '.join(saved_paths)}"
    )


if __name__ == "__main__":
    main()
