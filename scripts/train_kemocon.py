"""Train one or all repeated dyad-independent K-EmoCon folds."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, TextIO, TypedDict, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SOURCE = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_LOCAL_SOURCE))

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from emotion_model.data import (
    AlignedMultimodalBatch,
    AlignedMultimodalSample,
    DatasetPartition,
    KEmoConPhysioAdapter,
    ParticipantSplit,
    PartitionedManifest,
    apply_modality_keep_masks,
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
    KEmoConExperimentConfig,
    KEmoConModalityMode,
    ModalityAblationMode,
    append_jsonl_artifact,
    build_ablation_evaluation_summary,
    build_cross_fold_evaluation_summary,
    build_configured_kemocon_split,
    build_environment_summary,
    build_evaluation_summary,
    build_kemocon_class_weights,
    build_kemocon_dataset,
    build_kemocon_lr_scheduler,
    build_kemocon_model,
    build_kemocon_objective,
    build_kemocon_optimizer,
    build_valence_class_participant_sampling_weights,
    build_model_parameter_summary,
    build_runtime_availability_audit,
    build_split_summary,
    emotion_metrics_summary,
    fit_kemocon_train_normalizer,
    iter_both_modality_ablation_batches,
    load_kemocon_experiment_config,
    resolve_project_relative,
    write_effective_config_snapshot,
    write_json_artifact,
)
from emotion_model.multimodal import MultimodalEmotionClassifier
from emotion_model.physiology import ChannelwiseZScoreNormalizer
from emotion_model.speech import LightweightNoiseConditionedSpeechClassifier
from emotion_model.training import (
    EmotionTaskClassWeights,
    MultimodalEpochOutput,
    MultimodalTrainingObjective,
    MultimodalTrainingState,
    advance_training_state,
    load_multimodal_checkpoint,
    run_multimodal_training_epoch,
    save_multimodal_checkpoint,
)


class _RunLogger:
    """Write concise timestamped messages to a file and safely above tqdm."""

    def __init__(self, path: Path, *, append: bool) -> None:
        self._stream: TextIO = path.open(
            "a" if append else "w",
            encoding="utf-8",
            newline="\n",
        )

    def emit(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        self._stream.write(f"{timestamp} | {message}\n")
        self._stream.flush()
        tqdm.write(message)

    def close(self) -> None:
        self._stream.close()

    def write_traceback(self) -> None:
        self._stream.write(traceback.format_exc())
        self._stream.write("\n")
        self._stream.flush()


class _LightweightFineTuningDiagnostics(TypedDict):
    """JSON-compatible epoch diagnostics for the lightweight speech path."""

    trainable_wavlm_parameter_count: int
    trainable_downstream_parameter_count: int
    downstream_learning_rate: float | None
    wavlm_learning_rate: float | None
    emotion_layer_weights: dict[str, float]


class _OverfitSelectionDiagnostics(TypedDict):
    """JSON-compatible metadata for one cached diagnostic subset."""

    requested_sample_count: int
    selected_sample_count: int
    candidate_count: int
    inspected_candidate_count: int
    sample_ids: list[str]
    participant_ids: list[str]
    quadrant_counts: dict[str, int]
    modality_mode: str
    speech_availability: dict[str, int]


_OVERFIT_ACCURACY_TARGET = 0.95
_OVERFIT_MACRO_F1_TARGET = 0.95
_OVERFIT_DEFAULT_EPOCHS = 200
_OVERFIT_LEARNING_RATE = 1.0e-3


def _relative_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "paths must be project-relative and cannot contain '..'."
        )
    return path


def _overfit_sample_count(value: str) -> int:
    """Parse a diagnostic subset size of at least four records."""

    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--overfit-samples must be an integer."
        ) from error
    if count < 4:
        raise argparse.ArgumentTypeError(
            "--overfit-samples must be at least 4 for four-quadrant balancing."
        )
    return count


def _positive_epoch_count(value: str) -> int:
    """Parse a positive diagnostic epoch count."""

    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--overfit-epochs must be an integer."
        ) from error
    if count <= 0:
        raise argparse.ArgumentTypeError("--overfit-epochs must be positive.")
    return count


def _configure_overfit_diagnostic(
    config: KEmoConExperimentConfig,
    *,
    sample_count: int,
    epochs: int,
    modality_mode: KEmoConModalityMode,
) -> KEmoConExperimentConfig:
    """Return an isolated, regularization-free small-sample configuration.

    Args:
        config: Validated production configuration.
        sample_count: Diagnostic subset size, at least four.
        epochs: Positive number of repeated passes over the cached subset.
        modality_mode: Must be the V4.2 multimodal mode.

    Returns:
        A new frozen configuration. It uses a separate output directory,
        zero dropout/weight decay/class weighting, no scheduler, no modality
        dropout, fixed 0.5 thresholds, and validation-loss checkpointing.
    """

    if not isinstance(config, KEmoConExperimentConfig):
        raise TypeError("config must be KEmoConExperimentConfig.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer, not bool.")
    if sample_count < 4:
        raise ValueError("sample_count must be at least four.")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be an integer, not bool.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not isinstance(modality_mode, KEmoConModalityMode):
        raise TypeError("modality_mode must be KEmoConModalityMode.")
    directory_name = f"overfit_{sample_count}"
    output_dir = config.paths.output_dir / directory_name
    return replace(
        config,
        paths=replace(config.paths, output_dir=output_dir),
        model=replace(config.model, dropout=0.0),
        loss=replace(
            config.loss,
            class_weight_power=0.0,
            speech_aux_weight=config.loss.speech_aux_weight,
            physiology_aux_weight=config.loss.physiology_aux_weight,
        ),
        training=replace(
            config.training,
            modality_mode=modality_mode,
            batch_size=min(16, sample_count),
            num_workers=0,
            pin_memory=False,
            epochs=epochs,
            learning_rate=_OVERFIT_LEARNING_RATE,
            weight_decay=0.0,
            early_stopping_patience=epochs + 1,
            checkpoint_selection_metric="validation_loss",
            lr_scheduler_enabled=False,
            speech_modality_dropout=0.0,
            physiology_modality_dropout=0.0,
            sampling_policy="uniform",
            valence_low_sampling_mass=(
                config.training.valence_low_sampling_mass
            ),
            threshold_calibration_enabled=False,
            evaluate_ablation=False,
        ),
    )


def _overfit_config_overrides(
    config: KEmoConExperimentConfig,
) -> dict[str, dict[str, object]]:
    """Return YAML-compatible effective overrides for a diagnostic run."""

    return {
        "paths": {"output_dir": config.paths.output_dir.as_posix()},
        "model": {"dropout": config.model.dropout},
        "loss": {
            "speech_aux_weight": config.loss.speech_aux_weight,
            "physiology_aux_weight": config.loss.physiology_aux_weight,
            "class_weight_power": config.loss.class_weight_power,
        },
        "training": {
            "modality_mode": config.training.modality_mode.value,
            "batch_size": config.training.batch_size,
            "num_workers": config.training.num_workers,
            "pin_memory": config.training.pin_memory,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "early_stopping_patience": config.training.early_stopping_patience,
            "checkpoint_selection_metric": (
                config.training.checkpoint_selection_metric
            ),
            "lr_scheduler_enabled": config.training.lr_scheduler_enabled,
            "speech_modality_dropout": (
                config.training.speech_modality_dropout
            ),
            "physiology_modality_dropout": (
                config.training.physiology_modality_dropout
            ),
            "sampling_policy": config.training.sampling_policy,
            "valence_low_sampling_mass": (
                config.training.valence_low_sampling_mass
            ),
            "threshold_calibration_enabled": (
                config.training.threshold_calibration_enabled
            ),
            "threshold_calibration_shrinkage": (
                config.training.threshold_calibration_shrinkage
            ),
            "evaluate_ablation": config.training.evaluate_ablation,
        },
    }


def _select_balanced_overfit_samples(
    dataset: Dataset[AlignedMultimodalSample],
    *,
    sample_count: int,
    seed: int,
    modality_mode: KEmoConModalityMode,
) -> tuple[tuple[AlignedMultimodalSample, ...], _OverfitSelectionDiagnostics]:
    """Cache a deterministic, balanced subset valid for one modality mode.

    Args:
        dataset: Lazy training dataset returning speech ``[L]`` and physiology
            ``[T,C]`` samples with ``True=valid`` masks.
        sample_count: Requested subset size, at least four.
        seed: Integer used only to deterministically shuffle candidates.
        modality_mode: Required runtime modality availability.

    Returns:
        The selected CPU samples and JSON-compatible selection diagnostics.
        Every sample has the modality required by ``modality_mode`` and valid
        binary labels. Multimodal selection requires both modalities.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If any quadrant cannot supply its balanced quota.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a torch Dataset.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer, not bool.")
    if sample_count < 4:
        raise ValueError("sample_count must be at least four.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool.")
    if not isinstance(modality_mode, KEmoConModalityMode):
        raise TypeError("modality_mode must be KEmoConModalityMode.")
    if sample_count > len(dataset):
        raise ValueError(
            f"sample_count={sample_count} exceeds training records={len(dataset)}."
        )

    quotas = {
        quadrant: sample_count // 4 + int(quadrant < sample_count % 4)
        for quadrant in range(4)
    }
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected: dict[int, list[AlignedMultimodalSample]] = {
        quadrant: [] for quadrant in range(4)
    }
    inspected = 0
    for index in indices:
        if all(len(selected[item]) >= quotas[item] for item in range(4)):
            break
        sample = dataset[index]
        inspected += 1
        if not sample.speech_available:
            continue
        if not sample.physiology_available:
            continue
        if sample.arousal_label not in (0, 1) or sample.valence_label not in (0, 1):
            continue
        quadrant = sample.arousal_label + 2 * sample.valence_label
        if len(selected[quadrant]) < quotas[quadrant]:
            selected[quadrant].append(sample)

    counts = {quadrant: len(values) for quadrant, values in selected.items()}
    missing = {
        quadrant: quotas[quadrant] - counts[quadrant]
        for quadrant in range(4)
        if counts[quadrant] < quotas[quadrant]
    }
    if missing:
        raise RuntimeError(
            "could not build a balanced overfit subset for "
            f"modality_mode={modality_mode.value}; "
            f"requested quotas={quotas}, found={counts}, missing={missing}."
        )
    samples = tuple(
        sample
        for quadrant in range(4)
        for sample in selected[quadrant]
    )
    participant_ids = sorted({sample.record.participant_id for sample in samples})
    speech_statistics = FullWindowSpeechAvailabilityStatistics()
    speech_statistics.observe(collate_aligned_multimodal_samples(samples))
    diagnostics: _OverfitSelectionDiagnostics = {
        "requested_sample_count": sample_count,
        "selected_sample_count": len(samples),
        "candidate_count": len(dataset),
        "inspected_candidate_count": inspected,
        "sample_ids": [sample.record.sample_id for sample in samples],
        "participant_ids": participant_ids,
        "quadrant_counts": {
            ("LALV", "HALV", "LAHV", "HAHV")[quadrant]: counts[quadrant]
            for quadrant in range(4)
        },
        "modality_mode": modality_mode.value,
        "speech_availability": speech_statistics.to_summary(),
    }
    return samples, diagnostics


def _diagnostic_parameter_group(parameter_name: str) -> str:
    if "batch_scheduler.speech_classifier" in parameter_name:
        if (
            "relation_differential" in parameter_name
            or "relation_denoiser" in parameter_name
        ):
            return "speech_relation_differential"
        return "speech"
    if "batch_scheduler.physiology_classifier" in parameter_name:
        return "physiology"
    if parameter_name.startswith("multimodal_fusion."):
        return "fusion"
    if parameter_name.startswith(
        (
            "classifier_trunk.",
            "arousal_head.",
            "valence_head.",
            "quadrant_head.",
        )
    ):
        return "classifier"
    return parameter_name.split(".", maxsplit=1)[0]


def _snapshot_trainable_parameters(
    model: MultimodalEmotionClassifier,
) -> dict[str, torch.Tensor]:
    """Clone every trainable parameter to finite float CPU tensors ``[...]``."""

    return {
        name: parameter.detach().to(device="cpu", dtype=torch.float64).clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_update_diagnostics(
    model: MultimodalEmotionClassifier,
    initial: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Measure parameter deltas and last-batch gradient norms by module.

    Args:
        model: Trained classifier with arbitrary parameter tensor shapes.
        initial: CPU float snapshots ``[...]`` keyed by trainable parameter name.

    Returns:
        JSON-compatible global and per-module tensor counts, parameter counts,
        delta L2 norms, relative delta norms, and last-batch gradient L2 norms.
    """

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(initial):
        raise RuntimeError(
            "trainable parameter names changed during diagnostic training."
        )
    accumulators: dict[str, dict[str, float | int]] = {}
    for name, parameter in current.items():
        group = _diagnostic_parameter_group(name)
        accumulator = accumulators.setdefault(
            group,
            {
                "tensor_count": 0,
                "parameter_count": 0,
                "changed_tensor_count": 0,
                "gradient_tensor_count": 0,
                "delta_squared": 0.0,
                "initial_squared": 0.0,
                "gradient_squared": 0.0,
            },
        )
        value = parameter.detach().to(device="cpu", dtype=torch.float64)
        baseline = initial[name]
        delta = value - baseline
        accumulator["tensor_count"] += 1
        accumulator["parameter_count"] += parameter.numel()
        accumulator["changed_tensor_count"] += int(not torch.equal(value, baseline))
        accumulator["delta_squared"] += float(torch.sum(delta * delta).item())
        accumulator["initial_squared"] += float(
            torch.sum(baseline * baseline).item()
        )
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(dtype=torch.float64)
            accumulator["gradient_tensor_count"] += 1
            accumulator["gradient_squared"] += float(
                torch.sum(gradient * gradient).item()
            )

    modules: dict[str, object] = {}
    global_delta_squared = 0.0
    global_initial_squared = 0.0
    global_gradient_squared = 0.0
    global_tensor_count = 0
    global_changed_tensor_count = 0
    global_gradient_tensor_count = 0
    global_parameter_count = 0
    for name, accumulator in sorted(accumulators.items()):
        delta_squared = float(accumulator.pop("delta_squared"))
        initial_squared = float(accumulator.pop("initial_squared"))
        gradient_squared = float(accumulator.pop("gradient_squared"))
        delta_l2 = math.sqrt(delta_squared)
        initial_l2 = math.sqrt(initial_squared)
        modules[name] = {
            **accumulator,
            "delta_l2": delta_l2,
            "relative_delta_l2": delta_l2 / max(initial_l2, 1.0e-12),
            "last_batch_gradient_l2": math.sqrt(gradient_squared),
        }
        global_delta_squared += delta_squared
        global_initial_squared += initial_squared
        global_gradient_squared += gradient_squared
        global_tensor_count += int(accumulator["tensor_count"])
        global_changed_tensor_count += int(accumulator["changed_tensor_count"])
        global_gradient_tensor_count += int(accumulator["gradient_tensor_count"])
        global_parameter_count += int(accumulator["parameter_count"])
    global_delta_l2 = math.sqrt(global_delta_squared)
    return {
        "global": {
            "tensor_count": global_tensor_count,
            "parameter_count": global_parameter_count,
            "changed_tensor_count": global_changed_tensor_count,
            "gradient_tensor_count": global_gradient_tensor_count,
            "delta_l2": global_delta_l2,
            "relative_delta_l2": global_delta_l2
            / max(math.sqrt(global_initial_squared), 1.0e-12),
            "last_batch_gradient_l2": math.sqrt(global_gradient_squared),
        },
        "modules": modules,
    }


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("training.device must be 'cuda' or 'cpu'.")


def _seed_worker(worker_id: int) -> None:
    """Seed Python and PyTorch from the DataLoader-assigned worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader(
    dataset: object,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader[AlignedMultimodalBatch]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampler: WeightedRandomSampler | None = None
    if sample_weights is not None:
        if (
            sample_weights.dtype != torch.float64
            or sample_weights.device.type != "cpu"
            or tuple(sample_weights.shape) != (len(cast(Dataset[object], dataset)),)
            or not bool(torch.isfinite(sample_weights).all())
            or not bool((sample_weights > 0.0).all())
        ):
            raise ValueError(
                "sample_weights must be positive finite float64 CPU [N]."
            )
        sampler_generator = torch.Generator(device="cpu")
        sampler_generator.manual_seed(seed + 1_000_000)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=sample_weights.numel(),
            replacement=True,
            generator=sampler_generator,
        )
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_aligned_multimodal_samples,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _audit_dataset_runtime_availability(
    dataset: Dataset[AlignedMultimodalSample],
    *,
    description: str,
) -> dict[str, object]:
    """Load each sample once and return its declaration/runtime audit."""

    indices = tqdm(
        range(len(dataset)),
        total=len(dataset),
        desc=description,
        unit="sample",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.25,
    )
    return build_runtime_availability_audit(dataset[index] for index in indices)


def _progress(
    loader: DataLoader[AlignedMultimodalBatch],
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
) -> Iterable[AlignedMultimodalBatch]:
    return cast(
        Iterable[AlignedMultimodalBatch],
        tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch:02d}/{total_epochs:02d} {phase}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.25,
        ),
    )


def _track_speech_availability(
    batches: Iterable[AlignedMultimodalBatch],
    statistics: FullWindowSpeechAvailabilityStatistics,
) -> Iterable[AlignedMultimodalBatch]:
    """Observe reporting metadata and yield each logical batch unchanged."""

    for batch in batches:
        statistics.observe(batch)
        yield batch


class _ModalityDropoutStats:
    """Mutable scalar counts for one training epoch."""

    def __init__(self) -> None:
        self.eligible_count = 0
        self.speech_dropped_count = 0
        self.physiology_dropped_count = 0

    @property
    def dropped_count(self) -> int:
        """Return legacy physiology-drop count."""

        return self.physiology_dropped_count


def _apply_training_symmetric_modality_dropout(
    batches: Iterable[AlignedMultimodalBatch],
    *,
    speech_probability: float,
    physiology_probability: float,
    stats: _ModalityDropoutStats,
    speech_statistics: FullWindowSpeechAvailabilityStatistics | None = None,
) -> Iterable[AlignedMultimodalBatch]:
    """Drop at most one modality and optionally report natural/dropout masks."""

    for batch in batches:
        eligible = batch.speech_available & batch.physiology_available
        eligible_count = int(eligible.sum().item())
        stats.eligible_count += eligible_count
        if (
            speech_probability == 0.0
            and physiology_probability == 0.0
        ) or eligible_count == 0:
            if speech_statistics is not None:
                speech_statistics.observe(batch)
            yield batch
            continue
        draws = torch.rand(len(batch.records), device="cpu")
        speech_dropped = eligible & (draws < speech_probability)
        physiology_dropped = eligible & (
            (draws >= speech_probability)
            & (
                draws
                < speech_probability + physiology_probability
            )
        )
        stats.speech_dropped_count += int(speech_dropped.sum().item())
        stats.physiology_dropped_count += int(
            physiology_dropped.sum().item()
        )
        effective_batch = apply_modality_keep_masks(
            batch,
            speech_keep_mask=~speech_dropped,
            physiology_keep_mask=~physiology_dropped,
        )
        if speech_statistics is not None:
            speech_statistics.observe(batch, effective_batch)
        yield effective_batch


def _apply_training_physiology_dropout(
    batches: Iterable[AlignedMultimodalBatch],
    *,
    probability: float,
    stats: _ModalityDropoutStats,
) -> Iterable[AlignedMultimodalBatch]:
    yield from _apply_training_symmetric_modality_dropout(
        batches,
        speech_probability=0.0,
        physiology_probability=probability,
        stats=stats,
    )


def _epoch_payload(output: MultimodalEpochOutput) -> dict[str, object]:
    return {
        "batch_count": output.batch_count,
        "supervised_batch_count": output.supervised_batch_count,
        "empty_supervision_batch_count": output.empty_supervision_batch_count,
        "optimizer_step_count": output.optimizer_step_count,
        "active_target_count": output.active_target_count,
        "max_gradient_norm": output.max_gradient_norm,
        "loss": asdict(output.loss_averages),
    }


def _class_weight_payload(
    weights: object,
) -> dict[str, object]:
    arousal = weights.arousal.detach().cpu().tolist()  # type: ignore[attr-defined]
    valence = weights.valence.detach().cpu().tolist()  # type: ignore[attr-defined]
    return {"arousal": arousal, "valence": valence}


def _data_line(partitioned: PartitionedManifest) -> str:
    return (
        f"Data | train={len(partitioned.train_records)} "
        f"validation={len(partitioned.validation_records)} "
        f"test={len(partitioned.test_records)}"
    )


def _key_test_metrics(
    summary: Mapping[str, object],
) -> tuple[float, float, float, float]:
    overall = summary["overall"]
    if not isinstance(overall, Mapping):
        raise RuntimeError("test summary overall section is invalid.")
    arousal = overall["arousal"]
    valence = overall["valence"]
    if not isinstance(arousal, Mapping) or not isinstance(valence, Mapping):
        raise RuntimeError("test summary task sections are invalid.")
    return (
        float(arousal["accuracy"]),
        float(arousal["macro_f1"]),
        float(valence["accuracy"]),
        float(valence["macro_f1"]),
    )


def _validation_selection_value(
    *,
    metric_name: str,
    validation_loss: float,
    arousal_macro_f1: float,
    valence_macro_f1: float,
    participant_arousal_macro_f1: float | None = None,
    participant_valence_macro_f1: float | None = None,
) -> float:
    if metric_name == "validation_loss":
        return validation_loss
    if metric_name == "mean_macro_f1":
        return (arousal_macro_f1 + valence_macro_f1) / 2.0
    if metric_name in {
        "participant_mean_macro_f1",
        "calibrated_participant_mean_macro_f1",
    }:
        if (
            participant_arousal_macro_f1 is None
            or participant_valence_macro_f1 is None
        ):
            raise ValueError(
                "participant macro F1 values are required for participant "
                "checkpoint selection."
            )
        return (
            participant_arousal_macro_f1
            + participant_valence_macro_f1
        ) / 2.0
    raise ValueError(f"unsupported checkpoint selection metric: {metric_name}")


def _selection_improved(
    *,
    metric_name: str,
    candidate: float,
    best: float | None,
    candidate_validation_loss: float,
    best_validation_loss: float | None,
) -> bool:
    if best is None:
        return True
    tolerance = 1e-12
    if metric_name == "validation_loss":
        return candidate < best - tolerance
    if candidate > best + tolerance:
        return True
    return (
        abs(candidate - best) <= tolerance
        and (
            best_validation_loss is None
            or candidate_validation_loss < best_validation_loss
        )
    )


def _training_summary_base(
    *,
    status: str,
    fold_index: int,
    started_utc: str,
    best_checkpoint: Path,
) -> dict[str, object]:
    return {
        "status": status,
        "fold_index": fold_index,
        "started_utc": started_utc,
        "best_checkpoint": best_checkpoint.as_posix(),
    }


def _lightweight_fine_tuning_diagnostics(
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
) -> _LightweightFineTuningDiagnostics | None:
    """Return scalar LR/count diagnostics and H9--H12 weights.

    Args:
        model: Emotion classifier whose speech waveform input has shape ``[B,L]``.
        optimizer: Optimizer containing scalar-LR named parameter groups.

    Returns:
        JSON-compatible scalar diagnostics for a lightweight speech branch, or
        ``None`` when the selected model has no such branch.
    """

    speech_classifier = model.batch_scheduler.speech_classifier
    if not isinstance(
        speech_classifier,
        LightweightNoiseConditionedSpeechClassifier,
    ):
        return None
    wavlm_parameters = tuple(speech_classifier.wavlm_encoder.model.parameters())
    wavlm_parameter_ids = {id(parameter) for parameter in wavlm_parameters}
    group_learning_rates: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        learning_rate = group.get("lr")
        if isinstance(group_name, str) and isinstance(learning_rate, (int, float)):
            group_learning_rates[group_name] = float(learning_rate)
    weights = (
        speech_classifier.emotion_layer_aggregation.normalized_weights()
        .detach()
        .cpu()
        .tolist()
    )
    return {
        "trainable_wavlm_parameter_count": sum(
            parameter.numel()
            for parameter in wavlm_parameters
            if parameter.requires_grad
        ),
        "trainable_downstream_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in wavlm_parameter_ids
        ),
        "downstream_learning_rate": group_learning_rates.get("downstream"),
        "wavlm_learning_rate": group_learning_rates.get("wavlm"),
        "emotion_layer_weights": {
            name: float(weight)
            for name, weight in zip(("H9", "H10", "H11", "H12"), weights, strict=True)
        },
    }


def _evaluate_same_sample_ablation(
    *,
    model: MultimodalEmotionClassifier,
    objective: MultimodalTrainingObjective,
    evaluation_loader: DataLoader[AlignedMultimodalBatch],
    split: ParticipantSplit,
    partition: DatasetPartition,
    class_weights: EmotionTaskClassWeights,
    binary_thresholds: BinaryDecisionThresholds,
    fold_index: int,
    checkpoint_relative: Path,
    description_prefix: str,
) -> dict[str, object]:
    """Evaluate a multimodal model under three conditions on identical rows.

    Args:
        model: Multimodal classifier accepting strict aligned batches.
        objective: Training objective used to calculate evaluation loss.
        evaluation_loader: Re-iterable loader of logical batches with speech
            ``[Bs,L]`` and physiology ``[Bp,T,C]`` compact tensors.
        split: Participant split defining the requested partition membership.
        partition: Train, validation, or test membership enforced in metrics.
        class_weights: Binary task class weights ``[2]``.
        binary_thresholds: Arousal/valence high-class decision thresholds.
        fold_index: Zero-based split index recorded in artifacts.
        checkpoint_relative: Project-relative checkpoint path recorded in output.
        description_prefix: Progress-bar label before each ablation mode.

    Returns:
        JSON-compatible full/speech-only/physiology-only summaries evaluated
        over the same originally-both-valid sample identifiers.
    """

    results: dict[str, ParticipantIndependentEvaluationOutput] = {}
    for mode in ModalityAblationMode:
        progressed = tqdm(
            evaluation_loader,
            total=len(evaluation_loader),
            desc=f"{description_prefix} {mode.value}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.25,
        )
        result = evaluate_participant_independent(
            model,
            objective,
            iter_both_modality_ablation_batches(progressed, mode),
            scope=ParticipantEvaluationScope(
                split,
                partition,
                require_all_partition_participants=False,
            ),
            class_weights=class_weights,
            binary_thresholds=binary_thresholds,
        )
        results[mode.value] = result
    return build_ablation_evaluation_summary(
        results,
        fold_index=fold_index,
        checkpoint=checkpoint_relative,
    )


def _run(
    *,
    arguments: argparse.Namespace,
    fold_index: int,
    device: torch.device,
    fold_directory: Path,
    logger: _RunLogger,
    started_utc: str,
    started_clock: float,
    fold_relative: Path,
) -> None:
    config_path = resolve_project_relative(_PROJECT_ROOT, arguments.config)
    config = load_kemocon_experiment_config(config_path)
    overfit_sample_count: int | None = arguments.overfit_samples
    if overfit_sample_count is not None:
        config = _configure_overfit_diagnostic(
            config,
            sample_count=overfit_sample_count,
            epochs=arguments.overfit_epochs,
            modality_mode=KEmoConModalityMode.MULTIMODAL,
        )
    report_full_window_speech = (
        config.dataset.speech_activity.availability_policy == "source_presence"
    )
    manifest_path = resolve_project_relative(_PROJECT_ROOT, config.paths.manifest)
    data_root = resolve_project_relative(_PROJECT_ROOT, config.paths.data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{config.paths.manifest} is missing; run "
            "python scripts/prepare_kemocon.py first."
        )

    write_effective_config_snapshot(
        config_path,
        fold_directory / "run_config.yaml",
        fold_index=fold_index,
        overrides=(
            _overfit_config_overrides(config)
            if overfit_sample_count is not None
            else None
        ),
    )
    environment = build_environment_summary(device)
    environment["command"] = [Path(sys.executable).name, *sys.argv]
    environment["overfit_diagnostic"] = overfit_sample_count is not None
    if overfit_sample_count is not None:
        environment["overfit_modality_mode"] = config.training.modality_mode.value
    write_json_artifact(
        fold_directory / "environment.json",
        environment,
    )

    manifest = read_manifest_json(manifest_path)
    split = build_configured_kemocon_split(
        manifest,
        config.split,
        fold_index=fold_index,
        label_protocol=config.dataset.label_protocol,
    )
    partitioned = partition_manifest_by_participant(manifest, split)
    split_summary = build_split_summary(
        partitioned,
        fold_index=fold_index,
        split_seed=config.split.seed,
        label_protocol=config.dataset.label_protocol,
    )
    split_summary["split_strategy"] = config.split.strategy
    split_summary["training_modality_mode"] = (
        config.training.modality_mode.value
    )
    if config.split.strategy == "fixed_ratio":
        split_summary["requested_record_fractions"] = {
            "train": config.split.train_fraction,
            "validation": config.split.validation_fraction,
            "test": config.split.test_fraction,
        }
    write_json_artifact(fold_directory / "split_summary.json", split_summary)
    logger.emit(
        f"Run | fold={fold_index} device={device} "
        f"batch={config.training.batch_size} workers={config.training.num_workers}"
    )
    if overfit_sample_count is not None:
        logger.emit(
            "Overfit diagnostic | "
            f"samples={overfit_sample_count} epochs={config.training.epochs} "
            f"modality={config.training.modality_mode.value} "
            f"learning_rate={config.training.learning_rate:.2e} "
            "same_subset_train_eval=True thresholds=0.5"
        )
    logger.emit(_data_line(partitioned))

    logger.emit("Fitting training-only physiology normalization statistics...")
    normalizer = fit_kemocon_train_normalizer(
        partitioned.train_records,
        manifest.channel_specs,
        KEmoConPhysioAdapter(data_root),
    )
    write_json_artifact(
        fold_directory / "normalizer.json",
        normalizer.state_dict(),
    )
    train_dataset = build_kemocon_dataset(
        partitioned.train_records,
        manifest.channel_specs,
        data_root=data_root,
        config=config.dataset,
        normalizer=normalizer,
        modality_mode=config.training.modality_mode,
    )
    validation_dataset = build_kemocon_dataset(
        partitioned.validation_records,
        manifest.channel_specs,
        data_root=data_root,
        config=config.dataset,
        normalizer=normalizer,
        modality_mode=config.training.modality_mode,
    )
    test_dataset = build_kemocon_dataset(
        partitioned.test_records,
        manifest.channel_specs,
        data_root=data_root,
        config=config.dataset,
        normalizer=normalizer,
        modality_mode=config.training.modality_mode,
    )
    logger.emit("Auditing declared versus runtime modality availability...")
    runtime_audit = {
        "train": _audit_dataset_runtime_availability(
            train_dataset,
            description=f"Fold {fold_index} train availability audit",
        ),
        "validation": _audit_dataset_runtime_availability(
            validation_dataset,
            description=f"Fold {fold_index} validation availability audit",
        ),
        "test": _audit_dataset_runtime_availability(
            test_dataset,
            description=f"Fold {fold_index} test availability audit",
        ),
    }
    split_summary["runtime_modality_availability_audit"] = runtime_audit
    write_json_artifact(fold_directory / "split_summary.json", split_summary)
    for partition_name, audit in runtime_audit.items():
        physiology = cast(
            dict[str, int],
            cast(dict[str, object], audit["modality_counts"])["physiology"],
        )
        logger.emit(
            "Availability audit | "
            f"partition={partition_name} records={audit['record_count']} "
            f"physiology_declared={physiology['declared_count']} "
            f"physiology_runtime={physiology['runtime_available_count']} "
            "physiology_declared_but_unavailable="
            f"{physiology['declared_but_runtime_unavailable_count']}"
        )
    overfit_samples: tuple[AlignedMultimodalSample, ...] | None = None
    overfit_selection: _OverfitSelectionDiagnostics | None = None
    if overfit_sample_count is not None:
        logger.emit(
            "Selecting and caching a balanced training subset for "
            f"{config.training.modality_mode.value}..."
        )
        overfit_samples, overfit_selection = _select_balanced_overfit_samples(
            train_dataset,
            sample_count=overfit_sample_count,
            seed=config.training.seed + fold_index,
            modality_mode=config.training.modality_mode,
        )
        training_data: object = overfit_samples
        validation_data: object = overfit_samples
        effective_train_records = tuple(
            sample.record for sample in overfit_samples
        )
        split_summary["overfit_diagnostic"] = dict(overfit_selection)
        logger.emit(
            "Overfit subset | "
            f"selected={len(overfit_samples)} "
            f"inspected={overfit_selection['inspected_candidate_count']} "
            f"quadrants={overfit_selection['quadrant_counts']}"
        )
    else:
        training_data = train_dataset
        validation_data = validation_dataset
        effective_train_records = partitioned.train_records
    training_sample_weights = (
        build_valence_class_participant_sampling_weights(
            effective_train_records,
            protocol=config.dataset.label_protocol,
            low_class_mass=config.training.valence_low_sampling_mass,
        )
        if config.training.sampling_policy
        == "soft_valence_class_participant_balanced"
        else None
    )
    split_summary["sampling_policy"] = config.training.sampling_policy
    split_summary["sampling_weight_summary"] = (
        None
        if training_sample_weights is None
        else {
            "count": training_sample_weights.numel(),
            "sum": float(training_sample_weights.sum().item()),
            "minimum": float(training_sample_weights.min().item()),
            "maximum": float(training_sample_weights.max().item()),
            "effective_sample_size": float(
                1.0 / training_sample_weights.square().sum().item()
            ),
            "expected_valence_class_mass": {
                "low": config.training.valence_low_sampling_mass,
                "high": 1.0 - config.training.valence_low_sampling_mass,
            },
            "within_class_participant_mass": "equal",
        }
    )
    train_loader = _loader(
        training_data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        shuffle=True,
        seed=config.training.seed + fold_index,
        sample_weights=training_sample_weights,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        shuffle=False,
        seed=config.training.seed + 10_000 + fold_index,
    )

    logger.emit(
        "Building model | modality_mode="
        f"{config.training.modality_mode.value}"
    )
    model = build_kemocon_model(
        config,
        manifest.channel_specs,
        project_root=_PROJECT_ROOT,
        device=device,
    )
    objective = build_kemocon_objective(config.loss)
    optimizer = build_kemocon_optimizer(model, config)
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    initial_parameter_snapshot = (
        _snapshot_trainable_parameters(model)
        if overfit_sample_count is not None
        else None
    )
    scheduler_mode: Literal["min", "max"] = (
        "min"
        if config.training.checkpoint_selection_metric == "validation_loss"
        else "max"
    )
    scheduler = build_kemocon_lr_scheduler(
        optimizer,
        config,
        mode=scheduler_mode,
    )
    class_weights = build_kemocon_class_weights(
        effective_train_records,
        protocol=config.dataset.label_protocol,
        device=device,
        participant_balanced=True,
        class_weight_power=config.loss.class_weight_power,
    )
    split_summary["class_weights"] = _class_weight_payload(class_weights)
    split_summary["class_weighting_policy"] = (
        "participant_balanced_inverse_frequency"
    )
    split_summary["class_weight_power"] = config.loss.class_weight_power
    split_summary["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    split_summary["total_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    split_summary["model_parameter_summary"] = build_model_parameter_summary(
        model
    )
    split_summary["optimizer_parameter_groups"] = [
        {
            "group_name": group["group_name"],
            "learning_rate": float(group["lr"]),
            "parameter_count": sum(
                parameter.numel() for parameter in group["params"]
            ),
        }
        for group in optimizer.param_groups
    ]
    write_json_artifact(fold_directory / "split_summary.json", split_summary)

    state = MultimodalTrainingState()
    best_epoch: int | None = None
    best_selection_value: float | None = None
    best_checkpoint_validation_loss: float | None = None
    best_binary_thresholds: BinaryDecisionThresholds | None = None
    if arguments.resume is not None:
        loaded = load_multimodal_checkpoint(
            resolve_project_relative(_PROJECT_ROOT, arguments.resume),
            model=model,
            optimizer=optimizer,
            objective=objective,
        )
        state = loaded.training_state
        metadata_modality_mode = loaded.extra_metadata.get(
            "training_modality_mode"
        )
        if (
            metadata_modality_mode is not None
            and metadata_modality_mode != config.training.modality_mode.value
        ):
            raise ValueError(
                "resume checkpoint training modality mode does not match the "
                "current configuration."
            )
        metadata_best_epoch = loaded.extra_metadata.get("best_epoch")
        if isinstance(metadata_best_epoch, int) and not isinstance(
            metadata_best_epoch,
            bool,
        ):
            best_epoch = metadata_best_epoch
        metadata_selection_metric = loaded.extra_metadata.get(
            "checkpoint_selection_metric"
        )
        if (
            metadata_selection_metric is not None
            and metadata_selection_metric
            != config.training.checkpoint_selection_metric
        ):
            raise ValueError(
                "resume checkpoint selection metric does not match the "
                "current configuration."
            )
        metadata_class_weight_power = loaded.extra_metadata.get(
            "class_weight_power"
        )
        if (
            metadata_class_weight_power is not None
            and (
                isinstance(metadata_class_weight_power, bool)
                or not isinstance(metadata_class_weight_power, (int, float))
                or not math.isclose(
                    float(metadata_class_weight_power),
                    config.loss.class_weight_power,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
        ):
            raise ValueError(
                "resume checkpoint class_weight_power does not match the "
                "current configuration."
            )
        metadata_threshold_shrinkage = loaded.extra_metadata.get(
            "threshold_calibration_shrinkage"
        )
        if (
            isinstance(metadata_threshold_shrinkage, bool)
            or not isinstance(metadata_threshold_shrinkage, (int, float))
            or not math.isclose(
                float(metadata_threshold_shrinkage),
                config.training.threshold_calibration_shrinkage,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "resume checkpoint threshold calibration shrinkage does not "
                "match the current configuration."
            )
        metadata_best_selection = loaded.extra_metadata.get(
            "best_selection_value"
        )
        if (
            isinstance(metadata_best_selection, (int, float))
            and not isinstance(metadata_best_selection, bool)
            and math.isfinite(float(metadata_best_selection))
        ):
            best_selection_value = float(metadata_best_selection)
        metadata_best_validation = loaded.extra_metadata.get(
            "best_checkpoint_validation_loss"
        )
        if (
            isinstance(metadata_best_validation, (int, float))
            and not isinstance(metadata_best_validation, bool)
            and math.isfinite(float(metadata_best_validation))
        ):
            best_checkpoint_validation_loss = float(metadata_best_validation)
        metadata_thresholds = loaded.extra_metadata.get(
            "binary_decision_thresholds"
        )
        if metadata_thresholds is not None:
            best_binary_thresholds = BinaryDecisionThresholds.from_metadata(
                metadata_thresholds
            )
        if (
            config.training.threshold_calibration_enabled
            and best_binary_thresholds is None
        ):
            raise ValueError(
                "resume checkpoint lacks calibrated binary decision thresholds."
            )
        if (
            best_selection_value is None
            and config.training.checkpoint_selection_metric == "validation_loss"
            and state.best_validation_loss is not None
        ):
            best_selection_value = state.best_validation_loss
            best_checkpoint_validation_loss = state.best_validation_loss
        if scheduler is not None and best_selection_value is not None:
            scheduler.step(best_selection_value)
        logger.emit(
            f"Resume | checkpoint={arguments.resume.as_posix()} "
            f"completed_epochs={state.completed_epochs}"
        )

    write_json_artifact(fold_directory / "split_summary.json", split_summary)

    history_path = fold_directory / "history.jsonl"
    if arguments.resume is None:
        history_path.write_text("", encoding="utf-8")
    epochs_without_improvement = (
        max(0, state.completed_epochs - best_epoch)
        if best_epoch is not None
        else 0
    )
    stop_reason = "maximum_epochs"
    logger.emit(
        f"Training | epochs={config.training.epochs} "
        f"modality_mode={config.training.modality_mode.value} "
        f"selection={config.training.checkpoint_selection_metric} "
        f"early_stopping_patience={config.training.early_stopping_patience} "
        f"speech_dropout={config.training.speech_modality_dropout:.2f} "
        f"physiology_dropout="
        f"{config.training.physiology_modality_dropout:.2f} "
        f"class_weight_power={config.loss.class_weight_power:.2f} "
        f"threshold_calibration="
        f"{config.training.threshold_calibration_enabled} "
        f"threshold_shrinkage="
        f"{config.training.threshold_calibration_shrinkage:.2f} "
        f"sampling_policy={config.training.sampling_policy} "
        f"valence_low_sampling_mass="
        f"{config.training.valence_low_sampling_mass:.2f}"
    )
    if overfit_sample_count is not None:
        logger.emit(
            "Overfit controls | model_dropout=0.00 modality_dropout=0.00 "
            "weight_decay=0.00 class_weight_power=0.00 scheduler=False "
            "early_stopping=False"
        )

    for epoch_index in range(state.completed_epochs, config.training.epochs):
        epoch_number = epoch_index + 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_clock = perf_counter()
        train_clock = perf_counter()
        modality_dropout_stats = _ModalityDropoutStats()
        training_speech_statistics = (
            FullWindowSpeechAvailabilityStatistics()
            if report_full_window_speech
            else None
        )
        validation_speech_statistics = (
            FullWindowSpeechAvailabilityStatistics()
            if report_full_window_speech
            else None
        )
        training_epoch = run_multimodal_training_epoch(
            model,
            objective,
            _apply_training_symmetric_modality_dropout(
                _progress(
                    train_loader,
                    epoch=epoch_number,
                    total_epochs=config.training.epochs,
                    phase="train",
                ),
                speech_probability=(
                    config.training.speech_modality_dropout
                ),
                physiology_probability=(
                    config.training.physiology_modality_dropout
                ),
                stats=modality_dropout_stats,
                speech_statistics=training_speech_statistics,
            ),
            optimizer,
            class_weights=class_weights,
            max_gradient_norm=config.training.max_gradient_norm,
        )
        parameter_diagnostics = (
            _parameter_update_diagnostics(model, initial_parameter_snapshot)
            if initial_parameter_snapshot is not None
            else None
        )
        training_seconds = perf_counter() - train_clock
        validation_clock = perf_counter()
        validation_batches: Iterable[AlignedMultimodalBatch] = _progress(
            validation_loader,
            epoch=epoch_number,
            total_epochs=config.training.epochs,
            phase="validation",
        )
        if validation_speech_statistics is not None:
            validation_batches = _track_speech_availability(
                validation_batches,
                validation_speech_statistics,
            )
        validation_result = evaluate_participant_independent(
            model,
            objective,
            validation_batches,
            scope=ParticipantEvaluationScope(
                split,
                (
                    DatasetPartition.TRAIN
                    if overfit_sample_count is not None
                    else DatasetPartition.VALIDATION
                ),
                require_all_partition_participants=(
                    overfit_sample_count is None
                ),
            ),
            class_weights=class_weights,
            calibrate_thresholds=(
                config.training.threshold_calibration_enabled
            ),
            threshold_calibration_shrinkage=(
                config.training.threshold_calibration_shrinkage
            ),
        )
        validation_epoch = validation_result.loss_epoch
        validation_seconds = perf_counter() - validation_clock
        state = advance_training_state(
            state,
            training_epoch,
            validation_epoch,
        )
        current_validation = validation_epoch.loss_averages.total_loss
        arousal_validation_f1 = (
            validation_result.overall_metrics.arousal.macro_f1
        )
        valence_validation_f1 = (
            validation_result.overall_metrics.valence.macro_f1
        )
        participant_arousal_validation_f1 = (
            validation_result.participant_macro_metrics.arousal.mean_macro_f1
        )
        participant_valence_validation_f1 = (
            validation_result.participant_macro_metrics.valence.mean_macro_f1
        )
        selection_value = _validation_selection_value(
            metric_name=config.training.checkpoint_selection_metric,
            validation_loss=current_validation,
            arousal_macro_f1=arousal_validation_f1,
            valence_macro_f1=valence_validation_f1,
            participant_arousal_macro_f1=(
                participant_arousal_validation_f1
            ),
            participant_valence_macro_f1=(
                participant_valence_validation_f1
            ),
        )
        improved = _selection_improved(
            metric_name=config.training.checkpoint_selection_metric,
            candidate=selection_value,
            best=best_selection_value,
            candidate_validation_loss=current_validation,
            best_validation_loss=best_checkpoint_validation_loss,
        )
        if improved:
            best_epoch = epoch_number
            best_selection_value = selection_value
            best_checkpoint_validation_loss = current_validation
            best_binary_thresholds = (
                validation_result.binary_decision_thresholds
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        learning_rates_before_scheduler = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if scheduler is not None:
            scheduler.step(selection_value)
        learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]

        peak_allocated_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved_gib = (
            torch.cuda.max_memory_reserved(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        metadata = {
            "dataset": "K-EmoCon",
            "training_modality_mode": config.training.modality_mode.value,
            "fold_index": fold_index,
            "best_epoch": best_epoch,
            "config": "run_config.yaml",
            "manifest": config.paths.manifest.as_posix(),
            "annotation_perspective": config.dataset.annotation_perspective,
            "label_protocol": config.dataset.label_protocol.value,
            "checkpoint_selection_metric": (
                config.training.checkpoint_selection_metric
            ),
            "class_weight_power": config.loss.class_weight_power,
            "threshold_calibration_shrinkage": (
                config.training.threshold_calibration_shrinkage
            ),
            "binary_decision_thresholds": (
                validation_result.binary_decision_thresholds.to_metadata()
            ),
            "best_selection_value": best_selection_value,
            "best_checkpoint_validation_loss": (
                best_checkpoint_validation_loss
            ),
            "validation_arousal_macro_f1": arousal_validation_f1,
            "validation_valence_macro_f1": valence_validation_f1,
            "validation_participant_arousal_macro_f1": (
                participant_arousal_validation_f1
            ),
            "validation_participant_valence_macro_f1": (
                participant_valence_validation_f1
            ),
            "lr_scheduler_enabled": config.training.lr_scheduler_enabled,
        }
        if improved:
            save_multimodal_checkpoint(
                fold_directory / "best.pt",
                model=model,
                optimizer=optimizer,
                objective=objective,
                training_state=state,
                extra_metadata=metadata,
            )

        epoch_seconds = perf_counter() - epoch_clock
        history_record = {
            "epoch": epoch_number,
            "completed_utc": datetime.now(UTC).isoformat(),
            "training": _epoch_payload(training_epoch),
            "validation": _epoch_payload(validation_epoch),
            "validation_metrics": emotion_metrics_summary(
                validation_result.overall_metrics
            ),
            "validation_participant_macro": asdict(
                validation_result.participant_macro_metrics
            ),
            "validation_fusion_diagnostics": dict(
                validation_result.fusion_diagnostics
            ),
            "binary_decision_thresholds": (
                validation_result.binary_decision_thresholds.to_metadata()
            ),
            "checkpoint_selection_metric": (
                config.training.checkpoint_selection_metric
            ),
            "selection_value": selection_value,
            "best_selection_value": best_selection_value,
            "minimum_validation_loss": state.best_validation_loss,
            "best_checkpoint_validation_loss": (
                best_checkpoint_validation_loss
            ),
            "learning_rates_before_scheduler": (
                learning_rates_before_scheduler
            ),
            "learning_rates": learning_rates,
            "speech_modality_dropout": {
                "configured_probability": (
                    config.training.speech_modality_dropout
                ),
                "eligible_count": modality_dropout_stats.eligible_count,
                "dropped_count": (
                    modality_dropout_stats.speech_dropped_count
                ),
                "actual_rate": (
                    modality_dropout_stats.speech_dropped_count
                    / modality_dropout_stats.eligible_count
                    if modality_dropout_stats.eligible_count > 0
                    else 0.0
                ),
            },
            "physiology_modality_dropout": {
                "configured_probability": (
                    config.training.physiology_modality_dropout
                ),
                "eligible_count": modality_dropout_stats.eligible_count,
                "dropped_count": modality_dropout_stats.dropped_count,
                "actual_rate": (
                    modality_dropout_stats.dropped_count
                    / modality_dropout_stats.eligible_count
                    if modality_dropout_stats.eligible_count > 0
                    else 0.0
                ),
            },
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "timing_seconds": {
                "training": training_seconds,
                "validation": validation_seconds,
                "epoch": epoch_seconds,
            },
            "cuda_peak_allocated_gib": peak_allocated_gib,
            "cuda_peak_reserved_gib": peak_reserved_gib,
            "global_optimizer_steps": state.global_optimizer_steps,
        }
        if (
            training_speech_statistics is not None
            and validation_speech_statistics is not None
        ):
            history_record["full_window_speech_statistics"] = {
                "training_after_sampler": (
                    training_speech_statistics.to_summary()
                ),
                "validation": validation_speech_statistics.to_summary(),
            }
        fine_tuning_diagnostics = _lightweight_fine_tuning_diagnostics(
            model,
            optimizer,
        )
        if fine_tuning_diagnostics is not None:
            history_record.update(dict(fine_tuning_diagnostics))
        if parameter_diagnostics is not None:
            history_record["parameter_diagnostics"] = parameter_diagnostics
            history_record["evaluation_scope"] = "same_training_subset"
        for activity_name in (
            "speech_available_count",
            "speech_source_unavailable_count",
            "mean_speech_activity_ratio",
            "median_speech_activity_ratio",
            "p10_speech_activity_ratio",
            "p90_speech_activity_ratio",
        ):
            if activity_name in validation_result.fusion_diagnostics:
                history_record[activity_name] = (
                    validation_result.fusion_diagnostics[activity_name]
                )
        append_jsonl_artifact(history_path, history_record)
        marker = "*" if improved else "-"
        logger.emit(
            f"Epoch {epoch_number:02d}/{config.training.epochs:02d} {marker} | "
            f"train={training_epoch.loss_averages.total_loss:.4f} "
            f"val={current_validation:.4f} "
            f"ar_f1={arousal_validation_f1:.4f} "
            f"va_f1={valence_validation_f1:.4f} "
            f"p_ar_f1={participant_arousal_validation_f1:.4f} "
            f"p_va_f1={participant_valence_validation_f1:.4f} "
            f"thresholds=[ar "
            f"{validation_result.binary_decision_thresholds.arousal_high:.2f}, "
            f"va "
            f"{validation_result.binary_decision_thresholds.valence_high:.2f}] "
            f"select={selection_value:.4f} "
            f"best={best_selection_value:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={epoch_seconds:.1f}s "
            f"vram={peak_allocated_gib:.2f}GiB"
        )
        if training_speech_statistics is not None:
            speech_summary = training_speech_statistics.to_summary()
            logger.emit(
                "Full-window speech | "
                f"total={speech_summary['total_records']} "
                f"source_present={speech_summary['speech_source_present']} "
                f"ratio_eq_0={speech_summary['speech_activity_ratio_eq_0']} "
                "ratio_0_to_0.1="
                f"{speech_summary['speech_activity_ratio_0_to_0_1']} "
                "ratio_0.1_to_0.25="
                f"{speech_summary['speech_activity_ratio_0_1_to_0_25']} "
                "ratio_ge_0.25="
                f"{speech_summary['speech_activity_ratio_ge_0_25']} "
                "source_unavailable="
                f"{speech_summary['speech_source_unavailable']} "
                "artificial_dropout="
                f"{speech_summary['artificial_speech_modality_dropout']}"
            )
        if validation_speech_statistics is not None:
            logger.emit(
                "Validation full-window speech | "
                + " ".join(
                    f"{name}={value}"
                    for name, value in (
                        validation_speech_statistics.to_summary().items()
                    )
                )
            )
        if parameter_diagnostics is not None:
            global_parameters = parameter_diagnostics["global"]
            if not isinstance(global_parameters, Mapping):
                raise RuntimeError("global parameter diagnostics are invalid.")
            logger.emit(
                "Parameter update | "
                f"changed={global_parameters['changed_tensor_count']}/"
                f"{global_parameters['tensor_count']} "
                f"delta_l2={global_parameters['delta_l2']:.6g} "
                "last_batch_grad_l2="
                f"{global_parameters['last_batch_gradient_l2']:.6g}"
            )
        if fine_tuning_diagnostics is not None:
            emotion_weights = fine_tuning_diagnostics["emotion_layer_weights"]
            wavlm_lr = fine_tuning_diagnostics["wavlm_learning_rate"]
            downstream_lr = fine_tuning_diagnostics["downstream_learning_rate"]
            wavlm_lr_text = "none" if wavlm_lr is None else f"{float(wavlm_lr):.2e}"
            downstream_lr_text = (
                "none"
                if downstream_lr is None
                else f"{float(downstream_lr):.2e}"
            )
            speech_classifier = model.batch_scheduler.speech_classifier
            if not isinstance(
                speech_classifier,
                LightweightNoiseConditionedSpeechClassifier,
            ):
                raise RuntimeError(
                    "lightweight diagnostics branch changed unexpectedly."
                )
            top_n = speech_classifier.wavlm_encoder.unfreeze_last_n_layers
            logger.emit(
                f"WavLM Top-{top_n} | wavlm_lr={wavlm_lr_text} | "
                f"downstream_lr={downstream_lr_text} | "
                "H9-H12=["
                + ",".join(
                    f"{float(emotion_weights[name]):.4f}"
                    for name in ("H9", "H10", "H11", "H12")
                )
                + "]"
            )
        gate_diagnostics = validation_result.fusion_diagnostics
        if "shared_speech_weight_both" in gate_diagnostics:
            logger.emit(
                "Fusion | "
                f"shared=[speech "
                f"{gate_diagnostics['shared_speech_weight_both']:.3f}, "
                f"physiology "
                f"{gate_diagnostics['shared_physiology_weight_both']:.3f}] "
                f"predicted_low=[arousal "
                f"{gate_diagnostics['arousal_predicted_low_fraction']:.3f}, "
                f"valence "
                f"{gate_diagnostics['valence_predicted_low_fraction']:.3f}]"
            )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            stop_reason = "early_stopping"
            logger.emit(
                f"Early stopping | no selection improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    best_path = fold_directory / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("training completed without producing best.pt.")
    final_parameter_diagnostics = (
        _parameter_update_diagnostics(model, initial_parameter_snapshot)
        if initial_parameter_snapshot is not None
        else None
    )
    training_finished_utc = datetime.now(UTC).isoformat()
    training_duration = perf_counter() - started_clock
    training_summary = {
        **_training_summary_base(
            status="training_complete",
            fold_index=fold_index,
            started_utc=started_utc,
            best_checkpoint=fold_relative / "best.pt",
        ),
        "training_finished_utc": training_finished_utc,
        "training_duration_seconds": training_duration,
        "stop_reason": stop_reason,
        "completed_epochs": state.completed_epochs,
        "global_optimizer_steps": state.global_optimizer_steps,
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": (
            config.training.checkpoint_selection_metric
        ),
        "best_selection_value": best_selection_value,
        "best_checkpoint_validation_loss": best_checkpoint_validation_loss,
        "minimum_validation_loss": state.best_validation_loss,
        "class_weight_power": config.loss.class_weight_power,
        "threshold_calibration_enabled": (
            config.training.threshold_calibration_enabled
        ),
        "threshold_calibration_shrinkage": (
            config.training.threshold_calibration_shrinkage
        ),
        "training_modality_mode": config.training.modality_mode.value,
        "binary_decision_thresholds": (
            best_binary_thresholds.to_metadata()
            if best_binary_thresholds is not None
            else BinaryDecisionThresholds().to_metadata()
        ),
        "lr_scheduler": {
            "enabled": config.training.lr_scheduler_enabled,
            "factor": config.training.lr_scheduler_factor,
            "patience": config.training.lr_scheduler_patience,
            "min_learning_rate": config.training.min_learning_rate,
            "min_wavlm_learning_rate": (
                config.training.min_wavlm_learning_rate
            ),
        },
        "physiology_modality_dropout": (
            config.training.physiology_modality_dropout
        ),
        "speech_modality_dropout": (
            config.training.speech_modality_dropout
        ),
        "sampling_policy": config.training.sampling_policy,
        "valence_low_sampling_mass": (
            config.training.valence_low_sampling_mass
        ),
        "history": (
            config.paths.output_dir / f"fold_{fold_index}" / "history.jsonl"
        ).as_posix(),
    }
    if overfit_selection is not None:
        training_summary["overfit_diagnostic"] = {
            "enabled": True,
            "same_subset_train_eval": True,
            "selection": dict(overfit_selection),
            "accuracy_target": _OVERFIT_ACCURACY_TARGET,
            "macro_f1_target": _OVERFIT_MACRO_F1_TARGET,
            "final_parameter_diagnostics": final_parameter_diagnostics,
        }
    write_json_artifact(
        fold_directory / "training_summary.json",
        training_summary,
    )

    del train_loader, validation_loader, train_dataset
    if overfit_sample_count is None:
        del validation_dataset
    logger.emit(f"Testing best checkpoint from epoch {best_epoch}...")
    loaded_best = load_multimodal_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        restore_rng_state=False,
    )
    best_fine_tuning_diagnostics = _lightweight_fine_tuning_diagnostics(
        model,
        optimizer,
    )
    if best_fine_tuning_diagnostics is not None:
        training_summary["best_emotion_layer_weights"] = (
            best_fine_tuning_diagnostics["emotion_layer_weights"]
        )
    loaded_threshold_metadata = loaded_best.extra_metadata.get(
        "binary_decision_thresholds"
    )
    if loaded_threshold_metadata is None:
        if config.training.threshold_calibration_enabled:
            raise ValueError(
                "best checkpoint lacks calibrated binary decision thresholds."
            )
        best_binary_thresholds = BinaryDecisionThresholds()
    else:
        best_binary_thresholds = BinaryDecisionThresholds.from_metadata(
            loaded_threshold_metadata
        )
    checkpoint_relative = (
        config.paths.output_dir / f"fold_{fold_index}" / "best.pt"
    )
    if overfit_samples is not None:
        diagnostic_loader = _loader(
            overfit_samples,
            batch_size=config.training.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            seed=config.training.seed + 20_000 + fold_index,
        )
        diagnostic_batches: Iterable[AlignedMultimodalBatch] = tqdm(
            diagnostic_loader,
            total=len(diagnostic_loader),
            desc=f"Fold {fold_index} overfit evaluation",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.25,
        )
        diagnostic_speech_statistics = (
            FullWindowSpeechAvailabilityStatistics()
            if report_full_window_speech
            else None
        )
        if diagnostic_speech_statistics is not None:
            diagnostic_batches = _track_speech_availability(
                diagnostic_batches,
                diagnostic_speech_statistics,
            )
        diagnostic_result = evaluate_participant_independent(
            model,
            objective,
            diagnostic_batches,
            scope=ParticipantEvaluationScope(
                split,
                DatasetPartition.TRAIN,
                require_all_partition_participants=False,
            ),
            class_weights=class_weights,
            binary_thresholds=BinaryDecisionThresholds(),
        )
        diagnostic_summary = build_evaluation_summary(
            diagnostic_result,
            fold_index=fold_index,
            checkpoint=checkpoint_relative,
        )
        diagnostic_summary["diagnostic_mode"] = "same_training_subset"
        diagnostic_summary["training_modality_mode"] = (
            config.training.modality_mode.value
        )
        diagnostic_summary["selection"] = dict(overfit_selection or {})
        if diagnostic_speech_statistics is not None:
            diagnostic_summary["full_window_speech_statistics"] = (
                diagnostic_speech_statistics.to_summary()
            )
        write_json_artifact(
            fold_directory / "overfit_metrics.json",
            diagnostic_summary,
        )
        arousal_accuracy, arousal_f1, valence_accuracy, valence_f1 = (
            _key_test_metrics(diagnostic_summary)
        )
        overfit_success = all(
            (
                arousal_accuracy >= _OVERFIT_ACCURACY_TARGET,
                arousal_f1 >= _OVERFIT_MACRO_F1_TARGET,
                valence_accuracy >= _OVERFIT_ACCURACY_TARGET,
                valence_f1 >= _OVERFIT_MACRO_F1_TARGET,
            )
        )
        logger.emit(
            "Overfit result | "
            f"arousal acc={arousal_accuracy:.4f} f1={arousal_f1:.4f} | "
            f"valence acc={valence_accuracy:.4f} f1={valence_f1:.4f} | "
            f"success={overfit_success}"
        )
        overfit_ablation_key_metrics: dict[str, object] | None = None
        if config.training.modality_mode is KEmoConModalityMode.MULTIMODAL:
            logger.emit(
                "Running forced same-subset full/speech/physiology ablations..."
            )
            overfit_ablation_summary = _evaluate_same_sample_ablation(
                model=model,
                objective=objective,
                evaluation_loader=diagnostic_loader,
                split=split,
                partition=DatasetPartition.TRAIN,
                class_weights=class_weights,
                binary_thresholds=BinaryDecisionThresholds(),
                fold_index=fold_index,
                checkpoint_relative=checkpoint_relative,
                description_prefix=f"Fold {fold_index} overfit ablation",
            )
            overfit_ablation_summary["diagnostic_mode"] = (
                "same_training_subset_forced_modality_ablation"
            )
            overfit_ablation_summary["training_modality_mode"] = (
                config.training.modality_mode.value
            )
            overfit_ablation_summary["selection"] = dict(
                overfit_selection or {}
            )
            write_json_artifact(
                fold_directory / "overfit_ablation_metrics.json",
                overfit_ablation_summary,
            )
            ablation_modes = overfit_ablation_summary["modes"]
            if not isinstance(ablation_modes, Mapping):
                raise RuntimeError("overfit ablation modes section is invalid.")
            overfit_ablation_key_metrics = {}
            for mode in ModalityAblationMode:
                mode_summary = ablation_modes[mode.value]
                if not isinstance(mode_summary, Mapping):
                    raise RuntimeError(
                        f"overfit ablation summary for {mode.value} is invalid."
                    )
                (
                    mode_arousal_accuracy,
                    mode_arousal_f1,
                    mode_valence_accuracy,
                    mode_valence_f1,
                ) = _key_test_metrics(mode_summary)
                overfit_ablation_key_metrics[mode.value] = {
                    "arousal_accuracy": mode_arousal_accuracy,
                    "arousal_macro_f1": mode_arousal_f1,
                    "valence_accuracy": mode_valence_accuracy,
                    "valence_macro_f1": mode_valence_f1,
                }
                logger.emit(
                    f"Overfit ablation {mode.value} | "
                    f"arousal acc={mode_arousal_accuracy:.4f} "
                    f"f1={mode_arousal_f1:.4f} | "
                    f"valence acc={mode_valence_accuracy:.4f} "
                    f"f1={mode_valence_f1:.4f}"
                )
        training_summary.update(
            {
                "status": "complete",
                "finished_utc": datetime.now(UTC).isoformat(),
                "total_duration_seconds": perf_counter() - started_clock,
                "overfit_metrics": checkpoint_relative.with_name(
                    "overfit_metrics.json"
                ).as_posix(),
                "overfit_success": overfit_success,
                "overfit_modality_mode": config.training.modality_mode.value,
                "overfit_key_metrics": {
                    "arousal_accuracy": arousal_accuracy,
                    "arousal_macro_f1": arousal_f1,
                    "valence_accuracy": valence_accuracy,
                    "valence_macro_f1": valence_f1,
                },
                "test_evaluation_performed": False,
            }
        )
        if overfit_ablation_key_metrics is not None:
            training_summary.update(
                {
                    "overfit_ablation_metrics": checkpoint_relative.with_name(
                        "overfit_ablation_metrics.json"
                    ).as_posix(),
                    "overfit_ablation_key_metrics": (
                        overfit_ablation_key_metrics
                    ),
                }
            )
        write_json_artifact(
            fold_directory / "training_summary.json",
            training_summary,
        )
        logger.emit(
            "Complete | diagnostic artifacts="
            f"{fold_directory.relative_to(_PROJECT_ROOT)}"
        )
        return
    test_loader = _loader(
        test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        shuffle=False,
        seed=config.training.seed + 20_000 + fold_index,
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
    test_speech_statistics = (
        FullWindowSpeechAvailabilityStatistics()
        if report_full_window_speech
        else None
    )
    if test_speech_statistics is not None:
        test_batches = _track_speech_availability(
            test_batches,
            test_speech_statistics,
        )
    test_result = evaluate_participant_independent(
        model,
        objective,
        test_batches,
        scope=ParticipantEvaluationScope(split, DatasetPartition.TEST),
        class_weights=class_weights,
        binary_thresholds=best_binary_thresholds,
    )
    test_summary = build_evaluation_summary(
        test_result,
        fold_index=fold_index,
        checkpoint=checkpoint_relative,
    )
    test_summary["training_modality_mode"] = (
        config.training.modality_mode.value
    )
    if test_speech_statistics is not None:
        test_summary["full_window_speech_statistics"] = (
            test_speech_statistics.to_summary()
        )
    write_json_artifact(
        fold_directory / "test_metrics.json",
        test_summary,
    )
    arousal_accuracy, arousal_f1, valence_accuracy, valence_f1 = (
        _key_test_metrics(test_summary)
    )
    logger.emit(
        f"Test | arousal acc={arousal_accuracy:.4f} f1={arousal_f1:.4f} | "
        f"valence acc={valence_accuracy:.4f} f1={valence_f1:.4f}"
    )
    logger.emit(
        "Decision thresholds | "
        f"arousal_high={best_binary_thresholds.arousal_high:.2f} "
        f"valence_high={best_binary_thresholds.valence_high:.2f}"
    )
    if test_speech_statistics is not None:
        logger.emit(
            "Test full-window speech | "
            + " ".join(
                f"{name}={value}"
                for name, value in test_speech_statistics.to_summary().items()
            )
        )
    completion: dict[str, object] = {
        "status": "complete",
        "finished_utc": datetime.now(UTC).isoformat(),
        "total_duration_seconds": perf_counter() - started_clock,
        "test_metrics": checkpoint_relative.with_name(
            "test_metrics.json"
        ).as_posix(),
        "test_key_metrics": {
            "arousal_accuracy": arousal_accuracy,
            "arousal_macro_f1": arousal_f1,
            "valence_accuracy": valence_accuracy,
            "valence_macro_f1": valence_f1,
        },
        "test_ablation_enabled": config.training.evaluate_ablation,
        "binary_decision_thresholds": best_binary_thresholds.to_metadata(),
    }
    if test_speech_statistics is not None:
        completion["test_full_window_speech_statistics"] = (
            test_speech_statistics.to_summary()
        )
    if config.training.evaluate_ablation:
        logger.emit("Running controlled same-window modality ablations...")
        ablation_summary = _evaluate_same_sample_ablation(
            model=model,
            objective=objective,
            evaluation_loader=test_loader,
            split=split,
            partition=DatasetPartition.TEST,
            class_weights=class_weights,
            binary_thresholds=best_binary_thresholds,
            fold_index=fold_index,
            checkpoint_relative=checkpoint_relative,
            description_prefix=f"Fold {fold_index} ablation",
        )
        write_json_artifact(
            fold_directory / "test_ablation_metrics.json",
            ablation_summary,
        )
        ablation_modes = ablation_summary["modes"]
        if not isinstance(ablation_modes, Mapping):
            raise RuntimeError("ablation summary modes section is invalid.")
        ablation_key_metrics: dict[str, object] = {}
        for mode in ModalityAblationMode:
            mode_summary = ablation_modes[mode.value]
            if not isinstance(mode_summary, Mapping):
                raise RuntimeError(
                    f"ablation summary for {mode.value} is invalid."
                )
            _, mode_arousal_f1, _, mode_valence_f1 = _key_test_metrics(
                mode_summary
            )
            ablation_key_metrics[mode.value] = {
                "arousal_macro_f1": mode_arousal_f1,
                "valence_macro_f1": mode_valence_f1,
            }
            logger.emit(
                f"Ablation {mode.value} | "
                f"arousal f1={mode_arousal_f1:.4f} "
                f"valence f1={mode_valence_f1:.4f}"
            )
        completion.update(
            {
                "test_ablation_metrics": checkpoint_relative.with_name(
                    "test_ablation_metrics.json"
                ).as_posix(),
                "test_ablation_key_metrics": ablation_key_metrics,
            }
        )
    else:
        logger.emit("Ablation evaluation disabled by configuration.")
    training_summary.update(completion)
    write_json_artifact(
        fold_directory / "training_summary.json",
        training_summary,
    )
    logger.emit(f"Complete | artifacts={fold_directory.relative_to(_PROJECT_ROOT)}")


def _execute_fold(
    *,
    arguments: argparse.Namespace,
    config: KEmoConExperimentConfig,
    fold_index: int,
    device: torch.device,
) -> None:
    """Execute one fold with deterministic state independent of run order."""

    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    fold_relative = config.paths.output_dir / f"fold_{fold_index}"
    fold_directory = resolve_project_relative(_PROJECT_ROOT, fold_relative)
    fold_directory.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(UTC).isoformat()
    started_clock = perf_counter()
    logger = _RunLogger(
        fold_directory / "train.log",
        append=arguments.resume is not None,
    )
    write_json_artifact(
        fold_directory / "training_summary.json",
        _training_summary_base(
            status="running",
            fold_index=fold_index,
            started_utc=started_utc,
            best_checkpoint=fold_relative / "best.pt",
        ),
    )
    try:
        _run(
            arguments=arguments,
            fold_index=fold_index,
            device=device,
            fold_directory=fold_directory,
            logger=logger,
            started_utc=started_utc,
            started_clock=started_clock,
            fold_relative=fold_relative,
        )
    except Exception as error:
        logger.emit(f"Failed | {type(error).__name__}: {error}")
        logger.write_traceback()
        write_json_artifact(
            fold_directory / "training_summary.json",
            {
                **_training_summary_base(
                    status="failed",
                    fold_index=fold_index,
                    started_utc=started_utc,
                    best_checkpoint=fold_relative / "best.pt",
                ),
                "failed_utc": datetime.now(UTC).isoformat(),
                "duration_seconds": perf_counter() - started_clock,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    finally:
        logger.close()


def _write_cross_fold_summary(
    config: KEmoConExperimentConfig,
    fold_indices: Sequence[int],
) -> None:
    """Aggregate completed fold metrics and retain availability audits."""

    output_directory = resolve_project_relative(_PROJECT_ROOT, config.paths.output_dir)
    evaluations: list[Mapping[str, object]] = []
    availability_audits: list[dict[str, object]] = []
    assigned_test_participants: list[str] = []
    all_split_participants: set[str] = set()
    for fold_index in fold_indices:
        fold_directory = output_directory / f"fold_{fold_index}"
        with (fold_directory / "test_metrics.json").open(encoding="utf-8") as stream:
            evaluation = json.load(stream)
        if not isinstance(evaluation, dict):
            raise TypeError("test_metrics.json must contain a JSON object.")
        evaluations.append(evaluation)
        with (fold_directory / "split_summary.json").open(encoding="utf-8") as stream:
            split_summary = json.load(stream)
        if not isinstance(split_summary, dict):
            raise TypeError("split_summary.json must contain a JSON object.")
        partitions = split_summary.get("partitions")
        if not isinstance(partitions, dict):
            raise TypeError("split_summary.json must contain partition mappings.")
        for partition_name in ("train", "validation", "test"):
            partition = partitions.get(partition_name)
            if not isinstance(partition, dict):
                raise TypeError("every split partition must be a JSON object.")
            participant_ids = partition.get("participant_ids")
            if not isinstance(participant_ids, list) or not all(
                isinstance(value, str) for value in participant_ids
            ):
                raise TypeError("split participant_ids must be a list of strings.")
            all_split_participants.update(cast(list[str], participant_ids))
            if partition_name == "test":
                overlap = set(assigned_test_participants) & set(participant_ids)
                if overlap:
                    raise ValueError(
                        "test participants are assigned to multiple folds: "
                        f"{sorted(overlap)}."
                    )
                assigned_test_participants.extend(cast(list[str], participant_ids))
        availability_audits.append(
            {
                "fold_index": fold_index,
                "partitions": split_summary.get(
                    "runtime_modality_availability_audit",
                    {},
                ),
            }
        )
    summary = build_cross_fold_evaluation_summary(evaluations)
    summary["split_strategy"] = config.split.strategy
    summary["split_seed"] = config.split.seed
    complete_coverage = (
        set(assigned_test_participants) == all_split_participants
        and len(assigned_test_participants) == len(all_split_participants)
    )
    summary["split_coverage"] = {
        "participant_count": len(all_split_participants),
        "test_assignment_count": len(assigned_test_participants),
        "each_participant_tested_once": complete_coverage,
        "assigned_test_participant_ids": assigned_test_participants,
    }
    if not complete_coverage:
        raise ValueError("completed folds do not test every split participant once.")
    summary["runtime_modality_availability_audits"] = availability_audits
    write_json_artifact(output_directory / "cross_fold_summary.json", summary)


def main() -> None:
    """Run one fold or the complete repeated dyad-safe experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=_relative_cli_path,
        default=Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
    )
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument("--fold", type=int)
    fold_group.add_argument(
        "--all-folds",
        action="store_true",
        help="run every configured rotating fold and write cross_fold_summary.json",
    )
    parser.add_argument("--resume", type=_relative_cli_path)
    parser.add_argument(
        "--overfit-samples",
        type=_overfit_sample_count,
        help=(
            "cache a four-quadrant-balanced multimodal training subset and "
            "evaluate on identical samples"
        ),
    )
    parser.add_argument(
        "--overfit-epochs",
        type=_positive_epoch_count,
        default=_OVERFIT_DEFAULT_EPOCHS,
        help="diagnostic epochs used with --overfit-samples (default: 200)",
    )
    arguments = parser.parse_args()
    if arguments.overfit_samples is not None and arguments.resume is not None:
        parser.error("--overfit-samples cannot be combined with --resume.")
    if arguments.all_folds and arguments.resume is not None:
        parser.error("--all-folds cannot be combined with --resume.")
    if arguments.all_folds and arguments.overfit_samples is not None:
        parser.error("--all-folds cannot be combined with --overfit-samples.")
    config = load_kemocon_experiment_config(
        resolve_project_relative(_PROJECT_ROOT, arguments.config)
    )
    if arguments.overfit_samples is not None:
        config = _configure_overfit_diagnostic(
            config,
            sample_count=arguments.overfit_samples,
            epochs=arguments.overfit_epochs,
            modality_mode=KEmoConModalityMode.MULTIMODAL,
        )
    selected_fold = (
        config.split.fold_index if arguments.fold is None else arguments.fold
    )
    fold_indices = (
        tuple(range(config.split.run_count))
        if arguments.all_folds
        else (selected_fold,)
    )
    if any(not 0 <= fold_index < config.split.run_count for fold_index in fold_indices):
        raise ValueError("fold index lies outside configured runs.")
    device = _device(config.training.device)
    for fold_index in fold_indices:
        _execute_fold(
            arguments=arguments,
            config=config,
            fold_index=fold_index,
            device=device,
        )
    if arguments.all_folds:
        _write_cross_fold_summary(config, fold_indices)


if __name__ == "__main__":
    main()
