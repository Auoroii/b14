"""Configuration and assembly for local K-EmoCon experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]
from torch import Tensor
from transformers import WavLMModel

from emotion_model.common import LabelProtocol, binarize_emotion_scores
from emotion_model.data import (
    AlignedMultimodalDataset,
    KEmoConPhysioAdapter,
    KEmoConSpeechAdapter,
    MultimodalManifest,
    MultimodalWindowRecord,
    ParticipantSplit,
    SpeechActivityDetectorConfig,
    build_kemocon_dyad_ratio_split,
    build_kemocon_dyad_splits,
    build_kemocon_label_stratified_dyad_splits,
)
from emotion_model.multimodal import (
    FullWindowDynamicMultimodalFusion,
    MultimodalBatchScheduler,
    MultimodalEmotionClassifier,
)
from emotion_model.physiology import (
    ChannelwiseZScoreNormalizer,
    LightweightPhysioEmotionClassifier,
    NormalizationFitScope,
    PhysioChannelSpec,
)
from emotion_model.speech import (
    EmotionLayerAggregation,
    LightweightNoiseConditionedSpeechClassifier,
    NoiseConditionedRelationDifferentialDenoiser,
    WavLMEncoder,
)
from emotion_model.training import (
    ClassificationLossKind,
    EmotionTaskClassWeights,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
)

_MODEL_VARIANT = (
    "lightweight_shared_dynamic_relation_differential_full_window"
)


@dataclass(frozen=True)
class KEmoConPaths:
    """Project-relative paths used by one K-EmoCon experiment."""

    data_root: Path
    manifest: Path
    wavlm_model: Path
    output_dir: Path


@dataclass(frozen=True)
class KEmoConDatasetConfig:
    """K-EmoCon windowing, channel, alignment, and labeling settings."""

    annotation_perspective: str
    channel_names: tuple[str, ...]
    window_seconds: float
    speech_sample_rate_hz: int
    physio_target_sample_rate_hz: float
    label_protocol: LabelProtocol
    speech_activity: SpeechActivityDetectorConfig = SpeechActivityDetectorConfig()


@dataclass(frozen=True)
class KEmoConSplitConfig:
    """Dyad-aware random, label-stratified, or one-shot split settings."""

    strategy: str
    num_folds: int | None
    fold_index: int
    train_fraction: float | None
    validation_fraction: float | None
    test_fraction: float | None
    seed: int

    @property
    def run_count(self) -> int:
        """Return the number of legal CLI fold indices."""

        return self.num_folds if self.num_folds is not None else 1


@dataclass(frozen=True)
class KEmoConModelConfig:
    """V4.2 full-window relation-differential model parameters."""

    speech_embedding_dim: int
    noise_embedding_dim: int
    physiology_stem_channels: tuple[int, int]
    physiology_embedding_dim: int
    fusion_dim: int
    classifier_hidden_dim: int
    film_scale: float
    dropout: float
    freeze_wavlm: bool
    unfreeze_last_n_layers: int
    emotion_layer_aggregation: str = "fixed_mean"
    fusion_gate_hidden_dim: int = 32
    differential_dim: int = 32
    differential_heads: int = 4
    differential_lambda_init: float = 0.5
    differential_residual_scale: float = 0.1
    condition_gate_on_noise: bool = True
    variant: str = _MODEL_VARIANT


@dataclass(frozen=True)
class KEmoConLossConfig:
    """Classification loss kind and fused/unimodal scalar coefficients."""

    kind: ClassificationLossKind
    focal_gamma: float
    fused_weight: float
    speech_aux_weight: float
    physiology_aux_weight: float
    class_weight_power: float = 1.0


class KEmoConModalityMode(StrEnum):
    """The only supported V4.2 training mode."""

    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class KEmoConTrainingConfig:
    """Server training controls sized by default for a 24 GB RTX 4090."""

    device: str
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    epochs: int
    learning_rate: float
    wavlm_learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    early_stopping_patience: int
    checkpoint_selection_metric: str
    lr_scheduler_enabled: bool
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    min_learning_rate: float
    speech_modality_dropout: float
    physiology_modality_dropout: float
    sampling_policy: str
    valence_low_sampling_mass: float
    threshold_calibration_enabled: bool
    threshold_calibration_shrinkage: float
    evaluate_ablation: bool
    modality_mode: KEmoConModalityMode
    min_wavlm_learning_rate: float | None = None


@dataclass(frozen=True)
class KEmoConExperimentConfig:
    """Fully parsed K-EmoCon experiment configuration."""

    paths: KEmoConPaths
    dataset: KEmoConDatasetConfig
    split: KEmoConSplitConfig
    model: KEmoConModelConfig
    loss: KEmoConLossConfig
    training: KEmoConTrainingConfig


_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "paths": frozenset({"data_root", "manifest", "wavlm_model", "output_dir"}),
    "dataset": frozenset(
        {
            "annotation_perspective",
            "channels",
            "window_seconds",
            "speech_sample_rate_hz",
            "physio_target_sample_rate_hz",
            "label_protocol",
            "speech_activity",
        }
    ),
    "split": frozenset(
        {
            "strategy",
            "num_folds",
            "fold_index",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
            "seed",
        }
    ),
    "loss": frozenset(
        {
            "kind",
            "focal_gamma",
            "fused_weight",
            "speech_aux_weight",
            "physiology_aux_weight",
            "class_weight_power",
        }
    ),
    "training": frozenset(
        {
            "device",
            "seed",
            "batch_size",
            "num_workers",
            "pin_memory",
            "epochs",
            "learning_rate",
            "wavlm_learning_rate",
            "weight_decay",
            "max_gradient_norm",
            "early_stopping_patience",
            "checkpoint_selection_metric",
            "lr_scheduler_enabled",
            "lr_scheduler_factor",
            "lr_scheduler_patience",
            "min_learning_rate",
            "min_wavlm_learning_rate",
            "speech_modality_dropout",
            "physiology_modality_dropout",
            "sampling_policy",
            "valence_low_sampling_mass",
            "threshold_calibration_enabled",
            "threshold_calibration_shrinkage",
            "evaluate_ablation",
            "modality_mode",
        }
    ),
}
_MODEL_FIELDS = frozenset(
    {
        "variant",
        "speech_embedding_dim",
        "noise_embedding_dim",
        "physiology_stem_channels",
        "physiology_embedding_dim",
        "fusion_dim",
        "classifier_hidden_dim",
        "film_scale",
        "fusion_gate_hidden_dim",
        "dropout",
        "freeze_wavlm",
        "unfreeze_last_n_layers",
        "emotion_layer_aggregation",
        "differential_dim",
        "differential_heads",
        "differential_lambda_init",
        "differential_residual_scale",
        "condition_gate_on_noise",
    }
)


def _reject_unknown(
    section: Mapping[str, object],
    allowed: frozenset[str],
    *,
    name: str,
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}.")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return cast(Mapping[str, object], value)


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    if name not in config:
        raise ValueError(f"configuration section {name!r} is required.")
    return _mapping(config[name], name=name)


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    return value


def _real(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _optional_integer(
    section: Mapping[str, object],
    name: str,
) -> int | None:
    if name not in section:
        return None
    return _integer(section, name)


def _optional_real(
    section: Mapping[str, object],
    name: str,
) -> float | None:
    if name not in section:
        return None
    return _real(section, name)


def _boolean(section: Mapping[str, object], name: str) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _integer_pair(section: Mapping[str, object], name: str) -> tuple[int, int]:
    value = section.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of two integers.")
    items = tuple(value)
    if len(items) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in items
    ):
        raise TypeError(f"{name} must contain exactly two integers, not bool.")
    return cast(tuple[int, int], items)


def _parse_model_config(
    model: Mapping[str, object],
) -> KEmoConModelConfig:
    raw_variant = model.get("variant")
    if not isinstance(raw_variant, str) or not raw_variant.strip():
        raise ValueError("model.variant must be a non-empty string.")
    if raw_variant != _MODEL_VARIANT:
        raise ValueError(
            f"model.variant must be {_MODEL_VARIANT!r}; received {raw_variant!r}."
        )
    _reject_unknown(model, _MODEL_FIELDS, name="model")
    return KEmoConModelConfig(
        speech_embedding_dim=_integer(model, "speech_embedding_dim"),
        noise_embedding_dim=_integer(model, "noise_embedding_dim"),
        physiology_stem_channels=_integer_pair(
            model,
            "physiology_stem_channels",
        ),
        physiology_embedding_dim=_integer(
            model,
            "physiology_embedding_dim",
        ),
        fusion_dim=_integer(model, "fusion_dim"),
        classifier_hidden_dim=_integer(model, "classifier_hidden_dim"),
        film_scale=_real(model, "film_scale"),
        dropout=_real(model, "dropout"),
        freeze_wavlm=_boolean(model, "freeze_wavlm"),
        unfreeze_last_n_layers=_integer(model, "unfreeze_last_n_layers"),
        emotion_layer_aggregation=(
            _string(model, "emotion_layer_aggregation")
            if "emotion_layer_aggregation" in model
            else "fixed_mean"
        ),
        fusion_gate_hidden_dim=(
            _integer(model, "fusion_gate_hidden_dim")
            if "fusion_gate_hidden_dim" in model
            else 32
        ),
        differential_dim=_integer(model, "differential_dim"),
        differential_heads=_integer(model, "differential_heads"),
        differential_lambda_init=_real(model, "differential_lambda_init"),
        differential_residual_scale=_real(
            model,
            "differential_residual_scale",
        ),
        condition_gate_on_noise=_boolean(model, "condition_gate_on_noise"),
        variant=raw_variant,
    )


def _relative_path(section: Mapping[str, object], name: str) -> Path:
    path = Path(_string(section, name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a project-relative path without '..'.")
    return path


def resolve_project_relative(project_root: Path, relative_path: Path) -> Path:
    """Resolve one validated relative path below the project root.

    Args:
        project_root: Existing project directory.
        relative_path: Non-absolute path without parent traversal.

    Returns:
        Resolved filesystem path guaranteed to remain under ``project_root``.
    """

    if not isinstance(project_root, Path) or not project_root.is_dir():
        raise ValueError("project_root must be an existing directory Path.")
    if (
        not isinstance(relative_path, Path)
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError("relative_path must remain project-relative.")
    root = project_root.resolve()
    resolved = root.joinpath(relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("relative_path resolves outside project_root.") from error
    return resolved


def build_configured_kemocon_split(
    manifest: MultimodalManifest,
    config: KEmoConSplitConfig,
    *,
    fold_index: int,
    label_protocol: LabelProtocol | None = None,
) -> ParticipantSplit:
    """Build the configured dyad-safe split for one experiment run.

    Args:
        manifest: Complete K-EmoCon manifest.
        config: Validated rotating-fold, stratified-fold, or ratio settings.
        fold_index: Selected rotating fold, or exactly zero for fixed ratio.
        label_protocol: Required named label mapping for label-stratified folds.

    Returns:
        One participant split containing complete debate dyads.
    """

    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be MultimodalManifest.")
    if not isinstance(config, KEmoConSplitConfig):
        raise TypeError("config must be KEmoConSplitConfig.")
    if isinstance(fold_index, bool) or not isinstance(fold_index, int):
        raise TypeError("fold_index must be an integer, not bool.")
    if not 0 <= fold_index < config.run_count:
        raise ValueError("fold index lies outside configured runs.")
    if config.strategy == "rotating_folds":
        if config.num_folds is None:
            raise RuntimeError("validated rotating split lacks num_folds.")
        return build_kemocon_dyad_splits(
            manifest,
            num_folds=config.num_folds,
            seed=config.seed,
        )[fold_index]
    if config.strategy == "label_stratified_rotating_folds":
        if config.num_folds is None:
            raise RuntimeError("validated stratified split lacks num_folds.")
        if not isinstance(label_protocol, LabelProtocol):
            raise TypeError(
                "label_protocol must be LabelProtocol for label-stratified folds."
            )
        return build_kemocon_label_stratified_dyad_splits(
            manifest,
            label_protocol=label_protocol,
            num_folds=config.num_folds,
            seed=config.seed,
        )[fold_index]
    if config.strategy != "fixed_ratio":
        raise ValueError("unsupported split strategy.")
    fractions = (
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )
    if any(value is None for value in fractions):
        raise RuntimeError("validated fixed-ratio split lacks fractions.")
    train, validation, test = cast(tuple[float, float, float], fractions)
    return build_kemocon_dyad_ratio_split(
        manifest,
        train_fraction=train,
        validation_fraction=validation,
        test_fraction=test,
        seed=config.seed,
    )


def load_kemocon_experiment_config(path: str | Path) -> KEmoConExperimentConfig:
    """Parse a YAML experiment file whose stored paths are all relative."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = _mapping(raw, name="configuration")
    _reject_unknown(
        config,
        frozenset(
            {
                "paths",
                "dataset",
                "split",
                "model",
                "loss",
                "training",
            }
        ),
        name="configuration",
    )
    paths = _section(config, "paths")
    dataset = _section(config, "dataset")
    split = _section(config, "split")
    model = _section(config, "model")
    loss = _section(config, "loss")
    training = _section(config, "training")
    for section_name, section_value in (
        ("paths", paths),
        ("dataset", dataset),
        ("split", split),
        ("loss", loss),
        ("training", training),
    ):
        _reject_unknown(
            section_value,
            _SECTION_FIELDS[section_name],
            name=section_name,
        )

    speech_activity = _mapping(
        dataset.get("speech_activity", {}),
        name="dataset.speech_activity",
    )
    _reject_unknown(
        speech_activity,
        frozenset(
            {
                "enabled",
                "frame_ms",
                "hop_ms",
                "rms_threshold",
                "availability_policy",
            }
        ),
        name="dataset.speech_activity",
    )

    channels = dataset.get("channels")
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise TypeError("dataset.channels must be a non-string sequence.")
    channel_names = tuple(channels)
    if not channel_names or not all(
        isinstance(value, str) and value.strip() for value in channel_names
    ):
        raise ValueError("dataset.channels must contain non-empty strings.")

    parsed = KEmoConExperimentConfig(
        paths=KEmoConPaths(
            data_root=_relative_path(paths, "data_root"),
            manifest=_relative_path(paths, "manifest"),
            wavlm_model=_relative_path(paths, "wavlm_model"),
            output_dir=_relative_path(paths, "output_dir"),
        ),
        dataset=KEmoConDatasetConfig(
            annotation_perspective=_string(
                dataset,
                "annotation_perspective",
            ),
            channel_names=cast(tuple[str, ...], channel_names),
            window_seconds=_real(dataset, "window_seconds"),
            speech_sample_rate_hz=_integer(
                dataset,
                "speech_sample_rate_hz",
            ),
            physio_target_sample_rate_hz=_real(
                dataset,
                "physio_target_sample_rate_hz",
            ),
            label_protocol=LabelProtocol(_string(dataset, "label_protocol")),
            speech_activity=SpeechActivityDetectorConfig(
                enabled=(
                    _boolean(speech_activity, "enabled")
                    if "enabled" in speech_activity
                    else False
                ),
                frame_ms=(
                    _real(speech_activity, "frame_ms")
                    if "frame_ms" in speech_activity
                    else 20.0
                ),
                hop_ms=(
                    _real(speech_activity, "hop_ms")
                    if "hop_ms" in speech_activity
                    else 10.0
                ),
                rms_threshold=(
                    _real(speech_activity, "rms_threshold")
                    if "rms_threshold" in speech_activity
                    else 1.0e-4
                ),
                availability_policy=(
                    _string(speech_activity, "availability_policy")
                    if "availability_policy" in speech_activity
                    else "source_presence"
                ),
            ),
        ),
        split=KEmoConSplitConfig(
            strategy=(
                _string(split, "strategy")
                if "strategy" in split
                else "rotating_folds"
            ),
            num_folds=_optional_integer(split, "num_folds"),
            fold_index=_integer(split, "fold_index"),
            train_fraction=_optional_real(split, "train_fraction"),
            validation_fraction=_optional_real(
                split,
                "validation_fraction",
            ),
            test_fraction=_optional_real(split, "test_fraction"),
            seed=_integer(split, "seed"),
        ),
        model=_parse_model_config(model),
        loss=KEmoConLossConfig(
            kind=ClassificationLossKind(_string(loss, "kind")),
            focal_gamma=_real(loss, "focal_gamma"),
            fused_weight=_real(loss, "fused_weight"),
            speech_aux_weight=_real(loss, "speech_aux_weight"),
            physiology_aux_weight=_real(
                loss,
                "physiology_aux_weight",
            ),
            class_weight_power=(
                _real(loss, "class_weight_power")
                if "class_weight_power" in loss
                else 1.0
            ),
        ),
        training=KEmoConTrainingConfig(
            device=_string(training, "device"),
            seed=_integer(training, "seed"),
            batch_size=_integer(training, "batch_size"),
            num_workers=_integer(training, "num_workers"),
            pin_memory=_boolean(training, "pin_memory"),
            epochs=_integer(training, "epochs"),
            learning_rate=_real(training, "learning_rate"),
            wavlm_learning_rate=(
                _real(training, "wavlm_learning_rate")
                if "wavlm_learning_rate" in training
                else _real(training, "learning_rate")
            ),
            weight_decay=_real(training, "weight_decay"),
            max_gradient_norm=_real(training, "max_gradient_norm"),
            early_stopping_patience=_integer(
                training,
                "early_stopping_patience",
            ),
            checkpoint_selection_metric=_string(
                training,
                "checkpoint_selection_metric",
            ),
            lr_scheduler_enabled=_boolean(
                training,
                "lr_scheduler_enabled",
            ),
            lr_scheduler_factor=_real(
                training,
                "lr_scheduler_factor",
            ),
            lr_scheduler_patience=_integer(
                training,
                "lr_scheduler_patience",
            ),
            min_learning_rate=_real(
                training,
                "min_learning_rate",
            ),
            min_wavlm_learning_rate=_optional_real(
                training,
                "min_wavlm_learning_rate",
            ),
            speech_modality_dropout=(
                _real(training, "speech_modality_dropout")
                if "speech_modality_dropout" in training
                else 0.0
            ),
            physiology_modality_dropout=_real(
                training,
                "physiology_modality_dropout",
            ),
            sampling_policy=_string(training, "sampling_policy"),
            valence_low_sampling_mass=_real(
                training,
                "valence_low_sampling_mass",
            ),
            threshold_calibration_enabled=(
                _boolean(training, "threshold_calibration_enabled")
                if "threshold_calibration_enabled" in training
                else False
            ),
            threshold_calibration_shrinkage=(
                _real(training, "threshold_calibration_shrinkage")
                if "threshold_calibration_shrinkage" in training
                else 1.0
            ),
            evaluate_ablation=(
                _boolean(training, "evaluate_ablation")
                if "evaluate_ablation" in training
                else True
            ),
            modality_mode=KEmoConModalityMode(
                _string(training, "modality_mode")
                if "modality_mode" in training
                else KEmoConModalityMode.MULTIMODAL.value
            ),
        ),
    )
    _validate_config(parsed)
    return parsed


def _validate_config(config: KEmoConExperimentConfig) -> None:
    if config.model.variant != _MODEL_VARIANT:
        raise ValueError(f"model.variant must be {_MODEL_VARIANT!r}.")
    positive_integers = {
        "model.fusion_dim": config.model.fusion_dim,
        "model.classifier_hidden_dim": config.model.classifier_hidden_dim,
        "model.speech_embedding_dim": config.model.speech_embedding_dim,
        "model.noise_embedding_dim": config.model.noise_embedding_dim,
        "model.physiology_stem_channels[0]": (
            config.model.physiology_stem_channels[0]
        ),
        "model.physiology_stem_channels[1]": (
            config.model.physiology_stem_channels[1]
        ),
        "model.physiology_embedding_dim": config.model.physiology_embedding_dim,
        "model.fusion_gate_hidden_dim": config.model.fusion_gate_hidden_dim,
        "model.differential_dim": config.model.differential_dim,
        "model.differential_heads": config.model.differential_heads,
        "training.batch_size": config.training.batch_size,
        "training.epochs": config.training.epochs,
        "training.early_stopping_patience": (
            config.training.early_stopping_patience
        ),
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    rotating_strategies = {
        "rotating_folds",
        "label_stratified_rotating_folds",
    }
    if config.split.strategy in rotating_strategies:
        if config.split.num_folds is None:
            raise ValueError(
                "split.num_folds is required for rotating fold strategies."
            )
        if config.split.num_folds < 3:
            raise ValueError("split.num_folds must be at least 3.")
        if not 0 <= config.split.fold_index < config.split.num_folds:
            raise ValueError(
                "split.fold_index must lie within configured folds."
            )
        if any(
            value is not None
            for value in (
                config.split.train_fraction,
                config.split.validation_fraction,
                config.split.test_fraction,
            )
        ):
            raise ValueError(
                "split fractions are not allowed for rotating fold strategies."
            )
    elif config.split.strategy == "fixed_ratio":
        if config.split.num_folds is not None:
            raise ValueError(
                "split.num_folds is not allowed for fixed_ratio."
            )
        if config.split.fold_index != 0:
            raise ValueError("fixed_ratio requires split.fold_index=0.")
        fractions = (
            config.split.train_fraction,
            config.split.validation_fraction,
            config.split.test_fraction,
        )
        if any(value is None for value in fractions):
            raise ValueError(
                "fixed_ratio requires train/validation/test fractions."
            )
        numeric_fractions = cast(tuple[float, float, float], fractions)
        if any(value <= 0.0 for value in numeric_fractions):
            raise ValueError("split fractions must be positive.")
        if not math.isclose(
            sum(numeric_fractions),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("split fractions must sum to 1.")
    else:
        raise ValueError(
            "split.strategy must be 'rotating_folds', "
            "'label_stratified_rotating_folds', or 'fixed_ratio'."
        )
    if config.training.num_workers < 0:
        raise ValueError("training.num_workers must be non-negative.")
    if config.training.sampling_policy not in {
        "uniform",
        "soft_valence_class_participant_balanced",
    }:
        raise ValueError(
            "training.sampling_policy must be "
            "'soft_valence_class_participant_balanced' or 'uniform'."
        )
    if not 0.0 < config.training.valence_low_sampling_mass < 0.5:
        raise ValueError(
            "training.valence_low_sampling_mass must lie strictly between "
            "0 and 0.5."
        )
    if config.training.lr_scheduler_patience < 0:
        raise ValueError("training.lr_scheduler_patience must be non-negative.")
    if config.dataset.speech_activity.availability_policy != "source_presence":
        raise ValueError(
            "V4.2 requires dataset.speech_activity.availability_policy="
            "'source_presence'."
        )
    if not 0 <= config.model.unfreeze_last_n_layers <= 12:
        raise ValueError(
            "model.unfreeze_last_n_layers must lie in [0, 12]."
        )
    if config.model.freeze_wavlm != (config.model.unfreeze_last_n_layers == 0):
        raise ValueError(
            "model.freeze_wavlm must be true exactly when "
            "model.unfreeze_last_n_layers is 0."
        )
    if config.model.emotion_layer_aggregation not in {
        "fixed_mean",
        "learnable_weighted",
    }:
        raise ValueError(
            "lightweight model.emotion_layer_aggregation must be "
            "'fixed_mean' or 'learnable_weighted'."
        )
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must lie in [0, 1).")
    if not 0.0 < config.model.film_scale <= 1.0:
        raise ValueError("model.film_scale must lie in (0, 1].")
    if config.model.differential_dim % config.model.differential_heads != 0:
        raise ValueError(
            "model.differential_dim must be divisible by model.differential_heads."
        )
    if not 0.0 < config.model.differential_lambda_init < 1.0:
        raise ValueError("model.differential_lambda_init must lie in (0, 1).")
    if not 0.0 < config.model.differential_residual_scale <= 1.0:
        raise ValueError(
            "model.differential_residual_scale must lie in (0, 1]."
        )
    for name, positive_value in (
        ("dataset.window_seconds", config.dataset.window_seconds),
        (
            "dataset.physio_target_sample_rate_hz",
            config.dataset.physio_target_sample_rate_hz,
        ),
        ("training.learning_rate", config.training.learning_rate),
        ("training.wavlm_learning_rate", config.training.wavlm_learning_rate),
        ("training.max_gradient_norm", config.training.max_gradient_norm),
    ):
        if positive_value <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if config.training.weight_decay < 0.0:
        raise ValueError("training.weight_decay must be non-negative.")
    if not 0.0 <= config.loss.class_weight_power <= 1.0:
        raise ValueError("loss.class_weight_power must lie in [0, 1].")
    if config.training.checkpoint_selection_metric not in {
        "validation_loss",
        "mean_macro_f1",
        "participant_mean_macro_f1",
        "calibrated_participant_mean_macro_f1",
    }:
        raise ValueError(
            "training.checkpoint_selection_metric must be "
            "'validation_loss', 'mean_macro_f1', or "
            "'participant_mean_macro_f1', or "
            "'calibrated_participant_mean_macro_f1'."
        )
    if (
        config.training.checkpoint_selection_metric
        == "calibrated_participant_mean_macro_f1"
        and not config.training.threshold_calibration_enabled
    ):
        raise ValueError(
            "calibrated checkpoint selection requires "
            "training.threshold_calibration_enabled=true."
        )
    if not 0.0 <= config.training.threshold_calibration_shrinkage <= 1.0:
        raise ValueError(
            "training.threshold_calibration_shrinkage must lie in [0, 1]."
        )
    if not 0.0 < config.training.lr_scheduler_factor < 1.0:
        raise ValueError(
            "training.lr_scheduler_factor must lie strictly in (0, 1)."
        )
    shared_minimum = config.training.min_wavlm_learning_rate is None
    if config.training.min_learning_rate <= 0.0 or (
        config.training.min_learning_rate > config.training.learning_rate
        or (
            shared_minimum
            and config.training.min_learning_rate
            > config.training.wavlm_learning_rate
        )
    ):
        raise ValueError(
            "training.min_learning_rate must be positive and no greater than "
            "the learning rates to which it applies."
        )
    if config.training.min_wavlm_learning_rate is not None and (
        config.training.min_wavlm_learning_rate <= 0.0
        or config.training.min_wavlm_learning_rate
        > config.training.wavlm_learning_rate
    ):
        raise ValueError(
            "training.min_wavlm_learning_rate must be positive and no greater "
            "than training.wavlm_learning_rate."
        )
    if not 0.0 <= config.training.physiology_modality_dropout < 1.0:
        raise ValueError(
            "training.physiology_modality_dropout must lie in [0, 1)."
        )
    if not 0.0 <= config.training.speech_modality_dropout < 1.0:
        raise ValueError(
            "training.speech_modality_dropout must lie in [0, 1)."
        )
    if (
        config.training.speech_modality_dropout
        + config.training.physiology_modality_dropout
        >= 1.0
    ):
        raise ValueError(
            "speech and physiology modality dropout probabilities must sum "
            "to less than 1."
        )
    if config.training.modality_mode is not KEmoConModalityMode.MULTIMODAL:
        raise ValueError("training.modality_mode must be 'multimodal'.")


def fit_kemocon_train_normalizer(
    records: Sequence[MultimodalWindowRecord],
    channel_specs: Sequence[PhysioChannelSpec],
    adapter: KEmoConPhysioAdapter,
) -> ChannelwiseZScoreNormalizer:
    """Fit global channel z-scores from training windows only.

    Args:
        records: Non-empty training-partition records.
        channel_specs: Ordered channel semantics.
        adapter: K-EmoCon adapter returning CPU values/masks/timestamps ``[T]``.

    Returns:
        Fitted normalizer with one global key per channel. Missing values and
        channels do not contribute.
    """

    if not records:
        raise ValueError("records must be non-empty.")
    specs = tuple(channel_specs)
    if not specs:
        raise ValueError("channel_specs must be non-empty.")
    totals = {spec.name: [0.0, 0.0, 0] for spec in specs}
    spec_by_name = {spec.name: spec for spec in specs}
    for record in records:
        for source in record.physio_sources:
            spec = spec_by_name.get(source.channel_name)
            if spec is None:
                continue
            window = adapter.load_physio(source, spec)
            values = window.values[window.valid_mask].to(dtype=torch.float64)
            if values.numel() == 0:
                continue
            aggregate = totals[spec.name]
            aggregate[0] += float(values.sum().item())
            aggregate[1] += float((values * values).sum().item())
            aggregate[2] += int(values.numel())

    statistics: list[dict[str, object]] = []
    for spec in specs:
        total, square_total, count_value = totals[spec.name]
        count = int(count_value)
        if count <= 0:
            raise ValueError(
                f"training partition has no valid values for channel {spec.name!r}."
            )
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        statistics.append(
            {
                "channel_name": spec.name,
                "group_id": None,
                "mean": mean,
                "scale": max(math.sqrt(variance), 1.0e-6),
                "valid_count": count,
            }
        )
    normalizer = ChannelwiseZScoreNormalizer(epsilon=1.0e-6)
    normalizer.load_state_dict(
        {
            "version": 1,
            "epsilon": 1.0e-6,
            "fit_scope": NormalizationFitScope.TRAIN.value,
            "statistics": statistics,
        }
    )
    return normalizer


def build_kemocon_dataset(
    records: Sequence[MultimodalWindowRecord],
    channel_specs: Sequence[PhysioChannelSpec],
    *,
    data_root: Path,
    config: KEmoConDatasetConfig,
    normalizer: ChannelwiseZScoreNormalizer | None,
    modality_mode: KEmoConModalityMode = KEmoConModalityMode.MULTIMODAL,
) -> AlignedMultimodalDataset:
    """Build one lazy CPU dataset for an already selected partition.

    Speech output is ``[L]`` at 16 kHz and physiology output is ``[T, C]`` on
    the configured common alignment timeline.
    """

    if not isinstance(modality_mode, KEmoConModalityMode):
        raise TypeError("modality_mode must be KEmoConModalityMode.")
    if normalizer is None:
        raise ValueError("normalizer is required for V4.2 multimodal data.")

    return AlignedMultimodalDataset(
        records,
        channel_specs,
        label_protocol=config.label_protocol,
        speech_adapter=KEmoConSpeechAdapter(data_root),
        physio_adapter=KEmoConPhysioAdapter(data_root),
        required_speech_sample_rate_hz=config.speech_sample_rate_hz,
        physio_target_sample_rate_hz=config.physio_target_sample_rate_hz,
        physio_normalizer=normalizer,
        enable_speech=True,
        enable_physiology=True,
        speech_activity_config=config.speech_activity,
    )


def _balanced_binary_weights(
    labels: Tensor,
    *,
    device: torch.device,
    sample_weights: Tensor | None = None,
    class_weight_power: float = 1.0,
) -> Tensor:
    if isinstance(class_weight_power, bool) or not isinstance(
        class_weight_power,
        (int, float),
    ):
        raise TypeError("class_weight_power must be a real number, not bool.")
    power = float(class_weight_power)
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("class_weight_power must lie in [0, 1].")
    valid = labels >= 0
    if sample_weights is None:
        counts = torch.bincount(
            labels[valid],
            minlength=2,
        ).to(dtype=torch.float64)
    else:
        if (
            sample_weights.dtype != torch.float64
            or tuple(sample_weights.shape) != tuple(labels.shape)
        ):
            raise ValueError(
                "sample_weights must be float64 with the label shape."
            )
        counts = torch.zeros(2, dtype=torch.float64).scatter_add_(
            0,
            labels[valid],
            sample_weights[valid],
        )
    if bool((counts == 0).any()):
        raise ValueError(
            "training fold must contain both binary classes for class weighting."
        )
    weights = (counts.sum() / (2.0 * counts)).pow(power)
    return weights.to(device=device, dtype=torch.float32)


def build_valence_class_participant_sampling_weights(
    records: Sequence[MultimodalWindowRecord],
    *,
    protocol: LabelProtocol,
    low_class_mass: float,
    ignore_index: int = -100,
) -> Tensor:
    """Return soft Valence-class and participant-balanced weights ``[N]``.

    Args:
        records: Ordered training records of logical shape ``[N]``.
        protocol: Binary label protocol used by the training dataset.
        low_class_mass: Total expected sampling mass assigned to Valence Low.
            It must lie strictly between zero and 0.5, leaving the remainder
            for High and avoiding exact 50/50 oversampling.
        ignore_index: Label sentinel outside the binary classes.

    Returns:
        Positive float64 CPU weights ``[N]`` summing to one. Low receives
        ``low_class_mass`` and High receives the remainder; within a class,
        every participant containing that class receives equal mass. This
        changes only sampling, never labels or loss weights.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a non-string Sequence.")
    record_tuple = tuple(records)
    if not record_tuple or not all(
        isinstance(record, MultimodalWindowRecord) for record in record_tuple
    ):
        raise ValueError("records must contain at least one manifest record.")
    if not isinstance(protocol, LabelProtocol):
        raise TypeError("protocol must be LabelProtocol.")
    if (
        isinstance(low_class_mass, bool)
        or not isinstance(low_class_mass, (int, float))
        or not math.isfinite(float(low_class_mass))
    ):
        raise TypeError("low_class_mass must be a finite real number, not bool.")
    low_mass = float(low_class_mass)
    if not 0.0 < low_mass < 0.5:
        raise ValueError("low_class_mass must lie strictly between 0 and 0.5.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer, not bool.")
    labels, valid = binarize_emotion_scores(
        torch.tensor(
            [record.emotion_scores.valence for record in record_tuple],
            dtype=torch.float64,
        ),
        protocol,
        ignore_index=ignore_index,
    )
    if not bool(valid.all()):
        raise ValueError(
            "Valence-balanced sampling requires every training record to have "
            "a valid binary valence label."
        )
    participants_by_class: dict[int, set[str]] = {0: set(), 1: set()}
    stratum_counts: dict[tuple[int, str], int] = {}
    for record, label_tensor in zip(record_tuple, labels, strict=True):
        label = int(label_tensor.item())
        participants_by_class[label].add(record.participant_id)
        key = (label, record.participant_id)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    if any(not participants for participants in participants_by_class.values()):
        raise ValueError("training records must contain both Valence classes.")
    class_masses = {0: low_mass, 1: 1.0 - low_mass}
    weights = torch.tensor(
        [
            class_masses[int(label.item())]
            / len(participants_by_class[int(label.item())])
            / stratum_counts[(int(label.item()), record.participant_id)]
            for record, label in zip(record_tuple, labels, strict=True)
        ],
        dtype=torch.float64,
    )
    if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise RuntimeError("sampling weights failed to sum to one.")
    return weights


def build_kemocon_class_weights(
    records: Sequence[MultimodalWindowRecord],
    *,
    protocol: LabelProtocol,
    device: torch.device,
    ignore_index: int = -100,
    participant_balanced: bool = False,
    class_weight_power: float = 1.0,
) -> EmotionTaskClassWeights:
    """Fit arousal/valence class weights from training labels only.

    Args:
        records: Training records containing raw 1--5 scores.
        protocol: Explicit binary label protocol.
        device: Runtime model device.
        ignore_index: Label sentinel outside the two valid classes.
        participant_balanced: Whether each participant contributes total mass
            one before computing inverse-frequency task weights.
        class_weight_power: Exponent in ``[0, 1]`` applied to inverse-frequency
            weights. ``1`` preserves full weighting, ``0.5`` uses square-root
            weighting, and ``0`` produces uniform weights.

    Returns:
        Positive floating arousal and valence weights ``[2]`` on ``device``.
        No independent quadrant weight is produced.
    """

    if not records:
        raise ValueError("records must be non-empty.")
    if not isinstance(participant_balanced, bool):
        raise TypeError("participant_balanced must be bool.")
    arousal_scores = torch.tensor(
        [record.emotion_scores.arousal for record in records],
        dtype=torch.float64,
    )
    valence_scores = torch.tensor(
        [record.emotion_scores.valence for record in records],
        dtype=torch.float64,
    )
    arousal, _ = binarize_emotion_scores(
        arousal_scores,
        protocol,
        ignore_index=ignore_index,
    )
    valence, _ = binarize_emotion_scores(
        valence_scores,
        protocol,
        ignore_index=ignore_index,
    )
    sample_weights: Tensor | None = None
    if participant_balanced:
        participant_counts: dict[str, int] = {}
        for record in records:
            participant_counts[record.participant_id] = (
                participant_counts.get(record.participant_id, 0) + 1
            )
        sample_weights = torch.tensor(
            [
                1.0 / participant_counts[record.participant_id]
                for record in records
            ],
            dtype=torch.float64,
        )
    return EmotionTaskClassWeights(
        arousal=_balanced_binary_weights(
            arousal,
            device=device,
            sample_weights=sample_weights,
            class_weight_power=class_weight_power,
        ),
        valence=_balanced_binary_weights(
            valence,
            device=device,
            sample_weights=sample_weights,
            class_weight_power=class_weight_power,
        ),
        quadrant=None,
    )


def _build_kemocon_physiology_classifier(
    model_config: KEmoConModelConfig,
    specs: tuple[PhysioChannelSpec, ...],
) -> tuple[LightweightPhysioEmotionClassifier, int]:
    """Build a physiology classifier consuming ``[B,T,C]`` and return its width."""

    return (
        LightweightPhysioEmotionClassifier(
            specs,
            model_config.physiology_stem_channels,
            model_config.physiology_embedding_dim,
            dropout=model_config.dropout,
            stem_dropout=0.1,
        ),
        model_config.physiology_embedding_dim,
    )


def _build_kemocon_speech_classifier(
    model_config: KEmoConModelConfig,
    wavlm_encoder: WavLMEncoder,
    wavlm_dim: int,
) -> tuple[LightweightNoiseConditionedSpeechClassifier, int]:
    """Build a speech classifier consuming ``[B,L]`` and return its width."""

    relation_denoiser = NoiseConditionedRelationDifferentialDenoiser(
        wavlm_dim,
        model_config.differential_dim,
        model_config.differential_heads,
        lambda_init=model_config.differential_lambda_init,
        residual_scale=model_config.differential_residual_scale,
        condition_gate_on_noise=model_config.condition_gate_on_noise,
    )
    return (
        LightweightNoiseConditionedSpeechClassifier(
            wavlm_encoder,
            model_config.speech_embedding_dim,
            model_config.noise_embedding_dim,
            emotion_layer_aggregation=EmotionLayerAggregation(
                model_config.emotion_layer_aggregation,
            ),
            relation_denoiser=relation_denoiser,
            film_scale=model_config.film_scale,
            dropout=model_config.dropout,
        ),
        model_config.speech_embedding_dim,
    )


def build_kemocon_model(
    config: KEmoConExperimentConfig,
    channel_specs: Sequence[PhysioChannelSpec],
    *,
    project_root: Path,
    device: torch.device,
) -> MultimodalEmotionClassifier:
    """Build the configured K-EmoCon classifier on ``device``.

    Input batches retain CPU metadata. Compact waveform ``[Bs,L]`` and
    physiology ``[Bp,T,C]`` tensors are transferred by the scheduler.
    """

    model_config = config.model
    if model_config.variant != _MODEL_VARIANT:
        raise ValueError(f"model.variant must be {_MODEL_VARIANT!r}.")
    specs = tuple(channel_specs)
    model_path = resolve_project_relative(project_root, config.paths.wavlm_model)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"local WavLM model is missing at {config.paths.wavlm_model}; "
            "run scripts/cache_wavlm.py before training."
        )
    wavlm = WavLMModel.from_pretrained(model_path, local_files_only=True)
    if int(wavlm.config.num_hidden_layers) != 12:
        raise ValueError("configured WavLM must expose exactly 12 Transformer layers.")
    wavlm_dim = int(wavlm.config.hidden_size)
    wavlm_encoder = WavLMEncoder(
        wavlm,
        sample_rate=config.dataset.speech_sample_rate_hz,
        freeze_wavlm=model_config.freeze_wavlm,
        unfreeze_last_n_layers=model_config.unfreeze_last_n_layers,
        expected_num_hidden_layers=12,
    )
    speech_classifier, speech_dim = _build_kemocon_speech_classifier(
        model_config,
        wavlm_encoder,
        wavlm_dim,
    )
    physiology_classifier, physiology_dim = _build_kemocon_physiology_classifier(
        model_config,
        specs,
    )
    scheduler = MultimodalBatchScheduler(speech_classifier, physiology_classifier)
    fusion = FullWindowDynamicMultimodalFusion(
        speech_dim,
        physiology_dim,
        model_config.fusion_dim,
        gate_hidden_dim=model_config.fusion_gate_hidden_dim,
        dropout=model_config.dropout,
    )
    classifier = MultimodalEmotionClassifier(
        scheduler,
        fusion,
        model_config.fusion_dim,
        classifier_hidden_dim=model_config.classifier_hidden_dim,
        dropout=model_config.dropout,
        enable_independent_quadrant_head=False,
        model_variant=model_config.variant,
    )
    multimodal_result = classifier.to(device=device, dtype=torch.float32)
    parameter_summary = build_model_parameter_summary(multimodal_result)
    if parameter_summary["trainable_parameters_excluding_wavlm"] > 500_000:
        raise ValueError(
            "lightweight trainable parameters excluding WavLM exceed "
            "the 500,000 parameter budget."
        )
    return multimodal_result


def build_kemocon_optimizer(
    model: MultimodalEmotionClassifier,
    config: KEmoConExperimentConfig,
) -> torch.optim.AdamW:
    """Build disjoint downstream and optional WavLM parameter groups.

    Args:
        model: V4.2 classifier whose inputs include speech ``[B,L]`` and
            physiology ``[B,T,C]``.
        config: Validated experiment configuration providing the two rates.

    Returns:
        AdamW optimizer covering every trainable parameter exactly once. A
        trainable WavLM receives ``wavlm_learning_rate``; all heads, temporal
        branches, and fusion parameters receive ``learning_rate``.
    """

    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    if not isinstance(config, KEmoConExperimentConfig):
        raise TypeError("config must be KEmoConExperimentConfig.")
    speech_classifier = model.batch_scheduler.speech_classifier
    wavlm_parameters: tuple[Tensor, ...] = ()
    all_wavlm_parameter_ids: set[int] = set()
    if isinstance(
        speech_classifier,
        LightweightNoiseConditionedSpeechClassifier,
    ):
        wavlm = speech_classifier.wavlm_encoder.model
        all_wavlm_parameter_ids = {id(parameter) for parameter in wavlm.parameters()}
        wavlm_parameters = tuple(
            parameter for parameter in wavlm.parameters() if parameter.requires_grad
        )
    downstream_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in all_wavlm_parameter_ids
    )
    if not downstream_parameters:
        raise RuntimeError("K-EmoCon model has no trainable downstream parameters.")
    groups: list[dict[str, Any]] = [
        {
            "params": downstream_parameters,
            "lr": config.training.learning_rate,
            "group_name": "downstream",
        }
    ]
    if wavlm_parameters:
        groups.append(
            {
                "params": wavlm_parameters,
                "lr": config.training.wavlm_learning_rate,
                "group_name": "wavlm",
            }
        )
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in cast(Sequence[Tensor], group["params"])
    ]
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != expected_ids:
        raise RuntimeError("optimizer groups must partition all trainable parameters.")
    return torch.optim.AdamW(
        groups,
        weight_decay=config.training.weight_decay,
    )


def build_kemocon_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: KEmoConExperimentConfig,
    *,
    mode: str,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build a plateau scheduler with group-name-aware scalar LR floors.

    Args:
        optimizer: Optimizer whose parameter groups have scalar learning rates
            and explicit ``group_name`` values.
        config: Validated experiment configuration containing scheduler policy.
        mode: Plateau direction, either ``"min"`` or ``"max"``.

    Returns:
        A scheduler over the optimizer's scalar group learning rates, or
        ``None`` when disabled. An explicit WavLM minimum is aligned to the
        optimizer's actual group order through ``group_name``.
    """

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if not isinstance(config, KEmoConExperimentConfig):
        raise TypeError("config must be KEmoConExperimentConfig.")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    if not config.training.lr_scheduler_enabled:
        return None
    wavlm_minimum = config.training.min_wavlm_learning_rate
    minimum: float | list[float]
    if wavlm_minimum is None:
        minimum = config.training.min_learning_rate
    else:
        minimum = []
        for group in optimizer.param_groups:
            group_name = group.get("group_name")
            if group_name == "downstream":
                minimum.append(config.training.min_learning_rate)
            elif group_name == "wavlm":
                minimum.append(wavlm_minimum)
            else:
                raise ValueError(
                    "optimizer group_name must be 'downstream' or 'wavlm' "
                    "when separate scheduler minima are configured."
                )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
        min_lr=minimum,
    )


def build_kemocon_objective(
    config: KEmoConLossConfig,
) -> MultimodalTrainingObjective:
    """Build the V4.2 arousal/valence objective with optional auxiliaries."""

    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(
            loss_kind=config.kind,
            weights=MultimodalLossWeights(
                fused=config.fused_weight,
                speech_auxiliary=config.speech_aux_weight,
                physiology_auxiliary=config.physiology_aux_weight,
            ),
            focal_gamma=config.focal_gamma,
        )
    )


def build_model_parameter_summary(
    model: MultimodalEmotionClassifier,
) -> dict[str, int]:
    """Count model parameters by the V4.2 reporting boundary.

    Args:
        model: Constructed V4.2 multimodal classifier. Parameter tensors may
            have arbitrary shapes and devices.

    Returns:
        Integer counts for total, trainable, frozen, trainable excluding the
        injected WavLM, and trainable speech, physiology, fusion, and final
        classifier submodules, plus the optional relation-differential speech
        submodule. Counts are scalar Python integers.
    """
    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    parameters = tuple(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    wavlm_parameters = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if ".wavlm_encoder.model." in f".{name}"
    }
    speech_module = model.batch_scheduler.speech_classifier
    physiology_module = model.batch_scheduler.physiology_classifier
    speech_parameters = (
        tuple(speech_module.parameters()) if speech_module is not None else ()
    )
    physiology_parameters = (
        tuple(physiology_module.parameters())
        if physiology_module is not None
        else ()
    )
    fusion_module = getattr(model, "multimodal_fusion", None)
    fusion_parameters = (
        tuple(fusion_module.parameters()) if fusion_module is not None else ()
    )
    classifier_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("batch_scheduler.", "multimodal_fusion."))
    )
    relation_module = getattr(speech_module, "relation_denoiser", None)
    relation_parameters = (
        tuple(relation_module.parameters()) if relation_module is not None else ()
    )

    def trainable_count(values: Sequence[Tensor]) -> int:
        return sum(value.numel() for value in values if value.requires_grad)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_parameters_excluding_wavlm": sum(
            parameter.numel()
            for parameter in parameters
            if parameter.requires_grad and id(parameter) not in wavlm_parameters
        ),
        "speech_trainable_parameters": trainable_count(speech_parameters),
        "differential_trainable_parameters": trainable_count(
            relation_parameters
        ),
        "physiology_trainable_parameters": trainable_count(
            physiology_parameters
        ),
        "fusion_trainable_parameters": trainable_count(fusion_parameters),
        "classifier_trainable_parameters": trainable_count(
            classifier_parameters
        ),
    }


__all__ = [
    "KEmoConDatasetConfig",
    "KEmoConExperimentConfig",
    "KEmoConLossConfig",
    "KEmoConModalityMode",
    "KEmoConModelConfig",
    "KEmoConPaths",
    "KEmoConSplitConfig",
    "KEmoConTrainingConfig",
    "build_configured_kemocon_split",
    "build_kemocon_class_weights",
    "build_valence_class_participant_sampling_weights",
    "build_kemocon_dataset",
    "build_kemocon_lr_scheduler",
    "build_kemocon_model",
    "build_kemocon_objective",
    "build_kemocon_optimizer",
    "build_model_parameter_summary",
    "fit_kemocon_train_normalizer",
    "load_kemocon_experiment_config",
    "resolve_project_relative",
]
