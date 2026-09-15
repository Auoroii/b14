"""Reporting-only speech-activity stratification for evaluated predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emotion_model.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)

if TYPE_CHECKING:
    from emotion_model.evaluation.runner import EvaluationPredictions


class SpeechActivityBin(StrEnum):
    """Fixed reporting bins for an observed speech activity ratio ``[N]``."""

    RATIO_EQ_0 = "ratio_eq_0"
    RATIO_0_TO_0_1 = "ratio_0_to_0_1"
    RATIO_0_1_TO_0_25 = "ratio_0_1_to_0_25"
    RATIO_0_25_TO_0_5 = "ratio_0_25_to_0_5"
    RATIO_GE_0_5 = "ratio_ge_0_5"


@dataclass(frozen=True)
class FusionWeightStatistics:
    """Finite summary of shared modality weights selected from ``[N,2]``."""

    sample_count: int
    mean_speech_weight: float
    mean_physiology_weight: float
    std_speech_weight: float
    std_physiology_weight: float
    median_speech_weight: float
    median_physiology_weight: float

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count,
            int,
        ):
            raise TypeError("sample_count must be an integer, not bool.")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        values = (
            self.mean_speech_weight,
            self.mean_physiology_weight,
            self.std_speech_weight,
            self.std_physiology_weight,
            self.median_speech_weight,
            self.median_physiology_weight,
        )
        if any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in values
        ):
            raise ValueError("fusion weight statistics must be finite non-negative floats.")
        if self.sample_count == 0 and any(value != 0.0 for value in values):
            raise ValueError("empty fusion weight statistics must be zero.")


def speech_activity_bin_masks(
    ratios: Tensor,
    observed: Tensor,
) -> dict[SpeechActivityBin, Tensor]:
    """Assign observed ratios ``[N]`` to five disjoint boolean masks ``[N]``.

    Args:
        ratios: Floating tensor ``[N]`` containing finite values in ``[0, 1]``.
        observed: Boolean tensor ``[N]`` on the same device. False rows are
            excluded from every bin.

    Returns:
        A fresh mapping in :class:`SpeechActivityBin` order. Its boolean masks
        are exhaustive over ``observed`` rows and exclude all other rows.

    This is the project's only implementation of speech-activity bin
    boundaries. The masks are diagnostics and must never be used as model
    inputs, availability, sample selection, masking, fusion, or loss signals.
    """
    if not isinstance(ratios, Tensor) or not ratios.is_floating_point():
        raise TypeError("ratios must be a floating Tensor.")
    if ratios.ndim != 1:
        raise ValueError("ratios must have exact shape [N].")
    if not isinstance(observed, Tensor) or observed.dtype != torch.bool:
        raise TypeError("observed must be a boolean Tensor.")
    if tuple(observed.shape) != tuple(ratios.shape):
        raise ValueError("observed must have the same shape [N] as ratios.")
    if observed.device != ratios.device:
        raise ValueError("observed and ratios must be on the same device.")
    if not bool(
        torch.isfinite(ratios).all()
        and (ratios >= 0.0).all()
        and (ratios <= 1.0).all()
    ):
        raise ValueError("ratios must contain only finite values in [0, 1].")
    return {
        SpeechActivityBin.RATIO_EQ_0: observed & (ratios == 0.0),
        SpeechActivityBin.RATIO_0_TO_0_1: (
            observed & (ratios > 0.0) & (ratios < 0.10)
        ),
        SpeechActivityBin.RATIO_0_1_TO_0_25: (
            observed & (ratios >= 0.10) & (ratios < 0.25)
        ),
        SpeechActivityBin.RATIO_0_25_TO_0_5: (
            observed & (ratios >= 0.25) & (ratios < 0.50)
        ),
        SpeechActivityBin.RATIO_GE_0_5: observed & (ratios >= 0.50),
    }


@dataclass(frozen=True)
class SpeechActivityStratumMetrics:
    """Binary metrics for one reporting-only activity stratum.

    ``arousal`` and ``valence`` each contain a CPU confusion matrix ``[2,2]``
    and per-class CPU metric tensors ``[2]``. Counts include all rows assigned
    to the bin, including rows whose labels are ignored.
    """

    activity_bin: SpeechActivityBin
    record_count: int
    participant_count: int
    arousal: ClassificationMetrics
    valence: ClassificationMetrics
    availability_patterns: Mapping[str, int]
    all_valid_fusion: FusionWeightStatistics
    both_available_fusion: FusionWeightStatistics

    def __post_init__(self) -> None:
        if not isinstance(self.activity_bin, SpeechActivityBin):
            raise TypeError("activity_bin must be SpeechActivityBin.")
        for name in ("record_count", "participant_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.participant_count > self.record_count:
            raise ValueError("participant_count cannot exceed record_count.")
        for name in ("arousal", "valence"):
            metrics = getattr(self, name)
            if not isinstance(metrics, ClassificationMetrics):
                raise TypeError(f"{name} must be ClassificationMetrics.")
            if metrics.num_classes != 2:
                raise ValueError(f"{name} metrics must use two classes.")
            if metrics.evaluated_count > self.record_count:
                raise ValueError(
                    f"{name} evaluated_count cannot exceed record_count."
                )
        expected_patterns = {
            "both",
            "speech_only",
            "physiology_only",
            "neither",
        }
        if (
            not isinstance(self.availability_patterns, Mapping)
            or set(self.availability_patterns) != expected_patterns
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.availability_patterns.values()
            )
            or sum(self.availability_patterns.values()) != self.record_count
        ):
            raise ValueError(
                "availability_patterns must contain non-negative counts for "
                "all four patterns summing to record_count."
            )
        object.__setattr__(
            self,
            "availability_patterns",
            MappingProxyType(dict(self.availability_patterns)),
        )
        for name in ("all_valid_fusion", "both_available_fusion"):
            if not isinstance(getattr(self, name), FusionWeightStatistics):
                raise TypeError(f"{name} must be FusionWeightStatistics.")


def _fusion_weight_statistics(
    modality_weights: Tensor | None,
    rows: Tensor,
) -> FusionWeightStatistics:
    """Summarize selected detached modality weights ``[M,2]`` safely."""
    if modality_weights is None or not bool(rows.any()):
        return FusionWeightStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    selected = modality_weights[rows].to(dtype=torch.float64)
    return FusionWeightStatistics(
        sample_count=selected.shape[0],
        mean_speech_weight=float(selected[:, 0].mean().item()),
        mean_physiology_weight=float(selected[:, 1].mean().item()),
        std_speech_weight=float(selected[:, 0].std(correction=0).item()),
        std_physiology_weight=float(selected[:, 1].std(correction=0).item()),
        median_speech_weight=float(torch.quantile(selected[:, 0], 0.5).item()),
        median_physiology_weight=float(
            torch.quantile(selected[:, 1], 0.5).item()
        ),
    )


def compute_speech_activity_strata(
    predictions: EvaluationPredictions,
) -> tuple[SpeechActivityStratumMetrics, ...]:
    """Group retained CPU predictions ``[N]`` without another model forward.

    Only rows with a real speech source and an observed activity ratio are
    eligible. Existing predictions are never changed or recomputed. Invalid
    model rows are converted to ignored targets before the shared
    :func:`compute_classification_metrics` implementation is called.
    """
    from emotion_model.evaluation.runner import EvaluationPredictions

    if not isinstance(predictions, EvaluationPredictions):
        raise TypeError("predictions must be EvaluationPredictions.")
    observed = (
        predictions.speech_source_present
        & predictions.speech_activity_observed
    )
    masks = speech_activity_bin_masks(
        predictions.speech_activity_ratios,
        observed,
    )
    arousal_targets = predictions.arousal_targets.clone()
    valence_targets = predictions.valence_targets.clone()
    invalid = ~predictions.sample_valid
    arousal_targets[invalid] = predictions.ignore_index
    valence_targets[invalid] = predictions.ignore_index
    strata: list[SpeechActivityStratumMetrics] = []
    for activity_bin, rows in masks.items():
        participant_count = len(
            {
                participant_id
                for participant_id, included in zip(
                    predictions.participant_ids,
                    rows.tolist(),
                    strict=True,
                )
                if included
            }
        )
        both = rows & predictions.speech_available & predictions.physiology_available
        speech_only = (
            rows & predictions.speech_available & ~predictions.physiology_available
        )
        physiology_only = (
            rows & ~predictions.speech_available & predictions.physiology_available
        )
        neither = rows & ~predictions.sample_valid
        all_valid = rows & predictions.sample_valid
        strata.append(
            SpeechActivityStratumMetrics(
                activity_bin=activity_bin,
                record_count=int(rows.sum().item()),
                participant_count=participant_count,
                arousal=compute_classification_metrics(
                    arousal_targets[rows],
                    predictions.arousal_predictions[rows],
                    num_classes=2,
                    ignore_index=predictions.ignore_index,
                ),
                valence=compute_classification_metrics(
                    valence_targets[rows],
                    predictions.valence_predictions[rows],
                    num_classes=2,
                    ignore_index=predictions.ignore_index,
                ),
                availability_patterns={
                    "both": int(both.sum().item()),
                    "speech_only": int(speech_only.sum().item()),
                    "physiology_only": int(physiology_only.sum().item()),
                    "neither": int(neither.sum().item()),
                },
                all_valid_fusion=_fusion_weight_statistics(
                    predictions.modality_weights,
                    all_valid,
                ),
                both_available_fusion=_fusion_weight_statistics(
                    predictions.modality_weights,
                    both,
                ),
            )
        )
    return tuple(strata)


__all__ = [
    "FusionWeightStatistics",
    "SpeechActivityBin",
    "SpeechActivityStratumMetrics",
    "compute_speech_activity_strata",
    "speech_activity_bin_masks",
]
