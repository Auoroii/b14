"""Validation-only binary decision-threshold calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from emotion_model.evaluation.metrics import compute_classification_metrics

_CALIBRATION_POLICY = (
    "validation_pooled_macro_f1_grid_0.05_0.95_step_0.01"
)
_DEFAULT_POLICY = "fixed_high_probability_threshold_0.5"


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly in (0, 1).")
    return result


def _unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


@dataclass(frozen=True)
class BinaryDecisionThresholds:
    """High-class probability thresholds for arousal and valence.

    Attributes:
        arousal_high: Predict high arousal when ``P(high) > arousal_high``.
        valence_high: Predict high valence when ``P(high) > valence_high``.
        policy: Non-empty description of how the thresholds were selected.
    """

    arousal_high: float = 0.5
    valence_high: float = 0.5
    policy: str = _DEFAULT_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arousal_high",
            _probability(self.arousal_high, name="arousal_high"),
        )
        object.__setattr__(
            self,
            "valence_high",
            _probability(self.valence_high, name="valence_high"),
        )
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("policy must be a non-empty string.")

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible checkpoint representation."""

        return {
            "arousal_high": self.arousal_high,
            "valence_high": self.valence_high,
            "policy": self.policy,
        }

    @classmethod
    def from_metadata(
        cls,
        value: object,
    ) -> BinaryDecisionThresholds:
        """Parse an exact checkpoint metadata mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("binary decision threshold metadata must be a mapping.")
        if set(value) != {"arousal_high", "valence_high", "policy"}:
            raise ValueError(
                "binary decision threshold metadata fields do not match."
            )
        return cls(
            arousal_high=value["arousal_high"],  # type: ignore[arg-type]
            valence_high=value["valence_high"],  # type: ignore[arg-type]
            policy=value["policy"],  # type: ignore[arg-type]
        )


def _validate_calibration_inputs(
    high_probabilities: Tensor,
    targets: Tensor,
    participant_ids: Sequence[str],
    sample_valid: Tensor,
    *,
    ignore_index: int,
) -> tuple[str, ...]:
    if (
        not isinstance(high_probabilities, Tensor)
        or not high_probabilities.is_floating_point()
        or high_probabilities.ndim != 1
        or high_probabilities.device.type != "cpu"
    ):
        raise ValueError("high_probabilities must be floating CPU [N].")
    length = high_probabilities.numel()
    if (
        not isinstance(targets, Tensor)
        or targets.dtype != torch.long
        or tuple(targets.shape) != (length,)
        or targets.device.type != "cpu"
    ):
        raise ValueError("targets must be long CPU [N].")
    if (
        not isinstance(sample_valid, Tensor)
        or sample_valid.dtype != torch.bool
        or tuple(sample_valid.shape) != (length,)
        or sample_valid.device.type != "cpu"
    ):
        raise ValueError("sample_valid must be bool CPU [N].")
    if isinstance(participant_ids, (str, bytes)):
        raise TypeError("participant_ids must be a non-string sequence.")
    ids = tuple(participant_ids)
    if len(ids) != length or not all(
        isinstance(value, str) and value for value in ids
    ):
        raise ValueError("participant_ids must contain N non-empty strings.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer, not bool.")
    if 0 <= ignore_index < 2:
        raise ValueError("ignore_index must lie outside [0, 2).")
    if not bool(torch.isfinite(high_probabilities).all()):
        raise ValueError("high_probabilities must be finite.")
    if not bool(
        (high_probabilities >= 0.0).all()
        and (high_probabilities <= 1.0).all()
    ):
        raise ValueError("high_probabilities must lie in [0, 1].")
    legal_targets = (
        (targets == ignore_index) | ((targets >= 0) & (targets < 2))
    )
    if not bool(legal_targets.all()):
        raise ValueError("targets must contain binary classes or ignore_index.")
    return ids


def _pooled_macro_f1(
    high_probabilities: Tensor,
    targets: Tensor,
    sample_valid: Tensor,
    *,
    threshold: float,
    ignore_index: int,
) -> float:
    predictions = (high_probabilities > threshold).to(dtype=torch.long)
    working_targets = targets.clone()
    working_targets[~sample_valid] = ignore_index
    metrics = compute_classification_metrics(
        working_targets,
        predictions,
        num_classes=2,
        ignore_index=ignore_index,
    )
    if metrics.evaluated_count == 0:
        raise ValueError("threshold calibration has no evaluable sample.")
    return metrics.macro_f1


def select_pooled_macro_f1_threshold(
    high_probabilities: Tensor,
    targets: Tensor,
    participant_ids: Sequence[str],
    sample_valid: Tensor,
    *,
    ignore_index: int,
) -> float:
    """Select one validation threshold on a fixed ``0.05..0.95`` grid.

    Args:
        high_probabilities: Detached floating CPU tensor ``[N]``.
        targets: Binary long CPU labels ``[N]`` with ``ignore_index`` allowed.
        participant_ids: Participant identifier for every row, retained to
            validate row alignment and prevent accidental partition mixing.
        sample_valid: Boolean CPU tensor ``[N]`` with ``True=valid``.
        ignore_index: Label sentinel outside the binary classes.

    Returns:
        A Python float from the fixed 91-point grid. The primary objective is
        pooled binary macro-F1, which weights Low and High F1 equally without
        giving High-only participants separate objective mass. Ties prefer the
        threshold closest to 0.5, then the smaller threshold. Test rows are
        never accepted separately; callers must pass validation only.
    """

    _validate_calibration_inputs(
        high_probabilities,
        targets,
        participant_ids,
        sample_valid,
        ignore_index=ignore_index,
    )
    candidates = torch.linspace(
        0.05,
        0.95,
        91,
        dtype=torch.float64,
    ).tolist()
    return max(
        candidates,
        key=lambda threshold: (
            _pooled_macro_f1(
                high_probabilities,
                targets,
                sample_valid,
                threshold=threshold,
                ignore_index=ignore_index,
            ),
            -abs(threshold - 0.5),
            -threshold,
        ),
    )


def calibrate_binary_decision_thresholds(
    *,
    arousal_high_probabilities: Tensor,
    valence_high_probabilities: Tensor,
    arousal_targets: Tensor,
    valence_targets: Tensor,
    participant_ids: Sequence[str],
    sample_valid: Tensor,
    ignore_index: int,
    shrinkage: float = 1.0,
) -> BinaryDecisionThresholds:
    """Select and regularize validation thresholds from CPU vectors ``[N]``.

    ``shrinkage`` retains that fraction of each grid-selected threshold's
    displacement from 0.5. Thus ``1.0`` preserves the searched threshold,
    ``0.5`` moves it halfway toward 0.5, and ``0.0`` uses 0.5 exactly.
    """

    resolved_shrinkage = _unit_interval(shrinkage, name="shrinkage")
    raw_arousal = select_pooled_macro_f1_threshold(
        arousal_high_probabilities,
        arousal_targets,
        participant_ids,
        sample_valid,
        ignore_index=ignore_index,
    )
    raw_valence = select_pooled_macro_f1_threshold(
        valence_high_probabilities,
        valence_targets,
        participant_ids,
        sample_valid,
        ignore_index=ignore_index,
    )

    return BinaryDecisionThresholds(
        arousal_high=0.5 + resolved_shrinkage * (raw_arousal - 0.5),
        valence_high=0.5 + resolved_shrinkage * (raw_valence - 0.5),
        policy=(
            f"{_CALIBRATION_POLICY}_center_retention_"
            f"{resolved_shrinkage:g}"
        ),
    )


__all__ = [
    "BinaryDecisionThresholds",
    "calibrate_binary_decision_thresholds",
    "select_pooled_macro_f1_threshold",
]
