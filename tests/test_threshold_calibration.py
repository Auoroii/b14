"""Validation-only binary decision-threshold calibration tests."""

from __future__ import annotations

import pytest
import torch

from emotion_model.evaluation import (
    BinaryDecisionThresholds,
    calibrate_binary_decision_thresholds,
)


def test_calibration_finds_independent_pooled_macro_f1_thresholds() -> None:
    """Find task thresholds that separate both validation classes."""

    participant_ids = ("P1", "P1", "P2", "P2")
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    thresholds = calibrate_binary_decision_thresholds(
        arousal_high_probabilities=torch.tensor(
            [0.60, 0.80, 0.65, 0.90],
            dtype=torch.float64,
        ),
        valence_high_probabilities=torch.tensor(
            [0.30, 0.45, 0.40, 0.55],
            dtype=torch.float64,
        ),
        arousal_targets=targets,
        valence_targets=targets,
        participant_ids=participant_ids,
        sample_valid=torch.ones(4, dtype=torch.bool),
        ignore_index=-100,
    )

    assert 0.64 <= thresholds.arousal_high < 0.80
    assert 0.39 <= thresholds.valence_high < 0.45
    assert "validation_pooled_macro_f1" in thresholds.policy


def test_threshold_metadata_round_trip_is_exact_and_strict() -> None:
    """Persist finite threshold values without accepting unknown fields."""

    source = BinaryDecisionThresholds(
        arousal_high=0.63,
        valence_high=0.57,
        policy="validation-test-policy",
    )
    assert BinaryDecisionThresholds.from_metadata(
        source.to_metadata()
    ) == source
    with pytest.raises(ValueError, match="fields"):
        BinaryDecisionThresholds.from_metadata(
            {
                **source.to_metadata(),
                "unknown": 1,
            }
        )


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (True, TypeError),
        (0.0, ValueError),
        (1.0, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_thresholds_reject_invalid_values(
    value: object,
    exception: type[Exception],
) -> None:
    """Keep saved binary thresholds finite and strictly inside (0, 1)."""

    with pytest.raises(exception):
        BinaryDecisionThresholds(
            arousal_high=value,  # type: ignore[arg-type]
        )
