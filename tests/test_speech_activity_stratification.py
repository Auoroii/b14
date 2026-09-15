"""Tests for reporting-only speech activity stratification."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emotion_model.common import derive_quadrant_labels
from emotion_model.evaluation import (
    EvaluationPredictions,
    SpeechActivityBin,
    compute_classification_metrics,
    compute_speech_activity_strata,
    evaluate_participant_independent,
    speech_activity_bin_masks,
)
from emotion_model.experiments import (
    build_evaluation_summary,
    speech_activity_stratified_summary,
)
from tests.test_multimodal_classifier import _model
from tests.test_multimodal_routing import _batch
from tests.test_participant_evaluation import (
    _CountingModel,
    _objective,
    _scope,
    _with_record_ids,
)


_BOUNDARY_RATIOS = torch.tensor(
    [0.0, 0.0001, 0.0999, 0.10, 0.2499, 0.25, 0.4999, 0.50, 1.0],
    dtype=torch.float32,
)


def _predictions(
    ratios: torch.Tensor = _BOUNDARY_RATIOS,
) -> EvaluationPredictions:
    length = ratios.numel()
    arousal_targets = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1, 0])[:length]
    arousal_predictions = torch.tensor([0, 1, 1, 0, 0, 1, 1, 1, 0])[
        :length
    ]
    valence_targets = 1 - arousal_targets
    valence_predictions = 1 - arousal_predictions
    return EvaluationPredictions(
        sample_ids=tuple(f"sample-{index}" for index in range(length)),
        participant_ids=(
            "p0",
            "p0",
            "p1",
            "p1",
            "p2",
            "p2",
            "p3",
            "p3",
            "p3",
        )[:length],
        speech_available=torch.ones(length, dtype=torch.bool),
        physiology_available=torch.zeros(length, dtype=torch.bool),
        sample_valid=torch.ones(length, dtype=torch.bool),
        arousal_targets=arousal_targets,
        arousal_predictions=arousal_predictions,
        valence_targets=valence_targets,
        valence_predictions=valence_predictions,
        quadrant_targets=derive_quadrant_labels(
            arousal_targets,
            valence_targets,
            ignore_index=-100,
        ),
        quadrant_predictions=derive_quadrant_labels(
            arousal_predictions,
            valence_predictions,
            ignore_index=-100,
        ),
        ignore_index=-100,
        speech_source_present=torch.ones(length, dtype=torch.bool),
        speech_activity_observed=torch.ones(length, dtype=torch.bool),
        speech_activity_ratios=ratios,
    )


def test_activity_bin_boundaries_are_exact_and_exhaustive() -> None:
    """Assign all nine required boundary probes to exactly one fixed bin."""
    masks = speech_activity_bin_masks(
        _BOUNDARY_RATIOS,
        torch.ones(9, dtype=torch.bool),
    )
    assert tuple(masks) == tuple(SpeechActivityBin)
    assert masks[SpeechActivityBin.RATIO_EQ_0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks[SpeechActivityBin.RATIO_0_TO_0_1].nonzero().flatten().tolist() == [
        1,
        2,
    ]
    assert masks[
        SpeechActivityBin.RATIO_0_1_TO_0_25
    ].nonzero().flatten().tolist() == [3, 4]
    assert masks[
        SpeechActivityBin.RATIO_0_25_TO_0_5
    ].nonzero().flatten().tolist() == [5, 6]
    assert masks[SpeechActivityBin.RATIO_GE_0_5].nonzero().flatten().tolist() == [
        7,
        8,
    ]
    membership_count = torch.stack(tuple(masks.values())).sum(dim=0)
    assert torch.equal(membership_count, torch.ones(9, dtype=torch.long))


def test_strata_count_actual_participants_and_reuse_classification_metrics() -> None:
    """Count unique participants and match the shared metric implementation."""
    predictions = _predictions()
    strata = {
        stratum.activity_bin: stratum
        for stratum in compute_speech_activity_strata(predictions)
    }
    assert [strata[value].record_count for value in SpeechActivityBin] == [
        1,
        2,
        2,
        2,
        2,
    ]
    assert [strata[value].participant_count for value in SpeechActivityBin] == [
        1,
        2,
        2,
        2,
        1,
    ]
    expected = compute_classification_metrics(
        predictions.arousal_targets[torch.tensor([1, 2])],
        predictions.arousal_predictions[torch.tensor([1, 2])],
        num_classes=2,
        ignore_index=-100,
    )
    actual = strata[SpeechActivityBin.RATIO_0_TO_0_1].arousal
    assert torch.equal(actual.confusion_matrix, expected.confusion_matrix)
    assert torch.equal(actual.class_precision, expected.class_precision)
    assert torch.equal(actual.class_recall, expected.class_recall)
    assert torch.equal(actual.class_f1, expected.class_f1)


def test_only_observed_real_source_rows_are_stratified() -> None:
    """Exclude absent sources and source rows without observable diagnostics."""
    predictions = replace(
        _predictions(torch.tensor([0.0, 0.20, 0.70])),
        speech_source_present=torch.tensor([True, False, True]),
        speech_activity_observed=torch.tensor([True, False, False]),
    )
    strata = compute_speech_activity_strata(predictions)
    assert sum(stratum.record_count for stratum in strata) == 1
    assert strata[0].record_count == 1
    assert strata[0].participant_count == 1


def test_empty_activity_bins_report_finite_zero_metrics() -> None:
    """Serialize every empty bin with finite zero binary metrics."""
    summary = speech_activity_stratified_summary(
        _predictions(torch.tensor([0.0]))
    )
    assert tuple(summary) == tuple(value.value for value in SpeechActivityBin)
    for activity_bin in tuple(SpeechActivityBin)[1:]:
        empty = summary[activity_bin.value]
        assert empty["record_count"] == 0
        assert empty["participant_count"] == 0
        for task_name in ("arousal", "valence"):
            task = empty[task_name]
            assert task["accuracy"] == 0.0
            assert task["macro_precision"] == 0.0
            assert task["macro_recall"] == 0.0
            assert task["macro_f1"] == 0.0
            assert task["balanced_accuracy"] == 0.0
            assert task["confusion_matrix"] == [[0, 0], [0, 0]]
            assert task["class_support"] == [0, 0]
            assert task["per_class"]["low"]["f1"] == 0.0
            assert task["per_class"]["high"]["f1"] == 0.0
        fusion = empty["fusion"]
        assert fusion["availability_patterns"] == {
            "both": 0,
            "speech_only": 0,
            "physiology_only": 0,
            "neither": 0,
        }
        assert fusion["all_valid"]["sample_count"] == 0
        assert fusion["both_available"]["sample_count"] == 0
        assert all(
            value == 0.0
            for name, value in fusion["all_valid"].items()
            if name != "sample_count"
        )
        assert all(
            value == 0.0
            for name, value in fusion["both_available"].items()
            if name != "sample_count"
        )
    json.dumps(summary, allow_nan=False)


def test_fusion_statistics_separate_all_valid_from_both_available() -> None:
    """Summarize learned weights without single-modality 0/1 contamination."""
    predictions = replace(
        _predictions(torch.full((4,), 0.20)),
        speech_available=torch.tensor([True, True, True, False]),
        physiology_available=torch.tensor([True, True, False, True]),
        sample_valid=torch.ones(4, dtype=torch.bool),
        modality_weights=torch.tensor(
            [[0.2, 0.8], [0.6, 0.4], [1.0, 0.0], [0.0, 1.0]]
        ),
    )
    summary = speech_activity_stratified_summary(predictions)[
        "ratio_0_1_to_0_25"
    ]
    fusion = summary["fusion"]

    assert fusion["availability_patterns"] == {
        "both": 2,
        "speech_only": 1,
        "physiology_only": 1,
        "neither": 0,
    }
    assert fusion["all_valid"] == {
        "sample_count": 4,
        "mean_speech_weight": pytest.approx(0.45),
        "mean_physiology_weight": pytest.approx(0.55),
        "std_speech_weight": pytest.approx(
            torch.tensor([0.2, 0.6, 1.0, 0.0]).std(correction=0).item()
        ),
        "std_physiology_weight": pytest.approx(
            torch.tensor([0.8, 0.4, 0.0, 1.0]).std(correction=0).item()
        ),
        "median_speech_weight": pytest.approx(0.4),
        "median_physiology_weight": pytest.approx(0.6),
    }
    assert fusion["both_available"] == {
        "sample_count": 2,
        "mean_speech_weight": pytest.approx(0.4),
        "mean_physiology_weight": pytest.approx(0.6),
        "std_speech_weight": pytest.approx(0.2),
        "std_physiology_weight": pytest.approx(0.2),
        "median_speech_weight": pytest.approx(0.4),
        "median_physiology_weight": pytest.approx(0.6),
    }


def test_retained_modality_weights_enforce_availability_constraints() -> None:
    """Reject weights that violate shared masked-softmax availability rules."""
    base = _predictions(torch.tensor([0.20, 0.20]))
    with pytest.raises(ValueError, match="sum to one"):
        replace(
            base,
            physiology_available=torch.ones(2, dtype=torch.bool),
            modality_weights=torch.tensor([[0.7, 0.4], [0.5, 0.5]]),
        )
    with pytest.raises(ValueError, match="speech-only"):
        replace(
            base,
            modality_weights=torch.tensor([[0.9, 0.1], [1.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="physiology-only"):
        replace(
            base,
            speech_available=torch.zeros(2, dtype=torch.bool),
            physiology_available=torch.ones(2, dtype=torch.bool),
            modality_weights=torch.tensor([[0.1, 0.9], [0.0, 1.0]]),
        )


def test_activity_ratio_changes_only_reporting_groups_not_model_logits() -> None:
    """Keep model tensors fixed while moving identical predictions across bins."""
    base = _batch(((True, True), (True, True), (True, True)))
    first_batch = replace(
        base,
        speech_activity_ratios=torch.tensor([0.0, 0.0001, 0.0999]),
    )
    second_batch = replace(
        base,
        speech_activity_ratios=torch.tensor([0.10, 0.25, 0.50]),
    )
    model = _model(dropout=0.0).eval()
    with torch.no_grad():
        first_output = model(first_batch)
        second_output = model(second_batch)

    assert first_batch.speech is second_batch.speech
    assert first_batch.physiology is second_batch.physiology
    assert torch.equal(first_batch.speech_available, second_batch.speech_available)
    assert torch.equal(
        first_batch.physiology_available,
        second_batch.physiology_available,
    )
    assert torch.equal(first_output.arousal_logits, second_output.arousal_logits)
    assert torch.equal(first_output.valence_logits, second_output.valence_logits)
    assert torch.equal(first_output.fused_embedding, second_output.fused_embedding)
    assert torch.equal(
        first_output.fusion_output.modality_weights,
        second_output.fusion_output.modality_weights,
    )
    assert first_batch.speech_available[0].item() is True
    assert first_output.fusion_output.modality_weights[0, 0].item() > 0.0

    first_predictions = replace(
        _predictions(first_batch.speech_activity_ratios),
        physiology_available=torch.ones(3, dtype=torch.bool),
        modality_weights=(
            first_output.fusion_output.modality_weights.detach().cpu()
        ),
    )
    second_predictions = replace(
        first_predictions,
        speech_activity_ratios=second_batch.speech_activity_ratios,
    )
    assert torch.equal(
        first_predictions.arousal_predictions,
        second_predictions.arousal_predictions,
    )
    assert torch.equal(
        first_predictions.modality_weights,
        second_predictions.modality_weights,
    )
    first_counts = [
        item.record_count
        for item in compute_speech_activity_strata(first_predictions)
    ]
    second_counts = [
        item.record_count
        for item in compute_speech_activity_strata(second_predictions)
    ]
    assert first_counts == [1, 2, 0, 0, 0]
    assert second_counts == [0, 0, 1, 1, 1]


def test_test_metrics_stratification_reuses_one_evaluation_forward() -> None:
    """Build all five JSON strata after the runner's sole model forward."""
    batch = _with_record_ids(
        _batch(((True, True), (True, False))),
        ("p1", "p2"),
    )
    batch = replace(
        batch,
        speech_activity_ratios=torch.tensor([0.0, 0.50]),
    )
    model = _CountingModel()
    result = evaluate_participant_independent(
        model,
        _objective(),
        (batch,),
        scope=_scope(),
    )
    assert model.forward_calls == 1

    test_metrics = build_evaluation_summary(
        result,
        fold_index=0,
        checkpoint=Path("runs/fold_0/best.pt"),
    )

    assert model.forward_calls == 1
    stratified = test_metrics["speech_activity_stratified"]
    assert stratified["ratio_eq_0"]["record_count"] == 1
    assert stratified["ratio_ge_0_5"]["record_count"] == 1
    assert stratified["ratio_0_to_0_1"]["record_count"] == 0
    assert stratified["ratio_eq_0"]["fusion"]["availability_patterns"] == {
        "both": 1,
        "speech_only": 0,
        "physiology_only": 0,
        "neither": 0,
    }
    ratio_zero_fusion = stratified["ratio_eq_0"]["fusion"]
    assert ratio_zero_fusion["all_valid"]["sample_count"] == 1
    assert ratio_zero_fusion["both_available"]["sample_count"] == 1
    assert (
        ratio_zero_fusion["all_valid"]["mean_speech_weight"]
        == ratio_zero_fusion["both_available"]["mean_speech_weight"]
    )
    high_fusion = stratified["ratio_ge_0_5"]["fusion"]
    assert high_fusion["availability_patterns"]["speech_only"] == 1
    assert high_fusion["all_valid"]["mean_speech_weight"] == 1.0
    assert high_fusion["both_available"]["sample_count"] == 0
    assert (
        stratified["ratio_eq_0"]["arousal"]["per_class"]["low"]["recall"]
        == 0.0
    )
    json.dumps(test_metrics, allow_nan=False)


@pytest.mark.parametrize(
    "ratios",
    [
        torch.tensor([float("nan")]),
        torch.tensor([-0.01]),
        torch.tensor([1.01]),
    ],
)
def test_activity_bins_reject_nonfinite_or_out_of_range_ratios(
    ratios: torch.Tensor,
) -> None:
    """Reject diagnostic values that cannot belong to the fixed bins."""
    with pytest.raises(ValueError, match="finite values"):
        speech_activity_bin_masks(ratios, torch.ones(1, dtype=torch.bool))
