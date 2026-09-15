"""Tests for CPU classification metrics used by participant evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from emotion_model.evaluation import (
    ClassificationMetrics,
    EmotionTaskMetrics,
    compute_classification_metrics,
    compute_emotion_task_metrics,
)


def _binary_metrics() -> ClassificationMetrics:
    return compute_classification_metrics(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 1, 1]),
        num_classes=2,
        ignore_index=-100,
    )


def test_perfect_binary_metrics_and_matrix_direction() -> None:
    """Use rows as truth, columns as predictions, with perfect scalars."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0, 1, 1, 0]),
        num_classes=2,
        ignore_index=-100,
    )
    assert torch.equal(metrics.confusion_matrix, torch.tensor([[2, 0], [0, 2]]))
    assert torch.equal(metrics.class_support, torch.tensor([2, 2]))
    assert torch.equal(metrics.class_precision, torch.ones(2, dtype=torch.float64))
    assert torch.equal(metrics.class_recall, torch.ones(2, dtype=torch.float64))
    assert torch.equal(metrics.class_f1, torch.ones(2, dtype=torch.float64))
    assert metrics.evaluated_count == metrics.correct_count == 4
    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0


def test_asymmetric_all_wrong_confusion_is_not_transposed() -> None:
    """Distinguish true-row/predicted-column orientation with asymmetry."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([1, 1, 1, 0]),
        num_classes=2,
        ignore_index=-100,
    )
    assert torch.equal(metrics.confusion_matrix, torch.tensor([[0, 3], [1, 0]]))
    assert metrics.accuracy == 0.0
    assert metrics.correct_count == 0


def test_all_metric_scalars_match_an_independent_three_class_reference() -> None:
    """Independently hand-check counts, support, precision, recall, and F1."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 0, 0, 1, 1, 2]),
        torch.tensor([0, 0, 1, 0, 1, 1]),
        num_classes=3,
        ignore_index=-100,
    )
    assert torch.equal(
        metrics.confusion_matrix,
        torch.tensor([[2, 1, 0], [1, 1, 0], [0, 1, 0]]),
    )
    assert metrics.evaluated_count == 6
    assert metrics.correct_count == 3
    assert torch.equal(metrics.class_support, torch.tensor([3, 2, 1]))
    assert metrics.class_precision.tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    assert metrics.class_recall.tolist() == pytest.approx([2 / 3, 1 / 2, 0])
    assert metrics.class_f1.tolist() == pytest.approx([2 / 3, 2 / 5, 0])
    assert metrics.accuracy == pytest.approx(1 / 2)
    assert metrics.macro_precision == pytest.approx(
        ((2 / 3) + (1 / 3) + 0) / 3
    )
    assert metrics.macro_recall == pytest.approx(
        ((2 / 3) + (1 / 2) + 0) / 3
    )
    assert metrics.macro_f1 == pytest.approx(((2 / 3) + (2 / 5) + 0) / 3)
    assert metrics.balanced_accuracy == pytest.approx(
        ((2 / 3) + (1 / 2) + 0) / 3
    )


def test_zero_denominators_macro_and_balanced_absent_class_differ() -> None:
    """Average macro recall over all classes but balanced over support only."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        num_classes=2,
        ignore_index=-100,
    )
    assert metrics.macro_precision == 0.5
    assert metrics.macro_recall == 0.5
    assert metrics.macro_f1 == 0.5
    assert metrics.balanced_accuracy == 1.0


def test_precision_zero_denominator_and_hand_calculation() -> None:
    """Return zero for an unpredicted class and match independent arithmetic."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 0, 0]),
        num_classes=2,
        ignore_index=-100,
    )
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.macro_precision == pytest.approx((1 / 3 + 0) / 2)
    assert metrics.macro_recall == pytest.approx((1 + 0) / 2)
    assert metrics.macro_f1 == pytest.approx((0.5 + 0) / 2)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.class_precision.tolist() == pytest.approx([1 / 3, 0])
    assert metrics.class_recall.tolist() == pytest.approx([1, 0])
    assert metrics.class_f1.tolist() == pytest.approx([0.5, 0])


def test_zero_true_support_returns_finite_zero_recall_and_f1() -> None:
    """Return zero recall/F1 for an absent true class, even if predicted."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        num_classes=2,
        ignore_index=-100,
    )
    assert metrics.class_precision.tolist() == pytest.approx([1, 0])
    assert metrics.class_recall.tolist() == pytest.approx([0.5, 0])
    assert metrics.class_f1.tolist() == pytest.approx([2 / 3, 0])
    _assert_all_metrics_finite(metrics)


def _assert_all_metrics_finite(metrics: ClassificationMetrics) -> None:
    vectors = (
        metrics.class_precision,
        metrics.class_recall,
        metrics.class_f1,
    )
    assert all(vector.device.type == "cpu" for vector in vectors)
    assert all(vector.dtype == torch.float64 for vector in vectors)
    assert all(bool(torch.isfinite(vector).all()) for vector in vectors)
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            metrics.accuracy,
            metrics.macro_precision,
            metrics.macro_recall,
            metrics.macro_f1,
            metrics.balanced_accuracy,
        )
    )


def test_quadrant_four_class_metrics_include_absent_classes() -> None:
    """Keep all four configured quadrant classes in every macro denominator."""
    metrics = compute_classification_metrics(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 2]),
        num_classes=4,
        ignore_index=-100,
    )
    assert torch.equal(
        metrics.confusion_matrix,
        torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        ),
    )
    assert metrics.macro_recall == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("targets", "predictions"),
    [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        (torch.tensor([-100, -100]), torch.tensor([999, -999])),
    ],
)
def test_empty_and_all_ignored_return_zero_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Support empty input and ignore rows with deliberately invalid predictions."""
    metrics = compute_classification_metrics(
        targets,
        predictions,
        num_classes=2,
        ignore_index=-100,
    )
    assert metrics.evaluated_count == metrics.correct_count == 0
    assert not bool(metrics.confusion_matrix.any())
    assert not bool(metrics.class_support.any())
    assert not bool(metrics.class_precision.any())
    assert not bool(metrics.class_recall.any())
    assert not bool(metrics.class_f1.any())
    assert all(
        value == 0.0
        for value in (
            metrics.accuracy,
            metrics.macro_precision,
            metrics.macro_recall,
            metrics.macro_f1,
            metrics.balanced_accuracy,
        )
    )
    _assert_all_metrics_finite(metrics)


@pytest.mark.parametrize(
    ("targets", "predictions", "expected"),
    [
        (
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 0, 0]),
            torch.zeros(2, dtype=torch.float64),
        ),
        (
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0, 1, 1]),
            torch.ones(2, dtype=torch.float64),
        ),
        (
            torch.tensor([0, -100, 1, -100]),
            torch.tensor([0, 999, 1, -999]),
            torch.ones(2, dtype=torch.float64),
        ),
    ],
)
def test_binary_edge_cases_have_expected_finite_per_class_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Cover all-wrong, perfect, and ignored-label per-class metrics."""
    metrics = compute_classification_metrics(
        targets,
        predictions,
        num_classes=2,
        ignore_index=-100,
    )
    assert torch.equal(metrics.class_precision, expected)
    assert torch.equal(metrics.class_recall, expected)
    assert torch.equal(metrics.class_f1, expected)
    _assert_all_metrics_finite(metrics)


@pytest.mark.parametrize(
    ("targets", "predictions", "error_type"),
    [
        (torch.tensor([-1]), torch.tensor([0]), ValueError),
        (torch.tensor([2]), torch.tensor([0]), ValueError),
        (torch.tensor([0]), torch.tensor([-1]), ValueError),
        (torch.tensor([0]), torch.tensor([2]), ValueError),
        (torch.tensor([[0]]), torch.tensor([0]), ValueError),
        (torch.tensor([0]), torch.tensor([0, 1]), ValueError),
        (torch.tensor([0.0]), torch.tensor([0]), TypeError),
        (torch.tensor([0]), torch.tensor([0], dtype=torch.int32), TypeError),
    ],
)
def test_metric_function_rejects_invalid_labels_shapes_and_dtypes(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    error_type: type[Exception],
) -> None:
    """Reject malformed contracts only on rows selected for evaluation."""
    with pytest.raises(error_type):
        compute_classification_metrics(
            targets,
            predictions,
            num_classes=2,
            ignore_index=-100,
        )


@pytest.mark.parametrize(
    ("num_classes", "ignore_index", "error_type"),
    [
        (True, -100, TypeError),
        (1, -100, ValueError),
        (2, True, TypeError),
        (2, 0, ValueError),
        (2, 1, ValueError),
    ],
)
def test_metric_function_rejects_invalid_scalar_contracts(
    num_classes: int,
    ignore_index: int,
    error_type: type[Exception],
) -> None:
    """Reject boolean integers, too few classes, and conflicting ignore values."""
    with pytest.raises(error_type):
        compute_classification_metrics(
            torch.tensor([0]),
            torch.tensor([0]),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )


def test_metric_inputs_and_returned_storage_are_independent() -> None:
    """Never modify inputs or share returned matrix/support storage."""
    targets = torch.tensor([0, 1])
    predictions = torch.tensor([0, 1])
    targets_before = targets.clone()
    predictions_before = predictions.clone()
    metrics = compute_classification_metrics(
        targets,
        predictions,
        num_classes=2,
        ignore_index=-100,
    )
    metrics.confusion_matrix[0, 0] = 99
    metrics.class_precision[0] = 0.25
    assert torch.equal(targets, targets_before)
    assert torch.equal(predictions, predictions_before)
    assert torch.equal(metrics.class_support, torch.tensor([1, 1]))
    assert torch.equal(metrics.class_recall, torch.ones(2, dtype=torch.float64))
    assert torch.equal(metrics.class_f1, torch.ones(2, dtype=torch.float64))
    assert metrics.confusion_matrix.data_ptr() != metrics.class_support.data_ptr()
    assert metrics.class_precision.data_ptr() != metrics.class_recall.data_ptr()


def test_classification_metrics_direct_validation_and_frozen() -> None:
    """Validate count-derived tensor contracts and prevent field reassignment."""
    metrics = _binary_metrics()
    with pytest.raises(FrozenInstanceError):
        metrics.correct_count = 0  # type: ignore[misc]
    with pytest.raises(ValueError, match="row sums"):
        replace(metrics, class_support=torch.tensor([0, 4]))
    with pytest.raises(ValueError, match="trace"):
        replace(metrics, correct_count=0)
    with pytest.raises(TypeError, match="Python float"):
        replace(metrics, accuracy=1)  # type: ignore[arg-type]


def test_emotion_task_helper_reuses_class_counts_and_ignore_independently() -> None:
    """Compute binary/binary/quadrant tasks with task-specific ignored rows."""
    metrics = compute_emotion_task_metrics(
        arousal_targets=torch.tensor([0, -100, 1]),
        arousal_predictions=torch.tensor([0, 999, 0]),
        valence_targets=torch.tensor([1, 0, 1]),
        valence_predictions=torch.tensor([1, 0, 1]),
        quadrant_targets=torch.tensor([2, -100, 3]),
        quadrant_predictions=torch.tensor([2, -999, 1]),
        ignore_index=-100,
    )
    assert metrics.arousal.num_classes == 2
    assert metrics.valence.num_classes == 2
    assert metrics.quadrant.num_classes == 4
    assert metrics.arousal.evaluated_count == 2
    assert metrics.valence.evaluated_count == 3
    assert metrics.quadrant.evaluated_count == 2
    with pytest.raises(FrozenInstanceError):
        metrics.arousal = _binary_metrics()  # type: ignore[misc]


def test_emotion_task_dataclass_rejects_wrong_task_class_count() -> None:
    """Prevent a four-class metric from being installed as a binary task."""
    binary = _binary_metrics()
    quadrant = compute_classification_metrics(
        torch.tensor([0]),
        torch.tensor([0]),
        num_classes=4,
        ignore_index=-100,
    )
    with pytest.raises(ValueError, match="arousal"):
        EmotionTaskMetrics(
            arousal=quadrant,
            valence=binary,
            quadrant=quadrant,
        )
