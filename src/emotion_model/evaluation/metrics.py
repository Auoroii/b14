"""CPU classification metrics for participant-independent evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_metric_scalar(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a Python float.")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")


@dataclass(frozen=True)
class ClassificationMetrics:
    """Immutable metrics for one ``K``-class task.

    ``confusion_matrix`` is an independent CPU ``torch.long`` tensor
    ``[K, K]`` whose rows are true labels and columns are predictions.
    ``class_support`` is an independent CPU ``torch.long`` tensor ``[K]`` and
    equals the matrix row sums. Macro precision, recall, and F1 average over
    all configured classes, so absent classes contribute zero.
    ``balanced_accuracy`` instead averages recall only over classes with
    positive true support. With no evaluated rows, every scalar is ``0.0``.

    Invalid dimensions, counts, storage devices, dtypes, or inconsistent
    derived values raise :class:`TypeError` or :class:`ValueError`.
    """

    num_classes: int
    evaluated_count: int
    correct_count: int
    confusion_matrix: Tensor
    class_support: Tensor
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    balanced_accuracy: float

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.num_classes, name="num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        _validate_nonnegative_int(self.evaluated_count, name="evaluated_count")
        _validate_nonnegative_int(self.correct_count, name="correct_count")
        if self.correct_count > self.evaluated_count:
            raise ValueError("correct_count cannot exceed evaluated_count.")

        matrix = self.confusion_matrix
        support = self.class_support
        if not isinstance(matrix, Tensor):
            raise TypeError("confusion_matrix must be a Tensor.")
        if matrix.device.type != "cpu":
            raise ValueError("confusion_matrix must be on CPU.")
        if matrix.dtype != torch.long:
            raise TypeError("confusion_matrix must use torch.long.")
        expected_matrix_shape = (self.num_classes, self.num_classes)
        if tuple(matrix.shape) != expected_matrix_shape:
            raise ValueError(
                "confusion_matrix must have shape "
                f"{expected_matrix_shape}; received {tuple(matrix.shape)}."
            )
        if bool((matrix < 0).any()):
            raise ValueError("confusion_matrix counts must be non-negative.")
        if int(matrix.sum().item()) != self.evaluated_count:
            raise ValueError(
                "confusion_matrix sum must equal evaluated_count."
            )
        if int(matrix.diagonal().sum().item()) != self.correct_count:
            raise ValueError("confusion_matrix trace must equal correct_count.")

        if not isinstance(support, Tensor):
            raise TypeError("class_support must be a Tensor.")
        if support.device.type != "cpu":
            raise ValueError("class_support must be on CPU.")
        if support.dtype != torch.long:
            raise TypeError("class_support must use torch.long.")
        if tuple(support.shape) != (self.num_classes,):
            raise ValueError(
                f"class_support must have shape [{self.num_classes}]; "
                f"received {tuple(support.shape)}."
            )
        if not torch.equal(support, matrix.sum(dim=1)):
            raise ValueError(
                "class_support must equal confusion_matrix row sums."
            )

        for name in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        ):
            _validate_metric_scalar(getattr(self, name), name=name)

        # Frozen dataclasses do not make tensors deeply immutable. Defensive
        # copies prevent returned public tensors from sharing accumulator
        # storage with either each other or caller-owned inputs.
        object.__setattr__(self, "confusion_matrix", matrix.clone())
        object.__setattr__(self, "class_support", support.clone())


def _validate_metric_inputs(
    targets: Tensor,
    predictions: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    if not isinstance(targets, Tensor) or not isinstance(predictions, Tensor):
        raise TypeError("targets and predictions must be Tensor objects.")
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "targets and predictions must each have exact shape [N]."
        )
    if tuple(targets.shape) != tuple(predictions.shape):
        raise ValueError(
            "targets and predictions must have exactly the same shape; "
            f"received {tuple(targets.shape)} and {tuple(predictions.shape)}."
        )
    if targets.dtype != torch.long or predictions.dtype != torch.long:
        raise TypeError("targets and predictions must use torch.long.")
    if targets.device.type != "cpu" or predictions.device.type != "cpu":
        raise ValueError("targets and predictions must be on CPU.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer, not bool.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer, not bool.")
    if 0 <= ignore_index < num_classes:
        raise ValueError(
            "ignore_index must lie outside the configured class range."
        )


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Return elementwise ratios with exact zeros for zero denominators."""
    positive = denominator > 0
    safe_denominator = torch.where(
        positive,
        denominator,
        torch.ones_like(denominator),
    )
    return torch.where(
        positive,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


def compute_classification_metrics(
    targets: Tensor,
    predictions: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> ClassificationMetrics:
    """Compute one classification task from CPU label vectors ``[N]``.

    Args:
        targets: CPU ``torch.long`` tensor ``[N]``. Values equal to
            ``ignore_index`` do not participate.
        predictions: CPU ``torch.long`` tensor ``[N]``. Predictions paired
            with ignored targets are deliberately neither validated nor
            counted.
        num_classes: Configured class count ``K >= 2``.
        ignore_index: Integer outside ``[0, K)``.

    Returns:
        :class:`ClassificationMetrics` with independent CPU long confusion
        matrix ``[K, K]`` and support ``[K]``. Matrix rows are true labels and
        columns are predictions. Empty and all-ignored inputs return finite
        zero metrics.

    Raises:
        TypeError: If tensor, dtype, or integer contracts are invalid.
        ValueError: If shapes/devices mismatch, ``ignore_index`` conflicts
            with a class, or a non-ignored target/prediction is out of range.

    Inputs are never modified. The implementation uses tensor bincount and
    does not require NumPy or scikit-learn.
    """
    _validate_metric_inputs(
        targets,
        predictions,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    evaluation_mask = targets != ignore_index
    valid_targets = targets[evaluation_mask]
    valid_predictions = predictions[evaluation_mask]
    if bool(
        ((valid_targets < 0) | (valid_targets >= num_classes)).any()
    ):
        raise ValueError(
            f"non-ignored targets must lie in [0, {num_classes})."
        )
    if bool(
        ((valid_predictions < 0) | (valid_predictions >= num_classes)).any()
    ):
        raise ValueError(
            f"predictions paired with valid targets must lie in "
            f"[0, {num_classes})."
        )

    flat_indices = valid_targets * num_classes + valid_predictions
    matrix = torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    support = matrix.sum(dim=1)
    true_positives = matrix.diagonal()
    evaluated_count = int(valid_targets.numel())
    correct_count = int(true_positives.sum().item())

    matrix_float = matrix.to(dtype=torch.float64)
    true_positive_float = matrix_float.diagonal()
    support_float = matrix_float.sum(dim=1)
    predicted_float = matrix_float.sum(dim=0)
    precision = _safe_ratio(true_positive_float, predicted_float)
    recall = _safe_ratio(true_positive_float, support_float)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    supported = support_float > 0

    accuracy = (
        float(correct_count / evaluated_count)
        if evaluated_count > 0
        else 0.0
    )
    balanced_accuracy = (
        float(recall[supported].mean().item()) if bool(supported.any()) else 0.0
    )
    return ClassificationMetrics(
        num_classes=num_classes,
        evaluated_count=evaluated_count,
        correct_count=correct_count,
        confusion_matrix=matrix,
        class_support=support,
        accuracy=accuracy,
        macro_precision=float(precision.mean().item()),
        macro_recall=float(recall.mean().item()),
        macro_f1=float(f1.mean().item()),
        balanced_accuracy=balanced_accuracy,
    )


@dataclass(frozen=True)
class EmotionTaskMetrics:
    """Metrics for arousal ``[2]``, valence ``[2]``, and quadrant ``[4]``."""

    arousal: ClassificationMetrics
    valence: ClassificationMetrics
    quadrant: ClassificationMetrics

    def __post_init__(self) -> None:
        expected = (("arousal", self.arousal, 2), ("valence", self.valence, 2))
        for name, value, class_count in (
            *expected,
            ("quadrant", self.quadrant, 4),
        ):
            if not isinstance(value, ClassificationMetrics):
                raise TypeError(f"{name} must be ClassificationMetrics.")
            if value.num_classes != class_count:
                raise ValueError(
                    f"{name} metrics must use {class_count} classes."
                )


def compute_emotion_task_metrics(
    *,
    arousal_targets: Tensor,
    arousal_predictions: Tensor,
    valence_targets: Tensor,
    valence_predictions: Tensor,
    quadrant_targets: Tensor,
    quadrant_predictions: Tensor,
    ignore_index: int,
) -> EmotionTaskMetrics:
    """Compute emotion metrics from six CPU long vectors ``[N]``.

    Each task independently applies ``target != ignore_index``. Arousal and
    valence use two classes; quadrant uses four in
    ``[LALV, HALV, LAHV, HAHV]`` order. Inputs are not modified. Invalid
    tensor, dtype, device, shape, label, prediction, or ignore-index contracts
    raise :class:`TypeError` or :class:`ValueError`.
    """
    task_inputs = (
        (arousal_targets, arousal_predictions, 2),
        (valence_targets, valence_predictions, 2),
        (quadrant_targets, quadrant_predictions, 4),
    )
    metrics = tuple(
        compute_classification_metrics(
            targets,
            predictions,
            num_classes=class_count,
            ignore_index=ignore_index,
        )
        for targets, predictions, class_count in task_inputs
    )
    return EmotionTaskMetrics(
        arousal=metrics[0],
        valence=metrics[1],
        quadrant=metrics[2],
    )


__all__ = [
    "ClassificationMetrics",
    "EmotionTaskMetrics",
    "compute_classification_metrics",
    "compute_emotion_task_metrics",
]
