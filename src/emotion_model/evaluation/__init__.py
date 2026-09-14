"""Evaluation utilities for emotion recognition."""

from emotion_model.evaluation.calibration import (
    BinaryDecisionThresholds,
    calibrate_binary_decision_thresholds,
    select_equal_participant_macro_f1_threshold,
)
from emotion_model.evaluation.metrics import (
    ClassificationMetrics,
    EmotionTaskMetrics,
    compute_classification_metrics,
    compute_emotion_task_metrics,
)
from emotion_model.evaluation.runner import (
    EvaluationPredictions,
    ModalityPattern,
    ModalityStratumMetrics,
    ParticipantEvaluationMetrics,
    ParticipantEvaluationScope,
    ParticipantIndependentEvaluationOutput,
    ParticipantMacroEmotionMetrics,
    ParticipantMacroTaskMetrics,
    evaluate_participant_independent,
)

__all__ = [
    "BinaryDecisionThresholds",
    "ClassificationMetrics",
    "EmotionTaskMetrics",
    "EvaluationPredictions",
    "ModalityPattern",
    "ModalityStratumMetrics",
    "ParticipantEvaluationMetrics",
    "ParticipantEvaluationScope",
    "ParticipantIndependentEvaluationOutput",
    "ParticipantMacroEmotionMetrics",
    "ParticipantMacroTaskMetrics",
    "compute_classification_metrics",
    "compute_emotion_task_metrics",
    "calibrate_binary_decision_thresholds",
    "evaluate_participant_independent",
    "select_equal_participant_macro_f1_threshold",
]
