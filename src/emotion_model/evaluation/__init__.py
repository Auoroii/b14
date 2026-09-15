"""Evaluation utilities for emotion recognition."""

from emotion_model.evaluation.activity import (
    FusionWeightStatistics,
    SpeechActivityBin,
    SpeechActivityStratumMetrics,
    compute_speech_activity_strata,
    speech_activity_bin_masks,
)
from emotion_model.evaluation.calibration import (
    BinaryDecisionThresholds,
    calibrate_binary_decision_thresholds,
    select_pooled_macro_f1_threshold,
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
    "FusionWeightStatistics",
    "ModalityPattern",
    "ModalityStratumMetrics",
    "ParticipantEvaluationMetrics",
    "ParticipantEvaluationScope",
    "ParticipantIndependentEvaluationOutput",
    "ParticipantMacroEmotionMetrics",
    "ParticipantMacroTaskMetrics",
    "SpeechActivityBin",
    "SpeechActivityStratumMetrics",
    "compute_classification_metrics",
    "compute_emotion_task_metrics",
    "compute_speech_activity_strata",
    "calibrate_binary_decision_thresholds",
    "evaluate_participant_independent",
    "select_pooled_macro_f1_threshold",
    "speech_activity_bin_masks",
]
