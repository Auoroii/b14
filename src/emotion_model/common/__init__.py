"""Shared label, mask, and pooling utilities for emotion recognition."""

from emotion_model.common.labels import (
    LabelProtocol,
    binarize_emotion_scores,
    derive_quadrant_labels,
    derive_quadrant_probabilities,
)
from emotion_model.common.masking import (
    apply_query_mask,
    safe_masked_softmax,
    valid_to_key_padding_mask,
    validate_sequence_mask,
)
from emotion_model.common.pooling import (
    MaskAwareAttentiveStatisticsPooling,
    masked_mean,
    masked_mean_std,
    masked_population_std,
    masked_population_variance,
)

__all__ = [
    "LabelProtocol",
    "MaskAwareAttentiveStatisticsPooling",
    "apply_query_mask",
    "binarize_emotion_scores",
    "derive_quadrant_labels",
    "derive_quadrant_probabilities",
    "masked_mean",
    "masked_mean_std",
    "masked_population_std",
    "masked_population_variance",
    "safe_masked_softmax",
    "valid_to_key_padding_mask",
    "validate_sequence_mask",
]
