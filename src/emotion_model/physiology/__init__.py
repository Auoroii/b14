"""Physiology contracts, preprocessing, and the V4.2 classifier."""

from emotion_model.physiology.channel_metadata import (
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)
from emotion_model.physiology.lightweight import (
    ECG_ENCODER_TYPE,
    LightweightEcgEncoder,
    LightweightPhysioClassifierOutput,
    LightweightPhysioEmotionClassifier,
    MultiScaleDilatedConv1dStem,
    PHYSIOLOGY_ENCODER_MODES,
)
from emotion_model.physiology.normalization import (
    ChannelNormalizationStats,
    ChannelwiseZScoreNormalizer,
    NormalizationFitScope,
    NormalizationKey,
)
from emotion_model.physiology.preprocessing import (
    PhysioFilter,
    apply_channel_filter,
    combine_invalid_masks,
    detect_flatline_mask,
    detect_robust_outlier_mask,
    resample_channel_to_timeline,
    zero_invalid_values,
)
from emotion_model.physiology.quality import (
    PhysioChannelQuality,
    compute_channel_quality,
)

__all__ = [
    "ChannelNormalizationStats",
    "ChannelwiseZScoreNormalizer",
    "ECG_ENCODER_TYPE",
    "LightweightEcgEncoder",
    "LightweightPhysioClassifierOutput",
    "LightweightPhysioEmotionClassifier",
    "MultiScaleDilatedConv1dStem",
    "NormalizationFitScope",
    "NormalizationKey",
    "PhysioChannelQuality",
    "PHYSIOLOGY_ENCODER_MODES",
    "PhysioChannelSpec",
    "PhysioChannelWindow",
    "PhysioFilter",
    "PhysioSignalKind",
    "apply_channel_filter",
    "combine_invalid_masks",
    "compute_channel_quality",
    "detect_flatline_mask",
    "detect_robust_outlier_mask",
    "resample_channel_to_timeline",
    "zero_invalid_values",
]
