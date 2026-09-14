"""Top-level multimodal emotion recognition components."""

from emotion_model.multimodal.classifier import (
    MultimodalEmotionClassifier,
    MultimodalEmotionClassifierOutput,
)
from emotion_model.multimodal.full_window_dynamic_fusion import (
    FullWindowDynamicMultimodalFusion,
)
from emotion_model.multimodal.losses import (
    class_weighted_cross_entropy,
    multiclass_focal_loss,
)
from emotion_model.multimodal.fusion_output import (
    MultimodalFusionOutput,
)
from emotion_model.multimodal.routing import (
    ModalityAvailabilityMasks,
    MultimodalBatchScheduler,
    ScatteredModalityPrediction,
    ScheduledModalityOutputs,
    scatter_compact_rows,
)
__all__ = [
    "ModalityAvailabilityMasks",
    "FullWindowDynamicMultimodalFusion",
    "MultimodalBatchScheduler",
    "MultimodalEmotionClassifier",
    "MultimodalEmotionClassifierOutput",
    "MultimodalFusionOutput",
    "ScatteredModalityPrediction",
    "ScheduledModalityOutputs",
    "class_weighted_cross_entropy",
    "multiclass_focal_loss",
    "scatter_compact_rows",
]
