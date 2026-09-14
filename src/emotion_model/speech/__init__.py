"""V4.2 speech-modality components for emotion recognition."""

from emotion_model.speech.emotion_layer_aggregation import (
    EMOTION_LAYER_AGGREGATION_MODES,
    EMOTION_LAYER_INDICES,
    EMOTION_LAYER_NAMES,
    EmotionLayerAggregation,
)
from emotion_model.speech.lightweight_noise_conditioned import (
    LightweightNoiseConditionedSpeechClassifier,
    LightweightSpeechClassifierOutput,
)
from emotion_model.speech.relation_differential_denoising import (
    NoiseConditionedRelationDifferentialDenoiser,
    RelationDifferentialDenoisingOutput,
)
from emotion_model.speech.wavlm_encoder import (
    WavLMEncoder,
    WavLMEncoderOutput,
)

__all__ = [
    "EMOTION_LAYER_AGGREGATION_MODES",
    "EMOTION_LAYER_INDICES",
    "EMOTION_LAYER_NAMES",
    "EmotionLayerAggregation",
    "LightweightNoiseConditionedSpeechClassifier",
    "LightweightSpeechClassifierOutput",
    "NoiseConditionedRelationDifferentialDenoiser",
    "RelationDifferentialDenoisingOutput",
    "WavLMEncoder",
    "WavLMEncoderOutput",
]
