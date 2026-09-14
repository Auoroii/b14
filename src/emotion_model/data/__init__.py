"""Strict data contracts, time alignment, and participant split utilities."""

from emotion_model.data.alignment import (
    regular_sample_span,
    source_overlap,
    timestamp_index_span,
    validate_record_alignment,
)
from emotion_model.data.batch_transforms import (
    apply_modality_keep_masks,
    select_aligned_multimodal_batch,
)
from emotion_model.data.collate import (
    AlignedMultimodalBatch,
    PhysioSubBatch,
    SpeechSubBatch,
    collate_aligned_multimodal_samples,
)
from emotion_model.data.dataset import (
    AlignedMultimodalDataset,
    AlignedMultimodalSample,
)
from emotion_model.data.kemocon import (
    KEmoConManifestBuildReport,
    KEmoConPhysioAdapter,
    KEmoConSpeechAdapter,
    build_kemocon_dyad_ratio_split,
    build_kemocon_dyad_splits,
    build_kemocon_label_stratified_dyad_splits,
    build_kemocon_manifest,
    kemocon_channel_specs,
)
from emotion_model.data.manifest import (
    EmotionScores,
    IndexSpan,
    MultimodalManifest,
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
    TimedSourceRef,
    TimeInterval,
    manifest_from_dict,
    manifest_to_dict,
    read_manifest_json,
    write_manifest_json,
)
from emotion_model.data.source_adapters import (
    NormalizationKeyResolver,
    PhysioArtifactPolicy,
    PhysioSourceAdapter,
    SourceAdapterError,
    SpeechSourceAdapter,
    SpeechSourceData,
)
from emotion_model.data.speech_activity import (
    SpeechActivityDetectorConfig,
    detect_frame_rms_speech_activity,
    summarize_speech_activity_ratios,
)
from emotion_model.data.splits import (
    DatasetPartition,
    ParticipantSplit,
    PartitionedManifest,
    build_deterministic_participant_folds,
    build_rotating_participant_splits,
    partition_manifest_by_participant,
)

__all__ = [
    "AlignedMultimodalDataset",
    "AlignedMultimodalBatch",
    "AlignedMultimodalSample",
    "DatasetPartition",
    "EmotionScores",
    "IndexSpan",
    "KEmoConManifestBuildReport",
    "KEmoConPhysioAdapter",
    "KEmoConSpeechAdapter",
    "MultimodalManifest",
    "MultimodalWindowRecord",
    "NormalizationKeyResolver",
    "ParticipantSplit",
    "PartitionedManifest",
    "PhysioArtifactPolicy",
    "PhysioChannelSourceRef",
    "PhysioSubBatch",
    "PhysioSourceAdapter",
    "SourceAdapterError",
    "SpeechSourceAdapter",
    "SpeechSourceData",
    "SpeechActivityDetectorConfig",
    "detect_frame_rms_speech_activity",
    "summarize_speech_activity_ratios",
    "SpeechSubBatch",
    "TimeInterval",
    "TimedSourceRef",
    "apply_modality_keep_masks",
    "build_deterministic_participant_folds",
    "build_kemocon_dyad_ratio_split",
    "build_kemocon_dyad_splits",
    "build_kemocon_label_stratified_dyad_splits",
    "build_kemocon_manifest",
    "build_rotating_participant_splits",
    "collate_aligned_multimodal_samples",
    "manifest_from_dict",
    "manifest_to_dict",
    "kemocon_channel_specs",
    "partition_manifest_by_participant",
    "read_manifest_json",
    "regular_sample_span",
    "select_aligned_multimodal_batch",
    "source_overlap",
    "timestamp_index_span",
    "validate_record_alignment",
    "write_manifest_json",
]
