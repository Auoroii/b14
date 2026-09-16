"""CPU-only aligned multimodal Dataset with injectable source adapters.

Records must already have been selected from one participant partition.
This Dataset does not partition records and cannot detect if a caller mixes
train, validation, or test records. Likewise, an injected normalizer must
already be fitted from an authorized source; this module calls only
``transform`` and cannot prove real-world leakage is absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from emotion_model.common.labels import (
    LabelProtocol,
    binarize_emotion_scores,
    derive_quadrant_labels,
)
from emotion_model.data.alignment import (
    regular_sample_span,
    source_overlap,
    timestamp_index_span,
)
from emotion_model.data.manifest import (
    MultimodalWindowRecord,
    PhysioChannelSourceRef,
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
)
from emotion_model.physiology.channel_metadata import (
    PhysioChannelSpec,
    PhysioChannelWindow,
)
from emotion_model.physiology.normalization import (
    ChannelwiseZScoreNormalizer,
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
from emotion_model.physiology.quality import compute_channel_quality

_TIMELINE_TOLERANCE = 1.0e-9


def _require_cpu_tensor(value: Tensor, *, name: str) -> None:
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU.")


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool.")
    return value


def _is_string_like(value: object) -> bool:
    return isinstance(value, (str, bytes))


@dataclass(frozen=True)
class AlignedMultimodalSample:
    """One unbatched aligned sample ready for a future collate function.

    Speech waveform, attention, and optional activity fields are ``[L]``;
    physiology fields are ``[T, C]``, ``[T]``, ``[C]``, ``[C, 6]``, and
    ``[6*C]``. Every mask uses ``True=valid``. Speech attention is a strict
    prefix, while participant activity may contain internal holes.
    Physiology quality contains observable statistics, not a quality prior or
    calibrated reliability value.
    """

    record: MultimodalWindowRecord
    raw_arousal: float
    raw_valence: float
    arousal_label: int
    valence_label: int
    quadrant_label: int
    label_ignore_index: int
    speech_waveform: Tensor | None
    speech_attention_mask: Tensor | None
    speech_sample_rate_hz: int | None
    speech_available: bool
    physio_input: Tensor | None
    physio_valid_mask: Tensor | None
    physio_time_mask: Tensor | None
    physio_channel_mask: Tensor | None
    physio_timestamps_seconds: Tensor | None
    physio_channel_quality: Tensor | None
    physio_quality_features: Tensor | None
    physiology_available: bool
    channel_names: tuple[str, ...]
    speech_activity_mask: Tensor | None = None
    speech_activity_ratio: float | None = None
    ecg_values: Tensor | None = None
    ecg_valid_mask: Tensor | None = None
    ecg_timestamps_seconds: Tensor | None = None
    ecg_available: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.record, MultimodalWindowRecord):
            raise TypeError("record must be a MultimodalWindowRecord.")
        for name, value in (
            ("raw_arousal", self.raw_arousal),
            ("raw_valence", self.raw_valence),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if (
            self.raw_arousal != self.record.emotion_scores.arousal
            or self.raw_valence != self.record.emotion_scores.valence
        ):
            raise ValueError("raw scores must match record.emotion_scores.")
        if isinstance(self.label_ignore_index, bool) or not isinstance(
            self.label_ignore_index,
            int,
        ):
            raise TypeError("label_ignore_index must be an integer, not bool.")
        if self.label_ignore_index in (0, 1):
            raise ValueError("label_ignore_index must differ from 0 and 1.")
        for name, value in (
            ("arousal_label", self.arousal_label),
            ("valence_label", self.valence_label),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value not in (0, 1, self.label_ignore_index):
                raise ValueError(f"{name} must be 0, 1, or label_ignore_index.")
        if isinstance(self.quadrant_label, bool) or not isinstance(
            self.quadrant_label,
            int,
        ):
            raise TypeError("quadrant_label must be an integer.")
        if self.quadrant_label not in (0, 1, 2, 3, self.label_ignore_index):
            raise ValueError("quadrant_label must be 0..3 or label_ignore_index.")
        _require_bool(self.speech_available, name="speech_available")
        _require_bool(self.physiology_available, name="physiology_available")
        _require_bool(self.ecg_available, name="ecg_available")
        if not isinstance(self.channel_names, tuple) or not all(
            isinstance(name, str) for name in self.channel_names
        ):
            raise TypeError("channel_names must be a tuple of strings.")
        self._validate_speech_fields()
        self._validate_speech_activity_diagnostics()
        self._validate_physio_fields()
        self._validate_ecg_fields()

    def _validate_speech_fields(self) -> None:
        fields = (
            self.speech_waveform,
            self.speech_attention_mask,
            self.speech_sample_rate_hz,
            self.speech_activity_mask,
        )
        if not self.speech_available:
            if any(value is not None for value in fields):
                raise ValueError("unavailable speech requires all speech fields to be None.")
            return
        if not isinstance(self.speech_waveform, Tensor):
            raise TypeError("speech_waveform must be a Tensor when speech is available.")
        if self.speech_waveform.ndim != 1 or self.speech_waveform.numel() == 0:
            raise ValueError("speech_waveform must have non-empty shape [L].")
        if not self.speech_waveform.is_floating_point():
            raise TypeError("speech_waveform must be floating point.")
        _require_cpu_tensor(self.speech_waveform, name="speech_waveform")
        if not bool(torch.isfinite(self.speech_waveform).all()):
            raise ValueError("speech_waveform must be finite.")
        if not isinstance(self.speech_attention_mask, Tensor):
            raise TypeError("speech_attention_mask must be a Tensor.")
        if (
            self.speech_attention_mask.dtype != torch.bool
            or tuple(self.speech_attention_mask.shape)
            != tuple(self.speech_waveform.shape)
            or not bool(self.speech_attention_mask.all())
        ):
            raise ValueError(
                "speech_attention_mask must be all-True bool with shape [L]."
            )
        _require_cpu_tensor(
            self.speech_attention_mask,
            name="speech_attention_mask",
        )
        if self.speech_activity_mask is not None:
            if not isinstance(self.speech_activity_mask, Tensor):
                raise TypeError("speech_activity_mask must be a Tensor or None.")
            if (
                self.speech_activity_mask.dtype != torch.bool
                or tuple(self.speech_activity_mask.shape)
                != tuple(self.speech_waveform.shape)
            ):
                raise ValueError(
                    "speech_activity_mask must be bool with shape [L]."
                )
            _require_cpu_tensor(
                self.speech_activity_mask,
                name="speech_activity_mask",
            )
            if bool(
                (
                    self.speech_activity_mask
                    & ~self.speech_attention_mask
                ).any()
            ):
                raise ValueError(
                    "speech_activity_mask must be a subset of "
                    "speech_attention_mask."
                )
        if (
            isinstance(self.speech_sample_rate_hz, bool)
            or not isinstance(self.speech_sample_rate_hz, int)
            or self.speech_sample_rate_hz <= 0
        ):
            raise ValueError("speech_sample_rate_hz must be a positive integer.")

    def _validate_speech_activity_diagnostics(self) -> None:
        ratio = self.speech_activity_ratio
        if ratio is not None:
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError("speech_activity_ratio must be a real number or None.")
            if not math.isfinite(float(ratio)) or not 0.0 <= float(ratio) <= 1.0:
                raise ValueError("speech_activity_ratio must lie in [0, 1].")
        if self.speech_available and self.speech_activity_mask is not None:
            assert self.speech_attention_mask is not None
            expected_ratio = float(
                self.speech_activity_mask.sum().item()
                / self.speech_attention_mask.sum().item()
            )
            if ratio is not None and not math.isclose(
                float(ratio),
                expected_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "speech_activity_ratio must match the waveform masks."
                )

    def _validate_physio_fields(self) -> None:
        fields = (
            self.physio_input,
            self.physio_valid_mask,
            self.physio_time_mask,
            self.physio_channel_mask,
            self.physio_timestamps_seconds,
            self.physio_channel_quality,
            self.physio_quality_features,
        )
        if not self.channel_names:
            if any(value is not None for value in fields):
                raise ValueError(
                    "empty channel_names requires all physiology fields to be None."
                )
            if self.physiology_available and not self.ecg_available:
                raise ValueError(
                    "physiology cannot be available without a dense channel or ECG."
                )
            return
        if not all(isinstance(value, Tensor) for value in fields):
            raise TypeError(
                "all physiology tensor fields are required when channel_names is non-empty."
            )
        assert isinstance(self.physio_input, Tensor)
        assert isinstance(self.physio_valid_mask, Tensor)
        assert isinstance(self.physio_time_mask, Tensor)
        assert isinstance(self.physio_channel_mask, Tensor)
        assert isinstance(self.physio_timestamps_seconds, Tensor)
        assert isinstance(self.physio_channel_quality, Tensor)
        assert isinstance(self.physio_quality_features, Tensor)
        channel_count = len(self.channel_names)
        if self.physio_input.ndim != 2:
            raise ValueError("physio_input must have shape [T, C].")
        time_count = self.physio_input.shape[0]
        expected_matrix_shape = (time_count, channel_count)
        if tuple(self.physio_input.shape) != expected_matrix_shape:
            raise ValueError(
                f"physio_input must have shape {expected_matrix_shape}."
            )
        if not self.physio_input.is_floating_point():
            raise TypeError("physio_input must be floating point.")
        if (
            self.physio_valid_mask.dtype != torch.bool
            or tuple(self.physio_valid_mask.shape) != expected_matrix_shape
        ):
            raise ValueError("physio_valid_mask must be bool with shape [T, C].")
        if (
            self.physio_time_mask.dtype != torch.bool
            or tuple(self.physio_time_mask.shape) != (time_count,)
            or not torch.equal(
                self.physio_time_mask,
                self.physio_valid_mask.any(dim=1),
            )
        ):
            raise ValueError("physio_time_mask must equal valid_mask.any(dim=1).")
        if (
            self.physio_channel_mask.dtype != torch.bool
            or tuple(self.physio_channel_mask.shape) != (channel_count,)
            or not torch.equal(
                self.physio_channel_mask,
                self.physio_valid_mask.any(dim=0),
            )
        ):
            raise ValueError(
                "physio_channel_mask must equal valid_mask.any(dim=0)."
            )
        if (
            self.physio_timestamps_seconds.dtype != torch.float64
            or tuple(self.physio_timestamps_seconds.shape) != (time_count,)
        ):
            raise ValueError(
                "physio_timestamps_seconds must be float64 with shape [T]."
            )
        if not bool(torch.isfinite(self.physio_timestamps_seconds).all()):
            raise ValueError("physio_timestamps_seconds must be finite.")
        if time_count == 0 or (
            time_count > 1
            and not bool(
                (
                    self.physio_timestamps_seconds[1:]
                    > self.physio_timestamps_seconds[:-1]
                ).all()
            )
        ):
            raise ValueError("physio timestamps must be non-empty and increasing.")
        if tuple(self.physio_channel_quality.shape) != (channel_count, 6):
            raise ValueError("physio_channel_quality must have shape [C, 6].")
        if tuple(self.physio_quality_features.shape) != (6 * channel_count,):
            raise ValueError("physio_quality_features must have shape [6*C].")
        if (
            self.physio_channel_quality.dtype != self.physio_input.dtype
            or self.physio_quality_features.dtype != self.physio_input.dtype
        ):
            raise TypeError(
                "physiology input and quality tensors must share output dtype."
            )
        if not torch.equal(
            self.physio_quality_features,
            self.physio_channel_quality.reshape(-1),
        ):
            raise ValueError("physio_quality_features must be channel-major flattening.")
        for name, tensor in (
            ("physio_input", self.physio_input),
            ("physio_valid_mask", self.physio_valid_mask),
            ("physio_time_mask", self.physio_time_mask),
            ("physio_channel_mask", self.physio_channel_mask),
            ("physio_timestamps_seconds", self.physio_timestamps_seconds),
            ("physio_channel_quality", self.physio_channel_quality),
            ("physio_quality_features", self.physio_quality_features),
        ):
            _require_cpu_tensor(tensor, name=name)
        if not bool(torch.isfinite(self.physio_input).all()):
            raise ValueError("physio_input must be finite.")
        invalid_values = torch.where(
            self.physio_valid_mask,
            torch.zeros_like(self.physio_input),
            self.physio_input,
        )
        if bool(torch.count_nonzero(invalid_values)):
            raise ValueError("physio_input must be zero at invalid positions.")
        if not bool(
            torch.isfinite(self.physio_channel_quality).all()
            and (self.physio_channel_quality >= 0.0).all()
            and (self.physio_channel_quality <= 1.0).all()
        ):
            raise ValueError("physio quality values must be finite in [0, 1].")
        expected_available = bool(self.physio_valid_mask.any()) or self.ecg_available
        if self.physiology_available != expected_available:
            raise ValueError(
                "physiology_available must equal dense-channel OR ECG availability."
            )

    def _validate_ecg_fields(self) -> None:
        """Validate optional low-frequency ECG-HR tensors with shape ``[Te]``."""
        fields = (self.ecg_values, self.ecg_valid_mask, self.ecg_timestamps_seconds)
        if all(value is None for value in fields):
            if self.ecg_available:
                raise ValueError("ecg_available requires ECG tensors.")
            return
        if not all(isinstance(value, Tensor) for value in fields):
            raise TypeError("ECG values, mask, and timestamps must be provided together.")
        assert isinstance(self.ecg_values, Tensor)
        assert isinstance(self.ecg_valid_mask, Tensor)
        assert isinstance(self.ecg_timestamps_seconds, Tensor)
        if self.ecg_values.ndim != 1 or self.ecg_values.numel() <= 0:
            raise ValueError("ecg_values must have non-empty shape [Te].")
        if not self.ecg_values.is_floating_point():
            raise TypeError("ecg_values must be floating point.")
        if (
            self.ecg_valid_mask.dtype != torch.bool
            or tuple(self.ecg_valid_mask.shape) != tuple(self.ecg_values.shape)
        ):
            raise ValueError("ecg_valid_mask must be bool with shape [Te].")
        if (
            self.ecg_timestamps_seconds.dtype != torch.float64
            or tuple(self.ecg_timestamps_seconds.shape) != tuple(self.ecg_values.shape)
        ):
            raise ValueError("ecg_timestamps_seconds must be float64 with shape [Te].")
        for name, tensor in (
            ("ecg_values", self.ecg_values),
            ("ecg_valid_mask", self.ecg_valid_mask),
            ("ecg_timestamps_seconds", self.ecg_timestamps_seconds),
        ):
            _require_cpu_tensor(tensor, name=name)
        if not bool(torch.isfinite(self.ecg_values).all()):
            raise ValueError("ecg_values must be finite.")
        if bool(torch.count_nonzero(self.ecg_values[~self.ecg_valid_mask])):
            raise ValueError("ecg_values must be zero at invalid positions.")
        if not bool(torch.isfinite(self.ecg_timestamps_seconds).all()):
            raise ValueError("ECG timestamps must be finite.")
        if self.ecg_values.numel() > 1 and not bool(
            (self.ecg_timestamps_seconds[1:] > self.ecg_timestamps_seconds[:-1]).all()
        ):
            raise ValueError("ECG timestamps must be strictly increasing.")
        if self.ecg_available != bool(self.ecg_valid_mask.any()):
            raise ValueError("ecg_available must equal bool(ecg_valid_mask.any()).")


class AlignedMultimodalDataset(Dataset[AlignedMultimodalSample]):
    """Load and align one manifest record at a time on CPU.

    Args:
        records: Non-empty ordered sequence already selected from one experiment
            partition. The Dataset does not verify partition provenance.
        channel_specs: Ordered physiology channel semantics.
        label_protocol: Explicit public binary label protocol.
        speech_adapter: Optional structural adapter with callable
            ``load_speech``.
        physio_adapter: Optional structural adapter with callable
            ``load_physio``.
        required_speech_sample_rate_hz: Required positive integer speech rate.
        physio_target_sample_rate_hz: Positive common physiology alignment rate
            when channel specs exist.
        output_dtype: ``torch.float32`` or ``torch.float64``.
        physio_filters: Optional per-channel generic filter mapping.
        artifact_policy: Optional explicit generic detector policy. ``None``
            runs no detector and excludes no values.
        physio_normalizer: Optional already-fitted normalizer. It is never fit.
        normalization_key_resolver: Optional explicit key resolver. Without it,
            a global key for the current channel is used.
        label_ignore_index: Integer ignored-label value distinct from 0 and 1.
        enable_speech: Whether declared speech sources are loaded and returned.
        enable_physiology: Whether declared physiology sources are loaded and
            returned. At least one modality must be enabled.
        speech_activity_config: Optional clean-waveform frame-RMS detector and
            minimum active-duration policy. ``None`` preserves legacy speech.

    Source loading occurs only in :meth:`__getitem__`. No batching, caching,
    model execution, participant splitting, speech resampling, or amplitude
    normalization occurs here. Physiology interpolation is generic linear
    timestamp alignment, not channel-specific or medical-grade processing.
    """

    def __init__(
        self,
        records: Sequence[MultimodalWindowRecord],
        channel_specs: Sequence[PhysioChannelSpec],
        *,
        label_protocol: LabelProtocol,
        speech_adapter: SpeechSourceAdapter | None = None,
        physio_adapter: PhysioSourceAdapter | None = None,
        required_speech_sample_rate_hz: int = 16000,
        physio_target_sample_rate_hz: float | None = None,
        ecg_sample_rate_hz: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
        physio_filters: Mapping[str, PhysioFilter] | None = None,
        artifact_policy: PhysioArtifactPolicy | None = None,
        physio_normalizer: ChannelwiseZScoreNormalizer | None = None,
        normalization_key_resolver: NormalizationKeyResolver | None = None,
        label_ignore_index: int = -100,
        enable_speech: bool = True,
        enable_physiology: bool = True,
        speech_activity_config: SpeechActivityDetectorConfig | None = None,
    ) -> None:
        if _is_string_like(records) or not isinstance(records, Sequence):
            raise TypeError("records must be a non-string Sequence.")
        if not records:
            raise ValueError("records must be non-empty.")
        record_tuple = tuple(records)
        if not all(isinstance(record, MultimodalWindowRecord) for record in record_tuple):
            raise TypeError("records must contain MultimodalWindowRecord objects.")
        sample_ids = tuple(record.sample_id for record in record_tuple)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("records sample_id values must be unique.")
        if _is_string_like(channel_specs) or not isinstance(
            channel_specs,
            Sequence,
        ):
            raise TypeError("channel_specs must be a non-string Sequence.")
        spec_tuple = tuple(channel_specs)
        if not all(isinstance(spec, PhysioChannelSpec) for spec in spec_tuple):
            raise TypeError("channel_specs must contain PhysioChannelSpec objects.")
        all_channel_names = tuple(spec.name for spec in spec_tuple)
        if len(set(all_channel_names)) != len(all_channel_names):
            raise ValueError("channel_specs names must be unique.")
        known_channels = set(all_channel_names)
        ecg_specs = tuple(spec for spec in spec_tuple if spec.name == "ecg")
        if len(ecg_specs) > 1:
            raise ValueError("at most one ecg channel specification is allowed.")
        dense_specs = tuple(spec for spec in spec_tuple if spec.name != "ecg")
        channel_names = tuple(spec.name for spec in dense_specs)
        for record in record_tuple:
            for channel_source in record.physio_sources:
                if channel_source.channel_name not in known_channels:
                    raise ValueError(
                        f"record {record.sample_id!r} references unknown physiology "
                        f"channel {channel_source.channel_name!r}."
                    )
        if not isinstance(label_protocol, LabelProtocol):
            raise TypeError("label_protocol must be a LabelProtocol.")
        if (
            isinstance(required_speech_sample_rate_hz, bool)
            or not isinstance(required_speech_sample_rate_hz, int)
        ):
            raise TypeError(
                "required_speech_sample_rate_hz must be an integer, not bool."
            )
        if required_speech_sample_rate_hz <= 0:
            raise ValueError("required_speech_sample_rate_hz must be > 0.")
        if output_dtype not in (torch.float32, torch.float64):
            raise ValueError("output_dtype must be torch.float32 or torch.float64.")
        if isinstance(label_ignore_index, bool) or not isinstance(
            label_ignore_index,
            int,
        ):
            raise TypeError("label_ignore_index must be an integer, not bool.")
        if label_ignore_index in (0, 1):
            raise ValueError("label_ignore_index must differ from 0 and 1.")
        if not isinstance(enable_speech, bool):
            raise TypeError("enable_speech must be bool.")
        if not isinstance(enable_physiology, bool):
            raise TypeError("enable_physiology must be bool.")
        if not enable_speech and not enable_physiology:
            raise ValueError("at least one modality must be enabled.")
        if artifact_policy is not None and not isinstance(
            artifact_policy,
            PhysioArtifactPolicy,
        ):
            raise TypeError("artifact_policy must be PhysioArtifactPolicy or None.")
        if speech_activity_config is not None and not isinstance(
            speech_activity_config,
            SpeechActivityDetectorConfig,
        ):
            raise TypeError(
                "speech_activity_config must be SpeechActivityDetectorConfig "
                "or None."
            )
        if physio_normalizer is not None and not isinstance(
            physio_normalizer,
            ChannelwiseZScoreNormalizer,
        ):
            raise TypeError(
                "physio_normalizer must be ChannelwiseZScoreNormalizer or None."
            )
        if physio_normalizer is None and normalization_key_resolver is not None:
            raise ValueError(
                "normalization_key_resolver requires physio_normalizer."
            )
        if normalization_key_resolver is not None and not callable(
            normalization_key_resolver
        ):
            raise TypeError("normalization_key_resolver must be callable.")
        self._validate_adapter(speech_adapter, method_name="load_speech")
        self._validate_adapter(physio_adapter, method_name="load_physio")
        if enable_speech and any(
            record.speech_source is not None for record in record_tuple
        ):
            if speech_adapter is None:
                raise ValueError("speech_adapter is required for declared speech sources.")
        if enable_physiology and any(record.physio_sources for record in record_tuple):
            if physio_adapter is None:
                raise ValueError(
                    "physio_adapter is required for declared physiology sources."
                )

        if spec_tuple and enable_physiology:
            target_rate = self._validate_target_rate(physio_target_sample_rate_hz)
            ecg_rate = self._validate_ecg_rate(ecg_sample_rate_hz)
        elif spec_tuple:
            if any(
                value is not None
                for value in (
                    physio_adapter,
                    physio_target_sample_rate_hz,
                    physio_filters,
                    artifact_policy,
                    physio_normalizer,
                    normalization_key_resolver,
                )
            ):
                raise ValueError(
                    "physiology configuration must be None when physiology is disabled."
                )
            target_rate = None
            ecg_rate = None
        else:
            invalid_empty_spec_config = (
                physio_target_sample_rate_hz is not None
                or physio_adapter is not None
                or physio_filters is not None
                or artifact_policy is not None
                or physio_normalizer is not None
                or normalization_key_resolver is not None
            )
            if invalid_empty_spec_config:
                raise ValueError(
                    "physiology configuration must be None when channel_specs is empty."
                )
            target_rate = None
            ecg_rate = None

        filters: dict[str, PhysioFilter] = {}
        if physio_filters is not None:
            if not isinstance(physio_filters, Mapping):
                raise TypeError("physio_filters must be a mapping or None.")
            for channel_name, filter_fn in physio_filters.items():
                if not isinstance(channel_name, str):
                    raise TypeError("physio_filters keys must be strings.")
                if channel_name not in known_channels:
                    raise ValueError(
                        f"physio_filters references unknown channel {channel_name!r}."
                    )
                if not callable(filter_fn):
                    raise TypeError(
                        f"physio filter for {channel_name!r} must be callable."
                    )
                filters[channel_name] = filter_fn

        self._records = record_tuple
        self._channel_specs = spec_tuple
        self._dense_channel_specs = dense_specs
        self._ecg_spec = ecg_specs[0] if ecg_specs else None
        self._channel_names = channel_names
        self._label_protocol = label_protocol
        self._speech_adapter = speech_adapter
        self._physio_adapter = physio_adapter
        self._required_speech_sample_rate_hz = required_speech_sample_rate_hz
        self._physio_target_sample_rate_hz = target_rate
        self._ecg_sample_rate_hz = ecg_rate
        self._output_dtype = output_dtype
        self._physio_filters = filters
        self._artifact_policy = artifact_policy
        self._physio_normalizer = physio_normalizer
        self._normalization_key_resolver = normalization_key_resolver
        self._label_ignore_index = label_ignore_index
        self._enable_speech = enable_speech
        self._enable_physiology = enable_physiology
        self._speech_activity_config = speech_activity_config

    @staticmethod
    def _validate_adapter(adapter: object, *, method_name: str) -> None:
        if adapter is not None and not callable(getattr(adapter, method_name, None)):
            raise TypeError(f"adapter must provide callable {method_name}().")

    @staticmethod
    def _validate_target_rate(rate: object) -> float:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise TypeError(
                "physio_target_sample_rate_hz must be a real number, not bool."
            )
        result = float(rate)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(
                "physio_target_sample_rate_hz must be finite and > 0."
            )
        return result

    @staticmethod
    def _validate_ecg_rate(rate: object) -> float:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise TypeError("ecg_sample_rate_hz must be a real number, not bool.")
        result = float(rate)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("ecg_sample_rate_hz must be finite and > 0.")
        return result

    @property
    def records(self) -> tuple[MultimodalWindowRecord, ...]:
        """Return records in the original caller-provided order."""
        return self._records

    @property
    def channel_specs(self) -> tuple[PhysioChannelSpec, ...]:
        """Return configured channel specifications in fixed output order."""
        return self._channel_specs

    def __len__(self) -> int:
        """Return the number of manifest records without loading any source."""
        return len(self._records)

    def __getitem__(self, index: int) -> AlignedMultimodalSample:
        """Load and align one record by standard integer sequence index.

        Args:
            index: Non-boolean integer index. Negative indices follow standard
                Python sequence semantics; slices are not supported.

        Returns:
            A new :class:`AlignedMultimodalSample` containing unbatched CPU
            tensors. Repeated access may call adapters again; no cache exists.

        Raises:
            TypeError: If ``index`` is bool, slice, or another non-integer.
            IndexError: If the normalized index lies outside the Dataset.
            SourceAdapterError: If a declared source fails to load or violates
                its adapter/data alignment contract.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer, not bool or slice.")
        normalized_index = index
        if normalized_index < 0:
            normalized_index += len(self)
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}.")
        record = self._records[normalized_index]
        arousal_label, valence_label, quadrant_label = self._labels_for(record)
        if self._enable_speech:
            (
                speech_waveform,
                speech_attention_mask,
                speech_sample_rate,
                speech_activity_mask,
                speech_activity_ratio,
            ) = self._load_speech(record)
        else:
            speech_waveform = None
            speech_attention_mask = None
            speech_sample_rate = None
            speech_activity_mask = None
            speech_activity_ratio = None
        if self._enable_physiology:
            (
                physio_input,
                physio_valid_mask,
                physio_time_mask,
                physio_channel_mask,
                physio_timestamps,
                physio_channel_quality,
                physio_quality_features,
            ) = self._load_physiology(record)
            (
                ecg_values,
                ecg_valid_mask,
                ecg_timestamps,
            ) = self._load_ecg(record)
        else:
            physio_input = None
            physio_valid_mask = None
            physio_time_mask = None
            physio_channel_mask = None
            physio_timestamps = None
            physio_channel_quality = None
            physio_quality_features = None
            ecg_values = None
            ecg_valid_mask = None
            ecg_timestamps = None
        ecg_available = (
            False if ecg_valid_mask is None else bool(ecg_valid_mask.any())
        )
        physiology_available = (
            (False if physio_valid_mask is None else bool(physio_valid_mask.any()))
            or ecg_available
        )
        return AlignedMultimodalSample(
            record=record,
            raw_arousal=record.emotion_scores.arousal,
            raw_valence=record.emotion_scores.valence,
            arousal_label=arousal_label,
            valence_label=valence_label,
            quadrant_label=quadrant_label,
            label_ignore_index=self._label_ignore_index,
            speech_waveform=speech_waveform,
            speech_attention_mask=speech_attention_mask,
            speech_sample_rate_hz=speech_sample_rate,
            speech_available=speech_waveform is not None,
            physio_input=physio_input,
            physio_valid_mask=physio_valid_mask,
            physio_time_mask=physio_time_mask,
            physio_channel_mask=physio_channel_mask,
            physio_timestamps_seconds=physio_timestamps,
            physio_channel_quality=physio_channel_quality,
            physio_quality_features=physio_quality_features,
            physiology_available=physiology_available,
            channel_names=(self._channel_names if self._enable_physiology else ()),
            speech_activity_mask=speech_activity_mask,
            speech_activity_ratio=speech_activity_ratio,
            ecg_values=ecg_values,
            ecg_valid_mask=ecg_valid_mask,
            ecg_timestamps_seconds=ecg_timestamps,
            ecg_available=ecg_available,
        )

    def _labels_for(
        self,
        record: MultimodalWindowRecord,
    ) -> tuple[int, int, int]:
        arousal_tensor = torch.tensor(record.emotion_scores.arousal, dtype=torch.float64)
        valence_tensor = torch.tensor(record.emotion_scores.valence, dtype=torch.float64)
        arousal, _ = binarize_emotion_scores(
            arousal_tensor,
            self._label_protocol,
            ignore_index=self._label_ignore_index,
        )
        valence, _ = binarize_emotion_scores(
            valence_tensor,
            self._label_protocol,
            ignore_index=self._label_ignore_index,
        )
        quadrant = derive_quadrant_labels(
            arousal,
            valence,
            ignore_index=self._label_ignore_index,
        )
        return int(arousal.item()), int(valence.item()), int(quadrant.item())

    def _load_speech(
        self,
        record: MultimodalWindowRecord,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        int | None,
        Tensor | None,
        float | None,
    ]:
        source = record.speech_source
        if source is None:
            return None, None, None, None, None
        assert self._speech_adapter is not None
        try:
            loaded = self._speech_adapter.load_speech(source)
            if not isinstance(loaded, SpeechSourceData):
                raise TypeError(
                    "load_speech must return SpeechSourceData; "
                    f"received {type(loaded).__name__}."
                )
            SpeechSourceData(
                loaded.waveform,
                loaded.sample_rate_hz,
                loaded.timeline_start_seconds,
                loaded.speech_activity_mask,
            )
            if loaded.sample_rate_hz != self._required_speech_sample_rate_hz:
                raise ValueError(
                    "speech sample rate mismatch; "
                    f"expected {self._required_speech_sample_rate_hz}, "
                    f"received {loaded.sample_rate_hz}."
                )
            span = regular_sample_span(
                record.window,
                timeline_start_seconds=loaded.timeline_start_seconds,
                sample_rate_hz=loaded.sample_rate_hz,
                num_samples=loaded.waveform.numel(),
            )
            if span.length == 0:
                raise ValueError("record window maps to an empty speech sample span.")
            waveform = (
                loaded.waveform[span.start_index : span.end_index]
                .detach()
                .clone()
                .to(dtype=self._output_dtype)
                .contiguous()
            )
            attention_mask = torch.ones(
                waveform.shape,
                dtype=torch.bool,
                device="cpu",
            )
            activity_mask: Tensor | None
            if loaded.speech_activity_mask is not None:
                activity_mask = (
                    loaded.speech_activity_mask[
                        span.start_index : span.end_index
                    ]
                    .detach()
                    .clone()
                    .contiguous()
                )
            elif (
                self._speech_activity_config is not None
                and self._speech_activity_config.enabled
            ):
                activity_mask = detect_frame_rms_speech_activity(
                    waveform,
                    loaded.sample_rate_hz,
                    self._speech_activity_config,
                )
            else:
                activity_mask = None
            activity_ratio = (
                None
                if activity_mask is None
                else float(activity_mask.to(dtype=torch.float64).mean().item())
            )
            return (
                waveform,
                attention_mask,
                loaded.sample_rate_hz,
                activity_mask,
                activity_ratio,
            )
        except Exception as error:
            raise self._source_error(
                record=record,
                modality="speech",
                source_id=source.source_id,
                error=error,
            ) from error

    def _load_physiology(
        self,
        record: MultimodalWindowRecord,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        if not self._dense_channel_specs:
            return None, None, None, None, None, None, None
        assert self._physio_target_sample_rate_hz is not None
        target_timestamps = self._build_target_timeline(
            record,
            rate=self._physio_target_sample_rate_hz,
        )
        sources = {source.channel_name: source for source in record.physio_sources}
        channel_values: list[Tensor] = []
        channel_masks: list[Tensor] = []
        quality_values: list[Tensor] = []
        for spec in self._dense_channel_specs:
            source = sources.get(spec.name)
            values, valid_mask, quality = self._process_channel(
                record,
                spec,
                source,
                target_timestamps,
            )
            channel_values.append(values)
            channel_masks.append(valid_mask)
            quality_values.append(quality)

        physio_input = torch.stack(channel_values, dim=1)
        physio_valid_mask = torch.stack(channel_masks, dim=1)
        physio_time_mask = physio_valid_mask.any(dim=1)
        physio_channel_mask = physio_valid_mask.any(dim=0)
        channel_quality = torch.stack(quality_values, dim=0)
        quality_features = channel_quality.reshape(-1)
        return (
            physio_input,
            physio_valid_mask,
            physio_time_mask,
            physio_channel_mask,
            target_timestamps,
            channel_quality,
            quality_features,
        )

    def _load_ecg(
        self,
        record: MultimodalWindowRecord,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Load ECG-HR onto its own 1 Hz grid as three ``[Te]`` tensors.

        Multiple real Polar measurements in the same target interval are
        averaged. Empty intervals remain masked and no interpolation or
        extrapolation is performed.
        """
        spec = self._ecg_spec
        if spec is None:
            return None, None, None
        assert self._ecg_sample_rate_hz is not None
        target_timestamps = self._build_target_timeline(
            record,
            rate=self._ecg_sample_rate_hz,
        )
        values = torch.zeros(target_timestamps.shape, dtype=self._output_dtype)
        valid_mask = torch.zeros(target_timestamps.shape, dtype=torch.bool)
        source = next(
            (
                item
                for item in record.physio_sources
                if item.channel_name == spec.name
            ),
            None,
        )
        if source is None:
            return values, valid_mask, target_timestamps
        assert self._physio_adapter is not None
        try:
            loaded = self._physio_adapter.load_physio(source, spec)
            self._validate_loaded_physio(loaded, source=source, spec=spec)
            usable_interval = source_overlap(record.window, source.source)
            if usable_interval is None:
                raise ValueError("declared ECG source has no window overlap.")
            span = timestamp_index_span(loaded.timestamps_seconds, usable_interval)
            source_values = loaded.values[span.start_index : span.end_index]
            source_valid = loaded.valid_mask[span.start_index : span.end_index]
            source_times = loaded.timestamps_seconds[span.start_index : span.end_index]
            step = 1.0 / self._ecg_sample_rate_hz
            for index, start in enumerate(target_timestamps):
                in_bin = (
                    (source_times >= start)
                    & (source_times < min(float(start.item()) + step, record.window.end_seconds))
                    & source_valid
                )
                if bool(in_bin.any()):
                    values[index] = source_values[in_bin].to(self._output_dtype).mean()
                    valid_mask[index] = True
            window = PhysioChannelWindow(
                spec=spec,
                values=values,
                valid_mask=valid_mask,
                timestamps_seconds=target_timestamps,
            )
            normalized = self._normalize(record, window)
            values = torch.where(
                normalized.valid_mask,
                normalized.values,
                torch.zeros_like(normalized.values),
            ).to(dtype=self._output_dtype)
            valid_mask = normalized.valid_mask.clone()
        except Exception as error:
            raise self._source_error(
                record=record,
                modality="physiology",
                source_id=source.source.source_id,
                channel_name="ecg",
                error=error,
            ) from error
        return values.contiguous(), valid_mask.contiguous(), target_timestamps

    @staticmethod
    def _build_target_timeline(
        record: MultimodalWindowRecord,
        *,
        rate: float,
    ) -> Tensor:
        """Build a float64 CPU half-open timeline from integer sample indices.

        The small tolerance applies only to the dimensionless sample count and
        prevents an almost-integral ``duration * rate`` from gaining one point.
        Every returned timestamp is still checked directly against the window
        end, so no point at or beyond the half-open boundary is retained.
        """
        sample_count = max(
            1,
            math.ceil(record.window.duration_seconds * rate - _TIMELINE_TOLERANCE),
        )
        indices = torch.arange(sample_count, dtype=torch.float64)
        timestamps = indices / rate + record.window.start_seconds
        timestamps = timestamps[timestamps < record.window.end_seconds]
        if timestamps.numel() == 0:
            raise RuntimeError("physiology target timeline must be non-empty.")
        if not bool(torch.isfinite(timestamps).all()):
            raise ValueError("physiology target timeline must be finite.")
        if timestamps.numel() > 1 and not bool(
            (timestamps[1:] > timestamps[:-1]).all()
        ):
            raise ValueError(
                "physiology target timeline is not strictly increasing at "
                "float64 precision; adjust the logical time origin or rate."
            )
        return timestamps

    def _process_channel(
        self,
        record: MultimodalWindowRecord,
        spec: PhysioChannelSpec,
        source: PhysioChannelSourceRef | None,
        target_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if source is None:
            return self._missing_channel(spec, target_timestamps)
        assert self._physio_adapter is not None
        try:
            loaded = self._physio_adapter.load_physio(source, spec)
            self._validate_loaded_physio(loaded, source=source, spec=spec)
            usable_interval = source_overlap(record.window, source.source)
            if usable_interval is None:
                raise ValueError("declared physiology source has no window overlap.")
            span = timestamp_index_span(
                loaded.timestamps_seconds,
                usable_interval,
            )
            if span.length == 0:
                return self._missing_channel(spec, target_timestamps)
            cropped = PhysioChannelWindow(
                spec=spec,
                values=loaded.values[span.start_index : span.end_index].clone(),
                valid_mask=loaded.valid_mask[
                    span.start_index : span.end_index
                ].clone(),
                timestamps_seconds=loaded.timestamps_seconds[
                    span.start_index : span.end_index
                ].clone(),
            )
            filter_fn = self._physio_filters.get(spec.name)
            filtered = (
                cropped
                if filter_fn is None
                else apply_channel_filter(cropped, filter_fn)
            )
            aligned = resample_channel_to_timeline(filtered, target_timestamps)
        except Exception as error:
            raise self._source_error(
                record=record,
                modality="physiology",
                source_id=source.source.source_id,
                channel_name=spec.name,
                error=error,
            ) from error

        flatline_mask: Tensor | None = None
        outlier_mask: Tensor | None = None
        artifact_mask: Tensor | None = None
        policy = self._artifact_policy
        if policy is not None:
            flatline_mask = (
                detect_flatline_mask(
                    aligned.values,
                    aligned.valid_mask,
                    atol=policy.flatline_atol,
                    min_run_length=policy.flatline_min_run_length,
                )
                if policy.detect_flatline
                else torch.zeros_like(aligned.valid_mask)
            )
            outlier_mask = (
                detect_robust_outlier_mask(
                    aligned.values,
                    aligned.valid_mask,
                    threshold=policy.outlier_threshold,
                    mad_epsilon=policy.outlier_mad_epsilon,
                )
                if policy.detect_outliers
                else torch.zeros_like(aligned.valid_mask)
            )
            artifact_mask = flatline_mask | outlier_mask

        quality = compute_channel_quality(
            aligned,
            flatline_mask=flatline_mask,
            outlier_mask=outlier_mask,
            artifact_mask=artifact_mask,
        ).as_tensor()
        final_valid = (
            combine_invalid_masks(aligned.valid_mask, artifact_mask)
            if policy is not None
            and policy.exclude_detected_artifacts
            and artifact_mask is not None
            else aligned.valid_mask.clone()
        )
        final_window = PhysioChannelWindow(
            spec=spec,
            values=zero_invalid_values(aligned.values, final_valid),
            valid_mask=final_valid,
            timestamps_seconds=target_timestamps,
        )
        final_window = self._normalize(record, final_window)
        values = torch.where(
            final_window.valid_mask,
            final_window.values,
            torch.zeros_like(final_window.values),
        )
        return (
            values.detach().clone().to(dtype=self._output_dtype),
            final_window.valid_mask.detach().clone(),
            quality.detach().clone().to(dtype=self._output_dtype),
        )

    def _missing_channel(
        self,
        spec: PhysioChannelSpec,
        target_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        valid_mask = torch.zeros(
            target_timestamps.shape,
            dtype=torch.bool,
            device="cpu",
        )
        missing_window = PhysioChannelWindow(
            spec=spec,
            values=torch.zeros(
                target_timestamps.shape,
                dtype=self._output_dtype,
                device="cpu",
            ),
            valid_mask=valid_mask,
            timestamps_seconds=target_timestamps,
        )
        quality = compute_channel_quality(missing_window).as_tensor()
        return missing_window.values, valid_mask, quality

    @staticmethod
    def _validate_loaded_physio(
        loaded: object,
        *,
        source: PhysioChannelSourceRef,
        spec: PhysioChannelSpec,
    ) -> None:
        if not isinstance(loaded, PhysioChannelWindow):
            raise TypeError(
                "load_physio must return PhysioChannelWindow; "
                f"received {type(loaded).__name__}."
            )
        PhysioChannelWindow(
            loaded.spec,
            loaded.values,
            loaded.valid_mask,
            loaded.timestamps_seconds,
        )
        if loaded.spec != spec:
            raise ValueError(
                f"loaded physiology spec must equal configured spec {spec.name!r}."
            )
        if source.channel_name != spec.name:
            raise ValueError(
                "physiology source channel_name must equal configured spec name."
            )
        for name, tensor in (
            ("values", loaded.values),
            ("valid_mask", loaded.valid_mask),
            ("timestamps_seconds", loaded.timestamps_seconds),
        ):
            _require_cpu_tensor(tensor, name=f"loaded physiology {name}")

    def _normalize(
        self,
        record: MultimodalWindowRecord,
        window: PhysioChannelWindow,
    ) -> PhysioChannelWindow:
        normalizer = self._physio_normalizer
        if normalizer is None:
            return window
        try:
            key = (
                NormalizationKey(channel_name=window.spec.name, group_id=None)
                if self._normalization_key_resolver is None
                else self._normalization_key_resolver(record, window.spec)
            )
            if not isinstance(key, NormalizationKey):
                raise TypeError(
                    "normalization_key_resolver must return NormalizationKey."
                )
            if key.channel_name != window.spec.name:
                raise ValueError(
                    "normalization key channel_name must match current channel."
                )
            transformed = normalizer.transform(window, key=key)
            if not isinstance(transformed, PhysioChannelWindow):
                raise TypeError("normalizer.transform must return PhysioChannelWindow.")
            if (
                transformed.spec != window.spec
                or not torch.equal(transformed.valid_mask, window.valid_mask)
                or not torch.equal(
                    transformed.timestamps_seconds,
                    window.timestamps_seconds,
                )
            ):
                raise ValueError(
                    "normalizer.transform must preserve spec, mask, and timestamps."
                )
            return transformed
        except Exception as error:
            raise RuntimeError(
                "physiology normalization failed for "
                f"sample_id={record.sample_id!r}, channel={window.spec.name!r}, "
                f"original_exception={type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def _source_error(
        *,
        record: MultimodalWindowRecord,
        modality: str,
        source_id: str,
        error: Exception,
        channel_name: str | None = None,
    ) -> SourceAdapterError:
        channel_context = (
            "" if channel_name is None else f", channel={channel_name!r}"
        )
        return SourceAdapterError(
            f"source adapter failure: sample_id={record.sample_id!r}, "
            f"modality={modality!r}, source_id={source_id!r}{channel_context}, "
            f"original_exception={type(error).__name__}: {error}"
        )
