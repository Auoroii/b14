"""CPU-only collation into compact modality-specific subbatches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from emotion_model.common.labels import derive_quadrant_labels
from emotion_model.data.dataset import AlignedMultimodalSample
from emotion_model.data.manifest import MultimodalWindowRecord


def _is_string_like(value: object) -> bool:
    return isinstance(value, (str, bytes))


def _require_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    return value


def _require_cpu(tensor: Tensor, *, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU.")


def _require_contiguous(tensor: Tensor, *, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_batch_indices(
    batch_indices: object,
    *,
    expected_length: int,
    name: str,
) -> Tensor:
    tensor = _require_tensor(batch_indices, name=name)
    if tensor.dtype != torch.long or tuple(tensor.shape) != (expected_length,):
        raise ValueError(
            f"{name} must be torch.long with shape [{expected_length}]."
        )
    _require_cpu(tensor, name=name)
    if expected_length > 0 and bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain non-negative indices.")
    if expected_length > 1 and not bool((tensor[1:] > tensor[:-1]).all()):
        raise ValueError(f"{name} must be strictly increasing without duplicates.")
    return tensor


def _require_zero_where_invalid(
    values: Tensor,
    valid_mask: Tensor,
    *,
    name: str,
) -> None:
    invalid_values = torch.where(valid_mask, torch.zeros_like(values), values)
    if not bool((invalid_values == 0).all()):
        raise ValueError(f"{name} must be exactly zero at invalid positions.")


@dataclass(frozen=True)
class SpeechSubBatch:
    """A compact, right-padded speech-only subbatch.

    Attributes:
        waveform: Floating CPU tensor ``[Bs, Lmax]``.
        attention_mask: Boolean tensor ``[Bs, Lmax]`` with ``True=valid`` and
            strict valid-prefix/right-padding rows.
        sequence_lengths: Positive ``torch.long`` CPU tensor ``[Bs]``.
        sample_rate_hz: Shared positive integer sampling rate.
        batch_indices: Strictly increasing ``torch.long`` CPU tensor ``[Bs]``
            mapping rows back to the full logical batch.
        activity_mask: Optional boolean participant-activity tensor
            ``[Bs,Lmax]``. It is a subset of attention, permits internal
            holes, and is ``False`` in padding.

    Every row contains at least one valid sample. Invalid waveform padding is
    exactly zero. Construction raises ``TypeError`` for invalid tensor/dtype
    categories and ``ValueError`` for invalid shapes, masks, values, devices,
    lengths, rates, or indices.
    """

    waveform: Tensor
    attention_mask: Tensor
    sequence_lengths: Tensor
    sample_rate_hz: int
    batch_indices: Tensor
    activity_mask: Tensor | None = None

    def __post_init__(self) -> None:
        waveform = _require_tensor(self.waveform, name="waveform")
        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have exact shape [Bs, Lmax]; "
                f"received {tuple(waveform.shape)}."
            )
        if not waveform.is_floating_point():
            raise TypeError("waveform must be floating point.")
        batch_size, max_length = waveform.shape
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("waveform requires Bs > 0 and Lmax > 0.")
        _require_cpu(waveform, name="waveform")
        _require_contiguous(waveform, name="waveform")

        attention_mask = _require_tensor(
            self.attention_mask,
            name="attention_mask",
        )
        if (
            attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != tuple(waveform.shape)
        ):
            raise ValueError(
                "attention_mask must be bool with the same [Bs, Lmax] shape."
            )
        _require_cpu(attention_mask, name="attention_mask")
        _require_contiguous(attention_mask, name="attention_mask")

        lengths = _require_tensor(
            self.sequence_lengths,
            name="sequence_lengths",
        )
        if lengths.dtype != torch.long or tuple(lengths.shape) != (batch_size,):
            raise ValueError(
                f"sequence_lengths must be torch.long with shape [{batch_size}]."
            )
        _require_cpu(lengths, name="sequence_lengths")
        if not bool(((lengths > 0) & (lengths <= max_length)).all()):
            raise ValueError("sequence_lengths must be in [1, Lmax].")
        expected_mask = (
            torch.arange(max_length, device="cpu").unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(attention_mask, expected_mask):
            raise ValueError(
                "attention_mask must equal the strict valid prefix defined "
                "by sequence_lengths."
            )
        if not bool(torch.isfinite(waveform[attention_mask]).all()):
            raise ValueError("waveform valid positions must be finite.")
        _require_zero_where_invalid(
            waveform,
            attention_mask,
            name="waveform",
        )
        if self.activity_mask is not None:
            activity_mask = _require_tensor(
                self.activity_mask,
                name="activity_mask",
            )
            if (
                activity_mask.dtype != torch.bool
                or tuple(activity_mask.shape) != tuple(waveform.shape)
            ):
                raise ValueError(
                    "activity_mask must be bool with the same [Bs,Lmax] shape."
                )
            _require_cpu(activity_mask, name="activity_mask")
            _require_contiguous(activity_mask, name="activity_mask")
            if bool((activity_mask & ~attention_mask).any()):
                raise ValueError(
                    "activity_mask must be a subset of attention_mask."
                )

        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
        ):
            raise TypeError("sample_rate_hz must be an integer, not bool.")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0.")
        _validate_batch_indices(
            self.batch_indices,
            expected_length=batch_size,
            name="batch_indices",
        )

    def pin_memory(self) -> SpeechSubBatch:
        """Return an equivalent CPU subbatch in page-locked memory.

        Returns:
            New speech tensors with shapes ``[Bs,Lmax]`` and ``[Bs]`` suitable
            for non-blocking transfer by a CUDA scheduler.
        """

        return SpeechSubBatch(
            waveform=self.waveform.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            sequence_lengths=self.sequence_lengths.pin_memory(),
            sample_rate_hz=self.sample_rate_hz,
            batch_indices=self.batch_indices.pin_memory(),
            activity_mask=(
                None
                if self.activity_mask is None
                else self.activity_mask.pin_memory()
            ),
        )


@dataclass(frozen=True)
class PhysioSubBatch:
    """A compact, right-padded physiology-only subbatch.

    Attributes:
        physio_input: Floating CPU tensor ``[Bp, Tmax, C]``.
        physio_valid_mask: Boolean tensor ``[Bp, Tmax, C]`` with
            ``True=valid``.
        physio_time_mask: Boolean tensor ``[Bp, Tmax]``; ``True`` means at
            least one channel is actually valid at that time.
        physio_channel_mask: Boolean tensor ``[Bp, C]``.
        physio_timestamps_seconds: Float64 CPU tensor ``[Bp, Tmax]``.
        physio_timeline_mask: Boolean tensor ``[Bp, Tmax]``; ``True`` means
            the position belongs to the Dataset-created target timeline, even
            when every channel is missing there. Its trailing ``False`` values
            are batch right padding.
        timeline_lengths: Positive ``torch.long`` CPU tensor ``[Bp]`` holding
            original target-timeline lengths, not valid-time counts.
        physio_channel_quality: Observable quality tensor ``[Bp, C, 6]``.
        physio_quality_features: Channel-major flattening ``[Bp, 6*C]``.
        channel_names: Fixed ordered tuple of ``C`` unique non-empty names.
        batch_indices: Strictly increasing ``torch.long`` CPU tensor ``[Bp]``
            mapping rows back to the full logical batch.

    Each row contains at least one valid physiology value. Invalid values and
    right-padded timestamps are exactly zero. Construction raises ``TypeError``
    for invalid tensor/dtype categories and ``ValueError`` for incompatible
    shapes, masks, values, devices, timelines, quality, names, or indices.
    """

    physio_input: Tensor
    physio_valid_mask: Tensor
    physio_time_mask: Tensor
    physio_channel_mask: Tensor
    physio_timestamps_seconds: Tensor
    physio_timeline_mask: Tensor
    timeline_lengths: Tensor
    physio_channel_quality: Tensor
    physio_quality_features: Tensor
    channel_names: tuple[str, ...]
    batch_indices: Tensor

    def __post_init__(self) -> None:
        values = _require_tensor(self.physio_input, name="physio_input")
        if values.ndim != 3:
            raise ValueError(
                "physio_input must have exact shape [Bp, Tmax, C]; "
                f"received {tuple(values.shape)}."
            )
        if not values.is_floating_point():
            raise TypeError("physio_input must be floating point.")
        batch_size, max_length, channel_count = values.shape
        if batch_size <= 0 or max_length <= 0 or channel_count <= 0:
            raise ValueError("physio_input requires Bp > 0, Tmax > 0, and C > 0.")
        _require_cpu(values, name="physio_input")
        _require_contiguous(values, name="physio_input")

        valid_mask = _require_tensor(
            self.physio_valid_mask,
            name="physio_valid_mask",
        )
        if (
            valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != tuple(values.shape)
        ):
            raise ValueError(
                "physio_valid_mask must be bool with shape [Bp, Tmax, C]."
            )
        _require_cpu(valid_mask, name="physio_valid_mask")

        time_mask = _require_tensor(
            self.physio_time_mask,
            name="physio_time_mask",
        )
        if (
            time_mask.dtype != torch.bool
            or tuple(time_mask.shape) != (batch_size, max_length)
        ):
            raise ValueError(
                "physio_time_mask must be bool with shape [Bp, Tmax]."
            )
        _require_cpu(time_mask, name="physio_time_mask")
        if not torch.equal(time_mask, valid_mask.any(dim=2)):
            raise ValueError(
                "physio_time_mask must equal physio_valid_mask.any(dim=2)."
            )
        if not bool(time_mask.any(dim=1).all()):
            raise ValueError(
                "every PhysioSubBatch row must contain a valid physiology value."
            )

        channel_mask = _require_tensor(
            self.physio_channel_mask,
            name="physio_channel_mask",
        )
        if (
            channel_mask.dtype != torch.bool
            or tuple(channel_mask.shape) != (batch_size, channel_count)
        ):
            raise ValueError(
                "physio_channel_mask must be bool with shape [Bp, C]."
            )
        _require_cpu(channel_mask, name="physio_channel_mask")
        if not torch.equal(channel_mask, valid_mask.any(dim=1)):
            raise ValueError(
                "physio_channel_mask must equal physio_valid_mask.any(dim=1)."
            )

        timestamps = _require_tensor(
            self.physio_timestamps_seconds,
            name="physio_timestamps_seconds",
        )
        if (
            timestamps.dtype != torch.float64
            or tuple(timestamps.shape) != (batch_size, max_length)
        ):
            raise ValueError(
                "physio_timestamps_seconds must be float64 with shape "
                "[Bp, Tmax]."
            )
        _require_cpu(timestamps, name="physio_timestamps_seconds")

        timeline_mask = _require_tensor(
            self.physio_timeline_mask,
            name="physio_timeline_mask",
        )
        if (
            timeline_mask.dtype != torch.bool
            or tuple(timeline_mask.shape) != (batch_size, max_length)
        ):
            raise ValueError(
                "physio_timeline_mask must be bool with shape [Bp, Tmax]."
            )
        _require_cpu(timeline_mask, name="physio_timeline_mask")

        lengths = _require_tensor(self.timeline_lengths, name="timeline_lengths")
        if lengths.dtype != torch.long or tuple(lengths.shape) != (batch_size,):
            raise ValueError(
                f"timeline_lengths must be torch.long with shape [{batch_size}]."
            )
        _require_cpu(lengths, name="timeline_lengths")
        if not bool(((lengths > 0) & (lengths <= max_length)).all()):
            raise ValueError("timeline_lengths must be in [1, Tmax].")
        expected_timeline_mask = (
            torch.arange(max_length, device="cpu").unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(timeline_mask, expected_timeline_mask):
            raise ValueError(
                "physio_timeline_mask must equal the strict prefix defined "
                "by timeline_lengths."
            )
        if bool((time_mask & ~timeline_mask).any()):
            raise ValueError(
                "physio_time_mask must be a subset of physio_timeline_mask."
            )
        if bool((valid_mask & ~timeline_mask.unsqueeze(-1)).any()):
            raise ValueError(
                "physio_valid_mask cannot mark batch-padding positions valid."
            )

        if not bool(torch.isfinite(values[valid_mask]).all()):
            raise ValueError("physio_input valid positions must be finite.")
        _require_zero_where_invalid(values, valid_mask, name="physio_input")
        for row_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            active_timestamps = timestamps[row_index, :length]
            if not bool(torch.isfinite(active_timestamps).all()):
                raise ValueError(
                    "physio timestamps inside each original timeline must be finite."
                )
            if length > 1 and not bool(
                (active_timestamps[1:] > active_timestamps[:-1]).all()
            ):
                raise ValueError(
                    "physio timestamps inside each original timeline must be "
                    "strictly increasing."
                )
            if not bool((timestamps[row_index, length:] == 0).all()):
                raise ValueError(
                    "physio timestamp batch-padding positions must be exactly zero."
                )

        quality = _require_tensor(
            self.physio_channel_quality,
            name="physio_channel_quality",
        )
        quality_features = _require_tensor(
            self.physio_quality_features,
            name="physio_quality_features",
        )
        if tuple(quality.shape) != (batch_size, channel_count, 6):
            raise ValueError(
                "physio_channel_quality must have shape [Bp, C, 6]."
            )
        if tuple(quality_features.shape) != (batch_size, 6 * channel_count):
            raise ValueError(
                "physio_quality_features must have shape [Bp, 6*C]."
            )
        if (
            not quality.is_floating_point()
            or not quality_features.is_floating_point()
        ):
            raise TypeError("physiology quality tensors must be floating point.")
        if quality.dtype != values.dtype or quality_features.dtype != values.dtype:
            raise TypeError(
                "physio_input and physiology quality tensors must share dtype."
            )
        _require_cpu(quality, name="physio_channel_quality")
        _require_cpu(quality_features, name="physio_quality_features")
        if not bool(
            torch.isfinite(quality).all()
            and (quality >= 0).all()
            and (quality <= 1).all()
        ):
            raise ValueError("physio_channel_quality must be finite in [0, 1].")
        if not torch.equal(quality_features, quality.reshape(batch_size, -1)):
            raise ValueError(
                "physio_quality_features must be channel-major quality flattening."
            )

        if (
            not isinstance(self.channel_names, tuple)
            or len(self.channel_names) != channel_count
            or any(not isinstance(name, str) or not name for name in self.channel_names)
            or len(set(self.channel_names)) != channel_count
        ):
            raise ValueError(
                "channel_names must be a tuple of C unique non-empty strings."
            )
        _validate_batch_indices(
            self.batch_indices,
            expected_length=batch_size,
            name="batch_indices",
        )

    def pin_memory(self) -> PhysioSubBatch:
        """Return an equivalent page-locked physiology subbatch.

        Returns:
            New tensors preserving ``[Bp,Tmax,C]``, mask, timestamp, quality,
            and index shapes for non-blocking CUDA transfer.
        """

        return PhysioSubBatch(
            physio_input=self.physio_input.pin_memory(),
            physio_valid_mask=self.physio_valid_mask.pin_memory(),
            physio_time_mask=self.physio_time_mask.pin_memory(),
            physio_channel_mask=self.physio_channel_mask.pin_memory(),
            physio_timestamps_seconds=(
                self.physio_timestamps_seconds.pin_memory()
            ),
            physio_timeline_mask=self.physio_timeline_mask.pin_memory(),
            timeline_lengths=self.timeline_lengths.pin_memory(),
            physio_channel_quality=self.physio_channel_quality.pin_memory(),
            physio_quality_features=self.physio_quality_features.pin_memory(),
            channel_names=self.channel_names,
            batch_indices=self.batch_indices.pin_memory(),
        )


@dataclass(frozen=True)
class AlignedMultimodalBatch:
    """A full logical batch plus compact speech and physiology subbatches.

    Attributes:
        records: Original immutable manifest records in input order, length
            ``B``.
        raw_arousal: Float64 CPU tensor ``[B]``.
        raw_valence: Float64 CPU tensor ``[B]``.
        arousal_labels: Long CPU tensor ``[B]``.
        valence_labels: Long CPU tensor ``[B]``.
        quadrant_labels: Long CPU tensor ``[B]``.
        label_ignore_index: Shared integer ignore value outside ``0..3``.
        speech_available: Boolean CPU tensor ``[B]``.
        physiology_available: Boolean CPU tensor ``[B]``.
        speech: Compact rows selected by ``speech_available``, or ``None``.
        physiology: Compact rows selected by ``physiology_available``, or
            ``None``.
        speech_activity_ratios: Optional float32 diagnostic tensor ``[B]``.
        speech_activity_observed: Optional boolean diagnostic tensor ``[B]``;
            true only where the corresponding ratio is observable.

    The modality subbatch ``batch_indices`` must exactly equal the corresponding
    availability nonzero indices. Fully unavailable logical rows remain in the
    records and label tensors. Invalid shapes, values, labels, devices, or
    mappings raise ``TypeError`` or ``ValueError``.
    """

    records: tuple[MultimodalWindowRecord, ...]
    raw_arousal: Tensor
    raw_valence: Tensor
    arousal_labels: Tensor
    valence_labels: Tensor
    quadrant_labels: Tensor
    label_ignore_index: int
    speech_available: Tensor
    physiology_available: Tensor
    speech: SpeechSubBatch | None
    physiology: PhysioSubBatch | None
    speech_activity_ratios: Tensor | None = None
    speech_activity_observed: Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.records, tuple)
            or not self.records
            or not all(
                isinstance(record, MultimodalWindowRecord)
                for record in self.records
            )
        ):
            raise TypeError(
                "records must be a non-empty tuple of MultimodalWindowRecord."
            )
        batch_size = len(self.records)
        if (
            isinstance(self.label_ignore_index, bool)
            or not isinstance(self.label_ignore_index, int)
        ):
            raise TypeError("label_ignore_index must be an integer, not bool.")
        if self.label_ignore_index in (0, 1, 2, 3):
            raise ValueError("label_ignore_index must not conflict with labels 0..3.")

        for name, raw_scores in (
            ("raw_arousal", self.raw_arousal),
            ("raw_valence", self.raw_valence),
        ):
            tensor = _require_tensor(raw_scores, name=name)
            if tensor.dtype != torch.float64 or tuple(tensor.shape) != (batch_size,):
                raise ValueError(
                    f"{name} must be torch.float64 with shape [{batch_size}]."
                )
            _require_cpu(tensor, name=name)
            if not bool(
                torch.isfinite(tensor).all()
                and (tensor >= 1).all()
                and (tensor <= 5).all()
            ):
                raise ValueError(f"{name} must be finite in [1, 5].")
        expected_raw_arousal = torch.tensor(
            [record.emotion_scores.arousal for record in self.records],
            dtype=torch.float64,
        )
        expected_raw_valence = torch.tensor(
            [record.emotion_scores.valence for record in self.records],
            dtype=torch.float64,
        )
        if not torch.equal(self.raw_arousal, expected_raw_arousal):
            raise ValueError(
                "raw_arousal must match records in the same logical batch order."
            )
        if not torch.equal(self.raw_valence, expected_raw_valence):
            raise ValueError(
                "raw_valence must match records in the same logical batch order."
            )

        binary_labels: list[Tensor] = []
        for name, labels in (
            ("arousal_labels", self.arousal_labels),
            ("valence_labels", self.valence_labels),
        ):
            tensor = _require_tensor(labels, name=name)
            if tensor.dtype != torch.long or tuple(tensor.shape) != (batch_size,):
                raise ValueError(
                    f"{name} must be torch.long with shape [{batch_size}]."
                )
            _require_cpu(tensor, name=name)
            valid_values = (
                (tensor == 0)
                | (tensor == 1)
                | (tensor == self.label_ignore_index)
            )
            if not bool(valid_values.all()):
                raise ValueError(
                    f"{name} values must be 0, 1, or label_ignore_index."
                )
            binary_labels.append(tensor)

        quadrant = _require_tensor(self.quadrant_labels, name="quadrant_labels")
        if quadrant.dtype != torch.long or tuple(quadrant.shape) != (batch_size,):
            raise ValueError(
                f"quadrant_labels must be torch.long with shape [{batch_size}]."
            )
        _require_cpu(quadrant, name="quadrant_labels")
        valid_quadrants = (
            ((quadrant >= 0) & (quadrant <= 3))
            | (quadrant == self.label_ignore_index)
        )
        if not bool(valid_quadrants.all()):
            raise ValueError(
                "quadrant_labels values must be 0..3 or label_ignore_index."
            )
        expected_quadrants = derive_quadrant_labels(
            binary_labels[0],
            binary_labels[1],
            ignore_index=self.label_ignore_index,
        )
        if not torch.equal(quadrant, expected_quadrants):
            raise ValueError(
                "quadrant_labels must match the public arousal/valence mapping "
                "with ignore propagation."
            )

        speech_available = self._validate_availability(
            self.speech_available,
            name="speech_available",
            batch_size=batch_size,
        )
        physiology_available = self._validate_availability(
            self.physiology_available,
            name="physiology_available",
            batch_size=batch_size,
        )
        self._validate_subbatch_mapping(
            speech_available,
            self.speech,
            subbatch_type=SpeechSubBatch,
            name="speech",
        )
        self._validate_subbatch_mapping(
            physiology_available,
            self.physiology,
            subbatch_type=PhysioSubBatch,
            name="physiology",
        )
        activity_ratios = self.speech_activity_ratios
        activity_observed = self.speech_activity_observed
        if activity_ratios is not None:
            if (
                activity_ratios.dtype != torch.float32
                or tuple(activity_ratios.shape) != (batch_size,)
            ):
                raise ValueError(
                    "speech_activity_ratios must be float32 with shape [B]."
                )
            _require_cpu(activity_ratios, name="speech_activity_ratios")
            if not bool(
                torch.isfinite(activity_ratios).all()
                and (activity_ratios >= 0.0).all()
                and (activity_ratios <= 1.0).all()
            ):
                raise ValueError("speech_activity_ratios must be finite in [0, 1].")
            if activity_observed is None:
                activity_observed = torch.tensor(
                    [record.speech_source is not None for record in self.records],
                    dtype=torch.bool,
                )
                object.__setattr__(
                    self,
                    "speech_activity_observed",
                    activity_observed,
                )
            if (
                activity_observed.dtype != torch.bool
                or tuple(activity_observed.shape) != (batch_size,)
            ):
                raise ValueError(
                    "speech_activity_observed must be bool with shape [B]."
                )
            _require_cpu(activity_observed, name="speech_activity_observed")
            source_present = torch.tensor(
                [record.speech_source is not None for record in self.records],
                dtype=torch.bool,
            )
            if bool((activity_observed & ~source_present).any()):
                raise ValueError(
                    "speech activity can be observed only for a real source."
                )
        elif activity_observed is not None:
            raise ValueError(
                "speech_activity_observed requires speech_activity_ratios."
            )

    def pin_memory(self) -> AlignedMultimodalBatch:
        """Return an equivalent logical batch in page-locked CPU memory.

        Returns:
            A new batch retaining record identity and tensor shapes ``[B]``,
            with optional compact modality tensors recursively pinned.
        """

        return AlignedMultimodalBatch(
            records=self.records,
            raw_arousal=self.raw_arousal.pin_memory(),
            raw_valence=self.raw_valence.pin_memory(),
            arousal_labels=self.arousal_labels.pin_memory(),
            valence_labels=self.valence_labels.pin_memory(),
            quadrant_labels=self.quadrant_labels.pin_memory(),
            label_ignore_index=self.label_ignore_index,
            speech_available=self.speech_available.pin_memory(),
            physiology_available=self.physiology_available.pin_memory(),
            speech=None if self.speech is None else self.speech.pin_memory(),
            physiology=(
                None
                if self.physiology is None
                else self.physiology.pin_memory()
            ),
            speech_activity_ratios=(
                None
                if self.speech_activity_ratios is None
                else self.speech_activity_ratios.pin_memory()
            ),
            speech_activity_observed=(
                None
                if self.speech_activity_observed is None
                else self.speech_activity_observed.pin_memory()
            ),
        )

    @staticmethod
    def _validate_availability(
        value: object,
        *,
        name: str,
        batch_size: int,
    ) -> Tensor:
        tensor = _require_tensor(value, name=name)
        if tensor.dtype != torch.bool or tuple(tensor.shape) != (batch_size,):
            raise ValueError(f"{name} must be bool with shape [{batch_size}].")
        _require_cpu(tensor, name=name)
        return tensor

    @staticmethod
    def _validate_subbatch_mapping(
        availability: Tensor,
        subbatch: object,
        *,
        subbatch_type: type[SpeechSubBatch] | type[PhysioSubBatch],
        name: str,
    ) -> None:
        expected_indices = torch.nonzero(availability, as_tuple=False).flatten()
        if expected_indices.numel() == 0:
            if subbatch is not None:
                raise ValueError(f"{name} must be None when availability is all False.")
            return
        if not isinstance(subbatch, subbatch_type):
            raise TypeError(
                f"{name} must be {subbatch_type.__name__} when any row is available."
            )
        if not torch.equal(subbatch.batch_indices, expected_indices):
            raise ValueError(
                f"{name}.batch_indices must exactly equal availability nonzero indices."
            )


def _validate_sample(sample: AlignedMultimodalSample, *, index: int) -> None:
    try:
        sample.__post_init__()
    except (TypeError, ValueError) as error:
        raise type(error)(f"samples[{index}] violates its contract: {error}") from error
    if sample.label_ignore_index in (0, 1, 2, 3):
        raise ValueError(
            f"samples[{index}].label_ignore_index must not conflict with labels 0..3."
        )
    if sample.speech_available:
        assert sample.speech_waveform is not None
        if not sample.speech_waveform.is_contiguous():
            raise ValueError(f"samples[{index}].speech_waveform must be contiguous.")
    if sample.channel_names:
        if (
            any(not name for name in sample.channel_names)
            or len(set(sample.channel_names)) != len(sample.channel_names)
        ):
            raise ValueError(
                f"samples[{index}].channel_names must be unique and non-empty."
            )


def _collate_speech(
    samples: tuple[AlignedMultimodalSample, ...],
    batch_indices: Tensor,
) -> SpeechSubBatch:
    selected = tuple(samples[int(index)] for index in batch_indices.tolist())
    waveforms: list[Tensor] = []
    activity_masks: list[Tensor] = []
    lengths: list[int] = []
    sample_rate: int | None = None
    dtype: torch.dtype | None = None
    for sample in selected:
        assert sample.speech_waveform is not None
        assert sample.speech_attention_mask is not None
        assert sample.speech_sample_rate_hz is not None
        waveform = sample.speech_waveform
        if dtype is None:
            dtype = waveform.dtype
        elif waveform.dtype != dtype:
            raise TypeError("all available speech waveforms must share dtype.")
        if sample_rate is None:
            sample_rate = sample.speech_sample_rate_hz
        elif sample.speech_sample_rate_hz != sample_rate:
            raise ValueError("all available speech samples must share sample rate.")
        waveforms.append(waveform)
        activity_masks.append(
            sample.speech_attention_mask
            if sample.speech_activity_mask is None
            else sample.speech_activity_mask
        )
        lengths.append(waveform.numel())
    assert dtype is not None
    assert sample_rate is not None
    max_length = max(lengths)
    padded = torch.zeros((len(selected), max_length), dtype=dtype, device="cpu")
    attention_mask = torch.zeros(
        (len(selected), max_length),
        dtype=torch.bool,
        device="cpu",
    )
    activity_mask = torch.zeros_like(attention_mask)
    for row, (waveform, activity, length) in enumerate(
        zip(waveforms, activity_masks, lengths, strict=True)
    ):
        padded[row, :length].copy_(waveform)
        attention_mask[row, :length] = True
        activity_mask[row, :length].copy_(activity)
    return SpeechSubBatch(
        waveform=padded.contiguous(),
        attention_mask=attention_mask.contiguous(),
        sequence_lengths=torch.tensor(lengths, dtype=torch.long),
        sample_rate_hz=sample_rate,
        batch_indices=batch_indices.clone().contiguous(),
        activity_mask=activity_mask.contiguous(),
    )


def _collate_physiology(
    samples: tuple[AlignedMultimodalSample, ...],
    batch_indices: Tensor,
    *,
    channel_names: tuple[str, ...],
) -> PhysioSubBatch:
    selected = tuple(samples[int(index)] for index in batch_indices.tolist())
    first = selected[0]
    assert first.physio_input is not None
    dtype = first.physio_input.dtype
    channel_count = len(channel_names)
    lengths: list[int] = []
    for sample in samples:
        if sample.channel_names:
            assert sample.physio_input is not None
            assert sample.physio_channel_quality is not None
            assert sample.physio_quality_features is not None
            if (
                sample.physio_input.dtype != dtype
                or sample.physio_channel_quality.dtype != dtype
                or sample.physio_quality_features.dtype != dtype
            ):
                raise TypeError(
                    "all non-empty physiology-contract samples must share dtype."
                )
    for sample in selected:
        assert sample.physio_input is not None
        lengths.append(sample.physio_input.shape[0])
    max_length = max(lengths)
    batch_size = len(selected)
    values = torch.zeros(
        (batch_size, max_length, channel_count),
        dtype=dtype,
        device="cpu",
    )
    valid_mask = torch.zeros_like(values, dtype=torch.bool)
    time_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
        device="cpu",
    )
    timestamps = torch.zeros(
        (batch_size, max_length),
        dtype=torch.float64,
        device="cpu",
    )
    timeline_mask = torch.zeros_like(time_mask)
    channel_masks: list[Tensor] = []
    quality_matrices: list[Tensor] = []
    quality_features: list[Tensor] = []
    for row, (sample, length) in enumerate(zip(selected, lengths, strict=True)):
        assert sample.physio_input is not None
        assert sample.physio_valid_mask is not None
        assert sample.physio_time_mask is not None
        assert sample.physio_channel_mask is not None
        assert sample.physio_timestamps_seconds is not None
        assert sample.physio_channel_quality is not None
        assert sample.physio_quality_features is not None
        values[row, :length].copy_(sample.physio_input)
        valid_mask[row, :length].copy_(sample.physio_valid_mask)
        time_mask[row, :length].copy_(sample.physio_time_mask)
        timestamps[row, :length].copy_(sample.physio_timestamps_seconds)
        timeline_mask[row, :length] = True
        channel_masks.append(sample.physio_channel_mask)
        quality_matrices.append(sample.physio_channel_quality)
        quality_features.append(sample.physio_quality_features)
    return PhysioSubBatch(
        physio_input=values.contiguous(),
        physio_valid_mask=valid_mask.contiguous(),
        physio_time_mask=time_mask.contiguous(),
        physio_channel_mask=torch.stack(channel_masks).clone().contiguous(),
        physio_timestamps_seconds=timestamps.contiguous(),
        physio_timeline_mask=timeline_mask.contiguous(),
        timeline_lengths=torch.tensor(lengths, dtype=torch.long),
        physio_channel_quality=torch.stack(quality_matrices).clone().contiguous(),
        physio_quality_features=torch.stack(quality_features).clone().contiguous(),
        channel_names=channel_names,
        batch_indices=batch_indices.clone().contiguous(),
    )


def collate_aligned_multimodal_samples(
    samples: Sequence[AlignedMultimodalSample],
) -> AlignedMultimodalBatch:
    """Collate samples into one full batch and compact modality subbatches.

    Args:
        samples: Non-empty, non-string sequence of
            :class:`AlignedMultimodalSample` objects. Sample order is retained.

    Returns:
        :class:`AlignedMultimodalBatch` whose full label and availability
        tensors have shape ``[B]``. Its optional speech tensors have shapes
        ``[Bs, Lmax]``/``[Bs]`` and physiology tensors have shapes
        ``[Bp, Tmax, C]``, ``[Bp, Tmax]``, ``[Bp, C]``, ``[Bp, C, 6]``, and
        ``[Bp, 6*C]``. All tensors are new CPU tensors; masks use
        ``True=valid``.

    Raises:
        TypeError: If ``samples`` is not a sequence, contains an object whose
            exact type is not :class:`AlignedMultimodalSample`, or mixes
            incompatible dtypes.
        ValueError: If the sequence is empty, sample contracts conflict,
            labels are inconsistent, channel order differs, or available
            modalities have incompatible sampling rates.

    This function only validates, selects, stacks, and right-pads existing
    in-memory samples. It performs no source loading, resampling, inference,
    objective calculation, device transfer, memory pinning, or caching.
    """
    if _is_string_like(samples) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a non-string Sequence.")
    if not samples:
        raise ValueError("samples must be non-empty.")
    sample_tuple = tuple(samples)
    for index, sample in enumerate(sample_tuple):
        if type(sample) is not AlignedMultimodalSample:
            raise TypeError(
                f"samples[{index}] must be exactly AlignedMultimodalSample; "
                f"received {type(sample).__name__}."
            )
        _validate_sample(sample, index=index)

    ignore_index = sample_tuple[0].label_ignore_index
    if any(sample.label_ignore_index != ignore_index for sample in sample_tuple):
        raise ValueError("all samples must share label_ignore_index.")
    channel_names = sample_tuple[0].channel_names
    if any(sample.channel_names != channel_names for sample in sample_tuple):
        raise ValueError(
            "all samples must share exactly the same channel_names and order."
        )
    if channel_names:
        physiology_dtypes = {
            sample.physio_input.dtype
            for sample in sample_tuple
            if sample.physio_input is not None
        }
        if len(physiology_dtypes) != 1:
            raise TypeError(
                "all non-empty physiology-contract samples must share dtype."
            )

    records = tuple(sample.record for sample in sample_tuple)
    raw_arousal = torch.tensor(
        [sample.raw_arousal for sample in sample_tuple],
        dtype=torch.float64,
    )
    raw_valence = torch.tensor(
        [sample.raw_valence for sample in sample_tuple],
        dtype=torch.float64,
    )
    arousal_labels = torch.tensor(
        [sample.arousal_label for sample in sample_tuple],
        dtype=torch.long,
    )
    valence_labels = torch.tensor(
        [sample.valence_label for sample in sample_tuple],
        dtype=torch.long,
    )
    quadrant_labels = torch.tensor(
        [sample.quadrant_label for sample in sample_tuple],
        dtype=torch.long,
    )
    speech_available = torch.tensor(
        [sample.speech_available for sample in sample_tuple],
        dtype=torch.bool,
    )
    physiology_available = torch.tensor(
        [sample.physiology_available for sample in sample_tuple],
        dtype=torch.bool,
    )
    has_activity_diagnostics = any(
        sample.speech_activity_ratio is not None for sample in sample_tuple
    )
    speech_activity_observed = (
        torch.tensor(
            [
                sample.speech_activity_ratio is not None
                for sample in sample_tuple
            ],
            dtype=torch.bool,
        )
        if has_activity_diagnostics
        else None
    )
    speech_activity_ratios = (
        torch.tensor(
            [
                (
                    sample.speech_activity_ratio
                    if sample.speech_activity_ratio is not None
                    else (1.0 if sample.speech_available else 0.0)
                )
                for sample in sample_tuple
            ],
            dtype=torch.float32,
        )
        if has_activity_diagnostics
        else None
    )
    speech_indices = torch.nonzero(speech_available, as_tuple=False).flatten()
    physiology_indices = torch.nonzero(
        physiology_available,
        as_tuple=False,
    ).flatten()
    speech = (
        None
        if speech_indices.numel() == 0
        else _collate_speech(sample_tuple, speech_indices)
    )
    physiology = (
        None
        if physiology_indices.numel() == 0
        else _collate_physiology(
            sample_tuple,
            physiology_indices,
            channel_names=channel_names,
        )
    )
    return AlignedMultimodalBatch(
        records=records,
        raw_arousal=raw_arousal,
        raw_valence=raw_valence,
        arousal_labels=arousal_labels,
        valence_labels=valence_labels,
        quadrant_labels=quadrant_labels,
        label_ignore_index=ignore_index,
        speech_available=speech_available,
        physiology_available=physiology_available,
        speech=speech,
        physiology=physiology,
        speech_activity_ratios=speech_activity_ratios,
        speech_activity_observed=speech_activity_observed,
    )
