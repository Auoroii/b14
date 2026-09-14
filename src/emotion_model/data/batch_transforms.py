"""Pure CPU transformations for strict compact multimodal batches."""

from __future__ import annotations

import torch
from torch import Tensor

from emotion_model.data.collate import (
    AlignedMultimodalBatch,
    PhysioSubBatch,
    SpeechSubBatch,
)


def _validate_bool_mask(mask: object, *, name: str, length: int) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if mask.dtype != torch.bool or tuple(mask.shape) != (length,):
        raise ValueError(f"{name} must be bool with shape [{length}].")
    if mask.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU.")
    return mask


def _select_rows(tensor: Tensor, row_indices: Tensor) -> Tensor:
    selected = tensor.index_select(0, row_indices).contiguous()
    if tensor.is_pinned() and not selected.is_pinned():
        selected = selected.pin_memory()
    return selected


def _filter_speech(
    speech: SpeechSubBatch | None,
    availability: Tensor,
) -> SpeechSubBatch | None:
    if not bool(availability.any()):
        return None
    if speech is None:
        raise RuntimeError("speech subbatch is missing for retained rows.")
    row_keep = availability[speech.batch_indices]
    row_indices = torch.nonzero(row_keep, as_tuple=False).flatten()
    return SpeechSubBatch(
        waveform=_select_rows(speech.waveform, row_indices),
        attention_mask=_select_rows(speech.attention_mask, row_indices),
        sequence_lengths=_select_rows(speech.sequence_lengths, row_indices),
        sample_rate_hz=speech.sample_rate_hz,
        batch_indices=torch.nonzero(
            availability,
            as_tuple=False,
        ).flatten(),
        activity_mask=(
            None
            if speech.activity_mask is None
            else _select_rows(speech.activity_mask, row_indices)
        ),
    )


def _filter_physiology(
    physiology: PhysioSubBatch | None,
    availability: Tensor,
) -> PhysioSubBatch | None:
    if not bool(availability.any()):
        return None
    if physiology is None:
        raise RuntimeError("physiology subbatch is missing for retained rows.")
    row_keep = availability[physiology.batch_indices]
    row_indices = torch.nonzero(row_keep, as_tuple=False).flatten()
    return PhysioSubBatch(
        physio_input=_select_rows(physiology.physio_input, row_indices),
        physio_valid_mask=_select_rows(
            physiology.physio_valid_mask,
            row_indices,
        ),
        physio_time_mask=_select_rows(
            physiology.physio_time_mask,
            row_indices,
        ),
        physio_channel_mask=_select_rows(
            physiology.physio_channel_mask,
            row_indices,
        ),
        physio_timestamps_seconds=_select_rows(
            physiology.physio_timestamps_seconds,
            row_indices,
        ),
        physio_timeline_mask=_select_rows(
            physiology.physio_timeline_mask,
            row_indices,
        ),
        timeline_lengths=_select_rows(
            physiology.timeline_lengths,
            row_indices,
        ),
        physio_channel_quality=_select_rows(
            physiology.physio_channel_quality,
            row_indices,
        ),
        physio_quality_features=_select_rows(
            physiology.physio_quality_features,
            row_indices,
        ),
        channel_names=physiology.channel_names,
        batch_indices=torch.nonzero(
            availability,
            as_tuple=False,
        ).flatten(),
    )


def apply_modality_keep_masks(
    batch: AlignedMultimodalBatch,
    *,
    speech_keep_mask: Tensor,
    physiology_keep_mask: Tensor,
) -> AlignedMultimodalBatch:
    """Return ``B`` logical rows after independently dropping modalities.

    Args:
        batch: Original strict CPU batch with logical label tensors ``[B]``.
        speech_keep_mask: Boolean CPU tensor ``[B]``. ``False`` drops speech;
            it never creates speech for an unavailable row.
        physiology_keep_mask: Boolean CPU tensor ``[B]``. ``False`` drops
            physiology; it never creates physiology for an unavailable row.

    Returns:
        New strict batch with the same ``B`` records and labels. Compact
        speech ``[Bs,L]`` and physiology ``[Bp,T,C]`` rows are filtered so
        their indices exactly match the new availability masks. A logical row
        may become fully unavailable; downstream model validity handles it.

    Inputs are not modified, and ``True`` retains the project-wide valid-mask
    semantics.
    """

    if not isinstance(batch, AlignedMultimodalBatch):
        raise TypeError("batch must be AlignedMultimodalBatch.")
    batch_size = len(batch.records)
    speech_keep = _validate_bool_mask(
        speech_keep_mask,
        name="speech_keep_mask",
        length=batch_size,
    )
    physiology_keep = _validate_bool_mask(
        physiology_keep_mask,
        name="physiology_keep_mask",
        length=batch_size,
    )
    speech_available = batch.speech_available & speech_keep
    physiology_available = batch.physiology_available & physiology_keep
    return AlignedMultimodalBatch(
        records=batch.records,
        raw_arousal=batch.raw_arousal,
        raw_valence=batch.raw_valence,
        arousal_labels=batch.arousal_labels,
        valence_labels=batch.valence_labels,
        quadrant_labels=batch.quadrant_labels,
        label_ignore_index=batch.label_ignore_index,
        speech_available=speech_available,
        physiology_available=physiology_available,
        speech=_filter_speech(batch.speech, speech_available),
        physiology=_filter_physiology(
            batch.physiology,
            physiology_available,
        ),
        speech_activity_ratios=batch.speech_activity_ratios,
    )


def select_aligned_multimodal_batch(
    batch: AlignedMultimodalBatch,
    row_mask: Tensor,
) -> AlignedMultimodalBatch:
    """Select non-empty logical rows and reindex compact modality tensors.

    Args:
        batch: Original strict CPU batch with labels and availability ``[B]``.
        row_mask: Boolean CPU tensor ``[B]`` where ``True`` retains a row.

    Returns:
        New strict batch with ``Bselected > 0`` logical rows, label tensors
        ``[Bselected]``, speech ``[Bs,L]``, and physiology ``[Bp,T,C]``.
        Compact ``batch_indices`` are remapped into ``[0,Bselected)``.

    Raises:
        ValueError: If no logical row is selected.
    """

    if not isinstance(batch, AlignedMultimodalBatch):
        raise TypeError("batch must be AlignedMultimodalBatch.")
    selected = _validate_bool_mask(
        row_mask,
        name="row_mask",
        length=len(batch.records),
    )
    old_indices = torch.nonzero(selected, as_tuple=False).flatten()
    if old_indices.numel() == 0:
        raise ValueError("row_mask must select at least one logical row.")
    old_to_new = torch.full(
        (len(batch.records),),
        -1,
        dtype=torch.long,
    )
    old_to_new[old_indices] = torch.arange(old_indices.numel())

    speech_available = _select_rows(batch.speech_available, old_indices)
    physiology_available = _select_rows(
        batch.physiology_available,
        old_indices,
    )
    speech: SpeechSubBatch | None = None
    if bool(speech_available.any()):
        if batch.speech is None:
            raise RuntimeError("speech subbatch is missing for selected rows.")
        compact_keep = selected[batch.speech.batch_indices]
        compact_rows = torch.nonzero(
            compact_keep,
            as_tuple=False,
        ).flatten()
        retained_old_indices = _select_rows(
            batch.speech.batch_indices,
            compact_rows,
        )
        speech = SpeechSubBatch(
            waveform=_select_rows(batch.speech.waveform, compact_rows),
            attention_mask=_select_rows(
                batch.speech.attention_mask,
                compact_rows,
            ),
            sequence_lengths=_select_rows(
                batch.speech.sequence_lengths,
                compact_rows,
            ),
            sample_rate_hz=batch.speech.sample_rate_hz,
            batch_indices=old_to_new[retained_old_indices],
            activity_mask=(
                None
                if batch.speech.activity_mask is None
                else _select_rows(batch.speech.activity_mask, compact_rows)
            ),
        )

    physiology: PhysioSubBatch | None = None
    if bool(physiology_available.any()):
        if batch.physiology is None:
            raise RuntimeError("physiology subbatch is missing for selected rows.")
        compact_keep = selected[batch.physiology.batch_indices]
        compact_rows = torch.nonzero(
            compact_keep,
            as_tuple=False,
        ).flatten()
        retained_old_indices = _select_rows(
            batch.physiology.batch_indices,
            compact_rows,
        )
        physiology = PhysioSubBatch(
            physio_input=_select_rows(
                batch.physiology.physio_input,
                compact_rows,
            ),
            physio_valid_mask=_select_rows(
                batch.physiology.physio_valid_mask,
                compact_rows,
            ),
            physio_time_mask=_select_rows(
                batch.physiology.physio_time_mask,
                compact_rows,
            ),
            physio_channel_mask=_select_rows(
                batch.physiology.physio_channel_mask,
                compact_rows,
            ),
            physio_timestamps_seconds=_select_rows(
                batch.physiology.physio_timestamps_seconds,
                compact_rows,
            ),
            physio_timeline_mask=_select_rows(
                batch.physiology.physio_timeline_mask,
                compact_rows,
            ),
            timeline_lengths=_select_rows(
                batch.physiology.timeline_lengths,
                compact_rows,
            ),
            physio_channel_quality=_select_rows(
                batch.physiology.physio_channel_quality,
                compact_rows,
            ),
            physio_quality_features=_select_rows(
                batch.physiology.physio_quality_features,
                compact_rows,
            ),
            channel_names=batch.physiology.channel_names,
            batch_indices=old_to_new[retained_old_indices],
        )

    selected_records = tuple(batch.records[int(index)] for index in old_indices.tolist())
    return AlignedMultimodalBatch(
        records=selected_records,
        raw_arousal=_select_rows(batch.raw_arousal, old_indices),
        raw_valence=_select_rows(batch.raw_valence, old_indices),
        arousal_labels=_select_rows(batch.arousal_labels, old_indices),
        valence_labels=_select_rows(batch.valence_labels, old_indices),
        quadrant_labels=_select_rows(batch.quadrant_labels, old_indices),
        label_ignore_index=batch.label_ignore_index,
        speech_available=speech_available,
        physiology_available=physiology_available,
        speech=speech,
        physiology=physiology,
        speech_activity_ratios=(
            None
            if batch.speech_activity_ratios is None
            else _select_rows(batch.speech_activity_ratios, old_indices)
        ),
    )


__all__ = [
    "apply_modality_keep_masks",
    "select_aligned_multimodal_batch",
]
