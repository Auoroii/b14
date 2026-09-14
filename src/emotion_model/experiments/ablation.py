"""Controlled same-window modality ablations for multimodal evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import StrEnum

import torch

from emotion_model.data import (
    AlignedMultimodalBatch,
    apply_modality_keep_masks,
    select_aligned_multimodal_batch,
)


class ModalityAblationMode(StrEnum):
    """Supported evaluations over windows where both modalities are valid."""

    FULL = "full"
    SPEECH_ONLY = "speech_only"
    PHYSIOLOGY_ONLY = "physiology_only"


def iter_both_modality_ablation_batches(
    batches: Iterable[AlignedMultimodalBatch],
    mode: ModalityAblationMode,
) -> Iterator[AlignedMultimodalBatch]:
    """Yield the same both-valid logical rows under one modality condition.

    Args:
        batches: Ordered strict batches with logical availability masks ``[B]``.
        mode: Retain both modalities, speech only, or physiology only.

    Yields:
        Non-empty selected batches. Every yielded row originally has both
        modalities. Labels have shape ``[Bselected]``; compact speech and
        physiology retain ``[Bs,L]`` and ``[Bp,T,C]`` contracts. Across all
        modes, yielded sample identifiers are identical and ordered equally.
    """

    if not isinstance(mode, ModalityAblationMode):
        raise TypeError("mode must be ModalityAblationMode.")
    for batch in batches:
        if not isinstance(batch, AlignedMultimodalBatch):
            raise TypeError("batches must contain AlignedMultimodalBatch objects.")
        both = batch.speech_available & batch.physiology_available
        if not bool(both.any()):
            continue
        selected = select_aligned_multimodal_batch(batch, both)
        keep = torch.ones(len(selected.records), dtype=torch.bool)
        if mode is ModalityAblationMode.FULL:
            yield selected
        elif mode is ModalityAblationMode.SPEECH_ONLY:
            yield apply_modality_keep_masks(
                selected,
                speech_keep_mask=keep,
                physiology_keep_mask=torch.zeros_like(keep),
            )
        else:
            yield apply_modality_keep_masks(
                selected,
                speech_keep_mask=torch.zeros_like(keep),
                physiology_keep_mask=keep,
            )


__all__ = [
    "ModalityAblationMode",
    "iter_both_modality_ablation_batches",
]
