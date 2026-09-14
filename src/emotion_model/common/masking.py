"""Mask validation and safe masking operations."""

import torch
from torch import Tensor


def _validate_bool_mask(valid_mask: Tensor, *, name: str = "valid_mask") -> None:
    if valid_mask.dtype != torch.bool:
        raise TypeError(
            f"{name} must have dtype torch.bool with True meaning valid; "
            f"received {valid_mask.dtype}."
        )


def validate_sequence_mask(sequence: Tensor, valid_mask: Tensor) -> None:
    """Validate a sequence mask without modifying either input.

    Args:
        sequence: Target tensor with shape ``[B, T, ...]``. The first two axes
            are interpreted as batch and time.
        valid_mask: Boolean tensor with exact shape ``[B, T]`` where ``True``
            marks a valid time position and ``False`` marks padding or missing
            data.

    Returns:
        ``None``. An all-``False`` row is valid input and remains explicitly
        distinguishable by callers.

    Raises:
        TypeError: If ``valid_mask`` is not boolean.
        ValueError: If ``sequence`` has fewer than two dimensions, if
            ``valid_mask`` is not two-dimensional, if batch/time shapes do not
            match exactly, or if the tensors are on different devices. Shapes
            are never silently broadcast.
    """
    if sequence.ndim < 2:
        raise ValueError(
            "sequence must have at least two dimensions [B, T, ...]; "
            f"received shape {tuple(sequence.shape)}."
        )
    _validate_bool_mask(valid_mask)
    if valid_mask.ndim != 2:
        raise ValueError(
            "valid_mask must have exact shape [B, T]; "
            f"received shape {tuple(valid_mask.shape)}."
        )
    expected_shape = tuple(sequence.shape[:2])
    if tuple(valid_mask.shape) != expected_shape:
        raise ValueError(
            "valid_mask shape must exactly match sequence batch/time dimensions "
            f"{expected_shape}; received {tuple(valid_mask.shape)}."
        )
    if valid_mask.device != sequence.device:
        raise ValueError(
            "valid_mask and sequence must be on the same device; "
            f"received {valid_mask.device} and {sequence.device}."
        )


def valid_to_key_padding_mask(valid_mask: Tensor) -> Tensor:
    """Convert the project mask polarity to PyTorch key-padding polarity.

    Args:
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` means
            valid and ``False`` means padding or missing.

    Returns:
        Boolean tensor with shape ``[B, T]`` where ``True`` means the position
        must be ignored by a PyTorch attention interface. An all-padding input
        becomes an all-``True`` key padding mask; it is not treated as valid.

    Raises:
        TypeError: If ``valid_mask`` is not boolean.
        ValueError: If ``valid_mask`` does not have shape ``[B, T]``.
    """
    _validate_bool_mask(valid_mask)
    if valid_mask.ndim != 2:
        raise ValueError(
            "valid_mask must have exact shape [B, T]; "
            f"received shape {tuple(valid_mask.shape)}."
        )
    return torch.logical_not(valid_mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        raise IndexError(f"dim {dim} is out of range for a tensor with {ndim} dimensions.")
    return dim % ndim


def _expand_softmax_mask(logits: Tensor, valid_mask: Tensor, dim: int) -> Tensor:
    _validate_bool_mask(valid_mask)
    if valid_mask.device != logits.device:
        raise ValueError(
            "valid_mask and logits must be on the same device; "
            f"received {valid_mask.device} and {logits.device}."
        )

    if tuple(valid_mask.shape) == tuple(logits.shape):
        return valid_mask

    if (
        valid_mask.ndim == 2
        and logits.ndim >= 2
        and dim != 0
        and valid_mask.shape[0] == logits.shape[0]
        and valid_mask.shape[1] == logits.shape[dim]
    ):
        explicit_shape = [1] * logits.ndim
        explicit_shape[0] = logits.shape[0]
        explicit_shape[dim] = logits.shape[dim]
        return valid_mask.reshape(explicit_shape).expand(logits.shape)

    if valid_mask.ndim != logits.ndim:
        raise ValueError(
            "valid_mask must either have shape [B, T] aligned to the requested "
            "non-batch softmax dimension, or have the same rank as logits with "
            "explicit singleton broadcast dimensions; "
            f"received logits {tuple(logits.shape)} and mask {tuple(valid_mask.shape)}."
        )
    if logits.ndim >= 2 and valid_mask.shape[0] != logits.shape[0]:
        raise ValueError(
            "valid_mask batch size must exactly match logits and may not broadcast; "
            f"expected {logits.shape[0]}, received {valid_mask.shape[0]}."
        )
    if valid_mask.shape[dim] != logits.shape[dim]:
        raise ValueError(
            "valid_mask may not broadcast across the softmax dimension; "
            f"expected size {logits.shape[dim]} at dim {dim}, "
            f"received {valid_mask.shape[dim]}."
        )

    incompatible_dimensions = [
        index
        for index, (mask_size, logit_size) in enumerate(zip(valid_mask.shape, logits.shape))
        if mask_size not in (1, logit_size)
    ]
    if incompatible_dimensions:
        raise ValueError(
            "valid_mask has dimensions that cannot explicitly broadcast to logits: "
            f"indices {incompatible_dimensions}, logits {tuple(logits.shape)}, "
            f"mask {tuple(valid_mask.shape)}."
        )
    return valid_mask.expand(logits.shape)


def safe_masked_softmax(logits: Tensor, valid_mask: Tensor, dim: int = -1) -> Tensor:
    """Compute a numerically safe softmax over valid elements.

    Args:
        logits: Floating-point tensor with shape ``[...]``. Typical shapes are
            ``[B, T]`` and attention logits such as ``[B, H, Q, K]``.
        valid_mask: Boolean mask where ``True`` marks an element included in
            the softmax. It may match ``logits`` exactly, use the same rank with
            explicit singleton broadcast dimensions outside the batch and
            softmax axes, or have shape ``[B, T]`` aligned to the requested
            non-batch ``dim``. The batch dimension must match exactly.
        dim: Dimension over which probabilities are normalized.

    Returns:
        Tensor with the same shape and dtype as ``logits``. Masked positions
        are exactly zero. Slices containing at least one valid element sum to
        one along ``dim``. Fully masked slices are finite all-zero tensors and
        remain distinguishable as invalid through the input mask.

    Raises:
        TypeError: If ``logits`` is not floating point or ``valid_mask`` is not
            boolean.
        ValueError: If ``logits`` is scalar, tensor devices differ, or mask
            shapes cannot be explicitly aligned without ambiguous broadcasting.
            Broadcasting across the batch or softmax dimensions is rejected.
        IndexError: If ``dim`` is outside the valid range for ``logits``.
    """
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension; received a scalar.")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point; received {logits.dtype}.")

    normalized_dim = _normalize_dim(dim, logits.ndim)
    expanded_mask = _expand_softmax_mask(logits, valid_mask, normalized_dim)
    minimum = torch.finfo(logits.dtype).min
    masked_logits = torch.where(
        expanded_mask,
        logits,
        torch.full_like(logits, minimum),
    )
    probabilities = torch.softmax(masked_logits, dim=normalized_dim)
    masked_probabilities = torch.where(
        expanded_mask,
        probabilities,
        torch.zeros_like(probabilities),
    )
    normalizer = masked_probabilities.sum(dim=normalized_dim, keepdim=True)
    safe_normalizer = torch.where(
        normalizer > 0,
        normalizer,
        torch.ones_like(normalizer),
    )
    return masked_probabilities / safe_normalizer


def apply_query_mask(sequence: Tensor, valid_mask: Tensor) -> Tensor:
    """Zero invalid query positions in a batched sequence output.

    Args:
        sequence: Tensor with shape ``[B, T, D]`` containing query outputs.
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` keeps a
            query and ``False`` marks padding or a missing query.

    Returns:
        Tensor with shape ``[B, T, D]`` and the same dtype as ``sequence``.
        Valid queries are unchanged and invalid queries are exactly zero. An
        all-padding row therefore produces a finite all-zero row while
        remaining invalid according to ``valid_mask``.

    Raises:
        TypeError: If ``valid_mask`` is not boolean.
        ValueError: If ``sequence`` is not three-dimensional, if batch/time
            shapes differ, or if the tensors are on different devices.
    """
    if sequence.ndim != 3:
        raise ValueError(
            "sequence must have exact shape [B, T, D]; "
            f"received shape {tuple(sequence.shape)}."
        )
    validate_sequence_mask(sequence, valid_mask)
    return torch.where(
        valid_mask.unsqueeze(-1),
        sequence,
        torch.zeros_like(sequence),
    )
