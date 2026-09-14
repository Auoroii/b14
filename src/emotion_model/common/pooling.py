"""Numerically safe masked pooling for batched time sequences."""

import torch
from torch import Tensor

from emotion_model.common.masking import validate_sequence_mask


def _validate_pooling_inputs(sequence: Tensor, valid_mask: Tensor) -> None:
    if sequence.ndim != 3:
        raise ValueError(
            "sequence must have exact shape [B, T, D]; "
            f"received shape {tuple(sequence.shape)}."
        )
    if not sequence.is_floating_point():
        raise TypeError(f"sequence must be floating point; received {sequence.dtype}.")
    validate_sequence_mask(sequence, valid_mask)


def _masked_moments(sequence: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _validate_pooling_inputs(sequence, valid_mask)
    expanded_mask = valid_mask.unsqueeze(-1)
    masked_sequence = torch.where(
        expanded_mask,
        sequence,
        torch.zeros_like(sequence),
    )
    counts = valid_mask.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp_min(1).to(dtype=sequence.dtype)
    mean = masked_sequence.sum(dim=1) / safe_counts

    centered = torch.where(
        expanded_mask,
        sequence - mean.unsqueeze(1),
        torch.zeros_like(sequence),
    )
    variance = centered.square().sum(dim=1) / safe_counts
    sample_valid = valid_mask.any(dim=1)
    return mean, variance, sample_valid


def masked_mean(sequence: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the mean of valid time positions.

    Args:
        sequence: Floating-point tensor with shape ``[B, T, D]``.
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` means a
            valid time position and ``False`` means padding or missing.

    Returns:
        A tuple ``(mean, sample_valid)``. ``mean`` has shape ``[B, D]`` and the
        same dtype as ``sequence``. ``sample_valid`` has shape ``[B]`` and is
        ``True`` exactly when a sample has at least one valid time position.
        Fully padded samples return a finite all-zero mean and ``False`` state.

    Raises:
        TypeError: If ``sequence`` is not floating point or ``valid_mask`` is
            not boolean.
        ValueError: If inputs do not have exact shapes ``[B, T, D]`` and
            ``[B, T]``, their batch/time dimensions differ, or devices differ.
    """
    mean, _, sample_valid = _masked_moments(sequence, valid_mask)
    return mean, sample_valid


def masked_population_variance(
    sequence: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute population variance over valid time positions.

    Args:
        sequence: Floating-point tensor with shape ``[B, T, D]``.
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` means a
            valid time position and ``False`` means padding or missing.

    Returns:
        A tuple ``(variance, sample_valid)``. ``variance`` has shape ``[B, D]``
        and uses the population denominator, equivalent to ``unbiased=False``.
        ``sample_valid`` has shape ``[B]``. One valid position yields exact zero
        variance; fully padded samples yield finite zeros and ``False`` state.

    Raises:
        TypeError: If ``sequence`` is not floating point or ``valid_mask`` is
            not boolean.
        ValueError: If inputs do not have exact shapes ``[B, T, D]`` and
            ``[B, T]``, their batch/time dimensions differ, or devices differ.
    """
    _, variance, sample_valid = _masked_moments(sequence, valid_mask)
    return variance, sample_valid


def masked_population_std(
    sequence: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute population standard deviation over valid time positions.

    Args:
        sequence: Floating-point tensor with shape ``[B, T, D]``.
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` means a
            valid time position and ``False`` means padding or missing.

    Returns:
        A tuple ``(std, sample_valid)``. ``std`` has shape ``[B, D]`` and
        ``sample_valid`` has shape ``[B]``. One valid position and fully padded
        samples produce exact finite zero standard deviation; fully padded
        samples have ``False`` state.

    Raises:
        TypeError: If ``sequence`` is not floating point or ``valid_mask`` is
            not boolean.
        ValueError: If inputs do not have exact shapes ``[B, T, D]`` and
            ``[B, T]``, their batch/time dimensions differ, or devices differ.
    """
    _, variance, sample_valid = _masked_moments(sequence, valid_mask)
    positive_variance = variance > 0
    safe_variance = torch.where(
        positive_variance,
        variance,
        torch.ones_like(variance),
    )
    std = torch.where(
        positive_variance,
        torch.sqrt(safe_variance),
        torch.zeros_like(variance),
    )
    return std, sample_valid


def masked_mean_std(sequence: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Concatenate masked mean and population standard deviation.

    Args:
        sequence: Floating-point tensor with shape ``[B, T, D]``.
        valid_mask: Boolean tensor with shape ``[B, T]`` where ``True`` means a
            valid time position and ``False`` means padding or missing.

    Returns:
        A tuple ``(statistics, sample_valid)``. ``statistics`` has shape
        ``[B, 2D]`` with mean followed by population standard deviation.
        ``sample_valid`` has shape ``[B]``. Fully padded samples return finite
        all-zero statistics and ``False`` state.

    Raises:
        TypeError: If ``sequence`` is not floating point or ``valid_mask`` is
            not boolean.
        ValueError: If inputs do not have exact shapes ``[B, T, D]`` and
            ``[B, T]``, their batch/time dimensions differ, or devices differ.
    """
    mean, variance, sample_valid = _masked_moments(sequence, valid_mask)
    positive_variance = variance > 0
    safe_variance = torch.where(
        positive_variance,
        variance,
        torch.ones_like(variance),
    )
    std = torch.where(
        positive_variance,
        torch.sqrt(safe_variance),
        torch.zeros_like(variance),
    )
    return torch.cat((mean, std), dim=-1), sample_valid
