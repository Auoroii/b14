"""Numerically safe masked pooling for batched time sequences."""

import torch
from torch import Tensor, nn

from emotion_model.common.masking import safe_masked_softmax, validate_sequence_mask


class MaskAwareAttentiveStatisticsPooling(nn.Module):
    """Learn weighted mean/std statistics over valid sequence frames.

    Args:
        feature_dim: Input feature width ``D``.
        attention_hidden_dim: Hidden width of the lightweight score network.
        epsilon: Positive variance floor used before the square root.

    Forward accepts floating ``features`` with shape ``[B, T, D]`` and a
    boolean ``valid_mask`` with shape ``[B, T]``. It returns
    ``(statistics, attention_weights, sample_valid)`` with shapes ``[B, 2D]``,
    ``[B, T]``, and ``[B]`` respectively. Fully masked rows return finite
    zeros for both floating outputs and ``False`` validity.
    """

    def __init__(
        self,
        feature_dim: int,
        attention_hidden_dim: int = 64,
        *,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("attention_hidden_dim", attention_hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a real number, not bool.")
        self.feature_dim = feature_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.epsilon = float(epsilon)
        if not torch.isfinite(torch.tensor(self.epsilon)) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pool ``[B,T,D]`` features into mask-safe statistics ``[B,2D]``."""

        _validate_pooling_inputs(features, valid_mask)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features width must be {self.feature_dim}; "
                f"received {features.shape[-1]}."
            )
        safe_features = torch.where(
            valid_mask.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )
        logits = self.attention(safe_features).squeeze(-1)
        attention_weights = safe_masked_softmax(logits, valid_mask, dim=1)
        expanded_weights = attention_weights.unsqueeze(-1)
        mean = (expanded_weights * safe_features).sum(dim=1)
        centered = torch.where(
            valid_mask.unsqueeze(-1),
            safe_features - mean.unsqueeze(1),
            torch.zeros_like(safe_features),
        )
        variance = (expanded_weights * centered.square()).sum(dim=1)
        sample_valid = valid_mask.any(dim=1)
        std = torch.sqrt(variance.clamp_min(self.epsilon))
        std = torch.where(sample_valid.unsqueeze(1), std, torch.zeros_like(std))
        statistics = torch.cat((mean, std), dim=-1)
        statistics = torch.where(
            sample_valid.unsqueeze(1),
            statistics,
            torch.zeros_like(statistics),
        )
        return statistics, attention_weights, sample_valid


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
