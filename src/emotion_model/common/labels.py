"""Label protocols and derived quadrant utilities."""

from enum import StrEnum

import torch
from torch import Tensor

_PROBABILITY_ATOL = 1.0e-6
_PROBABILITY_RTOL = 1.0e-5


class LabelProtocol(StrEnum):
    """Named mappings from integer emotion scores to binary class labels."""

    OFFICIAL_MID_HIGH = "official_mid_high"
    STRICT_DROP_MID = "strict_drop_mid"
    MID_LOW = "mid_low"


def _resolve_protocol(protocol: LabelProtocol | str) -> LabelProtocol:
    try:
        return LabelProtocol(protocol)
    except (TypeError, ValueError) as error:
        available = ", ".join(item.value for item in LabelProtocol)
        raise ValueError(
            f"Unknown label protocol {protocol!r}; expected one of: {available}."
        ) from error


def _validate_ignore_index(ignore_index: int) -> None:
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError(f"ignore_index must be an integer; received {ignore_index!r}.")
    if ignore_index in (0, 1):
        raise ValueError("ignore_index must differ from valid binary labels 0 and 1.")


def _validate_scores(scores: Tensor) -> None:
    if scores.dtype == torch.bool or scores.is_complex():
        raise TypeError(
            "scores must use a real numeric dtype containing integer-valued ratings."
        )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values.")
    if scores.is_floating_point() and not bool(torch.eq(scores, torch.round(scores)).all()):
        raise ValueError(
            "scores must contain integer-valued ratings; fractional values are invalid."
        )
    if not bool(((scores >= 1) & (scores <= 5)).all()):
        raise ValueError("scores must lie in the inclusive range [1, 5].")


def binarize_emotion_scores(
    scores: Tensor,
    protocol: LabelProtocol | str,
    *,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """Map 1–5 emotion ratings to binary labels under a named protocol.

    Args:
        scores: Real numeric tensor of any shape ``[...]``. Every value must be
            finite, integer-valued, and within the inclusive range 1–5.
        protocol: A :class:`LabelProtocol` or its string value.
        ignore_index: Integer stored in ``labels`` for protocol-ignored scores.
            It must differ from valid binary labels 0 and 1.

    Returns:
        A tuple ``(labels, valid_mask)`` with the same shape as ``scores``.
        ``labels`` has dtype ``torch.long``. ``valid_mask`` has dtype
        ``torch.bool`` with ``True`` meaning a valid label. Under
        ``strict_drop_mid``, score 3 receives ``ignore_index`` and ``False``;
        all legal scores are valid under the other protocols.

    Raises:
        TypeError: If ``scores`` is boolean or complex, or ``ignore_index`` is
            not an integer.
        ValueError: If a score is non-finite, fractional, or outside 1–5; if
            ``protocol`` is unknown; or if ``ignore_index`` collides with 0/1.

    The input tensor is never modified.
    """
    resolved_protocol = _resolve_protocol(protocol)
    _validate_ignore_index(ignore_index)
    _validate_scores(scores)

    integer_scores = scores.to(dtype=torch.long)
    valid_mask = torch.ones_like(integer_scores, dtype=torch.bool)

    if resolved_protocol is LabelProtocol.OFFICIAL_MID_HIGH:
        labels = (integer_scores >= 3).to(dtype=torch.long)
    elif resolved_protocol is LabelProtocol.MID_LOW:
        labels = (integer_scores >= 4).to(dtype=torch.long)
    else:
        valid_mask = integer_scores != 3
        binary_labels = (integer_scores >= 4).to(dtype=torch.long)
        labels = torch.where(
            valid_mask,
            binary_labels,
            torch.full_like(binary_labels, ignore_index),
        )
    return labels, valid_mask


def _validate_binary_label_tensor(
    labels: Tensor,
    *,
    name: str,
    ignore_index: int,
) -> None:
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"{name} must use an integer dtype with values 0, 1, or ignore_index.")
    allowed = (labels == 0) | (labels == 1) | (labels == ignore_index)
    if not bool(allowed.all()):
        raise ValueError(f"{name} may contain only 0, 1, or ignore_index={ignore_index}.")


def derive_quadrant_labels(
    arousal_labels: Tensor,
    valence_labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Derive quadrant class labels from binary arousal and valence labels.

    Args:
        arousal_labels: Integer tensor with shape ``[...]`` containing 0 for low,
            1 for high, or ``ignore_index``.
        valence_labels: Integer tensor with exactly the same shape and allowed
            values as ``arousal_labels``.
        ignore_index: Integer propagated whenever either input is ignored. It
            must differ from valid binary labels 0 and 1.

    Returns:
        ``torch.long`` tensor with shape ``[...]``. Valid entries use
        ``arousal + 2 * valence``, giving the fixed order
        ``[LALV=0, HALV=1, LAHV=2, HAHV=3]``. Any ignored input produces
        ``ignore_index``.

    Raises:
        TypeError: If either label tensor is not integer or ``ignore_index`` is
            not an integer.
        ValueError: If input shapes or devices differ, a label is not 0, 1, or
            ``ignore_index``, or ``ignore_index`` collides with 0/1. Shapes are
            never broadcast.

    Neither input tensor is modified.
    """
    _validate_ignore_index(ignore_index)
    if tuple(arousal_labels.shape) != tuple(valence_labels.shape):
        raise ValueError(
            "arousal_labels and valence_labels must have exactly the same shape; "
            f"received {tuple(arousal_labels.shape)} and {tuple(valence_labels.shape)}."
        )
    if arousal_labels.device != valence_labels.device:
        raise ValueError(
            "arousal_labels and valence_labels must be on the same device; "
            f"received {arousal_labels.device} and {valence_labels.device}."
        )
    _validate_binary_label_tensor(
        arousal_labels,
        name="arousal_labels",
        ignore_index=ignore_index,
    )
    _validate_binary_label_tensor(
        valence_labels,
        name="valence_labels",
        ignore_index=ignore_index,
    )

    valid_mask = (arousal_labels != ignore_index) & (valence_labels != ignore_index)
    derived = arousal_labels.to(dtype=torch.long) + 2 * valence_labels.to(dtype=torch.long)
    return torch.where(
        valid_mask,
        derived,
        torch.full_like(derived, ignore_index),
    )


def _validate_binary_probabilities(
    probabilities: Tensor,
    *,
    name: str,
) -> None:
    if not probabilities.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    if probabilities.ndim == 0 or probabilities.shape[-1] != 2:
        raise ValueError(
            f"{name} must have shape [..., 2]; received {tuple(probabilities.shape)}."
        )
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError(f"{name} values must lie in the inclusive range [0, 1].")

    probability_sums = probabilities.sum(dim=-1)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        rtol=_PROBABILITY_RTOL,
        atol=_PROBABILITY_ATOL,
    ):
        raise ValueError(f"{name} must sum to 1 along its final dimension.")


def derive_quadrant_probabilities(
    arousal_probabilities: Tensor,
    valence_probabilities: Tensor,
) -> Tensor:
    """Derive four quadrant probabilities from two binary distributions.

    Args:
        arousal_probabilities: Floating-point tensor with shape ``[..., 2]`` in
            ``[low, high]`` order. Values must be finite, within [0, 1], and sum
            to approximately 1 along the final dimension.
        valence_probabilities: Tensor with exactly the same shape, dtype, device,
            ordering, and probability constraints as ``arousal_probabilities``.

    Returns:
        Tensor with shape ``[..., 4]`` and the same dtype/device, ordered as
        ``[LALV, HALV, LAHV, HAHV]``. Entries are the pairwise products
        ``[A_low*V_low, A_high*V_low, A_low*V_high, A_high*V_high]`` and sum
        approximately to 1. The result remains connected to both input
        computation graphs.

    Raises:
        TypeError: If either input is not floating point or dtypes differ.
        ValueError: If shapes or devices differ, the final dimension is not 2,
            values are non-finite or outside [0, 1], or a binary distribution
            does not sum approximately to 1. Shapes are never broadcast.

    Neither input tensor is modified or detached.
    """
    if tuple(arousal_probabilities.shape) != tuple(valence_probabilities.shape):
        raise ValueError(
            "arousal_probabilities and valence_probabilities must have exactly "
            f"the same shape; received {tuple(arousal_probabilities.shape)} and "
            f"{tuple(valence_probabilities.shape)}."
        )
    if arousal_probabilities.dtype != valence_probabilities.dtype:
        raise TypeError(
            "arousal_probabilities and valence_probabilities must have the same dtype; "
            f"received {arousal_probabilities.dtype} and {valence_probabilities.dtype}."
        )
    if arousal_probabilities.device != valence_probabilities.device:
        raise ValueError(
            "arousal_probabilities and valence_probabilities must be on the same device; "
            f"received {arousal_probabilities.device} and {valence_probabilities.device}."
        )
    _validate_binary_probabilities(
        arousal_probabilities,
        name="arousal_probabilities",
    )
    _validate_binary_probabilities(
        valence_probabilities,
        name="valence_probabilities",
    )

    arousal_low, arousal_high = arousal_probabilities.unbind(dim=-1)
    valence_low, valence_high = valence_probabilities.unbind(dim=-1)
    return torch.stack(
        (
            arousal_low * valence_low,
            arousal_high * valence_low,
            arousal_low * valence_high,
            arousal_high * valence_high,
        ),
        dim=-1,
    )
