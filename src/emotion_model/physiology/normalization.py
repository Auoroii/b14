"""Leakage-aware channelwise physiology normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch

from emotion_model.physiology.channel_metadata import PhysioChannelWindow


@dataclass(frozen=True)
class NormalizationKey:
    """Identify independent channel statistics without inferring group identity.

    Args:
        channel_name: Non-empty channel name matching ``window.spec.name``.
        group_id: Optional explicit grouping identifier. ``None`` denotes global
            channel statistics; a participant or session identifier may be
            supplied by the caller under its own explicit data contract.
    """

    channel_name: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel_name, str):
            raise TypeError("channel_name must be a string.")
        if not self.channel_name.strip():
            raise ValueError("channel_name must be non-empty.")
        if self.group_id is not None:
            if not isinstance(self.group_id, str):
                raise TypeError("group_id must be a string or None.")
            if not self.group_id.strip():
                raise ValueError("group_id must be non-empty when supplied.")


class NormalizationFitScope(StrEnum):
    """Authorized sources for fitting normalization statistics.

    ``TRAIN`` means statistics come only from the training fold.
    ``CALIBRATION`` means statistics come only from explicitly designated
    calibration windows. Test, full-validation, all-data, or future participant
    data are not legal fit scopes.
    """

    TRAIN = "train"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class ChannelNormalizationStats:
    """JSON-compatible statistics for one normalization key.

    Args:
        mean: Finite arithmetic mean of valid fit values.
        scale: Finite positive ``max(population_std, epsilon)``.
        valid_count: Positive number of fit values.
    """

    mean: float
    scale: float
    valid_count: int

    def __post_init__(self) -> None:
        for name, value in (("mean", self.mean), ("scale", self.scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if float(self.scale) <= 0.0:
            raise ValueError("scale must be > 0.")
        if isinstance(self.valid_count, bool) or not isinstance(self.valid_count, int):
            raise TypeError("valid_count must be an integer.")
        if self.valid_count <= 0:
            raise ValueError("valid_count must be > 0.")
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "scale", float(self.scale))


class ChannelwiseZScoreNormalizer:
    """Fit and apply explicit-key channelwise population z-score statistics.

    Args:
        epsilon: Finite positive minimum scale for constant channels.

    Repeated fitting is rejected. :meth:`fit` requires an explicit
    :class:`NormalizationFitScope`: training-fold windows or separately
    designated calibration windows only. The class cannot detect chronology or
    participant leakage itself; callers must not pass test, future-session, or
    full-participant data under an authorized name.
    """

    _STATE_VERSION = 1

    def __init__(self, *, epsilon: float = 1.0e-6) -> None:
        self._epsilon = self._validate_epsilon(epsilon)
        self._fit_scope: NormalizationFitScope | None = None
        self._statistics: dict[NormalizationKey, ChannelNormalizationStats] = {}

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a real number.")
        value = float(epsilon)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("epsilon must be finite and > 0.")
        return value

    @property
    def epsilon(self) -> float:
        """Return the finite positive scale floor."""
        return self._epsilon

    @property
    def fit_scope(self) -> NormalizationFitScope | None:
        """Return the explicit fit scope, or ``None`` before fitting."""
        return self._fit_scope

    @property
    def is_fitted(self) -> bool:
        """Return whether statistics have been fitted or loaded."""
        return self._fit_scope is not None

    @staticmethod
    def _resolve_scope(
        scope: NormalizationFitScope | str,
    ) -> NormalizationFitScope:
        try:
            return NormalizationFitScope(scope)
        except (TypeError, ValueError) as error:
            allowed = ", ".join(item.value for item in NormalizationFitScope)
            raise ValueError(
                f"Invalid normalization fit scope {scope!r}; expected {allowed}."
            ) from error

    def fit(
        self,
        samples: Mapping[NormalizationKey, Sequence[PhysioChannelWindow]],
        *,
        scope: NormalizationFitScope | str,
    ) -> ChannelwiseZScoreNormalizer:
        """Fit independent statistics using only effective values.

        Args:
            samples: Non-empty mapping from explicit keys to channel-window
                sequences. Each window contains ``values/mask/timestamps: [T]``
                and must have ``spec.name == key.channel_name``.
            scope: ``"train"`` or ``"calibration"``. Other strings, including
                ``"test"``, ``"validation_full"``, and ``"all_data"``, fail.

        Returns:
            ``self`` after fitting per-key mean, population standard deviation,
            and valid count.

        Raises:
            RuntimeError: If this instance is already fitted.
            TypeError: If mapping keys/windows have invalid types.
            ValueError: If samples are empty, a channel name mismatches, or a
                key has no valid values.

        Invalid placeholders, labels, and unlisted groups are never consumed.
        Inputs are not modified. Python mappings already enforce unique keys;
        no internal key overwrite occurs.
        """
        if self.is_fitted:
            raise RuntimeError("ChannelwiseZScoreNormalizer is already fitted.")
        resolved_scope = self._resolve_scope(scope)
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a mapping.")
        if not samples:
            raise ValueError("samples must be non-empty.")

        fitted_statistics: dict[NormalizationKey, ChannelNormalizationStats] = {}
        for key, windows in samples.items():
            if not isinstance(key, NormalizationKey):
                raise TypeError("samples keys must be NormalizationKey objects.")
            if not isinstance(windows, Sequence):
                raise TypeError("each samples value must be a window sequence.")
            effective_values: list[float] = []
            for window in windows:
                if not isinstance(window, PhysioChannelWindow):
                    raise TypeError("samples sequences must contain PhysioChannelWindow.")
                if window.spec.name != key.channel_name:
                    raise ValueError(
                        "window channel name must match normalization key; "
                        f"received {window.spec.name!r} and {key.channel_name!r}."
                    )
                effective_values.extend(
                    float(value)
                    for value in window.values[window.valid_mask]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .tolist()
                )
            if not effective_values:
                raise ValueError(
                    f"normalization key {key!r} has no valid fit values."
                )
            valid_count = len(effective_values)
            mean = math.fsum(effective_values) / valid_count
            variance = (
                math.fsum((value - mean) ** 2 for value in effective_values)
                / valid_count
            )
            population_std = math.sqrt(max(variance, 0.0))
            fitted_statistics[key] = ChannelNormalizationStats(
                mean=mean,
                scale=max(population_std, self._epsilon),
                valid_count=valid_count,
            )
        self._statistics = fitted_statistics
        self._fit_scope = resolved_scope
        return self

    def transform(
        self,
        window: PhysioChannelWindow,
        *,
        key: NormalizationKey,
    ) -> PhysioChannelWindow:
        """Normalize one window using one exact fitted key.

        Args:
            window: Source channel window with ``values/mask/timestamps: [T]``.
            key: Exact fitted channel/group key. Unknown keys never fall back to
                global statistics.

        Returns:
            New window with effective values ``(x - mean) / scale`` and invalid
            values exactly zero. Spec, mask, and timestamp objects are preserved.

        Raises:
            RuntimeError: If the normalizer is not fitted.
            TypeError: If window/key types are invalid.
            KeyError: If the exact key was not fitted.
            ValueError: If ``window.spec.name`` differs from ``key.channel_name``.

        Constant channels are finite because scale is at least ``epsilon``.
        Inputs are not modified.
        """
        if not self.is_fitted:
            raise RuntimeError("ChannelwiseZScoreNormalizer must be fitted first.")
        if not isinstance(window, PhysioChannelWindow):
            raise TypeError("window must be a PhysioChannelWindow.")
        if not isinstance(key, NormalizationKey):
            raise TypeError("key must be a NormalizationKey.")
        if window.spec.name != key.channel_name:
            raise ValueError(
                "window channel name must match normalization key; "
                f"received {window.spec.name!r} and {key.channel_name!r}."
            )
        if key not in self._statistics:
            raise KeyError(f"Unknown normalization key {key!r}.")
        statistics = self._statistics[key]
        safe_values = window.safe_values()
        normalized = (safe_values - statistics.mean) / statistics.scale
        normalized = torch.where(
            window.valid_mask,
            normalized,
            torch.zeros_like(normalized),
        )
        if not bool(torch.isfinite(normalized[window.valid_mask]).all()):
            raise RuntimeError("normalization produced NaN or Inf at valid positions.")
        return PhysioChannelWindow(
            spec=window.spec,
            values=normalized,
            valid_mask=window.valid_mask,
            timestamps_seconds=window.timestamps_seconds,
        )

    def state_dict(self) -> dict[str, object]:
        """Return a JSON-compatible fitted state without raw channel data.

        Returns:
            Mapping containing version, epsilon, fit scope, and a list of
            ``channel_name/group_id/mean/scale/valid_count`` records.

        Raises:
            RuntimeError: If the normalizer has not been fitted.
        """
        if self._fit_scope is None:
            raise RuntimeError("Cannot serialize an unfitted normalizer.")
        ordered_items = sorted(
            self._statistics.items(),
            key=lambda item: (
                item[0].channel_name,
                "" if item[0].group_id is None else item[0].group_id,
            ),
        )
        records: list[dict[str, object]] = []
        for key, statistics in ordered_items:
            records.append(
                {
                    "channel_name": key.channel_name,
                    "group_id": key.group_id,
                    "mean": statistics.mean,
                    "scale": statistics.scale,
                    "valid_count": statistics.valid_count,
                }
            )
        return {
            "version": self._STATE_VERSION,
            "epsilon": self._epsilon,
            "fit_scope": self._fit_scope.value,
            "statistics": records,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load a validated JSON-compatible state into an unfitted instance.

        Args:
            state: Mapping produced by :meth:`state_dict`. It contains no raw
                tensors, labels, or local paths.

        Returns:
            ``None``.

        Raises:
            RuntimeError: If this instance is already fitted.
            TypeError: If state or nested fields have invalid types.
            ValueError: If fields are missing/extra, version/scope/statistics
                are invalid, a key is duplicated, or scale is below epsilon.
        """
        if self.is_fitted:
            raise RuntimeError("Cannot load state into an already fitted normalizer.")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        expected_fields = {"version", "epsilon", "fit_scope", "statistics"}
        actual_fields = set(state)
        if actual_fields != expected_fields:
            raise ValueError(
                "normalizer state fields must be exactly "
                f"{sorted(expected_fields)}; received {sorted(actual_fields)}."
            )
        version = state["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != self._STATE_VERSION
        ):
            raise ValueError(
                f"Unsupported normalizer state version {version!r}."
            )
        epsilon_value = state["epsilon"]
        if isinstance(epsilon_value, bool) or not isinstance(
            epsilon_value,
            (int, float),
        ):
            raise TypeError("state epsilon must be a real number.")
        loaded_epsilon = self._validate_epsilon(float(epsilon_value))
        loaded_scope = self._resolve_scope(state["fit_scope"])  # type: ignore[arg-type]
        records = state["statistics"]
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("state statistics must be a sequence.")
        if not records:
            raise ValueError("state statistics must be non-empty.")
        loaded_statistics: dict[NormalizationKey, ChannelNormalizationStats] = {}
        expected_record_fields = {
            "channel_name",
            "group_id",
            "mean",
            "scale",
            "valid_count",
        }
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("each state statistics record must be a mapping.")
            if set(record) != expected_record_fields:
                raise ValueError(
                    "statistics record fields must be exactly "
                    f"{sorted(expected_record_fields)}."
                )
            channel_name = record["channel_name"]
            group_id = record["group_id"]
            if not isinstance(channel_name, str):
                raise TypeError("state channel_name must be a string.")
            if group_id is not None and not isinstance(group_id, str):
                raise TypeError("state group_id must be a string or None.")
            key = NormalizationKey(channel_name=channel_name, group_id=group_id)
            if key in loaded_statistics:
                raise ValueError(f"Duplicate normalization key in state: {key!r}.")
            mean = record["mean"]
            scale = record["scale"]
            valid_count = record["valid_count"]
            if (
                isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or isinstance(scale, bool)
                or not isinstance(scale, (int, float))
            ):
                raise TypeError("state mean and scale must be real numbers.")
            if isinstance(valid_count, bool) or not isinstance(valid_count, int):
                raise TypeError("state valid_count must be an integer.")
            statistics = ChannelNormalizationStats(
                mean=float(mean),
                scale=float(scale),
                valid_count=valid_count,
            )
            if statistics.scale < loaded_epsilon:
                raise ValueError("state scale must be at least state epsilon.")
            loaded_statistics[key] = statistics
        self._epsilon = loaded_epsilon
        self._fit_scope = loaded_scope
        self._statistics = loaded_statistics
