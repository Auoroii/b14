"""Tests for leakage-aware channelwise physiology normalization."""

import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from emotion_model.physiology import (
    ChannelNormalizationStats,
    ChannelwiseZScoreNormalizer,
    NormalizationFitScope,
    NormalizationKey,
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)


def _window(
    name: str,
    values: list[float],
    mask: list[bool] | None = None,
) -> PhysioChannelWindow:
    """Create an explicit named channel window."""
    tensor = torch.tensor(values, dtype=torch.float32)
    if mask is None:
        mask = [True] * len(values)
    return PhysioChannelWindow(
        PhysioChannelSpec(name, PhysioSignalKind.OTHER, 10.0, "a.u.", "test"),
        tensor,
        torch.tensor(mask),
        torch.arange(len(values), dtype=torch.float32) / 10.0,
    )


def test_normalization_key_and_scopes_are_explicit_and_frozen() -> None:
    """Represent global/participant groups and only authorized fit sources."""
    global_key = NormalizationKey("eda")
    participant_key = NormalizationKey("eda", "participant_01")

    assert global_key.group_id is None
    assert participant_key.group_id == "participant_01"
    assert {scope.value for scope in NormalizationFitScope} == {
        "train",
        "calibration",
    }
    with pytest.raises(FrozenInstanceError):
        global_key.channel_name = "bvp"
    with pytest.raises(ValueError, match="channel_name"):
        NormalizationKey("")
    with pytest.raises(ValueError, match="group_id"):
        NormalizationKey("eda", "")


@pytest.mark.parametrize(
    ("epsilon", "exception"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
    ],
)
def test_normalizer_rejects_invalid_epsilon(
    epsilon: float,
    exception: type[Exception],
) -> None:
    """Require a finite positive normalization scale floor."""
    with pytest.raises(exception):
        ChannelwiseZScoreNormalizer(epsilon=epsilon)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (NormalizationFitScope.TRAIN, NormalizationFitScope.TRAIN),
        ("train", NormalizationFitScope.TRAIN),
        (NormalizationFitScope.CALIBRATION, NormalizationFitScope.CALIBRATION),
        ("calibration", NormalizationFitScope.CALIBRATION),
    ],
)
def test_fit_accepts_only_explicit_authorized_scopes(
    scope: NormalizationFitScope | str,
    expected: NormalizationFitScope,
) -> None:
    """Store the caller-declared training or calibration provenance."""
    normalizer = ChannelwiseZScoreNormalizer().fit(
        {NormalizationKey("eda"): [_window("eda", [1.0, 2.0])]},
        scope=scope,
    )
    assert normalizer.fit_scope is expected


@pytest.mark.parametrize("scope", ["test", "validation_full", "all_data", "future"])
def test_fit_rejects_leaky_or_unknown_scope(scope: str) -> None:
    """Reject test, whole-validation, all-data, and unknown fit provenance."""
    with pytest.raises(ValueError, match="fit scope"):
        ChannelwiseZScoreNormalizer().fit(
            {NormalizationKey("eda"): [_window("eda", [1.0])]},
            scope=scope,
        )


def test_fit_computes_independent_population_statistics_and_ignores_padding() -> None:
    """Fit each channel/group independently using only valid values."""
    eda_global = NormalizationKey("eda")
    eda_participant = NormalizationKey("eda", "p01")
    bvp_global = NormalizationKey("bvp")
    padding_window = _window(
        "eda",
        [1.0, float("nan"), 3.0],
        [True, False, True],
    )
    normalizer = ChannelwiseZScoreNormalizer(epsilon=1.0e-4).fit(
        {
            eda_global: [padding_window],
            eda_participant: [_window("eda", [10.0, 14.0])],
            bvp_global: [_window("bvp", [100.0, 100.0])],
        },
        scope="train",
    )
    state = normalizer.state_dict()
    records = {
        (record["channel_name"], record["group_id"]): record
        for record in state["statistics"]  # type: ignore[union-attr]
    }

    assert records[("eda", None)]["mean"] == 2.0
    assert records[("eda", None)]["scale"] == 1.0
    assert records[("eda", None)]["valid_count"] == 2
    assert records[("eda", "p01")]["mean"] == 12.0
    assert records[("eda", "p01")]["scale"] == 2.0
    assert records[("bvp", None)]["scale"] == 1.0e-4
    torch.testing.assert_close(
        padding_window.values,
        torch.tensor([1.0, float("nan"), 3.0]),
        equal_nan=True,
    )


def test_transform_uses_exact_key_and_preserves_mask_timestamps_spec() -> None:
    """Apply (x-mean)/scale and zero invalid positions without fallback."""
    key = NormalizationKey("eda", "p01")
    fit_window = _window("eda", [10.0, 14.0])
    normalizer = ChannelwiseZScoreNormalizer().fit(
        {key: [fit_window]},
        scope="calibration",
    )
    source = _window(
        "eda",
        [8.0, float("inf"), 16.0],
        [True, False, True],
    )
    original = source.values.clone()

    transformed = normalizer.transform(source, key=key)

    assert torch.equal(transformed.values, torch.tensor([-2.0, 0.0, 2.0]))
    assert transformed.spec is source.spec
    assert transformed.valid_mask is source.valid_mask
    assert transformed.timestamps_seconds is source.timestamps_seconds
    torch.testing.assert_close(source.values, original, equal_nan=True)


def test_constant_channel_transform_is_finite() -> None:
    """Use epsilon for zero population variance without NaN or Inf."""
    key = NormalizationKey("temp")
    normalizer = ChannelwiseZScoreNormalizer(epsilon=0.125).fit(
        {key: [_window("temp", [5.0, 5.0, 5.0])]},
        scope="train",
    )
    transformed = normalizer.transform(
        _window("temp", [5.0, 5.125]),
        key=key,
    )
    torch.testing.assert_close(transformed.values, torch.tensor([0.0, 1.0]))
    assert bool(torch.isfinite(transformed.values).all())


def test_fit_and_transform_reject_invalid_lifecycle_and_keys() -> None:
    """Reject empty/no-valid fit, repeat fit, unknown keys, and name mismatch."""
    key = NormalizationKey("eda")
    with pytest.raises(ValueError, match="non-empty"):
        ChannelwiseZScoreNormalizer().fit({}, scope="train")
    with pytest.raises(ValueError, match="no valid"):
        ChannelwiseZScoreNormalizer().fit(
            {key: [_window("eda", [float("nan")], [False])]},
            scope="train",
        )
    with pytest.raises(ValueError, match="channel name"):
        ChannelwiseZScoreNormalizer().fit(
            {key: [_window("bvp", [1.0])]},
            scope="train",
        )

    normalizer = ChannelwiseZScoreNormalizer()
    with pytest.raises(RuntimeError, match="fitted"):
        normalizer.transform(_window("eda", [1.0]), key=key)
    normalizer.fit({key: [_window("eda", [1.0, 2.0])]}, scope="train")
    with pytest.raises(RuntimeError, match="already fitted"):
        normalizer.fit({key: [_window("eda", [1.0])]}, scope="train")
    with pytest.raises(KeyError, match="Unknown"):
        normalizer.transform(
            _window("eda", [1.0]),
            key=NormalizationKey("eda", "not-fitted"),
        )
    with pytest.raises(ValueError, match="channel name"):
        normalizer.transform(_window("bvp", [1.0]), key=key)


def test_state_round_trip_is_json_compatible_and_transform_equivalent() -> None:
    """Serialize only scalar statistics and reproduce transform exactly."""
    key = NormalizationKey("eda", "participant_01")
    source = ChannelwiseZScoreNormalizer(epsilon=1.0e-5).fit(
        {key: [_window("eda", [1.0, 2.0, 3.0])]},
        scope="train",
    )
    state = source.state_dict()
    encoded = json.dumps(state)
    restored_state = json.loads(encoded)
    restored = ChannelwiseZScoreNormalizer()
    restored.load_state_dict(restored_state)
    target = _window("eda", [0.0, 2.0, 4.0])

    expected = source.transform(target, key=key)
    actual = restored.transform(target, key=key)

    torch.testing.assert_close(actual.values, expected.values)
    assert restored.epsilon == source.epsilon
    assert restored.fit_scope is NormalizationFitScope.TRAIN
    assert "values" not in encoded
    assert "timestamps" not in encoded
    assert ":\\" not in encoded and ":/" not in encoded


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.pop("epsilon"),
        lambda state: state.update(extra_field="not-allowed"),
        lambda state: state.update(version=999),
        lambda state: state.update(version=True),
        lambda state: state.update(fit_scope="test"),
        lambda state: state.update(statistics=[]),
        lambda state: state["statistics"][0].pop("mean"),
        lambda state: state["statistics"][0].update(scale=0.0),
    ],
)
def test_load_state_rejects_missing_or_invalid_fields(mutator: object) -> None:
    """Reject incomplete, unknown-version, leaky, empty, and unsafe state."""
    key = NormalizationKey("eda")
    source = ChannelwiseZScoreNormalizer().fit(
        {key: [_window("eda", [1.0, 2.0])]},
        scope="train",
    )
    state = source.state_dict()
    mutator(state)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        ChannelwiseZScoreNormalizer().load_state_dict(state)


def test_load_state_rejects_duplicate_keys_and_fitted_target() -> None:
    """Do not silently overwrite duplicate serialized statistics."""
    key = NormalizationKey("eda")
    source = ChannelwiseZScoreNormalizer().fit(
        {key: [_window("eda", [1.0, 2.0])]},
        scope="train",
    )
    state = source.state_dict()
    records = state["statistics"]
    assert isinstance(records, list)
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        ChannelwiseZScoreNormalizer().load_state_dict(state)

    fitted = ChannelwiseZScoreNormalizer().fit(
        {key: [_window("eda", [1.0])]},
        scope="train",
    )
    with pytest.raises(RuntimeError, match="already fitted"):
        fitted.load_state_dict(source.state_dict())


def test_state_keeps_channel_group_keys_distinct_and_unknown_after_load() -> None:
    """Round-trip colliding names/groups without implicit global fallback."""
    keys = (
        NormalizationKey("eda"),
        NormalizationKey("eda", "p01"),
        NormalizationKey("bvp", "p01"),
    )
    source = ChannelwiseZScoreNormalizer().fit(
        {
            keys[0]: [_window("eda", [1.0, 2.0])],
            keys[1]: [_window("eda", [10.0, 12.0])],
            keys[2]: [_window("bvp", [100.0, 104.0])],
        },
        scope="train",
    )
    restored = ChannelwiseZScoreNormalizer()
    restored.load_state_dict(source.state_dict())

    expected = (
        source.transform(_window("eda", [2.0]), key=keys[0]).values,
        source.transform(_window("eda", [12.0]), key=keys[1]).values,
        source.transform(_window("bvp", [104.0]), key=keys[2]).values,
    )
    actual = (
        restored.transform(_window("eda", [2.0]), key=keys[0]).values,
        restored.transform(_window("eda", [12.0]), key=keys[1]).values,
        restored.transform(_window("bvp", [104.0]), key=keys[2]).values,
    )
    for actual_values, expected_values in zip(actual, expected):
        torch.testing.assert_close(actual_values, expected_values)
    with pytest.raises(KeyError, match="Unknown"):
        restored.transform(
            _window("eda", [1.0]),
            key=NormalizationKey("eda", "p02"),
        )


def test_stats_are_frozen_and_validate_population_state() -> None:
    """Keep serialized scalar statistics immutable and valid."""
    stats = ChannelNormalizationStats(mean=1.0, scale=2.0, valid_count=3)
    with pytest.raises(FrozenInstanceError):
        stats.mean = 0.0
    with pytest.raises(ValueError, match="scale"):
        ChannelNormalizationStats(mean=1.0, scale=0.0, valid_count=3)
    with pytest.raises(ValueError, match="valid_count"):
        ChannelNormalizationStats(mean=1.0, scale=1.0, valid_count=0)
