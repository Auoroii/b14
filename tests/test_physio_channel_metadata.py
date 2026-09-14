"""Tests for explicit physiology channel metadata and window contracts."""

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

import emotion_model.physiology.channel_metadata as metadata_module
import emotion_model.physiology.normalization as normalization_module
import emotion_model.physiology.preprocessing as preprocessing_module
import emotion_model.physiology.quality as quality_module
from emotion_model.physiology import (
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)


def _spec(
    *,
    name: str = "lead_i",
    kind: PhysioSignalKind | str = PhysioSignalKind.ECG_WAVEFORM,
    rate: float = 128.0,
    unit: str = "mV",
    description: str | None = None,
) -> PhysioChannelSpec:
    """Create an explicit test channel specification."""
    return PhysioChannelSpec(name, kind, rate, unit, description)


def test_signal_kind_values_are_stable_and_semantically_distinct() -> None:
    """Keep waveform, heart-rate, and interval meanings separate."""
    assert {kind.value for kind in PhysioSignalKind} == {
        "ecg_waveform",
        "heart_rate",
        "rr_interval",
        "bvp",
        "eda",
        "temperature",
        "other",
    }
    assert PhysioSignalKind.ECG_WAVEFORM is not PhysioSignalKind.HEART_RATE
    assert PhysioSignalKind.HEART_RATE is not PhysioSignalKind.RR_INTERVAL


def test_channel_spec_parses_explicit_string_and_is_frozen() -> None:
    """Parse only a declared stable signal value and preserve semantics."""
    spec = _spec(kind="heart_rate", unit="beats/min")

    assert spec.signal_kind is PhysioSignalKind.HEART_RATE
    assert spec.native_sample_rate_hz == 128.0
    with pytest.raises(FrozenInstanceError):
        spec.unit = "Hz"


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("name", "", ValueError, "name"),
        ("name", "   ", ValueError, "name"),
        ("unit", "", ValueError, "unit"),
        ("native_sample_rate_hz", 0.0, ValueError, "sample_rate"),
        ("native_sample_rate_hz", -1.0, ValueError, "sample_rate"),
        ("native_sample_rate_hz", float("inf"), ValueError, "sample_rate"),
        ("native_sample_rate_hz", True, TypeError, "sample_rate"),
        ("signal_kind", "ecg", ValueError, "Unknown"),
        ("signal_kind", "ECG", ValueError, "Unknown"),
    ],
)
def test_channel_spec_rejects_invalid_fields(
    field: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Reject empty metadata, bad rates, and guessed signal kinds."""
    arguments: dict[str, object] = {
        "name": "channel",
        "signal_kind": PhysioSignalKind.EDA,
        "native_sample_rate_hz": 4.0,
        "unit": "uS",
    }
    arguments[field] = value
    with pytest.raises(exception, match=message):
        PhysioChannelSpec(**arguments)  # type: ignore[arg-type]


def test_other_kind_requires_an_explicit_description() -> None:
    """Do not let an unknown semantic hide behind an unexplained OTHER value."""
    with pytest.raises(ValueError, match="description"):
        _spec(kind=PhysioSignalKind.OTHER)
    spec = _spec(
        kind=PhysioSignalKind.OTHER,
        description="Vendor-defined peripheral pulse amplitude.",
    )
    assert spec.signal_kind is PhysioSignalKind.OTHER


def test_channel_window_accepts_invalid_placeholders_and_safe_values() -> None:
    """Validate effective values while safely zeroing arbitrary placeholders."""
    values = torch.tensor([1.0, float("nan"), 3.0, float("inf")])
    mask = torch.tensor([True, False, True, False])
    timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3])
    original_values = values.clone()
    window = PhysioChannelWindow(_spec(), values, mask, timestamps)

    assert torch.equal(window.safe_values(), torch.tensor([1.0, 0.0, 3.0, 0.0]))
    torch.testing.assert_close(values, original_values, equal_nan=True)
    assert window.valid_mask is mask
    assert window.timestamps_seconds is timestamps
    with pytest.raises(FrozenInstanceError):
        window.spec = _spec(name="replacement")


@pytest.mark.parametrize(
    ("values", "mask", "timestamps", "exception", "message"),
    [
        (
            torch.ones((1, 3)),
            torch.ones(3, dtype=torch.bool),
            torch.arange(3, dtype=torch.float32),
            ValueError,
            "values",
        ),
        (
            torch.ones(3),
            torch.ones((1, 3), dtype=torch.bool),
            torch.arange(3, dtype=torch.float32),
            ValueError,
            "valid_mask",
        ),
        (
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            torch.ones((1, 3)),
            ValueError,
            "timestamps",
        ),
        (
            torch.ones(3),
            torch.ones(2, dtype=torch.bool),
            torch.arange(3, dtype=torch.float32),
            ValueError,
            "identical",
        ),
        (
            torch.ones(3),
            torch.ones(3),
            torch.arange(3, dtype=torch.float32),
            TypeError,
            "torch.bool",
        ),
        (
            torch.ones(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            torch.arange(3, dtype=torch.float32),
            TypeError,
            "floating",
        ),
        (
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            torch.arange(3),
            TypeError,
            "floating",
        ),
        (
            torch.tensor([]),
            torch.tensor([], dtype=torch.bool),
            torch.tensor([]),
            ValueError,
            "non-empty",
        ),
    ],
)
def test_channel_window_rejects_shape_dtype_and_length_errors(
    values: torch.Tensor,
    mask: torch.Tensor,
    timestamps: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    """Reject non-vector, mismatched, empty, and invalid dtype contracts."""
    with pytest.raises(exception, match=message):
        PhysioChannelWindow(_spec(), values, mask, timestamps)


@pytest.mark.parametrize(
    "timestamps",
    [
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([0.0, float("inf"), 2.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, 2.0, 1.0]),
    ],
)
def test_channel_window_rejects_invalid_or_nonincreasing_timestamps(
    timestamps: torch.Tensor,
) -> None:
    """Require finite, strictly increasing explicit time coordinates."""
    with pytest.raises(ValueError, match="timestamps_seconds"):
        PhysioChannelWindow(
            _spec(),
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            timestamps,
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_channel_window_rejects_nonfinite_valid_values(invalid_value: float) -> None:
    """Do not silently repair non-finite effective observations."""
    values = torch.tensor([1.0, invalid_value, 3.0])
    with pytest.raises(ValueError, match="valid positions"):
        PhysioChannelWindow(
            _spec(),
            values,
            torch.tensor([True, True, False]),
            torch.arange(3, dtype=torch.float32),
        )


def test_stage_scope_has_no_dataset_neural_branch_or_forbidden_logic() -> None:
    """Keep physiology infrastructure free of later models and prohibited logic."""
    modules = (
        metadata_module,
        preprocessing_module,
        quality_module,
        normalization_module,
    )
    source = "\n".join(inspect.getsource(module).lower() for module in modules)
    for identifier in (
        "clean_speech",
        "reconstruct_waveform",
        "denoise_output",
        "noise_subtraction",
        "speech_enhancement",
        "batchnorm",
        "k-emocon",
    ):
        assert identifier not in source
    for module in modules:
        declared_classes = [
            value
            for _, value in inspect.getmembers(module, inspect.isclass)
            if value.__module__ == module.__name__
        ]
        assert not any(
            issubclass(value, (torch.nn.Module, torch.utils.data.Dataset))
            for value in declared_classes
        )
