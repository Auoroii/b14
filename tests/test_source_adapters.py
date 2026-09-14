"""Tests for injectable source contracts and generic artifact policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from emotion_model.data import (
    NormalizationKeyResolver,
    PhysioArtifactPolicy,
    PhysioChannelSourceRef,
    PhysioSourceAdapter,
    SpeechSourceAdapter,
    SpeechSourceData,
    TimedSourceRef,
    TimeInterval,
)
from emotion_model.physiology import (
    NormalizationKey,
    PhysioChannelSpec,
    PhysioChannelWindow,
    PhysioSignalKind,
)


def test_speech_source_data_accepts_finite_mono_cpu_waveform() -> None:
    """Preserve a valid CPU waveform without normalization or mutation."""
    waveform = torch.tensor([-2.0, 0.0, 3.5], dtype=torch.float64)
    original = waveform.clone()

    data = SpeechSourceData(waveform, 16000, -1)

    assert data.waveform is waveform
    assert data.sample_rate_hz == 16000
    assert data.timeline_start_seconds == -1.0
    assert torch.equal(waveform, original)


@pytest.mark.parametrize(
    ("waveform", "exception"),
    [
        (torch.ones((1, 3)), ValueError),
        (torch.tensor([], dtype=torch.float32), ValueError),
        (torch.tensor([1, 2]), TypeError),
        (torch.tensor([1.0, float("nan")]), ValueError),
        (torch.tensor([1.0, float("inf")]), ValueError),
        (torch.empty(2, device="meta"), ValueError),
        ([1.0, 2.0], TypeError),
    ],
)
def test_speech_source_data_rejects_invalid_waveform(
    waveform: object,
    exception: type[Exception],
) -> None:
    """Reject non-vector, empty, integer, non-finite, non-CPU, and non-tensors."""
    with pytest.raises(exception):
        SpeechSourceData(waveform, 16000, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rate", "exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (16000.0, TypeError),
        ("16000", TypeError),
    ],
)
def test_speech_source_data_rejects_invalid_rate(
    rate: object,
    exception: type[Exception],
) -> None:
    """Require a genuinely integral positive sample rate."""
    with pytest.raises(exception):
        SpeechSourceData(torch.ones(2), rate, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "exception"),
    [
        (True, TypeError),
        ("0", TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_speech_source_data_rejects_invalid_timeline_start(
    start: object,
    exception: type[Exception],
) -> None:
    """Reject boolean, string, and non-finite timeline origins."""
    with pytest.raises(exception):
        SpeechSourceData(torch.ones(2), 2, start)  # type: ignore[arg-type]


def test_speech_source_data_is_frozen() -> None:
    """Prevent source contract reassignment."""
    data = SpeechSourceData(torch.ones(2), 2, 0.0)
    with pytest.raises(FrozenInstanceError):
        data.sample_rate_hz = 4  # type: ignore[misc]


def test_adapter_protocols_are_structural_without_base_classes() -> None:
    """Accept ordinary objects that implement the protocol methods."""

    class SpeechMemory:
        def load_speech(self, source: TimedSourceRef) -> SpeechSourceData:
            del source
            return SpeechSourceData(torch.ones(2), 2, 0.0)

    class PhysioMemory:
        def load_physio(
            self,
            source: PhysioChannelSourceRef,
            spec: PhysioChannelSpec,
        ) -> PhysioChannelWindow:
            del source
            return PhysioChannelWindow(
                spec,
                torch.ones(2),
                torch.ones(2, dtype=torch.bool),
                torch.tensor([0.0, 1.0]),
            )

    speech_adapter: SpeechSourceAdapter = SpeechMemory()
    physio_adapter: PhysioSourceAdapter = PhysioMemory()

    assert speech_adapter.load_speech(
        TimedSourceRef("memory:key", TimeInterval(0, 1))
    ).waveform.shape == (2,)
    spec = PhysioChannelSpec("eda", PhysioSignalKind.EDA, 2.0, "uS")
    source = PhysioChannelSourceRef(
        "eda",
        TimedSourceRef("not-a-path", TimeInterval(0, 1)),
    )
    assert physio_adapter.load_physio(source, spec).spec is spec


def test_normalization_key_resolver_protocol_is_structural() -> None:
    """Describe callable key selection without requiring inheritance."""

    class Resolver:
        def __call__(
            self,
            record: object,
            spec: PhysioChannelSpec,
        ) -> NormalizationKey:
            del record
            return NormalizationKey(spec.name, "participant")

    resolver: NormalizationKeyResolver = Resolver()  # type: ignore[assignment]
    spec = PhysioChannelSpec("eda", PhysioSignalKind.EDA, 2.0, "uS")
    assert resolver(object(), spec) == NormalizationKey("eda", "participant")  # type: ignore[arg-type]


def test_artifact_policy_defaults_are_explicit_and_frozen() -> None:
    """Expose valid generic detector defaults without a medical claim."""
    policy = PhysioArtifactPolicy()

    assert policy.detect_flatline
    assert policy.detect_outliers
    assert policy.exclude_detected_artifacts
    assert policy.flatline_min_run_length == 5
    with pytest.raises(FrozenInstanceError):
        policy.detect_flatline = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"detect_flatline": 1},
        {"detect_outliers": "yes"},
        {"exclude_detected_artifacts": 0},
        {"flatline_atol": True},
        {"flatline_atol": -1.0},
        {"flatline_atol": float("inf")},
        {"flatline_min_run_length": True},
        {"flatline_min_run_length": 1},
        {"outlier_threshold": False},
        {"outlier_threshold": 0.0},
        {"outlier_threshold": float("nan")},
        {"outlier_mad_epsilon": 0.0},
        {"outlier_mad_epsilon": float("inf")},
    ],
)
def test_artifact_policy_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    """Match existing detector parameter contracts exactly."""
    with pytest.raises((TypeError, ValueError)):
        PhysioArtifactPolicy(**kwargs)  # type: ignore[arg-type]
