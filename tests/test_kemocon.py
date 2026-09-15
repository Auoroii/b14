"""Offline tests for K-EmoCon preparation and source adapters."""

from __future__ import annotations

import csv
import shutil
import uuid
import wave
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import WavLMConfig, WavLMModel

from emotion_model.common import LabelProtocol
from emotion_model.data import (
    EmotionScores,
    KEmoConPhysioAdapter,
    KEmoConSpeechAdapter,
    MultimodalManifest,
    MultimodalWindowRecord,
    TimedSourceRef,
    TimeInterval,
    build_kemocon_dyad_ratio_split,
    build_kemocon_dyad_splits,
    build_kemocon_label_stratified_dyad_splits,
    build_kemocon_manifest,
    kemocon_channel_specs,
)
from emotion_model.experiments import (
    build_configured_kemocon_split,
    build_kemocon_class_weights,
    build_kemocon_model,
    build_valence_class_participant_sampling_weights,
    load_kemocon_experiment_config,
)


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """Provide a writable workspace-local directory on restricted Windows."""

    path = Path("tmp") / f"kemocon-tests-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_pcm(path: Path, *, seconds: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000 * seconds)


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "K-EmoCon"
    _write_csv(
        root / "metadata" / "subjects.csv",
        ("pid", "initTime", "startTime", "endTime"),
        (
            (1, 0, 100_000, 110_000),
            (2, 0, 100_000, 110_000),
        ),
    )
    _write_pcm(root / "debate_audios" / "p1.p2.wav")
    for participant in ("P1", "P2"):
        _write_pcm(root / "pre_audio" / f"{participant}.wav")
        _write_csv(
            root
            / "emotion_annotations"
            / "self_annotations"
            / f"{participant}.self.csv",
            ("seconds", "arousal", "valence"),
            (
                (5, 1, 5),
                (10, 3, 2),
                (15, 4, 4),
            ),
        )
    rows = (
        (100_000.0, 1, 0.0, "A00001"),
        (101_000.0, 1, 1.0, "A00001"),
        (104_000.0, 1, 2.0, "A00001"),
        (105_000.0, 1, 3.0, "A00001"),
    )
    for file_name in ("E4_BVP.csv", "E4_EDA.csv", "E4_TEMP.csv"):
        _write_csv(
            root / "e4_data" / "1" / file_name,
            ("timestamp", "pid", "value", "device_serial"),
            rows,
        )
    return root


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            "availability_policy: unsupported_policy",
            "availability_policy must be 'source_presence'",
        ),
        (
            "availability_policy: source_presence\n    minimum_active_seconds: 0.2",
            "unknown fields.*minimum_active_seconds",
        ),
    ],
)
def test_v4_2_rejects_removed_speech_availability_options(
    workspace_tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    """Accept only source-presence availability in the V4.2 config parser."""

    source = Path(
        "configs/kemocon_v4_2_full_window_relation_differential.yaml"
    ).read_text(encoding="utf-8")
    candidate = source.replace(
        "availability_policy: source_presence",
        replacement,
    )
    path = workspace_tmp_path / "invalid-availability.yaml"
    path.write_text(candidate, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_kemocon_experiment_config(path)


def test_manifest_uses_relative_sources_and_drops_audio_tail(
    workspace_tmp_path: Path,
) -> None:
    """Keep two complete windows per participant and portable source IDs."""

    report = build_kemocon_manifest(_data_root(workspace_tmp_path))
    assert report.participant_count == 2
    assert report.session_count == 1
    assert report.record_count == 4
    assert report.skipped_tail_windows == 2
    assert report.skipped_annotation_rows == 0
    first = report.manifest.records[0]
    assert first.sample_id == "P1-self-0000"
    assert first.session_id == "P1-P2"
    assert first.speech_source is not None
    assert first.speech_source.source_id == "pre_audio/P1.wav"
    assert tuple(source.channel_name for source in first.physio_sources) == (
        "bvp",
        "eda",
        "temperature",
    )
    assert all(
        not Path(source.source.source_id).is_absolute()
        for source in first.physio_sources
    )


def test_adapters_load_only_declared_window_and_preserve_mask_semantics(
    workspace_tmp_path: Path,
) -> None:
    """Return speech ``[80000]`` and channel values/masks ``[T]`` on CPU."""

    root = _data_root(workspace_tmp_path)
    report = build_kemocon_manifest(root)
    record = report.manifest.records[0]
    assert record.speech_source is not None
    speech = KEmoConSpeechAdapter(root).load_speech(record.speech_source)
    assert speech.waveform.shape == (80_000,)
    assert speech.sample_rate_hz == 16_000
    assert speech.timeline_start_seconds == 0.0

    sources = {source.channel_name: source for source in record.physio_sources}
    specs = {spec.name: spec for spec in report.manifest.channel_specs}
    adapter = KEmoConPhysioAdapter(root)
    bvp = adapter.load_physio(sources["bvp"], specs["bvp"])
    eda = adapter.load_physio(sources["eda"], specs["eda"])
    assert bvp.values.shape == bvp.valid_mask.shape
    assert bvp.timestamps_seconds.shape == bvp.values.shape
    assert bvp.valid_mask.tolist() == [True, True, True]
    assert eda.valid_mask.tolist() == [False, True, True]
    assert eda.values[0].item() == 0.0


def _split_manifest() -> MultimodalManifest:
    records = tuple(
        MultimodalWindowRecord(
            sample_id=f"P{participant}-sample",
            participant_id=f"P{participant}",
            session_id=f"P{left}-P{left + 1}",
            window=TimeInterval(0.0, 5.0),
            emotion_scores=EmotionScores(1.0, 5.0),
            speech_source=TimedSourceRef(
                f"pre_audio/P{participant}.wav",
                TimeInterval(0.0, 5.0),
            ),
        )
        for left in (1, 3, 5)
        for participant in (left, left + 1)
    )
    return MultimodalManifest(channel_specs=(), records=records)


def _record(
    sample_id: str,
    participant_id: str,
    session_id: str,
    start_seconds: float,
    *,
    scores: EmotionScores,
) -> MultimodalWindowRecord:
    """Build one synthetic labeled window for class-weight tests."""

    return MultimodalWindowRecord(
        sample_id=sample_id,
        participant_id=participant_id,
        session_id=session_id,
        window=TimeInterval(start_seconds, start_seconds + 1.0),
        emotion_scores=scores,
        speech_source=TimedSourceRef(
            f"pre_audio/{participant_id}.wav",
            TimeInterval(start_seconds, start_seconds + 1.0),
        ),
    )


def test_dyad_splits_never_separate_session_participants() -> None:
    """Assign both participants from every debate to one partition."""

    manifest = _split_manifest()
    splits = build_kemocon_dyad_splits(manifest, num_folds=3, seed=7)
    assert len(splits) == 3
    for split in splits:
        assignments = {
            participant: split.partition_for_participant(participant)
            for participant in (
                *split.train_participant_ids,
                *split.validation_participant_ids,
                *split.test_participant_ids,
            )
        }
        for left in (1, 3, 5):
            assert assignments[f"P{left}"] == assignments[f"P{left + 1}"]


def _balanced_label_split_manifest() -> MultimodalManifest:
    """Build six dyads that each expose all ``mid_low`` quadrants."""

    score_pairs = (
        EmotionScores(1.0, 1.0),
        EmotionScores(5.0, 1.0),
        EmotionScores(1.0, 5.0),
        EmotionScores(5.0, 5.0),
    )
    records = tuple(
        _record(
            f"P{participant}-q{quadrant}",
            f"P{participant}",
            f"P{left}-P{left + 1}",
            float(quadrant),
            scores=scores,
        )
        for left in range(1, 13, 2)
        for participant in (left, left + 1)
        for quadrant, scores in enumerate(score_pairs)
    )
    return MultimodalManifest(channel_specs=(), records=records)


def test_label_stratified_dyad_splits_are_complete_and_deterministic() -> None:
    """Keep dyads intact and require all task classes/quadrants in each fold."""

    manifest = _balanced_label_split_manifest()
    splits = build_kemocon_label_stratified_dyad_splits(
        manifest,
        label_protocol=LabelProtocol.MID_LOW,
        num_folds=3,
        seed=2026,
    )
    reordered = MultimodalManifest(
        channel_specs=(),
        records=tuple(reversed(manifest.records)),
    )
    assert splits == build_kemocon_label_stratified_dyad_splits(
        reordered,
        label_protocol=LabelProtocol.MID_LOW,
        num_folds=3,
        seed=2026,
    )
    assert len(splits) == 3
    for split in splits:
        assert len(split.test_participant_ids) == 4
        assert len(split.validation_participant_ids) == 4
        for left in range(1, 13, 2):
            assert split.partition_for_participant(
                f"P{left}"
            ) == split.partition_for_participant(f"P{left + 1}")


def test_label_stratified_dyad_splits_reject_missing_quadrants() -> None:
    """Fail before training when grouped labels cannot cover every fold."""

    with pytest.raises(ValueError, match="both task classes and all four"):
        build_kemocon_label_stratified_dyad_splits(
            _split_manifest(),
            label_protocol=LabelProtocol.MID_LOW,
            num_folds=3,
            seed=2026,
        )


def test_dyad_ratio_split_approximates_target_without_session_leakage() -> None:
    """Approximate 70/15/15 with complete, deterministic debate dyads."""

    records = tuple(
        MultimodalWindowRecord(
            sample_id=f"P{participant}-sample-{window_index}",
            participant_id=f"P{participant}",
            session_id=f"P{left}-P{left + 1}",
            window=TimeInterval(
                float(window_index * 5),
                float((window_index + 1) * 5),
            ),
            emotion_scores=EmotionScores(1.0, 5.0),
            speech_source=TimedSourceRef(
                f"pre_audio/P{participant}.wav",
                TimeInterval(0.0, 100.0),
            ),
        )
        for left in range(1, 21, 2)
        for participant in (left, left + 1)
        for window_index in range((left + 1) // 2)
    )
    manifest = MultimodalManifest(channel_specs=(), records=records)

    split = build_kemocon_dyad_ratio_split(
        manifest,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )
    reordered = MultimodalManifest(
        channel_specs=(),
        records=tuple(reversed(records)),
    )
    assert split == build_kemocon_dyad_ratio_split(
        reordered,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )
    fresh_split = build_kemocon_dyad_ratio_split(
        manifest,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=2026,
    )
    assert fresh_split.test_participant_ids != split.test_participant_ids
    assert tuple(
        map(
            len,
            (
                split.train_participant_ids,
                split.validation_participant_ids,
                split.test_participant_ids,
            ),
        )
    ) == (14, 4, 2)
    partition_counts = tuple(
        sum(
            record.participant_id in participant_ids
            for record in records
        )
        for participant_ids in (
            set(split.train_participant_ids),
            set(split.validation_participant_ids),
            set(split.test_participant_ids),
        )
    )
    total_count = sum(partition_counts)
    assert abs(partition_counts[0] / total_count - 0.70) < 0.10
    assert abs(partition_counts[1] / total_count - 0.15) < 0.10
    assert abs(partition_counts[2] / total_count - 0.15) < 0.10
    for left in range(1, 21, 2):
        assert split.partition_for_participant(
            f"P{left}"
        ) == split.partition_for_participant(f"P{left + 1}")


@pytest.mark.parametrize(
    ("fractions", "exception"),
    [
        ((0.70, 0.15, 0.10), ValueError),
        ((0.70, 0.30, 0.0), ValueError),
        ((0.70, 0.15, True), TypeError),
    ],
)
def test_dyad_ratio_split_rejects_invalid_fractions(
    fractions: tuple[object, object, object],
    exception: type[Exception],
) -> None:
    """Reject non-positive, non-numeric, or non-unit split fractions."""

    with pytest.raises(exception):
        build_kemocon_dyad_ratio_split(
            _split_manifest(),
            train_fraction=fractions[0],  # type: ignore[arg-type]
            validation_fraction=fractions[1],  # type: ignore[arg-type]
            test_fraction=fractions[2],  # type: ignore[arg-type]
        )


def test_participant_balanced_class_weights_equalize_person_mass() -> None:
    """Compute task class mass after giving every participant total weight one."""

    records = tuple(
        [
            _record(
                f"low-{index}",
                "P1",
                "P1-P2",
                float(index),
                scores=EmotionScores(1.0, 1.0),
            )
            for index in range(10)
        ]
        + [
            _record(
                "high-0",
                "P2",
                "P1-P2",
                10.0,
                scores=EmotionScores(5.0, 5.0),
            )
        ]
    )
    weights = build_kemocon_class_weights(
        records,
        protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        device=torch.device("cpu"),
        participant_balanced=True,
    )

    assert torch.allclose(weights.arousal, torch.ones(2))
    assert torch.allclose(weights.valence, torch.ones(2))


def test_valence_sampling_softly_balances_classes_then_people() -> None:
    """Give Valence Low 35% mass and equalize people within each class."""

    specifications = (
        ("low-p1-a", "P1", 1.0),
        ("low-p1-b", "P1", 1.0),
        ("low-p2", "P2", 1.0),
        ("high-p1", "P1", 5.0),
        ("high-p3-a", "P3", 5.0),
        ("high-p3-b", "P3", 5.0),
    )
    records = tuple(
        _record(
            sample_id,
            participant_id,
            "session",
            float(index),
            scores=EmotionScores(1.0, valence),
        )
        for index, (sample_id, participant_id, valence) in enumerate(specifications)
    )

    weights = build_valence_class_participant_sampling_weights(
        records,
        protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        low_class_mass=0.35,
    )

    assert weights.dtype == torch.float64
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(
        weights,
        torch.tensor(
            [0.0875, 0.0875, 0.175, 0.325, 0.1625, 0.1625],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("low_class_mass", [0.0, 0.5, 1.0])
def test_valence_sampling_rejects_non_soft_class_mass(
    low_class_mass: float,
) -> None:
    """Reject absent, exact, or reversed minority balancing."""

    records = (
        _record(
            "low",
            "P1",
            "session",
            0.0,
            scores=EmotionScores(1.0, 1.0),
        ),
        _record(
            "high",
            "P2",
            "session",
            1.0,
            scores=EmotionScores(1.0, 5.0),
        ),
    )

    with pytest.raises(ValueError, match="strictly between 0 and 0.5"):
        build_valence_class_participant_sampling_weights(
            records,
            protocol=LabelProtocol.OFFICIAL_MID_HIGH,
            low_class_mass=low_class_mass,
        )


def test_class_weight_power_softens_inverse_frequency_without_changing_classes() -> None:
    """Use square-root weighting while retaining the original class ordering."""

    records = tuple(
        [
            _record(
                f"low-{index}",
                "P1",
                "P1-P2",
                float(index),
                scores=EmotionScores(1.0, 1.0),
            )
            for index in range(3)
        ]
        + [
            _record(
                "high-0",
                "P2",
                "P1-P2",
                3.0,
                scores=EmotionScores(5.0, 5.0),
            )
        ]
    )
    full = build_kemocon_class_weights(
        records,
        protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        device=torch.device("cpu"),
        class_weight_power=1.0,
    )
    softened = build_kemocon_class_weights(
        records,
        protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        device=torch.device("cpu"),
        class_weight_power=0.5,
    )
    uniform = build_kemocon_class_weights(
        records,
        protocol=LabelProtocol.OFFICIAL_MID_HIGH,
        device=torch.device("cpu"),
        class_weight_power=0.0,
    )

    expected_full = torch.tensor([2.0 / 3.0, 2.0])
    assert torch.allclose(full.arousal, expected_full)
    assert torch.allclose(full.valence, expected_full)
    assert torch.allclose(softened.arousal, expected_full.sqrt())
    assert torch.allclose(softened.valence, expected_full.sqrt())
    assert torch.equal(uniform.arousal, torch.ones(2))
    assert torch.equal(uniform.valence, torch.ones(2))


@pytest.mark.parametrize(
    ("power", "exception"),
    [
        (True, TypeError),
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_class_weight_power_rejects_invalid_values(
    power: object,
    exception: type[Exception],
) -> None:
    """Reject booleans, non-finite values, and exponents outside [0, 1]."""

    records = (
        _record(
            "low",
            "P1",
            "P1-P2",
            0.0,
            scores=EmotionScores(1.0, 1.0),
        ),
        _record(
            "high",
            "P2",
            "P1-P2",
            1.0,
            scores=EmotionScores(5.0, 5.0),
        ),
    )
    with pytest.raises(exception, match="class_weight_power"):
        build_kemocon_class_weights(
            records,
            protocol=LabelProtocol.OFFICIAL_MID_HIGH,
            device=torch.device("cpu"),
            class_weight_power=power,  # type: ignore[arg-type]
        )
