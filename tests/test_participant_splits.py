"""Tests for deterministic participant-independent partition infrastructure."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from emotion_model.data import (
    DatasetPartition,
    EmotionScores,
    MultimodalManifest,
    MultimodalWindowRecord,
    ParticipantSplit,
    PartitionedManifest,
    TimedSourceRef,
    TimeInterval,
    build_deterministic_participant_folds,
    build_rotating_participant_splits,
    partition_manifest_by_participant,
)


def _record(
    sample_id: str,
    participant_id: str,
    session_id: str,
    start: float,
    *,
    scores: EmotionScores | None = None,
) -> MultimodalWindowRecord:
    """Create one speech-only record without external data access."""
    return MultimodalWindowRecord(
        sample_id,
        participant_id,
        session_id,
        TimeInterval(start, start + 1.0),
        EmotionScores(2, 4) if scores is None else scores,
        TimedSourceRef("abstract-source", TimeInterval(0, 20)),
    )


def _manifest() -> MultimodalManifest:
    """Create records that exercise repeated participants, sessions, and windows."""
    return MultimodalManifest(
        (),
        (
            _record("p1-a", "p1", "s1", 0),
            _record("p2-a", "p2", "s1", 1),
            _record("p1-b", "p1", "s2", 2),
            _record("p3-a", "p3", "s1", 3),
            _record("p2-b", "p2", "s1", 4),
            _record("p4-a", "p4", "s9", 5),
        ),
    )


def test_participant_split_lookup_and_empty_validation() -> None:
    """Resolve exact IDs and allow an explicitly empty validation partition."""
    split = ParticipantSplit(("p1",), (), ("p2",))

    assert split.partition_for_participant("p1") is DatasetPartition.TRAIN
    assert split.partition_for_participant("p2") is DatasetPartition.TEST
    assert split.partition_for_participant("P1") is None
    assert split.validation_participant_ids == ()


@pytest.mark.parametrize(
    ("train", "validation", "test", "exception"),
    [
        ((), (), ("p2",), ValueError),
        (("p1",), (), (), ValueError),
        (("p1", "p1"), (), ("p2",), ValueError),
        (("p1",), ("p1",), ("p2",), ValueError),
        (("p1",), ("p2",), ("p2",), ValueError),
        (("",), (), ("p2",), ValueError),
        ((1,), (), ("p2",), TypeError),
        (["p1"], (), ("p2",), TypeError),
    ],
)
def test_participant_split_rejects_invalid_contracts(
    train: object,
    validation: object,
    test: object,
    exception: type[Exception],
) -> None:
    """Reject empty required sets, duplicates, overlaps, bad IDs, and lists."""
    with pytest.raises(exception):
        ParticipantSplit(train, validation, test)  # type: ignore[arg-type]


def test_participant_split_is_frozen_and_preserves_order() -> None:
    """Retain caller order and prohibit reassignment."""
    split = ParticipantSplit(("p2", "p1"), ("p3",), ("p4",))
    assert split.train_participant_ids == ("p2", "p1")
    with pytest.raises(FrozenInstanceError):
        split.test_participant_ids = ("p5",)  # type: ignore[misc]


def test_partition_manifest_uses_only_participant_and_preserves_identity_order() -> None:
    """Keep all sessions/windows for a participant together as original objects."""
    manifest = _manifest()
    split = ParticipantSplit(("p2", "p1"), ("p3",), ("p4",))

    result = partition_manifest_by_participant(manifest, split)

    assert tuple(record.sample_id for record in result.train_records) == (
        "p1-a",
        "p2-a",
        "p1-b",
        "p2-b",
    )
    assert tuple(record.sample_id for record in result.validation_records) == ("p3-a",)
    assert tuple(record.sample_id for record in result.test_records) == ("p4-a",)
    assert result.train_records[0] is manifest.records[0]
    assert result.train_records[1] is manifest.records[1]
    assert {record.participant_id for record in result.train_records} == {"p1", "p2"}
    assert {record.participant_id for record in result.validation_records} == {"p3"}
    assert {record.participant_id for record in result.test_records} == {"p4"}
    all_ids = [
        *(record.sample_id for record in result.train_records),
        *(record.sample_id for record in result.validation_records),
        *(record.sample_id for record in result.test_records),
    ]
    assert len(all_ids) == len(set(all_ids))


def test_partition_is_label_independent() -> None:
    """Changing every raw score leaves participant assignment unchanged."""
    manifest = _manifest()
    changed = MultimodalManifest(
        (),
        tuple(
            replace(record, emotion_scores=EmotionScores(5, 1))
            for record in manifest.records
        ),
    )
    split = ParticipantSplit(("p1", "p2"), ("p3",), ("p4",))

    original_result = partition_manifest_by_participant(manifest, split)
    changed_result = partition_manifest_by_participant(changed, split)

    for partition_name in ("train_records", "validation_records", "test_records"):
        assert [
            record.sample_id for record in getattr(original_result, partition_name)
        ] == [record.sample_id for record in getattr(changed_result, partition_name)]


def test_partition_strict_and_non_strict_unassigned_behavior() -> None:
    """Reject unassigned manifest participants or explicitly exclude them."""
    manifest = _manifest()
    split = ParticipantSplit(("p1",), ("p2",), ("p3",))

    with pytest.raises(ValueError, match="absent from split"):
        partition_manifest_by_participant(manifest, split)
    result = partition_manifest_by_participant(
        manifest,
        split,
        require_all_participants=False,
    )
    included = {
        record.participant_id
        for records in (
            result.train_records,
            result.validation_records,
            result.test_records,
        )
        for record in records
    }
    assert included == {"p1", "p2", "p3"}


def test_partition_rejects_split_participant_absent_from_manifest() -> None:
    """Reject unknown split IDs even when non-strict exclusion is requested."""
    split = ParticipantSplit(("p1",), ("p2",), ("unknown",))
    with pytest.raises(ValueError, match="absent from manifest"):
        partition_manifest_by_participant(
            _manifest(),
            split,
            require_all_participants=False,
        )


def test_partition_inputs_remain_unchanged() -> None:
    """Do not mutate manifest order or participant split tuples."""
    manifest = _manifest()
    split = ParticipantSplit(("p1", "p2"), ("p3",), ("p4",))
    original_records = manifest.records
    original_train = split.train_participant_ids

    partition_manifest_by_participant(manifest, split)

    assert manifest.records is original_records
    assert split.train_participant_ids is original_train


def test_partitioned_manifest_validates_direct_construction() -> None:
    """Reject records placed under a contradictory participant partition."""
    manifest = _manifest()
    split = ParticipantSplit(("p1",), ("p2",), ("p3",))
    with pytest.raises(ValueError, match="does not belong"):
        PartitionedManifest(split, (manifest.records[1],), (), ())
    with pytest.raises(TypeError, match="tuple"):
        PartitionedManifest(split, [], (), ())  # type: ignore[arg-type]


@pytest.mark.parametrize("require_all", [0, "yes", None])
def test_partition_rejects_non_bool_strictness(require_all: object) -> None:
    """Do not interpret truthy values as the strictness flag."""
    split = ParticipantSplit(("p1", "p2"), ("p3",), ("p4",))
    with pytest.raises(TypeError, match="bool"):
        partition_manifest_by_participant(
            _manifest(),
            split,
            require_all_participants=require_all,  # type: ignore[arg-type]
        )


def test_deterministic_folds_are_balanced_complete_and_disjoint() -> None:
    """Assign every participant exactly once with size difference at most one."""
    participants = tuple(f"p{index}" for index in range(11))
    folds = build_deterministic_participant_folds(
        participants,
        num_folds=4,
        seed=17,
    )

    flattened = [participant for fold in folds for participant in fold]
    assert len(folds) == 4
    assert set(flattened) == set(participants)
    assert len(flattened) == len(set(flattened))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


def test_deterministic_folds_ignore_input_order_and_global_random_state() -> None:
    """Use stable hashing without consuming Python's global PRNG."""
    participants = ("a", "b", "c", "d", "e", "f")
    random.seed(1234)
    before = random.getstate()
    first = build_deterministic_participant_folds(
        participants,
        num_folds=3,
        seed=9,
    )
    after = random.getstate()
    reordered = build_deterministic_participant_folds(
        tuple(reversed(participants)),
        num_folds=3,
        seed=9,
    )

    assert first == reordered
    assert before == after
    assert first == build_deterministic_participant_folds(
        participants,
        num_folds=3,
        seed=9,
    )


def test_deterministic_folds_change_for_known_different_seed() -> None:
    """Use the integer seed in stable digest ordering."""
    participants = tuple(f"participant-{index}" for index in range(20))
    assert build_deterministic_participant_folds(
        participants,
        num_folds=4,
        seed=1,
    ) != build_deterministic_participant_folds(
        participants,
        num_folds=4,
        seed=2,
    )


@pytest.mark.parametrize(
    ("participants", "num_folds", "seed", "exception"),
    [
        ((), 2, 0, ValueError),
        (("p1", "p1"), 2, 0, ValueError),
        (("p1", ""), 2, 0, ValueError),
        (("p1", "p2"), 1, 0, ValueError),
        (("p1", "p2"), 3, 0, ValueError),
        (("p1", "p2"), True, 0, TypeError),
        (("p1", "p2"), 2, True, TypeError),
        ("p1", 2, 0, TypeError),
    ],
)
def test_deterministic_folds_reject_invalid_inputs(
    participants: object,
    num_folds: object,
    seed: object,
    exception: type[Exception],
) -> None:
    """Reject invalid ID sequences, fold counts, and boolean seeds."""
    with pytest.raises(exception):
        build_deterministic_participant_folds(
            participants,  # type: ignore[arg-type]
            num_folds=num_folds,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


def test_rotating_splits_cover_test_and_validation_exactly_once() -> None:
    """Rotate test and next-fold validation while keeping training non-empty."""
    participants = tuple(f"p{index}" for index in range(9))
    splits = build_rotating_participant_splits(
        participants,
        num_folds=3,
        seed=4,
    )

    assert len(splits) == 3
    assert sorted(
        participant
        for split in splits
        for participant in split.test_participant_ids
    ) == sorted(participants)
    assert sorted(
        participant
        for split in splits
        for participant in split.validation_participant_ids
    ) == sorted(participants)
    for split in splits:
        train = set(split.train_participant_ids)
        validation = set(split.validation_participant_ids)
        test = set(split.test_participant_ids)
        assert train
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(participants)


def test_rotating_splits_are_order_independent_and_support_lopo_shape() -> None:
    """Produce stable splits and one-member test/validation folds when N equals K."""
    participants = ("p1", "p2", "p3", "p4")
    first = build_rotating_participant_splits(
        participants,
        num_folds=4,
        seed=3,
    )
    second = build_rotating_participant_splits(
        tuple(reversed(participants)),
        num_folds=4,
        seed=3,
    )

    assert first == second
    assert all(len(split.test_participant_ids) == 1 for split in first)
    assert all(len(split.validation_participant_ids) == 1 for split in first)
    assert all(len(split.train_participant_ids) == 2 for split in first)


def test_rotating_split_can_partition_matching_manifest() -> None:
    """Feed a generated participant split directly to manifest partitioning."""
    manifest = _manifest()
    split = build_rotating_participant_splits(
        ("p1", "p2", "p3", "p4"),
        num_folds=4,
        seed=0,
    )[0]
    partitioned = partition_manifest_by_participant(manifest, split)
    assert sum(
        len(records)
        for records in (
            partitioned.train_records,
            partitioned.validation_records,
            partitioned.test_records,
        )
    ) == len(manifest.records)


@pytest.mark.parametrize("num_folds", [1, 2, 5])
def test_rotating_splits_require_legal_three_or_more_folds(num_folds: int) -> None:
    """Require at least three folds and no more folds than participants."""
    with pytest.raises(ValueError):
        build_rotating_participant_splits(
            ("p1", "p2", "p3", "p4"),
            num_folds=num_folds,
        )


def test_data_modules_do_not_contain_out_of_scope_implementations() -> None:
    """Keep stage 11A limited to contracts, alignment, and participant splits."""
    from emotion_model.data import alignment, manifest, splits

    module_paths = (manifest.__file__, alignment.__file__, splits.__file__)
    assert all(path is not None for path in module_paths)
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in module_paths
        if path is not None
    )
    forbidden = (
        "class Dataset(",
        "IterableDataset",
        "DataLoader",
        "__getitem__",
        "def collate",
        "torch.nn",
        "optimizer",
    )
    assert not any(term in source for term in forbidden)
