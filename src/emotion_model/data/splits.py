"""Participant-independent partitioning with explicit leakage boundaries.

These tools guarantee only that an exact participant identifier does not cross
train, validation, and test partitions. They cannot detect aliases for the same
person, copied or derived sources, or future-session leakage hidden behind
caller-provided identifiers.

Normalization with ``TRAIN`` scope must use only returned training records.
``CALIBRATION`` must use explicitly selected, finite calibration windows and
must not be treated as permission to fit on a test participant's full future
session. Apply label protocols only after partitioning. Label-aware group
balancing is a separate audited concern; this module does not claim to prevent
all real-world data leakage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from emotion_model.data.manifest import (
    MultimodalManifest,
    MultimodalWindowRecord,
)


def _validate_id(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _validate_id_tuple(values: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple.")
    for value in values:
        _validate_id(value, name=f"{name} entry")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate participant IDs.")
    return values


class DatasetPartition(StrEnum):
    """Stable formal dataset partition values."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class ParticipantSplit:
    """Define immutable, pairwise-disjoint participant partitions.

    Validation may be empty, although model selection normally requires an
    independent validation participant set. IDs are preserved exactly and are
    not case-folded or sorted.
    """

    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]
    test_participant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        train = _validate_id_tuple(
            self.train_participant_ids,
            name="train_participant_ids",
        )
        validation = _validate_id_tuple(
            self.validation_participant_ids,
            name="validation_participant_ids",
        )
        test = _validate_id_tuple(
            self.test_participant_ids,
            name="test_participant_ids",
        )
        if not train:
            raise ValueError("train_participant_ids must be non-empty.")
        if not test:
            raise ValueError("test_participant_ids must be non-empty.")
        if set(train) & set(validation):
            raise ValueError("train and validation participant IDs must be disjoint.")
        if set(train) & set(test):
            raise ValueError("train and test participant IDs must be disjoint.")
        if set(validation) & set(test):
            raise ValueError("validation and test participant IDs must be disjoint.")

    def partition_for_participant(
        self,
        participant_id: str,
    ) -> DatasetPartition | None:
        """Return the exact-ID partition, or ``None`` when unassigned."""
        _validate_id(participant_id, name="participant_id")
        if participant_id in self.train_participant_ids:
            return DatasetPartition.TRAIN
        if participant_id in self.validation_participant_ids:
            return DatasetPartition.VALIDATION
        if participant_id in self.test_participant_ids:
            return DatasetPartition.TEST
        return None


def _validate_record_tuple(
    records: object,
    *,
    name: str,
) -> tuple[MultimodalWindowRecord, ...]:
    if not isinstance(records, tuple):
        raise TypeError(f"{name} must be a tuple.")
    for record in records:
        if not isinstance(record, MultimodalWindowRecord):
            raise TypeError(f"{name} must contain MultimodalWindowRecord objects.")
    return records


@dataclass(frozen=True)
class PartitionedManifest:
    """Hold ordered original manifest records partitioned by participant ID."""

    split: ParticipantSplit
    train_records: tuple[MultimodalWindowRecord, ...]
    validation_records: tuple[MultimodalWindowRecord, ...]
    test_records: tuple[MultimodalWindowRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split, ParticipantSplit):
            raise TypeError("split must be a ParticipantSplit.")
        partitions = (
            (
                DatasetPartition.TRAIN,
                _validate_record_tuple(self.train_records, name="train_records"),
            ),
            (
                DatasetPartition.VALIDATION,
                _validate_record_tuple(
                    self.validation_records,
                    name="validation_records",
                ),
            ),
            (
                DatasetPartition.TEST,
                _validate_record_tuple(self.test_records, name="test_records"),
            ),
        )
        seen_sample_ids: set[str] = set()
        for expected_partition, records in partitions:
            for record in records:
                actual = self.split.partition_for_participant(record.participant_id)
                if actual is not expected_partition:
                    raise ValueError(
                        f"record {record.sample_id!r} participant "
                        f"{record.participant_id!r} does not belong to "
                        f"{expected_partition.value}."
                    )
                if record.sample_id in seen_sample_ids:
                    raise ValueError(
                        f"sample {record.sample_id!r} appears in multiple partitions."
                    )
                seen_sample_ids.add(record.sample_id)


def partition_manifest_by_participant(
    manifest: MultimodalManifest,
    split: ParticipantSplit,
    *,
    require_all_participants: bool = True,
) -> PartitionedManifest:
    """Partition records solely by exact participant ID.

    Args:
        manifest: Valid ordered manifest.
        split: Pairwise-disjoint participant assignment.
        require_all_participants: When ``True``, reject manifest participants
            absent from ``split``. When ``False``, exclude those records.

    Returns:
        Immutable partition tuples preserving manifest order and original record
        object identity.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If split participants do not exist in the manifest, or an
            unassigned manifest participant is encountered in strict mode.

    Emotion scores, sessions, timestamps, and record counts never affect the
    assignment.
    """
    if not isinstance(manifest, MultimodalManifest):
        raise TypeError("manifest must be a MultimodalManifest.")
    if not isinstance(split, ParticipantSplit):
        raise TypeError("split must be a ParticipantSplit.")
    if not isinstance(require_all_participants, bool):
        raise TypeError("require_all_participants must be bool.")

    manifest_participants = {record.participant_id for record in manifest.records}
    split_participants = (
        set(split.train_participant_ids)
        | set(split.validation_participant_ids)
        | set(split.test_participant_ids)
    )
    missing_from_manifest = split_participants - manifest_participants
    if missing_from_manifest:
        raise ValueError(
            "split contains participant IDs absent from manifest: "
            f"{sorted(missing_from_manifest)}."
        )

    train_records: list[MultimodalWindowRecord] = []
    validation_records: list[MultimodalWindowRecord] = []
    test_records: list[MultimodalWindowRecord] = []
    unassigned: set[str] = set()
    for record in manifest.records:
        partition = split.partition_for_participant(record.participant_id)
        if partition is DatasetPartition.TRAIN:
            train_records.append(record)
        elif partition is DatasetPartition.VALIDATION:
            validation_records.append(record)
        elif partition is DatasetPartition.TEST:
            test_records.append(record)
        else:
            unassigned.add(record.participant_id)

    if require_all_participants and unassigned:
        raise ValueError(
            "manifest contains participants absent from split: "
            f"{sorted(unassigned)}."
        )
    return PartitionedManifest(
        split=split,
        train_records=tuple(train_records),
        validation_records=tuple(validation_records),
        test_records=tuple(test_records),
    )


def _validated_participant_ids(
    participant_ids: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(participant_ids, (str, bytes)) or not isinstance(
        participant_ids,
        Sequence,
    ):
        raise TypeError("participant_ids must be a non-string Sequence.")
    values = tuple(participant_ids)
    if not values:
        raise ValueError("participant_ids must be non-empty.")
    for value in values:
        _validate_id(value, name="participant_id")
    if len(set(values)) != len(values):
        raise ValueError("participant_ids must be unique.")
    return values


def _validate_fold_arguments(
    participant_count: int,
    *,
    num_folds: int,
    seed: int,
    minimum_folds: int,
) -> None:
    if isinstance(num_folds, bool) or not isinstance(num_folds, int):
        raise TypeError("num_folds must be an integer, not bool.")
    if num_folds < minimum_folds:
        raise ValueError(f"num_folds must be >= {minimum_folds}.")
    if num_folds > participant_count:
        raise ValueError("num_folds must not exceed the participant count.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool.")


def _stable_participant_digest(participant_id: str, *, seed: int) -> bytes:
    payload = f"{seed}\0{participant_id}".encode()
    return hashlib.sha256(payload).digest()


def build_deterministic_participant_folds(
    participant_ids: Sequence[str],
    *,
    num_folds: int,
    seed: int = 0,
) -> tuple[tuple[str, ...], ...]:
    """Build balanced participant folds using stable SHA-256 ordering.

    Args:
        participant_ids: Non-empty, unique, non-string sequence of exact IDs.
        num_folds: Number of folds, from 2 through the participant count.
        seed: Integer incorporated into the stable digest.

    Returns:
        Pairwise-disjoint tuples whose union is the input ID set and whose sizes
        differ by at most one. Results depend on the ID set and seed, not input
        order, Python ``hash()``, labels, record counts, or global random state.
    """
    values = _validated_participant_ids(participant_ids)
    _validate_fold_arguments(
        len(values),
        num_folds=num_folds,
        seed=seed,
        minimum_folds=2,
    )
    ordered = sorted(
        values,
        key=lambda participant_id: (
            _stable_participant_digest(participant_id, seed=seed),
            participant_id,
        ),
    )
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for index, participant_id in enumerate(ordered):
        folds[index % num_folds].append(participant_id)
    return tuple(tuple(fold) for fold in folds)


def build_rotating_participant_splits(
    participant_ids: Sequence[str],
    *,
    num_folds: int,
    seed: int = 0,
) -> tuple[ParticipantSplit, ...]:
    """Build deterministic rotating participant-independent splits.

    Fold ``i`` is test, fold ``(i + 1) % num_folds`` is validation, and all
    remaining folds are concatenated in fold order as training participants.
    Each participant serves exactly once as test and once as validation.
    """
    values = _validated_participant_ids(participant_ids)
    _validate_fold_arguments(
        len(values),
        num_folds=num_folds,
        seed=seed,
        minimum_folds=3,
    )
    folds = build_deterministic_participant_folds(
        values,
        num_folds=num_folds,
        seed=seed,
    )
    splits: list[ParticipantSplit] = []
    for test_index in range(num_folds):
        validation_index = (test_index + 1) % num_folds
        train = tuple(
            participant_id
            for fold_index, fold in enumerate(folds)
            if fold_index not in {test_index, validation_index}
            for participant_id in fold
        )
        splits.append(
            ParticipantSplit(
                train_participant_ids=train,
                validation_participant_ids=folds[validation_index],
                test_participant_ids=folds[test_index],
            )
        )
    return tuple(splits)
