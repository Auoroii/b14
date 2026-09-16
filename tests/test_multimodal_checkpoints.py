"""Tests for strict atomic multimodal training checkpoints."""

from __future__ import annotations

import copy
import math
import os
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

import emotion_model.training.checkpoints as checkpoint_module
from emotion_model.training import (
    ClassificationLossKind,
    LoadedMultimodalCheckpoint,
    MultimodalCheckpointError,
    MultimodalLossWeights,
    MultimodalObjectiveConfig,
    MultimodalTrainingObjective,
    MultimodalTrainingState,
    advance_training_state,
    load_multimodal_checkpoint,
    run_multimodal_training_epoch,
    save_multimodal_checkpoint,
    train_multimodal_batch,
)
from tests.test_multimodal_classifier import _batch, _model


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide an isolated project-local directory despite host temp ACLs."""
    root = Path.cwd() / "tmp"
    root.mkdir(exist_ok=True)
    path = root / f"pytest-checkpoint-{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path)


def _objective(
    *,
    loss_kind: ClassificationLossKind = (
        ClassificationLossKind.WEIGHTED_CROSS_ENTROPY
    ),
    weights: MultimodalLossWeights | None = None,
    focal_gamma: float = 2.0,
    speech_aux_min_activity_ratio: float | None = None,
) -> MultimodalTrainingObjective:
    return MultimodalTrainingObjective(
        MultimodalObjectiveConfig(
            loss_kind=loss_kind,
            weights=weights or MultimodalLossWeights(),
            focal_gamma=focal_gamma,
            speech_aux_min_activity_ratio=speech_aux_min_activity_ratio,
        )
    )


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-3,
        betas=(0.8, 0.95),
        weight_decay=0.01,
    )


def _populate_optimizer(
    model: Any,
    objective: MultimodalTrainingObjective,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    result = train_multimodal_batch(model, objective, _batch(), optimizer)
    assert result.optimizer_step_performed
    assert optimizer.state


def _load_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert type(payload) is dict
    return payload


def _write_payload(path: Path, payload: object) -> None:
    torch.save(payload, path)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_checkpoint_basic_tree(value: object) -> None:
    """Independently require the documented weights-only payload value types."""
    if isinstance(value, Tensor) or value is None:
        return
    if type(value) in (str, int, float, bool):
        return
    if type(value) in (list, tuple):
        for item in value:
            _assert_checkpoint_basic_tree(item)
        return
    if type(value) is dict:
        assert all(type(key) in (str, int) for key in value)
        for item in value.values():
            _assert_checkpoint_basic_tree(item)
        return
    raise AssertionError(f"unsupported checkpoint value: {type(value).__name__}")


def test_checkpoint_round_trip_restores_model_optimizer_progress_metadata(
    tmp_path: Path,
) -> None:
    """Restore every public state category while preserving target mode."""
    torch.manual_seed(101)
    model = _model(dropout=0.2)
    objective = _objective()
    optimizer = _optimizer(model)
    _populate_optimizer(model, objective, optimizer)
    model.eval()
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    state = MultimodalTrainingState(3, 17, 0.75)
    metadata: dict[str, object] = {
        "run_id": "tiny-run",
        "fingerprints": {"manifest": "abc", "split": "def"},
        "folds": [1, 2],
        "tuple": ("x", True, None, 1.25),
    }
    path = tmp_path / "checkpoint.pt"
    save_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        training_state=state,
        extra_metadata=metadata,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    optimizer.param_groups[0]["lr"] = 0.5
    result = load_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
    )
    assert isinstance(result, LoadedMultimodalCheckpoint)
    assert result.training_state == state
    assert dict(result.extra_metadata) == metadata
    assert result.rng_state_restored
    assert not model.training
    _assert_nested_equal(model.state_dict(), saved_model)
    _assert_nested_equal(optimizer.state_dict(), saved_optimizer)
    with pytest.raises(FrozenInstanceError):
        result.rng_state_restored = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.extra_metadata["new"] = 1  # type: ignore[index]


def test_checkpoint_payload_has_exact_basic_schema_and_no_path(
    tmp_path: Path,
) -> None:
    """Persist only the fixed schema and caller-selected basic metadata."""
    model = _model()
    objective = _objective()
    optimizer = _optimizer(model)
    path = tmp_path / "strict.pt"
    save_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        training_state=MultimodalTrainingState(),
    )
    payload = _load_payload(path)
    assert set(payload) == checkpoint_module._TOP_LEVEL_FIELDS
    assert payload["checkpoint_schema_version"] == 1
    assert type(payload["model_state"]) is dict
    assert type(payload["optimizer_state"]) is dict
    assert type(payload["objective_config"]) is dict
    assert payload["objective_config"]["speech_aux_min_activity_ratio"] is None
    assert type(payload["training_state"]) is dict
    assert payload["extra_metadata"] == {}
    assert str(path) not in repr(payload)
    _assert_checkpoint_basic_tree(payload)


@dataclass(frozen=True)
class _MetadataDataclass:
    value: int


class _MetadataEnum(Enum):
    VALUE = "value"


@pytest.mark.parametrize(
    "value",
    [
        {1: "bad-key"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": torch.tensor(1.0)},
        {"value": Path("relative")},
        {"value": _MetadataEnum.VALUE},
        {"value": _MetadataDataclass(1)},
        {"value": math.sin},
        {"value": {1: "nested-bad-key"}},
    ],
)
def test_checkpoint_rejects_forbidden_metadata(
    tmp_path: Path,
    value: object,
) -> None:
    """Reject non-JSON, path, tensor, enum, dataclass, and callable metadata."""
    model = _model()
    with pytest.raises((TypeError, ValueError)):
        save_multimodal_checkpoint(
            tmp_path / "metadata.pt",
            model=model,
            optimizer=_optimizer(model),
            objective=_objective(),
            training_state=MultimodalTrainingState(),
            extra_metadata=value,  # type: ignore[arg-type]
        )


def test_metadata_is_defensively_copied_on_save_and_load(tmp_path: Path) -> None:
    """Prevent caller mutations from changing persisted or later loaded data."""
    model = _model()
    optimizer = _optimizer(model)
    objective = _objective()
    nested = [1, {"key": ["original"]}]
    metadata: dict[str, object] = {"nested": nested}
    path = tmp_path / "metadata-copy.pt"
    save_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        training_state=MultimodalTrainingState(),
        extra_metadata=metadata,
    )
    nested.append(2)
    first = load_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        restore_rng_state=False,
    )
    first_nested = first.extra_metadata["nested"]
    assert isinstance(first_nested, list)
    first_nested.append("caller-change")
    second = load_multimodal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        restore_rng_state=False,
    )
    assert second.extra_metadata["nested"] == [1, {"key": ["original"]}]


def test_save_validates_path_parent_and_directory(tmp_path: Path) -> None:
    """Reject empty paths, missing parents, and directory destinations."""
    model = _model()
    arguments = {
        "model": model,
        "optimizer": _optimizer(model),
        "objective": _objective(),
        "training_state": MultimodalTrainingState(),
    }
    with pytest.raises(ValueError, match="empty"):
        save_multimodal_checkpoint("", **arguments)
    with pytest.raises(FileNotFoundError, match="parent"):
        save_multimodal_checkpoint(
            tmp_path / "missing" / "file.pt",
            **arguments,
        )
    with pytest.raises(IsADirectoryError):
        save_multimodal_checkpoint(tmp_path, **arguments)


def test_save_uses_same_directory_unique_temp_and_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe a same-directory temporary file followed by os.replace."""
    model = _model()
    destination = tmp_path / "atomic.pt"
    calls: list[tuple[Path, Path]] = []
    original_replace = checkpoint_module.os.replace

    def tracked_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        calls.append((Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(checkpoint_module.os, "replace", tracked_replace)
    save_multimodal_checkpoint(
        destination,
        model=model,
        optimizer=_optimizer(model),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    assert len(calls) == 1
    assert calls[0][0].parent == destination.parent
    assert calls[0][1] == destination
    assert destination.exists()
    assert not calls[0][0].exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_atomic_replace_failure_cleans_temp_and_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat final replacement failure as atomic-save failure with cleanup."""
    model = _model()
    destination = tmp_path / "replace-failure.pt"
    destination.write_bytes(b"previous")

    def failing_replace(
        _source: os.PathLike[str],
        _target: os.PathLike[str],
    ) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)
    with pytest.raises(MultimodalCheckpointError) as captured:
        save_multimodal_checkpoint(
            destination,
            model=model,
            optimizer=_optimizer(model),
            objective=_objective(),
            training_state=MultimodalTrainingState(),
        )
    assert isinstance(captured.value.__cause__, OSError)
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_failed_save_cleans_temp_and_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave a previous checkpoint byte-for-byte intact on serialization error."""
    model = _model()
    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"old-checkpoint")

    def failing_save(_payload: object, stream: object) -> None:
        stream.write(b"partial")  # type: ignore[attr-defined]
        raise OSError("injected save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", failing_save)
    with pytest.raises(MultimodalCheckpointError) as captured:
        save_multimodal_checkpoint(
            destination,
            model=model,
            optimizer=_optimizer(model),
            objective=_objective(),
            training_state=MultimodalTrainingState(),
        )
    assert isinstance(captured.value.__cause__, OSError)
    assert destination.read_bytes() == b"old-checkpoint"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_save_preserves_rng_mode_and_does_not_call_forward(
    tmp_path: Path,
) -> None:
    """Checkpoint serialization has no model execution or runtime-mode effect."""
    model = _model()
    model.train()
    forward_calls = 0

    def hook(_module: object, _inputs: object, _output: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_hook(hook)
    rng_before = torch.get_rng_state().clone()
    save_multimodal_checkpoint(
        tmp_path / "passive.pt",
        model=model,
        optimizer=_optimizer(model),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    handle.remove()
    assert forward_calls == 0
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_save_preserves_model_optimizer_objective_and_progress_state(
    tmp_path: Path,
) -> None:
    """Prove serialization itself does not mutate any caller-owned state."""
    model = _model()
    objective = _objective()
    optimizer = _optimizer(model)
    _populate_optimizer(model, objective, optimizer)
    progress = MultimodalTrainingState(2, 1, 0.9)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    objective_config_before = objective.config
    save_multimodal_checkpoint(
        tmp_path / "unchanged.pt",
        model=model,
        optimizer=optimizer,
        objective=objective,
        training_state=progress,
    )
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert objective.config is objective_config_before
    assert progress == MultimodalTrainingState(2, 1, 0.9)


def test_optimizer_fingerprint_rejects_foreign_or_duplicate_parameters(
    tmp_path: Path,
) -> None:
    """Persist only unique optimizer parameters owned by the target model."""
    model = _model()
    objective = _objective()
    foreign = torch.nn.Parameter(torch.ones(1))
    foreign_optimizer = torch.optim.AdamW([foreign])
    with pytest.raises(ValueError, match="belong"):
        save_multimodal_checkpoint(
            tmp_path / "foreign.pt",
            model=model,
            optimizer=foreign_optimizer,
            objective=objective,
            training_state=MultimodalTrainingState(),
        )
    optimizer = _optimizer(model)
    first = optimizer.param_groups[0]["params"][0]
    optimizer.param_groups[0]["params"].append(first)
    with pytest.raises(ValueError, match="more than once"):
        save_multimodal_checkpoint(
            tmp_path / "duplicate.pt",
            model=model,
            optimizer=optimizer,
            objective=objective,
            training_state=MultimodalTrainingState(),
        )


def test_optimizer_parameter_subset_round_trip_is_supported(tmp_path: Path) -> None:
    """Allow an optimizer to own an ordered proper subset of model parameters."""
    source = _model()
    target = _model()
    source_named = dict(source.named_parameters())
    target_named = dict(target.named_parameters())
    selected_names = tuple(source_named)[:3]
    source_optimizer = torch.optim.AdamW(
        [source_named[name] for name in selected_names],
        lr=0.02,
    )
    target_optimizer = torch.optim.AdamW(
        [target_named[name] for name in selected_names],
        lr=0.5,
    )
    path = tmp_path / "subset.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    load_multimodal_checkpoint(
        path,
        model=target,
        optimizer=target_optimizer,
        objective=_objective(),
    )
    assert target_optimizer.param_groups[0]["lr"] == 0.02
    payload = _load_payload(path)
    fingerprint = payload["optimizer_parameter_groups"]
    assert fingerprint == (selected_names,)


def test_actual_optimizer_grouping_mismatch_fails_before_model_copy(
    tmp_path: Path,
) -> None:
    """Reject a differently grouped target even when it owns the same parameters."""
    source = _model()
    source_parameters = list(source.parameters())
    midpoint = len(source_parameters) // 2
    source_optimizer = torch.optim.AdamW(
        [
            {"params": source_parameters[:midpoint]},
            {"params": source_parameters[midpoint:]},
        ]
    )
    path = tmp_path / "groups.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    target = _model()
    target_before = copy.deepcopy(target.state_dict())
    target_parameters = list(target.parameters())
    target_optimizer = torch.optim.AdamW(
        [
            {"params": target_parameters[midpoint:]},
            {"params": target_parameters[:midpoint]},
        ]
    )
    with pytest.raises(MultimodalCheckpointError, match="parameter group"):
        load_multimodal_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            objective=_objective(),
        )
    _assert_nested_equal(target.state_dict(), target_before)


def test_save_rejects_unsupported_optimizer_payload_before_file_creation(
    tmp_path: Path,
) -> None:
    """Prevent arbitrary objects from reaching torch.save checkpoint payloads."""
    model = _model()
    optimizer = _optimizer(model)
    optimizer.param_groups[0]["unsafe_path"] = Path("not-serializable")
    destination = tmp_path / "unsafe-optimizer.pt"
    with pytest.raises(ValueError, match="unsupported"):
        save_multimodal_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            objective=_objective(),
            training_state=MultimodalTrainingState(),
        )
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def _mutate_payload(payload: dict[str, object], case: str) -> object:
    if case == "non_dict":
        return []
    if case == "non_string_top_key":
        payload[1] = payload.pop("training_state")  # type: ignore[index]
    elif case == "missing_top":
        payload.pop("training_state")
    elif case == "unknown_top":
        payload["unknown"] = 1
    elif case == "schema":
        payload["checkpoint_schema_version"] = 2
    elif case == "model_type":
        payload["model_type"] = "wrong.Model"
    elif case == "optimizer_type":
        payload["optimizer_type"] = "wrong.Optimizer"
    elif case == "objective_type":
        payload["objective_type"] = "wrong.Objective"
    elif case == "objective_config":
        config = payload["objective_config"]
        assert isinstance(config, dict)
        config["speech_auxiliary"] = 0.1
    elif case == "optimizer_group_count":
        payload["optimizer_parameter_groups"] = ()
    elif case == "optimizer_parameter_name":
        groups = payload["optimizer_parameter_groups"]
        assert isinstance(groups, tuple)
        first = list(groups[0])
        first[0] = "wrong.parameter"
        payload["optimizer_parameter_groups"] = (tuple(first), *groups[1:])
    elif case == "optimizer_parameter_order":
        groups = payload["optimizer_parameter_groups"]
        assert isinstance(groups, tuple)
        first = list(groups[0])
        first[0], first[1] = first[1], first[0]
        payload["optimizer_parameter_groups"] = (tuple(first), *groups[1:])
    elif case in {
        "model_missing_key",
        "model_unknown_key",
        "model_shape",
        "model_dtype",
        "model_nonfinite",
        "model_extra_state",
    }:
        state = payload["model_state"]
        assert isinstance(state, dict)
        tensor_key = next(
            key
            for key, value in state.items()
            if isinstance(value, Tensor)
            and value.is_floating_point()
            and value.numel() > 1
        )
        if case == "model_missing_key":
            state.pop(tensor_key)
        elif case == "model_unknown_key":
            state["unknown.weight"] = torch.zeros(1)
        elif case == "model_shape":
            state[tensor_key] = state[tensor_key].reshape(-1)[:1]
        elif case == "model_dtype":
            state[tensor_key] = state[tensor_key].to(torch.float64)
        elif case == "model_nonfinite":
            changed = state[tensor_key].clone()
            changed.reshape(-1)[0] = float("nan")
            state[tensor_key] = changed
        else:
            extra_key = next(key for key in state if key.endswith("_extra_state"))
            extra = copy.deepcopy(state[extra_key])
            assert isinstance(extra, dict)
            first_key = next(iter(extra))
            extra[first_key] = "incompatible"
            state[extra_key] = extra
    elif case == "optimizer_nonfinite":
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        groups = optimizer_state["param_groups"]
        assert isinstance(groups, list)
        parameter_id = groups[0]["params"][0]
        optimizer_state["state"][parameter_id] = {
            "invalid": torch.tensor(float("inf"))
        }
    elif case == "optimizer_state_groups":
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        optimizer_state["param_groups"] = []
    elif case == "optimizer_state_parameter_count":
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        groups = optimizer_state["param_groups"]
        assert isinstance(groups, list)
        groups[0]["params"].pop()
    elif case == "optimizer_state_option":
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        groups = optimizer_state["param_groups"]
        assert isinstance(groups, list)
        groups[0]["unknown_option"] = True
    elif case == "optimizer_state_parameter_order":
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        groups = optimizer_state["param_groups"]
        assert isinstance(groups, list)
        parameters = groups[0]["params"]
        parameters[0], parameters[1] = parameters[1], parameters[0]
    elif case == "rng":
        payload["torch_cpu_rng_state"] = torch.zeros(2, dtype=torch.int64)
    elif case == "rng_length":
        payload["torch_cpu_rng_state"] = torch.zeros(2, dtype=torch.uint8)
    elif case == "training_state":
        state = payload["training_state"]
        assert isinstance(state, dict)
        state["completed_epochs"] = -1
    elif case == "metadata":
        payload["extra_metadata"] = {"bad": torch.tensor(1)}
    else:
        raise AssertionError(f"unknown case {case}")
    return payload


@pytest.mark.parametrize(
    "case",
    [
        "non_dict",
        "non_string_top_key",
        "missing_top",
        "unknown_top",
        "schema",
        "model_type",
        "optimizer_type",
        "objective_type",
        "objective_config",
        "optimizer_group_count",
        "optimizer_parameter_name",
        "optimizer_parameter_order",
        "model_missing_key",
        "model_unknown_key",
        "model_shape",
        "model_dtype",
        "model_nonfinite",
        "model_extra_state",
        "optimizer_nonfinite",
        "optimizer_state_groups",
        "optimizer_state_parameter_count",
        "optimizer_state_option",
        "optimizer_state_parameter_order",
        "rng",
        "rng_length",
        "training_state",
        "metadata",
    ],
)
def test_preflight_rejects_corruption_without_mutating_targets(
    tmp_path: Path,
    case: str,
) -> None:
    """Reject every known incompatibility before loading model or optimizer."""
    source_model = _model()
    source_optimizer = _optimizer(source_model)
    source_objective = _objective()
    path = tmp_path / f"{case}.pt"
    save_multimodal_checkpoint(
        path,
        model=source_model,
        optimizer=source_optimizer,
        objective=source_objective,
        training_state=MultimodalTrainingState(1, 2, 0.5),
        extra_metadata={"valid": True},
    )
    payload = _mutate_payload(_load_payload(path), case)
    _write_payload(path, payload)

    target_model = _model()
    target_optimizer = _optimizer(target_model)
    model_before = copy.deepcopy(target_model.state_dict())
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(MultimodalCheckpointError):
        load_multimodal_checkpoint(
            path,
            model=target_model,
            optimizer=target_optimizer,
            objective=source_objective,
        )
    _assert_nested_equal(target_model.state_dict(), model_before)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "extra_state_key",
    [
        "batch_scheduler.physiology_classifier._extra_state",
        "multimodal_fusion._extra_state",
    ],
)
def test_nested_extra_state_mismatch_fails_before_any_state_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_state_key: str,
) -> None:
    """Exercise nested classifier and fusion fingerprints before state copying."""
    source = _model()
    path = tmp_path / "nested-extra.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=_optimizer(source),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    payload = _load_payload(path)
    model_state = payload["model_state"]
    assert isinstance(model_state, dict)
    extra_state = copy.deepcopy(model_state[extra_state_key])
    assert isinstance(extra_state, dict)
    first_key = next(iter(extra_state))
    extra_state[first_key] = "incompatible-nested-config"
    model_state[extra_state_key] = extra_state
    _write_payload(path, payload)

    target = _model()
    optimizer = _optimizer(target)
    model_calls = 0
    optimizer_calls = 0

    def forbidden_model_load(*_args: object, **_kwargs: object) -> None:
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("model.load_state_dict ran during preflight")

    def forbidden_optimizer_load(*_args: object, **_kwargs: object) -> None:
        nonlocal optimizer_calls
        optimizer_calls += 1
        raise AssertionError("optimizer.load_state_dict ran during preflight")

    monkeypatch.setattr(target, "load_state_dict", forbidden_model_load)
    monkeypatch.setattr(
        optimizer,
        "load_state_dict",
        forbidden_optimizer_load,
    )
    with pytest.raises(MultimodalCheckpointError, match="extra-state"):
        load_multimodal_checkpoint(
            path,
            model=target,
            optimizer=optimizer,
            objective=_objective(),
        )
    assert model_calls == 0
    assert optimizer_calls == 0


def test_load_validates_path_and_wraps_deserialization_error(
    tmp_path: Path,
) -> None:
    """Report absent, directory, and unreadable checkpoint sources clearly."""
    model = _model()
    arguments = {
        "model": model,
        "optimizer": _optimizer(model),
        "objective": _objective(),
    }
    with pytest.raises(FileNotFoundError):
        load_multimodal_checkpoint(tmp_path / "missing.pt", **arguments)
    with pytest.raises(IsADirectoryError):
        load_multimodal_checkpoint(tmp_path, **arguments)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(MultimodalCheckpointError) as captured:
        load_multimodal_checkpoint(bad, **arguments)
    assert captured.value.__cause__ is not None


def test_load_rejects_objective_semantics_before_state_copy(
    tmp_path: Path,
) -> None:
    """Do not overwrite the caller's same-shape but semantically different objective."""
    source_model = _model()
    path = tmp_path / "objective.pt"
    save_multimodal_checkpoint(
        path,
        model=source_model,
        optimizer=_optimizer(source_model),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    target_model = _model()
    target_before = copy.deepcopy(target_model.state_dict())
    target_objective = _objective(
        loss_kind=ClassificationLossKind.FOCAL,
        focal_gamma=1.0,
    )
    with pytest.raises(MultimodalCheckpointError, match="objective_config"):
        load_multimodal_checkpoint(
            path,
            model=target_model,
            optimizer=_optimizer(target_model),
            objective=target_objective,
        )
    assert target_objective.config.loss_kind is ClassificationLossKind.FOCAL
    _assert_nested_equal(target_model.state_dict(), target_before)


def test_load_rejects_speech_auxiliary_activity_threshold_mismatch(
    tmp_path: Path,
) -> None:
    """Fingerprint the activity-filtered speech supervision semantics."""

    source_model = _model()
    path = tmp_path / "speech-activity-objective.pt"
    save_multimodal_checkpoint(
        path,
        model=source_model,
        optimizer=_optimizer(source_model),
        objective=_objective(speech_aux_min_activity_ratio=0.0),
        training_state=MultimodalTrainingState(),
    )
    payload = _load_payload(path)
    assert payload["objective_config"]["speech_aux_min_activity_ratio"] == 0.0

    target_model = _model()
    with pytest.raises(MultimodalCheckpointError, match="objective_config"):
        load_multimodal_checkpoint(
            path,
            model=target_model,
            optimizer=_optimizer(target_model),
            objective=_objective(speech_aux_min_activity_ratio=None),
        )


def test_valid_restore_calls_each_loader_once_then_restores_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independently verify strict model, optimizer, then RNG restoration order."""
    source = _model()
    source_optimizer = _optimizer(source)
    path = tmp_path / "restore-order.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    target = _model()
    target.eval()
    optimizer = _optimizer(target)
    original_model_load = target.load_state_dict
    original_optimizer_load = optimizer.load_state_dict
    original_set_rng = torch.set_rng_state
    events: list[str] = []

    def tracked_model_load(
        state: dict[str, object],
        *,
        strict: bool = True,
    ) -> object:
        events.append("model")
        assert strict
        return original_model_load(state, strict=strict)

    def tracked_optimizer_load(state: dict[str, object]) -> None:
        events.append("optimizer")
        original_optimizer_load(state)

    def tracked_set_rng(state: Tensor) -> None:
        events.append("rng")
        original_set_rng(state)

    monkeypatch.setattr(target, "load_state_dict", tracked_model_load)
    monkeypatch.setattr(optimizer, "load_state_dict", tracked_optimizer_load)
    monkeypatch.setattr(torch, "set_rng_state", tracked_set_rng)
    load_multimodal_checkpoint(
        path,
        model=target,
        optimizer=optimizer,
        objective=_objective(),
    )
    assert events == ["model", "optimizer", "rng"]
    assert not target.training


def test_restore_rng_false_preserves_current_rng_and_model_mode(
    tmp_path: Path,
) -> None:
    """Allow state restoration without changing caller RNG or train/eval mode."""
    source = _model()
    path = tmp_path / "no-rng.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=_optimizer(source),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    target = _model()
    target.train()
    optimizer = _optimizer(target)
    torch.manual_seed(909)
    before = torch.get_rng_state().clone()
    loaded = load_multimodal_checkpoint(
        path,
        model=target,
        optimizer=optimizer,
        objective=_objective(),
        restore_rng_state=False,
    )
    assert not loaded.rng_state_restored
    assert torch.equal(torch.get_rng_state(), before)
    assert target.training
    with pytest.raises(TypeError):
        load_multimodal_checkpoint(
            path,
            model=target,
            optimizer=optimizer,
            objective=_objective(),
            restore_rng_state=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_round_trip_supports_core_floating_dtypes(
    tmp_path: Path,
    dtype: torch.dtype,
) -> None:
    """Preserve model tensor dtype exactly for CPU float32 and float64."""
    source = _model(dtype=dtype)
    target = _model(dtype=dtype)
    path = tmp_path / f"{dtype}.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=_optimizer(source),
        objective=_objective(),
        training_state=MultimodalTrainingState(),
    )
    load_multimodal_checkpoint(
        path,
        model=target,
        optimizer=_optimizer(target),
        objective=_objective(),
    )
    _assert_nested_equal(source.state_dict(), target.state_dict())


def test_deterministic_resume_restores_dropout_and_adamw_trajectory(
    tmp_path: Path,
) -> None:
    """Match next-epoch loss, parameters, moments, and progress after resume."""
    torch.manual_seed(2027)
    source = _model(dropout=0.35)
    objective = _objective()
    source_optimizer = _optimizer(source)
    first_epoch = run_multimodal_training_epoch(
        source,
        objective,
        [_batch()],
        source_optimizer,
    )
    saved_progress = advance_training_state(
        MultimodalTrainingState(),
        first_epoch,
    )
    assert source_optimizer.state
    path = tmp_path / "resume.pt"
    save_multimodal_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        objective=objective,
        training_state=saved_progress,
    )
    reference_epoch = run_multimodal_training_epoch(
        source,
        objective,
        [_batch()],
        source_optimizer,
    )
    reference_progress = advance_training_state(
        saved_progress,
        reference_epoch,
    )
    reference_model_state = copy.deepcopy(source.state_dict())
    reference_optimizer_state = copy.deepcopy(source_optimizer.state_dict())

    resumed = _model(dropout=0.35)
    resumed_optimizer = _optimizer(resumed)
    resumed.train()
    loaded = load_multimodal_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        objective=objective,
        restore_rng_state=True,
    )
    assert resumed.training
    assert loaded.training_state == saved_progress
    resumed_epoch = run_multimodal_training_epoch(
        resumed,
        objective,
        [_batch()],
        resumed_optimizer,
    )
    resumed_progress = advance_training_state(
        loaded.training_state,
        resumed_epoch,
    )
    assert resumed_epoch.loss_averages.total_loss == pytest.approx(
        reference_epoch.loss_averages.total_loss,
        rel=0.0,
        abs=1.0e-7,
    )
    assert resumed_progress == reference_progress
    _assert_nested_equal(resumed.state_dict(), reference_model_state)
    _assert_nested_equal(
        resumed_optimizer.state_dict(),
        reference_optimizer_state,
    )


def test_checkpoint_source_scope_excludes_future_runtime_features() -> None:
    """Keep checkpoint persistence local, CPU-only, and scheduler-free."""
    assert checkpoint_module.__file__ is not None
    source = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    forbidden_calls = (
        "torch.cuda.get_rng_state",
        "random.getstate",
        "numpy.random",
        "DataLoader(",
        "autocast(",
        "GradScaler(",
        "lr_scheduler.",
    )
    assert not any(token in source for token in forbidden_calls)
