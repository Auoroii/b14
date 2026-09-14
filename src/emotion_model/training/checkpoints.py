"""Strict, atomic checkpoints for multimodal training state."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import torch
from torch import Tensor, nn

from emotion_model.multimodal import MultimodalEmotionClassifier
from emotion_model.training.epochs import MultimodalTrainingState
from emotion_model.training.objectives import MultimodalTrainingObjective

_CHECKPOINT_SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(
    {
        "checkpoint_schema_version",
        "model_type",
        "model_state",
        "optimizer_type",
        "optimizer_state",
        "optimizer_parameter_groups",
        "objective_type",
        "objective_config",
        "training_state",
        "torch_cpu_rng_state",
        "extra_metadata",
    }
)
_OBJECTIVE_FIELDS = frozenset(
    {
        "loss_kind",
        "fused",
        "speech_auxiliary",
        "physiology_auxiliary",
        "fused_quadrant",
        "speech_quadrant",
        "physiology_quadrant",
        "focal_gamma",
    }
)
_TRAINING_STATE_FIELDS = frozenset(
    {
        "completed_epochs",
        "global_optimizer_steps",
        "best_validation_loss",
    }
)

_MetadataScalar: TypeAlias = str | int | float | bool | None
_MetadataValue: TypeAlias = (
    _MetadataScalar | list["_MetadataValue"] | tuple["_MetadataValue", ...]
    | dict[str, "_MetadataValue"]
)


class MultimodalCheckpointError(RuntimeError):
    """Raised when checkpoint I/O, schema validation, or restoration fails."""


@dataclass(frozen=True)
class LoadedMultimodalCheckpoint:
    """Validated training progress restored into caller-owned objects.

    Attributes:
        training_state: Immutable scalar epoch and optimizer-step progress.
        extra_metadata: Read-only, defensively copied JSON-compatible mapping.
            It contains no tensors, paths, callables, or model data.
        rng_state_restored: Whether the saved one-dimensional CPU ``uint8``
            PyTorch RNG tensor was applied.

    The model and optimizer are restored in place and are intentionally not
    returned. Only PyTorch CPU RNG state is handled; Python, NumPy, CUDA,
    iterator, sampler, and DataLoader state are outside this stage.
    """

    training_state: MultimodalTrainingState
    extra_metadata: Mapping[str, object]
    rng_state_restored: bool

    def __post_init__(self) -> None:
        if not isinstance(self.training_state, MultimodalTrainingState):
            raise TypeError("training_state must be MultimodalTrainingState.")
        if not isinstance(self.extra_metadata, Mapping):
            raise TypeError("extra_metadata must be a mapping.")
        if not isinstance(self.rng_state_restored, bool):
            raise TypeError("rng_state_restored must be bool.")
        copied = _copy_metadata(self.extra_metadata)
        object.__setattr__(
            self,
            "extra_metadata",
            MappingProxyType(dict(copied)),
        )


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_public_objects(
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
    objective: MultimodalTrainingObjective,
) -> None:
    if not isinstance(model, MultimodalEmotionClassifier):
        raise TypeError("model must be MultimodalEmotionClassifier.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if not isinstance(objective, MultimodalTrainingObjective):
        raise TypeError("objective must be MultimodalTrainingObjective.")


def _checkpoint_path(
    path: str | os.PathLike[str],
    *,
    for_load: bool,
) -> Path:
    if isinstance(path, bytes):
        raise TypeError("checkpoint path must be str or os.PathLike[str].")
    try:
        raw_path = os.fspath(path)
    except TypeError as error:
        raise TypeError(
            "checkpoint path must be str or os.PathLike[str]."
        ) from error
    if not isinstance(raw_path, str):
        raise TypeError("checkpoint path must resolve to a string.")
    if not raw_path.strip():
        raise ValueError("checkpoint path must not be empty.")
    resolved = Path(raw_path)
    if for_load:
        if not resolved.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
        if resolved.is_dir():
            raise IsADirectoryError(
                f"checkpoint path is a directory: {resolved}"
            )
    else:
        if not resolved.parent.exists():
            raise FileNotFoundError(
                f"checkpoint parent directory does not exist: {resolved.parent}"
            )
        if resolved.exists() and resolved.is_dir():
            raise IsADirectoryError(
                f"checkpoint path is a directory: {resolved}"
            )
    return resolved


def _copy_metadata_value(value: object, *, location: str) -> _MetadataValue:
    if (
        isinstance(value, (Tensor, os.PathLike, Enum))
        or is_dataclass(value)
        or callable(value)
    ):
        raise TypeError(
            f"{location} contains a forbidden non-JSON metadata value "
            f"of type {type(value).__name__}."
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} float values must be finite.")
        return value
    if isinstance(value, list):
        return [
            _copy_metadata_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _copy_metadata_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if type(value) is dict:
        copied: dict[str, _MetadataValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} dictionary keys must be strings.")
            copied[key] = _copy_metadata_value(
                item,
                location=f"{location}.{key}",
            )
        return copied
    raise TypeError(
        f"{location} must contain only JSON-compatible scalar, list, tuple, "
        "or string-key dict values."
    )


def _copy_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, _MetadataValue]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("extra_metadata must be a mapping or None.")
    copied: dict[str, _MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("extra_metadata keys must be strings.")
        copied[key] = _copy_metadata_value(
            value,
            location=f"extra_metadata.{key}",
        )
    return copied


def _objective_config(
    objective: MultimodalTrainingObjective,
) -> dict[str, str | float]:
    config = objective.config
    weights = config.weights
    return {
        "loss_kind": config.loss_kind.value,
        "fused": weights.fused,
        "speech_auxiliary": weights.speech_auxiliary,
        "physiology_auxiliary": weights.physiology_auxiliary,
        "fused_quadrant": weights.fused_quadrant,
        "speech_quadrant": weights.speech_quadrant,
        "physiology_quadrant": weights.physiology_quadrant,
        "focal_gamma": config.focal_gamma,
    }


def _training_state_payload(
    state: MultimodalTrainingState,
) -> dict[str, int | float | None]:
    if not isinstance(state, MultimodalTrainingState):
        raise TypeError("training_state must be MultimodalTrainingState.")
    return {
        "completed_epochs": state.completed_epochs,
        "global_optimizer_steps": state.global_optimizer_steps,
        "best_validation_loss": state.best_validation_loss,
    }


def _optimizer_parameter_groups(
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[str, ...], ...]:
    parameter_names = {
        parameter: name for name, parameter in model.named_parameters()
    }
    seen: set[nn.Parameter] = set()
    fingerprint: list[tuple[str, ...]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ValueError(
                f"optimizer parameter group {group_index} params must be a list."
            )
        names: list[str] = []
        for parameter_index, parameter in enumerate(parameters):
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    "optimizer parameters must be torch.nn.Parameter; "
                    f"group {group_index}, index {parameter_index} is invalid."
                )
            if parameter not in parameter_names:
                raise ValueError(
                    "every optimizer parameter must belong to model.named_parameters; "
                    f"group {group_index}, index {parameter_index} does not."
                )
            if parameter in seen:
                raise ValueError(
                    "an optimizer parameter must not occur more than once; "
                    f"duplicate {parameter_names[parameter]!r}."
                )
            seen.add(parameter)
            names.append(parameter_names[parameter])
        fingerprint.append(tuple(names))
    return tuple(fingerprint)


def _validate_exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> dict[object, object]:
    if type(value) is not dict:
        raise MultimodalCheckpointError(f"{name} must be a strict dict.")
    fields = set(value)
    if not all(isinstance(field, str) for field in fields):
        raise MultimodalCheckpointError(f"{name} keys must all be strings.")
    missing = expected - fields
    unknown = fields - expected
    if missing or unknown:
        raise MultimodalCheckpointError(
            f"{name} fields mismatch; missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}."
        )
    return value


def _parse_training_state(value: object) -> MultimodalTrainingState:
    payload = _validate_exact_fields(
        value,
        _TRAINING_STATE_FIELDS,
        name="training_state",
    )
    try:
        return MultimodalTrainingState(
            completed_epochs=payload["completed_epochs"],  # type: ignore[arg-type]
            global_optimizer_steps=payload["global_optimizer_steps"],  # type: ignore[arg-type]
            best_validation_loss=payload["best_validation_loss"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise MultimodalCheckpointError(
            f"training_state is invalid: {error}"
        ) from error


def _validate_rng_state(value: object) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.uint8
        or value.device.type != "cpu"
        or value.ndim != 1
    ):
        raise MultimodalCheckpointError(
            "torch_cpu_rng_state must be a one-dimensional CPU uint8 tensor."
        )
    expected_elements = torch.get_rng_state().numel()
    if value.numel() != expected_elements:
        raise MultimodalCheckpointError(
            "torch_cpu_rng_state length does not match the current PyTorch "
            f"CPU generator; expected {expected_elements}, received "
            f"{value.numel()}."
        )
    return value.clone()


def _validate_finite_tree(value: object, *, location: str) -> None:
    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise MultimodalCheckpointError(
                f"{location} contains NaN or Inf."
            )
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MultimodalCheckpointError(
                f"{location} contains a non-finite float."
            )
        return
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        for index, item in enumerate(sequence):
            _validate_finite_tree(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise MultimodalCheckpointError(
                    f"{location} dictionary keys must be exact str or int "
                    "values."
                )
            _validate_finite_tree(item, location=f"{location}[{key!r}]")
        return
    raise MultimodalCheckpointError(
        f"{location} contains unsupported type {type(value).__name__}."
    )


def _extra_state_equal(expected: object, loaded: object) -> bool:
    if type(expected) is not type(loaded):
        return False
    if isinstance(expected, dict):
        loaded_dict = cast(dict[object, object], loaded)
        if set(expected) != set(loaded_dict):
            return False
        return all(
            _extra_state_equal(expected[key], loaded_dict[key])
            for key in expected
        )
    if isinstance(expected, (list, tuple)):
        loaded_sequence = cast(list[object] | tuple[object, ...], loaded)
        return len(expected) == len(loaded_sequence) and all(
            _extra_state_equal(left, right)
            for left, right in zip(expected, loaded_sequence, strict=True)
        )
    return bool(expected == loaded)


def _validate_model_state(
    loaded: object,
    model: MultimodalEmotionClassifier,
) -> dict[str, object]:
    if type(loaded) is not dict:
        raise MultimodalCheckpointError("model_state must be a strict dict.")
    _validate_finite_tree(loaded, location="model_state")
    expected = dict(model.state_dict())
    loaded_keys = set(loaded)
    if not all(isinstance(key, str) for key in loaded_keys):
        raise MultimodalCheckpointError(
            "model_state keys must all be strings."
        )
    expected_keys = set(expected)
    if loaded_keys != expected_keys:
        raise MultimodalCheckpointError(
            "model_state keys mismatch; "
            f"missing={sorted(expected_keys - loaded_keys)!r}, "
            f"unknown={sorted(loaded_keys - expected_keys)!r}."
        )
    for key, expected_value in expected.items():
        loaded_value = loaded[key]
        if isinstance(expected_value, Tensor):
            if not isinstance(loaded_value, Tensor):
                raise MultimodalCheckpointError(
                    f"model_state[{key!r}] must be a tensor."
                )
            if loaded_value.shape != expected_value.shape:
                raise MultimodalCheckpointError(
                    f"model_state[{key!r}] shape mismatch: expected "
                    f"{tuple(expected_value.shape)}, received "
                    f"{tuple(loaded_value.shape)}."
                )
            if loaded_value.dtype != expected_value.dtype:
                raise MultimodalCheckpointError(
                    f"model_state[{key!r}] dtype mismatch: expected "
                    f"{expected_value.dtype}, received {loaded_value.dtype}."
                )
            _validate_finite_tree(
                loaded_value,
                location=f"model_state[{key!r}]",
            )
        elif not _extra_state_equal(expected_value, loaded_value):
            raise MultimodalCheckpointError(
                f"model_state extra-state {key!r} does not match the target "
                "construction fingerprint."
            )
    return loaded


def _validate_optimizer_state(
    value: object,
    *,
    optimizer: torch.optim.Optimizer,
    fingerprint: tuple[tuple[str, ...], ...],
) -> dict[str, object]:
    if type(value) is not dict:
        raise MultimodalCheckpointError("optimizer_state must be a strict dict.")
    if set(value) != {"state", "param_groups"}:
        raise MultimodalCheckpointError(
            "optimizer_state must contain exactly state and param_groups."
        )
    state = value["state"]
    groups = value["param_groups"]
    if type(state) is not dict or not isinstance(groups, list):
        raise MultimodalCheckpointError(
            "optimizer_state state/param_groups have invalid container types."
        )
    target_state = optimizer.state_dict()
    target_groups = target_state["param_groups"]
    if len(groups) != len(fingerprint):
        raise MultimodalCheckpointError(
            "optimizer_state parameter group count does not match the target."
        )
    parameter_ids: set[int] = set()
    for index, group in enumerate(groups):
        if type(group) is not dict or not isinstance(group.get("params"), list):
            raise MultimodalCheckpointError(
                f"optimizer_state group {index} must be a dict with params list."
            )
        target_group = target_groups[index]
        if set(group) != set(target_group):
            raise MultimodalCheckpointError(
                f"optimizer_state group {index} option keys do not match "
                "the target optimizer construction."
            )
        if len(group["params"]) != len(fingerprint[index]):
            raise MultimodalCheckpointError(
                f"optimizer_state group {index} parameter count does not "
                "match the target fingerprint."
            )
        if group["params"] != target_group["params"]:
            raise MultimodalCheckpointError(
                f"optimizer_state group {index} parameter index order does "
                "not match the target optimizer structure."
            )
        for parameter_id in group["params"]:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise MultimodalCheckpointError(
                    "optimizer_state parameter identifiers must be integers."
                )
            if parameter_id in parameter_ids:
                raise MultimodalCheckpointError(
                    "optimizer_state contains a duplicate parameter identifier."
                )
            parameter_ids.add(parameter_id)
    if not all(
        isinstance(key, int) and not isinstance(key, bool)
        for key in state
    ) or not set(state).issubset(parameter_ids):
        raise MultimodalCheckpointError(
            "optimizer_state moment keys must reference saved parameters."
        )
    _validate_finite_tree(value, location="optimizer_state")
    return value


def _validate_parameter_fingerprint(
    value: object,
    expected: tuple[tuple[str, ...], ...],
) -> None:
    if (
        not isinstance(value, tuple)
        or not all(
            isinstance(group, tuple)
            and all(isinstance(name, str) for name in group)
            for group in value
        )
    ):
        raise MultimodalCheckpointError(
            "optimizer_parameter_groups must be tuple[tuple[str, ...], ...]."
        )
    if value != expected:
        raise MultimodalCheckpointError(
            "optimizer parameter group names or ordering do not match target."
        )


def _preflight(
    payload_value: object,
    *,
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
    objective: MultimodalTrainingObjective,
) -> tuple[
    dict[str, object],
    dict[str, object],
    MultimodalTrainingState,
    Tensor,
    dict[str, _MetadataValue],
]:
    payload = _validate_exact_fields(
        payload_value,
        _TOP_LEVEL_FIELDS,
        name="checkpoint payload",
    )
    schema_version = payload["checkpoint_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _CHECKPOINT_SCHEMA_VERSION
    ):
        raise MultimodalCheckpointError(
            "unsupported checkpoint_schema_version; "
            f"expected {_CHECKPOINT_SCHEMA_VERSION!r}, received "
            f"{payload['checkpoint_schema_version']!r}."
        )
    expected_types = (
        ("model_type", _qualified_type(model)),
        ("optimizer_type", _qualified_type(optimizer)),
        ("objective_type", _qualified_type(objective)),
    )
    for field, expected in expected_types:
        if payload[field] != expected:
            raise MultimodalCheckpointError(
                f"{field} mismatch: expected {expected!r}, "
                f"received {payload[field]!r}."
            )
    objective_config = _validate_exact_fields(
        payload["objective_config"],
        _OBJECTIVE_FIELDS,
        name="objective_config",
    )
    if not _extra_state_equal(
        _objective_config(objective),
        objective_config,
    ):
        raise MultimodalCheckpointError(
            "objective_config does not match the target objective."
        )
    fingerprint = _optimizer_parameter_groups(model, optimizer)
    _validate_parameter_fingerprint(
        payload["optimizer_parameter_groups"],
        fingerprint,
    )
    training_state = _parse_training_state(payload["training_state"])
    rng_state = _validate_rng_state(payload["torch_cpu_rng_state"])
    if type(payload["extra_metadata"]) is not dict:
        raise MultimodalCheckpointError(
            "extra_metadata must be a strict string-key dict."
        )
    try:
        metadata = _copy_metadata(payload["extra_metadata"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise MultimodalCheckpointError(
            f"extra_metadata is invalid: {error}"
        ) from error
    model_state = _validate_model_state(payload["model_state"], model)
    optimizer_state = _validate_optimizer_state(
        payload["optimizer_state"],
        optimizer=optimizer,
        fingerprint=fingerprint,
    )
    return model_state, optimizer_state, training_state, rng_state, metadata


def save_multimodal_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
    objective: MultimodalTrainingObjective,
    training_state: MultimodalTrainingState,
    extra_metadata: Mapping[str, object] | None = None,
) -> None:
    """Atomically save model, optimizer, objective semantics, progress, and RNG.

    Args:
        path: Non-empty destination path whose parent already exists.
        model: Final multimodal classifier. Its tensor state and nested
            construction extra-state are saved without a forward or mode switch.
        optimizer: Existing optimizer over an ordered subset of model parameters.
        objective: Stage-13A objective whose scalar configuration is serialized.
        training_state: Immutable scalar epoch and optimizer-step progress.
        extra_metadata: Optional JSON-compatible, string-key metadata mapping.

    Raises:
        TypeError: If public object or metadata types are invalid.
        ValueError: If scalar metadata, training state, or optimizer ownership
            is invalid.
        FileNotFoundError: If the destination parent does not exist.
        IsADirectoryError: If the destination is a directory.
        MultimodalCheckpointError: If temporary serialization, synchronization,
            or atomic replacement fails. The original destination remains
            untouched when failure occurs before ``os.replace``.

    A uniquely named temporary file is written in the destination directory,
    flushed, best-effort ``fsync``-synchronized, and atomically installed with
    ``os.replace``. Only PyTorch CPU RNG is saved; Python, NumPy, CUDA, iterator,
    sampler, and DataLoader state are intentionally outside this stage.
    """
    _validate_public_objects(model, optimizer, objective)
    destination = _checkpoint_path(path, for_load=False)
    metadata = _copy_metadata(extra_metadata)
    training_payload = _training_state_payload(training_state)
    fingerprint = _optimizer_parameter_groups(model, optimizer)
    payload: dict[str, object] = {
        "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "model_type": _qualified_type(model),
        "model_state": dict(model.state_dict()),
        "optimizer_type": _qualified_type(optimizer),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_parameter_groups": fingerprint,
        "objective_type": _qualified_type(objective),
        "objective_config": _objective_config(objective),
        "training_state": training_payload,
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "extra_metadata": metadata,
    }
    try:
        _validate_finite_tree(payload, location="checkpoint payload")
    except MultimodalCheckpointError as error:
        raise ValueError(
            f"checkpoint payload contains unsupported or non-finite state: {error}"
        ) from error
    file_descriptor = -1
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = -1
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except Exception as error:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise MultimodalCheckpointError(
            f"failed to atomically save checkpoint to {destination}: {error}"
        ) from error


def load_multimodal_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: MultimodalEmotionClassifier,
    optimizer: torch.optim.Optimizer,
    objective: MultimodalTrainingObjective,
    restore_rng_state: bool = True,
) -> LoadedMultimodalCheckpoint:
    """Strictly preflight and restore a CPU multimodal checkpoint.

    Args:
        path: Existing checkpoint file.
        model: Target classifier with exactly matching type, tensor state
            shapes/dtypes, and every nested construction extra-state.
        optimizer: Target optimizer with matching type and ordered model
            parameter-name groups.
        objective: Target objective with exactly matching loss semantics.
        restore_rng_state: Whether to restore the saved PyTorch CPU RNG tensor.

    Returns:
        :class:`LoadedMultimodalCheckpoint` containing immutable progress,
        defensively copied read-only metadata, and the RNG restoration status.

    Raises:
        TypeError: If public inputs have invalid types.
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is a directory.
        MultimodalCheckpointError: If deserialization, strict preflight, or
            in-place state restoration fails.

    Preflight checks all project-known keys, tensor shapes/dtypes/finite values,
    nested model extra-state, objective semantics, optimizer parameter order,
    optimizer finite state, training progress, RNG, and metadata before calling
    either ``load_state_dict``. PyTorch's actual ``load_state_dict`` is not a
    universal transaction for arbitrary third-party custom modules; this
    complete preflight prevents known project configuration mismatches from
    causing partial copies but cannot promise universal transactional behavior.
    The target model's train/eval mode is never changed. Only CPU PyTorch RNG
    is optionally restored, after both state loads succeed.
    """
    _validate_public_objects(model, optimizer, objective)
    if not isinstance(restore_rng_state, bool):
        raise TypeError("restore_rng_state must be bool.")
    source = _checkpoint_path(path, for_load=True)
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise MultimodalCheckpointError(
            f"failed to read checkpoint {source}: {error}"
        ) from error
    try:
        (
            model_state,
            optimizer_state,
            training_state,
            rng_state,
            metadata,
        ) = _preflight(
            payload,
            model=model,
            optimizer=optimizer,
            objective=objective,
        )
    except MultimodalCheckpointError:
        raise
    except Exception as error:
        raise MultimodalCheckpointError(
            f"checkpoint preflight failed: {error}"
        ) from error
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        for name, value in (
            *model.named_parameters(),
            *model.named_buffers(),
        ):
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()
            ):
                raise RuntimeError(
                    f"restored model tensor {name!r} contains NaN or Inf."
                )
        if restore_rng_state:
            torch.set_rng_state(rng_state)
    except Exception as error:
        raise MultimodalCheckpointError(
            f"checkpoint state restoration failed: {error}"
        ) from error
    return LoadedMultimodalCheckpoint(
        training_state=training_state,
        extra_metadata=metadata,
        rng_state_restored=restore_rng_state,
    )


__all__ = [
    "LoadedMultimodalCheckpoint",
    "MultimodalCheckpointError",
    "load_multimodal_checkpoint",
    "save_multimodal_checkpoint",
]
