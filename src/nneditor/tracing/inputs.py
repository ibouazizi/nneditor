"""Input specification validation, deterministic binding, and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nneditor.cancellation import CancellationToken
from nneditor.ir.core import Document, Value
from nneditor.tracing.contracts import (
    InputBinding,
    InputSource,
    InputSpecification,
    resolved_file,
)

__all__ = [
    "InputSpecificationStore",
    "bind_inputs",
    "default_input_specification",
    "tensor_file_binding",
    "tensor_file_binding_for_input",
]

_DTYPES: dict[str, np.dtype[np.generic]] = {
    name: np.dtype(name)
    for name in (
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    )
}
_SIDECAR_VERSION = 1


def _file_hash(path: Path, token: CancellationToken | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if token is not None:
                token.raise_if_cancelled()
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def tensor_file_binding(
    name: str,
    element_type: str,
    shape: tuple[int, ...],
    path: Path | str,
    *,
    token: CancellationToken | None = None,
) -> InputBinding:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError(f"input tensor file {target} is not a readable file")
    return InputBinding(
        name,
        element_type,
        shape,
        InputSource.TENSOR_FILE,
        tensor_file=str(target),
        file_hash=_file_hash(target, token),
    )


def tensor_file_binding_for_input(
    document: Document,
    input_name: str,
    path: Path | str,
    *,
    token: CancellationToken | None = None,
) -> InputBinding:
    """Inspect a safe ``.npy`` header and bind it to one declared graph input."""
    values = {value.name or value.id: value for value in _input_values(document)}
    try:
        value = values[input_name]
    except KeyError:
        raise ValueError(f"{input_name!r} is not a model input") from None
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".npy":
        raise ValueError("trace tensor inputs must be safe .npy arrays")
    if not target.is_file():
        raise ValueError(f"input tensor file {target} is not a readable file")
    array = np.load(target, allow_pickle=False, mmap_mode="r")
    element_type = array.dtype.name
    if value.element_type != element_type:
        raise ValueError(
            f"tensor dtype {element_type} does not match model input "
            f"{input_name!r} ({value.element_type})"
        )
    binding = tensor_file_binding(
        input_name,
        element_type,
        tuple(int(item) for item in array.shape),
        target,
        token=token,
    )
    _validate_declared(value, binding, {})
    return binding


def _input_values(document: Document) -> tuple[Value, ...]:
    graph = document.main_graph
    return tuple(
        graph.value(value_id)
        for value_id in graph.inputs
        if value_id not in graph.initializers
    )


def _concrete_shape(
    value: Value,
    supplied: Mapping[str, tuple[int, ...]],
    known_symbols: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    declared = value.shape
    if declared is None:
        try:
            shape = supplied[value.name or value.id]
        except KeyError:
            raise ValueError(
                f"input {value.name or value.id!r} has no declared shape; "
                "provide every extent explicitly"
            ) from None
        return shape
    key = value.name or value.id
    provided = supplied.get(key)
    if all(isinstance(dimension, int) for dimension in declared):
        concrete = tuple(
            int(dimension) for dimension in declared if isinstance(dimension, int)
        )
        if provided is not None and provided != concrete:
            raise ValueError(
                f"input {key!r} shape {provided} does not match declared {concrete}"
            )
        return concrete
    if provided is None:
        symbols = known_symbols or {}
        inferred: list[int] = []
        for dimension in declared:
            if isinstance(dimension, int):
                inferred.append(dimension)
            elif isinstance(dimension, str) and dimension in symbols:
                inferred.append(symbols[dimension])
            else:
                break
        if len(inferred) == len(declared):
            return tuple(inferred)
        raise ValueError(
            f"input {key!r} has symbolic dimensions {declared}; provide every "
            "extent explicitly"
        )
    if len(provided) != len(declared):
        raise ValueError(
            f"input {key!r} rank {len(provided)} does not match declared "
            f"rank {len(declared)}"
        )
    for actual, expected in zip(provided, declared, strict=True):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"input {key!r} extent {actual} does not match declared {expected}"
            )
    return provided


def _is_mask_input(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized == "mask" or normalized.endswith("_mask")


def default_input_specification(
    document: Document,
    *,
    seed: int = 0,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
    bindings: Mapping[str, InputBinding] | None = None,
) -> InputSpecification:
    """Build a deterministic random specification, refusing guessed extents."""
    supplied = shapes or {}
    supplied_bindings = bindings or {}
    values = {value.name or value.id: value for value in _input_values(document)}
    extra = sorted(supplied_bindings.keys() - values.keys())
    if extra:
        raise ValueError(f"unknown model input binding(s): {extra}")
    symbols: dict[str, int] = {}
    for name, binding in supplied_bindings.items():
        if binding.name != name:
            raise ValueError(f"input binding {binding.name!r} cannot replace {name!r}")
        _validate_declared(values[name], binding, symbols)

    resolved_bindings: list[InputBinding] = []
    for position, value in enumerate(values.values()):
        name = value.name or value.id
        selected = supplied_bindings.get(name)
        if selected is not None:
            resolved_bindings.append(selected)
            continue
        if value.element_type not in _DTYPES:
            raise ValueError(
                f"input {name!r} element type {value.element_type!r} cannot be "
                "generated by the reference trace input binder"
            )
        shape = _concrete_shape(value, supplied, symbols)
        binding = (
            InputBinding(
                name,
                value.element_type,
                shape,
                InputSource.CONSTANT,
                constant=(1,),
            )
            if _is_mask_input(name)
            else InputBinding(
                name,
                value.element_type,
                shape,
                InputSource.SEEDED_RANDOM,
                seed=seed + position,
            )
        )
        _validate_declared(value, binding, symbols)
        resolved_bindings.append(binding)
    specification = InputSpecification(
        document.source.content_hash,
        tuple(resolved_bindings),
    )
    validated_symbols: dict[str, int] = {}
    for binding in specification.bindings:
        _validate_declared(values[binding.name], binding, validated_symbols)
    return specification


def _validate_declared(
    value: Value,
    binding: InputBinding,
    symbols: dict[str, int],
) -> None:
    if value.element_type != binding.element_type:
        raise ValueError(
            f"input {binding.name!r} dtype {binding.element_type} does not match "
            f"declared {value.element_type}"
        )
    if value.shape is None:
        return
    if len(value.shape) != len(binding.shape):
        raise ValueError(f"input {binding.name!r} rank does not match the graph")
    for declared, actual in zip(value.shape, binding.shape, strict=True):
        if isinstance(declared, int) and declared != actual:
            raise ValueError(
                f"input {binding.name!r} extent {actual} does not match "
                f"declared {declared}"
            )
        if isinstance(declared, str):
            previous = symbols.setdefault(declared, actual)
            if previous != actual:
                raise ValueError(
                    f"symbolic dimension {declared!r} was bound to both "
                    f"{previous} and {actual}"
                )


def _binding_array(
    binding: InputBinding,
    token: CancellationToken | None,
) -> NDArray[np.generic]:
    dtype = _DTYPES.get(binding.element_type)
    if dtype is None:
        raise ValueError(f"unsupported trace input dtype {binding.element_type!r}")
    if binding.source is InputSource.SEEDED_RANDOM:
        assert binding.seed is not None
        generator = np.random.default_rng(binding.seed)
        if np.issubdtype(dtype, np.floating):
            return generator.standard_normal(binding.shape).astype(dtype)
        if np.issubdtype(dtype, np.bool_):
            return generator.integers(0, 2, binding.shape).astype(dtype)
        return generator.integers(0, 8, binding.shape).astype(dtype)
    if binding.source is InputSource.TENSOR_FILE:
        path = resolved_file(binding)
        if _file_hash(path, token) != binding.file_hash:
            raise ValueError(
                f"input tensor file {path.name} changed after the specification "
                "was saved"
            )
        if path.suffix.lower() != ".npy":
            raise ValueError("trace tensor files currently use safe .npy arrays")
        # Map first so an invalid dtype/shape can be rejected without loading
        # a user-supplied payload into the editor process.
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return np.asarray(array)
    values = np.asarray(binding.constant, dtype=dtype)
    count = int(np.prod(binding.shape, dtype=np.int64))
    if values.size == 1 and count != 1:
        return np.full(binding.shape, values.item(), dtype=dtype)
    if values.size != count:
        raise ValueError(
            f"constant input {binding.name!r} has {values.size} value(s), "
            f"but shape {binding.shape} needs {count}"
        )
    return values.reshape(binding.shape)


def bind_inputs(
    document: Document,
    specification: InputSpecification,
    *,
    token: CancellationToken | None = None,
) -> dict[str, NDArray[np.generic]]:
    if specification.artifact_hash != document.source.content_hash:
        raise ValueError("input specification belongs to a different artifact")
    values = {value.name or value.id: value for value in _input_values(document)}
    bindings = {binding.name: binding for binding in specification.bindings}
    if bindings.keys() != values.keys():
        missing = sorted(values.keys() - bindings.keys())
        extra = sorted(bindings.keys() - values.keys())
        raise ValueError(
            f"input bindings do not match graph inputs; missing={missing}, "
            f"extra={extra}"
        )
    symbols: dict[str, int] = {}
    feeds: dict[str, NDArray[np.generic]] = {}
    for name, value in values.items():
        if token is not None:
            token.raise_if_cancelled()
        binding = bindings[name]
        _validate_declared(value, binding, symbols)
        array = _binding_array(binding, token)
        expected = _DTYPES[binding.element_type]
        if array.dtype != expected:
            raise ValueError(
                f"input {name!r} file dtype {array.dtype} does not match "
                f"specified {expected}"
            )
        if tuple(array.shape) != binding.shape:
            raise ValueError(
                f"input {name!r} shape {tuple(array.shape)} does not match "
                f"specified {binding.shape}"
            )
        feeds[name] = np.ascontiguousarray(array)
    return feeds


class InputSpecificationStore:
    """Atomic, versioned input specifications keyed by artifact identity."""

    __slots__ = ("_items", "_lock", "path")

    def __init__(self, directory: Path) -> None:
        self.path = directory / "trace-inputs.json"
        self._lock = threading.RLock()
        self._items = self._load()

    def _load(self) -> dict[str, dict[str, InputSpecification]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _SIDECAR_VERSION:
            return {}
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, dict):
            return {}
        loaded: dict[str, dict[str, InputSpecification]] = {}
        try:
            for artifact_hash, specs in artifacts.items():
                if not isinstance(specs, list):
                    raise ValueError
                parsed = [InputSpecification.from_json(item) for item in specs]
                loaded[str(artifact_hash)] = {item.hash: item for item in parsed}
        except (KeyError, TypeError, ValueError):
            return {}
        return loaded

    def save(self, specification: InputSpecification) -> None:
        with self._lock:
            self._items.setdefault(specification.artifact_hash, {})[
                specification.hash
            ] = specification
            payload = {
                "version": _SIDECAR_VERSION,
                "artifacts": {
                    artifact_hash: [
                        specification.to_json()
                        for specification in sorted(
                            specs.values(), key=lambda item: item.hash
                        )
                    ]
                    for artifact_hash, specs in sorted(self._items.items())
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_name(f"{self.path.name}.partial")
            try:
                with open(partial, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=1, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, self.path)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise

    def specifications(self, artifact_hash: str) -> tuple[InputSpecification, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._items.get(artifact_hash, {}).values(),
                    key=lambda item: item.hash,
                )
            )

    def get(self, artifact_hash: str, specification_hash: str) -> InputSpecification:
        with self._lock:
            try:
                return self._items[artifact_hash][specification_hash]
            except KeyError:
                raise KeyError(specification_hash) from None
