"""Stable contracts for consent-gated inference tracing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "ActivationRecord",
    "CaptureState",
    "InputBinding",
    "InputSource",
    "InputSpecification",
    "TraceApproval",
    "TraceBackend",
    "TraceDevice",
    "TraceKey",
    "TraceLimits",
    "TraceRequest",
    "TraceResult",
]


def _canonical_hash(payload: object, prefix: str) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


class InputSource(StrEnum):
    SEEDED_RANDOM = "seeded-random"
    TENSOR_FILE = "tensor-file"
    CONSTANT = "constant"


class CaptureState(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    DROPPED = "dropped"
    EVICTED = "evicted"


class TraceBackend(StrEnum):
    """Execution engine selected for one explicitly approved trace."""

    AUTO = "auto"
    ONNX_RUNTIME = "onnxruntime"
    REFERENCE = "reference"
    REFERENCE_NORMALIZED = "reference-normalized"


class TraceDevice(StrEnum):
    """Hardware class requested for one explicitly approved trace."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


Scalar = bool | int | float


@dataclass(frozen=True, slots=True)
class TraceLimits:
    wall_seconds: float = 30.0
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    capture_bytes: int = 256 * 1024 * 1024
    chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("trace wall-clock limit must be finite and positive")
        if self.memory_bytes <= 0:
            raise ValueError("trace memory limit must be positive")
        if self.capture_bytes <= 0:
            raise ValueError("trace capture limit must be positive")
        if self.chunk_bytes <= 0 or self.chunk_bytes > self.capture_bytes:
            raise ValueError(
                "trace chunk size must be positive and fit inside the capture limit"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "memory_bytes": self.memory_bytes,
            "capture_bytes": self.capture_bytes,
            "chunk_bytes": self.chunk_bytes,
        }

    @classmethod
    def from_json(cls, raw: object) -> TraceLimits:
        if not isinstance(raw, dict):
            raise ValueError("trace limits are malformed")
        return cls(
            wall_seconds=float(raw["wall_seconds"]),
            memory_bytes=int(raw["memory_bytes"]),
            capture_bytes=int(raw["capture_bytes"]),
            chunk_bytes=int(raw["chunk_bytes"]),
        )

    @property
    def hash(self) -> str:
        return _canonical_hash(self.to_json(), "limits")


@dataclass(frozen=True, slots=True)
class InputBinding:
    name: str
    element_type: str
    shape: tuple[int, ...]
    source: InputSource
    seed: int | None = None
    tensor_file: str | None = None
    file_hash: str | None = None
    constant: tuple[Scalar, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input binding needs a graph input name")
        if not self.element_type:
            raise ValueError(f"input {self.name!r} needs an element type")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"input {self.name!r} dimensions must be positive")
        if self.source is InputSource.SEEDED_RANDOM:
            if self.seed is None:
                raise ValueError(f"random input {self.name!r} needs a seed")
            if self.tensor_file is not None or self.constant:
                raise ValueError("random input cannot also carry file/constant data")
        elif self.source is InputSource.TENSOR_FILE:
            if self.tensor_file is None or self.file_hash is None:
                raise ValueError(
                    f"tensor-file input {self.name!r} needs a path and content hash"
                )
            if self.seed is not None or self.constant:
                raise ValueError("tensor-file input cannot carry random/constant data")
        elif not self.constant:
            raise ValueError(f"constant input {self.name!r} needs at least one value")

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "element_type": self.element_type,
            "shape": list(self.shape),
            "source": self.source.value,
            "seed": self.seed,
            "tensor_file": self.tensor_file,
            "file_hash": self.file_hash,
            "constant": list(self.constant),
        }

    @classmethod
    def from_json(cls, raw: object) -> InputBinding:
        if not isinstance(raw, dict):
            raise ValueError("input binding is malformed")
        constant = raw.get("constant", [])
        shape = raw.get("shape")
        if not isinstance(constant, list) or not isinstance(shape, list):
            raise ValueError("input binding shape/constant is malformed")
        values: list[Scalar] = []
        for value in constant:
            if not isinstance(value, bool | int | float):
                raise ValueError("constant inputs must contain numeric scalars")
            values.append(value)
        return cls(
            name=str(raw["name"]),
            element_type=str(raw["element_type"]),
            shape=tuple(int(item) for item in shape),
            source=InputSource(str(raw["source"])),
            seed=None if raw.get("seed") is None else int(raw["seed"]),
            tensor_file=(
                None if raw.get("tensor_file") is None else str(raw["tensor_file"])
            ),
            file_hash=(None if raw.get("file_hash") is None else str(raw["file_hash"])),
            constant=tuple(values),
        )


@dataclass(frozen=True, slots=True)
class InputSpecification:
    artifact_hash: str
    bindings: tuple[InputBinding, ...]

    def __post_init__(self) -> None:
        if not self.artifact_hash:
            raise ValueError("input specification needs an artifact hash")
        names = [binding.name for binding in self.bindings]
        if not names or len(names) != len(set(names)):
            raise ValueError("input specification needs unique named bindings")

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_hash": self.artifact_hash,
            "bindings": [
                binding.to_json()
                for binding in sorted(self.bindings, key=lambda item: item.name)
            ],
        }

    @classmethod
    def from_json(cls, raw: object) -> InputSpecification:
        if not isinstance(raw, dict) or not isinstance(raw.get("bindings"), list):
            raise ValueError("input specification is malformed")
        return cls(
            artifact_hash=str(raw["artifact_hash"]),
            bindings=tuple(InputBinding.from_json(item) for item in raw["bindings"]),
        )

    @property
    def hash(self) -> str:
        return _canonical_hash(self.to_json(), "inputs")


@dataclass(frozen=True, slots=True)
class TraceApproval:
    model_title: str
    artifact_hash: str
    input_specification_hash: str
    limits_hash: str
    approved: bool
    backend: TraceBackend = TraceBackend.AUTO
    device: TraceDevice = TraceDevice.AUTO

    @classmethod
    def approve(
        cls,
        model_title: str,
        artifact_hash: str,
        specification: InputSpecification,
        limits: TraceLimits,
        backend: TraceBackend = TraceBackend.AUTO,
        device: TraceDevice = TraceDevice.AUTO,
    ) -> TraceApproval:
        return cls(
            model_title,
            artifact_hash,
            specification.hash,
            limits.hash,
            True,
            backend,
            device,
        )

    def validate(
        self,
        *,
        model_title: str,
        artifact_hash: str,
        specification: InputSpecification,
        limits: TraceLimits,
        backend: TraceBackend = TraceBackend.AUTO,
        device: TraceDevice = TraceDevice.AUTO,
    ) -> None:
        if not self.approved:
            raise ValueError("inference tracing requires explicit per-run approval")
        expected = (
            model_title,
            artifact_hash,
            specification.hash,
            limits.hash,
            backend,
            device,
        )
        actual = (
            self.model_title,
            self.artifact_hash,
            self.input_specification_hash,
            self.limits_hash,
            self.backend,
            self.device,
        )
        if actual != expected:
            raise ValueError(
                "trace approval does not name this model, input specification, "
                "and resource limit set"
            )


@dataclass(frozen=True, slots=True)
class TraceKey:
    artifact_hash: str
    revision_id: str | None
    input_specification_hash: str
    backend: TraceBackend = TraceBackend.AUTO
    device: TraceDevice = TraceDevice.AUTO
    identity_version: int = 3

    def __post_init__(self) -> None:
        if not self.artifact_hash or not self.input_specification_hash:
            raise ValueError("trace key needs artifact and input-specification hashes")
        if self.identity_version not in {1, 2, 3}:
            raise ValueError("trace key identity version is unsupported")

    @property
    def id(self) -> str:
        payload: dict[str, object] = {
            "artifact_hash": self.artifact_hash,
            "revision_id": self.revision_id,
            "input_specification_hash": self.input_specification_hash,
        }
        # Version 1 predates selectable execution backends and version 2
        # predates selectable devices. Keeping their exact digests lets old
        # manifests load, while every new trace binds both choices into its
        # immutable identity.
        if self.identity_version >= 2:
            payload["backend"] = self.backend.value
        if self.identity_version >= 3:
            payload["device"] = self.device.value
        return _canonical_hash(
            payload,
            "trace",
        )


@dataclass(frozen=True, slots=True)
class TraceRequest:
    specification: InputSpecification
    limits: TraceLimits
    approval: TraceApproval
    value_ids: frozenset[str] = frozenset()
    backend: TraceBackend = TraceBackend.AUTO
    device: TraceDevice = TraceDevice.AUTO

    def __post_init__(self) -> None:
        if self.backend in {
            TraceBackend.REFERENCE,
            TraceBackend.REFERENCE_NORMALIZED,
        } and self.device not in {TraceDevice.AUTO, TraceDevice.CPU}:
            raise ValueError("reference tracing only supports CPU execution")


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    value_id: str
    value_name: str
    node_id: str | None
    role: str
    element_type: str
    numpy_dtype: str
    shape: tuple[int, ...]
    state: CaptureState
    full_byte_length: int
    stored_byte_length: int
    file_name: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not all(
            item
            for item in (
                self.value_id,
                self.value_name,
                self.role,
                self.element_type,
                self.numpy_dtype,
            )
        ):
            raise ValueError("activation record needs value, role, and dtype metadata")
        if any(dimension < -1 for dimension in self.shape):
            raise ValueError("activation dimensions cannot be less than -1")
        if (
            self.full_byte_length < 0
            or self.stored_byte_length < 0
            or self.stored_byte_length > self.full_byte_length
        ):
            raise ValueError("activation byte lengths are inconsistent")
        if self.state is CaptureState.COMPLETE and (
            self.stored_byte_length == 0
            or self.stored_byte_length != self.full_byte_length
            or self.file_name is None
        ):
            raise ValueError("complete activation records need their full payload")
        if self.state is CaptureState.TRUNCATED and (
            self.stored_byte_length == 0
            or self.stored_byte_length >= self.full_byte_length
            or self.file_name is None
        ):
            raise ValueError("truncated activation records need a captured prefix")
        if self.state in {CaptureState.DROPPED, CaptureState.EVICTED} and (
            self.stored_byte_length != 0 or self.file_name is not None
        ):
            raise ValueError("unreadable activation records cannot name payload bytes")

    @property
    def readable(self) -> bool:
        return (
            self.state in {CaptureState.COMPLETE, CaptureState.TRUNCATED}
            and self.stored_byte_length > 0
            and self.file_name is not None
        )

    def to_json(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "value_name": self.value_name,
            "node_id": self.node_id,
            "role": self.role,
            "element_type": self.element_type,
            "numpy_dtype": self.numpy_dtype,
            "shape": list(self.shape),
            "state": self.state.value,
            "full_byte_length": self.full_byte_length,
            "stored_byte_length": self.stored_byte_length,
            "file_name": self.file_name,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, raw: object) -> ActivationRecord:
        if not isinstance(raw, dict) or not isinstance(raw.get("shape"), list):
            raise ValueError("activation record is malformed")
        return cls(
            value_id=str(raw["value_id"]),
            value_name=str(raw["value_name"]),
            node_id=None if raw.get("node_id") is None else str(raw["node_id"]),
            role=str(raw["role"]),
            element_type=str(raw["element_type"]),
            numpy_dtype=str(raw["numpy_dtype"]),
            shape=tuple(int(item) for item in raw["shape"]),
            state=CaptureState(str(raw["state"])),
            full_byte_length=int(raw["full_byte_length"]),
            stored_byte_length=int(raw["stored_byte_length"]),
            file_name=(None if raw.get("file_name") is None else str(raw["file_name"])),
            reason=None if raw.get("reason") is None else str(raw["reason"]),
        )


@dataclass(frozen=True, slots=True)
class TraceResult:
    key: TraceKey
    records: tuple[ActivationRecord, ...]
    runtime: str
    diagnostics: tuple[str, ...] = ()
    execution_device: TraceDevice = TraceDevice.CPU
    execution_provider: str = "CPU"

    def __post_init__(self) -> None:
        value_ids = [record.value_id for record in self.records]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("trace result contains duplicate activation values")
        if not self.execution_provider:
            raise ValueError("trace result needs an execution provider")

    @property
    def id(self) -> str:
        return self.key.id

    @property
    def partial(self) -> bool:
        return bool(self.diagnostics) or any(
            record.state is not CaptureState.COMPLETE for record in self.records
        )

    @property
    def captured_value_ids(self) -> frozenset[str]:
        return frozenset(record.value_id for record in self.records if record.readable)

    def record(self, value_id: str) -> ActivationRecord:
        for record in self.records:
            if record.value_id == value_id:
                return record
        raise KeyError(value_id)

    def to_json(self) -> dict[str, object]:
        return {
            "key": {
                "artifact_hash": self.key.artifact_hash,
                "revision_id": self.key.revision_id,
                "input_specification_hash": self.key.input_specification_hash,
                "backend": self.key.backend.value,
                "device": self.key.device.value,
                "identity_version": self.key.identity_version,
            },
            "records": [record.to_json() for record in self.records],
            "runtime": self.runtime,
            "diagnostics": list(self.diagnostics),
            "execution_device": self.execution_device.value,
            "execution_provider": self.execution_provider,
        }

    @classmethod
    def from_json(cls, raw: object) -> TraceResult:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("key"), dict)
            or not isinstance(raw.get("records"), list)
            or not isinstance(raw.get("diagnostics"), list)
        ):
            raise ValueError("trace result is malformed")
        key = raw["key"]
        assert isinstance(key, dict)
        return cls(
            key=TraceKey(
                artifact_hash=str(key["artifact_hash"]),
                revision_id=(
                    None if key.get("revision_id") is None else str(key["revision_id"])
                ),
                input_specification_hash=str(key["input_specification_hash"]),
                backend=TraceBackend(str(key.get("backend", TraceBackend.AUTO.value))),
                device=TraceDevice(str(key.get("device", TraceDevice.AUTO.value))),
                identity_version=int(key.get("identity_version", 1)),
            ),
            records=tuple(ActivationRecord.from_json(item) for item in raw["records"]),
            runtime=str(raw["runtime"]),
            diagnostics=tuple(str(item) for item in raw["diagnostics"]),
            execution_device=TraceDevice(
                str(raw.get("execution_device", TraceDevice.CPU.value))
            ),
            execution_provider=str(raw.get("execution_provider", "CPU")),
        )


def resolved_file(binding: InputBinding) -> Path:
    if binding.tensor_file is None:
        raise ValueError(f"input {binding.name!r} has no tensor file")
    return Path(binding.tensor_file).expanduser().resolve()
