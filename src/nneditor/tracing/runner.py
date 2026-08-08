"""Parent-side orchestration for the isolated ONNX trace worker."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from nneditor.adapters.onnx.numerical import (
    _cap_worker_process,
    _close_job_object,
    _worker_environment,
)
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.core import Document
from nneditor.tracing.contracts import (
    ActivationRecord,
    TraceDevice,
    TraceKey,
    TraceLimits,
    TraceRequest,
    TraceResult,
    declared_byte_size,
)
from nneditor.tracing.inputs import bind_inputs
from nneditor.tracing.store import ActivationStore

__all__ = [
    "DeclaredValue",
    "TraceError",
    "estimated_capture_bytes",
    "recommended_trace_limits",
    "run_onnx_trace",
]

ModelBuilder = Callable[[Path, CancellationToken], None]
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_DEFAULT_TRACE_MEMORY_BYTES = 2 * _GIB
_DEFAULT_TRACE_CAPTURE_BYTES = 256 * _MIB
_DEFAULT_TRACE_CHUNK_BYTES = _MIB


class TraceError(RuntimeError):
    """An approved isolated trace could not produce a trustworthy result."""


def _darwin_process_memory_bytes(pid: int) -> int | None:
    """Return a Darwin process's physical memory footprint when available."""
    if sys.platform != "darwin":
        return None
    import ctypes

    class RUsageInfoV2(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
        ]

    try:
        libproc = ctypes.CDLL("libproc.dylib", use_errno=True)
        query = libproc.proc_pid_rusage
        query.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(RUsageInfoV2),
        )
        query.restype = ctypes.c_int
        info = RUsageInfoV2()
        if query(pid, 2, ctypes.byref(info)) != 0:
            return None
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return max(int(info.ri_resident_size), int(info.ri_phys_footprint))


def _check_darwin_worker_memory(
    process: subprocess.Popen[str], memory_limit: int
) -> None:
    """Fail closed when a live Darwin worker cannot honor its memory ceiling."""
    if sys.platform != "darwin":
        return
    used = _darwin_process_memory_bytes(process.pid)
    if used is None:
        if process.poll() is not None:
            return
        process.kill()
        process.wait()
        raise TraceError("trace worker memory ceiling could not be enforced on macOS")
    if used <= memory_limit:
        return
    process.kill()
    process.wait()
    raise TraceError(
        f"trace worker exceeded the approved {memory_limit / _MIB:,.0f} MiB "
        "memory ceiling"
    )


class DeclaredValue(Protocol):
    """Anything declaring an element type and shape, e.g. ``ir.core.Value``."""

    @property
    def element_type(self) -> str | None: ...

    @property
    def shape(self) -> Sequence[int | str | None] | None: ...


def estimated_capture_bytes(values: Iterable[DeclaredValue]) -> int:
    """Estimated decoded bytes of capturing these declared values in full.

    Intended for the UI before any run exists: pass a session's declared
    graph values (or the subset a capture selection names) to preview how
    much activation data a trace would materialize. Values with symbolic or
    unknown dimensions, or element types without a fixed byte width,
    contribute nothing, so the estimate is a lower bound.
    """
    total = 0
    for value in values:
        size = declared_byte_size(value.element_type, value.shape)
        if size is not None:
            total += size
    return total


def _capture_pressure_note(
    targets: list[dict[str, object]],
    limits: TraceLimits,
    capture_policy: str = "greedy",
) -> str | None:
    """A pre-run warning — never a refusal — when the capture looks too big.

    Declared shapes are promises, not measurements, so exceeding half the
    approved worker memory is worth a note while staying well short of
    grounds to block an explicitly approved trace.
    """
    estimated = 0
    for target in targets:
        raw_shape = target.get("shape")
        size = declared_byte_size(
            str(target.get("element_type")),
            raw_shape if isinstance(raw_shape, list) else None,
        )
        if size is not None:
            estimated += size
    if capture_policy == "preview":
        # A preview trace stores equal shares of the capture pool, so by
        # construction it fits its capture budget; the declared total alone
        # must not fire the note for it.
        estimated = min(estimated, limits.capture_bytes)
    if estimated <= limits.memory_bytes // 2:
        return None
    estimated_mib = math.ceil(estimated / _MIB)
    limit_mib = math.ceil(limits.memory_bytes / _MIB)
    return (
        f"the selected capture decodes to an estimated {estimated_mib:,} MiB "
        f"of activations, more than half the approved {limit_mib:,} MiB "
        "memory limit; a full capture may exhaust it — capture fewer values "
        "or raise the Memory limit"
    )


def _parse_worker_response(
    raw: object,
) -> tuple[
    tuple[ActivationRecord, ...],
    str,
    tuple[str, ...],
    tuple[str, ...],
    TraceDevice,
    str,
]:
    """Decode the worker's response JSON into typed result components."""
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("records"), list)
        or not isinstance(raw.get("diagnostics"), list)
    ):
        raise ValueError("malformed response")
    # Workers predating the notes channel simply omit the key.
    raw_notes = raw.get("notes", [])
    if not isinstance(raw_notes, list):
        raise ValueError("malformed response")
    return (
        tuple(ActivationRecord.from_json(item) for item in raw["records"]),
        str(raw["runtime"]),
        tuple(str(item) for item in raw["diagnostics"]),
        tuple(str(item) for item in raw_notes),
        TraceDevice(str(raw["execution_device"])),
        str(raw["execution_provider"]),
    )


def _spawn_worker(
    scratch: Path,
    request_path: Path,
) -> tuple[subprocess.Popen[str], Path, Path]:
    """Start the isolated worker with its streams landing in files.

    The caller's poll loop never drains streams, so a pipe would block the
    child the moment its buffer fills — ONNX Runtime's provider banners
    alone can do it — and a blocked child is indistinguishable from a
    wall-clock overrun.
    """
    stdout_path = scratch / "worker-stdout.log"
    stderr_path = scratch / "worker-stderr.log"
    out_handle = open(stdout_path, "w", encoding="utf-8", errors="replace")
    err_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
    try:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "nneditor.tracing.trace_worker",
                    str(request_path),
                ],
                cwd=scratch,
                env=_worker_environment("NNEDITOR_TRACE_WORKER"),
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                text=True,
            )
        except OSError as error:
            raise TraceError(f"trace worker could not start: {error}") from error
    finally:
        # The child owns its stream descriptors from the moment it spawns;
        # the parent's copies would only pin the log files.
        out_handle.close()
        err_handle.close()
    return process, stdout_path, stderr_path


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def recommended_trace_limits(model_bytes: int) -> TraceLimits:
    """Return practical isolated-worker defaults for an artifact's size."""
    if model_bytes < 0:
        raise ValueError("model byte size cannot be negative")
    # ONNX first reads the serialized protobuf, then retains parsed tensors,
    # evaluator state, and live activations; a GPU runtime adds its own
    # host-side weight copies and pinned transfer buffers on top. Twelve
    # model-sized working sets plus headroom is what a measured full-graph
    # CUDA trace of the 1.33 GB Qwen vision tower actually needed — its
    # host commit exceeded the previous six-set estimate and the job cap
    # turned driver allocations into spurious CUDA out-of-memory failures.
    estimated_memory = model_bytes * 12 + _GIB + _DEFAULT_TRACE_CAPTURE_BYTES
    memory_bytes = max(
        _DEFAULT_TRACE_MEMORY_BYTES,
        _next_power_of_two(estimated_memory),
    )
    size_units = math.ceil(model_bytes / (256 * _MIB))
    # Accelerator sessions can spend minutes before the first inference runs
    # — TensorRT compiles an engine per model, CUDA loads kernels — so even a
    # small model needs a first-run allowance far beyond its execution time,
    # and the reference-evaluator fallback is slower still.
    wall_seconds = max(300.0, float(size_units * 60))
    return TraceLimits(
        wall_seconds=wall_seconds,
        memory_bytes=memory_bytes,
        capture_bytes=_DEFAULT_TRACE_CAPTURE_BYTES,
        chunk_bytes=_DEFAULT_TRACE_CHUNK_BYTES,
    )


def _minimum_model_load_bytes(model_bytes: int, input_bytes: int) -> int:
    # onnx.load_model temporarily needs both serialized and parsed protobuf
    # storage. This is a hard preflight floor, not the larger practical default.
    return model_bytes * 2 + input_bytes + 256 * _MIB


def _staged_model_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _targets(
    document: Document,
    selected: frozenset[str],
) -> list[dict[str, object]]:
    graph = document.main_graph
    role_by_value: dict[str, tuple[str | None, str]] = {
        value_id: (None, "graph-input")
        for value_id in graph.inputs
        if value_id not in graph.initializers
    }
    for node in graph.nodes:
        for value_id in node.outputs:
            role_by_value[value_id] = (node.id, "node-output")
    wanted = selected or frozenset(role_by_value)
    unknown = wanted - frozenset(value.id for value in graph.values)
    if unknown:
        raise TraceError(
            f"capture selection contains unknown values: {sorted(unknown)}"
        )
    targets: list[dict[str, object]] = []
    for value in graph.values:
        if value.id not in wanted or value.id not in role_by_value:
            continue
        if not value.name:
            if selected:
                raise TraceError(
                    f"selected capture value {value.id!r} has no serialized name"
                )
            # Omitted optional ONNX outputs are positional IR placeholders,
            # not runtime values that can be added to graph outputs.
            continue
        node_id, role = role_by_value[value.id]
        targets.append(
            {
                "value_id": value.id,
                "value_name": value.name,
                "node_id": node_id,
                "role": role,
                "graph_output": value.id in graph.outputs,
                "element_type": value.element_type or "unknown",
                "shape": [
                    dimension if isinstance(dimension, int) else -1
                    for dimension in value.shape or ()
                ],
            }
        )
    if not targets:
        raise TraceError("trace selection contains no executable graph values")
    return targets


def run_onnx_trace(
    document: Document,
    key: TraceKey,
    request: TraceRequest,
    store: ActivationStore,
    build_model: ModelBuilder,
    *,
    token: CancellationToken,
) -> TraceResult:
    """Export a working revision, run it out of process, then atomically adopt."""
    request.approval.validate(
        model_title=Path(document.source.path).name,
        artifact_hash=document.source.content_hash,
        specification=request.specification,
        limits=request.limits,
        backend=request.backend,
        device=request.device,
    )
    token.raise_if_cancelled()
    try:
        input_bytes = sum(
            math.prod(binding.shape) * np.dtype(binding.element_type).itemsize
            for binding in request.specification.bindings
        )
    except TypeError as error:
        raise TraceError(f"trace input dtype is unsupported: {error}") from error
    if input_bytes > request.limits.memory_bytes:
        raise TraceError(
            f"trace inputs require {input_bytes:,} bytes before runtime "
            f"overhead, exceeding the {request.limits.memory_bytes:,}-byte "
            "worker memory ceiling"
        )
    feeds = bind_inputs(document, request.specification, token=token)
    staging = store.begin_staging()
    scratch = staging / "scratch"
    scratch.mkdir()
    model_path = scratch / "model.onnx"
    inputs_path = scratch / "inputs.npz"
    request_path = scratch / "request.json"
    response_path = scratch / "response.json"
    try:
        build_model(model_path, token)
        token.raise_if_cancelled()
        model_bytes = _staged_model_bytes(scratch)
        minimum_memory = _minimum_model_load_bytes(model_bytes, input_bytes)
        if request.limits.memory_bytes < minimum_memory:
            recommended = recommended_trace_limits(model_bytes)
            raise TraceError(
                f"the approved trace memory limit is "
                f"{request.limits.memory_bytes / _MIB:,.0f} MiB, but loading "
                f"this {model_bytes / _MIB:,.0f} MiB model requires at least "
                f"{minimum_memory / _MIB:,.0f} MiB before evaluation; set "
                f"Memory limit to {recommended.memory_bytes // _MIB:,} MiB "
                "and approve the trace again"
            )
        np.savez(inputs_path, **feeds)  # type: ignore[arg-type]
        targets = _targets(document, request.value_ids)
        # Known pre-launch: warn (as a note, not a refusal) when the declared
        # capture set alone approaches the approved worker memory.
        pressure_note = _capture_pressure_note(
            targets,
            request.limits,
            request.capture_policy,
        )
        payload = {
            "model": str(model_path),
            "inputs": str(inputs_path),
            "output": str(staging),
            "response": str(response_path),
            "targets": targets,
            "backend": request.backend.value,
            "device": request.device.value,
            "capture_policy": request.capture_policy,
            **request.limits.to_json(),
        }
        request_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        process, stdout_path, stderr_path = _spawn_worker(scratch, request_path)
        job = _cap_worker_process(process, request.limits.memory_bytes)
        if os.name == "nt" and job is None:
            process.kill()
            process.wait()
            raise TraceError(
                "trace worker memory ceiling could not be enforced on Windows"
            )
        started = time.monotonic()
        try:
            while process.poll() is None:
                try:
                    token.raise_if_cancelled()
                except OperationCancelled:
                    process.kill()
                    process.wait()
                    raise
                _check_darwin_worker_memory(process, request.limits.memory_bytes)
                if time.monotonic() - started > request.limits.wall_seconds:
                    process.kill()
                    process.wait()
                    raise TraceError(
                        f"trace worker exceeded {request.limits.wall_seconds:g} seconds"
                    )
                time.sleep(0.02)
        finally:
            _close_job_object(job)
        token.raise_if_cancelled()
        if process.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            message = stderr.strip() or stdout.strip()
            raise TraceError(f"trace worker failed: {message or process.returncode}")
        try:
            (
                records,
                runtime,
                diagnostics,
                notes,
                execution_device,
                execution_provider,
            ) = _parse_worker_response(
                json.loads(response_path.read_text(encoding="utf-8"))
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise TraceError(
                f"trace worker returned an invalid response: {error}"
            ) from error
        if pressure_note is not None:
            notes = (*notes, pressure_note)
        token.raise_if_cancelled()
        shutil.rmtree(scratch)
        return store.commit(
            key,
            records,
            runtime,
            diagnostics,
            staging,
            execution_device=execution_device,
            execution_provider=execution_provider,
            notes=notes,
        )
    except BaseException:
        if staging.exists():
            store.discard_staging(staging)
        raise
