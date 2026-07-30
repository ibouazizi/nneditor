"""Parent-side orchestration for the isolated ONNX trace worker."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

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
)
from nneditor.tracing.inputs import bind_inputs
from nneditor.tracing.store import ActivationStore

__all__ = ["TraceError", "recommended_trace_limits", "run_onnx_trace"]

ModelBuilder = Callable[[Path, CancellationToken], None]
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_DEFAULT_TRACE_MEMORY_BYTES = 2 * _GIB
_DEFAULT_TRACE_CAPTURE_BYTES = 256 * _MIB
_DEFAULT_TRACE_CHUNK_BYTES = _MIB


class TraceError(RuntimeError):
    """An approved isolated trace could not produce a trustworthy result."""


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def recommended_trace_limits(model_bytes: int) -> TraceLimits:
    """Return practical isolated-worker defaults for an artifact's size."""
    if model_bytes < 0:
        raise ValueError("model byte size cannot be negative")
    # ONNX first reads the serialized protobuf, then retains parsed tensors,
    # evaluator state, and live activations. Six model-sized working sets plus
    # fixed headroom covers attention-heavy inline-weight models such as the
    # Qwen vision tower while keeping small-model defaults unchanged.
    estimated_memory = model_bytes * 6 + 512 * _MIB + _DEFAULT_TRACE_CAPTURE_BYTES
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
        payload = {
            "model": str(model_path),
            "inputs": str(inputs_path),
            "output": str(staging),
            "response": str(response_path),
            "targets": _targets(document, request.value_ids),
            "backend": request.backend.value,
            "device": request.device.value,
            **request.limits.to_json(),
        }
        request_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            raise TraceError(f"trace worker could not start: {error}") from error
        job = _cap_worker_process(process, request.limits.memory_bytes)
        if os.name == "nt" and job is None:
            process.kill()
            process.communicate()
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
                    process.communicate()
                    raise
                if time.monotonic() - started > request.limits.wall_seconds:
                    process.kill()
                    process.communicate()
                    raise TraceError(
                        f"trace worker exceeded {request.limits.wall_seconds:g} seconds"
                    )
                time.sleep(0.02)
            stdout, stderr = process.communicate()
        finally:
            _close_job_object(job)
        token.raise_if_cancelled()
        if process.returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise TraceError(f"trace worker failed: {message or process.returncode}")
        try:
            raw = json.loads(response_path.read_text(encoding="utf-8"))
            if (
                not isinstance(raw, dict)
                or not isinstance(raw.get("records"), list)
                or not isinstance(raw.get("diagnostics"), list)
            ):
                raise ValueError("malformed response")
            records = tuple(ActivationRecord.from_json(item) for item in raw["records"])
            runtime = str(raw["runtime"])
            diagnostics = tuple(str(item) for item in raw["diagnostics"])
            execution_device = TraceDevice(str(raw["execution_device"]))
            execution_provider = str(raw["execution_provider"])
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
        )
    except BaseException:
        if staging.exists():
            store.discard_staging(staging)
        raise
