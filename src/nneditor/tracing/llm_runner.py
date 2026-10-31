"""Parent-side orchestration for the isolated GGUF/Hugging Face LLM worker."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from nneditor.cancellation import CancellationToken
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document
from nneditor.tracing.contracts import (
    TraceKey,
    TraceRequest,
    TraceResult,
)
from nneditor.tracing.inputs import bind_inputs
from nneditor.tracing.llm_support import validate_llm_trace
from nneditor.tracing.runner import (
    TraceError,
    _capture_pressure_note,
    _run_capped_worker,
    _targets,
    recommended_trace_limits,
)
from nneditor.tracing.store import ActivationStore

__all__ = ["run_llm_trace"]

_MIB = 1024 * 1024
_WORKER_MODULE = "nneditor.tracing.llm_worker"
_LLM_KINDS = frozenset({ArtifactKind.GGUF_MODEL, ArtifactKind.HUGGINGFACE_MODEL})


def run_llm_trace(
    document: Document,
    key: TraceKey,
    request: TraceRequest,
    store: ActivationStore,
    *,
    token: CancellationToken,
) -> TraceResult:
    """Trace a GGUF or Hugging Face snapshot in an isolated numpy worker."""
    if document.artifact_kind not in _LLM_KINDS:
        raise TraceError(
            f"{document.artifact_kind.value} has no numpy LLM trace worker"
        )
    request.approval.validate(
        model_title=Path(document.source.path).name,
        artifact_hash=document.source.content_hash,
        specification=request.specification,
        limits=request.limits,
        backend=request.backend,
        device=request.device,
    )
    token.raise_if_cancelled()
    validate_llm_trace(document)
    if key.revision_id is not None:
        raise TraceError(
            "LLM traces require unedited source weights; undo edits before tracing."
        )
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
    inputs_path = scratch / "inputs.npz"
    request_path = scratch / "request.json"
    response_path = scratch / "response.json"
    try:
        token.raise_if_cancelled()
        minimum_memory = input_bytes + 64 * _MIB
        if request.limits.memory_bytes < minimum_memory:
            recommended = recommended_trace_limits(
                document.source.byte_size, streaming_weights=True
            )
            raise TraceError(
                f"the approved trace memory limit is "
                f"{request.limits.memory_bytes / _MIB:,.0f} MiB, which is too "
                "small for the numpy LLM worker; set Memory limit to "
                f"{recommended.memory_bytes // _MIB:,} MiB and approve the "
                "trace again"
            )
        np.savez(inputs_path, **feeds)  # type: ignore[arg-type]
        targets = _targets(document, request.value_ids)
        pressure_note = _capture_pressure_note(
            targets,
            request.limits,
            request.capture_policy,
        )
        payload = {
            "artifact": document.source.path,
            "kind": document.artifact_kind.value,
            "artifact_hash": document.source.content_hash,
            "inputs": str(inputs_path),
            "output": str(staging),
            "response": str(response_path),
            "targets": targets,
            "backend": request.backend.value,
            "device": request.device.value,
            "capture_policy": request.capture_policy,
            "revision_id": key.revision_id,
            **request.limits.to_json(),
        }
        request_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        (
            records,
            runtime,
            diagnostics,
            notes,
            execution_device,
            execution_provider,
        ) = _run_capped_worker(
            scratch,
            request_path,
            response_path,
            request.limits,
            token,
            module=_WORKER_MODULE,
        )
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
