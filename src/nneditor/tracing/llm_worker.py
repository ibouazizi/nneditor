"""Isolated subprocess entry point for GGUF and Hugging Face activation capture."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.huggingface import open_huggingface
from nneditor.ir.capabilities import ArtifactKind
from nneditor.storage.store import TensorStore
from nneditor.tracing.contracts import CAPTURE_POLICIES, TraceBackend, TraceDevice
from nneditor.tracing.llm_runtime import LlmTraceError, trace_document

_KIND_OPENERS = {
    ArtifactKind.GGUF_MODEL.value: open_gguf,
    ArtifactKind.HUGGINGFACE_MODEL.value: open_huggingface,
}


def _limit_process(raw: dict[str, object]) -> None:
    if os.name == "nt":
        return
    try:
        import resource as resource_module
    except ImportError as error:
        raise RuntimeError(
            "this platform cannot enforce trace worker resource limits"
        ) from error
    resource = cast(Any, resource_module)
    try:
        cpu = max(1, int(float(str(raw["wall_seconds"]))))
        memory = int(str(raw["memory_bytes"]))

        def cap(resource_id: int, requested: int) -> None:
            _soft, hard = resource.getrlimit(resource_id)
            resolved = (
                requested
                if hard == resource.RLIM_INFINITY
                else min(requested, int(hard))
            )
            resource.setrlimit(resource_id, (resolved, resolved))

        cap(resource.RLIMIT_CPU, cpu)
        if sys.platform != "darwin":
            cap(resource.RLIMIT_AS, memory)
        cap(resource.RLIMIT_NOFILE, 64)
    except (AttributeError, OSError, ValueError) as error:
        raise RuntimeError(
            "trace worker resource limits could not be enforced"
        ) from error


def _write_capture(
    output: Path,
    target: dict[str, object],
    array: np.ndarray[Any, Any] | None,
    *,
    remaining: int,
    chunk_bytes: int,
    reason: str | None,
) -> tuple[dict[str, object], int]:
    base = {
        "value_id": str(target["value_id"]),
        "value_name": str(target["value_name"]),
        "node_id": target.get("node_id"),
        "role": str(target["role"]),
    }
    if array is None:
        raw_shape = target["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError("trace target shape is malformed")
        return (
            {
                **base,
                "element_type": str(target["element_type"]),
                "numpy_dtype": str(target["element_type"]),
                "shape": list(raw_shape),
                "state": "dropped",
                "full_byte_length": 0,
                "stored_byte_length": 0,
                "file_name": None,
                "reason": reason or "the numpy LLM runtime produced no value",
            },
            remaining,
        )
    contiguous = np.ascontiguousarray(array)
    full_length = int(contiguous.nbytes)
    itemsize = max(1, int(contiguous.dtype.itemsize))
    stored = min(full_length, remaining)
    stored -= stored % itemsize
    if stored <= 0:
        return (
            {
                **base,
                "element_type": str(contiguous.dtype),
                "numpy_dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "state": "dropped",
                "full_byte_length": full_length,
                "stored_byte_length": 0,
                "file_name": None,
                "reason": "capture byte ceiling was exhausted before this value",
            },
            remaining,
        )
    file_name = (
        "captures/"
        + hashlib.sha256(str(target["value_id"]).encode()).hexdigest()
        + ".bin"
    )
    path = output / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    view = memoryview(contiguous).cast("B")
    with open(path, "wb") as handle:
        for offset in range(0, stored, chunk_bytes):
            handle.write(view[offset : min(stored, offset + chunk_bytes)])
        handle.flush()
        os.fsync(handle.fileno())
    truncated = stored < full_length
    return (
        {
            **base,
            "element_type": str(contiguous.dtype),
            "numpy_dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "state": "truncated" if truncated else "complete",
            "full_byte_length": full_length,
            "stored_byte_length": stored,
            "file_name": file_name,
            "reason": (
                "capture was truncated at the configured byte ceiling"
                if truncated
                else None
            ),
        },
        remaining - stored,
    )


def _load_feeds(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(str(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _select_backend(raw: dict[str, object]) -> tuple[str, TraceDevice, str]:
    try:
        backend = TraceBackend(str(raw.get("backend", TraceBackend.AUTO.value)))
    except ValueError as error:
        raise ValueError(f"unknown trace backend {raw.get('backend')!r}") from error
    try:
        device = TraceDevice(str(raw.get("device", TraceDevice.AUTO.value)))
    except ValueError as error:
        raise ValueError(f"unknown trace device {raw.get('device')!r}") from error
    if backend is TraceBackend.ONNX_RUNTIME:
        raise LlmTraceError(
            "ONNX Runtime is not a GGUF or Hugging Face backend; use auto or reference"
        )
    if device not in {TraceDevice.AUTO, TraceDevice.CPU}:
        raise LlmTraceError("the numpy LLM worker runs on CPU only")
    return "numpy-llm", TraceDevice.CPU, "numpy"


class CaptureWriter:
    """Write selected values as they are produced, retaining only metadata."""

    def __init__(
        self,
        output: Path,
        targets: list[dict[str, object]],
        *,
        capture_bytes: int,
        chunk_bytes: int,
        capture_policy: str,
    ) -> None:
        self.output = output
        self.targets = {str(target["value_id"]): target for target in targets}
        self.remaining = capture_bytes
        self.chunk_bytes = chunk_bytes
        self.preview = capture_policy == "preview"
        self.written: dict[str, dict[str, object]] = {}

    def capture(self, value_id: str, array: np.ndarray[Any, Any] | None) -> None:
        if value_id not in self.targets or value_id in self.written:
            return
        count = len(self.targets) - len(self.written)
        budget = self.remaining // count if self.preview else self.remaining
        record, unused = _write_capture(
            self.output,
            self.targets[value_id],
            array,
            remaining=budget,
            chunk_bytes=self.chunk_bytes,
            reason=None,
        )
        self.remaining -= budget - unused
        self.written[value_id] = record

    def records(self) -> list[dict[str, object]]:
        for value_id in self.targets:
            if value_id not in self.written:
                self.capture(value_id, None)
        return [self.written[value_id] for value_id in self.targets]


def run(request_path: Path) -> None:
    if os.environ.get("NNEDITOR_TRACE_WORKER") != "1":
        raise RuntimeError("trace worker must be launched by NNEditor")
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("targets"), list):
        raise ValueError("trace request is malformed")
    _limit_process(raw)
    capture_policy = str(raw.get("capture_policy", "greedy"))
    if capture_policy not in CAPTURE_POLICIES:
        raise ValueError(f"unknown capture policy {raw.get('capture_policy')!r}")
    runtime, execution_device, execution_provider = _select_backend(raw)
    opener = _KIND_OPENERS.get(str(raw.get("kind")))
    if opener is None:
        raise LlmTraceError(f"unsupported LLM artifact kind {raw.get('kind')!r}")
    document = opener(Path(str(raw["artifact"])))
    if document.source.content_hash != raw.get("artifact_hash"):
        raise LlmTraceError(
            "The model changed after approval. Reopen it before tracing."
        )
    feeds = _load_feeds(Path(str(raw["inputs"])))
    notes: list[str] = []
    if raw.get("revision_id"):
        raise LlmTraceError("LLM traces require unedited source weights.")
    targets = [dict(item) for item in raw["targets"] if isinstance(item, dict)]
    writer = CaptureWriter(
        Path(str(raw["output"])),
        targets,
        capture_bytes=int(raw["capture_bytes"]),
        chunk_bytes=int(raw["chunk_bytes"]),
        capture_policy=capture_policy,
    )
    with TensorStore(document) as store:
        _captures, runtime_notes, diagnostics = trace_document(
            document, feeds, store, capture=writer.capture
        )
    notes.extend(runtime_notes)
    records = writer.records()
    response = {
        "runtime": runtime,
        "execution_device": execution_device.value,
        "execution_provider": execution_provider,
        "records": records,
        "diagnostics": list(diagnostics),
        "notes": notes,
    }
    Path(str(raw["response"])).write_text(
        json.dumps(response, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nneditor.tracing.llm_worker REQUEST")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
