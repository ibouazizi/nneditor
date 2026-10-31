"""Worker protocol, bounded capture, and failure behavior without process limits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.huggingface import open_huggingface
from nneditor.tracing import llm_worker
from nneditor.tracing.llm_runtime import LlmTraceError
from nneditor.tracing.runner import _targets
from tests.fixtures.huggingface_models import write_traceable_llama_snapshot


def _request(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    document = open_huggingface(root)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, tokens=np.array([[1, 2, 3]], dtype=np.int64))
    payload: dict[str, Any] = {
        "artifact": str(root),
        "kind": document.artifact_kind.value,
        "artifact_hash": document.source.content_hash,
        "inputs": str(inputs),
        "output": str(tmp_path / "output"),
        "response": str(tmp_path / "response.json"),
        "targets": _targets(document, frozenset()),
        "backend": "auto",
        "device": "cpu",
        "capture_bytes": 64,
        "chunk_bytes": 7,
        "capture_policy": "preview",
        "memory_bytes": 2**30,
        "wall_seconds": 10,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_worker_writes_bounded_response_and_rejects_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _request(tmp_path)
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(llm_worker, "_limit_process", lambda _raw: None)
    llm_worker.run(path)
    response = json.loads(Path(payload["response"]).read_text())
    assert response["runtime"] == "numpy-llm"
    assert response["execution_device"] == "cpu"
    assert sum(record["stored_byte_length"] for record in response["records"]) <= 64
    assert all(record["state"] == "truncated" for record in response["records"])
    payload["artifact_hash"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(LlmTraceError, match="changed after approval"):
        llm_worker.run(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("backend", "onnxruntime", "ONNX Runtime"),
        ("backend", "bogus", "unknown trace backend"),
        ("device", "gpu", "CPU only"),
        ("device", "bogus", "unknown trace device"),
        ("kind", "unknown", "unsupported LLM artifact"),
        ("capture_policy", "unknown", "capture policy"),
        ("revision_id", "edited", "unedited source"),
        ("targets", "malformed", "request is malformed"),
    ],
)
def test_worker_rejects_incompatible_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    path, payload = _request(tmp_path)
    payload[field] = value
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(llm_worker, "_limit_process", lambda _raw: None)
    with pytest.raises((LlmTraceError, ValueError), match=message):
        llm_worker.run(path)


def test_worker_requires_parent_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NNEDITOR_TRACE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="launched by NNEditor"):
        llm_worker.run(tmp_path / "nonexistent.json")


def test_missing_capture_has_explicit_dropped_record(tmp_path: Path) -> None:
    target: dict[str, object] = {
        "value_id": "missing",
        "value_name": "missing",
        "role": "node-output",
        "shape": [1],
        "element_type": "float32",
    }
    writer = llm_worker.CaptureWriter(
        tmp_path, [target], capture_bytes=32, chunk_bytes=4, capture_policy="greedy"
    )
    record = writer.records()[0]
    assert record["state"] == "dropped"
    assert "produced no value" in str(record["reason"])
    assert not list(tmp_path.iterdir())


def test_worker_does_not_load_pickle_feeds(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, tokens=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        llm_worker._load_feeds(path)


def test_worker_resource_caps_are_applied_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = pytest.importorskip("resource")

    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _id: (-1, -1))
    monkeypatch.setattr(
        resource, "setrlimit", lambda number, values: calls.append((number, values))
    )
    monkeypatch.setattr(llm_worker, "sys", SimpleNamespace(platform="linux"))
    llm_worker._limit_process({"wall_seconds": 10.0, "memory_bytes": 2**30})
    assert (resource.RLIMIT_AS, (2**30, 2**30)) in calls
    assert (resource.RLIMIT_CPU, (10, 10)) in calls
    assert (resource.RLIMIT_NOFILE, (64, 64)) in calls

    def cannot_cap(*_args: Any) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(resource, "setrlimit", cannot_cap)
    with pytest.raises(RuntimeError, match="could not be enforced"):
        llm_worker._limit_process({"wall_seconds": 10.0, "memory_bytes": 2**30})
