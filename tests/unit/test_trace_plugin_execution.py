"""Real extension workers, capture correctness, approval binding, and failures."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.application.session import ApplicationService, ModelSession, SessionError
from nneditor.cancellation import OperationCancelled
from nneditor.ir.capabilities import Availability, Capability
from nneditor.storage.store import TensorStore
from nneditor.tracing import plugin_worker
from nneditor.tracing.contracts import (
    TraceApproval,
    TraceDevice,
    TraceLimits,
    TraceRequest,
    TraceResult,
)
from nneditor.tracing.llm_runtime import trace_document
from nneditor.tracing.plugin_manifest import MANIFEST_NAME, load_manifest
from nneditor.tracing.runner import TraceError, _targets
from nneditor.tracing.sdk import TraceContext, TracePluginError
from nneditor.tracing.store import ActivationStore
from tests.fixtures.qwen35_models import write_qwen35
from tests.fixtures.tracing_plugins import (
    LINEAR_PLUGIN,
    write_linear_gguf,
    write_trace_plugin,
)


def _request(
    session: ModelSession,
    root: Path,
    identity: str,
    *,
    seconds: float = 20,
    capture_bytes: int = 1024**2,
) -> TraceRequest:
    path = root / "tokens.npy"
    np.save(path, np.array([[0, 2, 1]], dtype=np.int64))
    binding = session.trace_tensor_input("tokens", path)
    spec = session.default_trace_inputs(bindings={"tokens": binding})
    limits = TraceLimits(
        wall_seconds=seconds,
        memory_bytes=2**31,
        capture_bytes=capture_bytes,
        chunk_bytes=11,
    )
    approval = TraceApproval.approve(
        session.title,
        session.document.source.content_hash,
        spec,
        limits,
        tracer=identity,
    )
    return TraceRequest(spec, limits, approval, tracer=identity)


def test_new_architecture_is_traceable_without_changing_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "plugin-imported"
    prefix = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('worker imported')\n"
    )
    root = write_trace_plugin(tmp_path / "plugin", code=prefix + LINEAR_PLUGIN)
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        assert (
            session.document.capability(Capability.TRACING).availability
            is Availability.UNAVAILABLE
        )
        assert (
            session.capability(Capability.TRACING).availability
            is Availability.AVAILABLE
        )
        assert not marker.exists()
        request = _request(session, tmp_path, plugin.identity)
        result = session.trace_async(request).result(timeout=25)
        assert marker.exists()
        assert result.runtime == plugin.id and result.execution_provider == "test-numpy"
        assert result.execution_device is TraceDevice.CPU
        assert result.key.tracer == plugin.identity
        assert all(r.readable for r in result.records) and len(result.records) == 3
        head = next(r for r in result.records if r.value_name == "lm-head")
        actual = np.frombuffer(
            session.activations(result.id).read(head.value_id), np.float32
        ).reshape(head.shape)
        np.testing.assert_array_equal(actual, [[[1, 2, 8], [5, 6, 28], [3, 4, 18]]])
        assert TraceResult.from_json(result.to_json()) == result
        different = replace(result.key, tracer="another@0.1.0")
        assert different.id != result.id
        with pytest.raises(ValueError, match="version 4"):
            replace(result.key, identity_version=3)


def test_extension_identity_is_required_in_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    plugin = load_manifest(root / MANIFEST_NAME)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        request = _request(session, tmp_path, plugin.identity)
        mismatched = replace(request, approval=replace(request.approval, tracer=None))
        with pytest.raises(ValueError, match="tracer extension"):
            session.trace_async(mismatched).result(timeout=25)
        automatic = replace(
            request, tracer=None, approval=replace(request.approval, tracer=None)
        )
        with pytest.raises(SessionError, match="approve its identity"):
            session.trace_async(automatic)
        assert not session.traces()


def test_plugin_update_invalidates_approval_and_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    plugin = load_manifest(root / MANIFEST_NAME)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        request = _request(session, tmp_path, plugin.identity)
        (root / "helper.py").write_text("SCALE = 2\n")
        with pytest.raises(TracePluginError, match="changed after approval"):
            session.trace_async(request).result(timeout=25)
        session.refresh_tracers()
        with pytest.raises(SessionError, match="changed or was removed"):
            session.trace_async(request)
        assert not session.traces()


@pytest.mark.parametrize(
    "code,message",
    [
        (
            "def trace(ctx):\n    raise RuntimeError('deliberate extension failure')\n",
            "deliberate extension failure",
        ),
        (
            "def trace(ctx):\n    ctx.capture('typo', __import__('numpy').zeros(1))\n",
            "must identify one graph value",
        ),
        ("trace = 7\n", "is not callable"),
        ("def trace(ctx):\n    return {}\n", "must return TraceOutcome or None"),
        ("import nneditor_nonexistent_dependency\n", "nneditor_nonexistent_dependency"),
    ],
)
def test_failures_publish_no_partial_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, message: str
) -> None:
    root = write_trace_plugin(tmp_path / "plugin", code=code)
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    store = ActivationStore(tmp_path / "activations")
    with ApplicationService(prewarm_layouts=False, activation_store=store) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        with pytest.raises(TraceError, match=message):
            session.trace_async(_request(session, tmp_path, plugin.identity)).result(
                timeout=25
            )
        assert not session.traces()
        assert not any(store.root.iterdir())


def test_preview_capture_budget_is_owned_by_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        request = replace(
            _request(session, tmp_path, plugin.identity, capture_bytes=24),
            capture_policy="preview",
        )
        result = session.trace_async(request).result(timeout=25)
        assert sum(r.stored_byte_length for r in result.records) <= 24
        assert all(r.readable for r in result.records)
        assert result.partial


def test_cancellation_terminates_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "started"
    code = f"""import time
from pathlib import Path
def trace(ctx):
    Path({str(marker)!r}).touch()
    time.sleep(60)
"""
    root = write_trace_plugin(tmp_path / "plugin", code=code)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    plugin = load_manifest(root / MANIFEST_NAME)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        job = session.trace_async(_request(session, tmp_path, plugin.identity))
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        job.cancel()
        with pytest.raises(OperationCancelled):
            job.result(timeout=10)
        assert not session.traces()


def test_worker_caps_and_protocol_checks_precede_plugin_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        request = _request(session, tmp_path, plugin.identity)
        inputs = tmp_path / "inputs.npz"
        np.savez(inputs, tokens=np.array([[0, 2, 1]], dtype=np.int64))
        raw: dict[str, Any] = {
            "artifact": session.document.source.path,
            "kind": "gguf_model",
            "artifact_hash": session.document.source.content_hash,
            "inputs": str(inputs),
            "output": str(tmp_path / "output"),
            "response": str(tmp_path / "response.json"),
            "targets": _targets(session.document, frozenset()),
            "tracer": plugin.identity,
            "tracer_manifest": str(plugin.manifest),
            **request.limits.to_json(),
        }
        path = tmp_path / "request.json"
        path.write_text(json.dumps(raw))
        monkeypatch.delenv("NNEDITOR_TRACE_WORKER", raising=False)
        with pytest.raises(TracePluginError, match="launched by NNEditor"):
            plugin_worker.run(path)
        monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
        capped: list[bool] = []
        monkeypatch.setattr(
            plugin_worker, "_limit_process", lambda _: capped.append(True)
        )
        original = plugin_worker._load_function

        def load(p: Any) -> Any:
            assert capped
            return original(p)

        monkeypatch.setattr(plugin_worker, "_load_function", load)
        plugin_worker.run(path)
        assert (
            json.loads((tmp_path / "response.json").read_text())["runtime"] == plugin.id
        )
        for key, value, match in (
            ("artifact_hash", "different", "model changed"),
            ("tracer", "different", "Tracer files changed"),
            ("revision_id", "edited", "unedited source"),
            ("device", "gpu", "does not support device"),
            ("backend", "onnxruntime", "chooses its own"),
            ("capture_policy", "bad", "Unknown capture policy"),
        ):
            path.write_text(json.dumps({**raw, key: value}))
            with pytest.raises(TracePluginError, match=match):
                plugin_worker.run(path)


def test_bonsai_builtin_through_extension_host(tmp_path: Path) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=143)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        plugin = session.trace_registry.preferred(session.document)
        assert plugin is not None and plugin.id == "builtin.numpy-llm"
        result = session.trace_async(
            _request(session, tmp_path, plugin.identity)
        ).result(timeout=25)
        head = next(r for r in result.records if r.value_name == "lm-head")
        actual = np.frombuffer(
            session.activations(result.id).read(head.value_id), np.float32
        ).reshape(head.shape)
        with TensorStore(session.document) as tensors:
            expected, _, _ = trace_document(
                session.document, {"tokens": np.array([[0, 2, 1]])}, tensors
            )
        np.testing.assert_allclose(actual, expected[head.value_id], atol=2e-5)
        assert result.runtime == "numpy-llm"


def test_sdk_context_rejects_bad_arrays_and_resolves_weights(tmp_path: Path) -> None:
    from nneditor.adapters.gguf import open_gguf

    document = open_gguf(write_linear_gguf(tmp_path / "model.gguf"))
    emitted: list[str] = []
    with TensorStore(document) as tensors:
        ctx = TraceContext(
            document,
            {"tokens": np.array([[1]])},
            tensors,
            lambda key, _array: emitted.append(key),
            limits=TraceLimits(),
        )
        assert ctx.metadata["general.architecture"] == "plugin_linear"
        ctx.capture_inputs()
        assert len(emitted) == 1
        np.testing.assert_array_equal(
            ctx.load_weight("token_embd.weight"), [[1, 2], [3, 4], [5, 6]]
        )
        for name in ("missing", "missing.weight"):
            with pytest.raises(TracePluginError, match="one tensor"):
                ctx.load_weight(name)
        for bad in (np.array([object()]), np.array(["text"]), np.array([1j])):
            with pytest.raises(TracePluginError, match="numeric arrays"):
                ctx.capture("embed", bad)
        empty = TraceContext(document, {}, tensors, ctx.capture, limits=TraceLimits())
        with pytest.raises(TracePluginError, match="Missing input"):
            empty.capture_inputs()


def test_onnx_can_use_an_external_tracer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fixtures.onnx_models import build_embedded_model

    code = """def trace(context):
    context.capture_inputs()
    scaled = context.inputs["input"] * context.load_weight("weight")
    context.capture("scaled", scaled)
    context.capture("output", scaled + context.load_weight("bias"))
"""
    root = write_trace_plugin(
        tmp_path / "plugin",
        code=code,
        artifact_kinds=["onnx_model"],
        architectures=[],
        operators=["Mul", "Add"],
    )
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    path = tmp_path / "model.onnx"
    weights = build_embedded_model(path, elements=4)
    values = np.array([-2, -1, 2, 3], dtype=np.float32)
    np.save(tmp_path / "input.npy", values)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        binding = session.trace_tensor_input("input", tmp_path / "input.npy")
        spec = session.default_trace_inputs(bindings={"input": binding})
        limits = TraceLimits(wall_seconds=20, memory_bytes=2**31)
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            spec,
            limits,
            tracer=plugin.identity,
        )
        request = TraceRequest(spec, limits, approval, tracer=plugin.identity)
        result = session.trace_async(request).result(timeout=25)
        assert len(result.records) == 3 and all(r.readable for r in result.records)
        output = next(r for r in result.records if r.value_name == "output")
        actual = np.frombuffer(
            session.activations(result.id).read(output.value_id), np.float32
        )
        np.testing.assert_array_equal(actual, values * weights + 0.5)
