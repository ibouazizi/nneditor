"""Numpy LLM tracing for GGUF and Hugging Face snapshots."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.huggingface import open_huggingface
from nneditor.application.session import (
    ApplicationService,
    ModelSession,
    SessionError,
)
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.prompt_inputs import generate_prompt_token_tensor
from nneditor.storage.store import TensorStore
from nneditor.tracing import (
    TraceApproval,
    TraceBackend,
    TraceDevice,
    TraceLimits,
    TraceRequest,
)
from nneditor.tracing.contracts import InputBinding, InputSource, InputSpecification
from nneditor.tracing.llm_runtime import (
    LlmTraceError,
    hyperparams_from_document,
    trace_document,
)
from nneditor.tracing.runner import recommended_trace_limits
from tests.fixtures.gguf_models import write_traceable_llama_gguf
from tests.fixtures.gguf_tokenizers import write_tokenized_gguf
from tests.fixtures.huggingface_models import (
    write_llama_snapshot,
    write_traceable_llama_snapshot,
)


def _limits() -> TraceLimits:
    return TraceLimits(
        wall_seconds=20,
        memory_bytes=2 * 1024 * 1024 * 1024,
        capture_bytes=1024 * 1024,
        chunk_bytes=64,
    )


def test_traceable_gguf_declares_tracing(tmp_path: Path) -> None:
    path = write_traceable_llama_gguf(tmp_path / "model.gguf")
    document = open_gguf(path)
    tracing = document.capability(Capability.TRACING).availability
    execution = document.capability(Capability.EXECUTION).availability
    assert tracing is Availability.AVAILABLE
    assert execution is Availability.PARTIAL
    params = hyperparams_from_document(document)
    assert params.n_embd == 4
    assert params.n_head == 2
    assert params.n_kv_head == 2
    assert params.hidden_act == "silu"


def test_traceable_huggingface_declares_tracing(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    document = open_huggingface(root)
    assert document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL
    tracing = document.capability(Capability.TRACING).availability
    assert tracing is Availability.AVAILABLE
    params = hyperparams_from_document(document)
    assert params.n_embd == 4
    assert params.n_head == 2
    assert params.hidden_act == "silu"


def test_incomplete_huggingface_snapshot_explains_unavailable_tracing(
    tmp_path: Path,
) -> None:
    root = write_llama_snapshot(tmp_path / "partial")
    document = open_huggingface(root)
    tracing = document.capability(Capability.TRACING).availability
    assert tracing is Availability.UNAVAILABLE
    assert "missing" in document.capability(Capability.TRACING).reason


def test_numpy_forward_captures_each_synthesized_operator(tmp_path: Path) -> None:
    path = write_traceable_llama_gguf(tmp_path / "model.gguf")
    document = open_gguf(path)
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    with TensorStore(document) as store:
        captures, notes, diagnostics = trace_document(
            document, {"tokens": tokens}, store
        )
    assert diagnostics == ()
    names = {document.main_graph.value(value_id).name for value_id in captures}
    assert "tokens" in names
    assert "embed" in names
    assert "block-000-attn" in names
    assert "block-000-ffn" in names
    assert "output-norm" in names
    assert "lm-head" in names
    logits = next(
        array
        for value_id, array in captures.items()
        if document.main_graph.value(value_id).name == "lm-head"
    )
    assert logits.shape == (1, 3, 8)
    assert np.isfinite(logits).all()
    assert notes == ()


def test_missing_ffn_up_projection_is_a_clear_error(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "incomplete")
    document = open_huggingface(root)
    tokens = np.array([[1, 2]], dtype=np.int64)
    with TensorStore(document) as store, pytest.raises(LlmTraceError, match="missing"):
        trace_document(document, {"tokens": tokens}, store)


def _approved_request(
    session: ModelSession, specification: InputSpecification
) -> TraceRequest:
    limits = _limits()
    approval = TraceApproval.approve(
        session.title,
        session.document.source.content_hash,
        specification,
        limits,
    )
    return TraceRequest(specification, limits, approval)


def test_isolated_gguf_trace_captures_activations(tmp_path: Path) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    os.environ.pop("NNEDITOR_TRACE_WORKER", None)
    with ApplicationService() as service:
        session = service.open_model(path)
        generated = generate_prompt_token_tensor(
            tmp_path / "tokens.npy", document=session.document, text="Hello world!"
        )
        binding = session.trace_tensor_input("tokens", generated.path)
        specification = session.default_trace_inputs(bindings={"tokens": binding})
        result = session.trace_async(_approved_request(session, specification)).result(
            timeout=20
        )
        assert result.runtime == "numpy-llm"
        assert result.execution_device is TraceDevice.CPU
        assert result.execution_provider == "numpy"
        names = {record.value_name for record in result.records}
        assert {"tokens", "embed", "block-000-attn", "lm-head"} <= names
        assert all(record.readable for record in result.records)
        assert os.environ.get("NNEDITOR_TRACE_WORKER") is None
        logits = next(
            record for record in result.records if record.value_name == "lm-head"
        )
        payload = session.activations(result.id).read(logits.value_id)
        array = np.frombuffer(payload, dtype=np.float32).reshape(logits.shape)
        assert np.isfinite(array).all()


def test_isolated_huggingface_trace_captures_activations(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    with ApplicationService() as service:
        session = service.open_model(root)
        generated = generate_prompt_token_tensor(
            tmp_path / "tokens.npy", document=session.document, prompt_id="primes"
        )
        binding = session.trace_tensor_input("tokens", generated.path)
        specification = session.default_trace_inputs(bindings={"tokens": binding})
        result = session.trace_async(_approved_request(session, specification)).result(
            timeout=20
        )
        assert result.runtime == "numpy-llm"
        names = {record.value_name for record in result.records}
        assert "lm-head" in names
        assert all(record.readable for record in result.records)


def test_onnx_runtime_backend_is_refused_for_gguf(tmp_path: Path) -> None:
    path = write_traceable_llama_gguf(tmp_path / "model.gguf")
    with ApplicationService() as service:
        session = service.open_model(path)
        tokens = np.array([[1, 2]], dtype=np.int64)
        token_path = tmp_path / "ids.npy"
        np.save(token_path, tokens)
        binding = session.trace_tensor_input("tokens", token_path)
        specification = session.default_trace_inputs(bindings={"tokens": binding})
        limits = _limits()
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            specification,
            limits,
            backend=TraceBackend.ONNX_RUNTIME,
        )
        request = TraceRequest(
            specification,
            limits,
            approval,
            backend=TraceBackend.ONNX_RUNTIME,
        )
        with pytest.raises(Exception, match="ONNX Runtime is not a GGUF"):
            session.trace_async(request).result(timeout=20)


def test_streaming_trace_limits_are_tighter_than_onnx_full_load() -> None:
    model_bytes = 1_326_283_019
    onnx_limits = recommended_trace_limits(model_bytes)
    llm_limits = recommended_trace_limits(model_bytes, streaming_weights=True)
    assert llm_limits.memory_bytes < onnx_limits.memory_bytes
    assert llm_limits.memory_bytes >= 2 * 1024 * 1024 * 1024


def test_pickle_snapshot_does_not_advertise_tracing(tmp_path: Path) -> None:
    root = tmp_path / "pickle"
    root.mkdir()
    (root / "config.json").write_text(
        '{"model_type": "llama", "num_hidden_layers": 1}', encoding="utf-8"
    )
    (root / "pytorch_model.bin").write_bytes(b"not a pickle")
    document = open_huggingface(root)
    tracing = document.capability(Capability.TRACING).availability
    assert tracing is Availability.UNAVAILABLE
    with ApplicationService() as service:
        session = service.open_model(root)
        specification = InputSpecification(
            session.document.source.content_hash,
            (
                InputBinding(
                    "tokens",
                    "int64",
                    (1, 1),
                    InputSource.CONSTANT,
                    constant=(1,),
                ),
            ),
        )
        limits = _limits()
        with pytest.raises(SessionError):
            session.trace_async(
                TraceRequest(
                    specification,
                    limits,
                    TraceApproval.approve(
                        session.title,
                        session.document.source.content_hash,
                        specification,
                        limits,
                    ),
                )
            )
