"""LLM import, cache confinement, identity, tokenizer and capture regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.huggingface import open_huggingface
from nneditor.adapters.pytorch.safetensors import (
    index_safetensors_file,
    write_safetensors,
)
from nneditor.application.session import ApplicationService
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import Availability, Capability
from nneditor.prompt_inputs import codebook_for_document, generate_prompt_token_tensor
from nneditor.storage.paths import UnsafePathError, resolve_snapshot_member
from nneditor.storage.store import TensorStore
from nneditor.tokenization import WordHashCodebook, _byte_encoder
from nneditor.tracing.llm_runtime import LlmTraceError, trace_document
from nneditor.tracing.llm_worker import CaptureWriter
from tests.fixtures.gguf_models import write_traceable_llama_gguf
from tests.fixtures.huggingface_models import (
    write_sharded_snapshot,
    write_traceable_llama_snapshot,
)


def test_weight_and_bias_bindings_have_unique_names(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    path = root / "model.safetensors"
    entries, _, _ = index_safetensors_file(path)
    data = path.read_bytes()
    tensors = [
        (e.name, e.element_type, e.dims, data[e.offset : e.offset + e.length])
        for e in entries
    ]
    tensors.append(("model.layers.0.self_attn.q_proj.bias", "float32", (4,), bytes(16)))
    path.unlink()
    write_safetensors(path, tensors)
    document = open_huggingface(root)
    attention = next(
        node for node in document.main_graph.nodes if node.op_type == "Attention"
    )
    names = [attribute.name for attribute in attention.attributes]
    assert len(names) == len(set(names))
    assert any(name.endswith("q_proj.bias") for name in names)


def test_cache_snapshot_links_are_readable_without_leaving_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "models--example--tiny"
    root = write_sharded_snapshot(repository / "snapshots" / "revision")
    blobs = repository / "blobs"
    blobs.mkdir()
    for child in list(root.iterdir()):
        blob = blobs / hashlib.sha256(child.read_bytes()).hexdigest()
        child.rename(blob)
        child.symlink_to(Path("../../blobs") / blob.name)
    document = open_huggingface(root)
    assert len(document.tensors) == 4
    with TensorStore(document) as store:
        assert all(
            len(store.read(tensor.id)) == tensor.external.length
            for tensor in document.tensors.values()
            if tensor.external
        )
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(bytes(32))
    unsafe = root / "evil.safetensors"
    unsafe.symlink_to(outside)
    with pytest.raises(UnsafePathError):
        resolve_snapshot_member(root, unsafe.name)
    for location in (
        str(outside),
        "../blobs/anything",
        "C:\\secret",
        "\\\\server\\share",
    ):
        with pytest.raises(UnsafePathError):
            resolve_snapshot_member(root, location)


def test_same_size_header_changes_invalidate_identity(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    before = open_huggingface(root)
    path = root / "model.safetensors"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"model.norm.weight", b"model.norm.weighX"))
    after = open_huggingface(root)
    assert path.stat().st_size == len(original)
    assert set(before.tensors) != set(after.tensors)
    assert before.source.content_hash != after.source.content_hash


@pytest.mark.parametrize("kind", ["missing-bytes", "metaspace", "too-large"])
def test_incompatible_tokenizer_uses_in_range_fallback(
    tmp_path: Path, kind: str
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    vocab = (
        {f"word{i}": i for i in range(300)}
        if kind == "missing-bytes"
        else {value: key for key, value in _byte_encoder().items()}
    )
    payload: dict[str, Any] = {"model": {"type": "BPE", "vocab": vocab, "merges": []}}
    if kind == "metaspace":
        payload["pre_tokenizer"] = {"type": "Metaspace"}
    (root / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    document = open_huggingface(root)
    codebook = codebook_for_document(document)
    assert isinstance(codebook, WordHashCodebook)
    generated = generate_prompt_token_tensor(tmp_path / "tokens.npy", document=document)
    array = np.load(generated.path, allow_pickle=False)
    assert int(array.min()) >= 0 and int(array.max()) < 8


def test_complete_byte_tokenizer_is_selected(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    config = json.loads((root / "config.json").read_text())
    config["vocab_size"] = 256
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {value: key for key, value in _byte_encoder().items()},
                    "merges": [],
                },
                "pre_tokenizer": {"type": "ByteLevel"},
            }
        )
    )
    codebook = codebook_for_document(open_huggingface(root))
    assert codebook.name == "tokenizer-json"
    assert codebook.encode("hello") == tuple(b"hello")


def test_unsupported_architecture_disables_trace_before_weight_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    config = json.loads((root / "config.json").read_text())
    config["model_type"] = "mamba"
    (root / "config.json").write_text(json.dumps(config))

    def unexpected_read(*_args: Any, **_kwargs: Any) -> bytes:
        pytest.fail("preflight must not read weight payloads")

    monkeypatch.setattr(TensorStore, "read", unexpected_read)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(root)
        status = session.capability(Capability.TRACING)
        assert status.availability is Availability.UNAVAILABLE
        assert "mamba" in status.reason
        assert "inspect" in status.reason


@pytest.mark.parametrize("policy", ["greedy", "preview"])
def test_streaming_capture_obeys_selection_and_byte_ceiling(
    tmp_path: Path, policy: str
) -> None:
    targets: list[dict[str, object]] = [
        {
            "value_id": key,
            "value_name": key,
            "role": "node-output",
            "element_type": "float32",
            "shape": [16],
        }
        for key in ("first", "second")
    ]
    writer = CaptureWriter(
        tmp_path, targets, capture_bytes=32, chunk_bytes=7, capture_policy=policy
    )
    array = np.arange(16, dtype=np.float32)
    writer.capture("unselected", array)
    writer.capture("first", array)
    writer.capture("second", array)
    records = writer.records()
    assert len(records) == 2
    assert sum(int(str(record["stored_byte_length"])) for record in records) == 32
    for record in records:
        if file_name := record["file_name"]:
            data = (tmp_path / str(file_name)).read_bytes()
            assert data == array.tobytes()[: len(data)]
    if policy == "preview":
        assert all(record["state"] == "truncated" for record in records)


@pytest.mark.parametrize("kind", ["hf", "gguf"])
def test_llm_import_cancellation_is_preserved(tmp_path: Path, kind: str) -> None:
    token = CancellationToken()
    token.cancel()
    if kind == "hf":
        root = write_traceable_llama_snapshot(tmp_path / "hf")
        with pytest.raises(OperationCancelled):
            open_huggingface(root, token=token)
    else:
        path = write_traceable_llama_gguf(tmp_path / "model.gguf")
        with pytest.raises(OperationCancelled):
            open_gguf(path, token=token)


@pytest.mark.parametrize(
    "array,message",
    [
        (np.array([[1.0]]), "integer ids"),
        (np.array([[[1]]]), "shape"),
        (np.array([[]], dtype=np.int64), "at least one token"),
        (np.array([[-1]]), "outside"),
        (np.array([[8]]), "outside"),
    ],
)
def test_invalid_prompt_tensors_fail_clearly(
    tmp_path: Path, array: np.ndarray, message: str
) -> None:
    document = open_huggingface(write_traceable_llama_snapshot(tmp_path / "hf"))
    with TensorStore(document) as store, pytest.raises(LlmTraceError, match=message):
        trace_document(document, {"tokens": array}, store)


def test_missing_prompt_feed_is_explained(tmp_path: Path) -> None:
    document = open_huggingface(write_traceable_llama_snapshot(tmp_path / "hf"))
    with (
        TensorStore(document) as store,
        pytest.raises(LlmTraceError, match="do not include"),
    ):
        trace_document(document, {}, store)


def test_unindexed_shards_merge_and_report_duplicate_names(tmp_path: Path) -> None:
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"llama"}')
    for filename, extra in (
        ("one.safetensors", "unique_one"),
        ("two.safetensors", "unique_two"),
    ):
        write_safetensors(
            root / filename,
            (("shared", "float32", (1,), bytes(4)), (extra, "float32", (1,), bytes(4))),
        )
    document = open_huggingface(root)
    assert len(document.tensors) == 3
    assert "huggingface.duplicate-tensor-name" in {d.code for d in document.diagnostics}


def test_malformed_config_and_tokenizer_leave_weights_inspectable(
    tmp_path: Path,
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    (root / "config.json").write_text("[]")
    (root / "tokenizer.json").write_text("{")
    (root / "tokenizer_config.json").write_text("[]")
    document = open_huggingface(root)
    assert len(document.tensors) == 12
    assert (
        document.capability(Capability.WEIGHTS).availability is Availability.AVAILABLE
    )
    assert (
        document.capability(Capability.TRACING).availability is Availability.UNAVAILABLE
    )
    assert "huggingface.config-unreadable" in {d.code for d in document.diagnostics}


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("rope_scaling", {"type": "linear", "factor": 2}, "Scaled RoPE"),
        ("sliding_window", 128, "sliding-window"),
        ("hidden_act", "relu", "Activation"),
        ("rms_norm_eps", 0.0, "positive finite"),
        ("head_dim", -2, "positive finite"),
        ("num_attention_heads", 3, "multiple"),
    ],
)
def test_unsupported_execution_configs_still_allow_inspection(
    tmp_path: Path, key: str, value: Any, message: str
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    path = root / "config.json"
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    document = open_huggingface(root)
    status = document.capability(Capability.TRACING)
    assert status.availability is Availability.UNAVAILABLE
    assert message in status.reason
    assert (
        document.capability(Capability.WEIGHTS).availability is Availability.AVAILABLE
    )
