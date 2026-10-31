"""Native GGUF parser: every published version, laziness, and IR conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.detect import detect_artifact_kind
from nneditor.adapters.gguf import (
    GgufIndexError,
    index_gguf,
    load_block_tensors,
    open_gguf,
    tensor_source_name,
)
from nneditor.adapters.gguf.constants import (
    KV_TYPE_ARRAY,
    KV_TYPE_STRING,
    KV_TYPE_UINT32,
    packed_nbytes,
)
from nneditor.adapters.registry import default_artifact_adapter_registry
from nneditor.analysis.lod import DetailLevel
from nneditor.application.hierarchy import HierarchyController, pipeline_for_document
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import PayloadRange, Storage
from nneditor.ir.dtypes import dtype_info
from nneditor.storage.reader import ArtifactReader, ByteRange
from nneditor.storage.store import TensorStore
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

_HERO_27B = Path(
    "/mnt/d/workspace/npu/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
)

_F32 = 0
_Q4_0 = 2
_UNKNOWN_FORK = 143


def _tiny_llm(
    path: Path,
    *,
    version: int = 3,
    endian: str = "little",
    extra: tuple[GgufFixtureTensor, ...] = (),
) -> Path:
    embed = bytes(range(16))
    q = bytes(range(16, 32))
    k = bytes(range(32, 48))
    down = bytes(range(48, 64))
    tensors = (
        GgufFixtureTensor("token_embd.weight", (2, 2), _F32, embed),
        GgufFixtureTensor("blk.0.attn_q.weight", (2, 2), _F32, q),
        GgufFixtureTensor("blk.0.attn_k.weight", (2, 2), _F32, k),
        GgufFixtureTensor("blk.0.ffn_down.weight", (2, 2), _F32, down),
        *extra,
    )
    return write_gguf(
        path,
        version=version,
        endian=endian,
        kv=(
            ("general.architecture", KV_TYPE_STRING, "llama"),
            ("general.name", KV_TYPE_STRING, "tiny"),
            ("llama.block_count", KV_TYPE_UINT32, 1),
            ("llama.embedding_length", KV_TYPE_UINT32, 2),
            ("llama.vocab_size", KV_TYPE_UINT32, 2),
        ),
        tensors=tensors,
    )


def _as_range(payload: ByteRange | PayloadRange) -> ByteRange:
    return ByteRange(payload.offset, payload.length)


@pytest.mark.parametrize("version", (1, 2, 3))
def test_native_parser_reads_every_little_endian_version(
    tmp_path: Path, version: int
) -> None:
    path = _tiny_llm(tmp_path / f"v{version}.gguf", version=version)
    index = index_gguf(path)
    assert index.version == version
    assert index.endian == "little"
    assert index.n_tensors == 4
    assert index.content_hash.startswith("gguf-index:")
    names = [tensor.name for tensor in index.tensors]
    assert names[0] == "token_embd.weight"
    assert all(tensor.payload is not None for tensor in index.tensors)
    for tensor in index.tensors:
        payload = tensor.payload
        assert payload is not None
        assert not index.stats.touched_logically(payload)


def test_v3_big_endian_decodes_swapped_fields(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "be.gguf", version=3, endian="big")
    index = index_gguf(path)
    assert index.version == 3
    assert index.endian == "big"
    assert "gguf.big-endian" in index.diagnostics.codes()
    document = open_gguf(path)
    assert document.artifact_kind is ArtifactKind.GGUF_MODEL
    embed = next(
        tensor
        for tensor in document.tensors.values()
        if tensor_source_name(tensor.id) == "token_embd.weight"
    )
    with TensorStore(document) as store:
        assert store.read(embed.id) == bytes(range(16))


def test_detect_and_registry_dispatch_on_magic(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "model.bin")
    assert detect_artifact_kind(path) is ArtifactKind.GGUF_MODEL
    document = default_artifact_adapter_registry().open(path)
    assert document.artifact_kind is ArtifactKind.GGUF_MODEL
    assert document.source.content_hash.startswith("gguf-index:")


def test_open_does_not_hash_payloads_on_first_tensor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _tiny_llm(tmp_path / "lazy.gguf")
    document = open_gguf(path)

    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("must not hash the whole GGUF file")

    monkeypatch.setattr(ArtifactReader, "content_hash", boom)
    embed = next(iter(document.tensors))
    with TensorStore(document) as store:
        payload = store.read(embed)
    assert payload == bytes(range(16))


def test_reading_one_tensor_does_not_touch_another(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "isolated.gguf")
    document = open_gguf(path)
    tensors = list(document.tensors.values())
    first, second = tensors[0], tensors[1]
    with TensorStore(document) as store:
        store.read(first.id)
        reader = next(iter(store._readers.values()))
        assert first.payload is not None and second.payload is not None
        assert reader.stats.touched_logically(_as_range(first.payload))
        assert not reader.stats.touched_logically(_as_range(second.payload))


def test_read_group_coalesces_adjacent_block_tensors(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "block.gguf")
    document = open_gguf(path)
    with TensorStore(document) as store:
        loaded = load_block_tensors(store, 0)
    assert set(loaded) == {
        tensor.id
        for tensor in document.tensors.values()
        if tensor_source_name(tensor.id).startswith("blk.0.")
    }
    with TensorStore(document) as store:
        load_block_tensors(store, 0)
        reader = next(iter(store._readers.values()))
        logical = reader.stats.coalesced_logical()
        assert len(logical) == 1
        block_payloads = [
            _as_range(tensor.payload)
            for tensor in document.tensors.values()
            if tensor.payload is not None
            and tensor_source_name(tensor.id).startswith("blk.0.")
        ]
        span = logical[0]
        assert all(span.contains(payload) for payload in block_payloads)


def test_synthetic_graph_and_source_hierarchy(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "arch.gguf")
    document = open_gguf(path)
    ops = [node.op_type for node in document.main_graph.nodes]
    assert ops == ["Embedding", "Attention", "FFN", "Linear"]
    assert all(node.domain == "nneditor.llm" for node in document.main_graph.nodes)
    assert any(
        node.source_name == "llm/block-000/attn" for node in document.main_graph.nodes
    )
    assert any(
        entry.operation == "synthesize-llm-architecture"
        for entry in document.provenance
    )
    pipeline = pipeline_for_document(document)
    assert len(pipeline.detectors) == 1
    controller = HierarchyController(document)
    hierarchy = controller.hierarchy(document.entry_graph)
    labels = {group.label for group in hierarchy.groups}
    assert "block-000" in labels
    embed = next(
        tensor
        for tensor in document.tensors.values()
        if tensor_source_name(tensor.id) == "token_embd.weight"
    )
    assert embed.dims == (2, 2)
    model = dict(document.extensions)["x-gguf.model"]
    assert model["architecture"] == "llama"
    assert model["tied_embeddings"] is True
    assert model["kv_end"] > 0
    assert model["tensor_table_end"] >= model["kv_end"]
    kv_records = dict(document.extensions)["x-gguf.kv"]
    assert isinstance(kv_records, list)
    keys = {item["key"] for item in kv_records if isinstance(item, dict)}
    assert "general.architecture" in keys


def test_operator_detail_caps_to_layer(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "lod.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        sliced = session.scene(detail_level=DetailLevel.OPERATOR)
    assert sliced.detail_level is DetailLevel.LAYER


def test_unknown_fork_type_is_named_not_special_cased(tmp_path: Path) -> None:
    extra = (
        GgufFixtureTensor("blk.0.attn_v.weight", (2, 2), _UNKNOWN_FORK, bytes(20)),
    )
    path = _tiny_llm(tmp_path / "fork.gguf", extra=extra)
    document = open_gguf(path)
    tensor = next(
        item
        for item in document.tensors.values()
        if tensor_source_name(item.id) == "blk.0.attn_v.weight"
    )
    assert tensor.element_type == "gguf.type_143"
    info = dtype_info("gguf.type_143")
    assert info is not None
    assert info.decoder is None
    assert "gguf.unknown-tensor-type" in {item.code for item in document.diagnostics}


def test_tokenizer_string_array_is_elided(tmp_path: Path) -> None:
    tokens = tuple(f"t{index}" for index in range(300))
    path = write_gguf(
        tmp_path / "tok.gguf",
        version=3,
        kv=(("tokenizer.ggml.tokens", KV_TYPE_ARRAY, (KV_TYPE_STRING, tokens)),),
        tensors=(GgufFixtureTensor("token_embd.weight", (2, 2), _F32, bytes(16)),),
    )
    index = index_gguf(path)
    assert "gguf.kv-value-elided" in index.diagnostics.codes()
    entry = index.kv_map["tokenizer.ggml.tokens"]
    assert entry.elided
    assert isinstance(entry.value, dict)
    assert entry.value["count"] == 300


def test_packed_nbytes_matches_public_ggml_table() -> None:
    assert packed_nbytes(_Q4_0, 32) == 18
    assert packed_nbytes(_F32, 4) == 16
    assert packed_nbytes(_UNKNOWN_FORK, 4) is None
    q4 = dtype_info("q4_0")
    assert q4 is not None and q4.decoder is None
    assert "block quantization" in (q4.unavailable_reason or "")


def test_truncated_header_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "short.gguf"
    path.write_bytes(b"GGUF\x03\x00\x00\x00")
    with pytest.raises(GgufIndexError, match="truncated"):
        index_gguf(path)


def test_weights_stay_available_on_a_healthy_file(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "ok.gguf")
    document = open_gguf(path)
    topology = document.capability(Capability.TOPOLOGY).availability
    weights = document.capability(Capability.WEIGHTS).availability
    assert topology is Availability.PARTIAL
    assert weights is Availability.AVAILABLE
    assert all(
        tensor.storage is Storage.EMBEDDED_RAW for tensor in document.tensors.values()
    )


@pytest.mark.skipif(not _HERO_27B.is_file(), reason="27B GGUF is not on this machine")
def test_27b_open_is_header_only() -> None:
    index = index_gguf(_HERO_27B)
    assert index.version in {2, 3}
    assert index.n_tensors > 100
    assert index.content_hash.startswith("gguf-index:")
    for tensor in index.tensors:
        if tensor.payload is not None:
            assert not index.stats.touched_logically(tensor.payload)
    document = open_gguf(_HERO_27B)
    assert document.artifact_kind is ArtifactKind.GGUF_MODEL
    assert len(document.tensors) == index.n_tensors
