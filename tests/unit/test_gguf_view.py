"""Headless GGUF container presentation: layout, metadata, tensors, weights."""

from __future__ import annotations

from pathlib import Path

from nneditor.adapters.gguf import open_gguf, preview_kv_array, tensor_source_name
from nneditor.adapters.gguf.constants import (
    KV_TYPE_ARRAY,
    KV_TYPE_STRING,
    KV_TYPE_UINT32,
)
from nneditor.ui.gguf_view import (
    gguf_container_summary,
    gguf_kv_rows,
    gguf_layout_segments,
    gguf_tensor_catalog,
    gguf_tokenizer_rows,
    matches_query,
    tensor_owner_node_id,
)
from nneditor.ui.viewmodel import group_tensor_ids, node_details, node_tensor_ids
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

_F32 = 0


def _tiny(path: Path) -> Path:
    embed = bytes(range(16))
    q = bytes(range(16, 32))
    k = bytes(range(32, 48))
    down = bytes(range(48, 64))
    tokens = tuple(f"tok{index}" for index in range(300))
    return write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", KV_TYPE_STRING, "llama"),
            ("general.name", KV_TYPE_STRING, "tiny"),
            ("llama.block_count", KV_TYPE_UINT32, 1),
            ("llama.embedding_length", KV_TYPE_UINT32, 2),
            ("llama.vocab_size", KV_TYPE_UINT32, 2),
            ("llama.context_length", KV_TYPE_UINT32, 128),
            ("tokenizer.ggml.model", KV_TYPE_STRING, "llama"),
            ("tokenizer.ggml.tokens", KV_TYPE_ARRAY, (KV_TYPE_STRING, tokens)),
        ),
        tensors=(
            GgufFixtureTensor("token_embd.weight", (2, 2), _F32, embed),
            GgufFixtureTensor("blk.0.attn_q.weight", (2, 2), _F32, q),
            GgufFixtureTensor("blk.0.attn_k.weight", (2, 2), _F32, k),
            GgufFixtureTensor("blk.0.ffn_down.weight", (2, 2), _F32, down),
        ),
    )


def test_container_summary_and_layout(tmp_path: Path) -> None:
    document = open_gguf(_tiny(tmp_path / "tiny.gguf"))
    summary = gguf_container_summary(document)
    assert summary is not None
    assert summary.architecture == "llama"
    assert summary.name == "tiny"
    assert summary.version == 3
    assert summary.block_count == 1
    assert summary.vocab_size == 2
    assert summary.context_length == 128
    assert summary.n_tensors == 4
    assert summary.n_kv >= 7
    assert summary.payload_bytes > 0
    keys = [segment.key for segment in gguf_layout_segments(document)]
    assert keys[:3] == ["header", "metadata", "tensors"]
    assert keys[-1] == "weights"
    header = gguf_layout_segments(document)[0]
    assert header.offset == 0 and header.length == 24


def test_metadata_groups_and_tokenizer_preview(tmp_path: Path) -> None:
    path = _tiny(tmp_path / "tok.gguf")
    document = open_gguf(path)
    rows = {row.key: row for row in gguf_kv_rows(document)}
    assert rows["general.architecture"].value_text == "llama"
    assert rows["general.architecture"].group == "general"
    assert rows["llama.block_count"].group == "architecture"
    tokens = rows["tokenizer.ggml.tokens"]
    assert tokens.group == "tokenizer"
    assert tokens.elided
    assert tokens.previewable
    assert "300" in tokens.value_text
    tokenizer = gguf_tokenizer_rows(document)
    assert any(row.key == "tokenizer.ggml.tokens" for row in tokenizer)
    recorded = tokens.value
    assert isinstance(recorded, dict)
    offset, length = recorded["offset"], recorded["length"]
    assert isinstance(offset, int) and not isinstance(offset, bool)
    assert isinstance(length, int) and not isinstance(length, bool)
    items, total = preview_kv_array(
        path,
        offset=offset,
        length=length,
        version=3,
        endian="little",
        limit=8,
    )
    assert total == 300
    assert items[0] == "tok0"
    assert len(items) == 8


def test_tensor_catalog_binds_weights_to_nodes(tmp_path: Path) -> None:
    document = open_gguf(_tiny(tmp_path / "cat.gguf"))
    catalog = {row.name: row for row in gguf_tensor_catalog(document)}
    assert catalog["token_embd.weight"].group == "embeddings"
    assert catalog["blk.0.attn_q.weight"].group == "block-000"
    assert catalog["blk.0.attn_q.weight"].block == 0
    assert catalog["blk.0.attn_q.weight"].offset is not None
    attn = next(
        node for node in document.main_graph.nodes if node.op_type == "Attention"
    )
    weights = node_tensor_ids(document, document.entry_graph, attn.id)
    names = {tensor_source_name(tensor_id) for tensor_id in weights}
    assert names == {"blk.0.attn_q.weight", "blk.0.attn_k.weight"}
    details = dict(node_details(document, document.entry_graph, attn.id))
    assert any(key.startswith("Weight ") for key in details)
    q_id = catalog["blk.0.attn_q.weight"].tensor_id
    assert tensor_owner_node_id(document, q_id) == attn.id
    grouped = group_tensor_ids(document, document.entry_graph, (attn.id,))
    assert q_id in grouped


def test_search_matches_any_field() -> None:
    assert matches_query("Q4", "blk.0.attn_q.weight", "q4_0")
    assert not matches_query("ssm", "blk.0.attn_q.weight", "q4_0")
    assert matches_query("  ", "anything")
