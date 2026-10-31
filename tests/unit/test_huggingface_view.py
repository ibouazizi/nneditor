"""Headless Hugging Face snapshot presentation."""

from __future__ import annotations

from pathlib import Path

from nneditor.adapters.huggingface import open_huggingface, tensor_source_name
from nneditor.ui.hf_view import (
    hf_config_rows,
    hf_container_summary,
    hf_layout_segments,
    hf_tensor_catalog,
    hf_tokenizer_rows,
    matches_query,
    tensor_owner_node_id,
)
from nneditor.ui.viewmodel import group_tensor_ids, node_tensor_ids
from tests.fixtures.huggingface_models import write_llama_snapshot


def test_layout_and_config_and_tokenizer(tmp_path: Path) -> None:
    document = open_huggingface(write_llama_snapshot(tmp_path / "tiny"))
    summary = hf_container_summary(document)
    assert summary is not None
    assert summary.architecture == "llama"
    assert summary.block_count == 1
    assert summary.vocab_size == 8
    keys = {segment.key for segment in hf_layout_segments(document)}
    assert "config" in keys
    assert "weights" in keys
    assert "tokenizer" in keys
    config_keys = {row.key for row in hf_config_rows(document)}
    assert "model_type" in config_keys
    assert "architectures" in config_keys
    tokenizer = hf_tokenizer_rows(document)
    names = {row.key for row in tokenizer}
    assert "tokenizer.json" in names
    assert "vocab" in names


def test_tensor_catalog_binds_attention_weights(tmp_path: Path) -> None:
    document = open_huggingface(write_llama_snapshot(tmp_path / "tiny"))
    catalog = hf_tensor_catalog(document)
    q_proj = next(
        row for row in catalog if row.name.endswith("self_attn.q_proj.weight")
    )
    assert q_proj.group == "block-000"
    assert q_proj.block == 0
    assert q_proj.owner_node_id is not None
    owner = q_proj.owner_node_id
    graph = document.main_graph
    bound = node_tensor_ids(document, graph.id, owner)
    names = {tensor_source_name(tensor_id) for tensor_id in bound}
    assert "model.layers.0.self_attn.q_proj.weight" in names
    grouped = group_tensor_ids(document, graph.id, (owner,))
    assert q_proj.tensor_id in grouped
    assert tensor_owner_node_id(document, q_proj.tensor_id) == owner


def test_matches_query_is_case_insensitive() -> None:
    assert matches_query("Q_PROJ", "model.layers.0.self_attn.q_proj.weight")
    assert not matches_query("missing", "model.embed_tokens.weight")
