"""Hugging Face snapshot adapter: detection, laziness, and IR conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.huggingface import (
    HuggingFaceError,
    index_huggingface,
    open_huggingface,
    preview_tokenizer_vocab,
    tensor_source_name,
)
from nneditor.adapters.huggingface import snapshot as hf_snapshot
from nneditor.adapters.huggingface.architecture import plan_architecture
from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.adapters.registry import default_artifact_adapter_registry
from nneditor.analysis.lod import DetailLevel
from nneditor.application.hierarchy import HierarchyController, pipeline_for_document
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import Storage
from nneditor.storage.store import TensorStore
from tests.fixtures.huggingface_models import (
    write_adapter_snapshot,
    write_llama_snapshot,
    write_sharded_snapshot,
)


def test_directory_detects_as_huggingface(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "llama")
    assert detect_artifact_kind(root) is ArtifactKind.HUGGINGFACE_MODEL


def test_lone_safetensors_file_stays_safetensors(tmp_path: Path) -> None:
    path = write_safetensors(
        tmp_path / "model.safetensors",
        (("weight", "float32", (2, 2), bytes(16)),),
    )
    assert detect_artifact_kind(path) is ArtifactKind.SAFETENSORS


def test_config_without_weights_is_not_a_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DetectionError, match="no recognized model package"):
        detect_artifact_kind(root)


def test_detect_and_registry_dispatch_on_directory(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "model")
    document = default_artifact_adapter_registry().open(root)
    assert document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL
    assert document.source.content_hash.startswith("hf-index:")


def test_synthetic_graph_and_source_hierarchy(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "arch")
    document = open_huggingface(root)
    ops = [node.op_type for node in document.main_graph.nodes]
    assert ops == ["Embedding", "Attention", "FFN", "RMSNorm", "Linear"]
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
    model = dict(document.extensions)["x-huggingface.model"]
    assert isinstance(model, dict)
    assert model["architecture"] == "llama"
    assert model["block_count"] == 1
    assert model["vocab_size"] == 8
    config = dict(document.extensions)["x-huggingface.config"]
    assert isinstance(config, dict)
    assert config["model_type"] == "llama"


def test_tensors_are_external_and_lazy(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "lazy")
    document = open_huggingface(root)
    embed = next(
        tensor
        for tensor in document.tensors.values()
        if tensor_source_name(tensor.id) == "model.embed_tokens.weight"
    )
    assert embed.storage is Storage.EXTERNAL
    assert embed.external is not None
    assert embed.external.location == "model.safetensors"
    expected = bytes((index) % 256 for index in range(128))
    with TensorStore(document) as store:
        payload = store.read(embed.id)
    assert payload == expected


def test_reading_one_tensor_does_not_touch_another(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "isolated")
    document = open_huggingface(root)
    first, second = list(document.tensors.values())[:2]
    with TensorStore(document) as store:
        store.read(first.id)
        reader = next(iter(store._readers.values()))
        assert first.external is not None and second.external is not None
        from nneditor.storage.reader import ByteRange

        first_range = ByteRange(first.external.offset, first.external.length or 0)
        second_range = ByteRange(second.external.offset, second.external.length or 0)
        assert reader.stats.touched_logically(first_range)
        assert not reader.stats.touched_logically(second_range)


def test_open_does_not_hash_weight_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_llama_snapshot(tmp_path / "hash")
    hashed: list[str] = []

    def tracking(path: Path | str, **_kwargs: object) -> str:
        hashed.append(Path(path).name)
        raise AssertionError("must not hash snapshot weight files")

    monkeypatch.setattr("nneditor.storage.reader.hash_file", tracking)
    monkeypatch.setattr("nneditor.adapters.pytorch.safetensors.hash_file", tracking)
    document = open_huggingface(root)
    assert document.source.content_hash.startswith("hf-index:")
    assert hashed == []


def test_sharded_index_uses_weight_map(tmp_path: Path) -> None:
    root = write_sharded_snapshot(tmp_path / "sharded")
    document = open_huggingface(root)
    shards = {
        tensor.external.location
        for tensor in document.tensors.values()
        if tensor.external is not None
    }
    assert shards == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    model = dict(document.extensions)["x-huggingface.model"]
    assert isinstance(model, dict)
    assert model["n_shards"] == 2
    q_proj = next(
        tensor
        for tensor in document.tensors.values()
        if tensor_source_name(tensor.id) == "model.layers.0.self_attn.q_proj.weight"
    )
    with TensorStore(document) as store:
        payload = store.read(q_proj.id)
    assert payload == bytes((10 + index) % 256 for index in range(64))


def test_pickle_weights_are_listed_without_bytes(tmp_path: Path) -> None:
    root = tmp_path / "pickle"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 1}),
        encoding="utf-8",
    )
    (root / "pytorch_model.bin").write_bytes(b"not a pickle")
    document = open_huggingface(root)
    assert document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL
    assert list(document.tensors) == []
    weights = document.capability(Capability.WEIGHTS).availability
    execution = document.capability(Capability.EXECUTION).availability
    tracing = document.capability(Capability.TRACING).availability
    assert weights is Availability.UNAVAILABLE
    assert execution is Availability.UNAVAILABLE
    assert tracing is Availability.UNAVAILABLE
    assert "huggingface.pickle-weights" in {item.code for item in document.diagnostics}


def test_missing_shard_lists_tensors_without_bytes(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    document = open_huggingface(root)
    tensor = next(iter(document.tensors.values()))
    assert tensor.storage is Storage.ABSENT
    tracing = document.capability(Capability.TRACING).availability
    assert tracing is Availability.UNAVAILABLE
    assert "huggingface.missing-shard" in {item.code for item in document.diagnostics}


def test_operator_detail_caps_to_layer(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "lod")
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(root)
        sliced = session.scene(detail_level=DetailLevel.OPERATOR)
    assert sliced.detail_level is DetailLevel.LAYER


def test_tokenizer_preview_reads_vocab(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "tok")
    items, total = preview_tokenizer_vocab(root, limit=2)
    assert items == ("a", "b")
    assert total == 3


def test_file_path_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(HuggingFaceError, match="not a snapshot directory"):
        index_huggingface(path)


def test_unreadable_config_is_diagnosed(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "bad-config")
    (root / "config.json").write_text("{", encoding="utf-8")
    document = open_huggingface(root)
    assert "huggingface.config-unreadable" in {
        item.code for item in document.diagnostics
    }


def test_peft_adapter_directory_is_a_snapshot(tmp_path: Path) -> None:
    root = write_adapter_snapshot(tmp_path / "lora")
    assert detect_artifact_kind(root) is ArtifactKind.HUGGINGFACE_MODEL
    document = open_huggingface(root)
    names = {tensor_source_name(tensor_id) for tensor_id in document.tensors}
    assert "base_model.model.layers.0.self_attn.q_proj.lora_A.weight" in names
    assert all(
        tensor.storage is Storage.EXTERNAL for tensor in document.tensors.values()
    )


def test_post_attention_norm_binds_to_ffn() -> None:
    plan = plan_architecture(
        {"model_type": "llama", "num_hidden_layers": 1},
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
        ),
    )
    by_name = {spec.name: spec for spec in plan.layers}
    ffn_names = {binding.tensor_name for binding in by_name["block-000-ffn"].weights}
    attn_names = {binding.tensor_name for binding in by_name["block-000-attn"].weights}
    assert "model.layers.0.post_attention_layernorm.weight" in ffn_names
    assert "model.layers.0.post_attention_layernorm.weight" not in attn_names
    assert "model.layers.0.mlp.down_proj.weight" in ffn_names


def test_identity_ignores_weight_payload_bytes(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "stable")
    first = open_huggingface(root).source.content_hash
    weights = root / "model.safetensors"
    blob = bytearray(weights.read_bytes())
    blob[-1] = (blob[-1] + 1) % 256
    weights.write_bytes(blob)
    second = open_huggingface(root).source.content_hash
    assert first == second
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["hidden_size"] = 8
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    third = open_huggingface(root).source.content_hash
    assert third != first


def test_unsafe_shard_path_is_diagnosed(tmp_path: Path) -> None:
    root = tmp_path / "unsafe"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.embed_tokens.weight": "../escape.safetensors"}}
        ),
        encoding="utf-8",
    )
    document = open_huggingface(root)
    tensor = next(iter(document.tensors.values()))
    assert tensor.storage is Storage.ABSENT
    assert "huggingface.unsafe-shard-path" in {
        item.code for item in document.diagnostics
    }


def test_pickle_index_lists_tensor_names_without_bytes(tmp_path: Path) -> None:
    root = tmp_path / "pickle-index"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    (root / "pytorch_model.bin").write_bytes(b"not a pickle")
    (root / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "pytorch_model.bin"}}),
        encoding="utf-8",
    )
    document = open_huggingface(root)
    tensor = next(iter(document.tensors.values()))
    assert tensor_source_name(tensor.id) == "model.embed_tokens.weight"
    assert tensor.storage is Storage.ABSENT
    assert "huggingface.pickle-weights" in {item.code for item in document.diagnostics}


def test_oversized_tokenizer_is_elided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_llama_snapshot(tmp_path / "elide")
    monkeypatch.setattr(hf_snapshot, "MAX_TOKENIZER_PARSE", 16)
    document = open_huggingface(root)
    assert "huggingface.tokenizer-elided" in {
        item.code for item in document.diagnostics
    }
    tokenizer = dict(document.extensions)["x-huggingface.tokenizer"]
    assert isinstance(tokenizer, dict)
    assert tokenizer.get("elided") is True


def test_export_plan_is_weights_only(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "export")
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(root)
        plan = session.export_plan()
    assert plan.available
    assert plan.allowed_extensions == ("safetensors",)


def test_float_config_counts_are_accepted(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "floats")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 1.0
    config["vocab_size"] = 8.0
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    document = open_huggingface(root)
    model = dict(document.extensions)["x-huggingface.model"]
    assert isinstance(model, dict)
    assert model["block_count"] == 1
    assert model["vocab_size"] == 8


def test_weights_stay_available_on_a_healthy_snapshot(tmp_path: Path) -> None:
    root = write_llama_snapshot(tmp_path / "ok")
    document = open_huggingface(root)
    topology = document.capability(Capability.TOPOLOGY).availability
    weights = document.capability(Capability.WEIGHTS).availability
    execution = document.capability(Capability.EXECUTION).availability
    tracing = document.capability(Capability.TRACING).availability
    assert topology is Availability.PARTIAL
    assert weights is Availability.AVAILABLE
    assert execution is Availability.UNAVAILABLE
    assert tracing is Availability.UNAVAILABLE
    assert all(
        tensor.storage is Storage.EXTERNAL for tensor in document.tensors.values()
    )
