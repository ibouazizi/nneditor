"""Modality graphs name real stages without importing a model library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.huggingface.to_ir import open_huggingface
from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.tracing.plugins import default_trace_registry
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf


def _snapshot(root: Path, config: dict[str, Any]) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    write_safetensors(
        root / "model.safetensors",
        [("placeholder.weight", "float32", (1,), b"\0" * 4)],
    )
    return root


def _names(path: Path) -> set[str]:
    graph = open_huggingface(path).main_graph
    return {graph.value(key).name for key in graph.inputs} | {
        node.op_type for node in graph.nodes
    }


def test_gguf_vision_and_audio_blocks_are_separate_stages(tmp_path: Path) -> None:
    payload = np.zeros(4, dtype=np.float32).tobytes()

    def tensor(name: str) -> GgufFixtureTensor:
        return GgufFixtureTensor(name, (4,), 0, payload)

    path = write_gguf(
        tmp_path / "media.gguf",
        version=3,
        kv=(("general.architecture", 8, "qwen2vl"),),
        tensors=(
            tensor("v.blk.0.attn_q.weight"),
            tensor("v.blk.1.attn_q.weight"),
            tensor("encoder.blk.0.attn_q.weight"),
            tensor("decoder.blk.0.attn_q.weight"),
            tensor("blk.0.attn_q.weight"),
        ),
    )
    graph = open_gguf(path).main_graph
    ops = [node.op_type for node in graph.nodes]
    assert ops[:4] == ["VisionBlock", "VisionBlock", "EncoderBlock", "DecoderBlock"]
    assert "Attention" in ops
    assert graph.value(graph.inputs[0]).name == "pixel-values"


def test_qwen2_vl_gemma3_and_bitnet_graphs(tmp_path: Path) -> None:
    qwen = _names(
        _snapshot(
            tmp_path / "qwen",
            {
                "model_type": "qwen2_vl",
                "vision_config": {
                    "depth": 2,
                    "in_channels": 3,
                    "temporal_patch_size": 2,
                    "patch_size": 14,
                },
                "text_config": {"num_hidden_layers": 1, "vocab_size": 32},
            },
        )
    )
    assert {"pixel-values", "image-grid", "token-types", "VisionBlock"} <= qwen
    gemma = open_huggingface(
        _snapshot(
            tmp_path / "gemma",
            {
                "model_type": "gemma3",
                "vision_config": {
                    "num_hidden_layers": 1,
                    "image_size": 16,
                    "patch_size": 8,
                    "num_channels": 3,
                },
                "text_config": {"num_hidden_layers": 1, "vocab_size": 32},
            },
        )
    )
    assert all(node.domain == "nneditor.vision" for node in gemma.main_graph.nodes)
    bitnet = _names(
        _snapshot(
            tmp_path / "bitnet",
            {"model_type": "bitnet", "num_hidden_layers": 2, "vocab_size": 32},
        )
    )
    assert {"tokens", "DecoderBlock"} <= bitnet


def test_llava_graph_connects_the_tower_projector_and_decoder(tmp_path: Path) -> None:
    path = _snapshot(
        tmp_path / "llava",
        {
            "model_type": "llava",
            "vision_config": {
                "num_hidden_layers": 2,
                "image_size": 8,
                "num_channels": 3,
            },
            "text_config": {"num_hidden_layers": 2, "vocab_size": 32},
        },
    )
    document = open_huggingface(path)
    graph = document.main_graph
    inputs = {graph.value(key).name: graph.value(key) for key in graph.inputs}
    assert inputs["pixel-values"].shape == (1, 3, 8, 8)
    assert inputs["tokens"].element_type == "int64"
    assert {node.op_type for node in graph.nodes} >= {
        "VisionEmbed",
        "VisionBlock",
        "Projector",
        "DecoderBlock",
        "Output",
    }
    assert sum(node.op_type == "VisionBlock" for node in graph.nodes) == 2
    assert all(node.domain == "nneditor.vision" for node in graph.nodes)
    plugin = next(
        p for p in default_trace_registry([]).plugins if p.id == "nneditor.vision"
    )
    assert plugin.incompatibility(document) is None


def test_incomplete_llava_stays_inspectable(tmp_path: Path) -> None:
    document = open_huggingface(_snapshot(tmp_path / "llava", {"model_type": "llava"}))
    error = dict(document.extensions)["x-huggingface.trace-graph-error"]
    assert isinstance(error, str) and "vision_config" in error
    assert document.tensors


def test_speech_seq2seq_and_rwkv_graphs(tmp_path: Path) -> None:
    whisper = _names(
        _snapshot(
            tmp_path / "whisper",
            {
                "model_type": "whisper",
                "encoder_layers": 1,
                "decoder_layers": 1,
                "num_mel_bins": 8,
                "vocab_size": 32,
            },
        )
    )
    assert {
        "input-features",
        "decoder-tokens",
        "EncoderBlock",
        "DecoderBlock",
    } <= whisper
    t5 = _names(
        _snapshot(
            tmp_path / "t5",
            {"model_type": "t5", "num_layers": 1, "vocab_size": 32},
        )
    )
    assert {"source-tokens", "decoder-tokens", "EncoderBlock"} <= t5
    rwkv = _names(
        _snapshot(
            tmp_path / "rwkv",
            {"model_type": "rwkv", "num_hidden_layers": 2, "vocab_size": 32},
        )
    )
    assert {"tokens", "RecurrentBlock"} <= rwkv


def test_dit_and_video_graphs_declare_their_latents(tmp_path: Path) -> None:
    sd3 = open_huggingface(
        _snapshot(
            tmp_path / "sd3",
            {
                "_class_name": "SD3Transformer2DModel",
                "sample_size": 8,
                "patch_size": 2,
                "in_channels": 4,
                "num_layers": 1,
                "joint_attention_dim": 8,
                "pooled_projection_dim": 8,
            },
        )
    ).main_graph
    inputs = {sd3.value(key).name: sd3.value(key) for key in sd3.inputs}
    assert inputs["sample"].shape == (1, 4, 8, 8)
    assert "DiTBlock" in {node.op_type for node in sd3.nodes}
    flux = open_huggingface(
        _snapshot(
            tmp_path / "flux",
            {
                "_class_name": "FluxTransformer2DModel",
                "in_channels": 4,
                "num_layers": 1,
                "num_single_layers": 1,
                "joint_attention_dim": 8,
                "pooled_projection_dim": 8,
                "guidance_embeds": True,
            },
        )
    ).main_graph
    flux_inputs = {flux.value(key).name for key in flux.inputs}
    assert {"sample", "img-ids", "txt-ids", "guidance"} <= flux_inputs
    assert sum(node.op_type == "DiTBlock" for node in flux.nodes) == 2
    cog = open_huggingface(
        _snapshot(
            tmp_path / "cog",
            {
                "_class_name": "CogVideoXTransformer3DModel",
                "sample_frames": 2,
                "in_channels": 4,
                "sample_height": 8,
                "sample_width": 8,
                "text_embed_dim": 16,
                "num_layers": 1,
            },
        )
    ).main_graph
    assert cog.value(cog.inputs[0]).shape == ("batch", 2, 4, 8, 8)
    wan = open_huggingface(
        _snapshot(
            tmp_path / "wan",
            {
                "_class_name": "WanTransformer3DModel",
                "in_channels": 4,
                "text_dim": 16,
                "num_layers": 1,
            },
        )
    ).main_graph
    wan_shape = wan.value(wan.inputs[0]).shape
    assert wan_shape is not None and wan_shape[1] == 4
