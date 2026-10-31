"""Deterministic Hugging Face snapshot fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nneditor.adapters.pytorch.safetensors import write_safetensors

__all__ = [
    "write_adapter_snapshot",
    "write_llama_snapshot",
    "write_sharded_snapshot",
    "write_traceable_llama_snapshot",
]

_LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "vocab_size": 8,
    "max_position_embeddings": 32,
    "intermediate_size": 8,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def _weight_bytes(elements: int, start: int = 0) -> bytes:
    return bytes((start + index) % 256 for index in range(elements * 4))


def write_llama_snapshot(root: Path) -> Path:
    """A one-block Llama-style snapshot with a tiny tokenizer."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_LLAMA_CONFIG), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 32,
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}},
                "added_tokens": [],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        root / "model.safetensors",
        (
            ("model.embed_tokens.weight", "float32", (8, 4), _weight_bytes(32, 0)),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "float32",
                (4, 4),
                _weight_bytes(16, 10),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "float32",
                (4, 4),
                _weight_bytes(16, 20),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "float32",
                (4, 4),
                _weight_bytes(16, 30),
            ),
            ("model.norm.weight", "float32", (4,), _weight_bytes(4, 40)),
            ("lm_head.weight", "float32", (8, 4), _weight_bytes(32, 50)),
        ),
    )
    return root


def write_sharded_snapshot(root: Path) -> Path:
    """Two safetensors shards plus a Hugging Face weight-map index."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_LLAMA_CONFIG), encoding="utf-8")
    first = (
        ("model.embed_tokens.weight", "float32", (8, 4), _weight_bytes(32, 0)),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "float32",
            (4, 4),
            _weight_bytes(16, 10),
        ),
    )
    second = (
        (
            "model.layers.0.mlp.down_proj.weight",
            "float32",
            (4, 4),
            _weight_bytes(16, 30),
        ),
        ("lm_head.weight", "float32", (8, 4), _weight_bytes(32, 50)),
    )
    write_safetensors(root / "model-00001-of-00002.safetensors", first)
    write_safetensors(root / "model-00002-of-00002.safetensors", second)
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name, _t, _d, _b in first
    }
    weight_map.update(
        {name: "model-00002-of-00002.safetensors" for name, _t, _d, _b in second}
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 256}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return root


def _f32_bytes(shape: tuple[int, ...], seed: int, scale: float = 0.02) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(shape).astype("<f4") * np.float32(scale)
    return array.tobytes()


def _ones_bytes(shape: tuple[int, ...]) -> bytes:
    return np.ones(shape, dtype="<f4").tobytes()


def write_traceable_llama_snapshot(root: Path) -> Path:
    """A one-block Llama snapshot with every weight a numpy trace needs."""
    root.mkdir(parents=True, exist_ok=True)
    config = {
        **_LLAMA_CONFIG,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 32,
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}},
                "added_tokens": [],
            }
        ),
        encoding="utf-8",
    )
    hidden, vocab, intermediate = 4, 8, 8
    write_safetensors(
        root / "model.safetensors",
        (
            (
                "model.embed_tokens.weight",
                "float32",
                (vocab, hidden),
                _f32_bytes((vocab, hidden), 1),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "float32",
                (hidden,),
                _ones_bytes((hidden,)),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "float32",
                (hidden, hidden),
                _f32_bytes((hidden, hidden), 2),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "float32",
                (hidden, hidden),
                _f32_bytes((hidden, hidden), 3),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "float32",
                (hidden, hidden),
                _f32_bytes((hidden, hidden), 4),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "float32",
                (hidden, hidden),
                _f32_bytes((hidden, hidden), 5),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "float32",
                (hidden,),
                _ones_bytes((hidden,)),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "float32",
                (intermediate, hidden),
                _f32_bytes((intermediate, hidden), 6),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "float32",
                (intermediate, hidden),
                _f32_bytes((intermediate, hidden), 7),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "float32",
                (hidden, intermediate),
                _f32_bytes((hidden, intermediate), 8),
            ),
            ("model.norm.weight", "float32", (hidden,), _ones_bytes((hidden,))),
            (
                "lm_head.weight",
                "float32",
                (vocab, hidden),
                _f32_bytes((vocab, hidden), 9),
            ),
        ),
    )
    return root


def write_adapter_snapshot(root: Path) -> Path:
    """A PEFT adapter directory: adapter_config.json plus adapter weights."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "tiny-llama",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        root / "adapter_model.safetensors",
        (
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "float32",
                (2, 4),
                _weight_bytes(8, 1),
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
                "float32",
                (4, 2),
                _weight_bytes(8, 2),
            ),
        ),
    )
    return root
