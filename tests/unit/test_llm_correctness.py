"""Numerical parity with independent PyTorch decoder implementations."""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.gguf.constants import KV_TYPE_STRING, KV_TYPE_UINT32
from nneditor.adapters.huggingface import open_huggingface
from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.storage.store import TensorStore
from nneditor.tracing.llm_runtime import trace_document
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _snapshot(
    root: Path,
    config: dict[str, Any],
    weights: dict[str, np.ndarray],
) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(
        root / "model.safetensors",
        (
            (name, "float32", array.shape, array.astype("<f4").tobytes())
            for name, array in weights.items()
        ),
    )
    return root


def _run(root: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    document = open_huggingface(root)
    with TensorStore(document) as store:
        captures, notes, diagnostics = trace_document(document, {"tokens": ids}, store)
    assert notes == diagnostics == ()
    return {
        document.main_graph.value(key).name or key: value
        for key, value in captures.items()
    }


@pytest.mark.parametrize("activation", ["gelu_new", "gelu"])
def test_gpt2_matches_torch_with_biases_tied_head_and_square_conv1d(
    tmp_path: Path,
    activation: str,
) -> None:
    rng = np.random.default_rng(78)
    # Odd head dimensions also prove GPT-2 never enters the RoPE path.
    hidden, heads, vocab = 6, 2, 13
    shapes = {
        "transformer.wte.weight": (vocab, hidden),
        "transformer.wpe.weight": (11, hidden),
        "transformer.h.0.ln_1.weight": (hidden,),
        "transformer.h.0.ln_1.bias": (hidden,),
        "transformer.h.0.attn.c_attn.weight": (hidden, 3 * hidden),
        "transformer.h.0.attn.c_attn.bias": (3 * hidden,),
        "transformer.h.0.attn.c_proj.weight": (hidden, hidden),
        "transformer.h.0.attn.c_proj.bias": (hidden,),
        "transformer.h.0.ln_2.weight": (hidden,),
        "transformer.h.0.ln_2.bias": (hidden,),
        "transformer.h.0.mlp.c_fc.weight": (hidden, 9),
        "transformer.h.0.mlp.c_fc.bias": (9,),
        "transformer.h.0.mlp.c_proj.weight": (9, hidden),
        "transformer.h.0.mlp.c_proj.bias": (hidden,),
        "transformer.ln_f.weight": (hidden,),
        "transformer.ln_f.bias": (hidden,),
    }
    weights = {
        name: rng.normal(0, 0.3, shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    root = _snapshot(
        tmp_path / "gpt2",
        {
            "model_type": "gpt2",
            "n_embd": hidden,
            "n_head": heads,
            "n_layer": 1,
            "vocab_size": vocab,
            "tie_word_embeddings": True,
            "layer_norm_epsilon": 1e-5,
            "activation_function": activation,
        },
        weights,
    )
    ids = np.array([[1, 4, 2, 6], [3, 2, 5, 1]], dtype=np.int64)
    w = {key: torch.from_numpy(value) for key, value in weights.items()}
    x = (
        w["transformer.wte.weight"][torch.from_numpy(ids)]
        + w["transformer.wpe.weight"][:4]
    )
    expected = {"embed": x}
    z = F.layer_norm(
        x,
        (hidden,),
        w["transformer.h.0.ln_1.weight"],
        w["transformer.h.0.ln_1.bias"],
        1e-5,
    )
    q, k, v = (
        z @ w["transformer.h.0.attn.c_attn.weight"]
        + w["transformer.h.0.attn.c_attn.bias"]
    ).chunk(3, dim=-1)
    q, k, v = (t.reshape(2, 4, heads, 3).transpose(1, 2) for t in (q, k, v))
    mixed = (
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        .transpose(1, 2)
        .reshape(2, 4, hidden)
    )
    x = (
        x
        + mixed @ w["transformer.h.0.attn.c_proj.weight"]
        + w["transformer.h.0.attn.c_proj.bias"]
    )
    expected["block-000-attn"] = x
    z = F.layer_norm(
        x,
        (hidden,),
        w["transformer.h.0.ln_2.weight"],
        w["transformer.h.0.ln_2.bias"],
        1e-5,
    )
    z = F.gelu(
        z @ w["transformer.h.0.mlp.c_fc.weight"] + w["transformer.h.0.mlp.c_fc.bias"],
        approximate="tanh" if activation == "gelu_new" else "none",
    )
    x = (
        x
        + z @ w["transformer.h.0.mlp.c_proj.weight"]
        + w["transformer.h.0.mlp.c_proj.bias"]
    )
    expected["block-000-ffn"] = x
    x = F.layer_norm(
        x, (hidden,), w["transformer.ln_f.weight"], w["transformer.ln_f.bias"], 1e-5
    )
    expected["output-norm"] = x
    expected["lm-head"] = F.linear(x, w["transformer.wte.weight"])
    actual = _run(root, ids)
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name], value.numpy(), rtol=2e-5, atol=2e-6, err_msg=name
        )


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_gqa_and_qk_normalization_match_torch(
    tmp_path: Path, architecture: str
) -> None:
    rng = np.random.default_rng(183)
    hidden, heads, kv_heads, vocab = 8, 2, 1, 17
    dim = 6 if architecture == "qwen3" else 4
    shapes = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.layers.0.input_layernorm.weight": (hidden,),
        "model.layers.0.self_attn.q_proj.weight": (heads * dim, hidden),
        "model.layers.0.self_attn.k_proj.weight": (kv_heads * dim, hidden),
        "model.layers.0.self_attn.v_proj.weight": (kv_heads * dim, hidden),
        "model.layers.0.self_attn.o_proj.weight": (hidden, heads * dim),
        "model.layers.0.post_attention_layernorm.weight": (hidden,),
        "model.layers.0.mlp.up_proj.weight": (12, hidden),
        "model.layers.0.mlp.gate_proj.weight": (12, hidden),
        "model.layers.0.mlp.down_proj.weight": (hidden, 12),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    if architecture == "qwen3":
        shapes.update(
            {
                "model.layers.0.self_attn.q_norm.weight": (dim,),
                "model.layers.0.self_attn.k_norm.weight": (dim,),
            }
        )
    weights = {
        name: rng.normal(0, 0.4, shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    root = _snapshot(
        tmp_path / architecture,
        {
            "model_type": architecture,
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "num_hidden_layers": 1,
            "vocab_size": vocab,
            "head_dim": dim,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        },
        weights,
    )
    ids = np.array([[1, 4, 2, 6]], dtype=np.int64)
    w = {key: torch.from_numpy(value) for key, value in weights.items()}
    x = w["model.embed_tokens.weight"][torch.from_numpy(ids)]
    z = F.rms_norm(x, (hidden,), w["model.layers.0.input_layernorm.weight"], 1e-5)
    q = F.linear(z, w["model.layers.0.self_attn.q_proj.weight"]).reshape(
        1, 4, heads, dim
    )
    k = F.linear(z, w["model.layers.0.self_attn.k_proj.weight"]).reshape(
        1, 4, kv_heads, dim
    )
    v = F.linear(z, w["model.layers.0.self_attn.v_proj.weight"]).reshape(
        1, 4, kv_heads, dim
    )
    if architecture == "qwen3":
        q = F.rms_norm(q, (dim,), w["model.layers.0.self_attn.q_norm.weight"], 1e-5)
        k = F.rms_norm(k, (dim,), w["model.layers.0.self_attn.k_norm.weight"], 1e-5)
    frequencies = torch.outer(
        torch.arange(4), 10000.0 ** (-torch.arange(0, dim, 2) / dim)
    )
    angles = torch.cat((frequencies, frequencies), dim=-1)[None, :, None, :]

    def rotate(t: Any) -> Any:
        a, b = t.chunk(2, dim=-1)
        return t * angles.cos() + torch.cat((-b, a), dim=-1) * angles.sin()

    q, k = rotate(q), rotate(k)
    mixed = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(heads, dim=2).transpose(1, 2),
        v.repeat_interleave(heads, dim=2).transpose(1, 2),
        is_causal=True,
    )
    x = x + F.linear(
        mixed.transpose(1, 2).reshape(1, 4, heads * dim),
        w["model.layers.0.self_attn.o_proj.weight"],
    )
    expected_attention = x
    z = F.rms_norm(
        x, (hidden,), w["model.layers.0.post_attention_layernorm.weight"], 1e-5
    )
    z = F.silu(F.linear(z, w["model.layers.0.mlp.gate_proj.weight"])) * F.linear(
        z, w["model.layers.0.mlp.up_proj.weight"]
    )
    x = x + F.linear(z, w["model.layers.0.mlp.down_proj.weight"])
    x = F.rms_norm(x, (hidden,), w["model.norm.weight"], 1e-5)
    logits = F.linear(x, w["lm_head.weight"])
    actual = _run(root, ids)
    np.testing.assert_allclose(
        actual["block-000-attn"], expected_attention.numpy(), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(actual["lm-head"], logits.numpy(), rtol=2e-5, atol=2e-6)
    if architecture == "llama":
        _check_gguf_llama(tmp_path / "llama.gguf", weights, ids, logits.numpy())


def _check_gguf_llama(
    path: Path,
    weights: dict[str, np.ndarray],
    ids: np.ndarray,
    expected: np.ndarray,
) -> None:
    names = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
        "model.layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
        **{
            f"model.layers.0.self_attn.{role}_proj.weight": (
                f"blk.0.attn_{target}.weight"
            )
            for role, target in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "output"))
        },
        **{
            f"model.layers.0.mlp.{role}_proj.weight": f"blk.0.ffn_{role}.weight"
            for role in ("gate", "up", "down")
        },
    }
    tensors = []
    for name, array in weights.items():
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = 2 if name.endswith("q_proj.weight") else 1
            # HF split-half RoPE channels become GGUF's adjacent pairs.
            array = array.reshape(heads, 2, 2, 8).swapaxes(1, 2).reshape(array.shape)
        tensors.append(
            GgufFixtureTensor(
                names[name],
                array.shape,
                0,
                array.astype("<f4").tobytes(),
            )
        )
    write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", KV_TYPE_STRING, "llama"),
            ("llama.embedding_length", KV_TYPE_UINT32, 8),
            ("llama.attention.head_count", KV_TYPE_UINT32, 2),
            ("llama.attention.head_count_kv", KV_TYPE_UINT32, 1),
            ("llama.vocab_size", KV_TYPE_UINT32, 17),
            ("llama.block_count", KV_TYPE_UINT32, 1),
        ),
        tensors=tuple(tensors),
    )
    document = open_gguf(path)
    with TensorStore(document) as store:
        captures, _, _ = trace_document(document, {"tokens": ids}, store)
    np.testing.assert_allclose(
        captures[document.main_graph.outputs[0]], expected, rtol=2e-5, atol=2e-6
    )


def test_streamed_trace_releases_activations(tmp_path: Path) -> None:
    from tests.fixtures.huggingface_models import write_traceable_llama_snapshot

    document = open_huggingface(write_traceable_llama_snapshot(tmp_path / "hf"))
    values: list[weakref.ReferenceType[np.ndarray]] = []

    def capture(_key: str, value: np.ndarray) -> None:
        values.append(weakref.ref(value))

    with TensorStore(document) as store:
        captures, _, _ = trace_document(
            document, {"tokens": np.array([[1, 2, 3]])}, store, capture=capture
        )
    assert captures == {}
    assert len(values) == len(document.main_graph.nodes) + 1
    assert all(reference() is None for reference in values)
