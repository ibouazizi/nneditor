"""Text-only Qwen3.5 dense decoder: gated attention and GatedDeltaNet.

GGUF tensor conventions follow Prism/llama.cpp's qwen35.cpp. Each call
starts fresh recurrent and convolution states; batches never share state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nneditor.adapters.gguf.dequant import supports_dequantization
from nneditor.ir.core import AttrKind, Document, JsonValue, Storage
from nneditor.storage.store import TensorStore
from nneditor.tracing.llm_ops import apply_rope, rms_norm, rope_cos_sin, silu, softmax
from nneditor.tracing.llm_runtime import (
    LlmHyperparams,
    LlmTraceError,
    _extension_map,
    _tensor_name,
    _tokens_from_feeds,
)
from nneditor.tracing.llm_weights import GgufWeights, validate_prism

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Qwen35Config:
    blocks: int
    interval: int
    kernel: int
    state: int
    key_heads: int
    value_heads: int
    inner: int
    rotary: int
    head_dim: int
    context: int

    @classmethod
    def from_document(cls, document: Document) -> Qwen35Config:
        meta = _extension_map(document, "x-gguf.metadata")

        def integer(key: str) -> int:
            value = meta.get("qwen35." + key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LlmTraceError(f"Qwen3.5 requires positive {key} metadata")
            return value

        return cls(
            *(
                integer(key)
                for key in (
                    "block_count",
                    "full_attention_interval",
                    "ssm.conv_kernel",
                    "ssm.state_size",
                    "ssm.group_count",
                    "ssm.time_step_rank",
                    "ssm.inner_size",
                    "rope.dimension_count",
                    "attention.key_length",
                    "context_length",
                )
            )
        )


def _require_prism_head(meta: dict[str, JsonValue], names: set[str]) -> None:
    """Version 1 needs its own head; version 2 ties the latent embedding."""
    version = meta.get("prism.hadamard.version")
    if version == 1 and "output.weight" not in names:
        raise LlmTraceError("Prism v1 requires an explicit output projection")
    if version != 2:
        return
    if "output.weight" in names:
        raise LlmTraceError(
            "Prism Hadamard tied output requires output.weight to be absent"
        )
    inverses = meta.get("prism.hadamard.inverse_weight_names")
    if inverses == ["token_embd.weight"]:
        return
    retained = (
        isinstance(inverses, dict)
        and inverses.get("type") == "array"
        and inverses.get("count") == 1
    )
    if not retained:
        raise LlmTraceError(
            "Prism Hadamard tied output requires a latent token embedding"
        )


def validate_qwen35(document: Document, params: LlmHyperparams) -> Qwen35Config:
    """Check the complete dense text decoder by names and shapes, without IO."""
    validate_prism(document)
    cfg = Qwen35Config.from_document(document)
    meta = _extension_map(document, "x-gguf.metadata")
    if cfg.blocks > len(document.tensors):
        raise LlmTraceError("Qwen3.5 block count exceeds the tensor catalog")
    if (
        min(
            params.n_embd,
            params.n_head,
            params.n_kv_head,
            params.vocab_size,
            params.n_ff or 0,
        )
        <= 0
        or params.n_head % params.n_kv_head
        or cfg.value_heads % cfg.key_heads
        or cfg.inner != cfg.value_heads * cfg.state
        or cfg.rotary > cfg.head_dim
        or cfg.rotary % 2
        or meta.get("qwen35.attention.value_length", cfg.head_dim) != cfg.head_dim
        or not math.isfinite(params.rms_eps)
        or params.rms_eps <= 0
        or not math.isfinite(params.rope_theta)
        or params.rope_theta <= 0
    ):
        raise LlmTraceError("Invalid Qwen3.5 head geometry or normalization parameters")
    if meta.get("qwen35.rope.scaling.type", "none") != "none":
        raise LlmTraceError("Scaled Qwen3.5 RoPE cannot be traced")
    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (params.vocab_size, params.n_embd),
        "output_norm.weight": (params.n_embd,),
    }
    tensors = {_tensor_name(t.id): t for t in document.tensors.values()}
    _require_prism_head(meta, set(tensors))
    if "output.weight" in tensors:
        expected["output.weight"] = (params.vocab_size, params.n_embd)
    h, ff, d = params.n_embd, params.n_ff or 0, cfg.head_dim
    conv = 2 * cfg.key_heads * cfg.state + cfg.inner
    for block in range(cfg.blocks):
        shapes = {
            "attn_norm.weight": (h,),
            "post_attention_norm.weight": (h,),
            "ffn_gate.weight": (ff, h),
            "ffn_up.weight": (ff, h),
            "ffn_down.weight": (h, ff),
        }
        if (block + 1) % cfg.interval == 0:
            shapes |= {
                "attn_q.weight": (params.n_head * d * 2, h),
                "attn_k.weight": (params.n_kv_head * d, h),
                "attn_v.weight": (params.n_kv_head * d, h),
                "attn_output.weight": (h, params.n_head * d),
                "attn_q_norm.weight": (d,),
                "attn_k_norm.weight": (d,),
            }
        else:
            shapes |= {
                "attn_qkv.weight": (conv, h),
                "attn_gate.weight": (cfg.inner, h),
                "ssm_conv1d.weight": (conv, cfg.kernel),
                "ssm_dt.bias": (cfg.value_heads,),
                "ssm_a": (cfg.value_heads,),
                "ssm_alpha.weight": (cfg.value_heads, h),
                "ssm_beta.weight": (cfg.value_heads, h),
                "ssm_norm.weight": (cfg.state,),
                "ssm_out.weight": (h, cfg.inner),
            }
        expected.update({f"blk.{block}.{key}": shape for key, shape in shapes.items()})
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if tensor is None:
            raise LlmTraceError(f"Qwen3.5 is missing required weight {name}")
        if tensor.dims != shape:
            raise LlmTraceError(f"{name}: expected shape {shape}, got {tensor.dims}")
        if tensor.storage is Storage.ABSENT or not supports_dequantization(
            tensor.element_type
        ):
            raise LlmTraceError(
                f"{name}: unreadable or unsupported {tensor.element_type} weight"
            )
    if extra := tensors.keys() - expected.keys():
        raise LlmTraceError(f"Unsupported Qwen3.5 weight: {min(extra)}")
    return cfg


def sigmoid(x: FloatArray) -> FloatArray:
    return np.exp(-np.logaddexp(np.float32(0), -x))


def gated_delta(
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    decay: FloatArray,
    beta: FloatArray,
    eps: float,
) -> FloatArray:
    """Causal delta recurrence; native Qwen3.5 heads use cyclic K broadcasting."""
    batch, seq, heads, dim = v.shape
    q = np.asarray(
        q / np.maximum(np.sqrt(np.sum(q * q, axis=-1, keepdims=True)), eps),
        dtype=np.float32,
    )
    k = np.asarray(
        k / np.maximum(np.sqrt(np.sum(k * k, axis=-1, keepdims=True)), eps),
        dtype=np.float32,
    )
    index = np.arange(heads) % q.shape[-2]
    q, k = q[:, :, index], k[:, :, index]
    state = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    output = np.empty_like(v)
    for t in range(seq):
        state *= np.exp(decay[:, t, :, None, None])
        kt = k[:, t]
        predicted = np.einsum("bhvk,bhk->bhv", state, kt)
        delta = (v[:, t] - predicted) * beta[:, t, :, None]
        state += delta[..., None] * kt[:, :, None, :]
        output[:, t] = np.einsum("bhvk,bhk->bhv", state, q[:, t]) / np.float32(
            np.sqrt(dim)
        )
    return output


def recurrent(
    x: FloatArray,
    prefix: str,
    w: GgufWeights,
    cfg: Qwen35Config,
    eps: float,
) -> FloatArray:
    mixed = w.linear(prefix + "attn_qkv.weight", x)
    z = w.linear(prefix + "attn_gate.weight", x)
    beta = sigmoid(w.linear(prefix + "ssm_beta.weight", x))
    alpha = w.linear(prefix + "ssm_alpha.weight", x) + w.load(prefix + "ssm_dt.bias")
    decay = np.logaddexp(np.float32(0), alpha) * w.load(prefix + "ssm_a")
    kernel = w.load(prefix + "ssm_conv1d.weight")
    conv = np.zeros_like(mixed)
    for offset in range(min(cfg.kernel, mixed.shape[1])):
        conv[:, offset:] += (
            mixed[:, : mixed.shape[1] - offset] * kernel[:, cfg.kernel - 1 - offset]
        )
    conv = silu(conv)
    key_dim = cfg.key_heads * cfg.state
    shape = (*x.shape[:2], cfg.key_heads, cfg.state)
    q, k = (
        conv[..., :key_dim].reshape(shape),
        conv[..., key_dim : 2 * key_dim].reshape(shape),
    )
    v = conv[..., 2 * key_dim :].reshape(*x.shape[:2], cfg.value_heads, cfg.state)
    output = gated_delta(q, k, v, decay, beta, eps)
    output = rms_norm(output, w.load(prefix + "ssm_norm.weight"), eps=eps)
    output *= silu(z.reshape(output.shape))
    return w.linear(prefix + "ssm_out.weight", output.reshape(*x.shape[:2], cfg.inner))


def full_attention(
    x: FloatArray,
    prefix: str,
    w: GgufWeights,
    cfg: Qwen35Config,
    p: LlmHyperparams,
) -> FloatArray:
    batch, seq = x.shape[:2]
    qg = w.linear(prefix + "attn_q.weight", x).reshape(
        batch, seq, p.n_head, 2 * cfg.head_dim
    )
    q, gate = np.split(qg, 2, axis=-1)
    k = w.linear(prefix + "attn_k.weight", x).reshape(
        batch, seq, p.n_kv_head, cfg.head_dim
    )
    v = w.linear(prefix + "attn_v.weight", x).reshape(k.shape)
    q = rms_norm(q, w.load(prefix + "attn_q_norm.weight"), eps=p.rms_eps)
    k = rms_norm(k, w.load(prefix + "attn_k_norm.weight"), eps=p.rms_eps)
    cos, sin = rope_cos_sin(seq, cfg.rotary, p.rope_theta)
    # Equal time/height/width positions reduce text MRoPE to partial NeoX RoPE.
    for tensor in (q, k):
        tensor[..., : cfg.rotary] = apply_rope(tensor[..., : cfg.rotary], cos, sin)
    k, v = (np.repeat(t, p.n_head // p.n_kv_head, axis=2) for t in (k, v))
    scores = np.einsum("bthd,bshd->bhts", q, k) / np.float32(np.sqrt(cfg.head_dim))
    scores = np.where(np.triu(np.ones((seq, seq), dtype=bool), 1), -np.inf, scores)
    output = np.einsum("bhts,bshd->bthd", softmax(scores, axis=-1), v) * sigmoid(gate)
    return w.linear(prefix + "attn_output.weight", output.reshape(batch, seq, -1))


def trace_qwen35(
    document: Document,
    feeds: dict[str, np.ndarray],
    store: TensorStore,
    params: LlmHyperparams,
    capture: Callable[[str, np.ndarray], None] | None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[str, ...]]:
    cfg = validate_qwen35(document, params)
    graph = document.main_graph
    tokens = _tokens_from_feeds(
        feeds, graph.inputs, {v.id: v.name or v.id for v in graph.values}
    )
    if (
        tokens.shape[1] > cfg.context
        or np.any(tokens < 0)
        or np.any(tokens >= params.vocab_size)
    ):
        raise LlmTraceError(
            "Token IDs or prompt length exceed this Qwen3.5 model's limits"
        )
    weights = GgufWeights(document, store, params.endian)
    captures: dict[str, np.ndarray] = {}
    emit = capture if capture is not None else captures.__setitem__
    emit(graph.inputs[0], tokens)
    x = weights.embedding(tokens)
    for node in graph.nodes:
        if node.op_type == "Embedding":
            pass
        elif node.op_type in {"Attention", "SSM", "FFN"}:
            # Source bindings, rather than display names, identify the block.
            name = _tensor_name(
                str(next(a.value for a in node.attributes if a.kind is AttrKind.TENSOR))
            )
            prefix = ".".join(name.split(".")[:2]) + "."
            norm = (
                "post_attention_norm.weight"
                if node.op_type == "FFN"
                else "attn_norm.weight"
            )
            hidden = rms_norm(x, weights.load(prefix + norm), eps=params.rms_eps)
            if node.op_type == "FFN":
                gate = silu(weights.linear(prefix + "ffn_gate.weight", hidden))
                mixed = gate * weights.linear(prefix + "ffn_up.weight", hidden)
                x = x + weights.linear(prefix + "ffn_down.weight", mixed)
            elif node.op_type == "SSM":
                x = x + recurrent(hidden, prefix, weights, cfg, params.rms_eps)
            else:
                x = x + full_attention(hidden, prefix, weights, cfg, params)
        elif node.op_type == "RMSNorm":
            x = rms_norm(x, weights.load("output_norm.weight"), eps=params.rms_eps)
        elif node.op_type == "Linear":
            name = (
                "output.weight"
                if "output.weight" in weights.ids
                else "token_embd.weight"
            )
            x = weights.linear(name, x)
        else:
            raise LlmTraceError(f"Unsupported Qwen3.5 operator {node.op_type}")
        if node.outputs:
            emit(node.outputs[0], x)
    return (
        captures,
        ("Qwen3.5 text decoder; recurrent states reset for this prompt.",),
        (),
    )
