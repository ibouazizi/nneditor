"""Numpy primitives for a decoder-only transformer forward pass."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LlmTraceError",
    "attention",
    "ffn",
    "gelu",
    "linear",
    "repeat_kv",
    "rms_norm",
    "rope_cos_sin",
    "silu",
    "softmax",
]

Float32Array = NDArray[np.float32]


class LlmTraceError(RuntimeError):
    """The numpy LLM worker could not evaluate this graph."""


def rms_norm(hidden: Float32Array, weight: Float32Array, eps: float) -> Float32Array:
    """Llama-style RMSNorm over the last axis."""
    variance = np.mean(np.square(hidden, dtype=np.float32), axis=-1, keepdims=True)
    return hidden * np.reciprocal(np.sqrt(variance + eps)) * weight


def layer_norm(
    hidden: Float32Array,
    weight: Float32Array,
    eps: float,
    bias: Float32Array | None = None,
) -> Float32Array:
    """GPT-2 LayerNorm, including centering and the learned bias."""
    centered = hidden - np.mean(hidden, axis=-1, keepdims=True)
    normalized = rms_norm(centered, weight, eps)
    return normalized if bias is None else normalized + bias


def silu(hidden: Float32Array) -> Float32Array:
    return (hidden * (1.0 / (1.0 + np.exp(-hidden)))).astype(np.float32, copy=False)


def gelu(hidden: Float32Array) -> Float32Array:
    """Tanh GELU used by GPT-2 (`gelu_new`)."""
    inner = math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden**3)
    return (0.5 * hidden * (1.0 + np.tanh(inner))).astype(np.float32, copy=False)


def _activation(hidden: Float32Array, name: str) -> Float32Array:
    if name == "silu":
        return silu(hidden)
    if name == "gelu_exact":
        erf = np.fromiter(
            (math.erf(float(value) / math.sqrt(2)) for value in hidden.flat),
            dtype=np.float32,
            count=hidden.size,
        ).reshape(hidden.shape)
        return (0.5 * hidden * (1.0 + erf)).astype(np.float32, copy=False)
    return gelu(hidden)


def softmax(scores: Float32Array, axis: int = -1) -> Float32Array:
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32, copy=False)


def linear(
    hidden: Float32Array,
    weight: Float32Array,
    bias: Float32Array | None = None,
    *,
    conv1d: bool = False,
) -> Float32Array:
    """Apply a Linear (`x @ W.T`) or Conv1D (`x @ W`) weight."""
    if weight.ndim != 2:
        raise LlmTraceError(f"expected a rank-2 weight, got shape {weight.shape}")
    width = hidden.shape[-1]
    matrix = weight if conv1d else weight.T
    if matrix.shape[0] != width:
        raise LlmTraceError(
            f"weight shape {weight.shape} does not match last input extent {width}"
        )
    result = hidden @ matrix
    if bias is not None:
        result = result + bias
    return result


def rope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    *,
    positions: NDArray[np.integer] | None = None,
) -> tuple[Float32Array, Float32Array]:
    """NeoX-style RoPE cos/sin tables of shape ``[seq, head_dim]``."""
    if head_dim % 2:
        raise LlmTraceError(f"RoPE head dimension {head_dim} must be even")
    index = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (index / head_dim))
    if positions is None:
        positions = np.arange(seq_len, dtype=np.float32)
    else:
        positions = positions.astype(np.float32, copy=False)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(
    hidden: Float32Array,
    cos: Float32Array,
    sin: Float32Array,
    *,
    interleaved: bool = False,
) -> Float32Array:
    """Rotate half-channels of ``[batch, seq, heads, dim]``."""
    half = hidden.shape[-1] // 2
    if interleaved:
        rotated = np.empty_like(hidden)
        rotated[..., ::2] = -hidden[..., 1::2]
        rotated[..., 1::2] = hidden[..., ::2]
        cos = np.repeat(cos[..., :half], 2, axis=-1)
        sin = np.repeat(sin[..., :half], 2, axis=-1)
    else:
        rotated = np.concatenate([-hidden[..., half:], hidden[..., :half]], axis=-1)
    table = cos[None, :, None, :]
    return (hidden * table + rotated * sin[None, :, None, :]).astype(
        np.float32, copy=False
    )


def repeat_kv(hidden: Float32Array, repeats: int) -> Float32Array:
    """Expand GQA key/value heads to the query head count."""
    if repeats == 1:
        return hidden
    batch, seq, heads, dim = hidden.shape
    expanded = np.repeat(hidden[:, :, :, None, :], repeats, axis=3)
    return expanded.reshape(batch, seq, heads * repeats, dim)


def _causal_mask(query_len: int, key_len: int) -> NDArray[np.bool_]:
    return np.triu(np.ones((query_len, key_len), dtype=bool), k=1)


def attention(
    query: Float32Array,
    key: Float32Array,
    value: Float32Array,
    *,
    n_head: int,
    n_kv_head: int,
    cos: Float32Array,
    sin: Float32Array,
    use_rope: bool = True,
    interleaved: bool = False,
    q_norm: Float32Array | None = None,
    k_norm: Float32Array | None = None,
    norm_eps: float = 1e-5,
) -> Float32Array:
    """Scaled dot-product attention with RoPE and an optional GQA repeat."""
    batch, seq, width = query.shape
    head_dim = width // n_head
    if head_dim * n_head != width:
        raise LlmTraceError(
            f"query width {width} is not divisible by head count {n_head}"
        )
    kv_dim = key.shape[-1] // n_kv_head
    if kv_dim != head_dim:
        raise LlmTraceError(
            f"key/value head dim {kv_dim} does not match query head dim {head_dim}"
        )
    query_h = query.reshape(batch, seq, n_head, head_dim)
    key_h = key.reshape(batch, seq, n_kv_head, kv_dim)
    if q_norm is not None:
        query_h = rms_norm(query_h, q_norm, norm_eps)
    if k_norm is not None:
        key_h = rms_norm(key_h, k_norm, norm_eps)
    if use_rope:
        query_h = apply_rope(query_h, cos, sin, interleaved=interleaved)
        key_h = apply_rope(key_h, cos, sin, interleaved=interleaved)
    value_h = value.reshape(batch, seq, n_kv_head, kv_dim)
    repeats = n_head // n_kv_head
    if repeats * n_kv_head != n_head:
        raise LlmTraceError(
            f"head count {n_head} is not a multiple of kv heads {n_kv_head}"
        )
    key_h = repeat_kv(key_h, repeats)
    value_h = repeat_kv(value_h, repeats)
    query_h = np.transpose(query_h, (0, 2, 1, 3))
    key_h = np.transpose(key_h, (0, 2, 1, 3))
    value_h = np.transpose(value_h, (0, 2, 1, 3))
    scale = 1.0 / math.sqrt(head_dim)
    scores = (query_h @ np.transpose(key_h, (0, 1, 3, 2))) * scale
    scores = np.where(_causal_mask(seq, key_h.shape[2]), np.float32(-1e9), scores)
    weights = softmax(scores.astype(np.float32), axis=-1)
    mixed = weights @ value_h
    mixed = np.transpose(mixed, (0, 2, 1, 3)).reshape(batch, seq, width)
    return mixed


def ffn(
    hidden: Float32Array,
    *,
    up: Float32Array,
    down: Float32Array,
    gate: Float32Array | None,
    activation: str,
    up_bias: Float32Array | None = None,
    down_bias: Float32Array | None = None,
    gate_bias: Float32Array | None = None,
    conv1d: bool = False,
) -> Float32Array:
    """SwiGLU (Llama) or GELU (GPT-2) feed-forward block without residual."""
    projected = linear(hidden, up, up_bias, conv1d=conv1d)
    if gate is not None:
        gated = linear(hidden, gate, gate_bias, conv1d=conv1d)
        activated = _activation(gated, activation)
        inner = activated * projected
    else:
        inner = _activation(projected, activation)
    return linear(inner, down, down_bias, conv1d=conv1d)
