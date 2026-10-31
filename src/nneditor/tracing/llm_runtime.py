"""Decoder-only numpy forward pass over a synthesized GGUF or Hugging Face graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from nneditor.adapters.gguf.dequant import GgufDequantError, dequantize
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import AttrKind, Document, JsonValue, Node, Storage
from nneditor.ir.identity import decode_segment
from nneditor.storage.store import TensorStore, TensorUnavailableError
from nneditor.tracing.llm_ops import (
    LlmTraceError,
    attention,
    ffn,
    layer_norm,
    linear,
    rms_norm,
    rope_cos_sin,
)

__all__ = [
    "LlmHyperparams",
    "LlmTraceError",
    "hyperparams_from_document",
    "load_weight",
    "trace_document",
]

Float32Array = NDArray[np.float32]
_CHAIN_OPS: Final = frozenset(
    {"Embedding", "Attention", "FFN", "SSM", "RMSNorm", "Linear"}
)
_SILU_ACTS: Final = frozenset({"silu", "swish", "silu_swiglu"})
_GELU_ACTS: Final = frozenset({"gelu", "gelu_new", "gelu_pytorch_tanh", "quick_gelu"})


@dataclass(frozen=True, slots=True)
class LlmHyperparams:
    """Runtime numbers needed to evaluate a synthesized decoder-only graph."""

    n_embd: int
    n_head: int
    n_kv_head: int
    n_ff: int | None
    vocab_size: int
    rms_eps: float
    rope_theta: float
    hidden_act: str
    endian: str
    architecture: str = "llama"
    layer_norm: bool = False
    use_rope: bool = True
    conv1d: bool = False
    head_dim: int | None = None
    rope_interleaved: bool = False


def _extension_map(document: Document, key: str) -> dict[str, JsonValue]:
    value = dict(document.extensions).get(key)
    return dict(value) if isinstance(value, dict) else {}


def _as_int(value: JsonValue | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
    return value


def _as_float(value: JsonValue | None, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _as_str(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) and value else None


def _arch_lookup(
    metadata: dict[str, JsonValue], architecture: str | None, suffix: str
) -> JsonValue | None:
    if architecture:
        value = metadata.get(f"{architecture}.{suffix}")
        if value is not None:
            return value
    return metadata.get(suffix)


def _activation_name(raw: str | None, architecture: str) -> str:
    lowered = (raw or "").casefold()
    if lowered == "gelu":
        return "gelu_exact"
    if lowered in {"gelu_new", "gelu_pytorch_tanh"} or (
        not lowered and architecture == "gpt2"
    ):
        return "gelu"
    if not lowered or lowered in _SILU_ACTS:
        return "silu"
    raise LlmTraceError(f"Activation {raw!r} cannot be traced by this runtime.")


def _gguf_hyperparams(document: Document) -> LlmHyperparams:
    model = _extension_map(document, "x-gguf.model")
    metadata = _extension_map(document, "x-gguf.metadata")
    architecture = _as_str(model.get("architecture"))
    n_embd = _as_int(model.get("embedding_length")) or _as_int(
        _arch_lookup(metadata, architecture, "embedding_length")
    )
    n_head = _as_int(_arch_lookup(metadata, architecture, "attention.head_count"))
    n_kv = _as_int(_arch_lookup(metadata, architecture, "attention.head_count_kv"))
    n_ff = _as_int(_arch_lookup(metadata, architecture, "feed_forward_length"))
    vocab = _as_int(model.get("vocab_size"))
    if n_embd is None or n_head is None or vocab is None:
        raise LlmTraceError(
            "GGUF metadata does not declare embedding length, head count, "
            "and vocabulary size needed to run a numpy trace"
        )
    gpt2 = architecture == "gpt2"
    return LlmHyperparams(
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv or n_head,
        n_ff=n_ff,
        vocab_size=vocab,
        rms_eps=_as_float(
            _arch_lookup(
                metadata,
                architecture,
                "attention.layer_norm_epsilon"
                if gpt2
                else "attention.layer_norm_rms_epsilon"
                if architecture == "qwen35"
                else "attention.layer_norm_rms_eps",
            ),
            1e-5,
        ),
        rope_theta=_as_float(
            _arch_lookup(metadata, architecture, "rope.freq_base"), 10000.0
        ),
        hidden_act=_activation_name(None, architecture or ""),
        endian=_as_str(model.get("endian")) or "little",
        architecture=architecture or "unknown",
        layer_norm=gpt2,
        use_rope=not gpt2,
        head_dim=_as_int(_arch_lookup(metadata, architecture, "attention.key_length")),
        rope_interleaved=architecture == "llama",
    )


def _hf_hyperparams(document: Document) -> LlmHyperparams:
    model = _extension_map(document, "x-huggingface.model")
    config = _extension_map(document, "x-huggingface.config")
    architecture = (
        _as_str(model.get("architecture")) or _as_str(config.get("model_type")) or ""
    )
    n_embd = _as_int(config.get("hidden_size")) or _as_int(
        model.get("embedding_length")
    )
    n_head = _as_int(config.get("num_attention_heads")) or _as_int(config.get("n_head"))
    n_kv = _as_int(config.get("num_key_value_heads")) or n_head
    n_ff = _as_int(config.get("intermediate_size")) or _as_int(config.get("n_inner"))
    vocab = _as_int(config.get("vocab_size")) or _as_int(model.get("vocab_size"))
    if n_embd is None or n_head is None or vocab is None:
        raise LlmTraceError(
            "Hugging Face config.json does not declare hidden size, head count, "
            "and vocabulary size needed to run a numpy trace"
        )
    eps = config.get("rms_norm_eps")
    if eps is None:
        eps = config.get("layer_norm_epsilon")
    if eps is None:
        eps = config.get("layer_norm_eps")
    return LlmHyperparams(
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv or n_head,
        n_ff=n_ff,
        vocab_size=vocab,
        rms_eps=_as_float(eps, 1e-5),
        rope_theta=_as_float(config.get("rope_theta"), 10000.0),
        hidden_act=_activation_name(
            _as_str(config.get("hidden_act"))
            or _as_str(config.get("activation_function")),
            architecture,
        ),
        endian="little",
        architecture=architecture,
        layer_norm=architecture == "gpt2",
        use_rope=architecture != "gpt2",
        conv1d=architecture == "gpt2",
        head_dim=_as_int(config.get("head_dim")),
    )


def hyperparams_from_document(document: Document) -> LlmHyperparams:
    """Read runtime hyperparameters from GGUF KV or Hugging Face config."""
    if document.artifact_kind is ArtifactKind.GGUF_MODEL:
        return _gguf_hyperparams(document)
    if document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL:
        return _hf_hyperparams(document)
    raise LlmTraceError(f"{document.artifact_kind.value} has no numpy LLM runtime")


def _tensor_name(tensor_id: str) -> str:
    _, separator, encoded = tensor_id.partition("#")
    if not separator:
        return tensor_id
    return decode_segment(encoded)


def load_weight(store: TensorStore, tensor_id: str, endian: str) -> Float32Array:
    """Read one tensor through the store and decode it to float32."""
    tensor = store.metadata(tensor_id)
    if tensor.storage is Storage.ABSENT:
        raise LlmTraceError(f"weight {_tensor_name(tensor_id)!r} has no readable bytes")
    try:
        payload = store.read(tensor_id)
    except TensorUnavailableError as error:
        raise LlmTraceError(str(error)) from error
    try:
        return dequantize(tensor.element_type, tensor.dims, payload, endian=endian)
    except GgufDequantError as error:
        raise LlmTraceError(str(error)) from error


def _bare_name(name: str) -> str:
    lowered = name.casefold().replace("-", "_")
    if lowered.endswith(".weight"):
        return lowered[: -len(".weight")]
    if lowered.endswith(".bias"):
        return lowered[: -len(".bias")]
    return lowered


def _ends_with(name: str, *suffixes: str) -> bool:
    lowered = _bare_name(name)
    return any(
        lowered.endswith(suffix) or f".{suffix}" in lowered or f"_{suffix}" in lowered
        for suffix in suffixes
    )


def _attention_role(name: str) -> str | None:
    if _ends_with(name, "attn_qkv", "c_attn", "qkv_proj", "query_key_value"):
        return "qkv"
    if _ends_with(name, "q_norm", "q_layernorm"):
        return "q_norm"
    if _ends_with(name, "k_norm", "k_layernorm"):
        return "k_norm"
    if _ends_with(name, "q_proj", "attn_q", "query", "wq"):
        return "q"
    if _ends_with(name, "k_proj", "attn_k", "key", "wk"):
        return "k"
    if _ends_with(name, "v_proj", "attn_v", "value", "wv"):
        return "v"
    if _ends_with(name, "o_proj", "attn_output", "out_proj", "wo", "c_proj"):
        return "o"
    if _ends_with(
        name,
        "attn_norm",
        "input_layernorm",
        "ln_1",
        "pre_attention_layernorm",
        "attention_norm",
    ):
        return "norm"
    return None


def _ffn_role(name: str) -> str | None:
    if _ends_with(name, "gate_proj", "ffn_gate", "w1"):
        return "gate"
    if _ends_with(name, "up_proj", "ffn_up", "w3", "c_fc"):
        return "up"
    if _ends_with(name, "down_proj", "ffn_down", "w2", "c_proj"):
        return "down"
    if _ends_with(
        name,
        "ffn_norm",
        "post_attention_layernorm",
        "ln_2",
        "pre_ffn_layernorm",
        "post_feedforward_layernorm",
    ):
        return "norm"
    return None


def _embed_role(name: str) -> str | None:
    if _ends_with(
        name, "token_embd", "embed_tokens", "wte", "tok_embeddings", "word_embeddings"
    ):
        return "token"
    if _ends_with(name, "position_embd", "wpe", "position_embeddings"):
        return "position"
    return "token"


def _node_label(node: Node) -> str:
    return node.source_name or node.op_type


def _weight_key(name: str, role: str) -> str:
    if name.casefold().endswith(".bias"):
        return f"{role}_bias"
    return role


def _node_weights(
    node: Node,
    store: TensorStore,
    endian: str,
    role_of: Callable[[str], str | None],
) -> dict[str, Float32Array]:
    loaded: dict[str, Float32Array] = {}
    for attribute in node.attributes:
        if attribute.kind is not AttrKind.TENSOR:
            continue
        tensor_id = str(attribute.value)
        name = _tensor_name(tensor_id)
        role = role_of(name)
        if role is None:
            continue
        loaded[_weight_key(name, role)] = load_weight(store, tensor_id, endian)
    return loaded


def _linear_role(
    hidden: Float32Array,
    weights: dict[str, Float32Array],
    role: str,
    node_name: str,
    *,
    required: bool = True,
    conv1d: bool = False,
) -> Float32Array:
    if role not in weights:
        if required:
            raise LlmTraceError(f"{node_name} is missing required weight {role!r}")
        return hidden
    return linear(hidden, weights[role], weights.get(f"{role}_bias"), conv1d=conv1d)


def _normalize(
    hidden: Float32Array, weights: dict[str, Float32Array], params: LlmHyperparams
) -> Float32Array:
    if "norm" not in weights:
        return hidden
    if params.layer_norm:
        return layer_norm(
            hidden, weights["norm"], params.rms_eps, weights.get("norm_bias")
        )
    return rms_norm(hidden, weights["norm"], params.rms_eps)


def _require(
    weights: dict[str, Float32Array], role: str, node_name: str
) -> Float32Array:
    try:
        return weights[role]
    except KeyError as error:
        raise LlmTraceError(
            f"{node_name} is missing required weight {role!r}"
        ) from error


def _tokens_from_feeds(
    feeds: dict[str, np.ndarray], graph_inputs: tuple[str, ...], names: dict[str, str]
) -> NDArray[np.integer]:
    if not graph_inputs:
        raise LlmTraceError("the synthesized graph has no token input")
    input_id = graph_inputs[0]
    name = names.get(input_id, "tokens")
    if name not in feeds and input_id not in feeds:
        raise LlmTraceError(f"trace feeds do not include input {name!r}")
    array = feeds.get(name, feeds.get(input_id))
    assert array is not None
    tokens = np.asarray(array)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise LlmTraceError("token input must contain integer ids")
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if tokens.ndim != 2:
        raise LlmTraceError(
            f"token input must have shape [batch, seq], got {tokens.shape}"
        )
    if not tokens.size:
        raise LlmTraceError("token input must contain at least one token")
    return tokens.astype(np.int64, copy=False)


def _split_qkv(
    fused: Float32Array, n_head: int, n_kv_head: int, head_dim: int
) -> tuple[Float32Array, Float32Array, Float32Array]:
    q_dim = n_head * head_dim
    kv_dim = n_kv_head * head_dim
    expected = q_dim + 2 * kv_dim
    if fused.shape[-1] != expected:
        raise LlmTraceError(
            f"fused qkv width {fused.shape[-1]} does not match "
            f"{expected} for {n_head} query and {n_kv_head} kv heads"
        )
    query = fused[..., :q_dim]
    key = fused[..., q_dim : q_dim + kv_dim]
    value = fused[..., q_dim + kv_dim :]
    return query, key, value


def _apply_embedding(
    tokens: NDArray[np.integer],
    node: Node,
    store: TensorStore,
    params: LlmHyperparams,
) -> Float32Array:
    weights = _node_weights(node, store, params.endian, _embed_role)
    table = _require(weights, "token", _node_label(node))
    if int(tokens.max(initial=0)) >= table.shape[0] or int(tokens.min(initial=0)) < 0:
        raise LlmTraceError(
            f"token ids fall outside the embedding vocabulary of {table.shape[0]}"
        )
    hidden = np.asarray(table[tokens], dtype=np.float32)
    position = weights.get("position")
    if position is not None:
        seq = tokens.shape[1]
        if seq > position.shape[0]:
            raise LlmTraceError(
                f"prompt has {seq} tokens; position table supports {position.shape[0]}"
            )
        hidden = hidden + position[:seq]
    return hidden


def _apply_attention(
    residual: Float32Array,
    node: Node,
    store: TensorStore,
    params: LlmHyperparams,
    cos: Float32Array,
    sin: Float32Array,
) -> Float32Array:
    weights = _node_weights(node, store, params.endian, _attention_role)
    hidden = _normalize(residual, weights, params)
    if "qkv" in weights:
        fused = _linear_role(
            hidden, weights, "qkv", _node_label(node), conv1d=params.conv1d
        )
        query, key, value = _split_qkv(
            fused,
            params.n_head,
            params.n_kv_head,
            params.head_dim or params.n_embd // params.n_head,
        )
    else:
        query = _linear_role(hidden, weights, "q", _node_label(node))
        key = _linear_role(hidden, weights, "k", _node_label(node))
        value = _linear_role(hidden, weights, "v", _node_label(node))
    mixed = attention(
        query,
        key,
        value,
        n_head=params.n_head,
        n_kv_head=params.n_kv_head,
        cos=cos,
        sin=sin,
        use_rope=params.use_rope,
        interleaved=params.rope_interleaved,
        q_norm=weights.get("q_norm"),
        k_norm=weights.get("k_norm"),
        norm_eps=params.rms_eps,
    )
    if "o" in weights:
        mixed = _linear_role(
            mixed, weights, "o", _node_label(node), conv1d=params.conv1d
        )
    return residual + mixed


def _apply_ffn(
    residual: Float32Array,
    node: Node,
    store: TensorStore,
    params: LlmHyperparams,
) -> Float32Array:
    weights = _node_weights(node, store, params.endian, _ffn_role)
    hidden = _normalize(residual, weights, params)
    up = weights.get("up")
    down = _require(weights, "down", _node_label(node))
    if up is None:
        # A lone down-projection cannot form an MLP; refuse rather than
        # silently treating it as a residual pass-through.
        raise LlmTraceError(f"{_node_label(node)} is missing an up or gate projection")
    mixed = ffn(
        hidden,
        up=up,
        down=down,
        gate=weights.get("gate"),
        activation=params.hidden_act,
        up_bias=weights.get("up_bias"),
        down_bias=weights.get("down_bias"),
        gate_bias=weights.get("gate_bias"),
        conv1d=params.conv1d,
    )
    return residual + mixed


def _apply_node(
    node: Node,
    residual: Float32Array,
    tokens: NDArray[np.integer],
    store: TensorStore,
    params: LlmHyperparams,
    cos: Float32Array,
    sin: Float32Array,
    notes: list[str],
) -> Float32Array:
    if node.op_type == "Embedding":
        return _apply_embedding(tokens, node, store, params)
    if node.op_type == "Attention":
        return _apply_attention(residual, node, store, params, cos, sin)
    if node.op_type == "FFN":
        return _apply_ffn(residual, node, store, params)
    if node.op_type == "RMSNorm":
        weights = _node_weights(node, store, params.endian, lambda _name: "norm")
        scale = _require(weights, "norm", _node_label(node))
        return _normalize(residual, {**weights, "norm": scale}, params)
    if node.op_type == "Linear":
        weights = _node_weights(node, store, params.endian, lambda _name: "weight")
        return _linear_role(residual, weights, "weight", _node_label(node))
    if node.op_type == "SSM":
        raise LlmTraceError(
            "State-space blocks cannot be traced by this runtime; "
            "weight inspection remains available."
        )
    raise LlmTraceError(f"unsupported synthesized operator {node.op_type!r}")


def trace_document(
    document: Document,
    feeds: dict[str, np.ndarray],
    store: TensorStore,
    *,
    capture: Callable[[str, np.ndarray], None] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[str, ...]]:
    """Evaluate the synthesized decoder-only chain, capturing each operator."""
    from nneditor.tracing.llm_support import validate_llm_trace

    validate_llm_trace(document)
    params = hyperparams_from_document(document)
    if params.architecture == "qwen35":
        from nneditor.tracing.llm_qwen35 import trace_qwen35

        return trace_qwen35(document, feeds, store, params, capture)
    if min(params.n_embd, params.n_head, params.n_kv_head) <= 0:
        raise LlmTraceError("embedding width and head counts must be positive")
    if params.head_dim is None and params.n_embd % params.n_head:
        raise LlmTraceError(
            f"embedding length {params.n_embd} is not divisible by "
            f"head count {params.n_head}"
        )
    graph = document.main_graph
    names = {value.id: value.name or value.id for value in graph.values}
    tokens = _tokens_from_feeds(feeds, graph.inputs, names)
    seq = int(tokens.shape[1])
    head_dim = params.head_dim or params.n_embd // params.n_head
    if params.use_rope:
        cos, sin = rope_cos_sin(seq, head_dim, params.rope_theta)
    else:
        cos = sin = np.empty((0, 0), dtype=np.float32)
    captures: dict[str, np.ndarray] = {}
    emit = capture if capture is not None else captures.__setitem__
    emit(graph.inputs[0], tokens)
    notes: list[str] = []
    diagnostics: list[str] = []
    residual = np.zeros((tokens.shape[0], seq, params.n_embd), dtype=np.float32)
    for node in graph.nodes:
        if node.op_type not in _CHAIN_OPS:
            continue
        residual = _apply_node(node, residual, tokens, store, params, cos, sin, notes)
        if node.outputs:
            emit(node.outputs[0], residual)
    return captures, tuple(notes), tuple(diagnostics)
