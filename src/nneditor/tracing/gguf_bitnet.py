"""Load a float BitNet GGUF into the Hugging Face module and capture it."""

from __future__ import annotations

import importlib
import re
from typing import Any, cast

import numpy as np

from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError

_LEAVES = {
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_sub_norm": "self_attn.attn_sub_norm.weight",
    "attn_norm": "input_layernorm.weight",
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
    "ffn_sub_norm": "mlp.ffn_sub_norm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}
_BLOCK = re.compile(r"blk\.(\d+)\.(.+)\.weight\Z")


def _integer(metadata: Any, key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TracePluginError(f"BitNet GGUF needs a positive {key}.")
    return int(value)


def _state_key(name: str) -> str | None:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    match = _BLOCK.match(name)
    if match is None:
        return None
    leaf = _LEAVES.get(match.group(2))
    if leaf is None:
        return None
    return f"model.layers.{match.group(1)}.{leaf}"


def _model(context: TraceContext) -> Any:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    meta = context.metadata
    config = transformers.BitNetConfig(
        hidden_size=_integer(meta, "bitnet.embedding_length"),
        intermediate_size=_integer(meta, "bitnet.feed_forward_length"),
        num_hidden_layers=_integer(meta, "bitnet.block_count"),
        num_attention_heads=_integer(meta, "bitnet.attention.head_count"),
        num_key_value_heads=_integer(meta, "bitnet.attention.head_count_kv"),
        vocab_size=_integer(meta, "bitnet.vocab_size"),
        bos_token_id=1,
        eos_token_id=2,
    )
    model = transformers.BitNetForCausalLM(config)
    mapped: dict[str, Any] = {}
    for tensor in context.document.tensors.values():
        from nneditor.tracing.llm_runtime import _tensor_name

        name = _tensor_name(tensor.id)
        key = _state_key(name)
        if key is None:
            continue
        array = context.load_weight(tensor.id)
        mapped[key] = torch.from_numpy(np.array(array, copy=True))
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise TracePluginError(
            "BitNet GGUF weights do not match the Hugging Face module layout."
        )
    model.eval()
    model.config.use_cache = False
    return model


def _hook_target(model: Any, node: Any) -> Any | None:
    modules = dict(model.named_modules())
    if node.op_type == "Embedding":
        return modules["model.embed_tokens"]
    if node.op_type == "RMSNorm":
        return modules["model.norm"]
    if node.op_type == "Linear":
        return modules["lm_head"]
    match = re.search(r"block-(\d+)", node.source_name or "")
    if match is None:
        return None
    index = int(match.group(1))
    if node.op_type == "Attention":
        return modules[f"model.layers.{index}.self_attn"]
    if node.op_type == "FFN":
        return modules[f"model.layers.{index}.mlp"]
    return None


def trace_gguf_bitnet(context: TraceContext) -> TraceOutcome:
    """Run one BitNet GGUF prompt and capture the synthesized decoder ports."""
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("transformers")
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-modalities] to run this tracer."
        ) from error
    tokens = context.inputs.get("tokens")
    if tokens is None or tokens.ndim != 2 or tokens.dtype.kind not in "iu":
        raise TracePluginError("BitNet needs integer tokens shaped [batch, sequence].")
    model = _model(context)
    graph = context.document.main_graph
    handles: list[Any] = []

    def capture(key: str) -> Any:
        def save(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                raise TracePluginError("A BitNet module did not return a tensor.")
            context.capture(
                key, cast(np.ndarray[Any, Any], value.detach().float().cpu().numpy())
            )

        return save

    try:
        for node in graph.nodes:
            target = _hook_target(model, node)
            if target is None:
                continue
            handles.append(target.register_forward_hook(capture(node.outputs[0])))
        context.capture_inputs()
        with torch.inference_mode():
            model(
                input_ids=torch.from_numpy(np.array(tokens, dtype=np.int64, copy=True)),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return TraceOutcome(
        provider="pytorch-cpu",
        notes=("BitNet GGUF: one CPU forward pass with fresh cache.",),
    )
