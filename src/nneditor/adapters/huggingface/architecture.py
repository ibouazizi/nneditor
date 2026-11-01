"""Derive a synthetic LLM graph from Hugging Face tensor names and config.

A Transformers snapshot stores weights, not operators. Tensor names
(``model.layers.N.*``, ``transformer.h.N.*``, embeddings, ``lm_head``) plus
``config.json`` become a chain of Embedding, Attention or SSM, FFN, RMSNorm,
and Linear nodes in the ``nneditor.llm`` domain.

``source_name`` uses hyphens (``llm/block-000/attn``) because the source-scope
detector splits on ``.``. Pattern and repeated detectors stay off when
provenance records the synthesis.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from nneditor.ir.core import JsonValue

__all__ = [
    "ArchitecturePlan",
    "LayerSpec",
    "WeightBinding",
    "is_embedding_name",
    "is_output_name",
    "layer_index",
    "plan_architecture",
]

_LAYER_PATTERNS: Final = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),
    re.compile(r"(?:^|\.)blocks\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
    re.compile(r"(?:^|\.)layer\.(\d+)\."),
    re.compile(r"(?:^|\.)block\.(\d+)\."),
)
_EMBED_MARKERS: Final = (
    "embed_tokens",
    "word_embeddings",
    "token_embd",
    "tok_embeddings",
    "embed_in",
    "wte",
    "wpe",
    "position_embeddings",
    "embeddings.position",
)
_OUTPUT_MARKERS: Final = (
    "lm_head",
    "embed_out",
    "pred_proj",
    "cls.predictions",
    "output.weight",
)
_NORM_MARKERS: Final = (
    "backbone.norm_f",
    "model.norm",
    "model.language_model.norm",
    "ln_f",
    "final_layernorm",
    "final_layer_norm",
    "gpt_neox.final_layer_norm",
)
_ATTN_MARKERS: Final = (
    "self_attn",
    "self_attention",
    "cross_attn",
    "attention",
    "attn",
    "mixer.qkv",
    "ln_1",
    "input_layernorm",
    "pre_attention_layernorm",
)
_FFN_MARKERS: Final = (
    "mlp",
    "ffn",
    "feed_forward",
    "moe",
    "experts",
    "intermediate",
    "ln_2",
    "post_attention_layernorm",
    "pre_ffn_layernorm",
)
_SSM_MARKERS: Final = ("ssm", "mamba")
# These FFN names contain the substring "attn", so they must win before the
# generic attention markers.
_PRE_FFN_NORM_MARKERS: Final = (
    "post_attention_layernorm",
    "post_feedforward_layernorm",
    "pre_ffn_layernorm",
)


@dataclass(frozen=True, slots=True)
class WeightBinding:
    """One tensor attached to a synthesized operator as a TENSOR attribute."""

    attribute: str
    tensor_name: str


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One synthesized operator, identified by a stable source name."""

    name: str
    op_type: str
    source_name: str
    weights: tuple[WeightBinding, ...]
    block: int | None


@dataclass(frozen=True, slots=True)
class ArchitecturePlan:
    """Everything the IR converter needs to build the synthetic graph."""

    architecture: str
    name: str
    block_count: int
    vocab_size: int | None
    embedding_length: int | None
    context_length: int | None
    tied_embeddings: bool
    layers: tuple[LayerSpec, ...]


def _json_str(config: Mapping[str, JsonValue], *keys: str) -> str:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _json_int(config: Mapping[str, JsonValue], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _json_bool(config: Mapping[str, JsonValue], key: str) -> bool | None:
    value = config.get(key)
    return value if isinstance(value, bool) else None


def _architecture_name(config: Mapping[str, JsonValue]) -> str:
    model_type = _json_str(config, "model_type", "_class_name")
    if model_type:
        return model_type
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        first = architectures[0]
        if isinstance(first, str) and first:
            return first
    return "huggingface"


def layer_index(name: str) -> int | None:
    """The transformer block index encoded in ``name``, if any."""
    return _layer_index(name)


def is_embedding_name(name: str) -> bool:
    """Whether ``name`` is a token or position embedding weight."""
    return _is_embed(name)


def is_output_name(name: str) -> bool:
    """Whether ``name`` is a language-model head weight."""
    return _is_output(name)


def _layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    return None


def _contains_marker(name: str, markers: tuple[str, ...]) -> bool:
    lowered = name.casefold()
    return any(marker in lowered for marker in markers)


def _classify_block(name: str) -> str:
    lowered = name.casefold()
    if any(marker in lowered for marker in _SSM_MARKERS):
        return "ssm"
    if any(marker in lowered for marker in _PRE_FFN_NORM_MARKERS):
        return "ffn"
    if any(marker in lowered for marker in _ATTN_MARKERS):
        return "attn"
    if any(marker in lowered for marker in _FFN_MARKERS):
        return "ffn"
    return "extra"


def _attr_name(tensor_name: str) -> str:
    return tensor_name


def _bindings(names: list[str]) -> tuple[WeightBinding, ...]:
    return tuple(WeightBinding(_attr_name(name), name) for name in names)


def _empty_kinds() -> dict[str, list[str]]:
    return {"attn": [], "ffn": [], "ssm": [], "extra": []}


def _partition_blocks(names: tuple[str, ...]) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = {}
    for name in names:
        index = _layer_index(name)
        if index is None:
            continue
        kinds = grouped.setdefault(index, _empty_kinds())
        kinds[_classify_block(name)].append(name)
    for kinds in grouped.values():
        for group in kinds.values():
            group.sort()
    return grouped


def _is_embed(name: str) -> bool:
    if _layer_index(name) is not None:
        return False
    return _contains_marker(name, _EMBED_MARKERS) or name.endswith(".embeddings.weight")


def _is_output(name: str) -> bool:
    if _layer_index(name) is not None:
        return False
    return _contains_marker(name, _OUTPUT_MARKERS)


def _is_output_norm(name: str) -> bool:
    if _layer_index(name) is not None:
        return False
    return _contains_marker(name, _NORM_MARKERS)


def _sublayer(
    kind: str, names: list[str], block: int, op_type: str
) -> LayerSpec | None:
    if not names:
        return None
    label = f"{block:03d}"
    return LayerSpec(
        name=f"block-{label}-{kind}",
        op_type=op_type,
        source_name=f"llm/block-{label}/{kind}",
        weights=_bindings(names),
        block=block,
    )


def _block_layers(grouped: dict[int, dict[str, list[str]]]) -> list[LayerSpec]:
    layers: list[LayerSpec] = []
    for block in sorted(grouped):
        kinds = grouped[block]
        for kind, op_type in (
            ("attn", "Attention"),
            ("ssm", "SSM"),
            ("ffn", "FFN"),
            ("extra", "Parameters"),
        ):
            spec = _sublayer(kind, kinds[kind], block, op_type)
            if spec is not None:
                layers.append(spec)
    return layers


def _global_layers(names: tuple[str, ...], *, tied: bool) -> list[LayerSpec]:
    embed = sorted(name for name in names if _is_embed(name))
    output_norm = sorted(name for name in names if _is_output_norm(name))
    output = sorted(name for name in names if _is_output(name))
    layers: list[LayerSpec] = []
    if embed:
        layers.append(
            LayerSpec("embed", "Embedding", "llm/embed", _bindings(embed), None)
        )
    if output_norm:
        layers.append(
            LayerSpec(
                "output-norm",
                "RMSNorm",
                "llm/norm",
                _bindings(output_norm),
                None,
            )
        )
    head = output
    if not head and tied:
        head = [
            name
            for name in embed
            if name.endswith(
                (
                    "embed_tokens.weight",
                    "wte.weight",
                    "word_embeddings.weight",
                    "tok_embeddings.weight",
                    "embed_in.weight",
                    "embeddings.weight",
                )
            )
        ][:1]
    if head:
        layers.append(
            LayerSpec("lm-head", "Linear", "llm/lm_head", _bindings(head), None)
        )
    return layers


def _fallback_layers(names: tuple[str, ...]) -> tuple[LayerSpec, ...]:
    if not names:
        return ()
    return (
        LayerSpec(
            "weights",
            "Parameters",
            "llm/weights",
            _bindings(sorted(names)),
            None,
        ),
    )


def _embedding_length(
    config: Mapping[str, JsonValue],
    names: tuple[str, ...],
    tensors: Mapping[str, tuple[int, ...]],
) -> int | None:
    declared = _json_int(config, "hidden_size", "n_embd", "d_model", "n_embed")
    if declared is not None:
        return declared
    for name in names:
        if _is_embed(name):
            dims = tensors.get(name, ())
            if dims:
                return dims[-1]
    return None


def _vocab_size(
    config: Mapping[str, JsonValue],
    names: tuple[str, ...],
    tensors: Mapping[str, tuple[int, ...]],
) -> int | None:
    declared = _json_int(config, "vocab_size")
    if declared is not None:
        return declared
    for name in names:
        if _is_embed(name) or _is_output(name):
            dims = tensors.get(name, ())
            if dims:
                return dims[0]
    return None


def plan_architecture(
    config: Mapping[str, JsonValue],
    tensor_names: tuple[str, ...],
    tensor_dims: Mapping[str, tuple[int, ...]] | None = None,
) -> ArchitecturePlan:
    """Group Hugging Face tensors into a stable, source-named LLM chain."""
    dims = tensor_dims or {}
    grouped = _partition_blocks(tensor_names)
    if _architecture_name(config) in {"mamba", "mamba2"}:
        # A Mamba block includes its normalization, selective mixer, and
        # residual. Capture the whole block as one SSM operation.
        grouped = {
            index: {
                "attn": [],
                "ffn": [],
                "extra": [],
                "ssm": sorted(name for group in kinds.values() for name in group),
            }
            for index, kinds in grouped.items()
        }
    declared_blocks = _json_int(
        config, "num_hidden_layers", "n_layer", "num_layers", "n_layers"
    )
    block_count = (
        declared_blocks
        if declared_blocks is not None
        else (max(grouped) + 1 if grouped else 0)
    )
    tied = _json_bool(config, "tie_word_embeddings")
    if tied is None:
        tied = not any(_is_output(name) for name in tensor_names) and any(
            _is_embed(name) for name in tensor_names
        )
    block_specs = _block_layers(grouped)
    global_specs = _global_layers(tensor_names, tied=tied)
    embed = [spec for spec in global_specs if spec.op_type == "Embedding"]
    tail = [spec for spec in global_specs if spec.op_type != "Embedding"]
    layers = (*embed, *block_specs, *tail)
    if not layers:
        layers = _fallback_layers(tensor_names)
    architectures = config.get("architectures")
    display = ""
    if isinstance(architectures, list) and architectures:
        first = architectures[0]
        if isinstance(first, str):
            display = first
    return ArchitecturePlan(
        architecture=_architecture_name(config),
        name=display or _json_str(config, "model_type") or "huggingface",
        block_count=block_count,
        vocab_size=_vocab_size(config, tensor_names, dims),
        embedding_length=_embedding_length(config, tensor_names, dims),
        context_length=_json_int(
            config,
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_sequence_length",
            "model_max_length",
        ),
        tied_embeddings=bool(tied),
        layers=layers,
    )
