"""Derive a synthetic LLM graph from GGUF tensor names and architecture KV.

GGUF stores weights, not operators. The viewer still needs a graph so
architecture/block navigation works: tensor names (`token_embd`, `blk.N.*`,
`output`) plus `{arch}.block_count` become a chain of Embedding, Attention or
SSM, FFN, RMSNorm, and Linear nodes in the `nneditor.llm` domain.

`source_name` uses hyphens (`llm/block-000/attn`) because the source-scope
detector splits on `.`. Pattern and repeated detectors are skipped for this
synthetic graph; SOURCE grouping is the intended block structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from nneditor.adapters.gguf.index import GgufIndex, GgufTensorIndex

__all__ = [
    "ArchitecturePlan",
    "LayerSpec",
    "WeightBinding",
    "plan_architecture",
]

_BLOCK_NAME: Final = re.compile(r"^blk\.(\d+)\.(.+)$")
_EMBED_PREFIXES: Final = ("token_embd", "position_embd", "token_types")
_EMPTY_KINDS: Final[dict[str, list[str]]] = {
    "attn": [],
    "ffn": [],
    "ssm": [],
    "extra": [],
}


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

    architecture: str | None
    name: str | None
    block_count: int
    vocab_size: int | None
    embedding_length: int | None
    tied_embeddings: bool
    layers: tuple[LayerSpec, ...]


def _kv_str(index: GgufIndex, key: str) -> str | None:
    value = index.kv_value(key)
    return value if isinstance(value, str) and value else None


def _kv_int(index: GgufIndex, key: str) -> int | None:
    value = index.kv_value(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _arch_int(index: GgufIndex, architecture: str | None, key: str) -> int | None:
    if architecture:
        value = _kv_int(index, f"{architecture}.{key}")
        if value is not None:
            return value
    return _kv_int(index, key)


def _elided_count(index: GgufIndex, key: str) -> int | None:
    entry = index.kv_map.get(key)
    if entry is None or not entry.elided or not isinstance(entry.value, dict):
        return None
    count = entry.value.get("count")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _n_elems_last_dim(tensor: GgufTensorIndex) -> int | None:
    if not tensor.dims:
        return None
    return tensor.dims[-1]


def _vocab_size(
    index: GgufIndex, architecture: str | None, tensors: dict[str, GgufTensorIndex]
) -> int | None:
    declared = _arch_int(index, architecture, "vocab_size")
    if declared is not None:
        return declared
    tokens = _elided_count(index, "tokenizer.ggml.tokens")
    if tokens is not None:
        return tokens
    embed = tensors.get("token_embd.weight")
    if embed is not None:
        return _n_elems_last_dim(embed)
    return None


def _attr_name(tensor_name: str) -> str:
    return tensor_name


def _bindings(names: list[str]) -> tuple[WeightBinding, ...]:
    return tuple(WeightBinding(_attr_name(name), name) for name in names)


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


def _classify_block_suffix(suffix: str) -> str:
    if suffix.startswith("ssm_"):
        return "ssm"
    if suffix.startswith("ffn_"):
        return "ffn"
    if suffix.startswith("attn_"):
        return "attn"
    return "extra"


def _partition_blocks(
    tensors: dict[str, GgufTensorIndex],
) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = {}
    for name in tensors:
        match = _BLOCK_NAME.match(name)
        if match is None:
            continue
        block = int(match.group(1))
        kind = _classify_block_suffix(match.group(2))
        kinds = grouped.setdefault(block, {key: [] for key in _EMPTY_KINDS})
        kinds[kind].append(name)
    for kinds in grouped.values():
        for names in kinds.values():
            names.sort()
    return grouped


def _is_embed(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".") for prefix in _EMBED_PREFIXES
    )


def _is_output_norm(name: str) -> bool:
    return name == "output_norm.weight" or name.startswith("output_norm.")


def _global_layers(
    tensors: dict[str, GgufTensorIndex], *, tied: bool
) -> list[LayerSpec]:
    names = list(tensors)
    embed = sorted(name for name in names if _is_embed(name))
    output_norm = sorted(name for name in names if _is_output_norm(name))
    output = [name for name in names if name == "output.weight"]
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
    head_weights = output
    if not head_weights and tied and "token_embd.weight" in tensors:
        head_weights = ["token_embd.weight"]
    if head_weights:
        layers.append(
            LayerSpec("lm-head", "Linear", "llm/lm_head", _bindings(head_weights), None)
        )
    return layers


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


def _fallback_layers(tensors: dict[str, GgufTensorIndex]) -> tuple[LayerSpec, ...]:
    names = sorted(tensors)
    if not names:
        return ()
    return (
        LayerSpec(
            "weights",
            "Parameters",
            "llm/weights",
            _bindings(names),
            None,
        ),
    )


def plan_architecture(index: GgufIndex) -> ArchitecturePlan:
    """Group GGUF tensors into a stable, source-named LLM operator chain."""
    tensors = {tensor.name: tensor for tensor in index.tensors}
    architecture = _kv_str(index, "general.architecture")
    name = _kv_str(index, "general.name")
    grouped = _partition_blocks(tensors)
    declared_blocks = _arch_int(index, architecture, "block_count")
    block_count = (
        declared_blocks
        if declared_blocks is not None
        else (max(grouped) + 1 if grouped else 0)
    )
    tied = "output.weight" not in tensors and "token_embd.weight" in tensors
    block_specs = _block_layers(grouped)
    global_specs = _global_layers(tensors, tied=tied)
    embed = [spec for spec in global_specs if spec.op_type == "Embedding"]
    tail = [spec for spec in global_specs if spec.op_type != "Embedding"]
    layers = (*embed, *block_specs, *tail)
    if not layers:
        layers = _fallback_layers(tensors)
    return ArchitecturePlan(
        architecture=architecture,
        name=name,
        block_count=block_count,
        vocab_size=_vocab_size(index, architecture, tensors),
        embedding_length=_arch_int(index, architecture, "embedding_length"),
        tied_embeddings=tied,
        layers=layers,
    )
