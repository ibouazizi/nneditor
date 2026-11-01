"""Show the weight layers inside a synthesized LLM block.

GGUF and Hugging Face checkpoints do not store an operator circuit. The
import builds one node for attention, one for a state-space block, and one
for the feed-forward, and hangs every weight on that node. The Blocks level
is that pair. The Layers level opens each node into the projections and
norms it actually contains, in execution order.

The execution graph is unchanged. Tracing still runs the coarse node; a
layer glyph only points back at it and names the tensors that belong to
that step.
"""

from __future__ import annotations

from itertools import pairwise

from nneditor.analysis.lod import DetailLevel, SemanticScene, _relayout
from nneditor.ir.core import Graph, Node
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene

__all__ = ["expand_llm_layers", "layer_label", "layer_owner", "layer_tensor_ids"]

_LAYER_MARKER = "::layer:"

# Execution order inside a block. Stems not listed keep their relative order
# after these, sorted by name so the picture stays stable.
_ORDER: tuple[str, ...] = (
    "attn_norm",
    "input_layernorm",
    "attn_q",
    "q_proj",
    "attn_k",
    "k_proj",
    "attn_v",
    "v_proj",
    "attn_q_norm",
    "q_norm",
    "attn_k_norm",
    "k_norm",
    "attn_qkv",
    "qkv_proj",
    "attn_gate",
    "ssm_conv1d",
    "ssm_alpha",
    "ssm_beta",
    "ssm_dt",
    "ssm_a",
    "ssm_norm",
    "ssm_out",
    "attn_output",
    "o_proj",
    "out_proj",
    "ffn_norm",
    "post_attention_layernorm",
    "post_attention_norm",
    "ffn_gate",
    "gate_proj",
    "ffn_up",
    "up_proj",
    "ffn_down",
    "down_proj",
)
_RANK = {stem: index for index, stem in enumerate(_ORDER)}

_LABELS: dict[str, str] = {
    "attn_norm": "Attention norm",
    "input_layernorm": "Input norm",
    "attn_q": "Query",
    "q_proj": "Query",
    "attn_k": "Key",
    "k_proj": "Key",
    "attn_v": "Value",
    "v_proj": "Value",
    "attn_q_norm": "Query norm",
    "q_norm": "Query norm",
    "attn_k_norm": "Key norm",
    "k_norm": "Key norm",
    "attn_qkv": "QKV projection",
    "qkv_proj": "QKV projection",
    "attn_gate": "Output gate",
    "ssm_conv1d": "Convolution",
    "ssm_alpha": "Decay projection",
    "ssm_beta": "Update gate",
    "ssm_dt": "Decay bias",
    "ssm_a": "Decay",
    "ssm_norm": "State norm",
    "ssm_out": "State output",
    "attn_output": "Attention output",
    "o_proj": "Attention output",
    "out_proj": "Attention output",
    "ffn_norm": "Feed-forward norm",
    "post_attention_layernorm": "Post-attention norm",
    "post_attention_norm": "Post-attention norm",
    "ffn_gate": "Gate",
    "gate_proj": "Gate",
    "ffn_up": "Up projection",
    "up_proj": "Up projection",
    "ffn_down": "Down projection",
    "down_proj": "Down projection",
}

_EXPAND_OPS = frozenset({"Attention", "SSM", "FFN", "Parameters"})


def layer_label(stem: str) -> str:
    """The name drawn on one layer glyph."""
    known = _LABELS.get(stem)
    if known is not None:
        return known
    return stem.replace("_", " ").capitalize()


def layer_owner(
    members_by_glyph: dict[str, frozenset[str]], glyph_id: str
) -> tuple[str, str] | None:
    """The operator node and layer stem a layer glyph stands for."""
    if _LAYER_MARKER not in glyph_id:
        return None
    members = members_by_glyph.get(glyph_id)
    if members is None or len(members) != 1:
        return None
    return next(iter(members)), glyph_id.split(_LAYER_MARKER, 1)[1]


def layer_tensor_ids(node: Node, stem: str) -> tuple[str, ...]:
    """Tensor ids on ``node`` that belong to one layer stem."""
    found: list[str] = []
    for attribute in node.attributes:
        if _stem(attribute.name) != stem:
            continue
        for tensor_id in attribute.referenced_tensor_ids():
            if tensor_id not in found:
                found.append(tensor_id)
    return tuple(found)


def expand_llm_layers(semantic: SemanticScene, graph: Graph) -> SemanticScene:
    """Replace coarse LLM block nodes with their ordered weight layers."""
    if semantic.detail_level is not DetailLevel.LAYER:
        return semantic
    replacements = {
        glyph.id: _layer_glyphs(glyph, graph.node(glyph.id))
        for glyph in semantic.scene.nodes
        if _expandable(glyph.id, semantic.members_by_glyph, graph)
    }
    if not replacements:
        return semantic
    nodes: list[NodeGlyph] = []
    ends: dict[str, tuple[str, str]] = {}
    members = dict(semantic.members_by_glyph)
    for glyph in semantic.scene.nodes:
        parts = replacements.get(glyph.id)
        if parts is None:
            nodes.append(glyph)
            continue
        nodes.extend(parts)
        ends[glyph.id] = (parts[0].id, parts[-1].id)
        owner = members.pop(glyph.id)
        for part in parts:
            members[part.id] = owner
    edges: list[EdgeGlyph] = []
    for edge in semantic.scene.edges:
        source = ends[edge.source][1] if edge.source in ends else edge.source
        target = ends[edge.target][0] if edge.target in ends else edge.target
        edges.append(
            EdgeGlyph(
                id=f"e:{source}->{target}",
                source=source,
                target=target,
                points=((0.0, 0.0), (0.0, 1.0)),
            )
        )
    for parts in replacements.values():
        for left, right in pairwise(parts):
            edges.append(
                EdgeGlyph(
                    id=f"e:{left.id}->{right.id}",
                    source=left.id,
                    target=right.id,
                    points=((0.0, 0.0), (0.0, 1.0)),
                )
            )
    return SemanticScene(
        _relayout(Scene(nodes, edges)),
        semantic.detail_level,
        members,
    )


def _expandable(
    glyph_id: str, members_by_glyph: dict[str, frozenset[str]], graph: Graph
) -> bool:
    members = members_by_glyph.get(glyph_id)
    if members != frozenset((glyph_id,)):
        return False
    try:
        node = graph.node(glyph_id)
    except KeyError:
        return False
    return (
        node.domain == "nneditor.llm"
        and node.op_type in _EXPAND_OPS
        and len(_grouped(node)) >= 2
    )


def _layer_glyphs(glyph: NodeGlyph, node: Node) -> tuple[NodeGlyph, ...]:
    prefix = _block_prefix(node)
    glyphs: list[NodeGlyph] = []
    for stem, _names in _grouped(node):
        label = f"{prefix}{layer_label(stem)}" if prefix else layer_label(stem)
        glyphs.append(
            NodeGlyph(
                id=f"{glyph.id}{_LAYER_MARKER}{stem}",
                x=0.0,
                y=0.0,
                width=min(240.0, max(140.0, 24.0 + 7.0 * len(label))),
                height=glyph.height,
                kind=_kind(stem),
                label=label,
                type_label="Layer",
            )
        )
    return tuple(glyphs)


def _grouped(node: Node) -> tuple[tuple[str, tuple[str, ...]], ...]:
    buckets: dict[str, list[str]] = {}
    for attribute in node.attributes:
        if not attribute.referenced_tensor_ids():
            continue
        buckets.setdefault(_stem(attribute.name), []).append(attribute.name)
    ordered = sorted(buckets, key=lambda stem: (_RANK.get(stem, len(_RANK)), stem))
    return tuple((stem, tuple(buckets[stem])) for stem in ordered)


def _stem(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
        parts = parts[2:]
    if parts and parts[-1] in {"weight", "bias", "scales", "qweight"}:
        parts = parts[:-1]
    return parts[-1] if parts else name


def _block_prefix(node: Node) -> str:
    for part in (node.source_name or "").split("/"):
        if part.startswith("block-"):
            return f"{part.removeprefix('block-')} · "
    return ""


def _kind(stem: str) -> str:
    if stem.endswith("norm") or stem.endswith("layernorm"):
        return "norm"
    if stem.startswith("ffn_") or stem in {"gate_proj", "up_proj", "down_proj"}:
        return "linear"
    return "attention"
