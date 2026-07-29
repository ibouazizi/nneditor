"""Renderer-neutral graph affordances for interactive activation tracing.

The IR deliberately models values as edges, while graph inputs and outputs are
boundary declarations rather than operator nodes.  For tracing, those
boundaries need to be visible and selectable.  This module derives a
presentation-only scene with input/output glyphs and records exactly which IR
values every selectable connection represents.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nneditor.ir.core import Graph
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene

__all__ = ["TraceGraphPresentation", "build_trace_graph"]

_BOUNDARY_WIDTH = 168.0
_BOUNDARY_HEIGHT = 54.0
_BOUNDARY_GAP = 72.0


def _identifier(kind: str, value_id: str) -> str:
    digest = hashlib.blake2b(value_id.encode(), digest_size=12).hexdigest()
    return f"trace:{kind}:{digest}"


@dataclass(frozen=True, slots=True)
class TraceGraphPresentation:
    """A graph scene plus trace-specific selection mappings."""

    scene: Scene
    values_by_glyph: dict[str, tuple[str, ...]]
    input_glyphs: dict[str, str]
    output_glyphs: dict[str, str]


def _spread_positions(
    items: list[tuple[str, float]],
    *,
    minimum_x: float,
    maximum_x: float,
) -> dict[str, float]:
    """Place boundary nodes near their dataflow anchors without overlaps."""
    if not items:
        return {}
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    step = _BOUNDARY_WIDTH + 18.0
    lefts: list[float] = []
    for _identifier_value, anchor in ordered:
        left = anchor - _BOUNDARY_WIDTH / 2.0
        if lefts:
            left = max(left, lefts[-1] + step)
        lefts.append(left)
    overflow = lefts[-1] + _BOUNDARY_WIDTH - maximum_x
    if overflow > 0:
        lefts = [left - overflow for left in lefts]
    underflow = minimum_x - lefts[0]
    if underflow > 0:
        lefts = [left + underflow for left in lefts]
    return {item[0]: left for item, left in zip(ordered, lefts, strict=True)}


def build_trace_graph(
    scene: Scene,
    graph: Graph,
    members_by_glyph: dict[str, frozenset[str]],
) -> TraceGraphPresentation:
    """Add selectable model boundaries and map connections back to values."""
    representatives = {
        member: glyph_id
        for glyph_id, members in members_by_glyph.items()
        for member in members
    }
    values_between: dict[tuple[str, str], set[str]] = {}
    exact_edges: dict[str, str] = {}
    for node in graph.nodes:
        target = representatives.get(node.id)
        if target is None:
            continue
        for port, value_id in enumerate(node.inputs):
            producer = graph.producer(value_id)
            if producer is None:
                continue
            source = representatives.get(producer[0])
            if source is None or source == target:
                continue
            values_between.setdefault((source, target), set()).add(value_id)
            exact_edges[f"e:{value_id}=>{node.id}:{port}"] = value_id

    values_by_glyph: dict[str, tuple[str, ...]] = {}
    for edge in scene.edges:
        exact = exact_edges.get(edge.id)
        values = (
            (exact,)
            if exact is not None
            else tuple(sorted(values_between.get((edge.source, edge.target), ())))
        )
        if values:
            values_by_glyph[edge.id] = values

    nodes = list(scene.nodes)
    edges = list(scene.edges)
    input_glyphs: dict[str, str] = {}
    output_glyphs: dict[str, str] = {}
    graph_input_ids = [
        value_id for value_id in graph.inputs if value_id not in graph.initializers
    ]

    if scene.nodes:
        minimum_x = scene.bounds.min_x
        maximum_x = scene.bounds.max_x
        top = scene.bounds.min_y - _BOUNDARY_GAP - _BOUNDARY_HEIGHT
        bottom = scene.bounds.max_y + _BOUNDARY_GAP
    else:
        minimum_x = 0.0
        maximum_x = max(
            _BOUNDARY_WIDTH,
            (len(graph_input_ids) + len(graph.outputs)) * (_BOUNDARY_WIDTH + 18.0),
        )
        top = 0.0
        bottom = _BOUNDARY_HEIGHT + _BOUNDARY_GAP

    input_targets: dict[str, tuple[str, ...]] = {}
    input_anchors: list[tuple[str, float]] = []
    for value_id in graph_input_ids:
        targets = tuple(
            dict.fromkeys(
                representative
                for node_id, _port in graph.consumers(value_id)
                if (representative := representatives.get(node_id)) is not None
            )
        )
        input_targets[value_id] = targets
        centers = [
            scene.node(target).x + scene.node(target).width / 2.0 for target in targets
        ]
        anchor = (
            sum(centers) / len(centers)
            if centers
            else minimum_x + _BOUNDARY_WIDTH / 2.0
        )
        input_anchors.append((value_id, anchor))
    input_lefts = _spread_positions(
        input_anchors,
        minimum_x=minimum_x,
        maximum_x=maximum_x,
    )

    for value_id in graph_input_ids:
        value = graph.value(value_id)
        glyph_id = _identifier("input", value_id)
        input_glyphs[value_id] = glyph_id
        values_by_glyph[glyph_id] = (value_id,)
        left = input_lefts[value_id]
        nodes.append(
            NodeGlyph(
                id=glyph_id,
                x=left,
                y=top,
                width=_BOUNDARY_WIDTH,
                height=_BOUNDARY_HEIGHT,
                kind="graph-input",
                label=value.name or "Model input",
                type_label="Model input · choose tensor",
            )
        )
        for position, target in enumerate(input_targets[value_id]):
            target_node = scene.node(target)
            edge_id = _identifier(f"input-edge:{position}:{target}", value_id)
            edges.append(
                EdgeGlyph(
                    id=edge_id,
                    source=glyph_id,
                    target=target,
                    points=(
                        (left + _BOUNDARY_WIDTH / 2.0, top + _BOUNDARY_HEIGHT),
                        (
                            target_node.x + target_node.width / 2.0,
                            target_node.y,
                        ),
                    ),
                )
            )
            values_by_glyph[edge_id] = (value_id,)

    output_sources: dict[str, str | None] = {}
    output_anchors: list[tuple[str, float]] = []
    for value_id in graph.outputs:
        producer = graph.producer(value_id)
        source = representatives.get(producer[0]) if producer is not None else None
        if source is None:
            source = input_glyphs.get(value_id)
        output_sources[value_id] = source
        anchor = (
            scene.node(source).x + scene.node(source).width / 2.0
            if source is not None and scene.has_node(source)
            else input_lefts.get(value_id, minimum_x) + _BOUNDARY_WIDTH / 2.0
        )
        output_anchors.append((value_id, anchor))
    output_lefts = _spread_positions(
        output_anchors,
        minimum_x=minimum_x,
        maximum_x=maximum_x,
    )

    for value_id in graph.outputs:
        value = graph.value(value_id)
        glyph_id = _identifier("output", value_id)
        output_glyphs[value_id] = glyph_id
        values_by_glyph[glyph_id] = (value_id,)
        left = output_lefts[value_id]
        nodes.append(
            NodeGlyph(
                id=glyph_id,
                x=left,
                y=bottom,
                width=_BOUNDARY_WIDTH,
                height=_BOUNDARY_HEIGHT,
                kind="graph-output",
                label=value.name or "Model output",
                type_label="Model output · click to inspect",
            )
        )
        source = output_sources[value_id]
        if source is None:
            continue
        source_node = next(node for node in nodes if node.id == source)
        edge_id = _identifier(f"output-edge:{source}", value_id)
        edges.append(
            EdgeGlyph(
                id=edge_id,
                source=source,
                target=glyph_id,
                points=(
                    (
                        source_node.x + source_node.width / 2.0,
                        source_node.y + source_node.height,
                    ),
                    (left + _BOUNDARY_WIDTH / 2.0, bottom),
                ),
            )
        )
        values_by_glyph[edge_id] = (value_id,)

    return TraceGraphPresentation(
        Scene(nodes, edges),
        values_by_glyph,
        input_glyphs,
        output_glyphs,
    )
