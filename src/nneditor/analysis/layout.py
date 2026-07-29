"""Deterministic layered layout for IR graphs (task P1.6).

Turns one IR graph into a renderer-agnostic :class:`Scene`: nodes are layered
by longest path from the sources, ordered within each layer by the barycenter
of their predecessors, and placed on a fixed grid. The algorithm is chosen for
determinism and speed, not typographic beauty — the ADR budgets require layout
to be cacheable and reproducible, so given the same graph and settings it must
produce byte-identical geometry on every machine.

Cycles cannot occur in valid ONNX dataflow, but a malformed artifact must not
hang the layouter: nodes left over after a Kahn pass are appended in
declaration order below the acyclic part, and the caller sees that through the
returned :class:`LayoutResult`.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from nneditor.cancellation import CancellationToken
from nneditor.ir.core import Graph, Node
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene

__all__ = ["LayoutResult", "LayoutSettings", "layered_order", "layout_graph"]

_KIND_BY_OP: Final[dict[str, str]] = {
    "Conv": "conv",
    "ConvTranspose": "conv",
    "Gemm": "linear",
    "MatMul": "linear",
    "Attention": "attention",
    "MultiHeadAttention": "attention",
    "BatchNormalization": "norm",
    "LayerNormalization": "norm",
    "GroupNormalization": "norm",
    "InstanceNormalization": "norm",
    "Relu": "activation",
    "LeakyRelu": "activation",
    "Sigmoid": "activation",
    "Tanh": "activation",
    "Gelu": "activation",
    "Softmax": "activation",
    "Clip": "activation",
    "Add": "elementwise",
    "Sub": "elementwise",
    "Mul": "elementwise",
    "Div": "elementwise",
    "Pow": "elementwise",
    "Reshape": "reshape",
    "Transpose": "reshape",
    "Squeeze": "reshape",
    "Unsqueeze": "reshape",
    "Concat": "reshape",
    "Split": "reshape",
    "Flatten": "reshape",
}
_DEFAULT_KIND: Final = "other"


@dataclass(frozen=True, slots=True)
class LayoutSettings:
    """Geometry knobs; part of every layout cache key."""

    node_width: float = 140.0
    node_height: float = 44.0
    column_gap: float = 48.0
    row_gap: float = 64.0
    ordering_passes: int = 3

    def __post_init__(self) -> None:
        if self.node_width <= 0 or self.node_height <= 0:
            raise ValueError("node size must be positive")
        if self.column_gap < 0 or self.row_gap < 0:
            raise ValueError("gaps must be non-negative")
        if self.ordering_passes < 0:
            raise ValueError("ordering_passes must be non-negative")


@dataclass(frozen=True, slots=True)
class LayoutResult:
    """A scene plus what the layouter had to work around."""

    scene: Scene
    layers: tuple[tuple[str, ...], ...]
    cyclic_nodes: tuple[str, ...]


def _node_kind(node: Node) -> str:
    if node.subgraphs:
        return "control-flow"
    return _KIND_BY_OP.get(node.op_type, _DEFAULT_KIND)


def _node_label(node: Node) -> str:
    return node.source_name or node.op_type


def _dataflow_edges(graph: Graph) -> list[tuple[str, str, str, int]]:
    """``(value id, producer node, consumer node, consumer port)`` tuples."""
    edges: list[tuple[str, str, str, int]] = []
    for node in graph.nodes:
        for port, value_id in enumerate(node.inputs):
            producer = graph.producer(value_id)
            if producer is not None:
                edges.append((value_id, producer[0], node.id, port))
    return edges


def _assign_layers(
    graph: Graph,
    edges: list[tuple[str, str, str, int]],
    token: CancellationToken | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Longest-path layering via Kahn's algorithm; leftovers are cyclic."""
    return _layer_identifiers(tuple(node.id for node in graph.nodes), edges, token)


def _layer_identifiers(
    identifiers: tuple[str, ...],
    edges: list[tuple[str, str, str, int]],
    token: CancellationToken | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """The graph-agnostic core of :func:`_assign_layers`."""
    order = {node_id: position for position, node_id in enumerate(identifiers)}
    indegree = dict.fromkeys(order, 0)
    successors: dict[str, list[str]] = {node_id: [] for node_id in order}
    for _value, source, target, _port in edges:
        indegree[target] += 1
        successors[source].append(target)
    layer = dict.fromkeys(order, 0)
    # A min-heap of declaration positions pops exactly what the previous
    # sorted-list queue popped — its minimum — without re-sorting the whole
    # queue on every pop, which was quadratic on wide graphs.
    ready = [order[node_id] for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    placed: list[str] = []
    while ready:
        if token is not None and len(placed) % 1024 == 0:
            token.raise_if_cancelled()
        current = identifiers[heapq.heappop(ready)]
        placed.append(current)
        for successor in successors[current]:
            layer[successor] = max(layer[successor], layer[current] + 1)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, order[successor])
    placed_set = set(placed)
    cyclic = tuple(node_id for node_id in order if node_id not in placed_set)
    if cyclic:
        bottom = (max(layer.values()) if layer else 0) + 1
        for node_id in cyclic:
            layer[node_id] = bottom
    return layer, cyclic


def _order_within_layers(
    layers: list[list[str]],
    edges: list[tuple[str, str, str, int]],
    passes: int,
    token: CancellationToken | None = None,
) -> None:
    """Barycenter ordering, alternating downward and upward sweeps in place."""
    predecessors: dict[str, list[str]] = {}
    successors: dict[str, list[str]] = {}
    for _value, source, target, _port in edges:
        predecessors.setdefault(target, []).append(source)
        successors.setdefault(source, []).append(target)

    def sweep(neighbor_of: dict[str, list[str]], sequence: list[list[str]]) -> None:
        position: dict[str, float] = {}
        for row in sequence:
            for column, node_id in enumerate(row):
                position[node_id] = float(column)
        for row in sequence:
            keyed = []
            for column, node_id in enumerate(row):
                neighbors = neighbor_of.get(node_id, ())
                anchored = [position[item] for item in neighbors if item in position]
                barycenter = (
                    sum(anchored) / len(anchored) if anchored else float(column)
                )
                keyed.append((barycenter, column, node_id))
            keyed.sort()
            row[:] = [node_id for _bary, _column, node_id in keyed]
            for column, node_id in enumerate(row):
                position[node_id] = float(column)

    for _ in range(passes):
        if token is not None:
            token.raise_if_cancelled()
        sweep(predecessors, layers)
        sweep(successors, layers[::-1])


def layered_order(
    identifiers: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    ordering_passes: int = 3,
) -> tuple[tuple[str, ...], ...]:
    """Deterministic layered ordering for an arbitrary node/edge list.

    Reuses the longest-path layering and barycenter ordering that
    :func:`layout_graph` applies to IR graphs, so other producers of small
    node/edge lists — collapsed semantic scenes in
    :mod:`nneditor.analysis.lod` — get the same deterministic placement
    discipline without duplicating the algorithm. Nodes on a cycle are placed
    on a bottom row rather than hanging the layering; rows that end up empty
    are dropped.
    """
    ids = tuple(identifiers)
    quads = [("", source, target, 0) for source, target in edges]
    layer_of, _cyclic = _layer_identifiers(ids, quads)
    layer_count = max(layer_of.values(), default=-1) + 1
    rows: list[list[str]] = [[] for _ in range(layer_count)]
    for node_id in ids:
        rows[layer_of[node_id]].append(node_id)
    _order_within_layers(rows, quads, ordering_passes)
    return tuple(tuple(row) for row in rows if row)


def layout_graph(
    graph: Graph,
    settings: LayoutSettings | None = None,
    *,
    token: CancellationToken | None = None,
) -> LayoutResult:
    """Lay out one IR graph as a scene.

    ``token`` is checked periodically so a caller can abandon a large layout
    within a bounded amount of work; ``None`` keeps the call uncancellable.
    """
    settings = settings or LayoutSettings()
    if token is not None:
        token.raise_if_cancelled()
    edges = _dataflow_edges(graph)
    layer_of, cyclic = _assign_layers(graph, edges, token)
    layer_count = max(layer_of.values(), default=-1) + 1
    layers: list[list[str]] = [[] for _ in range(layer_count)]
    for node in graph.nodes:
        layers[layer_of[node.id]].append(node.id)
    _order_within_layers(layers, edges, settings.ordering_passes, token)

    width_step = settings.node_width + settings.column_gap
    height_step = settings.node_height + settings.row_gap
    positions: dict[str, tuple[float, float]] = {}
    glyphs: list[NodeGlyph] = []
    # The widest row is loop-invariant; computing it per row made layout
    # O(layers²), which only showed up on deep real models (14k nodes took
    # 33 s, almost all of it here).
    widest = max(
        (len(item) * width_step - settings.column_gap for item in layers),
        default=0.0,
    )
    for row_number, row in enumerate(layers):
        # Center each row so narrow layers do not hug the left edge.
        row_width = len(row) * width_step - settings.column_gap
        left = (widest - row_width) / 2.0
        for column, node_id in enumerate(row):
            if token is not None and len(glyphs) % 1024 == 0:
                token.raise_if_cancelled()
            node = graph.node(node_id)
            x = left + column * width_step
            y = row_number * height_step
            positions[node_id] = (x, y)
            glyphs.append(
                NodeGlyph(
                    id=node_id,
                    x=x,
                    y=y,
                    width=settings.node_width,
                    height=settings.node_height,
                    kind=_node_kind(node),
                    label=_node_label(node),
                    type_label=node.qualified_op_type,
                )
            )

    edge_glyphs: list[EdgeGlyph] = []
    for value_id, source, target, port in edges:
        if token is not None and len(edge_glyphs) % 2048 == 0:
            token.raise_if_cancelled()
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        edge_glyphs.append(
            EdgeGlyph(
                id=f"e:{value_id}=>{target}:{port}",
                source=source,
                target=target,
                points=(
                    (
                        source_x + settings.node_width / 2.0,
                        source_y + settings.node_height,
                    ),
                    (target_x + settings.node_width / 2.0, target_y),
                ),
            )
        )
    return LayoutResult(
        scene=Scene(glyphs, edge_glyphs),
        layers=tuple(tuple(row) for row in layers),
        cyclic_nodes=cyclic,
    )
