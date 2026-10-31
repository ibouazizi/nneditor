"""Activation boundaries and user-facing labels for traced blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nneditor.ir.core import Graph
from nneditor.rendering.scene import EdgeGlyph, Scene
from nneditor.ui.trace_graph import TraceGraphPresentation


@dataclass(frozen=True, slots=True)
class TracePort:
    glyph_id: str
    direction: str
    values: tuple[str, ...]
    x: float
    y: float


def boundary_values(
    graph: Graph, members: frozenset[str], *, weights: set[str] | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exclude weights and internal wires; keep graph port order deterministic."""
    if weights is None:
        weights = {f"v{key[1:]}" for key in graph.initializers}
    inputs: dict[str, None] = {}
    outputs: dict[str, None] = {}
    for node_id in sorted(members):
        node = graph.node(node_id)
        for key in node.inputs:
            producer = graph.producer(key)
            if key not in weights and (
                (producer is not None and producer[0] not in members)
                or (producer is None and key in graph.inputs)
            ):
                inputs[key] = None
        for key in node.outputs:
            if key in graph.outputs or any(
                consumer not in members for consumer, _ in graph.consumers(key)
            ):
                outputs[key] = None
    return tuple(inputs), tuple(outputs)


def graph_ports(
    presentation: TraceGraphPresentation,
    graph: Graph,
    members_by_glyph: dict[str, frozenset[str]],
    *,
    scoped: bool = False,
) -> tuple[Scene, dict[str, tuple[TracePort, ...]]]:
    """Keep model block geometry; attach ports and route wires to their centers.

    Wide signatures use a final grouped port, so tiny blocks never accumulate
    overlapping hit targets. Each grouped value remains individually inspectable.
    """
    ports: dict[str, tuple[TracePort, ...]] = {}
    anchors: dict[tuple[str, str, str], tuple[float, float]] = {}
    weights = {f"v{key[1:]}" for key in graph.initializers}
    input_values = {glyph: value for value, glyph in presentation.input_glyphs.items()}
    output_values = {
        glyph: value for value, glyph in presentation.output_glyphs.items()
    }
    for node in presentation.scene.nodes:
        inputs: tuple[str, ...]
        outputs: tuple[str, ...]
        if node.id in input_values:
            inputs, outputs = (), (input_values[node.id],)
        elif node.id in output_values:
            inputs, outputs = (output_values[node.id],), ()
        else:
            inputs, outputs = boundary_values(
                graph, members_by_glyph.get(node.id, frozenset()), weights=weights
            )
        node_ports: list[TracePort] = []
        capacity = max(1, min(8, int(node.width // 28)))
        for direction, values in (("input", inputs), ("output", outputs)):
            groups: list[tuple[str, ...]] = [(value,) for value in values[:capacity]]
            if len(values) > capacity:
                groups[-1] = values[capacity - 1 :]
            for index, group in enumerate(groups):
                x = node.x + node.width * (index + 1) / (len(groups) + 1)
                y = node.y if direction == "input" else node.y + node.height
                node_ports.append(TracePort(node.id, direction, group, x, y))
                for value in group:
                    anchors[node.id, direction, value] = (x, y)
        ports[node.id] = tuple(node_ports)
    edges: list[EdgeGlyph] = []
    for edge in presentation.scene.edges:
        values = presentation.values_by_glyph.get(edge.id, ())
        if not values:
            edges.append(edge)
            continue
        for index, value in enumerate(values):
            start = anchors.get((edge.source, "output", value), edge.points[0])
            end = anchors.get((edge.target, "input", value), edge.points[-1])
            edges.append(
                EdgeGlyph(
                    edge.id if index == 0 else f"{edge.id}:port:{index}",
                    edge.source,
                    edge.target,
                    (start, *edge.points[1:-1], end),
                )
            )
    connected = {endpoint for edge in edges for endpoint in (edge.source, edge.target)}
    nodes = [
        replace(
            node,
            type_label="Model input" if node.kind == "graph-input" else "Model output",
        )
        if node.kind in {"graph-input", "graph-output"}
        else node
        for node in presentation.scene.nodes
        if not (
            scoped
            and node.kind in {"graph-input", "graph-output"}
            and node.id not in connected
        )
    ]
    return Scene(nodes, edges), {node.id: ports[node.id] for node in nodes}


def port_label(graph: Graph, value_id: str) -> str:
    value = graph.value(value_id)
    llm = bool(graph.nodes and graph.nodes[0].domain == "nneditor.llm")
    if llm:
        if value_id in graph.inputs:
            return "Prompt tokens"
        if value_id in graph.outputs:
            return "Logits"
        producer = graph.producer(value_id)
        if producer:
            return {
                "Embedding": "Token embeddings",
                "RMSNorm": "Normalized hidden state",
            }.get(graph.node(producer[0]).op_type, "Hidden state")
    return value.name or value.id
