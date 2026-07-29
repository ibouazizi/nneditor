"""Deterministic synthetic scenes for benchmarks and tests.

The Phase 0 renderer benchmark needs graphs far larger than any committed
fixture, with a realistic shape: layered like a compiled network, mostly local
edges plus long skip connections, and repeated blocks that can collapse. These
are generated, never stored — the same ``seed`` always yields byte-identical
scenes, so benchmark runs and golden tests stay comparable across machines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene, ScenePatch

__all__ = ["SyntheticGraph", "collapse_layers", "synthetic_graph"]

NODE_WIDTH: Final = 120.0
NODE_HEIGHT: Final = 40.0
COLUMN_GAP: Final = 40.0
ROW_GAP: Final = 60.0

_KINDS: Final = (
    "conv",
    "norm",
    "activation",
    "linear",
    "attention",
    "reshape",
    "elementwise",
)


@dataclass(frozen=True, slots=True)
class SyntheticGraph:
    """A generated scene plus the layer structure the generator used.

    The layers are what a collapse benchmark groups; real models will get this
    structure from block detection (Phase 2), but the renderer must not care
    where it came from.
    """

    scene: Scene
    layers: tuple[tuple[str, ...], ...]


def _node_position(layer: int, column: int) -> tuple[float, float]:
    x = column * (NODE_WIDTH + COLUMN_GAP)
    y = layer * (NODE_HEIGHT + ROW_GAP)
    return x, y


def _edge_points(
    source: NodeGlyph, target: NodeGlyph
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (source.x + source.width / 2.0, source.y + source.height),
        (target.x + target.width / 2.0, target.y),
    )


def synthetic_graph(
    node_count: int,
    *,
    layer_width: int = 25,
    seed: int = 0,
    mean_extra_edges: float = 1.5,
    skip_probability: float = 0.1,
) -> SyntheticGraph:
    """Generate a layered DAG scene with roughly ``node_count`` nodes.

    Every non-first-layer node gets one edge from the previous layer so the
    graph is connected, plus a Poisson-ish number of extra edges controlled by
    ``mean_extra_edges``; each extra edge becomes a skip connection to an
    earlier layer with ``skip_probability``. The default 1.5 extra edges per
    node gives a 10,000-node graph about 25,000 edges — the "tens of thousands"
    the benchmark plan calls for.
    """
    if node_count <= 0:
        raise ValueError(f"node_count must be positive, got {node_count}")
    if layer_width <= 0:
        raise ValueError(f"layer_width must be positive, got {layer_width}")
    rng = random.Random(seed)
    nodes: list[NodeGlyph] = []
    layers: list[tuple[str, ...]] = []
    index = 0
    while index < node_count:
        layer = len(layers)
        width = min(layer_width, node_count - index)
        layer_ids: list[str] = []
        for column in range(width):
            node_id = f"n{index}"
            x, y = _node_position(layer, column)
            nodes.append(
                NodeGlyph(
                    id=node_id,
                    x=x,
                    y=y,
                    width=NODE_WIDTH,
                    height=NODE_HEIGHT,
                    kind=rng.choice(_KINDS),
                    label=f"node_{index}",
                )
            )
            layer_ids.append(node_id)
            index += 1
        layers.append(tuple(layer_ids))
    by_id = {node.id: node for node in nodes}
    edges: list[EdgeGlyph] = []
    for layer_number in range(1, len(layers)):
        previous = layers[layer_number - 1]
        for node_id in layers[layer_number]:
            target = by_id[node_id]
            sources = {rng.choice(previous)}
            extra = int(mean_extra_edges) + (
                1 if rng.random() < mean_extra_edges % 1 else 0
            )
            for _ in range(extra):
                if rng.random() < skip_probability and layer_number > 1:
                    source_layer = layers[rng.randrange(0, layer_number - 1)]
                else:
                    source_layer = previous
                sources.add(rng.choice(source_layer))
            # ``sources`` is a set; iterating it directly would make edge ids
            # and order depend on PYTHONHASHSEED, breaking the byte-identical
            # contract in the module docstring.
            for source_id in sorted(sources):
                source = by_id[source_id]
                edges.append(
                    EdgeGlyph(
                        id=f"e{len(edges)}",
                        source=source_id,
                        target=node_id,
                        points=_edge_points(source, target),
                    )
                )
    return SyntheticGraph(scene=Scene(nodes, edges), layers=tuple(layers))


def collapse_layers(
    graph: SyntheticGraph, first_layer: int, last_layer: int
) -> ScenePatch:
    """A patch that folds layers ``[first_layer, last_layer]`` into one group.

    The group glyph covers the folded layers' area, and every edge crossing the
    group boundary is rerouted to it; edges fully inside disappear. Applying
    the returned patch and then its :meth:`~ScenePatch.inverse_for` restores
    the original scene, which is exactly the collapse/expand cycle the
    benchmark measures.
    """
    if not 0 <= first_layer <= last_layer < len(graph.layers):
        raise ValueError(
            f"layer range [{first_layer}, {last_layer}] is outside "
            f"0..{len(graph.layers) - 1}"
        )
    folded: set[str] = set()
    for layer in graph.layers[first_layer : last_layer + 1]:
        folded.update(layer)
    scene = graph.scene
    group_bounds = graph.scene.node(next(iter(folded))).bounds
    for node_id in folded:
        group_bounds = group_bounds.union(scene.node(node_id).bounds)
    group_id = f"g{first_layer}-{last_layer}"
    group = NodeGlyph(
        id=group_id,
        x=group_bounds.min_x,
        y=group_bounds.min_y,
        width=group_bounds.width,
        height=group_bounds.height,
        kind="group",
        label=f"block {first_layer}..{last_layer}",
    )
    removed_edges: set[str] = set()
    rerouted: list[EdgeGlyph] = []
    for edge in scene.edges:
        source_in = edge.source in folded
        target_in = edge.target in folded
        if not source_in and not target_in:
            continue
        removed_edges.add(edge.id)
        if source_in and target_in:
            continue
        source_id = group_id if source_in else edge.source
        target_id = group_id if target_in else edge.target
        source = group if source_in else scene.node(edge.source)
        target = group if target_in else scene.node(edge.target)
        rerouted.append(
            EdgeGlyph(
                id=f"{edge.id}:collapsed",
                source=source_id,
                target=target_id,
                points=_edge_points(source, target),
            )
        )
    return ScenePatch(
        remove_nodes=frozenset(folded),
        remove_edges=frozenset(removed_edges),
        upsert_nodes=(group,),
        upsert_edges=tuple(rerouted),
    )
