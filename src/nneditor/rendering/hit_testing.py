"""Shared deterministic node/connection hit testing for render adapters."""

from __future__ import annotations

import math
from itertools import pairwise

from nneditor.rendering.scene import Bounds, Scene
from nneditor.rendering.spatial import SpatialIndex

__all__ = ["hit_test_glyph"]


def _segment_distance(
    x: float,
    y: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length == 0.0:
        return math.hypot(x - start[0], y - start[1])
    fraction = ((x - start[0]) * delta_x + (y - start[1]) * delta_y) / squared_length
    fraction = min(1.0, max(0.0, fraction))
    nearest_x = start[0] + fraction * delta_x
    nearest_y = start[1] + fraction * delta_y
    return math.hypot(x - nearest_x, y - nearest_y)


def hit_test_glyph(
    scene: Scene,
    node_index: SpatialIndex,
    edge_index: SpatialIndex,
    node_order: dict[str, int],
    world_x: float,
    world_y: float,
    *,
    tolerance: float,
) -> str | None:
    """Pick a node first, then the nearest connection within ``tolerance``."""
    node = node_index.hit(world_x, world_y, order=node_order)
    if node is not None:
        return node
    candidates = edge_index.query(
        Bounds(
            world_x - tolerance,
            world_y - tolerance,
            world_x + tolerance,
            world_y + tolerance,
        )
    )
    ranked: list[tuple[float, str]] = []
    for edge_id in candidates:
        points = scene.edge(edge_id).points
        distance = min(
            _segment_distance(world_x, world_y, start, end)
            for start, end in pairwise(points)
        )
        if distance <= tolerance:
            ranked.append((distance, edge_id))
    return min(ranked)[1] if ranked else None
