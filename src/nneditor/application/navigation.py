"""Hierarchy-aware search, breadcrumbs, minimap, and keyboard navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from nneditor.analysis.hierarchy import Hierarchy
from nneditor.analysis.lod import SemanticScene
from nneditor.ir.core import Document
from nneditor.rendering.scene import Bounds, Scene, Viewport

__all__ = [
    "Breadcrumb",
    "Direction",
    "MiniMap",
    "MiniNode",
    "NavigationModel",
    "SearchHit",
]


@dataclass(frozen=True, slots=True)
class SearchHit:
    graph_id: str
    node_id: str
    label: str
    op_type: str


@dataclass(frozen=True, slots=True)
class Breadcrumb:
    id: str
    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class MiniNode:
    id: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class MiniMap:
    width: float
    height: float
    nodes: tuple[MiniNode, ...]
    viewport: Bounds | None
    world_bounds: Bounds
    scale: float
    offset_x: float
    offset_y: float

    def world_at(self, x: float, y: float) -> tuple[float, float]:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("minimap has no drawable area")
        world_x = self.world_bounds.min_x + (x - self.offset_x) / self.scale
        world_y = self.world_bounds.min_y + (y - self.offset_y) / self.scale
        return (
            min(self.world_bounds.max_x, max(self.world_bounds.min_x, world_x)),
            min(self.world_bounds.max_y, max(self.world_bounds.min_y, world_y)),
        )

    def project_viewport(self, viewport: Viewport) -> Bounds:
        """Project a world viewport into minimap coordinates.

        The node dots depend only on the scene, so a shell can keep them and
        reproject just the viewport rectangle as the user pans and zooms.
        """
        return Bounds(
            self.offset_x + (viewport.x - self.world_bounds.min_x) * self.scale,
            self.offset_y + (viewport.y - self.world_bounds.min_y) * self.scale,
            self.offset_x
            + (viewport.x + viewport.width - self.world_bounds.min_x) * self.scale,
            self.offset_y
            + (viewport.y + viewport.height - self.world_bounds.min_y) * self.scale,
        )


class Direction(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class NavigationModel:
    """Pure navigation calculations, independent of Flet controls."""

    def search(self, document: Document, text: str) -> tuple[SearchHit, ...]:
        needle = text.strip().lower()
        if not needle:
            return ()
        hits: list[SearchHit] = []
        for graph in document.graphs.values():
            for node in graph.nodes:
                label = node.source_name or node.op_type
                haystacks = (node.id, label, node.op_type, node.qualified_op_type)
                if any(needle in item.lower() for item in haystacks):
                    hits.append(
                        SearchHit(graph.id, node.id, label, node.qualified_op_type)
                    )
        return tuple(hits)

    def breadcrumbs(
        self,
        document: Document,
        hierarchies: dict[str, Hierarchy],
        graph_id: str,
        group_id: str | None = None,
    ) -> tuple[Breadcrumb, ...]:
        graph_path: list[str] = [graph_id]
        current = document.graphs[graph_id]
        seen = {graph_id}
        while current.parent_node is not None:
            owner_graph: str | None = None
            for graph in document.graphs.values():
                if any(node.id == current.parent_node for node in graph.nodes):
                    owner_graph = graph.id
                    break
            if owner_graph is None or owner_graph in seen:
                break
            graph_path.append(owner_graph)
            seen.add(owner_graph)
            current = document.graphs[owner_graph]
        crumbs = [
            Breadcrumb(
                id=item,
                label=document.graphs[item].name or item,
                kind="graph",
            )
            for item in reversed(graph_path)
        ]
        if group_id is not None:
            crumbs.extend(
                Breadcrumb(group.id, group.label, "group")
                for group in hierarchies[graph_id].breadcrumbs(group_id)
            )
        return tuple(crumbs)

    def directional(
        self, scene: Scene, current_id: str, direction: Direction
    ) -> str | None:
        try:
            current = scene.node(current_id)
        except KeyError:
            # A stale selection (the node left the scene after an LOD change
            # or edit) means there is nowhere to move from, not an error.
            return None
        current_x = current.x + current.width / 2
        current_y = current.y + current.height / 2
        candidates: list[tuple[float, float, str]] = []
        for node in scene.nodes:
            if node.id == current_id:
                continue
            x = node.x + node.width / 2
            y = node.y + node.height / 2
            dx = x - current_x
            dy = y - current_y
            if direction is Direction.LEFT and dx >= 0:
                continue
            if direction is Direction.RIGHT and dx <= 0:
                continue
            if direction is Direction.UP and dy >= 0:
                continue
            if direction is Direction.DOWN and dy <= 0:
                continue
            horizontal = direction in (Direction.LEFT, Direction.RIGHT)
            primary = abs(dx) if horizontal else abs(dy)
            secondary = abs(dy) if horizontal else abs(dx)
            angle_penalty = secondary / max(primary, 1e-9)
            score = math.hypot(dx, dy) * (1.0 + angle_penalty)
            candidates.append((score, secondary, node.id))
        return min(candidates)[2] if candidates else None

    def minimap(
        self,
        scene: Scene,
        *,
        width: float,
        height: float,
        viewport: Viewport | None = None,
    ) -> MiniMap:
        if width <= 0 or height <= 0:
            raise ValueError("minimap dimensions must be positive")
        bounds = scene.bounds
        world_width = max(bounds.width, 1.0)
        world_height = max(bounds.height, 1.0)
        scale = min(width / world_width, height / world_height)
        offset_x = (width - bounds.width * scale) / 2.0
        offset_y = (height - bounds.height * scale) / 2.0
        nodes = tuple(
            MiniNode(
                id=node.id,
                x=offset_x + (node.x - bounds.min_x) * scale,
                y=offset_y + (node.y - bounds.min_y) * scale,
                width=max(1.0, node.width * scale),
                height=max(1.0, node.height * scale),
            )
            for node in scene.nodes
        )
        model = MiniMap(
            width,
            height,
            nodes,
            None,
            bounds,
            scale,
            offset_x,
            offset_y,
        )
        if viewport is None:
            return model
        return replace(model, viewport=model.project_viewport(viewport))

    def visible_selection(
        self, logical_node_ids: frozenset[str], semantic: SemanticScene
    ) -> frozenset[str]:
        """Map operator selection to stable representatives after LOD changes."""
        selected = {
            semantic.representative_for(node_id) for node_id in logical_node_ids
        }
        return frozenset(item for item in selected if item is not None)
