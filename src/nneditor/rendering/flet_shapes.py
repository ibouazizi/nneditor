"""Shared culled shape building for the Flet canvas adapters (task P1.7).

Both the naive benchmark adapter and the production managed adapter turn the
same scene-and-viewport input into `flet.canvas` shapes; this module is that
one translation, so the two can never disagree about geometry or styling.

Level-of-detail enforcement lives here too: the ADR caps visible shapes at
roughly 2,000 because no measured Flet path survives an unbounded list. Over
the cap, labels go first, then edges — nodes are never dropped, because an
empty viewport reads as a broken app while missing edge detail reads as a
zoomed-out map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import flet as ft
import flet.canvas as cv

from nneditor.rendering.scene import Scene, Viewport
from nneditor.rendering.spatial import SpatialIndex

__all__ = [
    "BLOCK_BOUNDARY_COLOR",
    "BLOCK_FILL_COLORS",
    "CULL_MARGIN",
    "DEFAULT_NODE_COLOR",
    "EDGE_COLOR",
    "KIND_COLORS",
    "LABEL_MIN_SCALE",
    "SELECTION_COLOR",
    "BuiltShapes",
    "TintFunction",
    "build_shapes",
]

TintFunction = Callable[[str], "str | None"]
"""Overlay seam: node glyph id to an override fill color, or None to keep it.

Analysis overlays (numeric-anomaly tinting) plug in here instead of owning a
second shape builder; a None callable keeps the built output byte-identical
to the untinted path."""

KIND_COLORS: Final[dict[str, str]] = {
    "conv": "#DCEBFF",
    "linear": "#DCEBFF",
    "attention": "#EEE5FF",
    "norm": "#DDF7F2",
    "activation": "#FFF0D9",
    "elementwise": "#FFE5E7",
    "reshape": "#F2E9E4",
    "control-flow": "#E9ECF2",
    "group": "#E2E4FF",
    "graph-input": "#DDF7F2",
    "graph-output": "#E0EAFF",
}
BLOCK_FILL_COLORS: Final[tuple[str, ...]] = (
    "#F9C7C9",
    "#FCD7A1",
    "#F6E7A6",
    "#BFE8D2",
    "#BFDFF7",
    "#D3CCFA",
    "#EDC9F2",
    "#CBE7E5",
)
"""Distinct, soft block fills; assignment is stable by semantic group id."""

BLOCK_BOUNDARY_COLOR: Final = "#111111"
DEFAULT_NODE_COLOR: Final = "#E7EAF0"
EDGE_COLOR: Final = "#C5CAD4"
SELECTION_COLOR: Final = "#5B5CE2"
LABEL_COLOR: Final = "#172033"

LABEL_MIN_SCALE: Final = 0.6
"""Below this zoom a label would be under ~8px tall; drawing it wastes shapes."""

CULL_MARGIN: Final = 0.25
"""Extra viewport fraction culled in on every side so small pans stay covered."""


@dataclass(frozen=True, slots=True)
class BuiltShapes:
    """The drawable output for one viewport plus what LOD did to it."""

    shapes: list[cv.Shape]
    visible_nodes: int
    visible_edges: int
    labels_drawn: bool
    edges_drawn: bool
    dropped_nodes: int = 0
    """Nodes inside the viewport that the shape budget refused to draw.

    Non-zero means the view is deliberately incomplete: the shell reports it
    so an incomplete picture is never mistaken for the whole graph.
    """


def _block_fill(node_id: str) -> str:
    """A deterministic palette choice that does not depend on Python's hash."""
    palette_index = sum(
        (position + 1) * ord(character) for position, character in enumerate(node_id)
    ) % len(BLOCK_FILL_COLORS)
    return BLOCK_FILL_COLORS[palette_index]


def build_shapes(
    scene: Scene,
    viewport: Viewport,
    node_index: SpatialIndex,
    edge_index: SpatialIndex,
    node_order: dict[str, int],
    selection: frozenset[str],
    *,
    max_shapes: int | None = None,
    tint: TintFunction | None = None,
) -> BuiltShapes:
    """Build screen-space shapes for everything near the viewport.

    ``tint`` may override a node glyph's fill color (analysis overlays); when
    it is None or returns None the kind-based fill applies unchanged.
    """
    cull_bounds = viewport.expanded(CULL_MARGIN)
    visible_edges = edge_index.query(cull_bounds)
    visible_nodes = node_index.query(cull_bounds)
    scale = viewport.scale
    to_screen = viewport.to_screen

    draw_labels = scale >= LABEL_MIN_SCALE
    draw_edges = True
    dropped_nodes = 0
    if max_shapes is not None:
        group_outlines = sum(
            scene.node(node_id).kind == "group" for node_id in visible_nodes
        )
        selected_visible = len(selection & visible_nodes)
        # A polyline edge with N points emits N-1 Line shapes, so edges are
        # budgeted by segments, not by count.
        edge_segments = sum(
            len(scene.edge(edge_id).points) - 1 for edge_id in visible_edges
        )
        # Every node emits its fill rect, groups a boundary rect on top, and
        # selected glyphs a highlight rect; the budget counts them all, or an
        # all-selected viewport would emit double the cap.
        estimate = len(visible_nodes) + group_outlines + selected_visible
        if draw_labels and estimate + len(visible_nodes) > max_shapes:
            draw_labels = False
        if estimate + edge_segments > max_shapes:
            draw_edges = False
        elif draw_labels:
            draw_labels = estimate + edge_segments + len(visible_nodes) <= max_shapes
        # Dropping labels and edges is not always enough: a viewport holding
        # tens of thousands of nodes at operator detail would still emit one
        # rectangle each and freeze the UI for seconds. Semantic aggregation
        # (a coarser detail level) is the real answer, so here we keep the
        # frame cheap, draw a deterministic prefix, and report the remainder
        # instead of silently showing a partial graph.
        if estimate > max_shapes:
            candidates = sorted(
                visible_nodes,
                key=lambda node_id: (node_id not in selection, node_order[node_id]),
            )
            keep: list[str] = []
            used_shapes = 0
            for node_id in candidates:
                cost = 1
                if scene.node(node_id).kind == "group":
                    cost += 1
                if node_id in selection:
                    cost += 1
                if used_shapes + cost > max_shapes:
                    # Stop at the first candidate that does not fit: letting a
                    # later cost-1 node squeeze past a skipped selected group
                    # would break "selection first" at the cut and make the
                    # kept set depend on what happens to sit near the boundary.
                    break
                keep.append(node_id)
                used_shapes += cost
            dropped_nodes = len(visible_nodes) - len(keep)
            visible_nodes = set(keep)
            draw_edges = False
            draw_labels = False

    shapes: list[cv.Shape] = []
    if draw_edges:
        edge_paint = ft.Paint(
            color=EDGE_COLOR,
            style=ft.PaintingStyle.STROKE,
            stroke_width=max(1.0, scale),
        )
        selected_edge_paint = ft.Paint(
            color=SELECTION_COLOR,
            style=ft.PaintingStyle.STROKE,
            stroke_width=max(3.0, 3.0 * scale),
        )
        for edge_id in sorted(visible_edges):
            points = scene.edge(edge_id).points
            previous = to_screen(*points[0])
            for point in points[1:]:
                current = to_screen(*point)
                shapes.append(
                    cv.Line(
                        x1=previous[0],
                        y1=previous[1],
                        x2=current[0],
                        y2=current[1],
                        paint=(
                            selected_edge_paint if edge_id in selection else edge_paint
                        ),
                    )
                )
                previous = current
    for node_id in sorted(visible_nodes, key=node_order.__getitem__):
        node = scene.node(node_id)
        left, top = to_screen(node.x, node.y)
        width = node.width * scale
        height = node.height * scale
        fill = (
            _block_fill(node.id)
            if node.kind == "group"
            else KIND_COLORS.get(node.kind, DEFAULT_NODE_COLOR)
        )
        if tint is not None and (override := tint(node_id)) is not None:
            fill = override
        shapes.append(
            cv.Rect(
                x=left,
                y=top,
                width=width,
                height=height,
                border_radius=10.0 * scale,
                paint=ft.Paint(
                    color=fill,
                    style=ft.PaintingStyle.FILL,
                ),
            )
        )
        if node.kind == "group":
            shapes.append(
                cv.Rect(
                    x=left,
                    y=top,
                    width=width,
                    height=height,
                    border_radius=10.0 * scale,
                    paint=ft.Paint(
                        color=BLOCK_BOUNDARY_COLOR,
                        style=ft.PaintingStyle.STROKE,
                        stroke_width=max(1.5, 1.5 * scale),
                    ),
                )
            )
        if node_id in selection:
            shapes.append(
                cv.Rect(
                    x=left,
                    y=top,
                    width=width,
                    height=height,
                    border_radius=10.0 * scale,
                    paint=ft.Paint(
                        color=SELECTION_COLOR,
                        style=ft.PaintingStyle.STROKE,
                        stroke_width=max(2.0, 2.0 * scale),
                    ),
                )
            )
        if draw_labels:
            display_value = node.label
            if node.display_type.casefold() != node.label.casefold():
                display_value += f"\n{node.display_type}"
            shapes.append(
                cv.Text(
                    x=left + width / 2.0,
                    y=top + height / 2.0,
                    value=display_value,
                    style=ft.TextStyle(size=9.0 * scale, color=LABEL_COLOR),
                    alignment=ft.Alignment.CENTER,
                    text_align=ft.TextAlign.CENTER,
                    max_lines=2,
                    max_width=max(1.0, width - 8.0 * scale),
                    ellipsis="…",
                )
            )
    return BuiltShapes(
        shapes=shapes,
        visible_nodes=len(visible_nodes),
        visible_edges=len(visible_edges) if draw_edges else 0,
        labels_drawn=draw_labels,
        edges_drawn=draw_edges,
        dropped_nodes=dropped_nodes,
    )
