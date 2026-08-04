"""The production Flet renderer adapter under the ADR update regime (P1.7).

The naive adapter rebuilds its shape list on every viewport change, which the
Phase 0 benchmark measured at 170-1,000 ms per frame — Flet's per-update
diffing of a fresh object list dominates. This adapter applies the regime ADR
0001 makes mandatory:

* **Pans inside the culled region mutate.** The shape objects persist and only
  their coordinate fields change, which the probe measured at 18-23 ms per
  frame — inside the 33 ms budget.
* **Rebuilds are confined** to cull-boundary crossings, zoom changes, scene
  patches, and selection changes. They pay the naive cost, but at click and
  settle frequency instead of every frame.
* **A shape cap enforces level of detail.** Above :data:`MAX_SHAPES`, labels
  are dropped first, then edges; semantic aggregation (Phase 2) will raise
  the ceiling properly.

Like the naive adapter it is fully functional headless, which is how the unit
tests verify the regime: a pan inside the cull bounds must keep the same list
object, and a boundary crossing must replace it.
"""

from __future__ import annotations

import time
from typing import Final

import flet as ft
import flet.canvas as cv

from nneditor.rendering.contract import RenderStats, SelectionCallback
from nneditor.rendering.flet_shapes import CULL_MARGIN, TintFunction, build_shapes
from nneditor.rendering.hit_testing import hit_test_glyph
from nneditor.rendering.scene import Bounds, Scene, ScenePatch, Viewport
from nneditor.rendering.spatial import SpatialIndex, scene_indexes

__all__ = ["MAX_SHAPES", "ManagedCanvasRenderer"]

MAX_SHAPES: Final = 2000
"""ADR 0001's visible-shape ceiling."""


class ManagedCanvasRenderer:
    """`GraphRenderer` implementation with persistent, mutated shapes."""

    __slots__ = (
        "_additive_selection",
        "_built_cull",
        "_built_origin",
        "_built_scale",
        "_canvas",
        "_detector",
        "_edge_index",
        "_node_index",
        "_node_order",
        "_on_select",
        "_order_counter",
        "_rebuilds",
        "_scene",
        "_selection",
        "_shape_moves",
        "_stats",
        "_tint",
        "_viewport",
    )

    def __init__(self, *, on_select: SelectionCallback | None = None) -> None:
        self._canvas = cv.Canvas(expand=True)
        self._detector = ft.GestureDetector(
            content=self._canvas, on_tap_down=self._on_tap_down
        )
        self._on_select = on_select
        self._scene: Scene | None = None
        self._viewport: Viewport | None = None
        self._node_index: SpatialIndex | None = None
        self._edge_index: SpatialIndex | None = None
        self._node_order: dict[str, int] = {}
        # Monotonic z-order source: ``len(self._node_order)`` would reuse
        # orders after removals and make ties hash-seed-dependent.
        self._order_counter = 0
        self._selection: frozenset[str] = frozenset()
        self._additive_selection = False
        self._stats = RenderStats(0, 0, 0, 0.0)
        # The world position shapes were last built or shifted to, the culled
        # world bounds they cover, and the scale they were built at.
        self._built_origin: tuple[float, float] | None = None
        self._built_cull: Bounds | None = None
        self._built_scale = 0.0
        self._rebuilds = 0
        self._shape_moves = 0
        # Optional analysis overlay: node glyph id -> override fill color.
        # Applied on every rebuild; pans mutate the already-tinted shapes in
        # place, so the overlay survives the fast path untouched.
        self._tint: TintFunction | None = None

    # -- contract properties ----------------------------------------------

    @property
    def control(self) -> ft.Control:
        return self._detector

    @property
    def scene(self) -> Scene | None:
        return self._scene

    @property
    def viewport(self) -> Viewport | None:
        return self._viewport

    @property
    def stats(self) -> RenderStats:
        return self._stats

    @property
    def selection(self) -> frozenset[str]:
        return self._selection

    @property
    def rebuild_count(self) -> int:
        """Full shape-list rebuilds so far; tests assert the regime with it."""
        return self._rebuilds

    @property
    def move_count(self) -> int:
        """In-place pan shifts so far."""
        return self._shape_moves

    # -- contract operations ----------------------------------------------

    def replace_scene(self, scene: Scene, viewport: Viewport) -> None:
        self._scene = scene
        self._viewport = viewport
        self._node_index, self._edge_index = scene_indexes(scene)
        self._node_order = {node.id: order for order, node in enumerate(scene.nodes)}
        self._order_counter = len(self._node_order)
        self._selection = frozenset()
        self._rebuild()

    def apply_patch(self, patch: ScenePatch) -> None:
        if self._scene is None or self._node_index is None or self._edge_index is None:
            raise RuntimeError("apply_patch requires a scene; call replace_scene")
        self._scene = self._scene.apply(patch)
        for node_id in patch.remove_nodes:
            self._node_index.remove(node_id)
            self._node_order.pop(node_id, None)
        for edge_id in patch.remove_edges:
            self._edge_index.remove(edge_id)
        for node in patch.upsert_nodes:
            if node.id in self._node_index:
                self._node_index.replace(node.id, node.bounds)
            else:
                self._node_index.insert(node.id, node.bounds)
                self._node_order[node.id] = self._order_counter
                self._order_counter += 1
        for edge in patch.upsert_edges:
            if edge.id in self._edge_index:
                self._edge_index.replace(edge.id, edge.bounds)
            else:
                self._edge_index.insert(edge.id, edge.bounds)
        self._selection = frozenset(
            glyph_id
            for glyph_id in self._selection
            if self._scene.has_node(glyph_id) or self._scene.has_edge(glyph_id)
        )
        self._rebuild()

    def set_viewport(self, viewport: Viewport) -> None:
        if self._scene is None:
            raise RuntimeError("set_viewport requires a scene; call replace_scene")
        previous = self._viewport
        self._viewport = viewport
        if (
            previous is not None
            and self._built_origin is not None
            and self._built_cull is not None
            and viewport.scale == self._built_scale
            and self._covers(viewport)
        ):
            self._shift(viewport)
            return
        self._rebuild()

    def set_selection(self, ids: frozenset[str]) -> None:
        if self._scene is None:
            raise RuntimeError("set_selection requires a scene; call replace_scene")
        scene = self._scene
        unknown = [
            glyph_id
            for glyph_id in ids
            if not scene.has_node(glyph_id) and not scene.has_edge(glyph_id)
        ]
        if unknown:
            raise KeyError(f"selection references unknown glyphs: {sorted(unknown)}")
        self._selection = ids
        self._rebuild()

    def set_additive_selection(self, enabled: bool) -> None:
        """Toggle explicit multi-selection mode for hierarchy editing."""
        self._additive_selection = enabled

    def hit_test(self, world_x: float, world_y: float) -> str | None:
        if self._scene is None or self._node_index is None or self._edge_index is None:
            return None
        scale = self._viewport.scale if self._viewport is not None else 1.0
        return hit_test_glyph(
            self._scene,
            self._node_index,
            self._edge_index,
            self._node_order,
            world_x,
            world_y,
            tolerance=7.0 / scale,
        )

    # -- extras the shell uses --------------------------------------------

    def set_tint(self, tint: TintFunction | None) -> None:
        """Install or clear the per-glyph fill override, then redraw.

        A non-None callable always triggers a rebuild — even when the same
        callable is passed again — because the data behind it (freshly landed
        statistics) may have changed what it returns. Clearing an already
        clear tint stays a no-op so idle callers never pay a rebuild.
        """
        if tint is None and self._tint is None:
            return
        self._tint = tint
        if self._scene is not None:
            self._rebuild()

    def viewport_focused_on(
        self, node_id: str, *, width: float, height: float, scale: float | None = None
    ) -> Viewport:
        """A viewport centering ``node_id``; the shell applies it explicitly."""
        if self._scene is None:
            raise RuntimeError("focus requires a scene; call replace_scene")
        node = self._scene.node(node_id)
        resolved_scale = (
            scale
            if scale is not None
            else (self._viewport.scale if self._viewport is not None else 1.0)
        )
        center_x = node.x + node.width / 2.0
        center_y = node.y + node.height / 2.0
        world_width = width / resolved_scale
        world_height = height / resolved_scale
        return Viewport(
            x=center_x - world_width / 2.0,
            y=center_y - world_height / 2.0,
            width=world_width,
            height=world_height,
            scale=resolved_scale,
        )

    # -- internals ---------------------------------------------------------

    def _covers(self, viewport: Viewport) -> bool:
        assert self._built_cull is not None
        bounds = viewport.bounds
        return (
            self._built_cull.min_x <= bounds.min_x
            and self._built_cull.min_y <= bounds.min_y
            and bounds.max_x <= self._built_cull.max_x
            and bounds.max_y <= self._built_cull.max_y
        )

    def _shift(self, viewport: Viewport) -> None:
        """Move every existing shape by the pan delta, in place."""
        assert self._built_origin is not None
        started = time.perf_counter()
        delta_x = (self._built_origin[0] - viewport.x) * viewport.scale
        delta_y = (self._built_origin[1] - viewport.y) * viewport.scale
        if delta_x == 0.0 and delta_y == 0.0:
            return
        for shape in self._canvas.shapes or ():
            if isinstance(shape, cv.Line):
                shape.x1 = (shape.x1 or 0.0) + delta_x
                shape.y1 = (shape.y1 or 0.0) + delta_y
                shape.x2 = (shape.x2 or 0.0) + delta_x
                shape.y2 = (shape.y2 or 0.0) + delta_y
            elif isinstance(shape, (cv.Rect, cv.Text)):
                shape.x = (shape.x or 0.0) + delta_x
                shape.y = (shape.y or 0.0) + delta_y
        self._built_origin = (viewport.x, viewport.y)
        self._shape_moves += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._stats = RenderStats(
            visible_nodes=self._stats.visible_nodes,
            visible_edges=self._stats.visible_edges,
            shape_count=self._stats.shape_count,
            build_ms=elapsed_ms,
            dropped_nodes=self._stats.dropped_nodes,
            shifted=True,
        )
        self._flush()

    def _rebuild(self) -> None:
        scene = self._scene
        viewport = self._viewport
        assert scene is not None and viewport is not None
        assert self._node_index is not None and self._edge_index is not None
        started = time.perf_counter()
        built = build_shapes(
            scene,
            viewport,
            self._node_index,
            self._edge_index,
            self._node_order,
            self._selection,
            max_shapes=MAX_SHAPES,
            tint=self._tint,
        )
        self._canvas.shapes = built.shapes
        self._built_origin = (viewport.x, viewport.y)
        self._built_cull = viewport.expanded(CULL_MARGIN)
        self._built_scale = viewport.scale
        self._rebuilds += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._stats = RenderStats(
            visible_nodes=built.visible_nodes,
            visible_edges=built.visible_edges,
            shape_count=len(built.shapes),
            build_ms=elapsed_ms,
            dropped_nodes=built.dropped_nodes,
        )
        self._flush()

    def _on_tap_down(self, event: ft.TapEvent[ft.GestureDetector]) -> None:
        if self._viewport is None:
            return
        position = event.local_position
        if position is None:
            return
        world_x, world_y = self._viewport.to_world(position.x, position.y)
        hit = self.hit_test(world_x, world_y)
        if hit is None:
            selection: frozenset[str] = frozenset()
        elif self._additive_selection:
            selected = set(self._selection)
            if hit in selected:
                selected.remove(hit)
            else:
                selected.add(hit)
            selection = frozenset(selected)
        else:
            selection = frozenset((hit,))
        if selection != self._selection:
            # A no-op tap (empty space with nothing selected, re-tapping the
            # sole selected node) must not pay a full rebuild.
            self._selection = selection
            self._rebuild()
        if self._on_select is not None:
            self._on_select(self._selection)

    def _flush(self) -> None:
        try:
            page = self._canvas.page
        except Exception:
            return
        if page is not None:
            self._canvas.update()
