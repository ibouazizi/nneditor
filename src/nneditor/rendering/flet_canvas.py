"""Flet Canvas implementation of the renderer contract (task P0.5).

This is the pure-Flet candidate in the Phase 0 renderer comparison: glyphs
become ``flet.canvas`` shapes rebuilt in screen coordinates on every viewport
change. That is deliberately the *conservative* design — no Flutter-side
transform tricks — so the benchmark measures the worst honest per-frame cost.
If this passes the frame budget, cheaper variants only improve on it; the
optimization levers (GPU-side pan between rebuilds, shape reuse) are recorded
in ``docs/renderer-benchmark.md`` rather than implemented speculatively.

The adapter works without a mounted page: shape lists are plain objects, so
headless benchmarks and unit tests exercise exactly the code the UI runs, and
:meth:`FletCanvasRenderer._flush` simply does nothing until the control is on
a page.
"""

from __future__ import annotations

import time

import flet as ft
import flet.canvas as cv

from nneditor.rendering.contract import RenderStats, SelectionCallback
from nneditor.rendering.flet_shapes import (
    CULL_MARGIN,
    KIND_COLORS,
    LABEL_MIN_SCALE,
    build_shapes,
)
from nneditor.rendering.hit_testing import hit_test_glyph
from nneditor.rendering.scene import Scene, ScenePatch, Viewport
from nneditor.rendering.spatial import SpatialIndex, scene_indexes

__all__ = ["CULL_MARGIN", "KIND_COLORS", "LABEL_MIN_SCALE", "FletCanvasRenderer"]


class FletCanvasRenderer:
    """Draws one scene on a ``flet.canvas.Canvas``, culled to the viewport."""

    __slots__ = (
        "_canvas",
        "_detector",
        "_edge_index",
        "_node_index",
        "_node_order",
        "_on_select",
        "_order_counter",
        "_scene",
        "_selection",
        "_stats",
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
        self._stats = RenderStats(0, 0, 0, 0.0)

    @property
    def control(self) -> ft.Control:
        """The control to mount into a Flet layout."""
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
        # Selection may reference removed glyphs; keep only what still exists.
        self._selection = frozenset(
            glyph_id
            for glyph_id in self._selection
            if self._scene.has_node(glyph_id) or self._scene.has_edge(glyph_id)
        )
        self._rebuild()

    def set_viewport(self, viewport: Viewport) -> None:
        if self._scene is None:
            raise RuntimeError("set_viewport requires a scene; call replace_scene")
        self._viewport = viewport
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

    def _on_tap_down(self, event: ft.TapEvent[ft.GestureDetector]) -> None:
        if self._viewport is None:
            return
        position = event.local_position
        if position is None:
            return
        world_x, world_y = self._viewport.to_world(position.x, position.y)
        hit = self.hit_test(world_x, world_y)
        self._selection = frozenset() if hit is None else frozenset((hit,))
        self._rebuild()
        if self._on_select is not None:
            self._on_select(self._selection)

    def _rebuild(self) -> None:
        """Recompute the culled, screen-space shape list for the viewport."""
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
        )
        self._canvas.shapes = built.shapes
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._stats = RenderStats(
            visible_nodes=built.visible_nodes,
            visible_edges=built.visible_edges,
            shape_count=len(built.shapes),
            build_ms=elapsed_ms,
        )
        self._flush()

    def _flush(self) -> None:
        """Push the new shapes to Flutter when mounted; no-op when headless."""
        try:
            page = self._canvas.page
        except Exception:
            return
        if page is not None:
            self._canvas.update()
