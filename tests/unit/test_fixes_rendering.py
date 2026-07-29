"""Regression tests for verified rendering defects.

Each test pins one fix:

* monotonic painter order after patches (both canvas adapters),
* the shape budget counting selection highlights, group outlines, and edge
  segments,
* strict-prefix drop semantics at the budget boundary,
* hash-seed-independent synthetic edge generation,
* the additive ``RenderStats.shifted`` flag,
* hit-testing agreeing with painter's order after ``SpatialIndex.replace``,
* rejection of non-finite glyph coordinates,
* taps that do not change the selection skipping the rebuild.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import flet as ft
import flet.canvas as cv
import pytest

from nneditor.rendering.flet_canvas import FletCanvasRenderer
from nneditor.rendering.flet_canvas_managed import ManagedCanvasRenderer
from nneditor.rendering.flet_shapes import SELECTION_COLOR, build_shapes
from nneditor.rendering.scene import (
    Bounds,
    EdgeGlyph,
    NodeGlyph,
    Scene,
    SceneError,
    ScenePatch,
    Viewport,
)
from nneditor.rendering.spatial import SpatialIndex, scene_indexes
from nneditor.rendering.synthetic import synthetic_graph

VIEWPORT = Viewport(0.0, 0.0, 1600.0, 1000.0, scale=1.0)


def _node(
    node_id: str,
    x: float,
    y: float,
    *,
    kind: str = "conv",
    width: float = 100.0,
    height: float = 40.0,
) -> NodeGlyph:
    return NodeGlyph(
        id=node_id, x=x, y=y, width=width, height=height, kind=kind, label=node_id
    )


def _shapes_of(renderer: ManagedCanvasRenderer | FletCanvasRenderer) -> list[cv.Shape]:
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    canvas = detector.content
    assert isinstance(canvas, cv.Canvas)
    return list(canvas.shapes or [])


def _tap(renderer: ManagedCanvasRenderer, x: float, y: float) -> None:
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    renderer._on_tap_down(
        ft.TapEvent(name="tap_down", control=detector, local_position=ft.Offset(x, y))
    )


# -- finding 1: monotonic z-order counter ----------------------------------


@pytest.mark.parametrize("renderer_type", [ManagedCanvasRenderer, FletCanvasRenderer])
def test_patch_remove_then_add_keeps_orders_unique(
    renderer_type: type[ManagedCanvasRenderer] | type[FletCanvasRenderer],
) -> None:
    """Removing nodes then upserting new ones must never collide z-orders.

    With ``len(node_order)`` as the next order, {a:0,b:1,c:2} minus {a,b} plus
    {d,e} yielded {'c':2,'d':1,'e':2} — a tie that made paint and hit order
    hash-seed-dependent.
    """
    renderer = renderer_type()
    scene = Scene(
        [_node("a", 0.0, 0.0), _node("b", 200.0, 0.0), _node("c", 400.0, 0.0)]
    )
    renderer.replace_scene(scene, VIEWPORT)
    renderer.apply_patch(
        ScenePatch(
            remove_nodes=frozenset({"a", "b"}),
            upsert_nodes=(_node("d", 600.0, 0.0), _node("e", 450.0, 10.0)),
        )
    )
    orders = renderer._node_order
    assert len(set(orders.values())) == len(orders), f"z-order collision: {orders}"
    assert orders["e"] > orders["c"], "new nodes must paint above existing ones"
    # "e" overlaps "c" at (470, 20); the newer node must both paint on top and
    # win the hit test.
    assert renderer.hit_test(470.0, 20.0) == "e"
    fills = [shape for shape in _shapes_of(renderer) if isinstance(shape, cv.Rect)]
    xs = [rect.x for rect in fills]
    assert xs.index(450.0) > xs.index(400.0), "e's rect must be painted after c's"


# -- finding 2: selection highlights count against the budget ---------------


def test_all_selected_viewport_stays_within_the_shape_cap() -> None:
    nodes = [_node(f"n{ordinal}", ordinal * 150.0, 0.0) for ordinal in range(30)]
    scene = Scene(nodes)
    node_index, edge_index = scene_indexes(scene)
    order = {node.id: ordinal for ordinal, node in enumerate(nodes)}
    selection = frozenset(node.id for node in nodes)
    viewport = Viewport(0.0, 0.0, 5000.0, 1000.0, scale=1.0)
    built = build_shapes(
        scene, viewport, node_index, edge_index, order, selection, max_shapes=10
    )
    assert len(built.shapes) <= 10, "the cap includes selection highlights"
    # Each kept node costs a fill plus a highlight, so exactly five survive.
    assert built.visible_nodes == 5
    assert built.dropped_nodes == 25
    highlights = [
        shape
        for shape in built.shapes
        if isinstance(shape, cv.Rect)
        and shape.paint is not None
        and shape.paint.color == SELECTION_COLOR
    ]
    assert len(highlights) == 5


# -- finding 3: edges budgeted by segment count -----------------------------


def _polyline_scene(point_count: int) -> Scene:
    nodes = [_node("a", 0.0, 0.0), _node("b", 0.0, 800.0)]
    points = tuple(
        (50.0 + (ordinal % 2) * 10.0, 40.0 + ordinal * 30.0)
        for ordinal in range(point_count)
    )
    edge = EdgeGlyph(id="e", source="a", target="b", points=points)
    return Scene(nodes, [edge])


def test_polyline_edges_are_budgeted_by_segments() -> None:
    scene = _polyline_scene(21)  # 20 Line shapes, but only one edge id
    node_index, edge_index = scene_indexes(scene)
    order = {"a": 0, "b": 1}
    built = build_shapes(
        scene, VIEWPORT, node_index, edge_index, order, frozenset(), max_shapes=10
    )
    assert not built.edges_drawn, "20 segments exceed a 10-shape budget"
    assert built.visible_edges == 0
    assert not any(isinstance(shape, cv.Line) for shape in built.shapes)
    assert len(built.shapes) <= 10

    generous = build_shapes(
        scene, VIEWPORT, node_index, edge_index, order, frozenset(), max_shapes=30
    )
    assert generous.edges_drawn
    lines = [shape for shape in generous.shapes if isinstance(shape, cv.Line)]
    assert len(lines) == 20
    assert len(generous.shapes) <= 30


# -- finding 4: the kept set is a strict prefix at the budget cut -----------


def test_budget_drop_stops_at_the_first_non_fitting_candidate() -> None:
    nodes = [
        _node("s", 0.0, 0.0),
        _node("g", 150.0, 0.0, kind="group"),
        _node("p2", 300.0, 0.0),
        _node("p3", 450.0, 0.0),
        _node("p4", 600.0, 0.0),
    ]
    scene = Scene(nodes)
    node_index, edge_index = scene_indexes(scene)
    order = {node.id: ordinal for ordinal, node in enumerate(nodes)}
    selection = frozenset({"s", "g"})
    built = build_shapes(
        scene, VIEWPORT, node_index, edge_index, order, selection, max_shapes=4
    )
    # "s" costs 2 (fill + highlight); the selected group "g" costs 3 and does
    # not fit, and no later cost-1 node may sneak past it.
    assert built.visible_nodes == 1
    assert built.dropped_nodes == 4
    assert len(built.shapes) == 2
    highlights = [
        shape
        for shape in built.shapes
        if isinstance(shape, cv.Rect)
        and shape.paint is not None
        and shape.paint.color == SELECTION_COLOR
    ]
    assert len(highlights) == 1


# -- finding 5: synthetic scenes are hash-seed independent ------------------

_SIGNATURE_CODE = """
import hashlib
from nneditor.rendering.synthetic import synthetic_graph

graph = synthetic_graph(60, layer_width=8, seed=7)
payload = repr(
    [(edge.id, edge.source, edge.target, edge.points) for edge in graph.scene.edges]
)
print(hashlib.sha256(payload.encode()).hexdigest())
"""


def _edge_signature(hash_seed: str) -> str:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", _SIGNATURE_CODE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def test_synthetic_edges_do_not_depend_on_pythonhashseed() -> None:
    assert _edge_signature("0") == _edge_signature("12345")


# -- finding 6: shift frames are flagged in the stats -----------------------


def test_shift_frames_set_the_shifted_flag() -> None:
    renderer = ManagedCanvasRenderer()
    graph = synthetic_graph(200, layer_width=20, seed=3)
    renderer.replace_scene(graph.scene, VIEWPORT)
    assert renderer.stats.shifted is False
    rebuilt = renderer.stats

    renderer.set_viewport(Viewport(40.0, 30.0, 1600.0, 1000.0, scale=1.0))
    assert renderer.move_count == 1
    assert renderer.stats.shifted is True
    assert renderer.stats.visible_nodes == rebuilt.visible_nodes
    assert renderer.stats.visible_edges == rebuilt.visible_edges

    renderer.set_viewport(Viewport(5000.0, 5000.0, 1600.0, 1000.0, scale=1.0))
    assert renderer.stats.shifted is False, "a rebuild frame clears the flag"


# -- finding 7: hits agree with painter's order after replace ---------------


def test_spatial_hit_prefers_the_highest_painter_order() -> None:
    index = SpatialIndex(100.0)
    index.insert("below", Bounds(0.0, 0.0, 10.0, 10.0))
    index.insert("above", Bounds(5.0, 5.0, 15.0, 15.0))
    order = {"below": 0, "above": 1}
    assert index.hit(7.0, 7.0, order=order) == "above"
    # replace() re-inserts, so plain insertion order now puts "below" on top;
    # the painter-order lookup must still pick "above".
    index.replace("below", Bounds(0.0, 0.0, 10.0, 10.0))
    assert index.hit(7.0, 7.0) == "below"
    assert index.hit(7.0, 7.0, order=order) == "above"


def test_patched_node_is_not_hit_above_what_occludes_it() -> None:
    renderer = ManagedCanvasRenderer()
    scene = Scene([_node("under", 0.0, 0.0), _node("over", 50.0, 10.0)])
    renderer.replace_scene(scene, VIEWPORT)
    assert renderer.hit_test(75.0, 20.0) == "over"
    # Upserting "under" replaces its index entry; painter's order keeps it
    # below "over", so the click must keep selecting "over".
    renderer.apply_patch(ScenePatch(upsert_nodes=(_node("under", 0.0, 0.0),)))
    assert renderer.hit_test(75.0, 20.0) == "over"


# -- finding 8: non-finite coordinates are rejected -------------------------


def test_non_finite_coordinates_are_rejected() -> None:
    with pytest.raises(SceneError, match="non-finite"):
        Bounds(math.nan, 0.0, 1.0, 1.0)
    with pytest.raises(SceneError, match="non-finite"):
        Bounds(0.0, 0.0, math.inf, 1.0)
    with pytest.raises(SceneError, match="non-finite"):
        NodeGlyph(
            id="n", x=math.nan, y=0.0, width=10.0, height=10.0, kind="conv", label="n"
        )
    with pytest.raises(SceneError, match="non-finite"):
        NodeGlyph(
            id="n", x=0.0, y=0.0, width=math.inf, height=10.0, kind="conv", label="n"
        )
    with pytest.raises(SceneError, match="non-finite"):
        EdgeGlyph(id="e", source="a", target="b", points=((0.0, 0.0), (math.nan, 1.0)))


# -- finding 9: no-op taps skip the rebuild ---------------------------------


def test_taps_that_do_not_change_the_selection_skip_the_rebuild() -> None:
    picks: list[frozenset[str]] = []
    renderer = ManagedCanvasRenderer(on_select=picks.append)
    graph = synthetic_graph(100, layer_width=10, seed=1)
    renderer.replace_scene(graph.scene, VIEWPORT)
    node = graph.scene.nodes[0]
    assert renderer.rebuild_count == 1

    _tap(renderer, 1599.0, 999.0)  # empty space, empty selection: a no-op
    assert renderer.rebuild_count == 1
    _tap(renderer, node.x + 5.0, node.y + 5.0)
    assert renderer.rebuild_count == 2
    _tap(renderer, node.x + 5.0, node.y + 5.0)  # same node again: a no-op
    assert renderer.rebuild_count == 2
    assert picks == [frozenset(), frozenset({node.id}), frozenset({node.id})]
