"""Tests for the production managed canvas adapter (P1.7).

The regime itself is what these tests pin down: pans inside the culled region
must mutate the existing shape list, and only boundary crossings, zoom
changes, patches, and selection changes may rebuild it.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv
import pytest

from nneditor.rendering.contract import GraphRenderer
from nneditor.rendering.flet_canvas_managed import MAX_SHAPES, ManagedCanvasRenderer
from nneditor.rendering.flet_shapes import SELECTION_COLOR
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene, ScenePatch, Viewport
from nneditor.rendering.synthetic import collapse_layers, synthetic_graph

VIEWPORT = Viewport(0.0, 0.0, 1600.0, 1000.0, scale=1.0)


def shapes_of(renderer: ManagedCanvasRenderer) -> list[cv.Shape]:
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    canvas = detector.content
    assert isinstance(canvas, cv.Canvas)
    return list(canvas.shapes or [])


def raw_shape_list(renderer: ManagedCanvasRenderer) -> object:
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    canvas = detector.content
    assert isinstance(canvas, cv.Canvas)
    return canvas.shapes


def scene_10k() -> Scene:
    return synthetic_graph(10_000, layer_width=50, seed=42).scene


def test_satisfies_the_renderer_contract() -> None:
    assert isinstance(ManagedCanvasRenderer(), GraphRenderer)


def test_small_pans_mutate_instead_of_rebuilding() -> None:
    renderer = ManagedCanvasRenderer()
    renderer.replace_scene(scene_10k(), VIEWPORT)
    assert renderer.rebuild_count == 1
    before = raw_shape_list(renderer)
    first_shape = shapes_of(renderer)[0]
    assert isinstance(first_shape, cv.Line | cv.Rect)

    renderer.set_viewport(Viewport(40.0, 30.0, 1600.0, 1000.0, scale=1.0))
    assert renderer.rebuild_count == 1, "no rebuild inside the cull margin"
    assert renderer.move_count == 1
    assert raw_shape_list(renderer) is before, "the shape list object persists"

    if isinstance(first_shape, cv.Rect):
        moved_x = first_shape.x
    else:
        moved_x = first_shape.x1
    assert moved_x is not None


def test_shift_moves_coordinates_by_the_pan_delta() -> None:
    renderer = ManagedCanvasRenderer()
    renderer.replace_scene(scene_10k(), VIEWPORT)
    rect = next(s for s in shapes_of(renderer) if isinstance(s, cv.Rect))
    original_x = rect.x
    assert original_x is not None
    renderer.set_viewport(Viewport(100.0, 0.0, 1600.0, 1000.0, scale=1.0))
    assert rect.x == pytest.approx(original_x - 100.0)


def test_leaving_the_cull_margin_rebuilds() -> None:
    renderer = ManagedCanvasRenderer()
    renderer.replace_scene(scene_10k(), VIEWPORT)
    before = raw_shape_list(renderer)
    renderer.set_viewport(Viewport(2000.0, 2000.0, 1600.0, 1000.0, scale=1.0))
    assert renderer.rebuild_count == 2
    assert raw_shape_list(renderer) is not before


def test_zoom_changes_rebuild() -> None:
    renderer = ManagedCanvasRenderer()
    renderer.replace_scene(scene_10k(), VIEWPORT)
    renderer.set_viewport(Viewport(0.0, 0.0, 3200.0, 2000.0, scale=0.5))
    assert renderer.rebuild_count == 2


def test_lod_cap_drops_edges_when_everything_is_visible() -> None:
    renderer = ManagedCanvasRenderer()
    scene = scene_10k()
    fit_scale = 0.05
    overview = Viewport(
        x=scene.bounds.min_x,
        y=scene.bounds.min_y,
        width=1600.0 / fit_scale,
        height=1000.0 / fit_scale,
        scale=fit_scale,
    )
    renderer.replace_scene(scene, overview)
    stats = renderer.stats
    # Labels and edges go first; beyond that the node budget itself binds, so
    # a zoomed-out operator-level view stays a cheap frame instead of a
    # multi-second freeze. The remainder is reported, never silently hidden.
    assert stats.visible_edges == 0, "edges dropped by the shape cap"
    assert stats.visible_nodes == MAX_SHAPES
    assert stats.dropped_nodes == 10_000 - MAX_SHAPES
    assert stats.shape_count <= MAX_SHAPES
    assert not any(isinstance(s, cv.Line) for s in shapes_of(renderer))


def test_a_scene_within_budget_reports_no_drops() -> None:
    renderer = ManagedCanvasRenderer()
    graph = synthetic_graph(200, layer_width=20, seed=3)
    renderer.replace_scene(graph.scene, VIEWPORT)
    assert renderer.stats.dropped_nodes == 0


def test_selection_survives_the_node_budget() -> None:
    """A selected node must stay drawn even when most nodes are dropped."""
    renderer = ManagedCanvasRenderer()
    scene = scene_10k()
    renderer.replace_scene(scene, VIEWPORT)
    far = scene.nodes[-1]
    renderer.set_selection(frozenset({far.id}))
    overview = Viewport(
        x=scene.bounds.min_x,
        y=scene.bounds.min_y,
        width=scene.bounds.width * 1.1,
        height=scene.bounds.height * 1.1,
        scale=0.05,
    )
    renderer.set_viewport(overview)
    assert renderer.stats.dropped_nodes > 0
    # The highlight rect proves the selected node survived, and it is counted
    # against the budget, so the total never exceeds the cap.
    assert renderer.stats.shape_count == MAX_SHAPES
    highlights = [
        shape
        for shape in shapes_of(renderer)
        if isinstance(shape, cv.Rect)
        and shape.paint is not None
        and shape.paint.color == SELECTION_COLOR
    ]
    assert len(highlights) == 1


def test_selection_and_patch_rebuild_and_survive() -> None:
    graph = synthetic_graph(500, layer_width=25, seed=6)
    renderer = ManagedCanvasRenderer()
    renderer.replace_scene(graph.scene, VIEWPORT)
    keep = graph.layers[0][0]
    victim = graph.layers[5][0]
    renderer.set_selection(frozenset({keep, victim}))
    rebuilds = renderer.rebuild_count

    patch = collapse_layers(graph, 5, 9)
    renderer.apply_patch(patch)
    assert renderer.rebuild_count == rebuilds + 1
    assert renderer.selection == {keep}
    assert renderer.scene is not None and renderer.scene.has_node("g5-9")
    with pytest.raises(KeyError):
        renderer.set_selection(frozenset({"ghost"}))


def test_operations_require_a_scene() -> None:
    renderer = ManagedCanvasRenderer()
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.set_viewport(VIEWPORT)
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.set_selection(frozenset())
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.apply_patch(ScenePatch())
    with pytest.raises(RuntimeError, match="requires a scene"):
        renderer.viewport_focused_on("x", width=100, height=100)
    assert renderer.hit_test(0.0, 0.0) is None


def test_tap_selects_and_notifies() -> None:
    picks: list[frozenset[str]] = []
    renderer = ManagedCanvasRenderer(on_select=picks.append)
    graph = synthetic_graph(100, layer_width=10, seed=1)
    renderer.replace_scene(graph.scene, VIEWPORT)
    node = graph.scene.nodes[0]
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    renderer._on_tap_down(
        ft.TapEvent(
            name="tap_down",
            control=detector,
            local_position=ft.Offset(node.x + 5.0, node.y + 5.0),
        )
    )
    assert picks == [frozenset({node.id})]


def test_tap_selects_a_connection_when_no_node_is_under_pointer() -> None:
    picks: list[frozenset[str]] = []
    renderer = ManagedCanvasRenderer(on_select=picks.append)
    scene = Scene(
        (
            NodeGlyph("a", 0, 0, 100, 40, "conv", "a"),
            NodeGlyph("b", 0, 100, 100, 40, "conv", "b"),
        ),
        (EdgeGlyph("edge", "a", "b", ((50.0, 40.0), (50.0, 100.0))),),
    )
    renderer.replace_scene(scene, VIEWPORT)
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    renderer._on_tap_down(
        ft.TapEvent(
            name="tap_down",
            control=detector,
            local_position=ft.Offset(50.0, 70.0),
        )
    )
    assert renderer.selection == {"edge"}
    assert picks == [frozenset({"edge"})]


def test_explicit_additive_mode_toggles_multiple_nodes() -> None:
    renderer = ManagedCanvasRenderer()
    graph = synthetic_graph(100, layer_width=10, seed=1)
    renderer.replace_scene(graph.scene, VIEWPORT)
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    renderer.set_additive_selection(True)
    first, second = graph.scene.nodes[:2]

    def tap(node_x: float, node_y: float) -> None:
        renderer._on_tap_down(
            ft.TapEvent(
                name="tap_down",
                control=detector,
                local_position=ft.Offset(node_x + 5.0, node_y + 5.0),
            )
        )

    tap(first.x, first.y)
    tap(second.x, second.y)
    assert renderer.selection == {first.id, second.id}
    tap(first.x, first.y)
    assert renderer.selection == {second.id}


def test_focus_viewport_centers_the_node() -> None:
    renderer = ManagedCanvasRenderer()
    graph = synthetic_graph(100, layer_width=10, seed=1)
    renderer.replace_scene(graph.scene, VIEWPORT)
    node = graph.scene.nodes[42]
    focused = renderer.viewport_focused_on(node.id, width=800.0, height=600.0)
    center = (focused.x + focused.width / 2, focused.y + focused.height / 2)
    assert center[0] == pytest.approx(node.x + node.width / 2)
    assert center[1] == pytest.approx(node.y + node.height / 2)
    assert focused.scale == VIEWPORT.scale
