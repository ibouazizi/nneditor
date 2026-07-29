"""Tests for the Flet Canvas renderer adapter (P0.5).

Everything here runs headless: the adapter builds plain shape objects, so the
exact code the UI executes is exercised without a display server.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv
import pytest

from nneditor.rendering.contract import GraphRenderer
from nneditor.rendering.flet_canvas import LABEL_MIN_SCALE, FletCanvasRenderer
from nneditor.rendering.flet_shapes import BLOCK_BOUNDARY_COLOR, SELECTION_COLOR
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene, ScenePatch, Viewport
from nneditor.rendering.synthetic import collapse_layers, synthetic_graph

VIEWPORT = Viewport(0.0, 0.0, 800.0, 600.0, scale=1.0)


def small_scene() -> Scene:
    nodes = [
        NodeGlyph(id="a", x=0, y=0, width=100, height=40, kind="conv", label="a"),
        NodeGlyph(id="b", x=0, y=100, width=100, height=40, kind="norm", label="b"),
        NodeGlyph(
            id="far", x=5000, y=5000, width=100, height=40, kind="conv", label="far"
        ),
    ]
    edges = [
        EdgeGlyph(id="e", source="a", target="b", points=((50.0, 40.0), (50.0, 100.0)))
    ]
    return Scene(nodes, edges)


def shapes_of(renderer: FletCanvasRenderer) -> list[cv.Shape]:
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    canvas = detector.content
    assert isinstance(canvas, cv.Canvas)
    return list(canvas.shapes or [])


def test_satisfies_the_renderer_contract() -> None:
    assert isinstance(FletCanvasRenderer(), GraphRenderer)


def test_blocks_have_black_boundaries_and_distinct_fills() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(
        Scene(
            (
                NodeGlyph("group-a", 0, 0, 120, 60, "group", "Encoder"),
                NodeGlyph("group-b", 180, 0, 120, 60, "group", "Decoder"),
            )
        ),
        VIEWPORT,
    )
    rectangles = [shape for shape in shapes_of(renderer) if isinstance(shape, cv.Rect)]
    fills = [
        rectangle.paint.color
        for rectangle in rectangles
        if rectangle.paint is not None
        and rectangle.paint.style is ft.PaintingStyle.FILL
    ]
    boundaries = [
        rectangle
        for rectangle in rectangles
        if rectangle.paint is not None
        and rectangle.paint.style is ft.PaintingStyle.STROKE
        and rectangle.paint.color == BLOCK_BOUNDARY_COLOR
    ]
    assert len(set(fills)) == 2
    assert len(boundaries) == 2


def test_operations_require_a_scene_first() -> None:
    renderer = FletCanvasRenderer()
    assert renderer.scene is None and renderer.viewport is None
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.set_viewport(VIEWPORT)
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.apply_patch(ScenePatch())
    with pytest.raises(RuntimeError, match="replace_scene"):
        renderer.set_selection(frozenset())
    assert renderer.hit_test(0.0, 0.0) is None


def test_replace_scene_builds_culled_shapes() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    stats = renderer.stats
    assert stats.visible_nodes == 2, "the far node is culled"
    assert stats.visible_edges == 1
    # Two rects, one line, and one label per visible node.
    assert stats.shape_count == 5
    assert stats.build_ms >= 0.0
    assert len(shapes_of(renderer)) == 5


def test_connections_are_hit_testable_and_selectable() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    assert renderer.hit_test(50.0, 70.0) == "e"
    renderer.set_selection(frozenset({"e"}))
    selected_lines = [
        shape
        for shape in shapes_of(renderer)
        if isinstance(shape, cv.Line)
        and shape.paint is not None
        and shape.paint.color == SELECTION_COLOR
    ]
    assert len(selected_lines) == 1


def test_node_name_and_type_are_centered_inside_the_node() -> None:
    renderer = FletCanvasRenderer()
    scene = Scene(
        (
            NodeGlyph(
                id="named",
                x=20,
                y=30,
                width=140,
                height=44,
                kind="linear",
                label="projection",
                type_label="com.example::FusedLinear",
            ),
        )
    )
    renderer.replace_scene(scene, VIEWPORT)
    label = next(shape for shape in shapes_of(renderer) if isinstance(shape, cv.Text))
    assert label.value == "projection\ncom.example::FusedLinear"
    assert label.x == 90
    assert label.y == 52
    assert label.alignment == ft.Alignment.CENTER
    assert label.text_align is ft.TextAlign.CENTER
    assert label.max_lines == 2
    assert label.max_width == 132


def test_labels_disappear_below_the_zoom_threshold() -> None:
    renderer = FletCanvasRenderer()
    zoomed_out = Viewport(0.0, 0.0, 8000.0, 6000.0, scale=LABEL_MIN_SCALE / 2.0)
    renderer.replace_scene(small_scene(), zoomed_out)
    assert renderer.stats.visible_nodes == 3
    assert not any(isinstance(shape, cv.Text) for shape in shapes_of(renderer))


def test_panning_changes_the_visible_set() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    renderer.set_viewport(Viewport(4900.0, 4900.0, 800.0, 600.0, scale=1.0))
    assert renderer.stats.visible_nodes == 1
    assert renderer.viewport is not None and renderer.viewport.x == 4900.0


def test_hit_test_maps_world_coordinates_to_nodes() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    assert renderer.hit_test(50.0, 20.0) == "a"
    assert renderer.hit_test(50.0, 120.0) == "b"
    assert renderer.hit_test(300.0, 300.0) is None


def test_selection_adds_a_highlight_shape() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    plain = len(shapes_of(renderer))
    renderer.set_selection(frozenset({"a"}))
    assert renderer.selection == {"a"}
    assert len(shapes_of(renderer)) == plain + 1


def test_selecting_unknown_glyphs_is_an_error() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    with pytest.raises(KeyError, match="ghost"):
        renderer.set_selection(frozenset({"ghost"}))


def test_replace_scene_drops_selection() -> None:
    renderer = FletCanvasRenderer()
    renderer.replace_scene(small_scene(), VIEWPORT)
    renderer.set_selection(frozenset({"a"}))
    renderer.replace_scene(small_scene(), VIEWPORT)
    assert renderer.selection == frozenset()


def test_tap_selects_the_hit_node_and_notifies() -> None:
    picks: list[frozenset[str]] = []
    renderer = FletCanvasRenderer(on_select=picks.append)
    renderer.replace_scene(small_scene(), VIEWPORT)
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    event = ft.TapEvent(
        name="tap_down",
        control=detector,
        local_position=ft.Offset(50.0, 20.0),
    )
    renderer._on_tap_down(event)
    assert picks == [frozenset({"a"})]
    renderer._on_tap_down(
        ft.TapEvent(
            name="tap_down",
            control=detector,
            local_position=ft.Offset(300.0, 300.0),
        )
    )
    assert picks[-1] == frozenset()


def test_patch_updates_scene_indexes_and_selection() -> None:
    graph = synthetic_graph(500, layer_width=25, seed=6)
    renderer = FletCanvasRenderer()
    renderer.replace_scene(graph.scene, VIEWPORT)
    victim = graph.layers[5][0]
    renderer.set_selection(frozenset({victim, graph.layers[0][0]}))

    patch = collapse_layers(graph, 5, 9)
    inverse = patch.inverse_for(graph.scene)
    renderer.apply_patch(patch)
    assert renderer.scene is not None
    assert renderer.scene.has_node("g5-9")
    assert renderer.hit_test(*_center(renderer.scene.node("g5-9"))) == "g5-9"
    assert renderer.selection == {graph.layers[0][0]}, (
        "selection keeps only surviving glyphs"
    )

    renderer.apply_patch(inverse)
    assert renderer.scene.node_count == graph.scene.node_count
    assert renderer.hit_test(*_center(graph.scene.node(victim))) == victim


def _center(node: NodeGlyph) -> tuple[float, float]:
    return (node.x + node.width / 2.0, node.y + node.height / 2.0)
