"""Tests for the renderer-agnostic scene model (P0.5)."""

from __future__ import annotations

import pytest

from nneditor.rendering.scene import (
    Bounds,
    EdgeGlyph,
    NodeGlyph,
    PatchError,
    Scene,
    SceneError,
    ScenePatch,
    Viewport,
)


def node(node_id: str, x: float = 0.0, y: float = 0.0) -> NodeGlyph:
    return NodeGlyph(
        id=node_id, x=x, y=y, width=10.0, height=5.0, kind="conv", label=node_id
    )


def edge(edge_id: str, source: str, target: str) -> EdgeGlyph:
    return EdgeGlyph(
        id=edge_id, source=source, target=target, points=((0.0, 0.0), (1.0, 1.0))
    )


class TestBounds:
    def test_size_properties(self) -> None:
        bounds = Bounds(1.0, 2.0, 4.0, 8.0)
        assert bounds.width == 3.0
        assert bounds.height == 6.0

    def test_inverted_bounds_are_rejected(self) -> None:
        with pytest.raises(SceneError, match="inverted"):
            Bounds(2.0, 0.0, 1.0, 5.0)

    def test_intersection_includes_touching(self) -> None:
        a = Bounds(0.0, 0.0, 1.0, 1.0)
        assert a.intersects(Bounds(1.0, 1.0, 2.0, 2.0))
        assert not a.intersects(Bounds(1.1, 0.0, 2.0, 1.0))

    def test_contains_point_is_inclusive(self) -> None:
        bounds = Bounds(0.0, 0.0, 2.0, 2.0)
        assert bounds.contains_point(0.0, 2.0)
        assert not bounds.contains_point(2.1, 1.0)

    def test_union_covers_both(self) -> None:
        merged = Bounds(0.0, 0.0, 1.0, 1.0).union(Bounds(5.0, -1.0, 6.0, 0.5))
        assert merged == Bounds(0.0, -1.0, 6.0, 1.0)


class TestGlyphValidation:
    def test_node_needs_positive_size(self) -> None:
        with pytest.raises(SceneError, match="positive size"):
            NodeGlyph(id="a", x=0, y=0, width=0, height=5, kind="conv", label="a")

    def test_node_needs_an_id(self) -> None:
        with pytest.raises(SceneError, match="non-empty"):
            NodeGlyph(id="", x=0, y=0, width=1, height=1, kind="conv", label="")

    def test_node_display_type_prefers_semantic_type_and_has_a_fallback(
        self,
    ) -> None:
        assert node("a").display_type == "Conv"
        typed = NodeGlyph(
            id="typed",
            x=0,
            y=0,
            width=10,
            height=5,
            kind="other",
            label="custom",
            type_label="com.example::Custom",
        )
        assert typed.display_type == "com.example::Custom"

    def test_edge_needs_two_points(self) -> None:
        with pytest.raises(SceneError, match="at least two points"):
            EdgeGlyph(id="e", source="a", target="b", points=((0.0, 0.0),))

    def test_edge_bounds_cover_the_polyline(self) -> None:
        glyph = EdgeGlyph(
            id="e",
            source="a",
            target="b",
            points=((0.0, 5.0), (3.0, -1.0), (2.0, 2.0)),
        )
        assert glyph.bounds == Bounds(0.0, -1.0, 3.0, 5.0)


class TestScene:
    def test_duplicate_node_ids_are_rejected(self) -> None:
        with pytest.raises(SceneError, match="duplicate"):
            Scene([node("a"), node("a")])

    def test_edge_ids_share_the_node_id_space(self) -> None:
        with pytest.raises(SceneError, match="duplicate"):
            Scene([node("a"), node("b")], [edge("a", "a", "b")])

    def test_dangling_edges_are_rejected(self) -> None:
        with pytest.raises(SceneError, match="missing node"):
            Scene([node("a")], [edge("e", "a", "ghost")])

    def test_bounds_union_all_nodes(self) -> None:
        scene = Scene([node("a", 0.0, 0.0), node("b", 100.0, 50.0)])
        assert scene.bounds == Bounds(0.0, 0.0, 110.0, 55.0)

    def test_empty_scene_has_zero_bounds(self) -> None:
        scene = Scene([])
        assert scene.bounds == Bounds(0.0, 0.0, 0.0, 0.0)
        assert scene.node_count == 0 and scene.edge_count == 0

    def test_lookups(self) -> None:
        scene = Scene([node("a"), node("b")], [edge("e", "a", "b")])
        assert scene.node("a").id == "a"
        assert scene.edge("e").target == "b"
        assert scene.has_node("b") and not scene.has_node("e")
        assert scene.has_edge("e") and not scene.has_edge("a")


class TestScenePatch:
    def make_scene(self) -> Scene:
        return Scene([node("a"), node("b"), node("c")], [edge("e", "a", "b")])

    def test_apply_removes_and_upserts(self) -> None:
        scene = self.make_scene()
        patched = scene.apply(
            ScenePatch(
                remove_nodes=frozenset({"c"}),
                upsert_nodes=(node("a", 9.0, 9.0), node("d")),
            )
        )
        assert not patched.has_node("c")
        assert patched.node("a").x == 9.0
        assert patched.has_node("d")
        assert scene.has_node("c"), "the original scene is untouched"

    def test_removing_unknown_ids_fails_loudly(self) -> None:
        with pytest.raises(PatchError, match="unknown node"):
            self.make_scene().apply(ScenePatch(remove_nodes=frozenset({"ghost"})))
        with pytest.raises(PatchError, match="unknown edge"):
            self.make_scene().apply(ScenePatch(remove_edges=frozenset({"ghost"})))

    def test_a_patch_cannot_create_dangling_edges(self) -> None:
        patch = ScenePatch(remove_nodes=frozenset({"a"}))
        with pytest.raises(PatchError, match="invalid scene"):
            self.make_scene().apply(patch)

    def test_is_empty(self) -> None:
        assert ScenePatch().is_empty
        assert not ScenePatch(remove_nodes=frozenset({"a"})).is_empty

    def test_inverse_restores_the_original_scene(self) -> None:
        scene = self.make_scene()
        patch = ScenePatch(
            remove_nodes=frozenset({"c"}),
            remove_edges=frozenset({"e"}),
            upsert_nodes=(node("a", 9.0, 9.0), node("d")),
            upsert_edges=(edge("f", "a", "b"),),
        )
        inverse = patch.inverse_for(scene)
        restored = scene.apply(patch).apply(inverse)
        assert {n.id for n in restored.nodes} == {"a", "b", "c"}
        assert restored.node("a").x == 0.0
        assert {e.id for e in restored.edges} == {"e"}


class TestViewport:
    def test_positive_size_and_scale_required(self) -> None:
        with pytest.raises(SceneError):
            Viewport(0, 0, 0, 10)
        with pytest.raises(SceneError):
            Viewport(0, 0, 10, 10, scale=0)

    def test_round_trip_between_screen_and_world(self) -> None:
        viewport = Viewport(100.0, 50.0, 800.0, 600.0, scale=2.0)
        world = viewport.to_world(30.0, 40.0)
        assert world == (115.0, 70.0)
        assert viewport.to_screen(*world) == (30.0, 40.0)

    def test_expanded_grows_every_side(self) -> None:
        viewport = Viewport(0.0, 0.0, 100.0, 50.0)
        assert viewport.expanded(0.1) == Bounds(-10.0, -5.0, 110.0, 55.0)
