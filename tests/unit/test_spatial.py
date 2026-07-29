"""Tests for the uniform-grid spatial index (P0.5)."""

from __future__ import annotations

import pytest

from nneditor.rendering.scene import Bounds, EdgeGlyph, NodeGlyph, Scene
from nneditor.rendering.spatial import SpatialIndex, scene_indexes


def box(x: float, y: float, size: float = 10.0) -> Bounds:
    return Bounds(x, y, x + size, y + size)


class TestSpatialIndex:
    def test_cell_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SpatialIndex(0.0)

    def test_query_finds_only_intersecting_items(self) -> None:
        index = SpatialIndex(50.0)
        index.insert("near", box(0.0, 0.0))
        index.insert("far", box(500.0, 500.0))
        assert index.query(Bounds(-5.0, -5.0, 20.0, 20.0)) == {"near"}

    def test_items_spanning_cells_are_found_once(self) -> None:
        index = SpatialIndex(10.0)
        index.insert("wide", Bounds(5.0, 5.0, 95.0, 8.0))
        assert index.query(Bounds(0.0, 0.0, 100.0, 100.0)) == {"wide"}

    def test_same_cell_but_not_intersecting_is_excluded(self) -> None:
        index = SpatialIndex(100.0)
        index.insert("a", box(0.0, 0.0))
        assert index.query(Bounds(50.0, 50.0, 60.0, 60.0)) == set()

    def test_duplicate_insert_is_an_error(self) -> None:
        index = SpatialIndex(10.0)
        index.insert("a", box(0.0, 0.0))
        with pytest.raises(ValueError, match="already indexed"):
            index.insert("a", box(1.0, 1.0))

    def test_remove_unknown_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            SpatialIndex(10.0).remove("ghost")

    def test_remove_then_query(self) -> None:
        index = SpatialIndex(10.0)
        index.insert("a", box(0.0, 0.0))
        index.remove("a")
        assert len(index) == 0
        assert index.query(Bounds(0.0, 0.0, 10.0, 10.0)) == set()

    def test_replace_moves_an_item(self) -> None:
        index = SpatialIndex(10.0)
        index.insert("a", box(0.0, 0.0))
        index.replace("a", box(200.0, 200.0))
        assert index.query(Bounds(0.0, 0.0, 20.0, 20.0)) == set()
        assert index.query(Bounds(195.0, 195.0, 205.0, 205.0)) == {"a"}
        assert "a" in index

    def test_hit_returns_topmost_by_insertion_order(self) -> None:
        index = SpatialIndex(100.0)
        index.insert("below", box(0.0, 0.0))
        index.insert("above", box(5.0, 5.0))
        assert index.hit(7.0, 7.0) == "above"
        assert index.hit(1.0, 1.0) == "below"
        assert index.hit(400.0, 400.0) is None

    def test_negative_coordinates_are_indexed_correctly(self) -> None:
        index = SpatialIndex(10.0)
        index.insert("neg", box(-25.0, -25.0))
        assert index.hit(-20.0, -20.0) == "neg"
        assert index.query(Bounds(-30.0, -30.0, -20.0, -20.0)) == {"neg"}


class TestSceneIndexes:
    def test_nodes_and_edges_are_indexed_separately(self) -> None:
        nodes = [
            NodeGlyph(id="a", x=0, y=0, width=10, height=10, kind="conv", label="a"),
            NodeGlyph(
                id="b", x=100, y=100, width=10, height=10, kind="conv", label="b"
            ),
        ]
        edges = [
            EdgeGlyph(
                id="e", source="a", target="b", points=((5.0, 10.0), (105.0, 100.0))
            )
        ]
        node_index, edge_index = scene_indexes(Scene(nodes, edges))
        everything = Bounds(-10.0, -10.0, 200.0, 200.0)
        assert node_index.query(everything) == {"a", "b"}
        assert edge_index.query(everything) == {"e"}
        assert node_index.hit(5.0, 5.0) == "a"

    def test_empty_scene_builds_empty_indexes(self) -> None:
        node_index, edge_index = scene_indexes(Scene([]))
        assert len(node_index) == 0 and len(edge_index) == 0


class TestQueryScaling:
    def test_a_huge_query_costs_content_not_geometry(self) -> None:
        """A viewport spanning millions of empty cells must stay cheap.

        The grid is sized from glyph extents, so a tall, narrow graph viewed
        zoomed out spans an enormous cell rectangle holding almost nothing.
        Walking that rectangle cell-by-cell made a 14k-node overview frame
        take 15 seconds; the query must scale with stored items instead.
        """
        import time

        index = SpatialIndex(10.0)
        for ordinal in range(2_000):
            index.insert(f"n{ordinal}", box(0.0, ordinal * 40.0, 8.0))

        enormous = Bounds(-5_000_000.0, -1000.0, 5_000_000.0, 5_000_000.0)
        started = time.perf_counter()
        found = index.query(enormous)
        elapsed = time.perf_counter() - started

        assert len(found) == 2_000, "every item is still returned"
        assert elapsed < 1.0, f"query took {elapsed:.2f}s; it must stay cheap"

    def test_both_strategies_agree(self) -> None:
        index = SpatialIndex(10.0)
        for ordinal in range(50):
            index.insert(f"n{ordinal}", box(ordinal * 12.0, ordinal * 7.0, 6.0))
        tight = Bounds(0.0, 0.0, 60.0, 60.0)
        wide = Bounds(-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0)
        assert index.query(tight) <= index.query(wide)
        assert index.query(wide) == {f"n{ordinal}" for ordinal in range(50)}
