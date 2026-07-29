"""Tests for the deterministic synthetic scene generator (P0.5)."""

from __future__ import annotations

import pytest

from nneditor.rendering.synthetic import collapse_layers, synthetic_graph


class TestSyntheticGraph:
    def test_rejects_non_positive_parameters(self) -> None:
        with pytest.raises(ValueError, match="node_count"):
            synthetic_graph(0)
        with pytest.raises(ValueError, match="layer_width"):
            synthetic_graph(10, layer_width=0)

    def test_exact_node_count_and_layering(self) -> None:
        graph = synthetic_graph(103, layer_width=25)
        assert graph.scene.node_count == 103
        assert [len(layer) for layer in graph.layers] == [25, 25, 25, 25, 3]

    def test_same_seed_is_byte_identical(self) -> None:
        first = synthetic_graph(200, seed=5)
        second = synthetic_graph(200, seed=5)
        assert first.scene.nodes == second.scene.nodes
        assert first.scene.edges == second.scene.edges

    def test_different_seeds_differ(self) -> None:
        first = synthetic_graph(200, seed=1)
        second = synthetic_graph(200, seed=2)
        assert first.scene.edges != second.scene.edges

    def test_every_non_input_node_has_a_predecessor(self) -> None:
        graph = synthetic_graph(300, layer_width=20, seed=3)
        targets = {edge.target for edge in graph.scene.edges}
        for layer in graph.layers[1:]:
            for node_id in layer:
                assert node_id in targets

    def test_edge_volume_matches_the_benchmark_requirement(self) -> None:
        graph = synthetic_graph(10_000, layer_width=50, seed=42)
        assert 20_000 <= graph.scene.edge_count <= 35_000

    def test_edges_only_point_to_earlier_layers(self) -> None:
        graph = synthetic_graph(200, layer_width=20, seed=4)
        layer_of = {
            node_id: number
            for number, layer in enumerate(graph.layers)
            for node_id in layer
        }
        for edge in graph.scene.edges:
            assert layer_of[edge.source] < layer_of[edge.target]


class TestCollapseLayers:
    def test_rejects_out_of_range_layers(self) -> None:
        graph = synthetic_graph(100, layer_width=25)
        with pytest.raises(ValueError, match="outside"):
            collapse_layers(graph, 2, 99)
        with pytest.raises(ValueError, match="outside"):
            collapse_layers(graph, 3, 2)

    def test_collapse_replaces_layers_with_one_group(self) -> None:
        graph = synthetic_graph(500, layer_width=25, seed=6)
        patch = collapse_layers(graph, 5, 9)
        collapsed = graph.scene.apply(patch)
        removed = 5 * 25
        assert collapsed.node_count == graph.scene.node_count - removed + 1
        group = collapsed.node("g5-9")
        assert group.kind == "group"
        for edge in collapsed.edges:
            assert collapsed.has_node(edge.source)
            assert collapsed.has_node(edge.target)

    def test_collapse_then_expand_restores_the_scene(self) -> None:
        graph = synthetic_graph(500, layer_width=25, seed=6)
        patch = collapse_layers(graph, 5, 9)
        inverse = patch.inverse_for(graph.scene)
        restored = graph.scene.apply(patch).apply(inverse)
        assert {n.id for n in restored.nodes} == {n.id for n in graph.scene.nodes}
        assert {e.id for e in restored.edges} == {e.id for e in graph.scene.edges}

    def test_boundary_edges_are_rerouted_to_the_group(self) -> None:
        graph = synthetic_graph(500, layer_width=25, seed=6)
        collapsed = graph.scene.apply(collapse_layers(graph, 5, 9))
        touching_group = [
            edge
            for edge in collapsed.edges
            if edge.source == "g5-9" or edge.target == "g5-9"
        ]
        assert touching_group, "the folded block must stay connected"
        for edge in touching_group:
            assert edge.id.endswith(":collapsed")
