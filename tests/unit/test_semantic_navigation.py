"""Phase 2 semantic LOD and hierarchy-aware navigation tests."""

from __future__ import annotations

import pytest

from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    Group,
    Hierarchy,
)
from nneditor.analysis.layout import layout_graph
from nneditor.analysis.lod import (
    DetailLevel,
    SemanticScene,
    detail_for_scale,
    semantic_scene,
)
from nneditor.application.navigation import Direction, NavigationModel
from nneditor.application.slices import GraphSlicer
from nneditor.ir.core import Graph
from nneditor.rendering.scene import Bounds, EdgeGlyph, NodeGlyph, Scene, Viewport
from tests.unit.test_hierarchy import linear_graph, make_document


def hierarchy_for(graph: Graph) -> Hierarchy:
    evidence = (DetectionEvidence("test.group", "fixture group"),)
    outer = Group(
        id="grp:outer",
        graph_id=graph.id,
        label="Encoder",
        members=frozenset(("n0", "n1", "n2", "n3")),
        kind=CandidateKind.REPEATED,
        confidence=0.95,
        evidence=evidence,
    )
    inner = Group(
        id="grp:inner",
        graph_id=graph.id,
        label="Conv layer",
        members=frozenset(("n0", "n1")),
        kind=CandidateKind.PATTERN,
        confidence=0.98,
        evidence=evidence,
        parent_id=outer.id,
    )
    return Hierarchy(graph.id, (outer, inner))


class TestSemanticLod:
    def test_scale_thresholds_cover_four_levels(self) -> None:
        assert detail_for_scale(0.01) is DetailLevel.ARCHITECTURE
        assert detail_for_scale(0.2) is DetailLevel.BLOCK
        assert detail_for_scale(0.5) is DetailLevel.LAYER
        assert detail_for_scale(1.0) is DetailLevel.OPERATOR

    def test_operator_layer_block_and_architecture_representations(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Relu"))
        scene = layout_graph(graph).scene
        hierarchy = hierarchy_for(graph)

        operator = semantic_scene(scene, hierarchy, DetailLevel.OPERATOR)
        layer = semantic_scene(scene, hierarchy, DetailLevel.LAYER)
        block = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        architecture = semantic_scene(scene, hierarchy, DetailLevel.ARCHITECTURE)

        assert operator.scene.node_count == 5
        assert operator.scene is scene
        assert operator.members_by_glyph["n0"] == {"n0"}
        assert layer.scene.has_node("grp:inner")
        assert layer.scene.node_count == 4
        assert block.scene.has_node("grp:outer")
        assert block.scene.node_count == 2
        assert architecture.scene.has_node("grp:outer")
        assert architecture.scene.node_count == 2
        assert block.scene.edge_count == 1, "internal edges are aggregated away"
        assert operator.scene.node("n0").display_type == "Conv"
        assert block.scene.node("grp:outer").display_type == "Repeated block"
        assert layer.scene.node("grp:inner").display_type == "Pattern block"

    def test_root_group_limits_the_operator_scene(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Relu"))
        result = semantic_scene(
            layout_graph(graph).scene,
            hierarchy_for(graph),
            DetailLevel.OPERATOR,
            root_group="grp:inner",
        )
        assert {node.id for node in result.scene.nodes} == {"n0", "n1"}
        assert result.scene.edge_count == 1

    def test_architecture_fallback_enforces_bounded_overview(self) -> None:
        nodes = tuple(
            NodeGlyph(f"n{index}", float(index), 0.0, 1.0, 1.0, "other", "n")
            for index in range(2501)
        )
        scene = Scene(nodes)
        result = semantic_scene(
            scene,
            Hierarchy("g:main"),
            DetailLevel.ARCHITECTURE,
            max_architecture_nodes=1000,
        )
        assert result.scene.node_count <= 1000
        assert all(node.kind == "group" for node in result.scene.nodes)
        assert all(
            node.display_type == "Architecture region" for node in result.scene.nodes
        )
        represented = frozenset().union(*result.members_by_glyph.values())
        assert represented == {node.id for node in nodes}

    def test_aggregated_edges_are_deterministic_and_count_parallel_edges(self) -> None:
        nodes = (
            NodeGlyph("a", 0, 0, 10, 10, "other", "a"),
            NodeGlyph("b", 20, 0, 10, 10, "other", "b"),
            NodeGlyph("c", 40, 0, 10, 10, "other", "c"),
        )
        edges = (
            EdgeGlyph("e1", "a", "c", ((10, 5), (40, 5))),
            EdgeGlyph("e2", "b", "c", ((30, 5), (40, 5))),
        )
        hierarchy = Hierarchy(
            "g:main",
            (
                Group(
                    "grp:ab",
                    "g:main",
                    "AB",
                    frozenset(("a", "b")),
                    CandidateKind.MANUAL,
                    1.0,
                    (DetectionEvidence("manual", "manual group"),),
                    automatic=False,
                ),
            ),
        )
        first = semantic_scene(Scene(nodes, edges), hierarchy, DetailLevel.BLOCK)
        second = semantic_scene(Scene(nodes, edges), hierarchy, DetailLevel.BLOCK)
        assert first.scene.edges == second.scene.edges
        assert first.scene.edge_count == 1
        assert first.scene.edges[0].id.endswith(":2")

    def test_unknown_root_group_is_explicit(self) -> None:
        graph = linear_graph(("Conv", "Relu"))
        with pytest.raises(KeyError):
            semantic_scene(
                layout_graph(graph).scene,
                Hierarchy(graph.id),
                DetailLevel.OPERATOR,
                root_group="missing",
            )


class TestSemanticSlicer:
    def test_hierarchy_revision_and_detail_participate_in_cache_key(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Relu"))
        document = make_document(graph)
        hierarchy = hierarchy_for(graph)
        slicer = GraphSlicer()
        operator = slicer.slice_graph(
            document, hierarchy=hierarchy, detail_level=DetailLevel.OPERATOR
        )
        block = slicer.slice_graph(
            document, hierarchy=hierarchy, detail_level=DetailLevel.BLOCK
        )
        assert operator is not block
        assert slicer.cached_layouts == 2
        assert block.hierarchy_revision == hierarchy.revision
        assert block.representative_for("n0") == "grp:outer"

    def test_root_group_and_viewport_slices(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Relu"))
        document = make_document(graph)
        hierarchy = hierarchy_for(graph)
        slicer = GraphSlicer()
        rooted = slicer.slice_graph(
            document,
            hierarchy=hierarchy,
            root_group="grp:inner",
        )
        assert rooted.scene.node_count == 2
        first = rooted.scene.node("n0")
        cropped = slicer.slice_graph(
            document,
            hierarchy=hierarchy,
            root_group="grp:inner",
            viewport=first.bounds,
        )
        assert cropped.scene.node_count == 1
        assert cropped.members_by_glyph == {"n0": frozenset(("n0",))}

    def test_hierarchy_from_wrong_graph_is_rejected(self) -> None:
        graph = linear_graph(("Conv", "Relu"))
        with pytest.raises(ValueError, match="different graph"):
            GraphSlicer().slice_graph(
                make_document(graph), hierarchy=Hierarchy("g:other")
            )

    def test_graph_aware_search_identifies_owning_graph(self) -> None:
        main = linear_graph(("Conv",), graph_id="g:main")
        child = linear_graph(("Softmax",), graph_id="g:child")
        document = make_document(main, child)
        hits = GraphSlicer().search_hits(document, "soft")
        assert len(hits) == 1
        assert hits[0].graph_id == "g:child"
        assert hits[0].node_id == "n0"


class TestNavigation:
    def test_breadcrumbs_include_graph_and_nested_groups(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Relu"))
        document = make_document(graph)
        hierarchy = hierarchy_for(graph)
        crumbs = NavigationModel().breadcrumbs(
            document, {graph.id: hierarchy}, graph.id, "grp:inner"
        )
        assert [(item.label, item.kind) for item in crumbs] == [
            ("main", "graph"),
            ("Encoder", "group"),
            ("Conv layer", "group"),
        ]

    def test_directional_navigation_prefers_aligned_neighbor(self) -> None:
        scene = Scene(
            (
                NodeGlyph("center", 10, 10, 5, 5, "other", "center"),
                NodeGlyph("right", 30, 10, 5, 5, "other", "right"),
                NodeGlyph("diagonal", 25, 40, 5, 5, "other", "diagonal"),
                NodeGlyph("left", -10, 10, 5, 5, "other", "left"),
            )
        )
        navigation = NavigationModel()
        assert navigation.directional(scene, "center", Direction.RIGHT) == "right"
        assert navigation.directional(scene, "center", Direction.LEFT) == "left"
        assert navigation.directional(scene, "center", Direction.UP) is None
        # A stale selection id (the node left the scene) is "nowhere to move
        # from", not an error — the contract is to return None.
        assert navigation.directional(scene, "missing", Direction.DOWN) is None

    def test_minimap_geometry_and_world_mapping(self) -> None:
        scene = Scene(
            (
                NodeGlyph("a", 0, 0, 10, 10, "other", "a"),
                NodeGlyph("b", 90, 90, 10, 10, "other", "b"),
            )
        )
        viewport = Viewport(0, 0, 50, 50)
        minimap = NavigationModel().minimap(
            scene, width=100, height=100, viewport=viewport
        )
        assert len(minimap.nodes) == 2
        assert minimap.viewport == Bounds(0, 0, 50, 50)
        assert minimap.world_at(50, 50) == (50, 50)
        with pytest.raises(ValueError, match="positive"):
            NavigationModel().minimap(scene, width=0, height=100)

    def test_selection_maps_to_group_and_back_to_operator(self) -> None:
        group_scene = SemanticScene(
            Scene((NodeGlyph("grp", 0, 0, 10, 10, "group", "g"),)),
            DetailLevel.BLOCK,
            {"grp": frozenset(("n0", "n1"))},
        )
        operator_scene = SemanticScene(
            Scene(
                (
                    NodeGlyph("n0", 0, 0, 10, 10, "other", "n0"),
                    NodeGlyph("n1", 20, 0, 10, 10, "other", "n1"),
                )
            ),
            DetailLevel.OPERATOR,
            {"n0": frozenset(("n0",)), "n1": frozenset(("n1",))},
        )
        navigation = NavigationModel()
        logical = frozenset(("n0",))
        assert navigation.visible_selection(logical, group_scene) == {"grp"}
        assert navigation.visible_selection(logical, operator_scene) == {"n0"}
