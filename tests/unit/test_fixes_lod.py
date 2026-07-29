"""Regression tests for the LOD antichain, drill-down, drill-through,
collapsed-scene re-layout, and constant-feeder folding fixes."""

from __future__ import annotations

from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    Group,
    Hierarchy,
)
from nneditor.analysis.layout import layered_order, layout_graph
from nneditor.analysis.lod import DetailLevel, SemanticScene, semantic_scene
from nneditor.ir.core import Graph, Node, Value
from nneditor.ir.identity import NodeIdStability
from nneditor.rendering.scene import NodeGlyph, Scene
from tests.unit.test_hierarchy import linear_graph

_EVIDENCE = (DetectionEvidence("test.fixture", "fixture group"),)


def make_group(
    graph_id: str,
    group_id: str,
    label: str,
    members: frozenset[str],
    *,
    kind: CandidateKind = CandidateKind.SOURCE,
    confidence: float = 0.85,
    parent_id: str | None = None,
) -> Group:
    return Group(
        id=group_id,
        graph_id=graph_id,
        label=label,
        members=members,
        kind=kind,
        confidence=confidence,
        evidence=_EVIDENCE,
        parent_id=parent_id,
    )


def scope_fixture() -> tuple[Scene, Hierarchy]:
    """A 16-op chain under nested scopes encoder/layerN/{attn,ffn}.

    This is the shape that previously degenerated the four-level LOD to two:
    every level selected the outermost scope and rendered one glyph.
    """
    graph = linear_graph(("Gemm",) * 16)
    groups = [
        make_group(
            graph.id,
            "grp:encoder",
            "encoder",
            frozenset(f"n{index}" for index in range(16)),
        )
    ]
    for layer in range(4):
        base = layer * 4
        layer_id = f"grp:layer{layer}"
        groups.append(
            make_group(
                graph.id,
                layer_id,
                f"layer{layer}",
                frozenset(f"n{base + offset}" for offset in range(4)),
                parent_id="grp:encoder",
            )
        )
        groups.append(
            make_group(
                graph.id,
                f"grp:attn{layer}",
                f"attn{layer}",
                frozenset((f"n{base}", f"n{base + 1}")),
                parent_id=layer_id,
            )
        )
        groups.append(
            make_group(
                graph.id,
                f"grp:ffn{layer}",
                f"ffn{layer}",
                frozenset((f"n{base + 2}", f"n{base + 3}")),
                parent_id=layer_id,
            )
        )
    return layout_graph(graph).scene, Hierarchy(graph.id, groups)


def constant_heavy_graph(
    *, blocks: int = 2, ops_per_block: int = 10, feeders: int = 5
) -> Graph:
    """Independent op chains where every op is fed by ``feeders`` Constants."""
    values: list[Value] = []
    nodes: list[Node] = []
    for block in range(blocks):
        for position in range(ops_per_block):
            index = block * ops_per_block + position
            inputs: list[str] = [] if position == 0 else [f"v{index - 1}"]
            for feeder in range(feeders):
                constant_value = f"cv{index}_{feeder}"
                values.append(Value(id=constant_value, name=constant_value))
                nodes.append(
                    Node(
                        id=f"c{index}_{feeder}",
                        id_stability=NodeIdStability.NAMED,
                        op_type="Constant",
                        inputs=(),
                        outputs=(constant_value,),
                    )
                )
                inputs.append(constant_value)
            values.append(Value(id=f"v{index}", name=f"v{index}"))
            nodes.append(
                Node(
                    id=f"n{index}",
                    id_stability=NodeIdStability.NAMED,
                    op_type="Gemm",
                    inputs=tuple(inputs),
                    outputs=(f"v{index}",),
                )
            )
    return Graph(id="g:main", name="main", nodes=nodes, values=values)


def assert_exact_partition(semantic: SemanticScene, expected: frozenset[str]) -> None:
    """Every operator maps to exactly one glyph and none is lost."""
    assert set(semantic.members_by_glyph) == {node.id for node in semantic.scene.nodes}
    seen: dict[str, str] = {}
    for glyph_id, members in semantic.members_by_glyph.items():
        for member in members:
            assert member not in seen, (
                f"{member} appears in {seen[member]} and {glyph_id}"
            )
            seen[member] = glyph_id
    assert frozenset(seen) == expected


def overlapping_area(scene: Scene) -> list[tuple[str, str]]:
    """Pairs of node glyphs sharing area (touching edges do not count)."""
    pairs: list[tuple[str, str]] = []
    nodes = scene.nodes
    for position, left in enumerate(nodes):
        a = left.bounds
        for right in nodes[position + 1 :]:
            b = right.bounds
            if (
                a.min_x < b.max_x
                and b.min_x < a.max_x
                and a.min_y < b.max_y
                and b.min_y < a.max_y
            ):
                pairs.append((left.id, right.id))
    return pairs


class TestAntichainSelection:
    def test_block_prefers_the_most_specific_groups(self) -> None:
        scene, hierarchy = scope_fixture()
        block = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        glyph_ids = {node.id for node in block.scene.nodes}
        assert glyph_ids == {
            f"grp:{kind}{layer}" for kind in ("attn", "ffn") for layer in range(4)
        }, "nested scopes must not collapse block detail into the outer glyph"
        assert not block.scene.has_node("grp:encoder")
        assert_exact_partition(block, frozenset(f"n{index}" for index in range(16)))

    def test_layer_prefers_the_most_specific_pattern_group(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm"))
        outer = make_group(
            graph.id,
            "grp:outer",
            "Outer",
            frozenset(("n0", "n1", "n2")),
            kind=CandidateKind.PATTERN,
        )
        inner = make_group(
            graph.id,
            "grp:inner",
            "Inner",
            frozenset(("n0", "n1")),
            kind=CandidateKind.PATTERN,
            parent_id="grp:outer",
        )
        layer = semantic_scene(
            layout_graph(graph).scene,
            Hierarchy(graph.id, (outer, inner)),
            DetailLevel.LAYER,
        )
        assert layer.scene.has_node("grp:inner")
        assert not layer.scene.has_node("grp:outer")

    def test_architecture_without_domination_keeps_the_root_antichain(self) -> None:
        graph = linear_graph(("Gemm",) * 16)
        half_a = make_group(
            graph.id, "grp:a", "A", frozenset(f"n{index}" for index in range(8))
        )
        half_b = make_group(
            graph.id, "grp:b", "B", frozenset(f"n{index}" for index in range(8, 16))
        )
        child = make_group(
            graph.id,
            "grp:a0",
            "A0",
            frozenset(("n0", "n1")),
            parent_id="grp:a",
        )
        architecture = semantic_scene(
            layout_graph(graph).scene,
            Hierarchy(graph.id, (half_a, half_b, child)),
            DetailLevel.ARCHITECTURE,
        )
        assert {node.id for node in architecture.scene.nodes} == {"grp:a", "grp:b"}


class TestDrillDown:
    def test_drilling_into_a_group_shows_its_child_groups(self) -> None:
        scene, hierarchy = scope_fixture()
        drilled = semantic_scene(
            scene, hierarchy, DetailLevel.BLOCK, root_group="grp:layer0"
        )
        assert not drilled.scene.has_node("grp:layer0"), (
            "the drill root itself must not be reselected"
        )
        assert {node.id for node in drilled.scene.nodes} == {
            "grp:attn0",
            "grp:ffn0",
        }
        assert_exact_partition(drilled, hierarchy.group("grp:layer0").members)

    def test_drilling_into_a_leaf_group_shows_its_operators(self) -> None:
        scene, hierarchy = scope_fixture()
        drilled = semantic_scene(
            scene, hierarchy, DetailLevel.BLOCK, root_group="grp:attn1"
        )
        # n4's producer (n3) is cropped away by the drill, but n4 is not a
        # pure source in the full graph, so it must keep its own glyph.
        assert {node.id for node in drilled.scene.nodes} == {"n4", "n5"}

    def test_drilling_at_layer_detail_shows_the_interior(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm"))
        outer = make_group(
            graph.id,
            "grp:outer",
            "Outer",
            frozenset(("n0", "n1", "n2", "n3")),
            kind=CandidateKind.PATTERN,
        )
        inner = make_group(
            graph.id,
            "grp:inner",
            "Inner",
            frozenset(("n0", "n1")),
            kind=CandidateKind.PATTERN,
            parent_id="grp:outer",
        )
        drilled = semantic_scene(
            layout_graph(graph).scene,
            Hierarchy(graph.id, (outer, inner)),
            DetailLevel.LAYER,
            root_group="grp:outer",
        )
        assert not drilled.scene.has_node("grp:outer")
        assert {node.id for node in drilled.scene.nodes} == {
            "grp:inner",
            "n2",
            "n3",
        }

    def test_drilling_at_architecture_detail_selects_the_children(self) -> None:
        scene, hierarchy = scope_fixture()
        drilled = semantic_scene(
            scene, hierarchy, DetailLevel.ARCHITECTURE, root_group="grp:layer1"
        )
        assert {node.id for node in drilled.scene.nodes} == {
            "grp:attn1",
            "grp:ffn1",
        }


class TestDrillThrough:
    def test_architecture_descends_through_a_dominant_scope_chain(self) -> None:
        graph = linear_graph(("Gemm",) * 12)
        all_nodes = frozenset(f"n{index}" for index in range(12))
        root = make_group(graph.id, "grp:root", "model", all_nodes)
        wrapper = make_group(
            graph.id,
            "grp:wrapper",
            "blocks",
            frozenset(f"n{index}" for index in range(11)),
            parent_id="grp:root",
        )
        children = [
            make_group(
                graph.id,
                f"grp:block{block}",
                f"block{block}",
                frozenset(
                    f"n{index}" for index in range(block * 4, min(block * 4 + 4, 11))
                ),
                parent_id="grp:wrapper",
            )
            for block in range(3)
        ]
        hierarchy = Hierarchy(graph.id, (root, wrapper, *children))
        architecture = semantic_scene(
            layout_graph(graph).scene, hierarchy, DetailLevel.ARCHITECTURE
        )
        glyph_ids = {node.id for node in architecture.scene.nodes}
        assert "grp:root" not in glyph_ids
        assert "grp:wrapper" not in glyph_ids
        assert {"grp:block0", "grp:block1", "grp:block2"} <= glyph_ids
        assert_exact_partition(architecture, all_nodes)

    def test_architecture_stops_at_the_first_meaningful_fanout(self) -> None:
        scene, hierarchy = scope_fixture()
        architecture = semantic_scene(scene, hierarchy, DetailLevel.ARCHITECTURE)
        assert {node.id for node in architecture.scene.nodes} == {
            f"grp:layer{layer}" for layer in range(4)
        }, "descend past the all-covering scope, but not into attn/ffn"
        assert_exact_partition(
            architecture, frozenset(f"n{index}" for index in range(16))
        )

    def test_dominant_group_without_children_is_kept(self) -> None:
        graph = linear_graph(("Gemm",) * 4)
        only = make_group(
            graph.id, "grp:only", "only", frozenset(("n0", "n1", "n2", "n3"))
        )
        architecture = semantic_scene(
            layout_graph(graph).scene,
            Hierarchy(graph.id, (only,)),
            DetailLevel.ARCHITECTURE,
        )
        assert architecture.scene.has_node("grp:only")
        assert architecture.scene.node_count == 1


class TestConstantFolding:
    def test_constants_fold_into_their_consumer_glyph_at_block_detail(self) -> None:
        graph = constant_heavy_graph()
        members_a = frozenset(f"n{index}" for index in range(10))
        members_b = frozenset(f"n{index}" for index in range(10, 20))
        hierarchy = Hierarchy(
            graph.id,
            (
                make_group(
                    graph.id, "grp:a", "A", members_a, kind=CandidateKind.REPEATED
                ),
                make_group(
                    graph.id, "grp:b", "B", members_b, kind=CandidateKind.REPEATED
                ),
            ),
        )
        scene = layout_graph(graph).scene
        block = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        total = len(graph.nodes)
        assert block.scene.node_count == 2
        assert block.scene.node_count * 10 <= total, (
            "folding must reduce block glyphs by an order of magnitude"
        )
        assert block.representative_for("c0_0") == "grp:a"
        assert block.representative_for("c15_3") == "grp:b"
        assert_exact_partition(block, frozenset(node.id for node in graph.nodes))

        operator = semantic_scene(scene, hierarchy, DetailLevel.OPERATOR)
        assert operator.scene.node_count == total, (
            "operator detail keeps showing feeders individually"
        )
        assert operator.scene.has_node("c0_0")

    def test_entry_compute_ops_are_not_folded(self) -> None:
        graph = linear_graph(("Mul", "Add"))
        block = semantic_scene(
            layout_graph(graph).scene, Hierarchy(graph.id), DetailLevel.BLOCK
        )
        # "n0" consumes only graph inputs, making it structurally a pure
        # source, but it is a compute op and must keep its own glyph.
        assert {node.id for node in block.scene.nodes} == {"n0", "n1"}

    def test_shared_feeders_and_grouped_sources_are_not_folded(self) -> None:
        values = [
            Value(id="vcs", name="vcs"),
            Value(id="vct", name="vct"),
            Value(id="v1", name="v1"),
            Value(id="v2", name="v2"),
            Value(id="v3", name="v3"),
        ]
        nodes = [
            Node(
                id="cs",
                id_stability=NodeIdStability.NAMED,
                op_type="Constant",
                inputs=(),
                outputs=("vcs",),
            ),
            Node(
                id="ct",
                id_stability=NodeIdStability.NAMED,
                op_type="Constant",
                inputs=(),
                outputs=("vct",),
            ),
            Node(
                id="a1",
                id_stability=NodeIdStability.NAMED,
                op_type="Gemm",
                inputs=("vcs",),
                outputs=("v1",),
            ),
            Node(
                id="a2",
                id_stability=NodeIdStability.NAMED,
                op_type="Relu",
                inputs=("v1",),
                outputs=("v2",),
            ),
            Node(
                id="t",
                id_stability=NodeIdStability.NAMED,
                op_type="Add",
                inputs=("v2", "vcs", "vct"),
                outputs=("v3",),
            ),
        ]
        graph = Graph(id="g:main", name="main", nodes=nodes, values=values)
        hierarchy = Hierarchy(
            graph.id,
            (
                make_group(
                    graph.id,
                    "grp:a",
                    "A",
                    frozenset(("a1", "a2")),
                    kind=CandidateKind.REPEATED,
                ),
            ),
        )
        block = semantic_scene(layout_graph(graph).scene, hierarchy, DetailLevel.BLOCK)
        glyph_ids = {node.id for node in block.scene.nodes}
        # "cs" feeds two glyphs, so it stays; "ct" folds into the ungrouped
        # consumer "t", whose glyph then represents both nodes.
        assert glyph_ids == {"grp:a", "cs", "t"}
        assert block.representative_for("ct") == "t"
        assert block.members_by_glyph["t"] == frozenset(("t", "ct"))
        assert block.members_by_glyph["cs"] == frozenset(("cs",))


class TestCompactGeometry:
    def test_block_scene_is_compact_non_overlapping_and_connected(self) -> None:
        scene, hierarchy = scope_fixture()
        block = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        assert overlapping_area(block.scene) == []
        for node in block.scene.nodes:
            assert node.kind == "group"
            assert node.height == 64.0
            assert 180.0 <= node.width <= 420.0
        for edge in block.scene.edges:
            source = block.scene.node(edge.source)
            target = block.scene.node(edge.target)
            assert edge.points[0] == (
                source.x + source.width / 2.0,
                source.y + source.height,
            )
            assert edge.points[-1] == (target.x + target.width / 2.0, target.y)
        operator = semantic_scene(scene, hierarchy, DetailLevel.OPERATOR)
        assert block.scene.bounds.height < operator.scene.bounds.height, (
            "eight compact glyphs must not inherit the 16-op layout extents"
        )

    def test_collapsed_relayout_is_deterministic(self) -> None:
        scene, hierarchy = scope_fixture()
        first = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        second = semantic_scene(scene, hierarchy, DetailLevel.BLOCK)
        assert first.scene.nodes == second.scene.nodes
        assert first.scene.edges == second.scene.edges

    def test_overview_fallback_operates_on_the_relaid_scene(self) -> None:
        nodes = tuple(
            NodeGlyph(f"n{index}", float(index), 0.0, 1.0, 1.0, "other", "n")
            for index in range(2501)
        )
        overview = semantic_scene(
            Scene(nodes),
            Hierarchy("g:main"),
            DetailLevel.ARCHITECTURE,
            max_architecture_nodes=1000,
        )
        assert overview.scene.node_count <= 1000
        assert all(node.kind == "group" for node in overview.scene.nodes)
        assert overlapping_area(overview.scene) == []
        assert_exact_partition(overview, frozenset(node.id for node in nodes))


class TestLayeredOrder:
    def test_layers_respect_edges(self) -> None:
        rows = layered_order(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
        assert rows == (("a",), ("b",), ("c",))

    def test_cycles_do_not_hang_and_land_on_one_row(self) -> None:
        rows = layered_order(("a", "b"), (("a", "b"), ("b", "a")))
        assert len(rows) == 1
        assert set(rows[0]) == {"a", "b"}

    def test_ordering_is_deterministic(self) -> None:
        identifiers = tuple(f"n{index}" for index in range(20))
        edges = tuple((f"n{index}", f"n{index + 2}") for index in range(18))
        assert layered_order(identifiers, edges) == layered_order(identifiers, edges)
