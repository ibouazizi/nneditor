"""Tests for layered layout and graph slicing (P1.6 core)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.analysis.layout import LayoutSettings, layout_graph
from nneditor.application.slices import GraphSlicer
from nneditor.ir.core import Graph, Node, Value
from nneditor.ir.identity import NodeIdStability
from tests.fixtures.onnx_models import build_control_flow_model, build_embedded_model


def chain_graph() -> Graph:
    """a -> b -> c with a skip edge a -> c."""
    values = [Value(id=f"v{i}", name=f"v{i}") for i in range(4)]
    nodes = [
        Node(
            id="a",
            id_stability=NodeIdStability.NAMED,
            op_type="Conv",
            inputs=(),
            outputs=("v0", "v1"),
        ),
        Node(
            id="b",
            id_stability=NodeIdStability.NAMED,
            op_type="Relu",
            inputs=("v0",),
            outputs=("v2",),
        ),
        Node(
            id="c",
            id_stability=NodeIdStability.NAMED,
            op_type="Add",
            inputs=("v2", "v1"),
            outputs=("v3",),
        ),
    ]
    return Graph(id="g:main", name="main", nodes=nodes, values=values)


def cyclic_graph() -> Graph:
    """a and b feed each other — malformed, but layout must not hang."""
    values = [Value(id="va", name="va"), Value(id="vb", name="vb")]
    nodes = [
        Node(
            id="a",
            id_stability=NodeIdStability.NAMED,
            op_type="Mul",
            inputs=("vb",),
            outputs=("va",),
        ),
        Node(
            id="b",
            id_stability=NodeIdStability.NAMED,
            op_type="Mul",
            inputs=("va",),
            outputs=("vb",),
        ),
    ]
    return Graph(id="g:main", name="main", nodes=nodes, values=values)


class TestLayout:
    def test_layers_follow_longest_paths(self) -> None:
        result = layout_graph(chain_graph())
        assert result.layers == (("a",), ("b",), ("c",))
        assert result.cyclic_nodes == ()
        scene = result.scene
        assert scene.node_count == 3
        assert scene.edge_count == 3, "two chain edges plus the skip edge"
        assert scene.node("a").y < scene.node("b").y < scene.node("c").y

    def test_edges_connect_node_centers(self) -> None:
        settings = LayoutSettings()
        scene = layout_graph(chain_graph(), settings).scene
        edge = next(e for e in scene.edges if e.source == "a" and e.target == "b")
        a = scene.node("a")
        assert edge.points[0] == (
            a.x + settings.node_width / 2,
            a.y + settings.node_height,
        )

    def test_layout_is_deterministic(self) -> None:
        first = layout_graph(chain_graph())
        second = layout_graph(chain_graph())
        assert first.scene.nodes == second.scene.nodes
        assert first.scene.edges == second.scene.edges

    def test_cycles_do_not_hang_and_are_reported(self) -> None:
        result = layout_graph(cyclic_graph())
        assert set(result.cyclic_nodes) == {"a", "b"}
        assert result.scene.node_count == 2

    def test_kind_and_label_mapping(self) -> None:
        scene = layout_graph(chain_graph()).scene
        assert scene.node("a").kind == "conv"
        assert scene.node("b").kind == "activation"
        assert scene.node("c").kind == "elementwise"
        assert scene.node("b").label == "Relu"
        assert scene.node("b").type_label == "Relu"

    def test_settings_are_validated(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LayoutSettings(node_width=0)
        with pytest.raises(ValueError, match="non-negative"):
            LayoutSettings(column_gap=-1)

    def test_empty_graph_lays_out_empty(self) -> None:
        result = layout_graph(Graph(id="g:main", name="main"))
        assert result.scene.node_count == 0
        assert result.layers == ()


class TestSlicer:
    def test_real_model_slices_and_caches(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=32)
        document = index_to_document(index_model(path))
        slicer = GraphSlicer()
        first = slicer.slice_graph(document)
        assert first.scene.node_count == 2
        assert slicer.slice_graph(document) is first, "cache hit"
        assert slicer.cached_layouts == 1

    def test_subgraphs_are_sliceable(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_control_flow_model(path)
        document = index_to_document(index_model(path))
        branch = document.main_graph.nodes[0]
        child = GraphSlicer().slice_graph(document, branch.subgraphs[0])
        assert child.scene.node_count == 1

    def test_unknown_graph_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=32)
        document = index_to_document(index_model(path))
        with pytest.raises(KeyError):
            GraphSlicer().slice_graph(document, "g:ghost")

    def test_cache_capacity_is_bounded(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_control_flow_model(path)
        document = index_to_document(index_model(path))
        slicer = GraphSlicer(cache_capacity=1)
        slicer.slice_graph(document)
        branch = document.main_graph.nodes[0]
        slicer.slice_graph(document, branch.subgraphs[0])
        assert slicer.cached_layouts == 1

    def test_invalidate_by_artifact(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=32)
        document = index_to_document(index_model(path))
        slicer = GraphSlicer()
        slicer.slice_graph(document)
        slicer.invalidate("sha256:not-this-one")
        assert slicer.cached_layouts == 1
        slicer.invalidate(document.source.content_hash)
        assert slicer.cached_layouts == 0
        slicer.slice_graph(document)
        slicer.invalidate()
        assert slicer.cached_layouts == 0

    def test_search_matches_names_ops_and_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=32)
        document = index_to_document(index_model(path))
        slicer = GraphSlicer()
        assert len(slicer.search(document, "mul")) == 1
        assert len(slicer.search(document, "SCALE")) == 1, "source names match"
        assert slicer.search(document, "") == ()
        assert slicer.search(document, "zzz-no-match") == ()

    def test_neighborhood_expands_by_hops(self) -> None:
        graph = chain_graph()
        document_graphs = {"g:main": graph}
        slicer = GraphSlicer()

        class FakeDocument:
            graphs = document_graphs

        document = FakeDocument()
        assert slicer.neighborhood(document, "g:main", "b", hops=0) == {"b"}  # type: ignore[arg-type]
        assert slicer.neighborhood(document, "g:main", "b", hops=1) == {"a", "b", "c"}  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-negative"):
            slicer.neighborhood(document, "g:main", "b", hops=-1)  # type: ignore[arg-type]
        with pytest.raises(KeyError):
            slicer.neighborhood(document, "g:main", "ghost")  # type: ignore[arg-type]
