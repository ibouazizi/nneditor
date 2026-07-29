"""Phase 2 hierarchy and semantic-navigation vertical slice."""

from __future__ import annotations

from pathlib import Path

from nneditor.analysis.lod import DetailLevel
from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService
from tests.fixtures.onnx_models import (
    build_control_flow_model,
    build_embedded_model,
)


def test_manual_group_collapses_expands_and_preserves_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService() as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        node_ids = frozenset(node.id for node in graph.nodes)
        selected = graph.nodes[0].id
        group = session.hierarchy.group(graph.id, "Pipeline", node_ids)

        block = session.scene(detail_level=DetailLevel.BLOCK)
        assert block.scene.node_count == 1
        assert block.representative_for(selected) == group.id

        operator = session.scene(detail_level=DetailLevel.OPERATOR)
        assert operator.scene.node_count == 2
        assert operator.representative_for(selected) == selected
        assert operator.hierarchy_revision == block.hierarchy_revision


def test_hierarchy_sidecar_survives_application_restart(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    state_dir = tmp_path / "state"
    model_dir.mkdir()
    path = model_dir / "model.onnx"
    build_embedded_model(path, elements=32)

    state = SessionStateStore(state_dir)
    with ApplicationService(state_store=state) as service:
        session = service.open_model(path)
        members = frozenset(node.id for node in session.document.main_graph.nodes)
        session.hierarchy.group(
            session.document.entry_graph, "Remembered group", members
        )

    with ApplicationService(
        state_store=SessionStateStore(state_dir)
    ) as reopened_service:
        reopened = reopened_service.open_model(path)
        assert any(
            group.label == "Remembered group"
            for group in reopened.graph_hierarchy().groups
        )


def test_cross_graph_search_breadcrumbs_and_root_navigation(tmp_path: Path) -> None:
    path = tmp_path / "control.onnx"
    build_control_flow_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        child = next(
            graph
            for graph in session.document.graphs.values()
            if graph.id != session.document.entry_graph
        )
        target = child.nodes[0]
        hits = session.search_hits(target.op_type)
        hit = next(item for item in hits if item.node_id == target.id)
        assert hit.graph_id == child.id
        crumbs = session.breadcrumbs(child.id)
        assert crumbs[0].id == session.document.entry_graph
        assert crumbs[-1].id == child.id
        assert session.scene(child.id).scene.has_node(target.id)
