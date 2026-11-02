"""Traced blocks expose real activation boundaries with a separate inspection pane."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.editing.validation import RenameNodeRequest
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene
from nneditor.tracing.contracts import (
    CaptureState,
    TraceApproval,
    TraceLimits,
    TraceRequest,
)
from nneditor.ui.trace_graph import TraceGraphPresentation
from nneditor.ui.trace_ports import boundary_values, graph_ports
from tests.fixtures.qwen35_models import write_qwen35
from tests.unit.test_shell import make_shell, press_key
from tests.unit.test_watch_overlay import find_data, open_traced_shell, texts


def _event() -> Any:
    return SimpleNamespace()


def test_completed_trace_opens_own_pane_and_port_overlay(tmp_path: Path) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page, _session = open_traced_shell(service, tmp_path)
        pane = shell.traced_model
        assert shell.workspace_tabs.selected_index == 2
        assert not shell.left_toggle.visible and not shell.right_toggle.visible
        assert pane.trace_picker.value == shell.trace.active_trace_id
        ports = find_data(pane.body, "trace-port:")
        assert ports
        selection = shell.renderer.selection
        button = cast(ft.Container, ports[0])
        semantics = cast(ft.Semantics, button.content)
        assert semantics.hint_text and "Click to inspect" in semantics.hint_text
        ellipse = cast(ft.Container, semantics.content)
        assert ellipse.width == 20 and ellipse.height == 12
        cast(Any, button.on_click)(_event())
        overlay = shell.activations.value_overlay
        assert overlay.dialog is page.dialogs[-1]
        assert overlay.tabs.length == 4
        assert shell.renderer.selection == selection
        assert shell.activations.close_overlay()
        assert not overlay.dialog
        pane._on_size(cast(Any, SimpleNamespace(width=1100, height=550)))
        view = pane._view
        center = (view.x + view.width / 2, view.y + view.height / 2)
        pane._on_size(cast(Any, SimpleNamespace(width=760, height=400)))
        resized = pane._view
        assert (
            resized.x + resized.width / 2,
            resized.y + resized.height / 2,
        ) == pytest.approx(center)
        pane._configure_trace()
        assert shell.workspace_tabs.selected_index == 0
        assert shell._active_operation == "trace"
        shell._on_workspace_changed(_event())
        assert shell.left_toggle.visible
        shell._on_close_model(_event())
        assert "No trace yet" in texts(pane.control)
        assert pane.trace_picker.disabled


def test_blocks_own_boundary_tensors_and_drill_down_without_weights(
    tmp_path: Path,
) -> None:
    path, _, _ = write_qwen35(tmp_path / "qwen.gguf", packing=142)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        np.save(tmp_path / "tokens.npy", np.array([[1, 2, 3]], np.int64))
        shell.trace.bind_tensor(
            "tokens", session.trace_tensor_input("tokens", tmp_path / "tokens.npy")
        )
        shell.trace.run(_event())
        pane = shell.traced_model
        graph = session.document.main_graph
        layout = shell.current_slice
        assert layout is not None
        scene = pane.renderer.scene
        assert scene is not None and shell.renderer.scene is not None
        # Copy the actual graph structure, positions, sizes, labels, and wires.
        for node in scene.nodes:
            source = shell.renderer.scene.node(node.id)
            assert (node.x, node.y, node.width, node.height, node.label) == (
                source.x,
                source.y,
                source.width,
                source.height,
                source.label,
            )
        assert {(e.source, e.target) for e in scene.edges} == {
            (e.source, e.target) for e in shell.renderer.scene.edges
        }
        block = next(n for n in scene.nodes if n.label.startswith("block-000"))
        members = layout.members_by_glyph[block.id]
        by_name = {v.name: v.id for v in graph.values}
        inputs, outputs = boundary_values(graph, members)
        assert inputs == (by_name["embed"],)
        assert outputs == (by_name["block-000-ffn"],)
        ports = pane._ports[block.id]
        assert [p.values for p in ports] == [(inputs[0],), (outputs[0],)]
        assert ports[0].y == block.y
        assert ports[1].y == block.y + block.height
        assert len(members) == 2
        pane._on_selected(frozenset({block.id}))
        assert shell.current_root_group == block.id and pane.back.visible
        assert pane.renderer.scene is not None
        assert not any(
            n.kind in {"graph-input", "graph-output"} for n in pane.renderer.scene.nodes
        )
        assert any(
            by_name["block-000-ssm"] in p.values
            for ports in pane._ports.values()
            for p in ports
        )
        pane._navigate_parent()
        # Source scopes can wrap a block. Navigation uses the same hierarchy
        # as Model graph and eventually reaches the model root.
        for _ in range(3):
            if not pane.back.visible:
                break
            pane._navigate_parent()
        assert shell.current_root_group is None and not pane.back.visible
        pane.detail.value = "operator"
        pane._on_detail(_event())
        # A weights-only model has no operator circuit to drill into.
        assert shell.current_detail is session.finest_detail() is DetailLevel.LAYER
        assert shell.current_slice is not None
        assert any("::layer:" in node.id for node in shell.current_slice.scene.nodes)
        pane.search.value = "no-such-block"
        pane._find(_event())
        assert "No matching" in (pane.hint.value or "")
        pane.search.value = ""
        pane.mode.value = "tensors"
        pane._on_mode(_event())
        result = session.trace(shell.trace.active_trace_id or "")
        assert len(find_data(pane.body, "trace-tensor:")) == len(result.records)


def test_edited_model_keeps_captures_but_does_not_invent_old_topology(
    tmp_path: Path,
) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page, session = open_traced_shell(service, tmp_path)
        pane = shell.traced_model
        stale_button = cast(ft.Container, find_data(pane.body, "trace-port:")[0])
        node = session.document.main_graph.nodes[0]
        transaction = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "changed")
        )
        session.commit_edit(transaction)
        pane.refresh()
        assert pane.notice.visible and pane.mode.value == "tensors"
        assert pane.mode.options[0].disabled
        assert find_data(pane.body, "trace-tensor:")
        session.undo_edit()
        pane.mode.value = "graph"
        pane.refresh()
        assert not pane.notice.visible and not pane.mode.options[0].disabled
        shell._on_close_model(_event())
        cast(Any, stale_button.on_click)(_event())
        assert not page.dialogs


def test_pane_only_reads_metadata_until_a_port_is_clicked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nneditor.tracing.store import ActivationStore

    with ApplicationService(prewarm_layouts=False) as service:
        shell, _, _ = open_traced_shell(service, tmp_path)
        monkeypatch.setattr(
            ActivationStore,
            "read",
            lambda *_a, **_k: pytest.fail("diagram read payload"),
        )
        pane = shell.traced_model
        pane.refresh()
        pane.mode.value = "tensors"
        pane.refresh()
        pane.search.value = "output"
        pane.refresh()
        assert find_data(pane.body, "trace-tensor:")


def test_search_and_pagination_bound_the_rendered_model(tmp_path: Path) -> None:
    nodes = [
        helper.make_node(
            "Identity", ["input" if i == 0 else f"v{i - 1}"], [f"v{i}"], name=f"step{i}"
        )
        for i in range(55)
    ]
    graph = helper.make_graph(
        nodes,
        "chain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("v54", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    path = tmp_path / "chain.onnx"
    onnx.save(model, path)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        shell.trace.run(_event())
        pane = shell.traced_model
        pane.mode.value = "tensors"
        pane._on_mode(_event())
        assert pane.pager.visible and pane.previous.disabled and not pane.next.disabled
        assert len(find_data(pane.body, "trace-tensor:")) <= 40
        pane._move(1)
        assert not pane.previous.disabled
        pane.search.value = "v54"
        pane._on_search(_event())
        assert len(find_data(pane.body, "trace-tensor:")) == 1
        assert not pane.pager.visible
        pane.mode.value = "graph"
        pane.refresh()
        pane.search.value = "step54"
        pane._find(_event())
        assert "Match 1" in (pane.hint.value or "")
        assert len(pane.port_layer.controls) < 30
        pane.fit()
        assert not pane.port_layer.controls
        assert "Zoom in" in (pane.hint.value or "")


def test_missing_capture_has_an_explanation_in_all_overlay_tabs(tmp_path: Path) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page, session = open_traced_shell(service, tmp_path)
        graph = session.document.main_graph
        trace_id = shell.trace.active_trace_id or ""
        result = session.trace(trace_id)
        # The first run's identical-key captures are retained by design. Use a
        # different input specification so this run really omits the intermediate.
        spec = session.default_trace_inputs(seed=73)
        limits = TraceLimits(wall_seconds=20, memory_bytes=2**31)
        approval = TraceApproval.approve(
            session.title, session.document.source.content_hash, spec, limits
        )
        limited = session.trace_async(
            TraceRequest(
                spec, limits, approval, value_ids=frozenset({graph.outputs[0]})
            )
        ).result(timeout=25)
        intermediate = next(
            r.value_id
            for r in result.records
            if r.role == "node-output" and r.value_id not in graph.outputs
        )
        shell.activations.open_tensor(limited.id, intermediate)
        overlay = shell.activations.value_overlay
        assert overlay.record is None
        for index in (1, 2):
            overlay.tabs.selected_index = index
            overlay._on_tab(_event())
        assert "not selected for capture" in " ".join(texts(overlay.view_area))
        assert not overlay._jobs
        assert overlay.close()
        assert not page.dialogs[-1].open


def test_overlay_analysis_is_lazy_cached_and_independent_of_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page, session = open_traced_shell(service, tmp_path)
        trace_id = shell.trace.active_trace_id or ""
        value_id = session.document.main_graph.outputs[0]
        stats_method = ModelSession.compute_activation_statistics_async
        views_method = ModelSession.activation_visualizations_async
        calls: list[str] = []
        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append

        def statistics(self: ModelSession, *args: Any, **kwargs: Any) -> Any:
            calls.append("statistics")
            return stats_method(self, *args, **kwargs)

        def views(self: ModelSession, *args: Any, **kwargs: Any) -> Any:
            calls.append("visualization")
            return views_method(self, *args, **kwargs)

        monkeypatch.setattr(
            ModelSession, "compute_activation_statistics_async", statistics
        )
        monkeypatch.setattr(ModelSession, "activation_visualizations_async", views)
        shell.activations.open_tensor(trace_id, value_id)
        overlay = shell.activations.value_overlay
        assert not calls
        assert find_data(overlay.view_area, "tensor-value-table")
        overlay.column.value = "invalid"
        overlay._apply_slice()
        assert not find_data(overlay.view_area, "tensor-value-table")
        overlay.column.value = "0"
        overlay._move(0, 3)
        assert overlay.column.value == "3"
        assert find_data(overlay.view_area, "tensor-value-table")
        overlay.tabs.selected_index = 1
        overlay._on_tab(_event())
        overlay._on_tab(_event())
        assert calls == ["statistics"]
        callbacks.pop(0)()
        assert (trace_id, value_id) in shell.activations.activation_statistics
        overlay.tabs.selected_index = 2
        overlay._on_tab(_event())
        callbacks.pop(0)()
        assert "Complete capture" in " ".join(texts(overlay.visualization_area))
        assert calls == ["statistics", "visualization"]
        shell.activations.close_overlay()
        shell.activations.open_tensor(trace_id, value_id)
        for index in (1, 2):
            overlay.tabs.selected_index = index
            overlay._on_tab(_event())
        assert calls == ["statistics", "visualization"] and not callbacks
        source_view = shell.renderer.viewport
        press_key(shell, "+")
        assert shell.renderer.viewport == source_view
        press_key(shell, "Escape")
        assert overlay.dialog is None
        old_scale = shell.traced_model._view.scale
        press_key(shell, "+")
        assert shell.traced_model._view.scale > old_scale
        assert shell.renderer.viewport == source_view


def test_dismissed_overlay_cancels_jobs_and_ignores_late_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page, session = open_traced_shell(service, tmp_path)
        callbacks: list[Any] = []
        cancelled: list[bool] = []
        fake = SimpleNamespace(
            state=SimpleNamespace(is_terminal=False, value="running"),
            cancel=lambda: cancelled.append(True),
            wait=lambda: True,
            release_result=lambda: None,
        )
        monkeypatch.setattr(
            ModelSession, "compute_activation_statistics_async", lambda *_: fake
        )
        cast(Any, page).run_thread = callbacks.append
        graph = session.document.main_graph
        shell.activations.open_tensor(
            shell.trace.active_trace_id or "", graph.outputs[0]
        )
        overlay = shell.activations.value_overlay
        overlay.tabs.selected_index = 1
        overlay._on_tab(_event())
        assert overlay._jobs and callbacks
        shell.activations.open_tensor(
            shell.trace.active_trace_id or "", graph.inputs[0]
        )
        assert cancelled == [True]
        callbacks.pop(0)()
        assert overlay.value_id == graph.inputs[0] and not overlay._jobs
        assert not shell.activations.activation_statistics
        overlay._on_dismiss(cast(Any, SimpleNamespace(control=overlay.dialog)))
        assert overlay.dialog is None


def test_partial_and_evicted_tensors_disclose_available_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _, session = open_traced_shell(service, tmp_path)
        spec = session.default_trace_inputs(seed=87)
        limits = TraceLimits(
            wall_seconds=20, memory_bytes=2**30, capture_bytes=16, chunk_bytes=4
        )
        result = session.trace_async(
            TraceRequest(
                spec,
                limits,
                TraceApproval.approve(
                    session.title, session.document.source.content_hash, spec, limits
                ),
            )
        ).result(timeout=25)
        record = next(r for r in result.records if r.state is CaptureState.TRUNCATED)
        shell.activations.open_tensor(result.id, record.value_id)
        overlay = shell.activations.value_overlay
        assert overlay.dialog is not None
        assert "Preview" in " ".join(texts(overlay.dialog))
        table = cast(
            ft.DataTable, find_data(overlay.view_area, "tensor-value-table")[0]
        )
        assert any(
            cast(ft.Text, cell.content).value == "—"
            for row in table.rows
            for cell in row.cells
        )
        overlay.tabs.selected_index = 1
        overlay._on_tab(_event())
        assert "captured prefix" in " ".join(texts(overlay.statistics_area))
        evicted = replace(
            record,
            state=CaptureState.EVICTED,
            stored_byte_length=0,
            file_name=None,
            reason="Capture evicted from storage.",
        )
        replaced_result = replace(result, records=(evicted,))
        monkeypatch.setattr(ModelSession, "trace", lambda *_: replaced_result)
        shell.activations.open_tensor(result.id, record.value_id)
        for index in (1, 2):
            overlay.tabs.selected_index = index
            overlay._on_tab(_event())
        for area in (
            overlay.view_area,
            overlay.statistics_area,
            overlay.visualization_area,
        ):
            assert "evicted" in " ".join(texts(area))
        assert not overlay._jobs


def test_many_port_values_keep_connections_and_remain_individually_inspectable(
    tmp_path: Path,
) -> None:
    outputs = [f"branch{i}" for i in range(10)]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Split", ["input"], outputs, num_outputs=10),
                helper.make_node("Concat", outputs, ["output"], axis=0),
            ],
            "fan",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [20])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [20])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    path = tmp_path / "fan.onnx"
    onnx.save(model, path)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        producer, consumer = graph.nodes
        nodes = [
            NodeGlyph(n.id, 0, index * 140, 56, 48, "group", n.source_name or n.id)
            for index, n in enumerate(graph.nodes)
        ]
        edge = EdgeGlyph("edge", producer.id, consumer.id, ((28, 48), (28, 140)))
        presentation = TraceGraphPresentation(
            Scene(nodes, [edge]), {edge.id: producer.outputs}, {}, {}
        )
        scene, ports = graph_ports(
            presentation, graph, {n.id: frozenset({n.id}) for n in graph.nodes}
        )
        assert scene.nodes == tuple(nodes)
        assert len(scene.edges) == 10
        assert len(ports[producer.id]) == 3  # one input, two output ellipses
        grouped = ports[producer.id][-1]
        assert grouped.values == producer.outputs[1:]
        assert len(ports[consumer.id][1].values) == 9
        assert scene.edges[0].points[0] != scene.edges[1].points[0]
        assert all(e.points[0] == (grouped.x, grouped.y) for e in scene.edges[1:])
        shell, page = make_shell(service)
        shell.show_session(session)
        shell.trace.run(_event())
        pane = shell.traced_model
        trace_id = shell.trace.active_trace_id or ""
        pane._inspect(grouped.values, "9 output tensors", session, trace_id)
        assert pane._chooser is page.dialogs[-1]
        assert pane._chooser is not None
        content = cast(ft.Container, pane._chooser.content)
        items = cast(ft.ListView, content.content)
        assert len(items.controls) == 9
        button = cast(ft.TextButton, items.controls[-1])
        cast(Any, button.on_click)(_event())
        assert pane._chooser is None
        assert shell.activations.value_overlay.value_id == grouped.values[-1]
