"""Headless Flet wiring for Phase 9 states and consent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import flet.canvas as cv
import numpy as np
import pytest

from nneditor.application.session import ApplicationService
from nneditor.tracing import (
    ActivationRecord,
    CaptureState,
    TraceApproval,
    TraceLimits,
    TraceRequest,
    build_activation_visualizations,
)
from nneditor.ui.app import Shell
from tests.fixtures.onnx_models import build_embedded_model, build_masked_image_model
from tests.unit.test_shell import make_shell


def test_trace_shell_empty_consent_loading_and_activation_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        assert shell.run_trace_button.disabled
        assert "Open an artifact" in str(shell.trace_status.value)

        session = service.open_model(path)
        shell.show_session(session)
        assert not shell.run_trace_button.disabled
        assert "isolated" in str(shell.trace_status.value)
        assert shell.run_trace_button.content == "Approve & run trace"
        assert "authorizes one isolated trace" in str(shell.trace_approval_notice.value)

        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        callbacks: list[Any] = []
        page.run_thread = cast(Any, callbacks.append)
        shell._on_run_trace(event)
        assert "Loading trace" in str(shell.trace_status.value)
        assert not shell.error_banner.visible
        assert callbacks
        callbacks.pop()()
        assert shell.active_trace_id is not None
        assert "Complete trace" in str(shell.trace_status.value)

        node_id = session.document.main_graph.nodes[0].id
        shell.current_detail = shell.current_detail.OPERATOR
        shell._show_graph(session.document.entry_graph)
        shell._refresh_inspector(frozenset({node_id}))
        cards = [
            control
            for control in shell.inspector.controls
            if isinstance(control, ft.ExpansionTile)
            and isinstance(control.data, str)
            and control.data.startswith("activation-card:")
        ]
        assert cards
        assert any(
            "activation captured" in node.type_label
            for node in shell.renderer.scene.nodes
        )


def test_selecting_traced_node_builds_views_and_opens_large_overlay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append

        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        shell._on_run_trace(event)
        callbacks.pop(0)()
        assert shell.active_trace_id is not None

        node_id = session.document.main_graph.nodes[0].id
        shell.current_detail = shell.current_detail.OPERATOR
        shell._show_graph(session.document.entry_graph)
        shell.renderer.set_selection(frozenset({node_id}))
        shell._on_selected(frozenset({node_id}))

        assert callbacks
        assert any(
            isinstance(control, ft.ExpansionTile)
            and any(
                isinstance(child, ft.Text)
                and child.value == "Loading activation views\u2026"
                for child in (
                    cast(ft.Column, control.controls[0]).controls
                    if control.controls
                    else []
                )
            )
            for control in shell.inspector.controls
        )

        while callbacks:
            callbacks.pop(0)()

        cards = [
            control
            for control in shell.inspector.controls
            if isinstance(control, ft.ExpansionTile)
            and isinstance(control.data, str)
            and control.data.startswith("activation-card:")
        ]
        open_button = next(
            child
            for card in cards
            for body in card.controls or []
            for child in cast(ft.Column, body).controls
            if isinstance(child, ft.FilledButton) and child.content == "Open large view"
        )
        cast(Any, open_button.on_click)(event)

        assert page.dialogs
        dialog = page.dialogs[-1]
        assert isinstance(dialog.data, str)
        assert dialog.data.startswith("activation-overlay:")
        assert dialog.open
        content = cast(ft.Container, dialog.content)
        assert content.width == 860
        assert content.height == 620
        rows = cast(ft.ListView, content.content).controls
        plot = next(
            control
            for control in rows
            if isinstance(control, ft.Container)
            and isinstance(control.data, str)
            and control.data.startswith("activation-plot:")
        )
        canvas = cast(cv.Canvas, plot.content)
        assert canvas.width == 780
        assert canvas.height == 420

        close_button = cast(ft.TextButton, dialog.actions[0])
        cast(Any, close_button.on_click)(event)
        assert not dialog.open


def test_graph_first_tensor_picker_and_click_to_inspect_every_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    tensor_path = tmp_path / "chosen-input.npy"
    build_embedded_model(path, elements=8)
    expected = np.arange(8, dtype=np.float32)
    np.save(tensor_path, expected)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        presentation = shell._trace_graph_presentation
        assert presentation is not None
        graph = session.document.main_graph
        input_id = next(
            value_id for value_id in graph.inputs if value_id not in graph.initializers
        )
        input_glyph = presentation.input_glyphs[input_id]
        assert any(
            control.data == f"trace-input-action:{input_id}"
            for control in shell.graph_trace_actions.controls
        )

        async def pick_tensor(
            picker: ft.FilePicker,
            **kwargs: object,
        ) -> list[SimpleNamespace]:
            assert kwargs["allowed_extensions"] == ["npy"]
            return [SimpleNamespace(path=str(tensor_path))]

        monkeypatch.setattr(ft.FilePicker, "pick_files", pick_tensor)
        event = cast(Any, SimpleNamespace())
        asyncio.run(shell._choose_trace_tensor_handler(input_id)(event))
        assert shell._trace_input_bindings["input"].tensor_file == str(
            tensor_path.resolve()
        )
        shell._reset_trace_tensor_handler("input", input_glyph)(event)
        assert "input" not in shell._trace_input_bindings
        asyncio.run(shell._choose_trace_tensor_handler(input_id)(event))

        shell.renderer.set_selection(frozenset({input_glyph}))
        shell._on_selected(frozenset({input_glyph}))
        assert any(
            control.data == f"trace-value-metadata:{input_id}"
            for control in shell.inspector.controls
        )
        assert any(
            isinstance(control, ft.Row)
            and any(
                child.data == f"trace-input-picker:{input_id}"
                for child in control.controls
            )
            for control in shell.inspector.controls
        )

        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append
        shell._on_run_trace(cast(ft.Event[ft.Button], event))
        callbacks.pop()()
        assert shell.active_trace_id is not None
        captured = session.activations(shell.active_trace_id).read(input_id)
        np.testing.assert_array_equal(
            np.frombuffer(captured, dtype=np.float32),
            expected,
        )
        block_rows = shell._group_activation_rows(
            frozenset(node.id for node in graph.nodes),
            owner_id="whole-model",
        )
        assert any(
            isinstance(control, ft.ExpansionTile)
            and control.data == f"activation-card:{graph.outputs[0]}"
            for control in block_rows
        )

        presentation = shell._trace_graph_presentation
        assert presentation is not None
        connection, connection_value = next(
            (glyph_id, value_ids[0])
            for glyph_id, value_ids in presentation.values_by_glyph.items()
            if shell.renderer.scene is not None
            and shell.renderer.scene.has_edge(glyph_id)
            and value_ids
        )
        shell.renderer.set_selection(frozenset({connection}))
        shell._on_selected(frozenset({connection}))
        assert "Activation on this connection" in str(shell.inspector_subtitle.value)
        assert any(
            isinstance(control, ft.ExpansionTile)
            and control.data == f"activation-card:{connection_value}"
            for control in shell.inspector.controls
        )

        output_id = graph.outputs[0]
        output_glyph = presentation.output_glyphs[output_id]
        shell.renderer.set_selection(frozenset({output_glyph}))
        shell._on_selected(frozenset({output_glyph}))
        assert "Model output" in str(shell.inspector_subtitle.value)
        assert any(
            isinstance(control, ft.ExpansionTile)
            and control.data == f"activation-card:{output_id}"
            for control in shell.inspector.controls
        )


def test_required_mask_is_automatic_after_selecting_an_image(tmp_path: Path) -> None:
    model = tmp_path / "masked-image.onnx"
    image = tmp_path / "image.npy"
    build_masked_image_model(model)
    np.save(image, np.zeros((1, 3, 224, 224), dtype=np.float32))

    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(model)
        shell.show_session(session)
        shell._trace_input_bindings["pixel_values"] = session.trace_tensor_input(
            "pixel_values", image
        )
        shell._refresh_graph_trace_actions()

        graph = session.document.main_graph
        mask_id = next(
            value_id
            for value_id in graph.inputs
            if (graph.value(value_id).name or value_id) == "pixel_mask"
        )
        mask_action = next(
            control
            for control in shell.graph_trace_actions.controls
            if control.data == f"trace-input-action:{mask_id}"
        )
        assert isinstance(mask_action, ft.Container)
        mask_button = mask_action.content
        assert isinstance(mask_button, ft.IconButton)
        assert "all-valid mask automatically" in str(mask_button.tooltip)

        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append
        shell._on_run_trace(cast(Any, SimpleNamespace()))
        assert callbacks
        assert not shell.error_banner.visible
        callbacks.pop()()
        assert shell.active_trace_id is not None


def test_trace_shell_invalid_limits_and_web_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        shell.show_session(service.open_model(path))
        shell.trace_capture_mib.value = "0"
        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        shell._on_run_trace(event)
        assert "configuration is invalid" in str(
            cast(ft.Text, shell.error_banner.content).value
        )

        page.web = True  # type: ignore[attr-defined]
        shell._refresh_trace_actions()
        assert shell.run_trace_button.disabled
        assert "Phase 8" in str(shell.trace_status.value)


def test_trace_shell_discloses_partial_capture(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        specification = session.default_trace_inputs()
        limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=1024 * 1024 * 1024,
            capture_bytes=16,
            chunk_bytes=4,
        )
        result = session.trace_async(
            TraceRequest(
                specification,
                limits,
                TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    limits,
                ),
            )
        ).result(timeout=20)
        assert result.partial
        shell.active_trace_id = result.id
        shell._refresh_trace_actions()
        assert "Partial trace" in str(shell.trace_status.value)


def test_activation_plot_adapter_consumes_headless_view_models() -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    matrix_record = ActivationRecord(
        "matrix",
        "matrix",
        "node",
        "node-output",
        "float32",
        "float32",
        matrix.shape,
        CaptureState.COMPLETE,
        matrix.nbytes,
        matrix.nbytes,
        "captures/matrix.bin",
    )
    vector = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float32)
    vector_record = ActivationRecord(
        "vector",
        "vector",
        "node",
        "node-output",
        "float32",
        "float32",
        vector.shape,
        CaptureState.COMPLETE,
        vector.nbytes,
        vector.nbytes,
        "captures/vector.bin",
    )
    views = (
        *build_activation_visualizations(matrix_record, matrix.tobytes()),
        *build_activation_visualizations(vector_record, vector.tobytes()),
    )

    controls = [
        cast(ft.Container, Shell._activation_plot_control(view)) for view in views
    ]

    assert {control.data for control in controls} == {
        "activation-plot:heatmap",
        "activation-plot:histogram",
        "activation-plot:line",
    }
    assert all(
        isinstance(control.content, cv.Canvas) and control.content.shapes
        for control in controls
    )
    heatmap = next(view for view in views if view.kind.value == "heatmap")
    large_heatmap = cast(
        ft.Container,
        Shell._activation_plot_control(heatmap, width=780, height=420),
    )
    large_canvas = cast(cv.Canvas, large_heatmap.content)
    assert large_canvas.width == 420
    assert large_canvas.height == 420
