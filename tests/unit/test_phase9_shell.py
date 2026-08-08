"""Headless Flet wiring for Phase 9 states and consent."""

from __future__ import annotations

import asyncio
import dataclasses
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import flet.canvas as cv
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.desktop.windows_associations import FileAssociationError
from nneditor.input_generation import InputGenerationError
from nneditor.tokenization import TOKENIZER_CHOICES, WordHashCodebook
from nneditor.tracing import (
    ActivationRecord,
    CaptureState,
    TraceApproval,
    TraceBackend,
    TraceDevice,
    TraceLimits,
    TraceRequest,
    build_activation_visualizations,
)
from nneditor.tracing.preflight import RuntimeStatus
from nneditor.tracing.runner import estimated_capture_bytes
from nneditor.ui import activation_layers, viewmodel
from nneditor.ui.activation_inspector import (
    ActivationInspector,
    _visualization_cost,
    build_activation_plot,
)
from nneditor.ui.app import SHELL_PALETTE, Shell
from nneditor.ui.input_workspace import InputTarget
from tests.fixtures.onnx_models import (
    build_embedded_model,
    build_masked_image_model,
    build_optional_output_model,
    build_token_ids_model,
)
from tests.unit.test_shell import StubPage, make_shell, press_key


def test_trace_shell_empty_consent_loading_and_activation_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        assert shell.trace.run_button.disabled
        assert "Open an artifact" in str(shell.trace.status.value)
        assert shell.device_text.value == "Device: idle"

        session = service.open_model(path)
        shell.show_session(session)
        assert not shell.trace.run_button.disabled
        assert "isolated" in str(shell.trace.status.value)
        assert shell.trace.run_button.content == "Approve & run trace"
        assert "authorizes one isolated trace" in str(shell.trace.approval_notice.value)

        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        callbacks: list[Any] = []
        page.run_thread = cast(Any, callbacks.append)
        shell.trace.run(event)
        assert "Preparing approved trace" in str(shell.trace.status.value)
        assert shell.trace.progress.visible
        assert shell.trace.run_button.disabled
        assert not shell.error_banner.visible
        assert callbacks
        callbacks.pop(0)()
        assert callbacks
        callbacks.pop(0)()
        assert shell.trace.active_trace_id is not None
        assert "Complete trace" in str(shell.trace.status.value)
        assert not shell.trace.progress.visible
        assert shell.device_text.value == "CPU / ONNX Runtime"
        assert "CPUExecutionProvider" in str(shell.device_indicator.tooltip)

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
        marked = [
            node.type_label
            for node in shell.renderer.scene.nodes
            if "activation captured" in node.type_label
        ]
        assert marked
        # The capture status is a suffix after the glyph's own identity, so
        # a renderer that truncates long labels drops the status, not the
        # name — no glyph ever reads as a bare "• status" string.
        assert all(label.split("•")[0].strip() for label in marked)


def test_open_model_applies_model_aware_trace_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    suggested = TraceLimits(
        wall_seconds=300,
        memory_bytes=16384 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "nneditor.ui.app.recommended_trace_limits",
        lambda _model_bytes: suggested,
    )

    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(path))

        assert shell.trace.wall_seconds.value == "300"
        assert shell.trace.memory_mib.value == "16384"


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
        shell.trace.run(event)
        callbacks.pop(0)()
        callbacks.pop(0)()
        assert shell.trace.active_trace_id is not None

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
        presentation = shell.trace.presentation
        assert presentation is not None
        graph = session.document.main_graph
        input_id = next(
            value_id for value_id in graph.inputs if value_id not in graph.initializers
        )
        input_glyph = presentation.input_glyphs[input_id]
        assert any(
            control.data == f"trace-input-action:{input_id}"
            for control in shell.trace.graph_actions.controls
        )

        async def pick_tensor(
            picker: ft.FilePicker,
            **kwargs: object,
        ) -> list[SimpleNamespace]:
            assert kwargs["allowed_extensions"] == ["npy"]
            return [SimpleNamespace(path=str(tensor_path))]

        monkeypatch.setattr(ft.FilePicker, "pick_files", pick_tensor)
        event = cast(Any, SimpleNamespace())
        asyncio.run(shell.trace.choose_tensor_handler(input_id)(event))
        assert shell.trace.input_bindings["input"].tensor_file == str(
            tensor_path.resolve()
        )
        shell.trace.reset_tensor_handler("input", input_glyph)(event)
        assert "input" not in shell.trace.input_bindings
        asyncio.run(shell.trace.choose_tensor_handler(input_id)(event))

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
        shell.trace.run(cast(ft.Event[ft.Button], event))
        callbacks.pop(0)()
        callbacks.pop(0)()
        assert shell.trace.active_trace_id is not None
        captured = session.activations(shell.trace.active_trace_id).read(input_id)
        np.testing.assert_array_equal(
            np.frombuffer(captured, dtype=np.float32),
            expected,
        )
        block_rows = shell.activations.group_activation_rows(
            frozenset(node.id for node in graph.nodes),
            owner_id="whole-model",
        )
        assert any(
            isinstance(control, ft.ExpansionTile)
            and control.data == f"activation-card:{graph.outputs[0]}"
            for control in block_rows
        )

        presentation = shell.trace.presentation
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


def test_group_selection_shows_only_boundary_activations(tmp_path: Path) -> None:
    """A selected block lists boundary captures only, inputs before outputs.

    The model declares its initializers as graph inputs too (a common
    exporter layout), so the trace captures the weight values; neither
    those parameters nor the value flowing between the block's members may
    surface under "Block inputs & outputs".
    """
    path = tmp_path / "legacy.onnx"
    gain = helper.make_tensor(
        "gain",
        TensorProto.FLOAT,
        [4],
        np.full(4, 2.0, dtype=np.float32).tobytes(),
        raw=True,
    )
    shift = helper.make_tensor(
        "shift",
        TensorProto.FLOAT,
        [4],
        np.full(4, 0.5, dtype=np.float32).tobytes(),
        raw=True,
    )
    graph_proto = helper.make_graph(
        nodes=[
            helper.make_node("Mul", ["input", "gain"], ["hidden"], name="entry"),
            helper.make_node("Mul", ["hidden", "gain"], ["folded"], name="inner"),
            helper.make_node("Add", ["folded", "shift"], ["output"], name="exit"),
        ],
        name="legacy",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("gain", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("shift", TensorProto.FLOAT, [4]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        initializer=[gain, shift],
    )
    onnx.save_model(
        helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )

    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)

        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append
        shell.trace.run(event)
        while callbacks:
            callbacks.pop(0)()
        cast(Any, page).run_thread = lambda target: target()
        assert shell.trace.active_trace_id is not None

        graph = session.document.main_graph

        def value_id(name: str) -> str:
            return next(value.id for value in graph.values if value.name == name)

        # The defect needs the internals on record: the trace must have
        # captured the weight values and the between-members value, so their
        # absence below proves filtering, not a narrow capture.
        captured = {
            record.value_id
            for record in session.trace(shell.trace.active_trace_id).records
        }
        assert {value_id("gain"), value_id("shift"), value_id("folded")} <= captured

        members = frozenset(
            node.id for node in graph.nodes if node.source_name in {"inner", "exit"}
        )
        shell.renderer.set_selection(members)
        shell._on_selected(members)
        shell.group_label_field.value = "Boundary block"
        shell._on_group_selected(cast(Any, None))
        manual = next(
            group for group in session.graph_hierarchy().groups if not group.automatic
        )
        assert manual.members == members

        shell.renderer.set_selection(frozenset({manual.id}))
        shell._on_selected(frozenset({manual.id}))
        markers: list[tuple[str, str]] = []
        for control in shell.inspector.controls:
            if isinstance(control, ft.Row):
                heading = next(
                    (
                        child.value
                        for child in control.controls
                        if isinstance(child, ft.Text)
                    ),
                    None,
                )
                if isinstance(heading, str):
                    markers.append(("heading", heading))
            elif isinstance(control, ft.ExpansionTile) and isinstance(
                control.data, str
            ):
                markers.append(("card", control.data))

        hidden_card = f"activation-card:{value_id('hidden')}"
        output_card = f"activation-card:{value_id('output')}"
        cards = [data for kind, data in markers if kind == "card"]
        assert cards == [hidden_card, output_card]
        assert (
            markers.index(("heading", "Block inputs & outputs"))
            < markers.index(("heading", "Inputs"))
            < markers.index(("card", hidden_card))
            < markers.index(("heading", "Outputs"))
            < markers.index(("card", output_card))
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
        shell.trace.input_bindings["pixel_values"] = session.trace_tensor_input(
            "pixel_values", image
        )
        shell.trace.refresh_graph_actions()

        graph = session.document.main_graph
        mask_id = next(
            value_id
            for value_id in graph.inputs
            if (graph.value(value_id).name or value_id) == "pixel_mask"
        )
        mask_action = next(
            control
            for control in shell.trace.graph_actions.controls
            if control.data == f"trace-input-action:{mask_id}"
        )
        assert isinstance(mask_action, ft.Container)
        mask_button = mask_action.content
        assert isinstance(mask_button, ft.IconButton)
        assert "all-valid mask automatically" in str(mask_button.tooltip)

        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append
        shell.trace.run(cast(Any, SimpleNamespace()))
        assert callbacks
        assert not shell.error_banner.visible
        callbacks.pop(0)()
        callbacks.pop(0)()
        assert shell.trace.active_trace_id is not None


def test_generator_tab_saves_and_assigns_to_selected_graph_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    destination = tmp_path / "generated.npy"
    build_embedded_model(model, elements=8)

    async def save_file(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        assert kwargs["allowed_extensions"] == ["npy"]
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        session = service.open_model(model)
        shell.show_session(session)
        generator = shell.input_generator

        assert shell.workspace_tabs.length == 2
        assert generator.target_input.value == "input"
        assert not generator.generate_and_assign_button.disabled
        shell.workspace_tabs.selected_index = 1
        generator.kind.value = "tensor"
        generator._on_kind_changed(None)
        generator.tensor_shape.value = "8"
        generator.tensor_distribution.value = "ramp"
        generator.tensor_dtype.value = "float32"

        asyncio.run(generator._on_generate_and_assign(cast(Any, SimpleNamespace())))

        assert destination.is_file()
        np.testing.assert_array_equal(
            np.load(destination, allow_pickle=False),
            np.arange(8, dtype=np.float32),
        )
        binding = shell.trace.input_bindings["input"]
        assert binding.tensor_file == str(destination.resolve())
        assert shell.workspace_tabs.selected_index == 0
        presentation = shell.trace.presentation
        assert presentation is not None
        input_id = session.document.main_graph.inputs[0]
        assert shell.renderer.selection == frozenset(
            {presentation.input_glyphs[input_id]}
        )
        assert "Saved and assigned" in str(generator.status.value)


def test_generator_target_switches_mask_inputs_to_all_valid_mask(
    tmp_path: Path,
) -> None:
    model = tmp_path / "masked-image.onnx"
    build_masked_image_model(model)

    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generator = shell.input_generator
        generator.target_input.value = "pixel_mask"
        generator._on_target_changed(None)

        assert generator.kind.value == "automatic"
        assert generator.mask_fill.value == "ones"
        assert generator.mask_dtype.value == "int64"
        assert generator.automatic_section.visible
        assert "all-valid values" in str(generator.automatic_summary.value)

        generator.kind.value = "mask"
        generator._on_kind_changed(None)
        assert generator.mask_section.visible
        assert not generator.image_section.visible


def test_generator_automatic_source_uses_inferred_shape_without_image_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "masked-image.onnx"
    destination = tmp_path / "pixel_values.npy"
    build_masked_image_model(model)

    async def save_file(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        assert kwargs["file_name"] == "pixel_values.npy"
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generator = shell.input_generator

        assert generator.kind.value == "automatic"
        assert str(generator.kind.options[0].text) == "Automatic from input"
        assert generator.image_path.value in {None, ""}
        asyncio.run(generator._on_generate(cast(Any, SimpleNamespace())))

        generated = np.load(destination, allow_pickle=False)
        assert generated.shape == (1, 3, 224, 224)
        assert generated.dtype == np.dtype("float32")
        assert generator.status.value == "Saved pixel_values.npy"
        assert not shell.error_banner.visible


def test_generator_detects_and_preconfigures_every_model_input(tmp_path: Path) -> None:
    model = tmp_path / "masked-image.onnx"
    build_masked_image_model(model)

    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generator = shell.input_generator

        assert tuple(generator.input_presets) == ("pixel_values", "pixel_mask")
        pixels = generator.input_presets["pixel_values"]
        assert pixels.kind == "image"
        assert pixels.shape == (1, 3, 224, 224)
        assert pixels.dtype == "float32"
        assert pixels.assumptions == ("symbolic or unknown extents default to 1",)
        mask = generator.input_presets["pixel_mask"]
        assert mask.kind == "mask"
        assert mask.shape == (1, 64, 64)
        assert mask.dtype == "int64"
        assert mask.distribution == "ones"
        assert [
            control.data
            for control in generator.detected_inputs.controls
            if isinstance(control, ft.TextButton)
        ] == ["input-preset:pixel_values", "input-preset:pixel_mask"]
        assert generator.target_input.value == "pixel_values"
        assert generator.kind.value == "automatic"
        assert generator.automatic_section.visible
        assert "shape (1, 3, 224, 224)" in str(generator.automatic_summary.value)
        assert generator.image_layout.value == "NCHW"
        assert generator.image_height.value == "224"
        assert generator.image_width.value == "224"
        assert not generator.generate_all_and_assign_button.disabled

        mask_row = generator.detected_inputs.controls[2]
        assert isinstance(mask_row, ft.TextButton)
        cast(Any, mask_row.on_click)(cast(Any, SimpleNamespace()))
        assert generator.target_input.value == "pixel_mask"
        assert generator.kind.value == "automatic"
        assert generator.mask_shape.value == "1,64,64"


def test_generator_generates_and_assigns_complete_detected_input_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "masked-image.onnx"
    output_directory = tmp_path / "inputs"
    output_directory.mkdir()
    build_masked_image_model(model)

    async def get_directory_path(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        assert kwargs["dialog_title"] == (
            "Choose a directory for generated model inputs"
        )
        return str(output_directory)

    monkeypatch.setattr(ft.FilePicker, "get_directory_path", get_directory_path)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generator = shell.input_generator

        asyncio.run(generator.generate_all_and_assign())

        pixel_path = output_directory / "01-pixel_values.npy"
        mask_path = output_directory / "02-pixel_mask.npy"
        pixels = np.load(pixel_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)
        assert pixels.shape == (1, 3, 224, 224)
        assert pixels.dtype == np.dtype("float32")
        assert mask.shape == (1, 64, 64)
        assert mask.dtype == np.dtype("int64")
        np.testing.assert_array_equal(mask, np.ones(mask.shape, dtype=np.int64))
        assert set(shell.trace.input_bindings) == {"pixel_values", "pixel_mask"}
        assert shell.trace.input_bindings["pixel_values"].tensor_file == str(
            pixel_path.resolve()
        )
        assert shell.trace.input_bindings["pixel_mask"].tensor_file == str(
            mask_path.resolve()
        )
        assert generator.status.value == "Generated and assigned 2 model inputs"
        assert "pixel_values" in str(generator.summary.value)
        assert "pixel_mask" in str(generator.summary.value)


def test_generator_complete_set_never_overwrites_existing_input(
    tmp_path: Path,
) -> None:
    model = tmp_path / "masked-image.onnx"
    output_directory = tmp_path / "inputs"
    output_directory.mkdir()
    existing = output_directory / "01-pixel_values.npy"
    existing.write_bytes(b"user data")
    build_masked_image_model(model)

    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))

        with pytest.raises(InputGenerationError, match="never overwrites"):
            shell.input_generator.generate_configured_inputs(output_directory)

        assert existing.read_bytes() == b"user data"
        assert not (output_directory / "02-pixel_mask.npy").exists()


def test_generator_validates_complete_set_before_binding_any_input(
    tmp_path: Path,
) -> None:
    model = tmp_path / "masked-image.onnx"
    output_directory = tmp_path / "inputs"
    output_directory.mkdir()
    build_masked_image_model(model)

    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generated = shell.input_generator.generate_configured_inputs(output_directory)
        mask_path = generated[1][1].path
        np.save(mask_path, np.ones((1, 64, 64), dtype=np.float32), allow_pickle=False)

        with pytest.raises(ValueError, match="dtype float32"):
            shell._assign_generated_inputs(generated)

        assert shell.trace.input_bindings == {}


def test_generator_target_switches_qwen3vl_pixels_to_patch_profile() -> None:
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        generator = shell.input_generator
        generator.refresh_targets(
            [
                InputTarget(
                    name="pixel_values",
                    element_type="float32",
                    shape=(880, 1536),
                )
            ],
            can_assign=True,
        )

        assert generator.kind.value == "automatic"
        assert generator.image_layout.value == "QWEN3VL_PATCHES"
        assert generator.image_width.value == "640"
        assert generator.image_height.value == "352"
        assert generator.image_normalization.value == "minus-one-one"
        assert generator.automatic_section.visible


@pytest.mark.parametrize(
    ("shape", "layout", "height", "width", "color"),
    [
        ((1, 3, 320, 640), "NCHW", "320", "640", "rgb"),
        ((1, 320, 640, 3), "NHWC", "320", "640", "rgb"),
        ((3, 320, 640), "CHW", "320", "640", "rgb"),
        ((1, 320, 640), "CHW", "320", "640", "grayscale"),
        ((320, 640, 3), "HWC", "320", "640", "rgb"),
        ((320, 640), "HW", "320", "640", "grayscale"),
    ],
)
def test_generator_maps_tensor_height_and_width_without_transposing(
    shape: tuple[int, ...],
    layout: str,
    height: str,
    width: str,
    color: str,
) -> None:
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        generator = shell.input_generator
        generator.refresh_targets(
            [
                InputTarget(
                    name="image",
                    element_type="float32",
                    shape=shape,
                )
            ],
            can_assign=True,
        )

        assert generator.kind.value == "automatic"
        assert generator.image_layout.value == layout
        assert generator.image_height.value == height
        assert generator.image_width.value == width
        assert generator.image_color.value == color
        assert generator.image_height.label == "Height (H)"
        assert generator.image_width.label == "Width (W)"


def test_generator_form_builds_image_mask_csv_and_time_series(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8), mode="RGB").save(image)
    csv = tmp_path / "series.csv"
    csv.write_text("1,2\n3,4\n", encoding="utf-8")

    with ApplicationService() as service:
        shell, page = make_shell(service)
        generator = shell.input_generator
        event = cast(Any, SimpleNamespace())

        generator.kind.value = "image"
        generator.image_path.value = str(image)
        generator.image_width.value = "3"
        generator.image_height.value = "2"
        generated = generator.generate_current_input(tmp_path / "image.npy")
        assert generated.shape == (1, 3, 2, 3)

        generator.kind.value = "mask"
        generator._on_kind_changed(event)
        generator.mask_shape.value = "1,2,3"
        generated = generator.generate_current_input(tmp_path / "mask.npy")
        assert generated.shape == (1, 2, 3)
        assert page.updates

        generator.kind.value = "csv"
        generator._on_kind_changed(None)
        generator.csv_path.value = str(csv)
        generator.csv_columns.value = "1"
        generator.csv_add_batch.value = True
        generated = generator.generate_current_input(tmp_path / "csv.npy")
        assert generated.shape == (1, 2, 1)

        generator.kind.value = "time-series"
        generator._on_kind_changed(None)
        generator.series_samples.value = "8"
        generator.series_channels.value = "2"
        generated = generator.generate_current_input(tmp_path / "series.npy")
        assert generated.shape == (1, 8, 2)


def test_generator_source_pickers_update_the_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    csv = tmp_path / "series.csv"
    vocabulary = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    responses = [
        [SimpleNamespace(path=str(image), name=image.name)],
        [SimpleNamespace(path=str(csv), name=csv.name)],
        [SimpleNamespace(path=str(vocabulary), name=vocabulary.name)],
        [SimpleNamespace(path=str(merges), name=merges.name)],
    ]

    async def pick_files(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> list[SimpleNamespace]:
        assert kwargs["allow_multiple"] is False
        return responses.pop(0)

    monkeypatch.setattr(ft.FilePicker, "pick_files", pick_files)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        generator = shell.input_generator
        event = cast(Any, SimpleNamespace())
        asyncio.run(generator._on_choose_image(event))
        asyncio.run(generator._on_choose_csv(event))
        asyncio.run(generator._on_choose_vocabulary(event))
        asyncio.run(generator._on_choose_merges(event))

        assert generator.image_path.value == str(image)
        assert generator.csv_path.value == str(csv)
        assert generator.token_vocabulary_path.value == str(vocabulary)
        assert generator.token_merges_path.value == str(merges)
        assert generator.status.value == f"Selected {merges.name}"


def test_generator_offers_every_tokenizer_choice_with_its_note_and_files() -> None:
    with ApplicationService() as service:
        shell, page = make_shell(service)
        generator = shell.input_generator

        assert [str(option.key) for option in generator.token_codebook.options] == [
            choice.id for choice in TOKENIZER_CHOICES
        ]
        assert [str(option.text) for option in generator.token_codebook.options] == [
            choice.label for choice in TOKENIZER_CHOICES
        ]

        for choice in TOKENIZER_CHOICES:
            generator.token_codebook.value = choice.id
            generator._on_codebook_changed(cast(Any, SimpleNamespace()))
            assert generator.token_note.value == choice.note
            assert generator.token_vocabulary_row.visible is choice.needs_vocabulary
            assert generator.token_merges_row.visible is choice.needs_merges
            # A file-backed vocabulary dictates the size, so the field is only
            # editable for the self-contained codebooks.
            assert generator.token_vocab_size.disabled is choice.needs_files

        generator.token_codebook.value = "tokenizer-json"
        generator._on_codebook_changed(None)
        assert generator.token_vocabulary_path.label == "Vocabulary (tokenizer.json)"
        assert not generator.token_merges_row.visible
        assert page.updates


def test_generator_text_tokens_save_and_assign_to_a_token_id_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "language-model.onnx"
    destination = tmp_path / "tokens.npy"
    build_token_ids_model(model, length=8)

    async def save_file(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        assert kwargs["file_name"] == "tokens.npy"
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(model))
        generator = shell.input_generator

        # Automatic generation is the default, while the inferred token form
        # remains preconfigured as a manual alternative.
        assert generator.kind.value == "automatic"
        assert generator.automatic_section.visible
        generator.kind.value = "text-tokens"
        generator._on_kind_changed(None)
        assert generator.token_section.visible
        assert not generator.image_section.visible
        assert generator.token_sequence_length.value == "8"
        assert generator.token_dtype.value == "int64"

        generator.token_codebook.value = "word-hash"
        generator._on_codebook_changed(None)
        generator.token_vocab_size.value = "512"
        generator.token_text.value = "hello token world"
        generator.token_bos_id.value = "1"

        asyncio.run(generator._on_generate_and_assign(cast(Any, SimpleNamespace())))

        saved = np.load(destination, allow_pickle=False)
        expected = (1, *WordHashCodebook(vocab_size=512).encode("hello token world"))
        assert saved.dtype == np.dtype("int64")
        assert saved.shape == (1, 8)
        np.testing.assert_array_equal(saved[0, : len(expected)], np.asarray(expected))
        np.testing.assert_array_equal(
            saved[0, len(expected) :],
            np.zeros(8 - len(expected), dtype=np.int64),
        )
        assert shell.trace.input_bindings["input_ids"].tensor_file == str(
            destination.resolve()
        )
        assert "Saved and assigned" in str(generator.status.value)
        # The fidelity disclosure travels with the tensor, undiluted.
        assert "not the model's own ids" in str(generator.summary.value)


def test_generator_text_tokens_refuse_missing_vocabulary_and_merge_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "tokens.npy"
    vocabulary = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"

    async def save_file(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        generator = shell.input_generator
        banner = cast(ft.Text, shell.error_banner.content)
        generator.kind.value = "text-tokens"
        generator._on_kind_changed(None)
        generator.token_codebook.value = "gpt2-bpe"
        generator._on_codebook_changed(None)
        generator.token_text.value = "hello"

        asyncio.run(generator.generate(assign=False))
        assert "choose a vocabulary file" in str(banner.value)
        assert not destination.exists()

        generator.token_vocabulary_path.value = str(vocabulary)
        asyncio.run(generator.generate(assign=False))
        assert "choose a merges file" in str(banner.value)
        assert not destination.exists()

        # Both slots filled, so the loader runs and reports its own refusal.
        vocabulary.write_text("not json", encoding="utf-8")
        merges.write_text("#version: 0.2\n", encoding="utf-8")
        generator.token_merges_path.value = str(merges)
        asyncio.run(generator.generate(assign=False))
        assert "not valid JSON" in str(banner.value)
        assert not destination.exists()


def test_generator_reports_validation_and_assignment_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    destination = tmp_path / "wrong.npy"
    build_embedded_model(model, elements=4)

    async def save_file(
        picker: ft.FilePicker,
        **kwargs: object,
    ) -> str:
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        generator = shell.input_generator
        asyncio.run(generator.generate(assign=True))
        banner = cast(ft.Text, shell.error_banner.content)
        assert "Open a model" in str(banner.value)

        shell.show_session(service.open_model(model))
        generator.kind.value = "tensor"
        generator.tensor_shape.value = "4"
        generator.tensor_dtype.value = "int64"
        asyncio.run(generator.generate(assign=True))

        assert destination.is_file()
        assert "could not assign" in str(banner.value)
        assert "assignment failed" in str(generator.status.value)

        generator.tensor_shape.value = "not-a-shape"
        asyncio.run(generator.generate(assign=False))
        assert "Could not generate" in str(banner.value)


def test_file_type_button_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        settings = 0

        def open_settings() -> None:
            nonlocal settings
            settings += 1

        monkeypatch.setattr(
            "nneditor.ui.app.register_file_associations",
            lambda **kwargs: SimpleNamespace(extensions=(".onnx", ".pt")),
        )
        monkeypatch.setattr(
            "nneditor.ui.app.open_default_apps_settings",
            open_settings,
        )
        shell._on_register_file_types(cast(Any, SimpleNamespace()))

        assert settings == 1
        assert "Registered 2" in str(shell.status_text.value)

        def fail(**kwargs: object) -> None:
            raise FileAssociationError("denied")

        monkeypatch.setattr("nneditor.ui.app.register_file_associations", fail)
        shell._on_register_file_types(cast(Any, SimpleNamespace()))
        banner = cast(ft.Text, shell.error_banner.content)
        assert "denied" in str(banner.value)


def test_trace_shell_invalid_limits_and_web_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        shell.show_session(service.open_model(path))
        shell.trace.capture_mib.value = "0"
        event = cast(ft.Event[ft.Button], cast(Any, SimpleNamespace()))
        shell.trace.run(event)
        assert "configuration is invalid" in str(
            cast(ft.Text, shell.error_banner.content).value
        )
        assert not shell.trace.progress.visible
        assert not shell.trace.run_button.disabled

        page.web = True  # type: ignore[attr-defined]
        shell.trace.refresh_actions()
        assert shell.trace.run_button.disabled
        assert "Phase 8" in str(shell.trace.status.value)


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
        shell.trace.active_trace_id = result.id
        shell.trace.refresh_actions()
        assert "Partial trace" in str(shell.trace.status.value)


def test_trace_shell_reads_truncated_but_readable_capture_as_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page, _callbacks, trace_id, session = _traced_shell(service, path)
        base = session.trace(trace_id)
        assert not base.partial

        # Equal-share preview budgeting: every requested value keeps a
        # readable prefix. Nothing was dropped, so the status must read
        # "Preview trace", not flag the run as partial.
        previewed = dataclasses.replace(
            base,
            records=tuple(
                dataclasses.replace(
                    record,
                    state=CaptureState.TRUNCATED,
                    stored_byte_length=record.full_byte_length // 2,
                    reason="preview share",
                )
                for record in base.records
            ),
        )
        assert previewed.partial
        monkeypatch.setattr(ModelSession, "trace", lambda self, _trace_id: previewed)
        monkeypatch.setattr(ModelSession, "traces", lambda self: (previewed,))
        shell.trace.refresh_actions()

        status = str(shell.trace.status.value)
        assert "Preview trace" in status
        assert "Partial" not in status
        assert "most values were not captured" not in status


def _select_operators(shell: Shell, session: ModelSession, *node_ids: str) -> None:
    """Show the operator graph and select those nodes, as a click would."""
    shell.current_detail = DetailLevel.OPERATOR
    shell._show_graph(session.document.entry_graph)
    selection = frozenset(node_ids)
    shell.renderer.set_selection(selection)
    shell._on_selected(selection)


def _approved_request(
    shell: Shell,
    page: StubPage,
    monkeypatch: pytest.MonkeyPatch,
) -> TraceRequest:
    """Approve one trace and return the request, without starting a worker."""
    requests: list[TraceRequest] = []

    def record(session: ModelSession, request: TraceRequest) -> Any:
        requests.append(request)
        return SimpleNamespace(
            state=SimpleNamespace(is_terminal=False, value="running"),
            cancel=lambda: None,
        )

    monkeypatch.setattr(ModelSession, "trace_async", record)
    callbacks: list[Any] = []
    cast(Any, page).run_thread = callbacks.append
    shell.trace.run(cast(ft.Event[ft.Button], cast(Any, SimpleNamespace())))
    callbacks.pop(0)()
    shell.trace.job = None
    (request,) = requests
    return request


def _tick_capture_scope(shell: Shell, ticked: bool) -> None:
    shell.trace.capture_selected_only.value = ticked
    cast(Any, shell.trace.capture_selected_only.on_change)(cast(Any, SimpleNamespace()))


def _estimate_text(session: ModelSession, value_ids: frozenset[str]) -> str:
    """The human-compact decoded-bytes estimate the panel shows for a scope."""
    graph = session.document.main_graph
    return viewmodel.compact_bytes(
        estimated_capture_bytes(graph.value(value_id) for value_id in value_ids)
    )


def _boundary_ids(session: ModelSession) -> frozenset[str]:
    """Named model inputs and outputs, the boundaries-plus-selection floor."""
    graph = session.document.main_graph
    candidates = {
        value_id for value_id in graph.inputs if value_id not in graph.initializers
    }
    candidates.update(graph.outputs)
    return frozenset(value_id for value_id in candidates if graph.value(value_id).name)


def test_unticked_capture_scope_still_traces_every_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        _select_operators(shell, session, session.document.main_graph.nodes[0].id)

        assert not shell.trace.capture_selected_only.value
        assert shell.trace.capture_value_ids() == frozenset()
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == frozenset()
        assert request.capture_policy == "greedy"


def test_trace_backend_is_visible_and_bound_into_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        shell.trace.backend.value = TraceBackend.REFERENCE_NORMALIZED.value

        request = _approved_request(shell, page, monkeypatch)

        assert request.backend is TraceBackend.REFERENCE_NORMALIZED
        assert request.approval.backend is TraceBackend.REFERENCE_NORMALIZED
        assert request.device is TraceDevice.CPU
        assert request.approval.device is TraceDevice.CPU


def test_trace_gpu_selection_is_bound_into_per_run_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        shell.trace.backend.value = TraceBackend.ONNX_RUNTIME.value
        shell.trace.device.value = TraceDevice.GPU.value

        request = _approved_request(shell, page, monkeypatch)

        assert request.device is TraceDevice.GPU
        assert request.approval.device is TraceDevice.GPU
        assert shell.device_text.value == "GPU / selecting"


def test_ticked_capture_scope_narrows_the_request_to_selected_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        graph = session.document.main_graph
        selected = graph.nodes[0]
        _select_operators(shell, session, selected.id)
        _tick_capture_scope(shell, True)

        expected = frozenset(selected.outputs)
        assert expected
        assert shell.trace.capture_value_ids() == expected
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == expected
        # A selected-nodes capture always spends its budget on whole values.
        assert request.capture_policy == "greedy"

        # An empty selection narrows nothing, so the trace captures everything.
        shell.renderer.set_selection(frozenset())
        shell._on_selected(frozenset())
        assert shell.trace.capture_value_ids() == frozenset()


def test_capture_scope_drops_unnamed_placeholder_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "optional-output.onnx"
    build_optional_output_model(path)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        graph = session.document.main_graph
        dropout = next(node for node in graph.nodes if node.op_type == "Dropout")
        named = frozenset(
            value_id for value_id in dropout.outputs if graph.value(value_id).name
        )
        placeholder = frozenset(dropout.outputs) - named
        # The runner rejects a selected value with no serialized name, so the
        # panel must never put one in the request.
        assert len(placeholder) == 1
        _select_operators(shell, session, dropout.id)
        _tick_capture_scope(shell, True)

        assert shell.trace.capture_value_ids() == named
        assert _approved_request(shell, page, monkeypatch).value_ids == named
        assert "Capturing 1 value from 1 selected node (~" in str(
            shell.trace.capture_scope.value
        )


def test_capture_scope_summary_tracks_the_box_and_the_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        assert "Open an artifact" in str(shell.trace.capture_scope.value)

        session = service.open_model(path)
        shell.show_session(session)
        graph = session.document.main_graph
        all_ids = {
            value_id for value_id in graph.inputs if value_id not in graph.initializers
        }
        for node in graph.nodes:
            all_ids.update(node.outputs)
        all_ids = {value_id for value_id in all_ids if graph.value(value_id).name}
        full = _estimate_text(session, frozenset(all_ids))
        # One model input plus one output per node, none of them unnamed; the
        # summary always states the scope and its estimated decoded bytes.
        # "scaled" declares no shape, so the estimate is an honest lower
        # bound over the values that do rather than a pretend total.
        full_suffix = f"(≥{full}, 2 of 3 values declare shapes)."
        assert shell.trace.capture_scope.value == (
            f"Capturing all 3 values {full_suffix}"
        )

        _tick_capture_scope(shell, True)
        assert shell.trace.capture_scope.value == (
            f"No nodes selected; capturing all 3 values {full_suffix}"
        )

        _select_operators(shell, session, *(node.id for node in graph.nodes))
        selected = frozenset(
            value_id
            for node in graph.nodes
            for value_id in node.outputs
            if graph.value(value_id).name
        )
        assert shell.trace.capture_scope.value == (
            f"Capturing 2 values from 2 selected nodes "
            f"(≥{_estimate_text(session, selected)}, 1 of 2 values declare shapes)."
        )

        _tick_capture_scope(shell, False)
        assert shell.trace.capture_scope.value == (
            f"Capturing all 3 values {full_suffix}"
        )


# -- smart capture-scope default, estimates, and preflight -------------------


def test_big_model_defaults_to_preview_everything_with_per_value_share(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    # The declared input and output are 32 B each, so a 32-byte advisory
    # stands in for the real 1 GiB threshold a multi-GiB capture would trip.
    monkeypatch.setattr("nneditor.ui.trace_panel._FULL_CAPTURE_ADVISORY_BYTES", 32)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace

        assert trace.capture_scope_choice.visible
        assert trace.capture_scope_choice.value == "preview-everything"
        # Preview traces the full value set — nothing is narrowed — and the
        # summary states the mode and the per-value preview share of the
        # panel's 256 MiB capture limit.
        assert trace.capture_value_ids() == frozenset()
        per_value = viewmodel.compact_bytes(256 * 1024 * 1024 // 3)
        summary = str(trace.capture_scope.value)
        assert "Preview everything" in summary
        assert f"~{per_value} per value" in summary
        preview_option = next(
            option
            for option in trace.capture_scope_choice.options
            if option.key == "preview-everything"
        )
        assert "3 values" in str(preview_option.text)
        assert f"~{per_value} each" in str(preview_option.text)

        # The approved request carries the full scope and the preview policy.
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == frozenset()
        assert request.capture_policy == "preview"


def test_boundaries_choice_narrows_the_request_and_stays_greedy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    monkeypatch.setattr("nneditor.ui.trace_panel._FULL_CAPTURE_ADVISORY_BYTES", 32)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace
        assert trace.capture_scope_choice.value == "preview-everything"

        trace.capture_scope_choice.value = "boundaries-plus-selection"
        cast(Any, trace.capture_scope_choice.on_select)(cast(Any, SimpleNamespace()))
        assert "boundaries + selection" in str(trace.capture_scope.value)

        # The current selection joins the boundary scope, the choice
        # survives selection churn, and the approved request carries exactly
        # that narrowed set under whole-value greedy budgeting.
        selected = session.document.main_graph.nodes[0]
        _select_operators(shell, session, selected.id)
        trace.refresh_actions()
        assert trace.capture_scope_choice.value == "boundaries-plus-selection"
        expected = _boundary_ids(session) | frozenset(selected.outputs)
        assert trace.capture_value_ids() == expected
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == expected
        assert request.capture_policy == "greedy"


def test_small_model_keeps_the_capture_everything_default(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(path))

        # The scope control is offered for every open session — hiding it
        # behind the byte advisory hid "Preview everything" on exactly the
        # models that needed it — while a small model still defaults to the
        # greedy capture-everything scope.
        assert shell.trace.capture_scope_choice.visible
        assert shell.trace.capture_scope_choice.value == "everything"
        assert shell.trace.capture_value_ids() == frozenset()
        assert shell.trace.capture_policy() == "greedy"
        assert str(shell.trace.capture_scope.value).startswith(
            "Capturing all 3 values (≥"
        )


def test_undeclared_shape_count_defaults_to_preview_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    # "scaled" declares no shape, so the byte estimate stays at a tiny 64 B —
    # far under the 1 GiB advisory — and only the always-known capturable
    # value count can move the default. Two stands in for the real 256-value
    # threshold a thousand-node model would cross.
    monkeypatch.setattr("nneditor.ui.trace_panel._PREVIEW_DEFAULT_VALUE_COUNT", 2)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace

        assert trace.capture_scope_choice.visible
        assert trace.capture_scope_choice.value == "preview-everything"
        assert trace.capture_value_ids() == frozenset()
        assert trace.capture_policy() == "preview"
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == frozenset()
        assert request.capture_policy == "preview"


def test_partially_declared_shapes_read_as_a_lower_bound(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        full = _estimate_text(session, _boundary_ids(session))

        # The 64 B sum covers only the input and output — "scaled" declares
        # no shape — so every estimate string must read as a lower bound
        # over the values it covers, never as the total.
        assert shell.trace.capture_scope.value == (
            f"Capturing all 3 values (≥{full}, 2 of 3 values declare shapes)."
        )
        assert shell.trace.capture_estimate.value == (
            f"Estimated decoded activations for this scope: "
            f"≥{full}, 2 of 3 values declare shapes."
        )
        everything_option = next(
            option
            for option in shell.trace.capture_scope_choice.options
            if option.key == "everything"
        )
        assert f"≥{full}" in str(everything_option.text)


def test_budget_dropped_majority_status_names_the_preview_way_out(
    tmp_path: Path,
) -> None:
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
        # The 16-byte pool stores one whole 16 B value and drops the other
        # two for budget, so most of the requested values are unreadable and
        # the status must name the way out rather than read as failure.
        dropped = [
            record
            for record in result.records
            if "capture byte ceiling" in (record.reason or "")
        ]
        assert 2 * len(dropped) > len(result.records)
        shell.trace.active_trace_id = result.id
        shell.trace.refresh_actions()
        status = str(shell.trace.status.value)
        assert "Partial trace" in status
        assert (
            "most values were not captured — switch Capture scope to "
            "Preview everything to cover every node"
        ) in status


def test_explicit_widen_sticks_for_the_session_but_not_the_next_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    monkeypatch.setattr("nneditor.ui.trace_panel._FULL_CAPTURE_ADVISORY_BYTES", 32)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace
        assert trace.capture_scope_choice.value == "preview-everything"

        trace.capture_scope_choice.value = "everything"
        cast(Any, trace.capture_scope_choice.on_select)(cast(Any, SimpleNamespace()))
        assert trace.capture_value_ids() == frozenset()
        assert str(trace.capture_scope.value).startswith("Capturing all 3 values (≥")

        # Selection churn and later refreshes keep the explicit choice, and
        # the approved request drops back to whole-value greedy budgeting.
        _select_operators(shell, session, session.document.main_graph.nodes[0].id)
        trace.refresh_actions()
        assert trace.capture_scope_choice.value == "everything"
        assert trace.capture_value_ids() == frozenset()
        request = _approved_request(shell, page, monkeypatch)
        assert request.value_ids == frozenset()
        assert request.capture_policy == "greedy"

        # A new session returns to the smart default; the choice was per-session.
        second_path = tmp_path / "second.onnx"
        build_embedded_model(second_path, elements=8)
        second = service.open_model(second_path)
        shell.show_session(second)
        assert shell.trace.capture_scope_choice.value == "preview-everything"
        assert shell.trace.capture_value_ids() == frozenset()
        assert shell.trace.capture_policy() == "preview"


def test_over_half_memory_estimate_warns_inline_before_the_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace

        assert trace.capture_estimate.visible
        assert "Estimated decoded activations" in str(trace.capture_estimate.value)
        assert trace.capture_estimate.color == shell.palette.muted

        trace.memory_mib.value = "0"
        trace.refresh_capture_scope()
        warning = str(trace.capture_estimate.value)
        assert "decodes to an estimated" in warning
        assert "more than half the approved 0 MiB memory limit" in warning
        assert "capture fewer values or raise the Memory limit" in warning
        assert trace.capture_estimate.color == shell.palette.warning

        trace.memory_mib.value = "2048"
        trace.refresh_capture_scope()
        assert "Estimated decoded activations" in str(trace.capture_estimate.value)
        assert trace.capture_estimate.color == shell.palette.muted


def test_runtime_preflight_probes_off_thread_and_renders_the_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    statuses = [
        RuntimeStatus(
            ("CUDAExecutionProvider", "CPUExecutionProvider"), "1.22.0", None
        ),
        RuntimeStatus(
            (),
            None,
            "ONNX Runtime is not installed in this environment; install "
            "nneditor[runtime] (or an accelerator extra such as "
            "nneditor[runtime-gpu]) to enable it",
        ),
    ]
    refreshes: list[bool] = []

    def fake_status(
        timeout_seconds: float = 10.0, *, refresh: bool = False
    ) -> RuntimeStatus:
        refreshes.append(refresh)
        return statuses.pop(0)

    monkeypatch.setattr("nneditor.ui.trace_panel.runtime_status", fake_status)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        assert "Open an artifact" in str(shell.trace.runtime_text.value)

        deferred: list[Any] = []
        cast(Any, page).run_thread = deferred.append
        shell.show_session(service.open_model(path))

        # The panel is fully constructed while the probe is still pending: a
        # quiet placeholder shows and the device dropdown keeps working.
        assert "Probing ONNX Runtime" in str(shell.trace.runtime_text.value)
        assert not shell.trace.device.disabled
        assert not refreshes

        while deferred:
            deferred.pop(0)()
        rendered = str(shell.trace.runtime_text.value)
        assert "CUDAExecutionProvider, CPUExecutionProvider" in rendered
        assert "1.22.0" in rendered
        assert refreshes == [False]

        # The refresh icon re-probes; a failure renders the error, which names
        # the nneditor[runtime] extra, without disturbing the device choice.
        cast(Any, page).run_thread = lambda target: target()
        cast(Any, shell.trace.runtime_refresh.on_click)(cast(Any, SimpleNamespace()))
        assert refreshes == [False, True]
        assert "nneditor[runtime]" in str(shell.trace.runtime_text.value)
        assert shell.trace.runtime_text.color == shell.palette.warning
        assert not shell.trace.device.disabled


def test_complete_trace_notes_render_as_info_and_never_read_as_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page, _callbacks, trace_id, session = _traced_shell(service, path)
        base = session.trace(trace_id)
        assert not base.partial

        noted = dataclasses.replace(
            base,
            notes=("CUDAExecutionProvider was skipped: device unavailable",),
        )
        monkeypatch.setattr(ModelSession, "trace", lambda self, _trace_id: noted)
        monkeypatch.setattr(ModelSession, "traces", lambda self: (noted,))
        shell.trace.refresh_actions()

        assert "Complete trace" in str(shell.trace.status.value)
        assert "Partial" not in str(shell.trace.status.value)
        assert shell.trace.result_annotations.visible
        notes = [
            control
            for control in shell.trace.result_annotations.controls
            if isinstance(control, ft.Container)
            and isinstance(control.data, str)
            and control.data.startswith("trace-note:")
        ]
        assert len(notes) == 1
        note_text = cast(ft.Text, notes[0].content)
        assert "skipped" in str(note_text.value)
        assert note_text.color == shell.palette.info
        assert notes[0].bgcolor == shell.palette.info_soft
        assert not any(
            isinstance(control.data, str)
            and control.data.startswith("trace-diagnostic:")
            for control in shell.trace.result_annotations.controls
        )

        # Diagnostics stay visually distinct (warning palette) and only they
        # accompany a partial presentation.
        flawed = dataclasses.replace(noted, diagnostics=("capture stopped early",))
        monkeypatch.setattr(ModelSession, "trace", lambda self, _trace_id: flawed)
        monkeypatch.setattr(ModelSession, "traces", lambda self: (flawed,))
        shell.trace.refresh_actions()

        assert "Partial trace" in str(shell.trace.status.value)
        diagnostics = [
            control
            for control in shell.trace.result_annotations.controls
            if isinstance(control, ft.Container)
            and isinstance(control.data, str)
            and control.data.startswith("trace-diagnostic:")
        ]
        assert len(diagnostics) == 1
        diagnostic_text = cast(ft.Text, diagnostics[0].content)
        assert diagnostic_text.color == shell.palette.warning
        assert diagnostics[0].bgcolor == shell.palette.warning_soft
        assert diagnostic_text.color != note_text.color


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
        cast(ft.Container, build_activation_plot(view, palette=SHELL_PALETTE))
        for view in views
    ]

    assert {control.data for control in controls} == {
        "activation-plot:heatmap",
        "activation-plot:histogram",
        "activation-plot:line",
        "activation-plot:tensor-layer-stack",
    }
    heatmap_control = next(
        control for control in controls if control.data == "activation-plot:heatmap"
    )
    assert isinstance(heatmap_control.content, ft.Image)
    assert all(
        isinstance(control.content, cv.Canvas) and control.content.shapes
        for control in controls
        if control.data in {"activation-plot:histogram", "activation-plot:line"}
    )
    heatmap = next(view for view in views if view.kind.value == "heatmap")
    large_heatmap = cast(
        ft.Container,
        build_activation_plot(heatmap, palette=SHELL_PALETTE, width=780, height=420),
    )
    large_image = cast(ft.Image, large_heatmap.content)
    assert large_image.width == 420
    assert large_image.height == 420


def test_histogram_plot_labels_its_axes_with_the_captured_range() -> None:
    vector = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float32)
    record = ActivationRecord(
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
    histogram = next(
        view
        for view in build_activation_visualizations(record, vector.tobytes())
        if view.kind.value == "histogram"
    )

    control = cast(
        ft.Container, build_activation_plot(histogram, palette=SHELL_PALETTE)
    )
    canvas = cast(cv.Canvas, control.content)
    labels = [shape.value for shape in canvas.shapes if isinstance(shape, cv.Text)]
    lines = [shape for shape in canvas.shapes if isinstance(shape, cv.Line)]
    bars = [shape for shape in canvas.shapes if isinstance(shape, cv.Rect)]

    assert len(bars) == len(histogram.values)
    # Bars sit inside the axis margin instead of at the canvas edge.
    assert all(bar.x > 0 for bar in bars)
    # Two axis lines plus one tick mark per label.
    assert len(lines) == 2 + len(labels)
    # The count axis shows zero and the tallest bar's actual count; the
    # value axis spans the captured data's actual bin-edge range.
    assert {"0", "1", "2", "3"} <= set(labels)

    large = cast(
        ft.Container,
        build_activation_plot(histogram, palette=SHELL_PALETTE, width=780, height=420),
    )
    large_canvas = cast(cv.Canvas, large.content)
    large_labels = [
        shape.value for shape in large_canvas.shapes if isinstance(shape, cv.Text)
    ]
    # Five value ticks across the same 1.0-3.0 range, three count ticks.
    assert {"1", "1.5", "2", "2.5", "3"} <= set(large_labels)
    assert len(large_labels) == 8


def test_feature_plot_uses_exact_resolution_raster() -> None:
    tensor = np.arange(2 * 18 * 47, dtype=np.float32).reshape(1, 2, 18, 47)
    record = ActivationRecord(
        "feature",
        "feature",
        "node",
        "node-output",
        "float32",
        "float32",
        tensor.shape,
        CaptureState.COMPLETE,
        tensor.nbytes,
        tensor.nbytes,
        "captures/feature.bin",
    )
    feature = next(
        view
        for view in build_activation_visualizations(record, tensor.tobytes())
        if view.kind.value == "feature-map-grid"
    )

    control = cast(
        ft.Container,
        build_activation_plot(feature, palette=SHELL_PALETTE, width=780, height=420),
    )

    assert isinstance(control.content, ft.Image)
    assert control.content.src == feature.raster_png
    assert feature.shape[-2:] == (18, 47)
    assert control.width == pytest.approx(780)
    assert control.height == pytest.approx(780 * 18 / 95)


def test_tensor_layer_viewer_rotates_and_brings_selected_plane_forward() -> None:
    tensor = np.arange(3 * 18 * 47, dtype=np.float32).reshape(3, 18, 47)
    record = ActivationRecord(
        "layers",
        "layers",
        "node",
        "node-output",
        "float32",
        "float32",
        tensor.shape,
        CaptureState.COMPLETE,
        tensor.nbytes,
        tensor.nbytes,
        "captures/layers.bin",
    )
    stack = next(
        view
        for view in build_activation_visualizations(record, tensor.tobytes())
        if view.kind.value == "tensor-layer-stack"
    )
    control = cast(
        ft.Container,
        build_activation_plot(stack, palette=SHELL_PALETTE, width=780, height=420),
    )
    controller = cast(Any, control).activation_layer_viewer
    initial_yaw = controller.yaw
    initial_pitch = controller.pitch

    controller.rotate(12, -8)
    controller.select(1)

    assert controller.yaw != initial_yaw
    assert controller.pitch != initial_pitch
    assert controller.selected == 1
    assert controller.scene.controls[-1] is controller.planes[1]
    assert controller.planes[1].opacity == 1.0
    assert controller.planes[0].opacity < 1.0
    assert "source 1" in str(controller.selection_text.value)


def test_tensor_layer_viewer_zoom_steps_clamps_and_survives_rotation() -> None:
    tensor = np.arange(3 * 18 * 47, dtype=np.float32).reshape(3, 18, 47)
    record = ActivationRecord(
        "layers",
        "layers",
        "node",
        "node-output",
        "float32",
        "float32",
        tensor.shape,
        CaptureState.COMPLETE,
        tensor.nbytes,
        tensor.nbytes,
        "captures/layers.bin",
    )
    stack = next(
        view
        for view in build_activation_visualizations(record, tensor.tobytes())
        if view.kind.value == "tensor-layer-stack"
    )
    control = cast(
        ft.Container,
        build_activation_plot(stack, palette=SHELL_PALETTE, width=780, height=420),
    )
    controller = cast(Any, control).activation_layer_viewer

    assert controller.scale == pytest.approx(1.0)
    controller.zoom_in()
    assert controller.scale == pytest.approx(1.25)
    controller.zoom_out()
    assert controller.scale == pytest.approx(1.0)

    for _ in range(12):
        controller.zoom_in()
    assert controller.scale == pytest.approx(4.0)
    for _ in range(24):
        controller.zoom_out()
    assert controller.scale == pytest.approx(0.4)

    row = cast(ft.Row, cast(ft.Column, control.content).controls[1])
    buttons = {
        button.data: button
        for button in row.controls
        if isinstance(button, ft.IconButton)
    }
    zoom_in_button = buttons["layer-zoom-in"]
    zoom_out_button = buttons["layer-zoom-out"]
    assert zoom_in_button.tooltip == "Zoom in"
    assert zoom_out_button.tooltip == "Zoom out"
    event = cast("ft.Event[ft.IconButton]", cast(Any, SimpleNamespace()))
    assert zoom_in_button.on_click is not None
    assert zoom_out_button.on_click is not None
    cast(Any, zoom_in_button.on_click)(event)
    assert controller.scale == pytest.approx(0.5)
    cast(Any, zoom_out_button.on_click)(event)
    assert controller.scale == pytest.approx(0.4)

    cast(Any, zoom_in_button.on_click)(event)
    zoomed = controller.scale
    controller.rotate(12, -8)
    controller.select(1)
    assert controller.scale == pytest.approx(zoomed)
    assert controller.selected == 1
    matrix = controller.planes[1].transform.matrix
    scale_ops = [op for op in matrix.ops if op.name == "scale"]
    assert scale_ops and scale_ops[0].args == [pytest.approx(zoomed)]


def test_tensor_layer_viewer_keeps_stack_inside_perspective_focal_depth() -> None:
    tensor = np.arange(16 * 18 * 47, dtype=np.float32).reshape(16, 18, 47)
    record = ActivationRecord(
        "layers",
        "layers",
        "node",
        "node-output",
        "float32",
        "float32",
        tensor.shape,
        CaptureState.COMPLETE,
        tensor.nbytes,
        tensor.nbytes,
        "captures/layers.bin",
    )
    stack = next(
        view
        for view in build_activation_visualizations(record, tensor.tobytes())
        if view.kind.value == "tensor-layer-stack"
    )
    assert len(stack.layer_pngs) == 16
    control = cast(
        ft.Container,
        build_activation_plot(stack, palette=SHELL_PALETTE, width=780, height=420),
    )
    controller = cast(Any, control).activation_layer_viewer

    # Worst case for the perspective divisor: maximum zoom, pitch pinned to
    # its clamp, arbitrary yaw, and the selected plane pulled clear of the
    # stack (the deepest translate the viewer ever issues).
    for _ in range(12):
        controller.zoom_in()
    assert controller.scale == pytest.approx(4.0)
    controller.rotate(131.0, -1000.0)
    assert controller.pitch == pytest.approx(1.35)
    controller.select(15)

    budget = activation_layers._DEPTH_SAFETY * controller.focal_depth
    # The documented invariant: even the deepest translate, zoomed to the
    # 4.0 maximum, stays within the safety fraction of the focal depth.
    deepest = controller.layer_spacing * (len(controller.planes) + 1)
    assert deepest * 4.0 <= budget

    for plane in controller.planes:
        matrix = plane.transform.matrix
        entry = next(op for op in matrix.ops if op.name == "set_entry")
        assert entry.args == [3, 2, pytest.approx(1.0 / controller.focal_depth)]
        translate = next(op for op in matrix.ops if op.name == "translate")
        # Rotation never grows a translate's z beyond its own magnitude and
        # the uniform zoom multiplies it by at most the current scale, so
        # this bounds the plane's effective z. Staying under the safety
        # fraction keeps homogeneous w positive: the plane cannot flip
        # behind the camera and vanish.
        effective = abs(float(translate.args[2])) * controller.scale
        assert effective <= budget
        assert effective < controller.focal_depth

    # Neither the scene stack nor the viewer body may crop planes swinging
    # outside the box while rotating.
    assert controller.scene.clip_behavior is ft.ClipBehavior.NONE
    assert control.clip_behavior is ft.ClipBehavior.NONE


def _traced_shell(
    service: ApplicationService,
    path: Path,
) -> tuple[Shell, StubPage, list[Any], str, ModelSession]:
    """Open the model, run one approved trace, and defer run_thread waiters."""
    shell, page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    callbacks: list[Any] = []
    cast(Any, page).run_thread = callbacks.append
    shell.trace.run(cast(ft.Event[ft.Button], cast(Any, SimpleNamespace())))
    callbacks.pop(0)()
    callbacks.pop(0)()
    trace_id = shell.trace.active_trace_id
    assert trace_id is not None
    return shell, page, callbacks, trace_id, session


def _activation_card(shell: Shell, value_id: str) -> ft.ExpansionTile:
    return next(
        control
        for control in shell.inspector.controls
        if isinstance(control, ft.ExpansionTile)
        and control.data == f"activation-card:{value_id}"
    )


def _build_views_button(card: ft.ExpansionTile) -> ft.TextButton | None:
    assert card.controls
    body = cast(ft.Column, card.controls[0])
    return next(
        (
            child
            for child in body.controls
            if isinstance(child, ft.TextButton)
            and child.content == "Build activation views now"
        ),
        None,
    )


def _count_view_submissions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every visualization job the session is asked to start."""
    submitted: list[str] = []
    original = ModelSession.activation_visualizations_async

    def counting(
        session: ModelSession,
        trace_id: str,
        value_id: str,
        *,
        attention: bool = False,
    ) -> Any:
        submitted.append(value_id)
        return original(session, trace_id, value_id, attention=attention)

    monkeypatch.setattr(ModelSession, "activation_visualizations_async", counting)
    return submitted


def test_visualization_cache_evicts_under_budget_and_rebuilds_on_click(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    # Measure what the selection's two view tuples really cost, on a first
    # trace of the same deterministic model, so the injected budget admits
    # either entry alone but never both together.
    with ApplicationService() as measuring:
        _shell, _page, _callbacks, trace_id, session = _traced_shell(measuring, path)
        node_id = session.document.main_graph.nodes[0].id
        records = session.node_activations(trace_id, node_id)
        view = session.activations(trace_id)
        costs = [
            _visualization_cost(
                build_activation_visualizations(record, view.read(record.value_id))
            )
            for record in records
        ]
        assert len(costs) == 2

    monkeypatch.setattr(
        "nneditor.ui.app.ActivationInspector",
        partial(ActivationInspector, visualization_cache_budget=max(costs)),
    )
    with ApplicationService() as service:
        shell, page, callbacks, trace_id, session = _traced_shell(service, path)
        cache = shell.activations.activation_visualizations
        assert cache.budget == max(costs)

        node_id = session.document.main_graph.nodes[0].id
        _select_operators(shell, session, node_id)
        while callbacks:
            callbacks.pop(0)()

        first, second = session.node_activations(trace_id, node_id)
        # Both views were built; the budget retained only the more recent one.
        assert cache.stats.evictions == 1
        assert cache.peek((trace_id, first.value_id)) is None
        assert cache.peek((trace_id, second.value_id)) is not None

        shell._refresh_inspector(frozenset({node_id}))
        button = _build_views_button(_activation_card(shell, first.value_id))
        assert button is not None
        assert _build_views_button(_activation_card(shell, second.value_id)) is None

        # The manual button is the recovery path: clicking it rebuilds the
        # evicted entry through a fresh job.
        cast(Any, page).run_thread = lambda target: target()
        cast(Any, button.on_click)(cast(Any, SimpleNamespace()))
        assert cache.peek((trace_id, first.value_id)) is not None
        rebuilt_card = _activation_card(shell, first.value_id)
        assert rebuilt_card.controls
        rebuilt = cast(ft.Column, rebuilt_card.controls[0])
        assert any(
            isinstance(child, ft.FilledButton) and child.content == "Open large view"
            for child in rebuilt.controls
        )


def test_selection_autoload_is_capped_and_the_rest_keep_the_manual_button(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    monkeypatch.setattr("nneditor.ui.activation_inspector._AUTOLOAD_VIEW_LIMIT", 1)
    with ApplicationService() as service:
        shell, _page, callbacks, trace_id, session = _traced_shell(service, path)
        submitted = _count_view_submissions(monkeypatch)

        node_id = session.document.main_graph.nodes[0].id
        _select_operators(shell, session, node_id)

        first, second = session.node_activations(trace_id, node_id)
        # Two readable records but a cap of one: exactly the first record's
        # job starts, and the second keeps its manual build button.
        assert submitted == [first.value_id]
        assert _build_views_button(_activation_card(shell, first.value_id)) is None
        assert _build_views_button(_activation_card(shell, second.value_id)) is not None

        while callbacks:
            callbacks.pop(0)()
        assert submitted == [first.value_id]
        assert _build_views_button(_activation_card(shell, second.value_id)) is not None


def test_inflight_view_request_is_not_resubmitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page, callbacks, trace_id, session = _traced_shell(service, path)
        submitted = _count_view_submissions(monkeypatch)

        node_id = session.document.main_graph.nodes[0].id
        _select_operators(shell, session, node_id)
        records = session.node_activations(trace_id, node_id)
        assert submitted == [record.value_id for record in records]

        # The waiters have not run, so every request is still in flight; a
        # repeated selection and even a forced manual request submit nothing.
        before = list(submitted)
        shell.activations.autoload_views(frozenset({node_id}))
        first = records[0]
        assert not shell.activations._request_views(trace_id, first.value_id, node_id)
        assert not shell.activations._request_views(
            trace_id, first.value_id, node_id, force=True
        )
        assert submitted == before

        while callbacks:
            callbacks.pop(0)()
        # Resolved views are cached, so a repeat request stays a no-op.
        assert not shell.activations._request_views(trace_id, first.value_id, node_id)
        assert submitted == before


# -- export affordances ------------------------------------------------------


def test_activation_array_reconstructs_full_and_truncated_captures() -> None:
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    full, truncated = viewmodel.activation_array("float32", (2, 3), data.tobytes())
    assert not truncated
    assert full.dtype == np.float32
    np.testing.assert_array_equal(full, data)

    # Nine bytes hold two full elements plus one dangling byte: the prefix
    # decodes flat and the dangling byte is dropped, flagged as truncated.
    prefix, truncated = viewmodel.activation_array(
        "float32", (2, 3), data.tobytes()[:9]
    )
    assert truncated
    assert prefix.shape == (2,)
    np.testing.assert_array_equal(prefix, data.ravel()[:2])

    with pytest.raises(ValueError, match="not reconstructible"):
        viewmodel.activation_array("not-a-dtype", (2,), b"\x00")


def test_activation_card_saves_the_captured_tensor_as_npy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    destination = tmp_path / "activation.npy"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, _page, callbacks, trace_id, session = _traced_shell(service, path)
        node_id = session.document.main_graph.nodes[0].id
        _select_operators(shell, session, node_id)
        while callbacks:
            callbacks.pop(0)()

        record = session.node_activations(trace_id, node_id)[0]
        card = _activation_card(shell, record.value_id)
        assert card.controls
        body = cast(ft.Column, card.controls[0])
        button = next(
            child
            for child in body.controls
            if isinstance(child, ft.TextButton)
            and child.data == f"activation-save-npy:{record.value_id}"
        )

        async def save_file(picker: ft.FilePicker, **kwargs: object) -> str:
            assert kwargs["allowed_extensions"] == ["npy"]
            return str(destination)

        monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
        asyncio.run(cast(Any, button.on_click)(cast(Any, SimpleNamespace())))

        saved = np.load(destination, allow_pickle=False)
        raw = session.activations(trace_id).read(record.value_id)
        assert saved.dtype == np.dtype(record.numpy_dtype)
        assert saved.shape == tuple(record.shape)
        assert saved.tobytes() == raw
        assert "Saved" in str(shell.status_text.value)
        assert "truncated" not in str(shell.status_text.value)


def test_truncated_activation_save_discloses_the_stored_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    destination = tmp_path / "prefix.npy"
    build_embedded_model(path, elements=8)
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
        record = next(
            item for item in result.records if item.state is CaptureState.TRUNCATED
        )

        async def save_file(picker: ft.FilePicker, **kwargs: object) -> str:
            return str(destination)

        monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
        handler = shell.activations._save_activation_handler(result.id, record)
        asyncio.run(handler(cast(Any, SimpleNamespace())))

        saved = np.load(destination, allow_pickle=False)
        width = np.dtype(record.numpy_dtype).itemsize
        assert saved.ndim == 1
        assert saved.nbytes == record.stored_byte_length - (
            record.stored_byte_length % width
        )
        status = str(shell.status_text.value)
        assert "truncated" in status
        assert f"{record.stored_byte_length:,} of {record.full_byte_length:,}" in status


def test_save_view_as_png_writes_the_exact_raster_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    record = ActivationRecord(
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
    heatmap = next(
        view
        for view in build_activation_visualizations(record, matrix.tobytes())
        if view.raster_png
    )
    destination = tmp_path / "view.png"

    async def save_file(picker: ft.FilePicker, **kwargs: object) -> str:
        assert kwargs["allowed_extensions"] == ["png"]
        return str(destination)

    monkeypatch.setattr(ft.FilePicker, "save_file", save_file)
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        handler = shell.activations._save_view_png_handler(heatmap, "matrix")
        asyncio.run(handler(cast(Any, SimpleNamespace())))

    assert destination.read_bytes() == heatmap.raster_png
    assert "Saved" in str(shell.status_text.value)


def test_raster_views_offer_a_png_save_button_on_the_card(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    raster_record = ActivationRecord(
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
    raster_views = tuple(
        view
        for view in build_activation_visualizations(raster_record, matrix.tobytes())
        if view.raster_png
    )
    assert raster_views
    with ApplicationService() as service:
        shell, _page, _callbacks, trace_id, session = _traced_shell(service, path)
        node_id = session.document.main_graph.nodes[0].id
        record = session.node_activations(trace_id, node_id)[0]
        shell.activations.activation_visualizations.put(
            (trace_id, record.value_id), raster_views
        )
        _select_operators(shell, session, node_id)

        card = _activation_card(shell, record.value_id)
        assert card.controls
        body = cast(ft.Column, card.controls[0])
        assert any(
            isinstance(child, ft.TextButton)
            and isinstance(child.data, str)
            and child.data.startswith(f"activation-save-png:{record.value_id}:")
            for child in body.controls
        )


def test_escape_closes_the_large_view_overlay_before_clearing_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page, callbacks, _trace_id, session = _traced_shell(service, path)
        node_id = session.document.main_graph.nodes[0].id
        _select_operators(shell, session, node_id)
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
        cast(Any, open_button.on_click)(cast(Any, SimpleNamespace()))
        assert page.dialogs
        dialog = page.dialogs[-1]
        assert dialog.open
        # The overlay itself carries the tensor and raster export affordances.
        rows = cast(ft.ListView, cast(ft.Container, dialog.content).content).controls
        assert any(
            isinstance(row, ft.TextButton)
            and isinstance(row.data, str)
            and row.data.startswith("activation-overlay-save-npy:")
            for row in rows
        )

        press_key(shell, "Escape")
        assert not dialog.open
        assert shell.renderer.selection == frozenset({node_id})

        # The operations drawer is next in the dismissal chain: it closes
        # before the selection is touched.
        cast(Any, shell.operation_buttons["edit"].on_click)(cast(Any, None))
        assert shell.operations_drawer.visible
        press_key(shell, "Escape")
        assert not shell.operations_drawer.visible
        assert shell.renderer.selection == frozenset({node_id})

        press_key(shell, "Escape")
        assert shell.renderer.selection == frozenset()
