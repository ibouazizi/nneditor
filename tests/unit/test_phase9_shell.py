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
from PIL import Image

from nneditor.application.session import ApplicationService
from nneditor.desktop.windows_associations import FileAssociationError
from nneditor.tokenization import TOKENIZER_CHOICES, WordHashCodebook
from nneditor.tracing import (
    ActivationRecord,
    CaptureState,
    TraceApproval,
    TraceLimits,
    TraceRequest,
    build_activation_visualizations,
)
from nneditor.ui.activation_inspector import build_activation_plot
from nneditor.ui.app import SHELL_PALETTE
from nneditor.ui.input_workspace import InputTarget
from tests.fixtures.onnx_models import (
    build_embedded_model,
    build_masked_image_model,
    build_token_ids_model,
)
from tests.unit.test_shell import make_shell


def test_trace_shell_empty_consent_loading_and_activation_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        shell, page = make_shell(service)
        assert shell.trace.run_button.disabled
        assert "Open an artifact" in str(shell.trace.status.value)

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

        assert generator.kind.value == "mask"
        assert generator.mask_fill.value == "ones"
        assert generator.mask_dtype.value == "int64"
        assert generator.mask_section.visible
        assert not generator.image_section.visible


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

        assert generator.kind.value == "image"
        assert generator.image_layout.value == "QWEN3VL_PATCHES"
        assert generator.image_width.value == "640"
        assert generator.image_height.value == "352"
        assert generator.image_normalization.value == "minus-one-one"
        assert generator.image_section.visible


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

        assert generator.kind.value == "image"
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

        # A rank-2 integer input is a token-id batch, so the form opens there.
        assert generator.kind.value == "text-tokens"
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
