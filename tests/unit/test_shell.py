"""Headless wiring tests for the Flet shell (P1.8).

A stub page satisfies the page surface the shell uses (overlay, services,
update, run_thread), so session display, graph switching, search, focus, and
error states can be verified without a display server. What cannot be tested
headlessly — the window actually painting — is covered by the P1.10 smoke
run.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.analysis.lod import DetailLevel
from nneditor.application.hierarchy import OrganizationMode
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.artifact_formats import MODEL_FILE_EXTENSIONS
from nneditor.rendering import (
    InteractiveGraphRenderer,
    SelectionCallback,
    create_flet_renderer,
)
from nneditor.ui import shell_layout
from nneditor.ui.app import SHELL_PALETTE, Shell
from tests.fixtures.onnx_models import build_control_flow_model, build_embedded_model


class StubPage:
    """The minimal page surface the shell touches."""

    def __init__(self) -> None:
        # Mirrors the real Page: Controls go in `overlay`, Services in
        # `services`. Modelling only `overlay` is what let a Service being
        # registered as a Control reach users.
        self.overlay: list[ft.Control] = []
        self.services: list[object] = []
        self.dialogs: list[ft.AlertDialog] = []
        self.updates = 0
        self.width = 1400

    def update(self) -> None:
        self.updates += 1

    def run_thread(self, target: Any) -> None:
        target()

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        dialog.open = True
        self.dialogs.append(dialog)
        self.update()

    def pop_dialog(self) -> ft.AlertDialog | None:
        if not self.dialogs:
            return None
        dialog = self.dialogs[-1]
        dialog.open = False
        self.update()
        return dialog


@pytest.fixture
def service() -> Any:
    with ApplicationService() as instance:
        yield instance


def make_shell(service: ApplicationService) -> tuple[Shell, StubPage]:
    page = StubPage()
    shell = Shell(cast(ft.Page, page), service)
    shell.build()
    return shell, page


def press_key(
    shell: Shell,
    key: str,
    *,
    ctrl: bool = False,
    shift: bool = False,
) -> None:
    """Deliver one global keyboard event to the shell."""
    shell.on_keyboard(
        ft.KeyboardEvent(
            name="keyboard",
            control=shell.page,
            key=key,
            shift=shift,
            ctrl=ctrl,
            alt=False,
            meta=False,
        )
    )


def test_shell_uses_the_injected_renderer_factory(
    service: ApplicationService,
) -> None:
    page = StubPage()
    created: list[InteractiveGraphRenderer] = []

    def factory(
        on_select: SelectionCallback | None,
    ) -> InteractiveGraphRenderer:
        renderer = create_flet_renderer(on_select)
        created.append(renderer)
        return renderer

    shell = Shell(cast(ft.Page, page), service, renderer_factory=factory)
    assert created == [shell.renderer]
    assert shell.renderer_control is shell.renderer.control


def test_show_session_populates_all_panels(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))

    assert shell.renderer.scene is not None
    assert (
        sum(
            node.kind not in {"graph-input", "graph-output"}
            for node in shell.renderer.scene.nodes
        )
        == 2
    )
    assert {node.kind for node in shell.renderer.scene.nodes} >= {
        "graph-input",
        "graph-output",
    }
    assert len(shell.graph_list.controls) == 1
    assert shell.model_info.controls, "the Model section shows the model summary"
    assert shell.inspector.controls, "the Selection section explains the empty state"
    assert shell.hierarchy_list.controls
    assert shell.minimap_model is not None
    assert "model.onnx" in (shell.title_text.value or "")
    assert "Opened" in (shell.status_text.value or "")
    assert shell.current_detail is DetailLevel.ARCHITECTURE
    assert shell.detail_segment.selected == [DetailLevel.ARCHITECTURE.value]
    assert not shell.reset_view_button.disabled
    assert not shell.empty_state.visible
    assert not shell.loading_overlay.visible
    assert not shell.error_banner.visible


def test_command_line_path_opens_through_the_normal_background_job(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    shell, _page = make_shell(service)

    shell.open_path(path)

    assert shell.session is not None
    assert Path(shell.session.document.source.path).resolve() == path.resolve()
    assert "Opened" in str(shell.status_text.value)


def test_save_action_tracks_unsaved_revisions(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nneditor.editing.validation import RenameNodeRequest

    path = tmp_path / "model.onnx"
    destination = tmp_path / "saved.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)

    assert shell.close_model_button.visible
    assert not shell.save_model_button.visible

    node = session.document.main_graph.nodes[0]
    shell.renderer.set_selection(frozenset({node.id}))
    shell.pending_edit = session.prepare_edit(
        RenameNodeRequest(session.document.entry_graph, node.id, "first revision")
    )
    shell.pending_target = node.id
    shell._on_commit_edit(cast(Any, None))

    assert shell.save_model_button.visible
    assert not shell.save_model_button.disabled

    async def choose_destination(*args: Any, **kwargs: Any) -> Path:
        return destination

    monkeypatch.setattr(type(shell.picker), "save_file", choose_destination)
    asyncio.run(shell._on_save_model(cast(Any, None)))

    assert destination.is_file()
    assert not shell.save_model_button.visible
    assert "Exported" in (shell.status_text.value or "")

    shell.renderer.set_selection(frozenset({node.id}))
    shell.pending_edit = session.prepare_edit(
        RenameNodeRequest(session.document.entry_graph, node.id, "second revision")
    )
    shell.pending_target = node.id
    shell._on_commit_edit(cast(Any, None))
    assert shell.save_model_button.visible


def test_close_model_releases_session_and_clears_the_shell(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)

    shell._on_close_model(cast(Any, None))

    assert session.closed
    assert service.open_sessions == ()
    assert shell.session is None
    assert shell.current_graph is None
    assert shell.renderer.scene is not None
    assert shell.renderer.scene.node_count == 0
    assert shell.renderer.scene.edge_count == 0
    assert shell.empty_state.visible
    assert shell.reset_view_button.disabled
    assert not shell.close_model_button.visible
    assert not shell.save_model_button.visible
    assert shell.graph_list.controls == []
    assert shell.hierarchy_list.controls == []
    assert shell.inspector.controls == []
    assert shell.model_info.controls == []
    assert shell.status_text.value == "Closed model.onnx"


def test_model_overview_uses_structured_metadata_sections(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))

    sections = {
        control.data: control
        for control in shell.model_info.controls
        if isinstance(control.data, str)
    }
    assert {
        "model-overview-summary",
        "model-overview-artifact",
        "model-overview-capabilities",
    } <= sections.keys()

    summary = cast(ft.Column, sections["model-overview-summary"])
    metrics = next(
        control for control in summary.controls if isinstance(control, ft.ResponsiveRow)
    )
    assert metrics.data == "model-overview-metrics"
    assert len(metrics.controls) == 4
    assert {
        control.data
        for control in metrics.controls
        if isinstance(control, ft.Container)
    } == {
        "overview-metric:graphs",
        "overview-metric:operators",
        "overview-metric:tensors",
        "overview-metric:model-size",
    }

    capabilities = cast(ft.Column, sections["model-overview-capabilities"])
    capability_cards = [
        control
        for control in capabilities.controls
        if isinstance(control, ft.Container)
        and isinstance(control.data, str)
        and control.data.startswith("overview-capability:")
    ]
    assert len(capability_cards) == 7
    assert shell.inspector.spacing == 14
    assert "ONNX Model" in (shell.inspector_subtitle.value or "")


def test_typed_preview_requires_explicit_consent(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    bias_id = session.document.main_graph.initializers[1]

    prompt = shell.activations.preview_row(bias_id)
    assert isinstance(prompt, ft.TextButton)
    assert "full typed tensor" in str(prompt.content)
    assert session.store.open_file_count == 0

    shell.activations.typed_preview_handler(bias_id)(cast(Any, None))
    preview = shell.activations.preview_row(bias_id)
    assert isinstance(preview, ft.Text)
    assert "Preview:" in (preview.value or "")
    assert session.store.open_file_count == 1


def test_graph_switching_replaces_the_scene(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_control_flow_model(path)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    assert shell.renderer.scene is not None
    assert (
        sum(
            node.kind not in {"graph-input", "graph-output"}
            for node in shell.renderer.scene.nodes
        )
        == 1
    )

    branch = session.document.main_graph.nodes[0]
    shell._show_graph(branch.subgraphs[0])
    assert shell.current_graph == branch.subgraphs[0]
    assert shell.renderer.scene.node_count == 1


def test_search_and_focus_select_the_node(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)

    shell.search_field.value = "shift"
    shell._on_search(cast(Any, None))
    assert len(shell.search_results.controls) == 1
    assert "1 match(es)" in (shell.status_text.value or "")

    (hit_id,) = session.search("shift")
    shell.focus_node(hit_id)
    assert shell.renderer.selection == {hit_id}
    viewport = shell.renderer.viewport
    assert viewport is not None
    scene_node = shell.renderer.scene.node(hit_id)  # type: ignore[union-attr]
    center_x = viewport.x + viewport.width / 2
    assert center_x == pytest.approx(scene_node.x + scene_node.width / 2)
    metadata = next(
        control
        for control in shell.inspector.controls
        if control.data == "selection-item-metadata"
    )
    assert isinstance(metadata, ft.Column)
    card = metadata.controls[1]
    assert isinstance(card, ft.Container)
    details = card.content
    assert isinstance(details, ft.Column)
    inspector_text = " ".join(
        text.value or ""
        for row in details.controls
        if isinstance(row, ft.Column)
        for text in row.controls
        if isinstance(text, ft.Text)
    )
    assert "Add" in inspector_text, "the inspector shows the focused node"


def test_failed_async_open_shows_the_error_banner(
    tmp_path: Path, service: ApplicationService
) -> None:
    bogus = tmp_path / "bad.onnx"
    bogus.write_bytes(b"not a model")
    shell, page = make_shell(service)
    job = service.open_model_async(bogus)
    shell._watch_open(job)
    assert shell.error_banner.visible
    assert shell.session is None
    assert page.updates > 0


def test_open_dialog_filters_every_supported_model_extension(
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, _page = make_shell(service)
    calls: list[dict[str, Any]] = []

    async def pick_files(*args: Any, **kwargs: Any) -> list[ft.FilePickerFile]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(type(shell.picker), "pick_files", pick_files)
    asyncio.run(shell._on_open_clicked(cast(Any, None)))

    assert len(calls) == 1
    assert calls[0]["file_type"] is ft.FilePickerFileType.CUSTOM
    assert calls[0]["allowed_extensions"] == list(MODEL_FILE_EXTENSIONS)
    assert set(calls[0]["allowed_extensions"]) == {
        "onnx",
        "pb",
        "pt2",
        "pt",
        "pth",
        "ckpt",
        "bin",
        "safetensors",
        "mlir",
        "stablehlo",
        "dlc",
    }


def test_narrow_layout_collapses_side_panels(service: ApplicationService) -> None:
    shell, _page = make_shell(service)
    shell.apply_layout_for_width(800.0)
    assert not shell.left_panel.visible
    assert not shell.right_panel.visible
    shell.apply_layout_for_width(1400.0)
    assert shell.left_panel.visible
    assert shell.right_panel.visible


def test_pan_and_scroll_update_the_view(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    scale = shell.view["scale"]
    x_before = shell.view["x"]

    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)
    pan = ft.DragUpdateEvent(
        name="pan_update",
        control=detector,
        local_position=ft.Offset(0.0, 0.0),
        global_position=ft.Offset(0.0, 0.0),
        local_delta=ft.Offset(-50.0, 0.0),
        timestamp=None,
    )
    shell._on_pan(cast(Any, pan))
    assert shell.view["x"] == pytest.approx(x_before + 50.0 / scale)

    scroll = ft.ScrollEvent(
        name="scroll",
        control=detector,
        local_position=ft.Offset(240.0, 180.0),
        global_position=ft.Offset(240.0, 180.0),
        scroll_delta=ft.Offset(0.0, -10.0),
    )
    world_before = (
        shell.view["x"] + 240.0 / shell.view["scale"],
        shell.view["y"] + 180.0 / shell.view["scale"],
    )
    shell._on_scroll(cast(Any, scroll))
    assert shell.view["scale"] > scale
    world_after = (
        shell.view["x"] + 240.0 / shell.view["scale"],
        shell.view["y"] + 180.0 / shell.view["scale"],
    )
    assert world_after == pytest.approx(world_before)


def test_cursor_anchor_survives_a_semantic_zoom_transition(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell.current_detail = DetailLevel.BLOCK
    shell.auto_detail = False
    shell._replace_current_representation()
    shell.auto_detail = True
    shell.view = {"scale": 0.34, "x": 25.0, "y": 40.0}
    shell.renderer.set_viewport(shell._current_viewport())
    cursor = ft.Offset(300.0, 220.0)
    anchor = (
        shell.view["x"] + cursor.x / shell.view["scale"],
        shell.view["y"] + cursor.y / shell.view["scale"],
    )
    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)

    shell._on_scroll(
        cast(
            Any,
            ft.ScrollEvent(
                name="scroll",
                control=detector,
                local_position=cursor,
                global_position=cursor,
                scroll_delta=ft.Offset(0.0, -10.0),
            ),
        )
    )

    assert shell.current_detail is DetailLevel.LAYER
    assert (
        shell.view["x"] + cursor.x / shell.view["scale"],
        shell.view["y"] + cursor.y / shell.view["scale"],
    ) == pytest.approx(anchor)


def test_reset_view_refits_without_changing_context_or_selection(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    fitted_view = dict(shell.view)
    scene = shell.renderer.scene
    assert scene is not None
    selected = scene.nodes[0].id
    shell.renderer.set_selection(frozenset({selected}))
    shell._on_selected(frozenset({selected}))

    shell.view = {"scale": 3.5, "x": 750.0, "y": -240.0}
    shell.renderer.set_viewport(shell._current_viewport())
    shell._on_reset_view(cast(Any, None))

    assert shell.view == pytest.approx(fitted_view)
    assert shell.current_detail is DetailLevel.ARCHITECTURE
    assert shell.current_root_group is None
    assert shell.renderer.selection == {selected}
    assert shell.status_text.value == "Architecture view reset"


def test_hover_shows_context_without_changing_selection(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, page = make_shell(service)
    shell.show_session(service.open_model(path))
    scene = shell.renderer.scene
    viewport = shell.renderer.viewport
    assert scene is not None and viewport is not None
    node = scene.nodes[0]
    screen_x, screen_y = viewport.to_screen(
        node.x + node.width / 2, node.y + node.height / 2
    )

    shell._on_surface_hover(
        cast(Any, SimpleNamespace(local_position=ft.Offset(screen_x, screen_y)))
    )

    assert shell.hover_card.visible
    assert shell.hover_title.value == node.label
    assert node.display_type in (shell.hover_summary.value or "")
    assert not shell.renderer.selection
    updates = page.updates
    shell._on_surface_exit(cast(Any, None))
    assert not shell.hover_card.visible
    assert page.updates > updates


def test_manual_group_control_collapses_and_resets_selection(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node_ids = frozenset(node.id for node in session.document.main_graph.nodes)
    shell.renderer.set_selection(node_ids)
    shell._on_selected(node_ids)
    shell.group_label_field.value = "Manual block"
    shell._on_group_selected(cast(Any, None))

    hierarchy = session.graph_hierarchy()
    manual = next(group for group in hierarchy.groups if not group.automatic)
    assert manual.label == "Manual block"
    assert shell.current_detail is DetailLevel.BLOCK
    assert shell.renderer.scene is not None
    assert shell.renderer.scene.has_node(manual.id)

    shell.renderer.set_selection(frozenset({manual.id}))
    shell._on_selected(frozenset({manual.id}))
    assert shell.inspector_title.value == "Manual block"
    assert shell.open_selection_button.visible
    assert {
        control.data
        for control in shell.inspector.controls
        if isinstance(control.data, str)
    } >= {"selection-block-summary", "selection-item-metadata"}
    assert not any(isinstance(control, ft.Text) for control in shell.inspector.controls)

    shell._on_surface_double_tap(cast(Any, None))
    assert shell.current_root_group == manual.id
    assert str(shell.current_detail) == "layer"
    assert not shell.open_selection_button.visible
    assert {
        control.data
        for control in shell.inspector.controls
        if isinstance(control.data, str)
    } >= {"selection-block-summary", "selection-item-metadata"}

    shell._on_back_to_parent(cast(Any, None))
    assert shell.current_root_group is None

    shell._on_reset_groups(cast(Any, None))
    assert all(group.automatic for group in session.graph_hierarchy().groups)


def test_detail_breadcrumb_minimap_and_keyboard_navigation(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)

    shell.detail_segment.selected = [DetailLevel.ARCHITECTURE.value]
    shell._on_detail_segment_changed(cast(Any, None))
    assert shell.current_detail is DetailLevel.ARCHITECTURE
    assert not shell.auto_detail
    assert shell.breadcrumbs_row.controls
    assert shell.minimap_canvas.shapes

    shell.detail_segment.selected = [DetailLevel.OPERATOR.value]
    shell._on_detail_segment_changed(cast(Any, None))
    scene = shell.renderer.scene
    assert scene is not None
    first = scene.nodes[0]
    shell.renderer.set_selection(frozenset((first.id,)))
    shell.on_keyboard(
        ft.KeyboardEvent(
            name="keyboard",
            control=shell.page,
            key="Arrow Down",
            shift=False,
            ctrl=False,
            alt=False,
            meta=False,
        )
    )
    assert shell.renderer.selection != {first.id}


def test_streamed_upload_progress_hands_off_to_the_open_worker(
    tmp_path: Path, service: ApplicationService
) -> None:
    uploaded = tmp_path / "streamed-upload.onnx"
    build_embedded_model(uploaded, elements=16)
    shell, page = make_shell(service)
    shell._pending_web_upload = (uploaded, "friendly.onnx")

    shell._on_upload_progress(
        ft.FilePickerUploadEvent(
            name="upload",
            control=shell.picker,
            file_name="friendly.onnx",
            progress=1.0,
        )
    )

    assert shell.session is not None
    assert shell.session.title == "friendly.onnx"
    assert shell._pending_web_upload is None
    assert shell.open_job is None
    assert not shell.loading_overlay.visible
    assert page.updates > 0


def test_streamed_upload_error_is_visible(service: ApplicationService) -> None:
    shell, _page = make_shell(service)
    shell._pending_web_upload = (Path("unused.onnx"), "broken.onnx")

    shell._on_upload_progress(
        ft.FilePickerUploadEvent(
            name="upload",
            control=shell.picker,
            file_name="broken.onnx",
            progress=0.25,
            error="network interrupted",
        )
    )

    assert shell.error_banner.visible
    assert shell._pending_web_upload is None


class TestBlockingUiFixes:
    """Regressions found by the UX audit; each fails without its fix."""

    def open_shell(
        self, tmp_path: Path, service: ApplicationService
    ) -> tuple[Shell, Any]:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=16)
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        return shell, session

    def test_panels_can_be_reopened_after_a_narrow_collapse(
        self, service: ApplicationService
    ) -> None:
        shell, _page = make_shell(service)
        shell.apply_layout_for_width(800.0)
        assert not shell.left_panel.visible and not shell.right_panel.visible
        assert shell.left_toggle.content == "Show panels"
        assert shell.right_toggle.content == "Show inspector"

        shell._on_toggle_left_panel(cast(Any, None))
        shell._on_toggle_right_panel(cast(Any, None))
        assert shell.left_panel.visible and shell.right_panel.visible

        # An explicit choice survives further resizing.
        shell.apply_layout_for_width(700.0)
        assert shell.left_panel.visible, "the user's choice outranks the width"
        assert shell.left_toggle.content == "Hide panels"

    def test_committing_below_the_operator_zoom_does_not_raise(
        self, tmp_path: Path, service: ApplicationService
    ) -> None:
        """Commit used to raise KeyError once the rebuild changed detail."""
        from nneditor.editing.validation import RenameNodeRequest

        shell, session = self.open_shell(tmp_path, service)
        node = session.document.main_graph.nodes[0]
        shell.renderer.set_selection(frozenset({node.id}))
        transaction = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "renamed")
        )
        shell.pending_edit = transaction
        shell.pending_target = node.id
        # Force the post-commit rebuild onto a coarser level, where the scene
        # holds group glyphs rather than this operator.
        shell.auto_detail = False
        shell.current_detail = DetailLevel.ARCHITECTURE

        shell._on_commit_edit(cast(Any, None))

        assert session.document.main_graph.node(node.id).source_name == "renamed"
        assert "committed" in (shell.status_text.value or "")
        assert not shell.error_banner.visible

    def test_a_pending_edit_is_discarded_when_the_selection_moves(
        self, tmp_path: Path, service: ApplicationService
    ) -> None:

        shell, session = self.open_shell(tmp_path, service)
        first, second = session.document.main_graph.nodes
        shell.renderer.set_selection(frozenset({first.id}))
        shell.edit_kind.value = "rename"
        shell.edit_primary.value = "renamed"
        shell._on_validate_edit(cast(Any, None))
        assert shell.pending_edit is not None
        assert shell.pending_target == first.id

        # The user clicks a different node before committing.
        shell.renderer.set_selection(frozenset({second.id}))
        shell._on_selected(frozenset({second.id}))

        assert shell.pending_edit is None, "the stale transaction is gone"
        assert shell.commit_edit_button.disabled
        assert "discarded" in (shell.status_text.value or "")
        # Committing now is a no-op rather than renaming the wrong node.
        shell._on_commit_edit(cast(Any, None))
        assert session.document.main_graph.node(first.id).source_name == "scale"


def _pan_event(shell: Shell, dx: float) -> ft.DragUpdateEvent[ft.GestureDetector]:
    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)
    return ft.DragUpdateEvent(
        name="pan_update",
        control=detector,
        local_position=ft.Offset(0.0, 0.0),
        global_position=ft.Offset(0.0, 0.0),
        local_delta=ft.Offset(dx, 0.0),
        timestamp=None,
    )


def test_gesture_events_are_throttled_at_frame_rate(
    service: ApplicationService,
) -> None:
    shell, _page = make_shell(service)
    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)
    # 0 (the Flet default) delivers every pointer move; each one costs a
    # canvas patch, so an unthrottled drag floods the event loop.
    assert detector.drag_interval == 16


def test_pan_moves_the_minimap_rectangle_without_rebuilding_dots(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))

    shapes = shell.minimap_canvas.shapes
    assert shapes, "the minimap draws after a session is shown"
    first_dot = shapes[0]
    rect = shell._minimap_view_rect
    assert rect is not None and rect is shapes[-1]
    assert rect.width > 0, "a shown session has a viewport to draw"
    x_before = rect.x

    shell._on_pan(cast(Any, _pan_event(shell, -50.0)))

    # Rebuilding these per pan event was the dominant interaction cost: one
    # fresh control per node, re-diffed by Flet on every pointer move.
    assert shell.minimap_canvas.shapes is shapes
    assert shell.minimap_canvas.shapes[0] is first_dot
    assert shell._minimap_view_rect is rect
    assert rect.x > x_before, "the viewport rectangle tracked the pan"


def test_minimap_dot_count_is_bounded(
    tmp_path: Path, service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nneditor.ui import app as app_module

    monkeypatch.setattr(app_module, "_MINIMAP_MAX_DOTS", 1)
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    # One sampled dot plus the persistent viewport rectangle.
    assert len(shell.minimap_canvas.shapes) <= 2


class DeferringTaskPage(StubPage):
    """A stub page whose run_task defers coroutines like a real event loop."""

    def __init__(self) -> None:
        super().__init__()
        self.scheduled: list[tuple[Any, tuple[Any, ...], Future[Any]]] = []

    def run_task(self, handler: Any, *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        self.scheduled.append((handler, args, future))
        return future

    def drain(self) -> None:
        while self.scheduled:
            handler, args, future = self.scheduled.pop(0)
            asyncio.run(handler(*args))
            future.set_result(None)


def test_pan_bursts_coalesce_into_one_apply(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    page = DeferringTaskPage()
    shell = Shell(cast(ft.Page, page), service)
    shell.build()
    shell.show_session(service.open_model(path))

    applies = 0
    original = shell._apply_viewport

    def counting_apply() -> None:
        nonlocal applies
        applies += 1
        original()

    shell._apply_viewport = counting_apply  # type: ignore[method-assign]
    scale = shell.view["scale"]
    x_before = shell.view["x"]

    for _ in range(3):
        shell._on_pan(cast(Any, _pan_event(shell, -50.0)))

    # The burst recorded three viewport targets but scheduled one drain.
    assert len(page.scheduled) == 1
    assert applies == 0
    page.drain()
    # The drain applied the latest viewport once, not once per event.
    assert applies == 1
    assert shell.view["x"] == pytest.approx(x_before + 150.0 / scale)


# -- zoom drill-through and blank-canvas recovery ---------------------------


def _scroll_event(
    shell: Shell, cursor: ft.Offset, delta_y: float
) -> ft.ScrollEvent[ft.GestureDetector]:
    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)
    return ft.ScrollEvent(
        name="scroll",
        control=detector,
        local_position=cursor,
        global_position=cursor,
        scroll_delta=ft.Offset(0.0, delta_y),
    )


def test_drill_helpers_measure_glyph_screen_coverage() -> None:
    from nneditor.ui import app as app_module

    assert app_module.next_detail_level(DetailLevel.ARCHITECTURE) is DetailLevel.BLOCK
    assert app_module.next_detail_level(DetailLevel.BLOCK) is DetailLevel.LAYER
    assert app_module.next_detail_level(DetailLevel.LAYER) is DetailLevel.OPERATOR
    assert app_module.next_detail_level(DetailLevel.OPERATOR) is None

    # A 20,000-world-unit region at the huge-model fit floor already covers a
    # third of the surface although the scale is far below every absolute
    # detail threshold.
    coverage = app_module.glyph_screen_coverage(20000.0, 400.0, 0.02, 1200.0, 800.0)
    assert coverage == pytest.approx(400.0 / 1200.0)
    assert app_module.glyph_screen_coverage(10.0, 10.0, 1.0, 0.0, 0.0) == 0.0

    from nneditor.rendering.scene import NodeGlyph

    near = NodeGlyph(id="a", x=0.0, y=0.0, width=10.0, height=10.0, kind="n", label="a")
    far = NodeGlyph(
        id="b", x=100.0, y=100.0, width=10.0, height=10.0, kind="n", label="b"
    )
    assert app_module.nearest_glyph_center((near, far), 8.0, 8.0) == (5.0, 5.0)


def _manual_block_shell(
    tmp_path: Path, service: ApplicationService
) -> tuple[Shell, str]:
    """A shell at BLOCK detail with every operator inside one manual group."""
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node_ids = frozenset(node.id for node in session.document.main_graph.nodes)
    shell.renderer.set_selection(node_ids)
    shell._on_selected(node_ids)
    shell.group_label_field.value = "Manual block"
    shell._on_group_selected(cast(Any, None))
    manual = next(
        group for group in session.graph_hierarchy().groups if not group.automatic
    )
    return shell, manual.id


def test_zoom_into_a_dominant_group_glyph_drills_through(
    tmp_path: Path, service: ApplicationService
) -> None:
    """A screen-filling group advances the detail level below the absolute
    scale thresholds, anchored at the cursor."""
    shell, group_id = _manual_block_shell(tmp_path, service)
    assert shell.current_detail is DetailLevel.BLOCK
    shell.auto_detail = True
    shell._sync_detail_controls()
    scene = shell.renderer.scene
    assert scene is not None
    glyph = scene.node(group_id)

    # A small surface makes the block glyph dominant on screen while the
    # scale stays far below the absolute LAYER threshold (0.35).
    shell.surface_size = (64.0, 40.0)
    scale = 0.2
    center_x = glyph.x + glyph.width / 2.0
    center_y = glyph.y + glyph.height / 2.0
    shell.view = {
        "scale": scale,
        "x": center_x - 32.0 / scale,
        "y": center_y - 20.0 / scale,
    }
    shell.renderer.set_viewport(shell._current_viewport())
    cursor = ft.Offset(32.0, 20.0)
    anchor = (
        shell.view["x"] + cursor.x / shell.view["scale"],
        shell.view["y"] + cursor.y / shell.view["scale"],
    )

    shell._on_scroll(cast(Any, _scroll_event(shell, cursor, -10.0)))

    # A fresh read defeats mypy's literal narrowing from the setup assert.
    landed: DetailLevel = shell.current_detail
    assert landed is DetailLevel.LAYER
    assert shell.auto_detail
    assert shell.view["scale"] == pytest.approx(scale * 1.1)
    assert (
        shell.view["x"] + cursor.x / shell.view["scale"],
        shell.view["y"] + cursor.y / shell.view["scale"],
    ) == pytest.approx(anchor)


def test_zoom_over_empty_space_does_not_drill(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, group_id = _manual_block_shell(tmp_path, service)
    shell.auto_detail = True
    shell._sync_detail_controls()
    scene = shell.renderer.scene
    assert scene is not None
    glyph = scene.node(group_id)

    shell.surface_size = (64.0, 40.0)
    scale = 0.2
    # Anchor just above the glyph: the cursor hits nothing, but the glyph
    # stays inside the viewport so no blank-canvas recovery fires either.
    anchor_x = glyph.x + glyph.width / 2.0
    anchor_y = glyph.y - 30.0
    shell.view = {
        "scale": scale,
        "x": anchor_x - 32.0 / scale,
        "y": anchor_y - 20.0 / scale,
    }
    shell.renderer.set_viewport(shell._current_viewport())

    shell._on_scroll(cast(Any, _scroll_event(shell, ft.Offset(32.0, 20.0), -10.0)))

    assert shell.current_detail is DetailLevel.BLOCK
    assert shell.view["scale"] == pytest.approx(scale * 1.1)


def test_blank_viewport_recenters_on_the_nearest_glyph(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell.auto_detail = False

    shell.view = {"scale": 1.0, "x": 500000.0, "y": 500000.0}
    shell._apply_viewport()

    assert shell.current_detail is DetailLevel.ARCHITECTURE
    assert shell.renderer.stats.visible_nodes > 0
    assert shell.view["x"] < 500000.0


def test_blank_viewport_advances_detail_once_under_auto(
    tmp_path: Path, service: ApplicationService
) -> None:
    """One recovery per gesture: a deeper level is tried, then the viewport
    is clamped onto a glyph; the recovery never loops down to OPERATOR."""
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell.auto_detail = True
    assert shell.current_detail is DetailLevel.ARCHITECTURE

    shell.view = {"scale": 0.05, "x": 500000.0, "y": 500000.0}
    shell._apply_viewport()

    recovered: DetailLevel = shell.current_detail
    assert recovered is DetailLevel.BLOCK
    assert shell.auto_detail
    assert shell.renderer.stats.visible_nodes > 0


def test_recovery_ignores_an_intentionally_empty_scene(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell._on_close_model(cast(Any, None))

    shell._apply_viewport()

    assert shell.renderer.stats.visible_nodes == 0


# -- keyboard shortcuts ------------------------------------------------------


def test_ctrl_o_opens_the_model_picker(
    service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell, _page = make_shell(service)
    calls: list[dict[str, Any]] = []

    async def pick_files(*args: Any, **kwargs: Any) -> list[ft.FilePickerFile]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(type(shell.picker), "pick_files", pick_files)
    press_key(shell, "O", ctrl=True)

    assert len(calls) == 1


def _commit_rename(shell: Shell, session: Any, name: str) -> Any:
    from nneditor.editing.validation import RenameNodeRequest

    node = session.document.main_graph.nodes[0]
    shell.renderer.set_selection(frozenset({node.id}))
    shell.pending_edit = session.prepare_edit(
        RenameNodeRequest(session.document.entry_graph, node.id, name)
    )
    shell.pending_target = node.id
    shell._on_commit_edit(cast(Any, None))
    return node


def test_ctrl_s_saves_only_when_the_action_is_enabled(
    tmp_path: Path, service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.onnx"
    destination = tmp_path / "saved.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)

    async def choose_destination(*args: Any, **kwargs: Any) -> Path:
        return destination

    monkeypatch.setattr(type(shell.picker), "save_file", choose_destination)

    # No unsaved revision: the Save action is hidden, so Ctrl+S must not export.
    press_key(shell, "S", ctrl=True)
    assert not destination.exists()

    _commit_rename(shell, session, "renamed by shortcut")
    assert shell.save_model_button.visible

    press_key(shell, "S", ctrl=True)
    assert destination.is_file()
    assert "Exported" in (shell.status_text.value or "")


def test_ctrl_z_and_ctrl_y_undo_and_redo(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node = _commit_rename(shell, session, "renamed")
    assert session.document.main_graph.node(node.id).source_name == "renamed"

    press_key(shell, "Z", ctrl=True)
    assert session.document.main_graph.node(node.id).source_name == "scale"

    press_key(shell, "Y", ctrl=True)
    assert session.document.main_graph.node(node.id).source_name == "renamed"


def test_ctrl_f_focuses_the_search_field(
    service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell, _page = make_shell(service)
    focused: list[bool] = []

    async def focus() -> None:
        focused.append(True)

    monkeypatch.setattr(shell.search_field, "focus", focus)
    press_key(shell, "F", ctrl=True)

    assert focused == [True]


def test_escape_clears_the_selection_when_no_overlay_is_open(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    scene = shell.renderer.scene
    assert scene is not None
    shell.renderer.set_selection(frozenset({scene.nodes[0].id}))

    press_key(shell, "Escape")

    assert shell.renderer.selection == frozenset()


def test_plus_and_minus_zoom_about_the_viewport_center(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    width, height = shell.surface_size
    scale = shell.view["scale"]
    center = (
        shell.view["x"] + (width / 2.0) / scale,
        shell.view["y"] + (height / 2.0) / scale,
    )

    press_key(shell, "+")
    assert shell.view["scale"] == pytest.approx(scale * 1.1)
    assert (
        shell.view["x"] + (width / 2.0) / shell.view["scale"],
        shell.view["y"] + (height / 2.0) / shell.view["scale"],
    ) == pytest.approx(center)

    press_key(shell, "-")
    assert shell.view["scale"] == pytest.approx(scale * 1.1 * 0.9)


def test_zero_fits_the_graph(tmp_path: Path, service: ApplicationService) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    fitted_view = dict(shell.view)

    shell.view = {"scale": 3.5, "x": 750.0, "y": -240.0}
    shell.renderer.set_viewport(shell._current_viewport())
    press_key(shell, "0")

    assert shell.view == pytest.approx(fitted_view)


def test_shortcuts_yield_to_a_focused_text_field(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node = _commit_rename(shell, session, "renamed")
    scale = shell.view["scale"]
    shell._text_input_active = True

    press_key(shell, "Z", ctrl=True)
    press_key(shell, "+")

    assert session.document.main_graph.node(node.id).source_name == "renamed"
    assert shell.view["scale"] == pytest.approx(scale)


# -- operations drawer and information sections ------------------------------


def test_operation_buttons_toggle_swap_and_close_the_drawer(
    service: ApplicationService,
) -> None:
    shell, _page = make_shell(service)
    assert shell.operations_drawer.data == "operations-drawer"
    assert not shell.operations_drawer.visible
    trace_button = shell.operation_buttons["trace"]
    edit_button = shell.operation_buttons["edit"]
    optimize_button = shell.operation_buttons["optimize"]
    assert trace_button.data == "operation:trace"
    assert edit_button.data == "operation:edit"
    assert optimize_button.data == "operation:optimize"

    cast(Any, trace_button.on_click)(cast(Any, None))
    assert shell.operations_drawer.visible
    assert shell.drawer_body.content is shell.trace.control
    assert shell.drawer_title.value == "Trace activations"
    assert trace_button.style is not None
    assert trace_button.style.color == shell.palette.accent

    # Opening a second operation swaps the drawer content in place.
    cast(Any, edit_button.on_click)(cast(Any, None))
    assert shell.operations_drawer.visible
    assert shell.drawer_body.content is shell._edit_controls
    assert shell.drawer_title.value == "Edit model"
    assert trace_button.style is not None
    assert trace_button.style.color == shell.palette.ink
    assert edit_button.style is not None
    assert edit_button.style.color == shell.palette.accent

    # Toggling the active operation closes the drawer.
    cast(Any, edit_button.on_click)(cast(Any, None))
    assert not shell.operations_drawer.visible
    assert shell.drawer_body.content is None
    assert edit_button.style is not None
    assert edit_button.style.color == shell.palette.ink

    cast(Any, optimize_button.on_click)(cast(Any, None))
    assert shell.drawer_body.content is shell._transformation_controls
    assert shell.drawer_title.value == "Optimize weights"
    cast(Any, shell.drawer_close_button.on_click)(cast(Any, None))
    assert not shell.operations_drawer.visible


def test_operations_drawer_survives_a_theme_rebuild(
    service: ApplicationService,
) -> None:
    shell, _page = make_shell(service)
    cast(Any, shell.operation_buttons["optimize"].on_click)(cast(Any, None))
    assert shell.operations_drawer.visible

    shell._on_toggle_theme(None)

    assert shell.operations_drawer.visible
    assert shell.drawer_body.content is shell._transformation_controls
    assert shell.operations_drawer.data == "operations-drawer"
    button = shell.operation_buttons["optimize"]
    assert button.style is not None
    assert button.style.color == shell.palette.accent


def test_escape_closes_overlay_then_drawer_then_clears_selection(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    scene = shell.renderer.scene
    assert scene is not None
    shell.renderer.set_selection(frozenset({scene.nodes[0].id}))
    cast(Any, shell.operation_buttons["trace"].on_click)(cast(Any, None))

    # A large activation overlay outranks the drawer and the selection.
    overlay_closes = iter([True, False, False])
    monkeypatch.setattr(
        shell.activations, "close_overlay", lambda: next(overlay_closes)
    )
    press_key(shell, "Escape")
    assert shell.operations_drawer.visible, "the overlay consumed this Escape"
    assert shell.renderer.selection

    press_key(shell, "Escape")
    assert not shell.operations_drawer.visible
    assert shell.renderer.selection, "closing the drawer keeps the selection"

    press_key(shell, "Escape")
    assert shell.renderer.selection == frozenset()


def test_left_panel_sections_carry_markers_and_persist_expansion(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    column = shell.left_panel.content
    assert isinstance(column, ft.Column)
    assert [tile.data for tile in column.controls] == [
        "info:model",
        "info:selection",
        "info:activations",
    ]
    assert all(isinstance(tile, ft.ExpansionTile) for tile in column.controls), (
        "the left panel is a column of collapsible information sections"
    )

    session = service.open_model(path)
    shell.show_session(session)
    assert shell.info_model_tile.expanded, "no selection: Model starts open"
    assert shell.info_selection_tile.expanded
    assert not shell.info_activations_tile.expanded, "no trace exists yet"

    node_id = session.document.main_graph.nodes[0].id
    shell._refresh_inspector(frozenset({node_id}))
    assert not shell.info_model_tile.expanded, (
        "a selection collapses the Model section by default"
    )
    assert shell.info_selection_tile.expanded

    # An explicit choice outranks the default and survives refreshes.
    assert shell.info_model_tile.on_change is not None
    cast(Any, shell.info_model_tile.on_change)(SimpleNamespace(data=True))
    shell._refresh_inspector(frozenset({node_id}))
    assert shell.info_model_tile.expanded

    # ... and a theme rebuild, which replaces the tile objects.
    previous_tile = shell.info_model_tile
    shell._on_toggle_theme(None)
    assert shell.info_model_tile is not previous_tile
    assert shell.info_model_tile.data == "info:model"
    assert shell.info_model_tile.expanded, "the stored choice survived"
    assert not shell.info_activations_tile.expanded

    # A collapse choice persists the same way.
    cast(Any, shell.info_selection_tile.on_change)(SimpleNamespace(data=False))
    shell._refresh_inspector(frozenset())
    assert not shell.info_selection_tile.expanded


def test_left_panel_hosts_no_operation_controls(
    service: ApplicationService,
) -> None:
    shell, _page = make_shell(service)

    def walk(control: Any) -> list[Any]:
        found = [control]
        for attribute in ("leading", "title", "content"):
            child = getattr(control, attribute, None)
            if isinstance(child, ft.Control):
                found.extend(walk(child))
        for child in getattr(control, "controls", None) or []:
            found.extend(walk(child))
        return found

    mounted = {id(control) for control in walk(shell.left_panel)}
    assert id(shell.trace.control) not in mounted
    assert id(shell._edit_controls) not in mounted
    assert id(shell._transformation_controls) not in mounted
    assert id(shell.watch.control) in mounted, "the watch strip is information"


# -- dark mode ---------------------------------------------------------------


def test_dark_palette_provides_a_distinct_counterpart_for_every_color() -> None:
    import dataclasses

    dark = shell_layout.DARK_SHELL_PALETTE
    for field in dataclasses.fields(shell_layout.ShellPalette):
        light_value = getattr(SHELL_PALETTE, field.name)
        dark_value = getattr(dark, field.name)
        if not isinstance(light_value, str):
            assert dark_value == light_value
            continue
        assert dark_value.startswith("#")
        assert dark_value != light_value, field.name
    # The core readability pairing holds: dark ink on a dark panel would be
    # unreadable, so ink must be a light tone and the panel a dark one.
    assert int(dark.ink.removeprefix("#"), 16) > int(dark.panel.removeprefix("#"), 16)


def test_theme_toggle_switches_palette_and_rebuilds_the_chrome(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, page = make_shell(service)
    shell.show_session(service.open_model(path))
    # An untyped view of the stub defeats both attr-defined noise and the
    # literal narrowing that would make later `is` checks non-overlapping.
    stub = cast(Any, page)
    assert shell.palette is SHELL_PALETTE
    assert stub.theme_mode is ft.ThemeMode.LIGHT

    shell._on_toggle_theme(None)

    dark = shell_layout.DARK_SHELL_PALETTE
    assert shell.palette is dark
    # A second untyped view: mypy pins member narrowing to the first `stub`
    # even across the toggle call.
    toggled = cast(Any, page)
    assert toggled.theme_mode is ft.ThemeMode.DARK
    assert toggled.bgcolor == dark.canvas
    assert shell.left_panel.bgcolor == dark.panel
    assert shell.right_panel.bgcolor == dark.panel
    assert shell.surface.bgcolor == dark.canvas
    assert shell.title_text.color == dark.ink
    assert shell.status_text.color == dark.muted
    assert shell.inspector_title.color == dark.ink
    assert shell.activations.palette is dark
    assert shell.trace.palette is dark
    # The open model stayed live through the rebuild.
    assert not shell.empty_state.visible
    assert shell.renderer.scene is not None
    assert shell.hierarchy_list.controls
    assert "dark mode" in (shell.status_text.value or "")

    # The left panel's Model-section content re-rendered with dark colors:
    # metadata labels/values bake their palette at build time, so stale light
    # ink here would be invisible on the dark panel.
    def texts(control: Any) -> list[ft.Text]:
        found = [control] if isinstance(control, ft.Text) else []
        for attribute in ("leading", "title", "content"):
            child = getattr(control, attribute, None)
            if isinstance(child, ft.Control):
                found.extend(texts(child))
        for child in getattr(control, "controls", None) or []:
            found.extend(texts(child))
        return found

    def artifact_colors() -> set[str | None]:
        section = next(
            control
            for control in shell.model_info.controls
            if control.data == "model-overview-artifact"
        )
        return {text.color for text in texts(section)}

    assert dark.ink in artifact_colors(), "metadata values use dark ink"
    assert dark.muted in artifact_colors(), "metadata labels use dark muted"
    model_info_colors = {
        text.color for control in shell.model_info.controls for text in texts(control)
    }
    assert SHELL_PALETTE.ink not in model_info_colors
    assert SHELL_PALETTE.muted not in model_info_colors
    # The Activations section's watch strip bakes its colors the same way.
    watch_hint = shell.watch.strip.controls[0]
    assert isinstance(watch_hint, ft.Text)
    assert watch_hint.color == dark.muted

    shell._on_toggle_theme(None)
    assert shell.palette is SHELL_PALETTE
    assert stub.theme_mode is ft.ThemeMode.LIGHT
    assert shell.left_panel.bgcolor == SHELL_PALETTE.panel
    assert SHELL_PALETTE.ink in artifact_colors(), "light ink is restored"
    assert dark.ink not in artifact_colors()
    watch_hint = shell.watch.strip.controls[0]
    assert isinstance(watch_hint, ft.Text)
    assert watch_hint.color == SHELL_PALETTE.muted


def test_theme_toggle_preserves_panel_visibility_choices(
    service: ApplicationService,
) -> None:
    shell, _page = make_shell(service)
    shell._on_toggle_left_panel(cast(Any, None))
    assert not shell.left_panel.visible

    shell._on_toggle_theme(None)

    assert not shell.left_panel.visible, "the rebuild honors the user's choice"
    assert shell.right_panel.visible


def test_initial_theme_follows_the_platform_brightness(
    service: ApplicationService,
) -> None:
    from nneditor.ui.app import resolve_initial_palette

    assert resolve_initial_palette(None) is SHELL_PALETTE
    assert (
        resolve_initial_palette(ft.Brightness.DARK) is shell_layout.DARK_SHELL_PALETTE
    )
    assert resolve_initial_palette(ft.Brightness.LIGHT) is SHELL_PALETTE

    page = StubPage()
    page.platform_brightness = ft.Brightness.DARK  # type: ignore[attr-defined]
    shell = Shell(cast(ft.Page, page), service)
    assert shell.palette is shell_layout.DARK_SHELL_PALETTE
    assert page.theme_mode is ft.ThemeMode.DARK  # type: ignore[attr-defined]


def test_node_details_json_collects_repeated_keys() -> None:
    import json

    from nneditor.ui import viewmodel

    payload = json.loads(
        viewmodel.node_details_json(
            (
                ("Operator", "Add"),
                ("Subgraph", "g:one"),
                ("Subgraph", "g:two"),
            )
        )
    )
    assert payload == {"Operator": "Add", "Subgraph": ["g:one", "g:two"]}


def test_copy_as_json_puts_node_metadata_on_the_clipboard(
    tmp_path: Path, service: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node = session.document.main_graph.nodes[0]
    shell.current_detail = DetailLevel.OPERATOR
    shell._show_graph(session.document.entry_graph)
    shell.renderer.set_selection(frozenset({node.id}))
    shell._refresh_inspector(frozenset({node.id}))

    button = next(
        control
        for control in shell.inspector.controls
        if isinstance(control, ft.TextButton)
        and control.data == f"copy-node-json:{node.id}"
    )
    copied: list[str] = []

    async def set_clipboard(clipboard: Any, value: str) -> None:
        copied.append(value)

    monkeypatch.setattr(ft.Clipboard, "set", set_clipboard)
    cast(Any, button.on_click)(cast(Any, SimpleNamespace()))

    assert len(copied) == 1
    payload = json.loads(copied[0])
    assert payload["Operator"] == node.qualified_op_type
    assert payload["Node id"] == node.id
    assert "copied" in (shell.status_text.value or "")


# -- ordered, collapsible blocks tree ---------------------------------------


def _build_identity_chain_model(path: Path, *, count: int = 8) -> None:
    """A flat chain of unscoped Identity ops.

    No detector finds structure in unscoped Identity nodes under SOURCE
    organization, so the Blocks pane shows exactly the manual groups a test
    creates.
    """
    nodes = [
        helper.make_node("Identity", [f"v{index}"], [f"v{index + 1}"], name=f"n{index}")
        for index in range(count)
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="chain",
        inputs=[helper.make_tensor_value_info("v0", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info(f"v{count}", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(model, path)


def _chain_shell(
    tmp_path: Path, service: ApplicationService
) -> tuple[Shell, ModelSession]:
    path = tmp_path / "chain.onnx"
    _build_identity_chain_model(path)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    session.hierarchy.set_mode(OrganizationMode.SOURCE)
    return shell, session


def _row_button(control: ft.Control) -> ft.TextButton | None:
    """The select button of a Blocks row, whether plain or chevroned."""
    if isinstance(control, ft.Container) and control.content is not None:
        # Highlighted rows are wrapped in a selection-wash container.
        control = control.content
    if isinstance(control, ft.TextButton):
        return control
    if isinstance(control, ft.Row):
        return next(
            (item for item in control.controls if isinstance(item, ft.TextButton)),
            None,
        )
    return None


def _selected_markers(shell: Shell) -> list[str]:
    """The hierarchy-selected data markers carried by highlighted rows."""
    return [
        control.data
        for control in shell.hierarchy_list.controls
        if isinstance(control, ft.Container)
        and isinstance(control.data, str)
        and control.data.startswith("hierarchy-selected:")
    ]


def _row_labels(shell: Shell) -> list[str]:
    labels: list[str] = []
    for control in shell.hierarchy_list.controls:
        button = _row_button(control)
        if button is not None:
            labels.append(str(button.content).split("  ·  ")[0])
    return labels


def _row_toggle(shell: Shell, label: str) -> ft.IconButton:
    for control in shell.hierarchy_list.controls:
        button = _row_button(control)
        if button is None or str(button.content).split("  ·  ")[0] != label:
            continue
        assert isinstance(control, ft.Row), f"row {label!r} has no chevron"
        return next(
            item for item in control.controls if isinstance(item, ft.IconButton)
        )
    raise AssertionError(f"no Blocks row labelled {label!r}")


def test_blocks_pane_orders_siblings_by_graph_position(
    tmp_path: Path, service: ApplicationService
) -> None:
    from nneditor.ui import app as app_module

    shell, session = _chain_shell(tmp_path, service)
    graph = session.document.main_graph
    pairs = [
        frozenset({graph.nodes[index].id, graph.nodes[index + 1].id})
        for index in range(0, 8, 2)
    ]
    # Created in a deliberately shuffled order. Member counts tie and both
    # insertion and lexicographic id order would put "block 10" before
    # "block 2", so only the graph position can produce the expected rows.
    session.hierarchy.group(graph.id, "block 10", pairs[3])
    session.hierarchy.group(graph.id, "block 2", pairs[1])
    session.hierarchy.group(graph.id, "block 1", pairs[0])
    session.hierarchy.group(graph.id, "block 3", pairs[2])

    shell._refresh_hierarchy()

    assert _row_labels(shell) == ["block 1", "block 2", "block 3", "block 10"]
    # The label tiebreak is numeric-aware, not lexicographic.
    assert app_module._natural_key("block 2") < app_module._natural_key("block 10")


def test_blocks_tree_collapses_deeper_levels_until_toggled(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, session = _chain_shell(tmp_path, service)
    graph = session.document.main_graph
    ids = [node.id for node in graph.nodes]
    session.hierarchy.group(graph.id, "Stage", frozenset(ids))
    session.hierarchy.group(graph.id, "Front", frozenset(ids[:4]))
    session.hierarchy.group(graph.id, "Core", frozenset(ids[:2]))

    shell._refresh_hierarchy()

    # Roots are expanded by default; deeper levels start collapsed.
    assert _row_labels(shell) == ["Stage", "Front"]
    toggle = _row_toggle(shell, "Front")
    assert toggle.icon == ft.Icons.CHEVRON_RIGHT_ROUNDED

    cast(Any, toggle.on_click)(cast(Any, None))

    assert _row_labels(shell) == ["Stage", "Front", "Core"]
    assert _row_toggle(shell, "Front").icon == ft.Icons.EXPAND_MORE_ROUNDED

    # Expansion survives inspector refreshes and a theme toggle.
    shell._refresh_inspector(frozenset())
    shell._on_toggle_theme(None)
    assert _row_labels(shell) == ["Stage", "Front", "Core"]

    # Closing the model clears the per-session expansion state.
    shell._on_close_model(cast(Any, None))
    assert shell._hierarchy_expanded == {}
    assert shell.hierarchy_list.controls == []


def test_blocks_row_cap_counts_only_visible_rows(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nneditor.ui import app as app_module

    shell, session = _chain_shell(tmp_path, service)
    graph = session.document.main_graph
    ids = [node.id for node in graph.nodes]
    session.hierarchy.group(graph.id, "Stage", frozenset(ids))
    session.hierarchy.group(graph.id, "Front", frozenset(ids[:4]))
    session.hierarchy.group(graph.id, "Back", frozenset(ids[4:]))
    session.hierarchy.group(graph.id, "Front head", frozenset(ids[:2]))
    session.hierarchy.group(graph.id, "Front tail", frozenset(ids[2:4]))
    monkeypatch.setattr(app_module, "_MAX_EXPLORER_ROWS", 3)

    shell._refresh_hierarchy()

    # Five groups exist but only three rows are visible, so the cap is not
    # reached and no truncation note appears.
    assert _row_labels(shell) == ["Stage", "Front", "Back"]
    assert not any(
        isinstance(control, ft.Text) for control in shell.hierarchy_list.controls
    )

    cast(Any, _row_toggle(shell, "Front").on_click)(cast(Any, None))

    # Expanding Front makes five rows visible; the cap trims the list and
    # notes the remainder.
    assert _row_labels(shell) == ["Stage", "Front", "Front head"]
    note = shell.hierarchy_list.controls[-1]
    assert isinstance(note, ft.Text)
    assert "2 more…" in str(note.value)


def test_blocks_tree_highlights_deepest_group_for_selection(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, session = _chain_shell(tmp_path, service)
    graph = session.document.main_graph
    ids = [node.id for node in graph.nodes]
    session.hierarchy.group(graph.id, "Stage", frozenset(ids))
    front = session.hierarchy.group(graph.id, "Front", frozenset(ids[:4]))
    core = session.hierarchy.group(graph.id, "Core", frozenset(ids[:2]))
    shell.detail_segment.selected = [DetailLevel.OPERATOR.value]
    shell._on_detail_segment_changed(cast(Any, None))
    assert _row_labels(shell) == ["Stage", "Front"]

    shell.renderer.set_selection(frozenset({ids[0]}))
    shell._on_selected(frozenset({ids[0]}))

    # The deepest containing group lights up; its collapsed ancestors were
    # opened so the highlighted row is visible, but stay unhighlighted.
    assert _row_labels(shell) == ["Stage", "Front", "Core"]
    assert shell._hierarchy_expanded[front.id] is True
    assert _selected_markers(shell) == [f"hierarchy-selected:{core.id}"]
    core_button = _row_button(shell.hierarchy_list.controls[2])
    assert core_button is not None
    assert core_button.style is not None
    assert core_button.style.color == shell.palette.accent

    # Selecting another node of the same deepest group keeps the highlight
    # unchanged and must not rebuild the rows.
    row_before = shell.hierarchy_list.controls[2]
    shell.renderer.set_selection(frozenset({ids[1]}))
    shell._on_selected(frozenset({ids[1]}))
    assert shell.hierarchy_list.controls[2] is row_before

    # Clearing the selection removes the highlight; expansion persists.
    shell.renderer.set_selection(frozenset())
    shell._on_selected(frozenset())
    assert _selected_markers(shell) == []
    assert shell._hierarchy_highlight == frozenset()
    assert _row_labels(shell) == ["Stage", "Front", "Core"]


def test_blocks_tree_highlights_selected_group_glyph(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, session = _chain_shell(tmp_path, service)
    graph = session.document.main_graph
    ids = [node.id for node in graph.nodes]
    session.hierarchy.group(graph.id, "Stage", frozenset(ids))
    front = session.hierarchy.group(graph.id, "Front", frozenset(ids[:4]))
    shell.detail_segment.selected = [DetailLevel.BLOCK.value]
    shell._on_detail_segment_changed(cast(Any, None))
    scene = shell.renderer.scene
    assert scene is not None and scene.has_node(front.id)

    shell.renderer.set_selection(frozenset({front.id}))
    shell._on_selected(frozenset({front.id}))

    assert _row_labels(shell) == ["Stage", "Front"]
    assert _selected_markers(shell) == [f"hierarchy-selected:{front.id}"]
    wash = shell.hierarchy_list.controls[1]
    assert isinstance(wash, ft.Container)
    assert wash.bgcolor == shell.palette.accent_soft

    # The highlight recolors from the live palette on a theme switch.
    shell._on_toggle_theme(None)
    assert shell.palette is shell_layout.DARK_SHELL_PALETTE
    wash = shell.hierarchy_list.controls[1]
    assert isinstance(wash, ft.Container)
    assert wash.bgcolor == shell_layout.DARK_SHELL_PALETTE.accent_soft
    assert _selected_markers(shell) == [f"hierarchy-selected:{front.id}"]

    # Closing the model clears the highlight with the rest of the tree state.
    shell._on_close_model(cast(Any, None))
    assert shell._hierarchy_highlight == frozenset()
    assert shell.hierarchy_list.controls == []
