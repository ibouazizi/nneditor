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
import pytest

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from nneditor.artifact_formats import MODEL_FILE_EXTENSIONS
from nneditor.rendering import (
    InteractiveGraphRenderer,
    SelectionCallback,
    create_flet_renderer,
)
from nneditor.ui.app import Shell
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
    assert shell.inspector.controls, "the inspector shows the model summary"
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
        for control in shell.inspector.controls
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

    prompt = shell._preview_row(bias_id)
    assert isinstance(prompt, ft.TextButton)
    assert "full typed tensor" in str(prompt.content)
    assert session.store.open_file_count == 0

    shell._typed_preview_handler(bias_id)(cast(Any, None))
    preview = shell._preview_row(bias_id)
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
