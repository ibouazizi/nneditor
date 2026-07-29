"""Regression tests for the shell defect sweep.

Each test pins one verified finding: session lifecycle on reopen, the open
worker's failure path, the zoom-linked Auto detail level, stale search
results, inspector staleness after commit/undo, keyboard focus handling,
export cancellation, web-upload interleaving, statistics-job completion,
search focus across drilled-in blocks, hierarchy root repair, tensor-card
expansion state, and the small cosmetic fixes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.analysis.lod import DetailLevel
from nneditor.application.editing import SidecarPersistenceError
from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService
from nneditor.ui import tensor_tools
from tests.fixtures.onnx_models import build_embedded_model
from tests.unit.test_shell import make_shell


@pytest.fixture
def service() -> Iterator[ApplicationService]:
    with ApplicationService() as instance:
        yield instance


def _build_chain_model(path: Path) -> None:
    """Two Mul+Add pairs, so a drilled-in block can exclude operators."""

    def _tensor(name: str) -> onnx.TensorProto:
        values = np.ones(4, dtype=np.float32)
        return helper.make_tensor(
            name, TensorProto.FLOAT, [4], values.tobytes(), raw=True
        )

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Mul", ["input", "w0"], ["a"], name="m0"),
            helper.make_node("Add", ["a", "b0"], ["b"], name="a0"),
            helper.make_node("Mul", ["b", "w1"], ["c"], name="m1"),
            helper.make_node("Add", ["c", "b1"], ["output"], name="a1"),
        ],
        name="chain",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        initializer=[_tensor("w0"), _tensor("b0"), _tensor("w1"), _tensor("b1")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(model, path)


# -- finding 1: reopen must close the previous session ----------------------


def test_show_session_closes_the_previous_session(
    tmp_path: Path, service: ApplicationService
) -> None:
    path_a = tmp_path / "a.onnx"
    path_b = tmp_path / "b.onnx"
    build_embedded_model(path_a, elements=16)
    build_embedded_model(path_b, elements=16)
    shell, _page = make_shell(service)
    first = service.open_model(path_a)
    shell.show_session(first)
    second = service.open_model(path_b)

    shell.show_session(second)

    assert first.closed, "the replaced session released its file handle"
    assert service.open_sessions == (second,)
    assert shell.session is second

    # Re-showing the current session must never close it.
    shell.show_session(second)
    assert not second.closed


# -- finding 2: the open worker's failure path ------------------------------


def test_open_failure_while_presenting_clears_the_overlay(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, page = make_shell(service)

    def boom(session: Any) -> None:
        raise RuntimeError("presentation exploded")

    shell.show_session = boom  # type: ignore[method-assign]
    job = service.open_model_async(path)
    shell.open_job = job
    shell.loading_overlay.visible = True

    shell._watch_open(job)

    assert not shell.loading_overlay.visible, "the modal overlay always clears"
    assert shell.open_job is None
    assert shell.error_banner.visible
    banner = shell.error_banner.content
    assert isinstance(banner, ft.Text)
    assert "presentation exploded" in (banner.value or "")
    assert page.updates > 0


# -- finding 3: the Auto detail level is reachable and zoom-linked ----------


def test_auto_detail_is_reachable_and_zoom_linked(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    assert "auto" in [segment.value for segment in shell.detail_segment.segments]

    shell.view = {"scale": 1.0, "x": 0.0, "y": 0.0}
    shell.renderer.set_viewport(shell._current_viewport())
    shell.detail_segment.selected = ["auto"]
    shell._on_detail_segment_changed(cast(Any, None))
    assert shell.auto_detail
    assert shell.current_detail is DetailLevel.OPERATOR
    assert shell.detail_segment.selected == ["auto"]

    # Zooming across a threshold switches the semantic level automatically.
    shell.view = {"scale": 0.34, "x": 25.0, "y": 40.0}
    shell.renderer.set_viewport(shell._current_viewport())
    detector = shell.renderer.control
    assert isinstance(detector, ft.GestureDetector)
    shell._on_scroll(
        cast(
            Any,
            ft.ScrollEvent(
                name="scroll",
                control=detector,
                local_position=ft.Offset(300.0, 220.0),
                global_position=ft.Offset(300.0, 220.0),
                scroll_delta=ft.Offset(0.0, -10.0),
            ),
        )
    )
    assert cast(DetailLevel, shell.current_detail) is DetailLevel.LAYER
    assert shell.auto_detail, "the zoom transition keeps Auto engaged"
    assert shell.detail_segment.selected == ["auto"]


# -- finding 4: stale search results and unvalidated jumps ------------------


def test_reopen_clears_search_results_and_validates_jumps(
    tmp_path: Path, service: ApplicationService
) -> None:
    path_a = tmp_path / "a.onnx"
    path_b = tmp_path / "b.onnx"
    build_embedded_model(path_a, elements=16)
    build_embedded_model(path_b, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path_a))
    shell.search_field.value = "shift"
    shell._on_search(cast(Any, None))
    assert shell.search_results.controls

    shell.show_session(service.open_model(path_b))
    assert shell.search_results.controls == []
    assert not shell.search_field.value

    # A jump captured from the old model can no longer wedge navigation.
    entry = shell.current_graph
    shell._jump_handler("graph:gone", "node:gone")(cast(Any, None))
    assert shell.current_graph == entry
    assert "previously opened" in (shell.status_text.value or "")

    shell._show_graph("graph:gone")
    assert shell.current_graph == entry


# -- finding 5: inspector staleness after commit and undo -------------------


def test_inspector_refreshes_after_commit_and_undo(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    shell.detail_segment.selected = [DetailLevel.OPERATOR.value]
    shell._on_detail_segment_changed(cast(Any, None))
    node = session.document.main_graph.nodes[0]
    shell.renderer.set_selection(frozenset({node.id}))
    shell._on_selected(frozenset({node.id}))
    assert shell.inspector_title.value == "scale"

    shell.edit_kind.value = "rename"
    shell.edit_primary.value = "renamed"
    shell._on_validate_edit(cast(Any, None))
    assert shell.pending_edit is not None and shell.pending_edit.ok
    shell._on_commit_edit(cast(Any, None))
    assert shell.inspector_title.value == "renamed", (
        "the inspector shows the committed name"
    )

    shell._on_undo_edit(cast(Any, None))
    assert shell.inspector_title.value == "scale", "the inspector shows the undone name"


def test_sidecar_failure_on_commit_warns_but_keeps_the_edit(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    shell.detail_segment.selected = [DetailLevel.OPERATOR.value]
    shell._on_detail_segment_changed(cast(Any, None))
    node = session.document.main_graph.nodes[0]
    shell.renderer.set_selection(frozenset({node.id}))
    shell._on_selected(frozenset({node.id}))
    shell.edit_kind.value = "rename"
    shell.edit_primary.value = "renamed"
    shell._on_validate_edit(cast(Any, None))
    assert shell.pending_edit is not None

    original = type(session).commit_edit

    def commit_then_fail(self: Any, transaction: Any) -> None:
        # The application layer raises AFTER the revision applied; mimic it.
        original(self, transaction)
        raise SidecarPersistenceError(
            cast(Any, SimpleNamespace(id="rev-1")), RuntimeError("disk full")
        )

    monkeypatch.setattr(type(session), "commit_edit", commit_then_fail)
    shell._on_commit_edit(cast(Any, None))

    # The edit succeeded: the shell must refresh as on success and warn only
    # about durability, never present the applied revision as a failure.
    assert session.document.main_graph.node(node.id).source_name == "renamed"
    assert shell.inspector_title.value == "renamed"
    assert not shell.error_banner.visible
    assert shell.pending_edit is None
    status = shell.status_text.value or ""
    assert "sidecar" in status and "disk full" in status


# -- finding 6: keyboard navigation yields to text inputs -------------------


def test_arrow_keys_yield_to_focused_text_inputs(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell.detail_segment.selected = [DetailLevel.OPERATOR.value]
    shell._on_detail_segment_changed(cast(Any, None))
    scene = shell.renderer.scene
    assert scene is not None
    first = scene.nodes[0]
    shell.renderer.set_selection(frozenset({first.id}))

    event = ft.KeyboardEvent(
        name="keyboard",
        control=shell.page,
        key="Arrow Down",
        shift=False,
        ctrl=False,
        alt=False,
        meta=False,
    )
    assert shell.search_field.on_focus is not None
    assert shell.search_field.on_blur is not None
    cast(Any, shell.search_field.on_focus)(None)
    shell.on_keyboard(event)
    assert shell.renderer.selection == {first.id}, (
        "navigation must not steal the caret from a focused field"
    )

    cast(Any, shell.search_field.on_blur)(None)
    shell.on_keyboard(event)
    assert shell.renderer.selection != {first.id}, "navigation resumes on blur"


# -- finding 7: export cancellation wording ---------------------------------


def test_export_cancel_is_a_status_not_a_web_error(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, page = make_shell(service)
    shell.show_session(service.open_model(path))

    async def cancelled(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(type(shell.picker), "save_file", cancelled)
    asyncio.run(shell._on_export_clicked(cast(Any, None)))
    assert shell.status_text.value == "Export cancelled"
    assert not shell.error_banner.visible

    cast(Any, page).web = True
    asyncio.run(shell._on_export_clicked(cast(Any, None)))
    assert "web client" in (shell.status_text.value or "")


# -- finding 8: web upload interleaving -------------------------------------


def test_a_second_pick_is_blocked_while_an_upload_is_pending(
    tmp_path: Path,
    service: ApplicationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, _page = make_shell(service)
    pending = (tmp_path / "upload.onnx", "first.onnx")
    shell._pending_web_upload = pending

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the picker must not open during a pending upload")

    monkeypatch.setattr(type(shell.picker), "pick_files", fail)
    asyncio.run(shell._on_open_clicked(cast(Any, None)))
    assert shell._pending_web_upload == pending


def test_upload_events_for_another_file_are_ignored(
    tmp_path: Path, service: ApplicationService
) -> None:
    uploaded = tmp_path / "streamed.onnx"
    build_embedded_model(uploaded, elements=16)
    shell, _page = make_shell(service)
    shell._pending_web_upload = (uploaded, "first.onnx")

    shell._on_upload_progress(
        ft.FilePickerUploadEvent(
            name="upload",
            control=shell.picker,
            file_name="second.onnx",
            progress=1.0,
        )
    )

    assert shell.session is None, "another file's completion opens nothing"
    assert shell._pending_web_upload == (uploaded, "first.onnx")


# -- finding 9: statistics-job completion -----------------------------------


def test_stats_completion_skips_a_changed_inspector_target(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=64)
    shell, page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node_id = session.document.main_graph.nodes[0].id
    tensor_id = session.document.main_graph.initializers[0]
    shell._refresh_inspector(frozenset({node_id}))

    deferred: list[Any] = []
    # Defer the completion callback the way a real thread would.
    cast(Any, page).run_thread = deferred.append
    shell._stats_handler(tensor_id, node_id)(cast(Any, None))
    shell._refresh_inspector(frozenset())  # the user moved to the overview
    (wait,) = deferred
    wait()

    assert shell.inspector_title.value == "Model overview", (
        "a finished job must not clobber the current inspector view"
    )
    assert session.statistics(tensor_id) is not None


def test_stats_completion_survives_a_model_reopen(
    tmp_path: Path, service: ApplicationService
) -> None:
    path_a = tmp_path / "a.onnx"
    path_b = tmp_path / "b.onnx"
    build_embedded_model(path_a, elements=64)
    build_embedded_model(path_b, elements=16)
    shell, page = make_shell(service)
    session_a = service.open_model(path_a)
    shell.show_session(session_a)
    node_id = session_a.document.main_graph.nodes[0].id
    tensor_id = session_a.document.main_graph.initializers[0]
    shell._refresh_inspector(frozenset({node_id}))

    deferred: list[Any] = []
    cast(Any, page).run_thread = deferred.append
    shell._stats_handler(tensor_id, node_id)(cast(Any, None))
    shell.show_session(service.open_model(path_b))
    (wait,) = deferred
    wait()  # must neither raise nor touch the new session's inspector

    assert shell.inspector_title.value == "Model overview"
    assert "b.onnx" in (shell.title_text.value or "")


# -- finding 10: search focus outside the drilled-in block ------------------


def test_focus_node_escapes_a_drilled_in_block(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "chain.onnx"
    _build_chain_model(path)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    graph = session.document.main_graph
    front = session.hierarchy.group(
        graph.id, "Front", frozenset({graph.nodes[0].id, graph.nodes[1].id})
    )
    shell.show_group(front.id)
    assert shell.current_root_group == front.id

    outside = graph.nodes[2].id
    shell.focus_node(outside)

    assert shell.current_root_group is None, "the view widened to the graph"
    assert shell.renderer.selection == {outside}
    assert (shell.status_text.value or "").startswith("Focused")


def test_focus_node_misses_report_a_status(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))

    shell.focus_node("node:not-there")

    assert "not part of the current graph" in (shell.status_text.value or "")


# -- finding 11: hierarchy mutations repair the drilled-in root -------------


def test_merge_repairs_a_root_consumed_by_the_merge(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "chain.onnx"
    _build_chain_model(path)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    graph = session.document.main_graph
    front = session.hierarchy.group(
        graph.id, "Front", frozenset({graph.nodes[0].id, graph.nodes[1].id})
    )
    back = session.hierarchy.group(
        graph.id, "Back", frozenset({graph.nodes[2].id, graph.nodes[3].id})
    )
    shell.detail_segment.selected = [DetailLevel.BLOCK.value]
    shell._on_detail_segment_changed(cast(Any, None))
    shell.renderer.set_selection(frozenset({front.id, back.id}))
    shell.current_root_group = front.id  # the user is drilled into Front
    shell.group_label_field.value = "Merged"

    shell._on_merge_selected(cast(Any, None))

    hierarchy = session.graph_hierarchy(graph.id)
    merged = next(group for group in hierarchy.groups if group.label == "Merged")
    assert shell.current_root_group == merged.id
    assert not shell.error_banner.visible


def test_group_repairs_a_missing_drilled_in_root(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "chain.onnx"
    _build_chain_model(path)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    graph = session.document.main_graph
    shell.logical_selection = frozenset({graph.nodes[0].id, graph.nodes[1].id})
    shell.current_root_group = "grp:user:gone"  # removed by an earlier edit
    shell.group_label_field.value = "Front"

    shell._on_group_selected(cast(Any, None))

    assert shell.current_root_group is None
    assert not shell.error_banner.visible
    hierarchy = session.graph_hierarchy(graph.id)
    assert any(group.label == "Front" for group in hierarchy.groups)


# -- finding 12: tensor-card expansion state --------------------------------


def test_tensor_card_expansion_survives_inspector_rebuilds(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=64)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    node_id = session.document.main_graph.nodes[0].id
    tensor_id = session.document.main_graph.initializers[0]
    shell._refresh_inspector(frozenset({node_id}))

    def tile() -> ft.ExpansionTile:
        found = next(
            control
            for control in shell.inspector.controls
            if getattr(control, "data", None) == f"tensor-card:{tensor_id}"
        )
        assert isinstance(found, ft.ExpansionTile)
        return found

    first = tile()
    assert first.expanded, "the first card starts open"
    assert first.on_change is not None
    cast(Any, first.on_change)(SimpleNamespace(data=False))  # user collapses

    # A hex-pager interaction rebuilds the inspector.
    shell._hex_page_handler(tensor_id, node_id, tensor_tools.HEX_PAGE_BYTES)(
        cast(Any, None)
    )

    assert not tile().expanded, "the rebuild keeps the user's collapse"


# -- finding 13: cosmetics --------------------------------------------------


def test_search_rows_name_unnamed_graphs_by_id(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    graph = session.document.main_graph
    graph.name = ""  # simulate an artifact whose graph carries no name

    shell.search_field.value = "shift"
    shell._on_search(cast(Any, None))

    (row,) = shell.search_results.controls
    assert isinstance(row, ft.TextButton)
    assert f"[{graph.id}]" in str(row.content)
    assert "None" not in str(row.content)


def test_offset_parser_matches_its_error_message() -> None:
    assert tensor_tools.parse_offset("16", total_bytes=64) == 16
    assert tensor_tools.parse_offset("0x10", total_bytes=64) == 16
    for rejected in ("0o17", "0b101", "ten"):
        with pytest.raises(ValueError, match="decimal integer or start with 0x"):
            tensor_tools.parse_offset(rejected, total_bytes=64)


def test_restore_view_updates_the_minimap_rectangle(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    store = SessionStateStore(tmp_path / "state")
    with ApplicationService(state_store=store) as first_service:
        shell, _page = make_shell(first_service)
        shell.show_session(first_service.open_model(path))
        shell.view = {"scale": 2.0, "x": 10.0, "y": 20.0}
        shell._apply_viewport()

    with ApplicationService(
        state_store=SessionStateStore(tmp_path / "state")
    ) as second_service:
        shell, _page = make_shell(second_service)
        shell.show_session(second_service.open_model(path))
        assert shell.view == {"scale": 2.0, "x": 10.0, "y": 20.0}
        rect = shell._minimap_view_rect
        viewport = shell.renderer.viewport
        assert rect is not None and viewport is not None
        assert shell.minimap_model is not None
        bounds = shell.minimap_model.project_viewport(viewport)
        assert rect.x == pytest.approx(bounds.min_x)
        assert rect.y == pytest.approx(bounds.min_y)
        assert rect.width == pytest.approx(bounds.width)
        assert rect.height == pytest.approx(bounds.height)
