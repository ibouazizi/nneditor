"""Headless GGUF container explorer wiring in the Flet shell."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flet as ft
import pytest

from nneditor.application.session import ApplicationService
from nneditor.ui.app import Shell
from nneditor.ui.gguf_view import gguf_tokenizer_rows
from tests.unit.test_gguf_view import _tiny
from tests.unit.test_shell import make_shell


@pytest.fixture
def service() -> Any:
    with ApplicationService() as instance:
        yield instance


def _walk(control: ft.Control) -> list[ft.Control]:
    found = [control]
    children = getattr(control, "controls", None)
    if isinstance(children, list):
        for child in children:
            if isinstance(child, ft.Control):
                found.extend(_walk(child))
    content = getattr(control, "content", None)
    if isinstance(content, ft.Control):
        found.extend(_walk(content))
    for attribute in ("title", "subtitle", "leading", "trailing"):
        child = getattr(control, attribute, None)
        if isinstance(child, ft.Control):
            found.extend(_walk(child))
    return found


def _all_controls(shell: Shell) -> list[ft.Control]:
    found: list[ft.Control] = []
    for control in shell.model_info.controls:
        found.extend(_walk(control))
    for control in shell.inspector.controls:
        found.extend(_walk(control))
    found.extend(_walk(shell.gguf_panel.root))
    return found


def _by_data(shell: Shell, data: str) -> ft.Control:
    return next(control for control in _all_controls(shell) if control.data == data)


def test_gguf_overview_shows_container_explorer(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = _tiny(tmp_path / "model.gguf")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))

    sections = {
        control.data: control
        for control in shell.model_info.controls
        if isinstance(control.data, str)
    }
    assert "model-overview-summary" in sections
    assert "gguf-explorer" in sections
    metrics = next(
        control
        for control in _walk(sections["model-overview-summary"])
        if isinstance(control, ft.ResponsiveRow)
        and control.data == "model-overview-metrics"
    )
    labels = {
        control.data
        for control in metrics.controls
        if isinstance(control, ft.Container)
    }
    assert "overview-metric:architecture" in labels
    assert "overview-metric:tensors" in labels
    assert "overview-metric:weights" in labels
    assert _by_data(shell, "gguf-layout:header")
    assert _by_data(shell, "gguf-layout:weights")
    assert _by_data(shell, "gguf-pane:tensors")
    assert _by_data(shell, "gguf-pane:metadata")
    assert _by_data(shell, "gguf-pane:tokenizer")
    assert "GGUF Model" in (shell.inspector_subtitle.value or "")
    hint = shell.inspector.controls[0]
    assert isinstance(hint, ft.Text)
    assert "GGUF container" in (hint.value or "")


def test_selecting_a_gguf_node_lists_its_weight_tensors(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = _tiny(tmp_path / "attn.gguf")
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    attn = next(
        node
        for node in session.document.main_graph.nodes
        if node.op_type == "Attention"
    )
    shell._refresh_inspector(frozenset({attn.id}))
    cards = [
        control.data
        for control in _all_controls(shell)
        if isinstance(getattr(control, "data", None), str)
        and str(control.data).startswith("tensor-card:")
    ]
    assert cards, "GGUF attention weights appear as tensor cards"
    text = " ".join(
        str(getattr(control, "value", "") or "")
        for control in _all_controls(shell)
        if isinstance(control, ft.Text)
    )
    assert "blk.0.attn_q.weight" in text


def test_inspecting_a_catalog_tensor_opens_the_owner(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = _tiny(tmp_path / "click.gguf")
    shell, _page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    embed = next(
        tensor_id
        for tensor_id in session.document.tensors
        if tensor_id.endswith("#token_embd.weight")
    )
    shell._on_gguf_tensor(embed)
    cards = [
        control
        for control in _all_controls(shell)
        if getattr(control, "data", None) == f"tensor-card:{embed}"
    ]
    assert cards, "inspecting a catalog tensor opens its weight card"


def test_metadata_pane_lists_architecture_keys(
    tmp_path: Path, service: ApplicationService
) -> None:
    path = _tiny(tmp_path / "meta.gguf")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(path))
    shell.gguf_panel.expanded_groups.update({"general", "architecture", "tokenizer"})
    shell.gguf_panel.pane = "metadata"
    shell.gguf_panel._rebuild()
    assert _by_data(shell, "gguf-kv:general.architecture")
    assert _by_data(shell, "gguf-kv:llama.block_count")


def test_tokenizer_preview_dialog(tmp_path: Path, service: ApplicationService) -> None:
    path = _tiny(tmp_path / "preview.gguf")
    shell, page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    shell.gguf_panel.pane = "tokenizer"
    shell.gguf_panel.expanded_groups.add("tokenizer")
    shell.gguf_panel._rebuild()
    preview = _by_data(shell, "gguf-kv-preview:tokenizer.ggml.tokens")
    assert isinstance(preview, ft.TextButton)
    row = next(
        item
        for item in gguf_tokenizer_rows(session.document)
        if item.key == "tokenizer.ggml.tokens"
    )
    shell._on_gguf_preview_kv(row)
    assert page.dialogs
    dialog = page.dialogs[-1]
    assert dialog.open
    body = " ".join(
        str(getattr(control, "value", "") or "")
        for control in _walk(dialog)
        if isinstance(control, ft.Text)
    )
    assert "tok0" in body
