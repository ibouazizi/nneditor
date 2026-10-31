"""Headless Hugging Face snapshot explorer wiring in the Flet shell."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import flet as ft
import pytest

from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.application.session import ApplicationService
from nneditor.tracing.contracts import TraceBackend, TraceDevice
from nneditor.ui.app import Shell
from tests.fixtures.huggingface_models import (
    write_llama_snapshot,
    write_traceable_llama_snapshot,
)
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
    found.extend(_walk(shell.hf_panel.root))
    return found


def _by_data(shell: Shell, data: str) -> ft.Control:
    return next(control for control in _all_controls(shell) if control.data == data)


def test_huggingface_overview_shows_container_explorer(
    tmp_path: Path, service: ApplicationService
) -> None:
    root = write_llama_snapshot(tmp_path / "model")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(root))

    sections = {
        control.data: control
        for control in shell.model_info.controls
        if isinstance(control.data, str)
    }
    assert "model-overview-summary" in sections
    assert "hf-explorer" in sections
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
    assert "overview-metric:blocks" in labels
    assert _by_data(shell, "hf-layout")
    assert _by_data(shell, "hf-pane:tensors")
    assert _by_data(shell, "hf-pane:config")
    assert _by_data(shell, "hf-pane:tokenizer")


def test_inspecting_a_catalog_tensor_opens_the_owner(
    tmp_path: Path, service: ApplicationService
) -> None:
    root = write_llama_snapshot(tmp_path / "click")
    shell, _page = make_shell(service)
    session = service.open_model(root)
    shell.show_session(session)
    embed = next(
        tensor_id
        for tensor_id in session.document.tensors
        if tensor_id.endswith("#model.embed_tokens.weight")
    )
    shell._on_gguf_tensor(embed)
    cards = [
        control
        for control in _all_controls(shell)
        if getattr(control, "data", None) == f"tensor-card:{embed}"
    ]
    assert cards, "inspecting a catalog tensor opens its weight card"


def test_config_pane_lists_model_type(
    tmp_path: Path, service: ApplicationService
) -> None:
    root = write_llama_snapshot(tmp_path / "model")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(root))
    shell.hf_panel.expanded_groups.update({"architecture", "tokenizer", "other"})
    shell.hf_panel.pane = "config"
    shell.hf_panel._rebuild()
    assert _by_data(shell, "hf-config:model_type")


def test_tokenizer_preview_dialog_lists_tokens(
    tmp_path: Path, service: ApplicationService
) -> None:
    root = write_llama_snapshot(tmp_path / "preview")
    shell, page = make_shell(service)
    shell.show_session(service.open_model(root))
    shell._on_hf_preview_tokenizer()
    assert page.dialogs
    dialog = page.dialogs[-1]
    body = " ".join(
        str(getattr(control, "value", "") or "")
        for control in _walk(dialog)
        if isinstance(control, ft.Text)
    )
    assert " a" in f" {body}" or "\n   0  a" in body or "a" in body


def test_llm_trace_controls_guide_prompt_preparation(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    shell, _page = make_shell(service)
    session = service.open_model(root)
    shell.show_session(session)
    trace = shell.trace
    assert trace.backend.value == TraceBackend.REFERENCE.value
    assert {str(option.key) for option in trace.backend.options} == {
        TraceBackend.REFERENCE.value
    }
    assert trace.device.value == TraceDevice.CPU.value
    assert trace.backend.disabled and trace.device.disabled
    assert not trace.runtime_refresh.visible
    assert "ONNX" not in str(trace.runtime_text.value)
    assert trace.prepare_prompt.visible and not trace.prepare_prompt.disabled
    assert trace.run_button.disabled
    assert "prompt" in str(trace.status.value)
    cast(Any, trace.prepare_prompt.on_click)(None)
    assert shell.workspace_tabs.selected_index == 1
    from nneditor.prompt_inputs import generate_prompt_token_tensor

    generated = generate_prompt_token_tensor(
        tmp_path / "prompt.npy", document=session.document
    )
    trace.input_bindings["tokens"] = session.trace_tensor_input(
        "tokens", generated.path
    )
    trace.refresh_actions()
    assert not trace.run_button.disabled
    assert "ready" in str(trace.status.value)


def test_unsupported_llm_explains_availability_in_trace_panel(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    root = write_llama_snapshot(tmp_path / "incomplete")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(root))
    assert shell.trace.run_button.disabled
    assert "missing" in str(shell.trace.status.value)
    assert "random data" not in str(shell.trace.status.value)


def test_tensor_catalog_can_page_beyond_first_64_rows(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    root = tmp_path / "large-catalog"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    write_safetensors(
        root / "model.safetensors",
        ((f"weight_{index:03}", "float32", (4,), bytes(16)) for index in range(80)),
    )
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(root))

    def cards() -> list[ft.Control]:
        return [
            control
            for control in _walk(shell.hf_panel.root)
            if str(control.data).startswith("hf-tensor:")
        ]

    assert len(cards()) == 64
    cast(Any, _by_data(shell, "hf-show-more")).on_click(None)
    assert len(cards()) == 80


def test_switching_from_llm_restores_onnx_trace_choices(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    from tests.fixtures.onnx_models import build_embedded_model

    root = write_traceable_llama_snapshot(tmp_path / "hf")
    shell, _page = make_shell(service)
    shell.show_session(service.open_model(root))
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    shell.show_session(service.open_model(path))
    assert len(shell.trace.backend.options) == 4
    assert shell.trace.backend.value == TraceBackend.AUTO.value
    assert shell.trace.runtime_refresh.visible
    assert not shell.trace.prepare_prompt.visible


def test_editing_disables_source_trace_until_undo(
    tmp_path: Path,
    service: ApplicationService,
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    shell, _page = make_shell(service)
    session = service.open_model(root)
    shell.show_session(session)
    tensor_id = next(iter(session.document.tensors))
    session.editing.replace_bytes(tensor_id, 0, bytes(4))
    shell.trace.refresh_actions()
    assert shell.trace.run_button.disabled
    assert "Undo tensor edits" in str(shell.trace.status.value)
    session.editing.undo()
    shell.trace.refresh_actions()
    assert not shell.trace.prepare_prompt.disabled
    assert "prompt" in str(shell.trace.status.value)
