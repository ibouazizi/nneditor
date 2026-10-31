"""Tracer selection, install/reload, approval wiring, and CLI authoring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import flet as ft
import numpy as np
import pytest

from nneditor.application.session import ApplicationService
from nneditor.tracing import plugins
from nneditor.tracing.plugin_manifest import MANIFEST_NAME, load_manifest
from nneditor.ui import app
from tests.fixtures.tracing_plugins import write_linear_gguf, write_trace_plugin
from tests.unit.test_shell import make_shell


def test_plugin_enables_unsupported_architecture_and_can_be_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    plugin = load_manifest(root / MANIFEST_NAME)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        shell.show_session(session)
        panel = shell.trace
        assert panel.tracer.value == plugin.identity
        assert panel.tracer_panel.visible and panel.run_settings.visible
        assert panel.blocker_title.value == "Prompt input required"
        assert panel.runtime_text.value == plugin.description
        assert panel.backend.value == "auto"
        ids = tmp_path / "tokens.npy"
        np.save(ids, np.array([[0, 2]], dtype=np.int64))
        panel.bind_tensor("tokens", session.trace_tensor_input("tokens", ids))
        panel.refresh_actions()
        assert not panel.run_button.disabled
        assert plugin.name in str(panel.approval_notice.value)
        builtin = next(
            p for p in session.trace_registry.plugins if p.id == "builtin.numpy-llm"
        )
        panel.tracer.value = builtin.identity
        panel._on_tracer_changed(cast(Any, None))
        assert panel.run_button.disabled
        assert panel.blocker_notice.visible
        panel.tracer.value = plugin.identity
        panel._on_tracer_changed(cast(Any, None))
        assert not panel.run_button.disabled


def test_ui_installs_and_reloads_without_importing_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_trace_plugin(tmp_path / "source")
    installed = tmp_path / "installed"
    monkeypatch.setattr(plugins, "trace_plugin_directory", lambda: installed)

    async def choose(*_args: Any, **_kwargs: Any) -> str:
        return str(source)

    monkeypatch.setattr(ft.FilePicker, "get_directory_path", choose)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page = make_shell(service)
        session = service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        shell.show_session(session)
        panel = shell.trace
        assert panel.run_button.disabled
        row = panel.tracer_panel.controls[0]
        assert isinstance(row, ft.Row)
        assert panel.tracer_install in row.controls
        asyncio.run(panel._on_install_tracer(cast(Any, None)))
        assert (installed / "test.linear" / MANIFEST_NAME).exists()
        assert panel.tracer.value == load_manifest(source / MANIFEST_NAME).identity
        assert panel.run_settings.visible
        (installed / "test.linear" / "helper.py").write_text("SCALE=2\n")
        old = panel.tracer.value
        panel._on_tracers_refresh()
        assert panel.tracer.value != old
        monkeypatch.setattr(page, "web", True, raising=False)
        panel.refresh_actions()
        assert panel.run_button.disabled and panel.tracer_install.disabled


def test_discovery_errors_are_visible_in_trace_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_trace_plugin(tmp_path / "bad", api_version=999)
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        shell.show_session(
            service.open_model(write_linear_gguf(tmp_path / "model.gguf"))
        )
        assert "Unsupported tracer API version" in str(shell.trace.tracer_info.value)
        assert shell.trace.tracer_panel.visible
        assert not shell.trace.run_settings.visible


def test_cli_creates_installs_and_lists_without_starting_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        plugins, "trace_plugin_directory", lambda: tmp_path / "installed"
    )
    monkeypatch.setattr(
        app, "run", lambda *_: pytest.fail("CLI must not launch the UI")
    )
    root = tmp_path / "created"
    app.cli(["--create-tracer", str(root)])
    assert "Created" in capsys.readouterr().out
    assert (root / "tracer.py").is_file()
    app.cli(["--install-tracer", str(root)])
    assert "Installed" in capsys.readouterr().out
    app.cli(["--list-tracers"])
    text = capsys.readouterr().out
    assert "builtin.onnx" in text and "example.numpy-llm" in text
    with pytest.raises(SystemExit):
        app.cli(["--install-tracer", str(root)])
    assert "already installed" in capsys.readouterr().err
