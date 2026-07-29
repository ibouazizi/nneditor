"""Tests for the Flet application entry point (P1.8)."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import flet as ft
import pytest
from pytest import MonkeyPatch

from nneditor.desktop.windows_associations import FileAssociationError
from nneditor.ui import app


class FakePage:
    """Small page double that records controls without starting Flutter."""

    def __init__(self) -> None:
        self.title = ""
        self.window = ft.Window()
        self.controls: list[ft.Control] = []
        self.overlay: list[ft.Control] = []
        self.services: list[object] = []
        self.width = 1400
        self.updates = 0
        self.on_resize: object = None

    def add(self, *controls: ft.Control) -> None:
        self.controls.extend(controls)

    def update(self) -> None:
        self.updates += 1

    def run_thread(self, target: Callable[[], None]) -> None:
        target()


def test_main_builds_the_shell() -> None:
    page = FakePage()

    app.main(cast(ft.Page, page))

    assert page.title == app.APP_TITLE
    assert page.window.icon == str(app.APP_WINDOW_ICON_PATH)
    assert len(page.controls) == 1, "one root column holds the whole shell"
    # The picker is a Flet Service; registering it as a Control made the
    # client refuse the page with "Unknown control: FilePicker".
    assert any(isinstance(item, ft.FilePicker) for item in page.services)
    assert not page.overlay, "no Service may be registered as a Control"
    assert all(isinstance(item, ft.Control) for item in page.overlay)
    assert page.on_resize is not None
    assert page.updates > 0


def test_run_delegates_to_flet(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[Callable[[ft.Page], None], str | None, str | None]] = []

    def fake_run(
        target: Callable[[ft.Page], None],
        *,
        assets_dir: str | None = None,
        upload_dir: str | None = None,
    ) -> None:
        calls.append((target, assets_dir, upload_dir))

    monkeypatch.setattr(ft, "run", fake_run)

    app.run()

    assert calls == [
        (
            app.main,
            str(app.APP_ASSETS_DIRECTORY),
            str(app._WEB_UPLOAD_DIRECTORY),
        )
    ]


def test_run_passes_a_desktop_launch_path_to_the_page_entry_point(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Callable[[ft.Page], None]] = []

    def fake_run(
        target: Callable[[ft.Page], None],
        **kwargs: object,
    ) -> None:
        calls.append(target)

    monkeypatch.setattr(ft, "run", fake_run)
    model = tmp_path / "model.onnx"

    app.run(model)

    assert len(calls) == 1
    assert isinstance(calls[0], partial)
    target = cast(partial[None], calls[0])
    assert target.func is app.main
    assert target.keywords == {"launch_path": model}


def test_cli_registers_file_types_without_starting_the_ui(
    monkeypatch: MonkeyPatch,
) -> None:
    registrations: list[Path] = []

    def register(*, icon_path: Path) -> SimpleNamespace:
        registrations.append(icon_path)
        return SimpleNamespace(extensions=(".onnx", ".pt"))

    monkeypatch.setattr(app, "register_file_associations", register)
    monkeypatch.setattr(
        app,
        "run",
        lambda path=None: (_ for _ in ()).throw(AssertionError("must not launch")),
    )

    app.cli(["--register-file-types"])

    assert registrations == [app.APP_WINDOW_ICON_PATH]


def test_cli_routes_unregister_settings_and_model_launch(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    actions: list[object] = []
    monkeypatch.setattr(
        app,
        "unregister_file_associations",
        lambda: actions.append("unregister"),
    )
    monkeypatch.setattr(
        app,
        "open_default_apps_settings",
        lambda: actions.append("settings"),
    )
    monkeypatch.setattr(app, "run", lambda path=None: actions.append(path))

    app.cli(["--unregister-file-types"])
    app.cli(["--choose-default-app"])
    model = tmp_path / "model.onnx"
    app.cli([str(model)])

    assert actions == ["unregister", "settings", model]


def test_cli_translates_file_association_errors(
    monkeypatch: MonkeyPatch,
) -> None:
    def fail() -> None:
        raise FileAssociationError("unsupported")

    monkeypatch.setattr(app, "unregister_file_associations", fail)
    with pytest.raises(SystemExit) as raised:
        app.cli(["--unregister-file-types"])
    assert raised.value.code == 2


def test_main_passes_launch_path_to_the_shell(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(app.Shell, "open_path", lambda self, path: opened.append(path))
    page = FakePage()
    model = tmp_path / "model.onnx"

    app.main(cast(ft.Page, page), launch_path=model)

    assert opened == [model]


def test_runtime_icon_assets_are_packaged() -> None:
    web_icons = (
        app.APP_ICON_PATH,
        app.APP_ASSETS_DIRECTORY / "favicon.png",
        app.APP_ASSETS_DIRECTORY / "icons" / "apple-touch-icon-192.png",
        app.APP_ASSETS_DIRECTORY / "icons" / "icon-192.png",
        app.APP_ASSETS_DIRECTORY / "icons" / "icon-512.png",
        app.APP_ASSETS_DIRECTORY / "icons" / "icon-maskable-192.png",
        app.APP_ASSETS_DIRECTORY / "icons" / "icon-maskable-512.png",
    )
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in web_icons)
    assert app.APP_WINDOW_ICON_PATH.read_bytes().startswith(b"\x00\x00\x01\x00")
    manifest = json.loads(
        (app.APP_ASSETS_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["name"] == app.APP_TITLE
