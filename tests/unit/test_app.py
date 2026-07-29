"""Tests for the Flet application entry point (P1.8)."""

import json
from collections.abc import Callable
from typing import cast

import flet as ft
from pytest import MonkeyPatch

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
