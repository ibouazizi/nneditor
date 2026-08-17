"""Tests for session-state persistence and safe open validation (P1.9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import flet as ft
import pytest

from nneditor.application.persistence import (
    MAX_RECENT,
    SessionState,
    SessionStateStore,
    ViewState,
)
from nneditor.application.session import ApplicationService, SessionError
from nneditor.ui.app import Shell
from tests.fixtures.onnx_models import build_embedded_model
from tests.unit.test_shell import StubPage


class TestSessionState:
    def test_record_open_keeps_most_recent_first(self) -> None:
        state = SessionState()
        state.record_open("a.onnx", "sha256:a", "a")
        state.record_open("b.onnx", "sha256:b", "b")
        state.record_open("a.onnx", "sha256:a", "a")
        assert [entry.content_hash for entry in state.recent] == [
            "sha256:a",
            "sha256:b",
        ]

    def test_recent_list_is_capped(self) -> None:
        state = SessionState()
        for index in range(MAX_RECENT + 5):
            state.record_open(f"m{index}.onnx", f"sha256:{index}", f"m{index}")
        assert len(state.recent) == MAX_RECENT
        assert state.recent[0].content_hash == f"sha256:{MAX_RECENT + 4}"


class TestSessionStateStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = SessionStateStore(tmp_path)
        store.state.record_open("model.onnx", "sha256:m", "model")
        store.state.record_view(
            "sha256:m",
            ViewState(graph_id="g:main", x=1.5, y=-2.0, scale=0.5, selection=("n:a",)),
        )
        store.save()

        reloaded = SessionStateStore(tmp_path)
        assert reloaded.state.recent[0].title == "model"
        view = reloaded.state.view_for("sha256:m")
        assert view == ViewState("g:main", 1.5, -2.0, 0.5, ("n:a",))

    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        store = SessionStateStore(tmp_path)
        assert store.state.recent == []
        assert store.state.views == {}

    def test_corrupt_file_is_discarded(self, tmp_path: Path) -> None:
        (tmp_path / "session-state.json").write_text("{nope", encoding="utf-8")
        assert SessionStateStore(tmp_path).state.recent == []

    def test_newer_version_is_discarded(self, tmp_path: Path) -> None:
        (tmp_path / "session-state.json").write_text(
            json.dumps({"version": 99, "recent": [{"bogus": True}]}),
            encoding="utf-8",
        )
        assert SessionStateStore(tmp_path).state.recent == []

    def test_malformed_entries_discard_everything(self, tmp_path: Path) -> None:
        (tmp_path / "session-state.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "recent": [{"path": "a", "content_hash": "h", "title": "a"}],
                    "views": {"h": {"graph_id": "g", "x": "not a number"}},
                }
            ),
            encoding="utf-8",
        )
        state = SessionStateStore(tmp_path).state
        assert state.recent == [] and state.views == {}

    def test_save_never_writes_beside_artifacts(self, tmp_path: Path) -> None:
        store = SessionStateStore(tmp_path / "state")
        store.save()
        assert (tmp_path / "state" / "session-state.json").exists()
        assert not list((tmp_path / "state").glob("*.partial"))


class TestSafeOpenBoundary:
    def test_missing_path_is_a_session_error(self, tmp_path: Path) -> None:
        with ApplicationService() as service:
            with pytest.raises(SessionError, match="does not exist"):
                service.open_model(tmp_path / "ghost.onnx")

    def test_unrecognized_directories_are_rejected(self, tmp_path: Path) -> None:
        # Directories are legal open targets since package bundles
        # (.mlpackage) arrived, so the rejection now comes from detection.
        with ApplicationService() as service:
            with pytest.raises(SessionError, match="no recognized model package"):
                service.open_model(tmp_path)


class TestServiceAndShellPersistence:
    def test_opening_records_recents(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=16)
        store = SessionStateStore(tmp_path / "state")
        with ApplicationService(state_store=store) as service:
            session = service.open_model(path)
            content_hash = session.document.source.content_hash
        reloaded = SessionStateStore(tmp_path / "state")
        assert reloaded.state.recent[0].content_hash == content_hash
        assert reloaded.state.recent[0].title == "model.onnx"

    def test_the_shell_restores_the_saved_view(self, tmp_path: Path) -> None:
        path = tmp_path / "model.onnx"
        build_embedded_model(path, elements=16)
        store = SessionStateStore(tmp_path / "state")

        with ApplicationService(state_store=store) as service:
            page = StubPage()
            shell = Shell(cast(ft.Page, page), service)
            shell.build()
            session = service.open_model(path)
            shell.show_session(session)
            node_id = session.search("scale")[0]
            shell.renderer.set_selection(frozenset({node_id}))
            shell.view = {"scale": 2.0, "x": 10.0, "y": 20.0}
            shell._apply_viewport()

        with ApplicationService(
            state_store=SessionStateStore(tmp_path / "state")
        ) as service:
            page = StubPage()
            shell = Shell(cast(ft.Page, page), service)
            shell.build()
            shell.show_session(service.open_model(path))
            assert shell.view == {"scale": 2.0, "x": 10.0, "y": 20.0}
            assert shell.renderer.selection == {node_id}
