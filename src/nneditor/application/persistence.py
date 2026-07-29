"""Session-state persistence (task P1.9).

What the viewer remembers between launches — recent models and each model's
last view — lives in the user's state directory, never beside a source
artifact (rule 1: sources stay untouched). View state is keyed by *content
hash*, not path, so a moved or renamed artifact keeps its view and two copies
of the same model share one.

The state file is disposable by design, like cached layouts: a corrupt or
newer-versioned file is discarded and rebuilt rather than migrated, because
nothing in it is authoritative.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = ["RecentModel", "SessionState", "SessionStateStore", "ViewState"]

STATE_VERSION: Final = 1
MAX_RECENT: Final = 10


def default_state_directory() -> Path:
    """The per-user state directory for this platform."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".local" / "state"
    return base / "nneditor"


@dataclass(frozen=True, slots=True)
class RecentModel:
    """One entry of the recent-models list."""

    path: str
    content_hash: str
    title: str


@dataclass(frozen=True, slots=True)
class ViewState:
    """Where the user last was in a model."""

    graph_id: str
    x: float
    y: float
    scale: float
    selection: tuple[str, ...] = ()


@dataclass(slots=True)
class SessionState:
    """Everything persisted between launches."""

    recent: list[RecentModel] = field(default_factory=list)
    views: dict[str, ViewState] = field(default_factory=dict)

    def record_open(self, path: str, content_hash: str, title: str) -> None:
        """Move or insert a model at the top of the recent list."""
        self.recent = [
            entry for entry in self.recent if entry.content_hash != content_hash
        ]
        self.recent.insert(0, RecentModel(path, content_hash, title))
        del self.recent[MAX_RECENT:]

    def record_view(self, content_hash: str, view: ViewState) -> None:
        self.views[content_hash] = view

    def view_for(self, content_hash: str) -> ViewState | None:
        return self.views.get(content_hash)


class SessionStateStore:
    """Loads and atomically saves one :class:`SessionState` file."""

    __slots__ = ("_save_lock", "path", "state")

    def __init__(self, directory: Path | None = None) -> None:
        base = directory if directory is not None else default_state_directory()
        self.path = base / "session-state.json"
        self.state = self._load()
        self._save_lock = threading.Lock()

    def _load(self) -> SessionState:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return SessionState()
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            return SessionState()
        state = SessionState()
        try:
            for entry in raw.get("recent", []):
                state.recent.append(
                    RecentModel(
                        path=str(entry["path"]),
                        content_hash=str(entry["content_hash"]),
                        title=str(entry["title"]),
                    )
                )
            for content_hash, view in raw.get("views", {}).items():
                state.views[str(content_hash)] = ViewState(
                    graph_id=str(view["graph_id"]),
                    x=float(view["x"]),
                    y=float(view["y"]),
                    scale=float(view["scale"]),
                    selection=tuple(str(item) for item in view.get("selection", [])),
                )
        except (KeyError, TypeError, ValueError):
            # Partially readable state is worse than none: half a recents
            # list next to missing views would look like data loss.
            return SessionState()
        del state.recent[MAX_RECENT:]
        return state

    def save(self) -> None:
        """Write the state atomically; never leaves a partial file."""
        # A unique temp name plus the lock keeps concurrent saves from
        # colliding on one .partial file, which raises PermissionError on
        # Windows when os.replace hits a file another thread holds open.
        # The payload snapshot is taken under the same lock so a concurrent
        # mutation cannot change the dicts mid-serialization.
        partial = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.partial")
        with self._save_lock:
            payload = {
                "version": STATE_VERSION,
                "recent": [
                    {
                        "path": entry.path,
                        "content_hash": entry.content_hash,
                        "title": entry.title,
                    }
                    for entry in self.state.recent
                ],
                "views": {
                    content_hash: {
                        "graph_id": view.graph_id,
                        "x": view.x,
                        "y": view.y,
                        "scale": view.scale,
                        "selection": list(view.selection),
                    }
                    for content_hash, view in self.state.views.items()
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(partial, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=1, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    partial.unlink(missing_ok=True)
                raise
