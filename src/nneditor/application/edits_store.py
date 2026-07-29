"""Sidecar persistence for revision chains (task P3.6).

A model's pending edits — its whole revision chain, including undone redo
candidates and the cursor — persist by content hash in the user state
directory, never beside the artifact. Killing the process mid-session loses
nothing that was committed: every mutation saves atomically, and a partial
write can never replace a good file.

A sidecar that fails to parse, or whose chain does not verify (broken parent
links, ids that do not match their content), is *quarantined*: renamed to
``*.corrupt`` so the user's history is preserved for inspection rather than
silently deleted, while the session recovers cleanly at the base revision.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Final

from nneditor.editing.cow import EditError
from nneditor.editing.revisions import (
    Revision,
    ValidationState,
    command_from_json,
    command_to_json,
)

__all__ = ["RevisionSidecarStore"]

REVISIONS_FORMAT_VERSION: Final = 1


class RevisionSidecarStore:
    """Loads, saves, and quarantines one artifact's revision chain."""

    __slots__ = ("_save_lock", "directory")

    def __init__(self, directory: Path) -> None:
        self.directory = directory / "revisions"
        self._save_lock = threading.Lock()

    def _path_for(self, content_hash: str) -> Path:
        safe = content_hash.replace(":", "-")
        return self.directory / f"{safe}.json"

    def load(self, content_hash: str) -> tuple[list[Revision], int] | None:
        """The persisted chain and cursor, or ``None`` (absent or quarantined)."""
        target = self._path_for(content_hash)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            # Absent, or transiently unreadable (an AV scanner holding the
            # file open is routine on Windows). Neither is corruption, so the
            # user's revision chain must survive untouched.
            return None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            self._quarantine(target)
            return None
        try:
            if (
                not isinstance(raw, dict)
                or raw.get("format_version") != REVISIONS_FORMAT_VERSION
            ):
                raise EditError("unknown sidecar format")
            if raw.get("content_hash") != content_hash:
                raise EditError("sidecar belongs to a different artifact")
            revisions: list[Revision] = []
            for entry in raw["revisions"]:
                commands = tuple(command_from_json(item) for item in entry["commands"])
                validation_raw = entry.get("validation", {})
                revisions.append(
                    Revision(
                        id=str(entry["id"]),
                        parent_id=(
                            None
                            if entry["parent_id"] is None
                            else str(entry["parent_id"])
                        ),
                        commands=commands,
                        validation=ValidationState(
                            ok=bool(validation_raw.get("ok", True)),
                            findings=tuple(
                                str(item) for item in validation_raw.get("findings", [])
                            ),
                        ),
                    )
                )
            cursor = int(raw["cursor"])
            if not 0 <= cursor <= len(revisions):
                raise EditError("cursor outside the chain")
            return revisions, cursor
        except (EditError, KeyError, TypeError, ValueError):
            self._quarantine(target)
            return None

    def save(self, content_hash: str, revisions: list[Revision], cursor: int) -> None:
        """Atomically replace the artifact's revision sidecar."""
        payload = {
            "format_version": REVISIONS_FORMAT_VERSION,
            "content_hash": content_hash,
            "cursor": cursor,
            "revisions": [
                {
                    "id": revision.id,
                    "parent_id": revision.parent_id,
                    "commands": [
                        command_to_json(command) for command in revision.commands
                    ],
                    "validation": {
                        "ok": revision.validation.ok,
                        "findings": list(revision.validation.findings),
                    },
                }
                for revision in revisions
            ],
        }
        target = self._path_for(content_hash)
        # A unique temp name plus the lock keeps concurrent saves from
        # colliding on one .partial file, which raises PermissionError on
        # Windows when os.replace hits a file another thread holds open.
        partial = target.with_name(f"{target.name}.{uuid.uuid4().hex}.partial")
        with self._save_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(partial, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=1, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, target)
            except BaseException:
                with contextlib.suppress(OSError):
                    partial.unlink(missing_ok=True)
                raise

    def discard(self, content_hash: str) -> None:
        """Remove the artifact's sidecar; discarding history is explicit."""
        self._path_for(content_hash).unlink(missing_ok=True)

    def quarantine(self, content_hash: str) -> None:
        """Preserve a chain that parsed but failed artifact replay."""
        self._quarantine(self._path_for(content_hash))

    @staticmethod
    def _quarantine(target: Path) -> None:
        """Best-effort rename to ``*.corrupt``; never deletes, never raises.

        Quarantine runs while an artifact is being opened; a failure here must
        not abort the open, and the fallback must never destroy the user's
        chain — a uniquely suffixed rename is attempted instead.
        """
        corrupt = target.with_name(target.name + ".corrupt")
        try:
            os.replace(target, corrupt)
        except OSError:
            unique = target.with_name(f"{target.name}.{uuid.uuid4().hex}.corrupt")
            with contextlib.suppress(OSError):
                os.replace(target, unique)
