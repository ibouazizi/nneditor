"""Sidecar persistence for tensor statistics (task P3.3).

Statistics are expensive to stream but derived entirely from immutable bytes,
so they persist by *content hash* and statistics version — the same artifact
opened tomorrow, or from a different path, reuses them. Like every sidecar,
the file lives in the user state directory, never beside the model, and a
corrupt or version-stale file is discarded and recomputed, not migrated.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from pathlib import Path

from nneditor.analysis.statistics import (
    STATISTICS_VERSION,
    TensorStatistics,
    statistics_from_json,
    statistics_to_json,
)

__all__ = ["StatisticsSidecarStore"]


class StatisticsSidecarStore:
    """Loads and saves one artifact's statistics keyed by content hash."""

    __slots__ = ("_save_lock", "directory")

    def __init__(self, directory: Path) -> None:
        self.directory = directory / "statistics"
        self._save_lock = threading.Lock()

    def _path_for(self, content_hash: str) -> Path:
        safe = content_hash.replace(":", "-")
        return self.directory / f"{safe}.json"

    def load(self, content_hash: str) -> dict[str, TensorStatistics]:
        """Every persisted, version-current statistic for one artifact."""
        try:
            raw = json.loads(self._path_for(content_hash).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(raw, dict)
            or raw.get("statistics_version") != STATISTICS_VERSION
            or raw.get("content_hash") != content_hash
            or not isinstance(raw.get("tensors"), dict)
        ):
            return {}
        loaded: dict[str, TensorStatistics] = {}
        for tensor_id, entry in raw["tensors"].items():
            if isinstance(entry, dict):
                stats = statistics_from_json(entry)
                if stats is not None:
                    loaded[str(tensor_id)] = stats
        return loaded

    def save(self, content_hash: str, stats: dict[str, TensorStatistics]) -> None:
        """Atomically replace the artifact's statistics file."""
        payload = {
            "statistics_version": STATISTICS_VERSION,
            "content_hash": content_hash,
            "tensors": {
                tensor_id: statistics_to_json(item)
                for tensor_id, item in sorted(stats.items())
            },
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
