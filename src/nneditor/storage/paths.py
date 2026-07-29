"""Confinement rules for artifact-supplied file paths.

An ONNX model's ``external_data`` location is attacker-controlled input: it is a
string inside a file the user downloaded. It is resolved against an approved
root and must stay there, including after symbolic links are followed.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

__all__ = ["UnsafePathError", "resolve_within"]


class UnsafePathError(ValueError):
    """Raised when an artifact-supplied path escapes its approved root."""


def _components(location: str) -> list[str]:
    """Split a location on both separators, as producers emit both."""
    pure = PurePosixPath(location.replace("\\", "/"))
    return [part for part in pure.parts if part not in ("", ".")]


def resolve_within(root: Path, location: str) -> Path:
    """Resolve ``location`` relative to ``root`` and prove it stays inside.

    Raises :class:`UnsafePathError` for empty, absolute, traversing, or
    link-escaping locations.
    """
    if not location or not location.strip():
        raise UnsafePathError("external data location is empty")
    if "\x00" in location:
        raise UnsafePathError("external data location contains a NUL byte")
    if PurePosixPath(location).is_absolute() or PureWindowsPath(location).is_absolute():
        raise UnsafePathError(f"external data location is absolute: {location!r}")
    if PureWindowsPath(location).drive:
        raise UnsafePathError(f"external data location names a drive: {location!r}")

    parts = _components(location)
    if not parts:
        raise UnsafePathError(f"external data location names no file: {location!r}")
    if ".." in parts:
        raise UnsafePathError(f"external data location traverses upward: {location!r}")

    real_root = Path(os.path.realpath(root))
    candidate = Path(os.path.realpath(real_root.joinpath(*parts)))
    if candidate != real_root and real_root not in candidate.parents:
        raise UnsafePathError(
            f"external data location escapes the approved root: {location!r}"
        )
    return candidate
