"""Tests for external-path confinement (P0.4, security checklist)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nneditor.storage.paths import UnsafePathError, resolve_within


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "root" / "nested").mkdir(parents=True)
    (tmp_path / "root" / "weights.bin").write_bytes(b"ok")
    (tmp_path / "root" / "nested" / "more.bin").write_bytes(b"ok")
    (tmp_path / "outside.bin").write_bytes(b"secret")
    return tmp_path / "root"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("weights.bin", "weights.bin"),
        ("./weights.bin", "weights.bin"),
        ("nested/more.bin", "nested/more.bin"),
        ("nested\\more.bin", "nested/more.bin"),
    ],
)
def test_locations_inside_the_root_resolve(
    root: Path, location: str, expected: str
) -> None:
    assert resolve_within(root, location) == Path(
        os.path.realpath(root / Path(expected))
    )


def test_a_location_may_name_a_file_that_does_not_exist_yet(root: Path) -> None:
    """Existence is the caller's check; confinement is this function's job."""
    assert resolve_within(root, "absent.bin").name == "absent.bin"


@pytest.mark.parametrize(
    "location",
    [
        "../outside.bin",
        "nested/../../outside.bin",
        "..",
        "/etc/passwd",
        "\\\\server\\share\\file.bin",
        "C:\\Windows\\system32\\config",
        "",
        "   ",
    ],
)
def test_hostile_locations_are_rejected(root: Path, location: str) -> None:
    with pytest.raises(UnsafePathError):
        resolve_within(root, location)


def test_a_drive_relative_location_is_rejected(root: Path) -> None:
    """``C:weights.bin`` is not absolute but still names another drive."""
    with pytest.raises(UnsafePathError, match="names a drive"):
        resolve_within(root, "C:weights.bin")


def test_a_nul_byte_is_rejected(root: Path) -> None:
    with pytest.raises(UnsafePathError, match="NUL"):
        resolve_within(root, "weights\x00.bin")


def test_a_location_naming_no_file_is_rejected(root: Path) -> None:
    with pytest.raises(UnsafePathError, match="names no file"):
        resolve_within(root, "./")


@pytest.mark.skipif(
    os.name == "nt", reason="symlink creation needs elevation on Windows"
)
def test_a_symlink_escaping_the_root_is_rejected(root: Path, tmp_path: Path) -> None:
    (root / "escape.bin").symlink_to(tmp_path / "outside.bin")
    with pytest.raises(UnsafePathError, match="escapes the approved root"):
        resolve_within(root, "escape.bin")


@pytest.mark.skipif(
    os.name == "nt", reason="symlink creation needs elevation on Windows"
)
def test_a_symlink_staying_inside_the_root_is_accepted(root: Path) -> None:
    (root / "alias.bin").symlink_to(root / "weights.bin")
    assert resolve_within(root, "alias.bin").name == "weights.bin"
