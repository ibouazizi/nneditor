"""Windows file-association plans stay explicit and user controlled."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from nneditor.artifact_formats import MODEL_FILE_SUFFIXES
from nneditor.desktop import windows_associations
from nneditor.desktop.windows_associations import (
    CAPABILITIES_PATH,
    PROG_ID,
    FileAssociationError,
    association_registration,
    current_launch_command,
    open_default_apps_settings,
    register_file_associations,
    unregister_file_associations,
)


def test_registration_advertises_every_supported_extension(tmp_path: Path) -> None:
    registration = association_registration(
        (r"C:\Program Files\NNEditor\nneditor.exe",),
        icon_path=tmp_path / "nneditor.ico",
    )

    assert registration.extensions == MODEL_FILE_SUFFIXES
    assert registration.command == (r'"C:\Program Files\NNEditor\nneditor.exe" "%1"')
    assert all(
        any(
            item.key == rf"{CAPABILITIES_PATH}\FileAssociations"
            and item.name == extension
            and item.value == PROG_ID
            for item in registration.values
        )
        for extension in MODEL_FILE_SUFFIXES
    )


def test_registration_never_writes_windows_userchoice() -> None:
    registration = association_registration(
        ("python.exe", "-m", "nneditor"),
        icon_path=Path("nneditor.ico"),
    )

    assert '"%1"' in registration.command
    assert all("userchoice" not in item.key.lower() for item in registration.values)


def test_launch_command_prefers_the_console_script_beside_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python = scripts / "python.exe"
    launcher = scripts / "nneditor.exe"
    python.touch()
    launcher.touch()
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.sys.executable",
        str(python),
    )

    assert current_launch_command() == (str(launcher.resolve()),)


def test_launch_command_falls_back_through_argv_path_and_python_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python.exe"
    python.touch()
    launcher = tmp_path / "nneditor-custom.exe"
    launcher.touch()
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.sys.argv",
        [str(launcher)],
    )
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.shutil.which",
        lambda name: None,
    )

    assert current_launch_command() == (str(launcher.resolve()),)

    launcher.unlink()
    assert current_launch_command() == (
        str(python.resolve()),
        "-m",
        "nneditor",
    )


class _FakeKey:
    def __enter__(self) -> _FakeKey:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _fake_winreg(
    *,
    written: list[tuple[str | None, str]] | None = None,
    deleted: list[str] | None = None,
) -> ModuleType:
    writes = written if written is not None else []
    deletions = deleted if deleted is not None else []

    def set_value(
        key: _FakeKey,
        name: str | None,
        reserved: int,
        kind: int,
        value: str,
    ) -> None:
        writes.append((name, value))

    def delete_value(key: _FakeKey, name: str) -> None:
        deletions.append(name)

    fake = SimpleNamespace(
        HKEY_CURRENT_USER=1,
        KEY_SET_VALUE=2,
        KEY_READ=4,
        KEY_WRITE=8,
        REG_SZ=1,
        CreateKeyEx=lambda *args: _FakeKey(),
        OpenKey=lambda *args: _FakeKey(),
        SetValueEx=set_value,
        DeleteValue=delete_value,
    )
    return cast(ModuleType, cast(Any, fake))


def test_register_and_unregister_use_only_owned_per_user_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[tuple[str | None, str]] = []
    deleted: list[str] = []
    fake_winreg = _fake_winreg(written=written, deleted=deleted)
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    notifications = 0

    def notify() -> None:
        nonlocal notifications
        notifications += 1

    removed_trees: list[tuple[int, str]] = []
    monkeypatch.setattr(windows_associations, "_notify_shell", notify)
    monkeypatch.setattr(
        windows_associations,
        "_delete_tree",
        lambda root, path: removed_trees.append((root, path)),
    )

    registration = register_file_associations(
        ("nneditor.exe",),
        icon_path=tmp_path / "icon.ico",
    )
    unregister_file_associations()

    assert len(written) == len(registration.values)
    assert deleted.count(PROG_ID) == len(MODEL_FILE_SUFFIXES)
    assert windows_associations.APPLICATION_NAME in deleted
    assert removed_trees == [
        (1, CAPABILITIES_PATH),
        (1, windows_associations.PROG_ID_PATH),
    ]
    assert notifications == 2


def test_registration_wraps_registry_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_winreg = _fake_winreg()
    cast(Any, fake_winreg).CreateKeyEx = lambda *args: (_ for _ in ()).throw(
        OSError("denied")
    )
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

    with pytest.raises(FileAssociationError, match="denied"):
        register_file_associations(
            ("nneditor.exe",),
            icon_path=tmp_path / "icon.ico",
        )


def test_default_apps_settings_uri_and_error_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.os.startfile",
        opened.append,
    )
    open_default_apps_settings()
    assert opened == ["ms-settings:defaultapps?registeredAppUser=NNEditor"]

    def fail(target: str) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.os.startfile",
        fail,
    )
    with pytest.raises(FileAssociationError, match="unavailable"):
        open_default_apps_settings()


@pytest.mark.parametrize(
    "action",
    [
        lambda: register_file_associations(("nneditor",), icon_path="icon.ico"),
        unregister_file_associations,
        open_default_apps_settings,
    ],
)
def test_windows_actions_reject_other_platforms(
    action: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nneditor.desktop.windows_associations.sys.platform",
        "linux",
    )
    with pytest.raises(FileAssociationError, match="only on Windows"):
        action()
