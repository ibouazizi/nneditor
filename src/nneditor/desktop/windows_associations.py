"""Per-user Windows registration for NNEditor model file types.

Windows 10 and 11 require the user to choose defaults through system UI.  The
registration here therefore advertises NNEditor as a capable ``Open with``
application and never overwrites the protected ``UserChoice`` default.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from nneditor.artifact_formats import MODEL_FILE_SUFFIXES

APPLICATION_NAME: Final = "NNEditor"
APPLICATION_DESCRIPTION: Final = (
    "Inspect, edit, optimize, and trace neural-network model artifacts."
)
PROG_ID: Final = "NNEditor.Model.1"
CAPABILITIES_PATH: Final = r"Software\NNEditor\Capabilities"
PROG_ID_PATH: Final = rf"Software\Classes\{PROG_ID}"
REGISTERED_APPLICATIONS_PATH: Final = r"Software\RegisteredApplications"


class FileAssociationError(OSError):
    """Windows file-association registration could not be completed."""


@dataclass(frozen=True, slots=True)
class RegistryValue:
    """One string value written beneath ``HKEY_CURRENT_USER``."""

    key: str
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class AssociationRegistration:
    """Complete, testable registry plan for one NNEditor launcher."""

    command: str
    icon: str
    extensions: tuple[str, ...]
    values: tuple[RegistryValue, ...]


def current_launch_command() -> tuple[str, ...]:
    """Return a stable command prefix that relaunches this NNEditor install."""
    executable = Path(sys.executable)
    if executable.name.lower() == "nneditor.exe" and executable.is_file():
        return (str(executable.resolve()),)
    sibling_script = executable.with_name("nneditor.exe")
    if sibling_script.is_file():
        return (str(sibling_script.resolve()),)
    argument_zero = Path(sys.argv[0]) if sys.argv else Path()
    if (
        argument_zero.suffix.lower() == ".exe"
        and argument_zero.name.lower().startswith("nneditor")
        and argument_zero.is_file()
    ):
        return (str(argument_zero.resolve()),)
    installed_script = shutil.which("nneditor")
    if installed_script is not None:
        return (str(Path(installed_script).resolve()),)
    return (str(executable.resolve()), "-m", "nneditor")


def association_registration(
    command_prefix: tuple[str, ...],
    *,
    icon_path: Path | str,
) -> AssociationRegistration:
    """Build the exact per-user registration without touching the registry."""
    if not command_prefix or any(not item for item in command_prefix):
        raise ValueError("the NNEditor launch command cannot be empty")
    command = " ".join(
        subprocess.list2cmdline((argument,)) for argument in command_prefix
    )
    command = f'{command} "%1"'
    icon = str(Path(icon_path).expanduser().resolve())
    values = [
        RegistryValue(PROG_ID_PATH, "", "Neural-network model artifact"),
        RegistryValue(rf"{PROG_ID_PATH}\DefaultIcon", "", icon),
        RegistryValue(rf"{PROG_ID_PATH}\shell", "", "open"),
        RegistryValue(
            rf"{PROG_ID_PATH}\shell\open", "FriendlyAppName", APPLICATION_NAME
        ),
        RegistryValue(rf"{PROG_ID_PATH}\shell\open\command", "", command),
        RegistryValue(CAPABILITIES_PATH, "ApplicationName", APPLICATION_NAME),
        RegistryValue(
            CAPABILITIES_PATH,
            "ApplicationDescription",
            APPLICATION_DESCRIPTION,
        ),
        RegistryValue(CAPABILITIES_PATH, "ApplicationIcon", icon),
        RegistryValue(
            REGISTERED_APPLICATIONS_PATH,
            APPLICATION_NAME,
            CAPABILITIES_PATH,
        ),
    ]
    for extension in MODEL_FILE_SUFFIXES:
        values.extend(
            (
                RegistryValue(
                    rf"Software\Classes\{extension}\OpenWithProgids",
                    PROG_ID,
                    "",
                ),
                RegistryValue(
                    rf"{CAPABILITIES_PATH}\FileAssociations",
                    extension,
                    PROG_ID,
                ),
            )
        )
    return AssociationRegistration(
        command=command,
        icon=icon,
        extensions=MODEL_FILE_SUFFIXES,
        values=tuple(values),
    )


def register_file_associations(
    command_prefix: tuple[str, ...] | None = None,
    *,
    icon_path: Path | str,
) -> AssociationRegistration:
    """Register NNEditor as an ``Open with`` candidate for the current user."""
    if sys.platform != "win32":
        raise FileAssociationError(
            "file-association registration is available only on Windows"
        )
    import winreg

    registration = association_registration(
        command_prefix or current_launch_command(),
        icon_path=icon_path,
    )
    try:
        for item in registration.values:
            with winreg.CreateKeyEx(
                winreg.HKEY_CURRENT_USER,
                item.key,
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.SetValueEx(key, item.name or None, 0, winreg.REG_SZ, item.value)
    except OSError as error:
        raise FileAssociationError(
            f"could not register NNEditor file types: {error}"
        ) from error
    _notify_shell()
    return registration


def unregister_file_associations() -> None:
    """Remove only NNEditor-owned registration values for the current user."""
    if sys.platform != "win32":
        raise FileAssociationError(
            "file-association registration is available only on Windows"
        )
    import winreg

    for extension in MODEL_FILE_SUFFIXES:
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                rf"Software\Classes\{extension}\OpenWithProgids",
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.DeleteValue(key, PROG_ID)
        except FileNotFoundError:
            pass
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            REGISTERED_APPLICATIONS_PATH,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.DeleteValue(key, APPLICATION_NAME)
    except FileNotFoundError:
        pass
    _delete_tree(winreg.HKEY_CURRENT_USER, CAPABILITIES_PATH)
    _delete_tree(winreg.HKEY_CURRENT_USER, PROG_ID_PATH)
    _notify_shell()


def open_default_apps_settings() -> None:
    """Open Windows Settings at NNEditor's registered default-app page."""
    if sys.platform != "win32":
        raise FileAssociationError("default-app settings are available only on Windows")
    target = (
        f"ms-settings:defaultapps?registeredAppUser={quote(APPLICATION_NAME, safe='')}"
    )
    try:
        os.startfile(target)
    except OSError as error:
        raise FileAssociationError(
            f"could not open Windows default-app settings: {error}"
        ) from error


def _delete_tree(root: int, path: str) -> None:
    if sys.platform != "win32":
        raise FileAssociationError(
            "file-association registration is available only on Windows"
        )
    import winreg

    try:
        with winreg.OpenKey(
            root,
            path,
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        ) as key:
            children: list[str] = []
            index = 0
            while True:
                try:
                    children.append(winreg.EnumKey(key, index))
                except OSError:
                    break
                index += 1
        for child in children:
            _delete_tree(root, rf"{path}\{child}")
        winreg.DeleteKey(root, path)
    except FileNotFoundError:
        pass


def _notify_shell() -> None:
    if sys.platform != "win32":
        raise FileAssociationError(
            "file-association registration is available only on Windows"
        )
    import ctypes

    # SHCNE_ASSOCCHANGED / SHCNF_IDLIST: invalidate Explorer's association
    # cache after the complete registration is in place.
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
