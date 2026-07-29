"""Desktop operating-system integration."""

from nneditor.desktop.windows_associations import (
    AssociationRegistration,
    FileAssociationError,
    association_registration,
    current_launch_command,
    open_default_apps_settings,
    register_file_associations,
    unregister_file_associations,
)

__all__ = [
    "AssociationRegistration",
    "FileAssociationError",
    "association_registration",
    "current_launch_command",
    "open_default_apps_settings",
    "register_file_associations",
    "unregister_file_associations",
]
