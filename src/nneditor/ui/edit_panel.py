"""Transactional graph-edit controls and request construction.

The application shell coordinates commits and navigation. This component owns
the edit form itself, including the translation from retained control values
to a typed validation request.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft

from nneditor.editing.validation import (
    EditRequest,
    InsertUnaryRequest,
    ReconnectInputRequest,
    RemoveUnaryRequest,
    RenameNodeRequest,
    ReplaceOperatorRequest,
    SetAttributeRequest,
    parse_attribute_value,
)
from nneditor.ir.core import AttrKind

__all__ = ["EditPanel"]

FilledButtonHandler = Callable[[ft.Event[ft.Button]], Any]
TextButtonHandler = Callable[[ft.Event[ft.TextButton]], Any]
OptionalTextButtonHandler = Callable[[ft.Event[ft.TextButton] | None], Any]


class EditPanel:
    """Retained controls for one transactional graph-edit workspace."""

    def __init__(
        self,
        *,
        on_validate: FilledButtonHandler,
        on_commit: FilledButtonHandler,
        on_reject: TextButtonHandler,
        on_undo: OptionalTextButtonHandler,
        on_redo: OptionalTextButtonHandler,
        on_export: TextButtonHandler,
    ) -> None:
        self.kind = ft.Dropdown(
            value="rename",
            label="Edit command",
            dense=True,
            options=[
                ft.DropdownOption(key="rename", text="Rename node"),
                ft.DropdownOption(key="attribute", text="Set attribute"),
                ft.DropdownOption(key="operator", text="Replace operator"),
                ft.DropdownOption(key="insert", text="Insert unary"),
                ft.DropdownOption(key="remove", text="Remove unary"),
                ft.DropdownOption(key="reconnect", text="Reconnect input"),
            ],
        )
        self.primary = ft.TextField(label="Name / operator / value ID", dense=True)
        self.secondary = ft.TextField(label="Value / domain", dense=True)
        self.attribute_kind = ft.Dropdown(
            value=AttrKind.STRING.value,
            label="Attribute type",
            dense=True,
            options=[
                ft.DropdownOption(key=kind.value, text=kind.value)
                for kind in (
                    AttrKind.INT,
                    AttrKind.FLOAT,
                    AttrKind.STRING,
                    AttrKind.INTS,
                    AttrKind.FLOATS,
                    AttrKind.STRINGS,
                )
            ],
        )
        self.port = ft.TextField(label="Input port", value="0", dense=True)
        self.validate_button = ft.FilledButton(
            content="Validate", on_click=on_validate, disabled=True
        )
        self.commit_button = ft.FilledButton(
            content="Commit", on_click=on_commit, disabled=True
        )
        self.reject_button = ft.TextButton(
            content="Reject", on_click=on_reject, disabled=True
        )
        self.undo_button = ft.TextButton(
            content="Undo", on_click=on_undo, disabled=True
        )
        self.redo_button = ft.TextButton(
            content="Redo", on_click=on_redo, disabled=True
        )
        self.export_button = ft.TextButton(
            content="Export…", on_click=on_export, disabled=True
        )
        self.findings = ft.ListView(height=90, spacing=1)
        self.control = ft.Column(
            controls=[
                self.kind,
                self.primary,
                self.secondary,
                ft.Row(controls=[self.attribute_kind, self.port], spacing=6),
                ft.Row(
                    controls=[
                        self.validate_button,
                        self.commit_button,
                        self.reject_button,
                    ],
                    wrap=True,
                    spacing=4,
                ),
                ft.Row(
                    controls=[
                        self.undo_button,
                        self.redo_button,
                        self.export_button,
                    ],
                    wrap=True,
                    spacing=2,
                ),
                self.findings,
            ],
            spacing=8,
        )

    @property
    def text_fields(self) -> tuple[ft.TextField, ...]:
        """Fields whose focus suppresses global keyboard navigation."""
        return (self.primary, self.secondary, self.port)

    def request(self, graph_id: str, node_id: str) -> EditRequest:
        """Build the typed edit request represented by the current form."""
        kind = self.kind.value or "rename"
        primary = (self.primary.value or "").strip()
        secondary = self.secondary.value or ""
        port = int(self.port.value or "0")
        if kind == "rename":
            return RenameNodeRequest(graph_id, node_id, primary)
        if kind == "attribute":
            attribute_kind = AttrKind(
                self.attribute_kind.value or AttrKind.STRING.value
            )
            return SetAttributeRequest(
                graph_id,
                node_id,
                primary,
                attribute_kind,
                parse_attribute_value(attribute_kind, secondary),
            )
        if kind == "operator":
            return ReplaceOperatorRequest(
                graph_id,
                node_id,
                primary,
                secondary.strip(),
            )
        if kind == "insert":
            return InsertUnaryRequest(
                graph_id,
                node_id,
                port,
                primary,
                secondary.strip(),
            )
        if kind == "remove":
            return RemoveUnaryRequest(graph_id, node_id)
        return ReconnectInputRequest(graph_id, node_id, port, primary)
