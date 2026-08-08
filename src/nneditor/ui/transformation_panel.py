"""Quantization and pruning controls for the operations drawer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft

from nneditor.transformations.engine import TransformationRequest
from nneditor.transformations.schema import Granularity
from nneditor.ui import viewmodel

__all__ = ["TransformationPanel"]

FilledButtonHandler = Callable[[ft.Event[ft.Button]], Any]
TextButtonHandler = Callable[[ft.Event[ft.TextButton]], Any]


class TransformationPanel:
    """Retained transformation controls and request translation."""

    def __init__(
        self,
        *,
        on_preview: FilledButtonHandler,
        on_commit: FilledButtonHandler,
        on_reject: TextButtonHandler,
    ) -> None:
        self.kind = ft.Dropdown(
            value="weight-quantization",
            label="Transformation",
            dense=True,
            options=[
                ft.DropdownOption(
                    key="weight-quantization", text="8-bit dequantized weight"
                ),
                ft.DropdownOption(key="graph-quantization", text="ONNX Q/DQ"),
                ft.DropdownOption(key="threshold-pruning", text="Threshold pruning"),
                ft.DropdownOption(key="mask-pruning", text="Explicit mask pruning"),
                ft.DropdownOption(key="nm-pruning", text="2:4 logical pruning"),
                ft.DropdownOption(
                    key="structured-pruning", text="Terminal MatMul channels"
                ),
            ],
        )
        self.granularity = ft.Dropdown(
            value=Granularity.PER_TENSOR.value,
            label="Quantization granularity",
            dense=True,
            options=[
                ft.DropdownOption(key=Granularity.PER_TENSOR.value, text="Per tensor"),
                ft.DropdownOption(
                    key=Granularity.PER_CHANNEL.value, text="Per channel"
                ),
            ],
        )
        self.axis = ft.TextField(label="Channel axis", value="0", dense=True)
        self.parameter = ft.TextField(
            label="Threshold / mask / kept channels", value="0.1", dense=True
        )
        self.preview_button = ft.FilledButton(
            content="Preview", on_click=on_preview, disabled=True
        )
        self.commit_button = ft.FilledButton(
            content="Apply", on_click=on_commit, disabled=True
        )
        self.reject_button = ft.TextButton(
            content="Reject", on_click=on_reject, disabled=True
        )
        self.findings = ft.ListView(height=125, spacing=1)
        self.control = ft.Column(
            controls=[
                self.kind,
                self.granularity,
                ft.Row(controls=[self.axis, self.parameter], spacing=6),
                ft.Row(
                    controls=[
                        self.preview_button,
                        self.commit_button,
                        self.reject_button,
                    ],
                    wrap=True,
                    spacing=4,
                ),
                self.findings,
            ],
            spacing=8,
        )

    @property
    def text_fields(self) -> tuple[ft.TextField, ...]:
        """Fields whose focus suppresses global keyboard navigation."""
        return (self.axis, self.parameter)

    @property
    def selected_kind(self) -> str:
        return self.kind.value or "weight-quantization"

    def request(
        self,
        *,
        graph_id: str,
        node_id: str,
        tensor_id: str | None,
    ) -> TransformationRequest:
        """Build the typed transformation represented by the current form."""
        if self.selected_kind != "structured-pruning" and tensor_id is None:
            raise ValueError("the selected operator has no initializer input")
        return viewmodel.transformation_request(
            kind=self.selected_kind,
            graph_id=graph_id,
            node_id=node_id,
            tensor_id=tensor_id,
            granularity_value=(self.granularity.value or Granularity.PER_TENSOR.value),
            axis_value=self.axis.value or "0",
            parameter=(self.parameter.value or "").strip(),
        )
