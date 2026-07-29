"""Interactive layered tensor presentation for activation visualizations."""

from __future__ import annotations

import math
from collections.abc import Callable

import flet as ft

from nneditor.tracing.visualization import ActivationVisualization

__all__ = ["ActivationLayerViewer", "build_activation_layer_viewer"]


class ActivationLayerViewer:
    """A rotatable stack whose selected tensor layer is painted in front."""

    def __init__(
        self,
        view: ActivationVisualization,
        *,
        width: float,
        height: float,
        panel_color: str,
        border_color: str,
        accent_color: str,
        muted_color: str,
    ) -> None:
        if (
            not view.layer_pngs
            or len(view.layer_pngs) != len(view.layer_labels)
            or len(view.layer_pngs) != len(view.layer_indices)
            or view.layer_shape is None
        ):
            raise ValueError("layer-stack visualization data is incomplete")
        self.view = view
        self.width = width
        self.height = height
        self._panel_color = panel_color
        self._border_color = border_color
        self._accent_color = accent_color
        self._muted_color = muted_color
        self.pitch = -0.52
        self.yaw = 0.58
        self.selected = 0
        self._scene_height = max(120.0, height - 42.0)
        rows, columns = view.layer_shape
        available_width = max(60.0, width * 0.68)
        available_height = max(60.0, self._scene_height * 0.68)
        scale = min(
            available_width / max(columns, 1),
            available_height / max(rows, 1),
        )
        self._plane_width = max(30.0, columns * scale)
        self._plane_height = max(30.0, rows * scale)
        left = (width - self._plane_width) / 2
        top = (self._scene_height - self._plane_height) / 2
        self.planes: list[ft.GestureDetector] = []
        for position, (png, label) in enumerate(
            zip(view.layer_pngs, view.layer_labels, strict=True)
        ):
            card = ft.Container(
                content=ft.Image(
                    src=png,
                    width=self._plane_width,
                    height=self._plane_height,
                    fit=ft.BoxFit.FILL,
                    filter_quality=ft.FilterQuality.NONE,
                ),
                width=self._plane_width,
                height=self._plane_height,
                bgcolor=panel_color,
                border=ft.Border.all(1, border_color),
                border_radius=5,
            )
            plane = ft.GestureDetector(
                data=f"activation-layer-plane:{position}",
                content=card,
                width=self._plane_width,
                height=self._plane_height,
                left=left,
                top=top,
                mouse_cursor=ft.MouseCursor.CLICK,
                on_tap=self._select_handler(position),
                tooltip=f"Select {label}",
            )
            self.planes.append(plane)
        self.scene = ft.Stack(
            data="activation-layer-scene",
            controls=[],
            width=width,
            height=self._scene_height,
            clip_behavior=ft.ClipBehavior.NONE,
        )
        self.gesture = ft.GestureDetector(
            data="activation-layer-rotation",
            content=self.scene,
            width=width,
            height=self._scene_height,
            drag_interval=16,
            mouse_cursor=ft.MouseCursor.MOVE,
            on_pan_update=self._on_pan_update,
            tooltip="Drag to rotate; click a layer to bring it forward",
        )
        self.selection_text = ft.Text(
            data="activation-layer-selection",
            value="",
            size=10,
            color=muted_color,
            expand=True,
        )
        previous = ft.IconButton(
            data="activation-layer-previous",
            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,
            tooltip="Previous layer",
            on_click=self._step_handler(-1),
            disabled=len(self.planes) < 2,
        )
        following = ft.IconButton(
            data="activation-layer-next",
            icon=ft.Icons.CHEVRON_RIGHT_ROUNDED,
            tooltip="Next layer",
            on_click=self._step_handler(1),
            disabled=len(self.planes) < 2,
        )
        reset = ft.IconButton(
            data="activation-layer-reset",
            icon=ft.Icons.THREED_ROTATION_ROUNDED,
            tooltip="Reset rotation",
            on_click=self._reset_rotation,
        )
        self.control = ft.Container(
            data=f"activation-plot:{view.kind.value}",
            content=ft.Column(
                controls=[
                    self.gesture,
                    ft.Row(
                        controls=[previous, self.selection_text, following, reset],
                        spacing=2,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=2,
                tight=True,
            ),
            width=width,
            height=height,
            padding=6,
            bgcolor=panel_color,
            border=ft.Border.all(1, border_color),
            border_radius=8,
        )
        self.control.activation_layer_viewer = self  # type: ignore[attr-defined]
        self._apply_view()

    def _select_handler(
        self,
        position: int,
    ) -> Callable[[ft.TapEvent[ft.GestureDetector]], None]:
        def select(_event: ft.TapEvent[ft.GestureDetector]) -> None:
            self.select(position)

        return select

    def _step_handler(
        self,
        delta: int,
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        def step(_event: ft.Event[ft.IconButton]) -> None:
            self.select((self.selected + delta) % len(self.planes))

        return step

    def select(self, position: int) -> None:
        if not 0 <= position < len(self.planes):
            raise IndexError(position)
        self.selected = position
        self._apply_view()
        self._update()

    def rotate(self, delta_x: float, delta_y: float) -> None:
        self.yaw = (self.yaw + delta_x * 0.012) % (2 * math.pi)
        self.pitch = max(-1.35, min(1.35, self.pitch - delta_y * 0.012))
        self._apply_view()
        self._update()

    def _on_pan_update(
        self,
        event: ft.DragUpdateEvent[ft.GestureDetector],
    ) -> None:
        delta = event.local_delta
        if delta is not None:
            self.rotate(float(delta.x), float(delta.y))

    def _reset_rotation(self, _event: ft.Event[ft.IconButton]) -> None:
        self.pitch = -0.52
        self.yaw = 0.58
        self._apply_view()
        self._update()

    def _apply_view(self) -> None:
        order = [
            position
            for position in range(len(self.planes))
            if position != self.selected
        ]
        order.append(self.selected)
        self.scene.controls = [self.planes[position] for position in order]
        spacing = min(34.0, max(12.0, 150.0 / max(len(self.planes), 1)))
        for rank, position in enumerate(order):
            plane = self.planes[position]
            selected = position == self.selected
            depth = (
                spacing * (len(order) + 1)
                if selected
                else spacing * (rank - (len(order) - 1) / 2)
            )
            matrix = ft.Matrix4.identity()
            matrix.set_entry(3, 2, 0.0018)
            matrix.rotate_x(self.pitch)
            matrix.rotate_y(self.yaw)
            matrix.translate(0, 0, depth)
            plane.transform = ft.Transform(
                matrix=matrix,
                alignment=ft.Alignment.CENTER,
                transform_hit_tests=True,
                filter_quality=ft.FilterQuality.NONE,
            )
            plane.opacity = 1.0 if selected else 0.68
            card = plane.content
            if isinstance(card, ft.Container):
                card.border = ft.Border.all(
                    3 if selected else 1,
                    self._accent_color if selected else self._border_color,
                )
        label = self.view.layer_labels[self.selected]
        source_index = self.view.layer_indices[self.selected]
        self.selection_text.value = (
            f"Layer {self.selected + 1}/{len(self.planes)} · "
            f"source {source_index} · {label}"
        )

    def _update(self) -> None:
        try:
            self.control.update()
        except RuntimeError:
            # Headless tests and controls not mounted yet still need their
            # state mutated; Flet will serialize it when the control is added.
            return


def build_activation_layer_viewer(
    view: ActivationVisualization,
    *,
    width: float,
    height: float,
    panel_color: str,
    border_color: str,
    accent_color: str,
    muted_color: str,
) -> ft.Container:
    """Build an interactive stack and retain its controller on the root."""
    return ActivationLayerViewer(
        view,
        width=width,
        height=height,
        panel_color=panel_color,
        border_color=border_color,
        accent_color=accent_color,
        muted_color=muted_color,
    ).control
