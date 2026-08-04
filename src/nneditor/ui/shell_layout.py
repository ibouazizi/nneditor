"""Declarative Flet composition for the application shell.

The shell controller owns state and event handling.  This module owns the
static arrangement of those controls, keeping layout construction out of the
controller's already large initialization path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import flet as ft

__all__ = [
    "DARK_SHELL_PALETTE",
    "ShellPalette",
    "build_empty_state",
    "build_hierarchy_tools",
    "build_left_panel",
    "build_loading_overlay",
    "build_right_panel",
    "build_surface",
]

EventHandler = Callable[..., Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class ShellPalette:
    panel: str
    border: str
    accent: str
    accent_soft: str
    ink: str
    muted: str
    canvas: str
    sidebar_width: int
    # Semantic states shared by the panels that report capture, consent, and
    # validation outcomes; they belong to the palette so no panel has to
    # redeclare a hex value the rest of the shell already uses.
    subtle: str = "#F8FAFC"
    info: str = "#175CD3"
    info_soft: str = "#EFF8FF"
    success: str = "#027A48"
    success_soft: str = "#ECFDF3"
    warning: str = "#B54708"
    warning_soft: str = "#FFFAEB"
    warning_border: str = "#FEDF89"
    danger: str = "#B42318"
    danger_soft: str = "#FEF3F2"
    # The translucent wash behind the modal loading card; canvas-tinted so
    # the workspace stays recognizable underneath it.
    scrim: str = "#A6F6F7FB"


DARK_SHELL_PALETTE = ShellPalette(
    panel="#1B2130",
    border="#2E3A50",
    accent="#8E8FF2",
    accent_soft="#2B2D55",
    ink="#E6EAF2",
    muted="#98A2B3",
    canvas="#10131C",
    sidebar_width=304,
    subtle="#232B3D",
    info="#7CB3F5",
    info_soft="#17293F",
    success="#4CC38A",
    success_soft="#12291E",
    warning="#F0B25C",
    warning_soft="#33270F",
    warning_border="#6B5320",
    danger="#F97066",
    danger_soft="#3A1A18",
    scrim="#A610131C",
)
"""Dark counterparts for every palette field.

The semantic states swap roles with their soft variants relative to light
mode: the text colors are the light tints (readable on dark panels) and the
soft backgrounds are deep washes of the same hue, keeping the same
foreground-on-soft pairings legible in both themes."""


def build_left_panel(
    *,
    palette: ShellPalette,
    inspector_title: ft.Text,
    inspector_subtitle: ft.Text,
    open_selection_button: ft.FilledButton,
    back_to_parent_button: ft.TextButton,
    inspector: ft.ListView,
    trace_controls: ft.Column,
    edit_controls: ft.Column,
    transformation_controls: ft.Column,
) -> ft.Container:
    """Compose the inspector and edit sidebar from stateful child controls."""
    return ft.Container(
        width=palette.sidebar_width,
        bgcolor=palette.panel,
        border=ft.Border.only(right=ft.BorderSide(1, palette.border)),
        padding=ft.Padding.only(left=18, right=18, top=18, bottom=12),
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Icon(
                                ft.Icons.INFO_OUTLINE_ROUNDED,
                                size=19,
                                color=palette.accent,
                            ),
                            width=34,
                            height=34,
                            alignment=ft.Alignment.CENTER,
                            bgcolor=palette.accent_soft,
                            border_radius=10,
                        ),
                        ft.Column(
                            controls=[inspector_title, inspector_subtitle],
                            spacing=0,
                            expand=True,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Row(
                    controls=[open_selection_button, back_to_parent_button],
                    wrap=True,
                    spacing=4,
                ),
                ft.Divider(height=12, color=palette.border),
                ft.Container(content=inspector, expand=True),
                ft.ExpansionTile(
                    data="trace-controls",
                    title=ft.Text("Trace activations", size=12),
                    leading=ft.Icon(ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED, size=17),
                    controls=[trace_controls],
                    controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
                    dense=True,
                ),
                ft.ExpansionTile(
                    title=ft.Text("Edit model", size=12),
                    leading=ft.Icon(ft.Icons.EDIT_ROUNDED, size=17),
                    controls=[edit_controls],
                    controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
                    dense=True,
                ),
                ft.ExpansionTile(
                    title=ft.Text("Optimize weights", size=12),
                    leading=ft.Icon(ft.Icons.TUNE_ROUNDED, size=17),
                    controls=[transformation_controls],
                    controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
                    dense=True,
                ),
            ],
            expand=True,
            spacing=8,
        ),
    )


def build_hierarchy_tools(
    *,
    multi_select_field: ft.Checkbox,
    group_label_field: ft.TextField,
    group_button: ft.TextButton,
    merge_button: ft.TextButton,
    rename_button: ft.TextButton,
    split_button: ft.TextButton,
    lock_button: ft.TextButton,
    reject_button: ft.TextButton,
    reset_groups_button: ft.TextButton,
) -> ft.ExpansionTile:
    """Compose the manual hierarchy command group."""
    return ft.ExpansionTile(
        title=ft.Text("Hierarchy tools", size=12),
        leading=ft.Icon(ft.Icons.TUNE_ROUNDED, size=17),
        dense=True,
        controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
        controls=[
            ft.Column(
                controls=[
                    multi_select_field,
                    group_label_field,
                    ft.Row(
                        controls=[group_button, merge_button, rename_button],
                        wrap=True,
                        spacing=2,
                    ),
                    ft.Row(
                        controls=[
                            split_button,
                            lock_button,
                            reject_button,
                            reset_groups_button,
                        ],
                        wrap=True,
                        spacing=2,
                    ),
                ],
                spacing=4,
            )
        ],
    )


def build_right_panel(
    *,
    palette: ShellPalette,
    search_field: ft.TextField,
    search_results: ft.ListView,
    graph_list: ft.ListView,
    hierarchy_list: ft.ListView,
    hierarchy_tools: ft.ExpansionTile,
    minimap: ft.GestureDetector,
) -> ft.Container:
    """Compose the graph, hierarchy, search, and minimap sidebar."""
    return ft.Container(
        width=palette.sidebar_width,
        bgcolor=palette.panel,
        border=ft.Border.only(left=ft.BorderSide(1, palette.border)),
        padding=ft.Padding.only(left=16, right=16, top=18, bottom=12),
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(
                            ft.Icons.ACCOUNT_TREE_ROUNDED,
                            color=palette.accent,
                            size=20,
                        ),
                        ft.Text(
                            "Model explorer",
                            size=16,
                            weight=ft.FontWeight.W_700,
                            color=palette.ink,
                        ),
                    ],
                    spacing=8,
                ),
                search_field,
                ft.Container(content=search_results, height=92),
                ft.Text(
                    "GRAPHS",
                    size=10,
                    weight=ft.FontWeight.W_700,
                    color=palette.muted,
                ),
                ft.Container(content=graph_list, height=108),
                ft.Text(
                    "BLOCKS",
                    size=10,
                    weight=ft.FontWeight.W_700,
                    color=palette.muted,
                ),
                ft.Container(content=hierarchy_list, expand=True),
                hierarchy_tools,
                ft.Text(
                    "MINIMAP",
                    size=10,
                    weight=ft.FontWeight.W_700,
                    color=palette.muted,
                ),
                ft.Container(
                    content=minimap,
                    padding=6,
                    bgcolor=palette.canvas,
                    border=ft.Border.all(1, palette.border),
                    border_radius=10,
                    alignment=ft.Alignment.CENTER,
                ),
            ],
            expand=True,
            spacing=9,
        ),
    )


def build_empty_state(
    *,
    palette: ShellPalette,
    on_open: EventHandler,
) -> ft.Container:
    """Compose the no-artifact workspace state."""
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Container(
                    content=ft.Icon(
                        ft.Icons.HUB_ROUNDED,
                        size=34,
                        color=palette.accent,
                    ),
                    width=64,
                    height=64,
                    alignment=ft.Alignment.CENTER,
                    bgcolor=palette.accent_soft,
                    border_radius=18,
                ),
                ft.Text(
                    "Explore a neural network",
                    size=22,
                    weight=ft.FontWeight.W_700,
                    color=palette.ink,
                ),
                ft.Text(
                    "Open a supported model artifact to inspect its available\n"
                    "architecture, weights, metadata, and capabilities.",
                    size=13,
                    color=palette.muted,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.FilledButton(
                    content="Open model",
                    icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                    on_click=on_open,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
            spacing=12,
        ),
        alignment=ft.Alignment.CENTER,
        expand=True,
    )


def build_loading_overlay(
    *,
    palette: ShellPalette,
    title: ft.Text,
    stage: ft.Text,
    cancel_button: ft.TextButton,
) -> ft.Container:
    """Compose the cancellable asynchronous-open overlay."""
    return ft.Container(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.ProgressRing(
                        width=40,
                        height=40,
                        stroke_width=3,
                        color=palette.accent,
                    ),
                    title,
                    stage,
                    cancel_button,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True,
                spacing=12,
            ),
            width=380,
            padding=28,
            bgcolor=palette.panel,
            border=ft.Border.all(1, palette.border),
            border_radius=18,
            shadow=ft.BoxShadow(
                blur_radius=36,
                color="#260F172A",
                offset=ft.Offset(0, 12),
            ),
        ),
        alignment=ft.Alignment.CENTER,
        bgcolor=palette.scrim,
        expand=True,
        visible=False,
    )


def build_surface(
    *,
    palette: ShellPalette,
    renderer_control: ft.Control,
    graph_actions: ft.Stack,
    empty_state: ft.Container,
    hover_card: ft.Container,
    loading_overlay: ft.Container,
    on_size_change: EventHandler,
) -> ft.Container:
    """Compose the renderer and its transient overlay layers."""
    return ft.Container(
        content=ft.Stack(
            controls=[
                renderer_control,
                graph_actions,
                empty_state,
                hover_card,
                loading_overlay,
            ],
            expand=True,
        ),
        expand=True,
        bgcolor=palette.canvas,
        on_size_change=on_size_change,
    )
