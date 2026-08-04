"""Reusable Flet controls for model and selection overviews.

Every builder takes the shell's active :class:`ShellPalette` so the same
control renders legibly in both the light and dark themes; nothing in this
module bakes a theme-specific hex value.
"""

from __future__ import annotations

from pathlib import Path

import flet as ft

from nneditor.ir.core import Document
from nneditor.ui import viewmodel
from nneditor.ui.shell_layout import ShellPalette

__all__ = [
    "metadata_section",
    "model_overview_controls",
    "section_heading",
    "selected_block_controls",
    "selected_region_controls",
]


def section_heading(
    title: str,
    icon: ft.IconData,
    *,
    palette: ShellPalette,
    trailing: str | None = None,
) -> ft.Row:
    controls: list[ft.Control] = [
        ft.Icon(icon, size=17, color=palette.accent),
        ft.Text(
            title,
            size=13,
            weight=ft.FontWeight.W_700,
            color=palette.ink,
            expand=True,
        ),
    ]
    if trailing:
        controls.append(
            ft.Container(
                content=ft.Text(
                    trailing,
                    size=9,
                    weight=ft.FontWeight.W_700,
                    color=palette.accent,
                ),
                padding=ft.Padding.symmetric(horizontal=8, vertical=3),
                bgcolor=palette.accent_soft,
                border_radius=20,
            )
        )
    return ft.Row(
        controls=controls,
        spacing=7,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _metric_card(
    label: str,
    value: str,
    icon: ft.IconData,
    *,
    palette: ShellPalette,
    role_prefix: str = "overview-metric",
) -> ft.Container:
    return ft.Container(
        data=f"{role_prefix}:{label.lower().replace(' ', '-')}",
        col=6,
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(icon, size=15, color=palette.accent),
                        ft.Text(
                            label.upper(),
                            size=9,
                            weight=ft.FontWeight.W_700,
                            color=palette.muted,
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(
                    value,
                    size=18,
                    weight=ft.FontWeight.W_700,
                    color=palette.ink,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    tooltip=value,
                ),
            ],
            spacing=7,
            tight=True,
        ),
        padding=12,
        bgcolor=palette.subtle,
        border=ft.Border.all(1, palette.border),
        border_radius=12,
    )


def metadata_section(
    items: tuple[tuple[str, str], ...],
    *,
    palette: ShellPalette,
    title: str = "Artifact",
    icon: ft.IconData = ft.Icons.INSERT_DRIVE_FILE_ROUNDED,
    role: str = "model-overview-artifact",
) -> ft.Column:
    details: list[ft.Control] = []
    for position, (label, value) in enumerate(items):
        details.append(
            ft.Column(
                controls=[
                    ft.Text(
                        label.upper(),
                        size=9,
                        weight=ft.FontWeight.W_700,
                        color=palette.muted,
                    ),
                    ft.Text(
                        value,
                        size=11,
                        color=palette.ink,
                        selectable=True,
                        tooltip=value,
                    ),
                ],
                spacing=2,
                tight=True,
            )
        )
        if position < len(items) - 1:
            details.append(ft.Divider(height=10, thickness=1, color=palette.border))
    return ft.Column(
        data=role,
        controls=[
            section_heading(title, icon, palette=palette),
            ft.Container(
                content=ft.Column(controls=details, spacing=0, tight=True),
                padding=12,
                bgcolor=palette.panel,
                border=ft.Border.all(1, palette.border),
                border_radius=12,
            ),
        ],
        spacing=8,
        tight=True,
    )


def selected_block_controls(
    *,
    palette: ShellPalette,
    pattern: str,
    members: int,
    confidence: float,
    explanation: str,
) -> list[ft.Control]:
    return [
        ft.Column(
            data="selection-block-summary",
            controls=[
                section_heading(
                    "Block summary",
                    ft.Icons.GRID_VIEW_ROUNDED,
                    palette=palette,
                ),
                ft.ResponsiveRow(
                    data="selection-block-metrics",
                    controls=[
                        _metric_card(
                            "Operators",
                            f"{members:,}",
                            ft.Icons.HUB_ROUNDED,
                            palette=palette,
                            role_prefix="selection-metric",
                        ),
                        _metric_card(
                            "Confidence",
                            f"{confidence:.0%}",
                            ft.Icons.VERIFIED_ROUNDED,
                            palette=palette,
                            role_prefix="selection-metric",
                        ),
                    ],
                    spacing=8,
                    run_spacing=8,
                ),
            ],
            spacing=8,
            tight=True,
        ),
        metadata_section(
            (
                ("Pattern", viewmodel.humanize_identifier(pattern)),
                ("Explanation", explanation),
            ),
            palette=palette,
            title="Block metadata",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            role="selection-item-metadata",
        ),
    ]


def selected_region_controls(
    members: int,
    *,
    palette: ShellPalette,
) -> list[ft.Control]:
    return [
        ft.Column(
            data="selection-region-summary",
            controls=[
                section_heading(
                    "Region summary",
                    ft.Icons.HUB_ROUNDED,
                    palette=palette,
                ),
                ft.ResponsiveRow(
                    data="selection-region-metrics",
                    controls=[
                        _metric_card(
                            "Operators",
                            f"{members:,}",
                            ft.Icons.DATA_OBJECT_ROUNDED,
                            palette=palette,
                            role_prefix="selection-metric",
                        ),
                        _metric_card(
                            "Next view",
                            "Blocks",
                            ft.Icons.GRID_VIEW_ROUNDED,
                            palette=palette,
                            role_prefix="selection-metric",
                        ),
                    ],
                    spacing=8,
                    run_spacing=8,
                ),
            ],
            spacing=8,
            tight=True,
        ),
        metadata_section(
            (
                ("Region", "Architecture overview"),
                ("Next step", "Open this region to inspect its blocks"),
            ),
            palette=palette,
            title="Region metadata",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            role="selection-item-metadata",
        ),
    ]


def _capability_style(
    availability: str,
    palette: ShellPalette,
) -> tuple[str, str, ft.IconData, str]:
    styles = {
        "available": (
            palette.success,
            palette.success_soft,
            ft.Icons.CHECK_CIRCLE_ROUNDED,
            "Ready",
        ),
        "partial": (
            palette.warning,
            palette.warning_soft,
            ft.Icons.INFO_OUTLINE_ROUNDED,
            "Partial",
        ),
        "unavailable": (
            palette.danger,
            palette.danger_soft,
            ft.Icons.CANCEL_ROUNDED,
            "Unavailable",
        ),
        "requires trusted mode": (
            palette.accent,
            palette.accent_soft,
            ft.Icons.LOCK_ROUNDED,
            "Trusted mode",
        ),
        "requires companion artifact": (
            palette.info,
            palette.info_soft,
            ft.Icons.LINK_ROUNDED,
            "Companion",
        ),
    }
    return styles.get(
        availability,
        (
            palette.muted,
            palette.subtle,
            ft.Icons.HELP_OUTLINE_ROUNDED,
            availability.title(),
        ),
    )


def _capability_card(
    name: str,
    availability: str,
    reason: str,
    *,
    palette: ShellPalette,
) -> ft.Container:
    color, background, icon, status_label = _capability_style(availability, palette)
    return ft.Container(
        data=f"overview-capability:{name}",
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(icon, size=16, color=color),
                        ft.Text(
                            viewmodel.humanize_identifier(name),
                            size=11,
                            weight=ft.FontWeight.W_700,
                            color=palette.ink,
                            expand=True,
                        ),
                        ft.Container(
                            content=ft.Text(
                                status_label,
                                size=8,
                                weight=ft.FontWeight.W_700,
                                color=color,
                            ),
                            padding=ft.Padding.symmetric(horizontal=7, vertical=3),
                            bgcolor=background,
                            border_radius=20,
                        ),
                    ],
                    spacing=7,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(reason, size=10, color=palette.muted, selectable=True),
            ],
            spacing=6,
            tight=True,
        ),
        padding=10,
        bgcolor=palette.panel,
        border=ft.Border.all(1, palette.border),
        border_radius=10,
    )


def _capabilities_section(
    capabilities: tuple[tuple[str, str, str], ...],
    *,
    palette: ShellPalette,
) -> ft.Column:
    ready = sum(availability == "available" for _, availability, _ in capabilities)
    return ft.Column(
        data="model-overview-capabilities",
        controls=[
            section_heading(
                "Capabilities",
                ft.Icons.VERIFIED_USER_ROUNDED,
                palette=palette,
                trailing=f"{ready}/{len(capabilities)} ready",
            ),
            *[
                _capability_card(name, availability, reason, palette=palette)
                for name, availability, reason in capabilities
            ],
        ],
        spacing=7,
        tight=True,
    )


def _finding_card(
    severity: str,
    title: str,
    message: str,
    *,
    palette: ShellPalette,
) -> ft.Container:
    styles = {
        "error": (
            palette.danger,
            palette.danger_soft,
            ft.Icons.ERROR_OUTLINE_ROUNDED,
        ),
        "warning": (
            palette.warning,
            palette.warning_soft,
            ft.Icons.WARNING_AMBER_ROUNDED,
        ),
        "info": (palette.info, palette.info_soft, ft.Icons.INFO_OUTLINE_ROUNDED),
    }
    color, background, icon = styles.get(
        severity,
        (palette.muted, palette.subtle, ft.Icons.INFO_OUTLINE_ROUNDED),
    )
    return ft.Container(
        data=f"overview-finding:{severity}",
        content=ft.Row(
            controls=[
                ft.Icon(icon, size=17, color=color),
                ft.Column(
                    controls=[
                        ft.Text(
                            title,
                            size=11,
                            weight=ft.FontWeight.W_700,
                            color=palette.ink,
                        ),
                        ft.Text(message, size=10, color=palette.muted, selectable=True),
                    ],
                    spacing=2,
                    tight=True,
                    expand=True,
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
        padding=10,
        bgcolor=background,
        border=ft.Border.all(1, color),
        border_radius=10,
    )


def model_overview_controls(
    document: Document,
    *,
    palette: ShellPalette,
) -> list[ft.Control]:
    summary = dict(viewmodel.model_summary(document))
    artifact_path = summary["Artifact"]
    artifact_items: list[tuple[str, str]] = [
        ("File", Path(artifact_path).name or artifact_path),
        ("Location", artifact_path),
        ("Format", viewmodel.humanize_identifier(summary["Kind"])),
    ]
    if producer := summary.get("Producer"):
        artifact_items.append(("Producer", producer))
    artifact_items.append(("Fingerprint", summary["Content hash"]))

    metrics = ft.ResponsiveRow(
        data="model-overview-metrics",
        controls=[
            _metric_card(
                "Graphs",
                f"{int(summary['Graphs']):,}",
                ft.Icons.ACCOUNT_TREE_ROUNDED,
                palette=palette,
            ),
            _metric_card(
                "Operators",
                f"{int(summary['Nodes']):,}",
                ft.Icons.HUB_ROUNDED,
                palette=palette,
            ),
            _metric_card(
                "Tensors",
                f"{int(summary['Tensors']):,}",
                ft.Icons.GRID_VIEW_ROUNDED,
                palette=palette,
            ),
            _metric_card(
                "Model size",
                viewmodel.compact_bytes(document.source.byte_size),
                ft.Icons.DATA_OBJECT_ROUNDED,
                palette=palette,
            ),
        ],
        spacing=8,
        run_spacing=8,
    )
    controls: list[ft.Control] = [
        ft.Column(
            data="model-overview-summary",
            controls=[
                section_heading(
                    "At a glance",
                    ft.Icons.DASHBOARD_ROUNDED,
                    palette=palette,
                ),
                metrics,
            ],
            spacing=8,
            tight=True,
        ),
        metadata_section(tuple(artifact_items), palette=palette),
        _capabilities_section(viewmodel.capability_lines(document), palette=palette),
    ]
    findings = viewmodel.diagnostic_lines(document)
    if findings:
        controls.append(
            ft.Column(
                data="model-overview-findings",
                controls=[
                    section_heading(
                        "Findings",
                        ft.Icons.RULE_ROUNDED,
                        palette=palette,
                        trailing=str(len(findings)),
                    ),
                    *[
                        _finding_card(severity, title, message, palette=palette)
                        for severity, title, message in findings
                    ],
                ],
                spacing=7,
                tight=True,
            )
        )
    return controls
