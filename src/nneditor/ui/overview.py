"""Reusable Flet controls for model and selection overviews."""

from __future__ import annotations

from pathlib import Path

import flet as ft

from nneditor.ir.core import Document
from nneditor.ui import viewmodel

__all__ = [
    "metadata_section",
    "model_overview_controls",
    "section_heading",
    "selected_block_controls",
    "selected_region_controls",
]

_ACCENT = "#7F56D9"
_ACCENT_SOFT = "#F4F3FF"
_BORDER = "#E4E7EC"
_DANGER = "#B42318"
_DANGER_SOFT = "#FEF3F2"
_INFO = "#175CD3"
_INFO_SOFT = "#EFF8FF"
_INK = "#101828"
_MUTED = "#667085"
_PANEL = "#FFFFFF"
_SUBTLE = "#F8FAFC"
_SUCCESS = "#067647"
_SUCCESS_SOFT = "#ECFDF3"
_WARNING = "#B54708"
_WARNING_SOFT = "#FFFAEB"


def section_heading(
    title: str,
    icon: ft.IconData,
    *,
    trailing: str | None = None,
) -> ft.Row:
    controls: list[ft.Control] = [
        ft.Icon(icon, size=17, color=_ACCENT),
        ft.Text(
            title,
            size=13,
            weight=ft.FontWeight.W_700,
            color=_INK,
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
                    color=_ACCENT,
                ),
                padding=ft.Padding.symmetric(horizontal=8, vertical=3),
                bgcolor=_ACCENT_SOFT,
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
    role_prefix: str = "overview-metric",
) -> ft.Container:
    return ft.Container(
        data=f"{role_prefix}:{label.lower().replace(' ', '-')}",
        col=6,
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(icon, size=15, color=_ACCENT),
                        ft.Text(
                            label.upper(),
                            size=9,
                            weight=ft.FontWeight.W_700,
                            color=_MUTED,
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(
                    value,
                    size=18,
                    weight=ft.FontWeight.W_700,
                    color=_INK,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    tooltip=value,
                ),
            ],
            spacing=7,
            tight=True,
        ),
        padding=12,
        bgcolor=_SUBTLE,
        border=ft.Border.all(1, _BORDER),
        border_radius=12,
    )


def metadata_section(
    items: tuple[tuple[str, str], ...],
    *,
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
                        color=_MUTED,
                    ),
                    ft.Text(
                        value,
                        size=11,
                        color=_INK,
                        selectable=True,
                        tooltip=value,
                    ),
                ],
                spacing=2,
                tight=True,
            )
        )
        if position < len(items) - 1:
            details.append(ft.Divider(height=10, thickness=1, color=_BORDER))
    return ft.Column(
        data=role,
        controls=[
            section_heading(title, icon),
            ft.Container(
                content=ft.Column(controls=details, spacing=0, tight=True),
                padding=12,
                bgcolor=_PANEL,
                border=ft.Border.all(1, _BORDER),
                border_radius=12,
            ),
        ],
        spacing=8,
        tight=True,
    )


def selected_block_controls(
    *,
    pattern: str,
    members: int,
    confidence: float,
    explanation: str,
) -> list[ft.Control]:
    return [
        ft.Column(
            data="selection-block-summary",
            controls=[
                section_heading("Block summary", ft.Icons.GRID_VIEW_ROUNDED),
                ft.ResponsiveRow(
                    data="selection-block-metrics",
                    controls=[
                        _metric_card(
                            "Operators",
                            f"{members:,}",
                            ft.Icons.HUB_ROUNDED,
                            role_prefix="selection-metric",
                        ),
                        _metric_card(
                            "Confidence",
                            f"{confidence:.0%}",
                            ft.Icons.VERIFIED_ROUNDED,
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
            title="Block metadata",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            role="selection-item-metadata",
        ),
    ]


def selected_region_controls(members: int) -> list[ft.Control]:
    return [
        ft.Column(
            data="selection-region-summary",
            controls=[
                section_heading("Region summary", ft.Icons.HUB_ROUNDED),
                ft.ResponsiveRow(
                    data="selection-region-metrics",
                    controls=[
                        _metric_card(
                            "Operators",
                            f"{members:,}",
                            ft.Icons.DATA_OBJECT_ROUNDED,
                            role_prefix="selection-metric",
                        ),
                        _metric_card(
                            "Next view",
                            "Blocks",
                            ft.Icons.GRID_VIEW_ROUNDED,
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
            title="Region metadata",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            role="selection-item-metadata",
        ),
    ]


def _capability_style(
    availability: str,
) -> tuple[str, str, ft.IconData, str]:
    styles = {
        "available": (
            _SUCCESS,
            _SUCCESS_SOFT,
            ft.Icons.CHECK_CIRCLE_ROUNDED,
            "Ready",
        ),
        "partial": (
            _WARNING,
            _WARNING_SOFT,
            ft.Icons.INFO_OUTLINE_ROUNDED,
            "Partial",
        ),
        "unavailable": (
            _DANGER,
            _DANGER_SOFT,
            ft.Icons.CANCEL_ROUNDED,
            "Unavailable",
        ),
        "requires trusted mode": (
            "#6941C6",
            "#F4F3FF",
            ft.Icons.LOCK_ROUNDED,
            "Trusted mode",
        ),
        "requires companion artifact": (
            _INFO,
            _INFO_SOFT,
            ft.Icons.LINK_ROUNDED,
            "Companion",
        ),
    }
    return styles.get(
        availability,
        (_MUTED, _SUBTLE, ft.Icons.HELP_OUTLINE_ROUNDED, availability.title()),
    )


def _capability_card(
    name: str,
    availability: str,
    reason: str,
) -> ft.Container:
    color, background, icon, status_label = _capability_style(availability)
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
                            color=_INK,
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
                ft.Text(reason, size=10, color=_MUTED, selectable=True),
            ],
            spacing=6,
            tight=True,
        ),
        padding=10,
        bgcolor=_PANEL,
        border=ft.Border.all(1, _BORDER),
        border_radius=10,
    )


def _capabilities_section(
    capabilities: tuple[tuple[str, str, str], ...],
) -> ft.Column:
    ready = sum(availability == "available" for _, availability, _ in capabilities)
    return ft.Column(
        data="model-overview-capabilities",
        controls=[
            section_heading(
                "Capabilities",
                ft.Icons.VERIFIED_USER_ROUNDED,
                trailing=f"{ready}/{len(capabilities)} ready",
            ),
            *[
                _capability_card(name, availability, reason)
                for name, availability, reason in capabilities
            ],
        ],
        spacing=7,
        tight=True,
    )


def _finding_card(severity: str, title: str, message: str) -> ft.Container:
    styles = {
        "error": (_DANGER, _DANGER_SOFT, ft.Icons.ERROR_OUTLINE_ROUNDED),
        "warning": (_WARNING, _WARNING_SOFT, ft.Icons.WARNING_AMBER_ROUNDED),
        "info": (_INFO, _INFO_SOFT, ft.Icons.INFO_OUTLINE_ROUNDED),
    }
    color, background, icon = styles.get(
        severity,
        (_MUTED, _SUBTLE, ft.Icons.INFO_OUTLINE_ROUNDED),
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
                            color=_INK,
                        ),
                        ft.Text(message, size=10, color=_MUTED, selectable=True),
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


def model_overview_controls(document: Document) -> list[ft.Control]:
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
            ),
            _metric_card(
                "Operators",
                f"{int(summary['Nodes']):,}",
                ft.Icons.HUB_ROUNDED,
            ),
            _metric_card(
                "Tensors",
                f"{int(summary['Tensors']):,}",
                ft.Icons.GRID_VIEW_ROUNDED,
            ),
            _metric_card(
                "Model size",
                viewmodel.compact_bytes(document.source.byte_size),
                ft.Icons.DATA_OBJECT_ROUNDED,
            ),
        ],
        spacing=8,
        run_spacing=8,
    )
    controls: list[ft.Control] = [
        ft.Column(
            data="model-overview-summary",
            controls=[
                section_heading("At a glance", ft.Icons.DASHBOARD_ROUNDED),
                metrics,
            ],
            spacing=8,
            tight=True,
        ),
        metadata_section(tuple(artifact_items)),
        _capabilities_section(viewmodel.capability_lines(document)),
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
                        trailing=str(len(findings)),
                    ),
                    *[
                        _finding_card(severity, title, message)
                        for severity, title, message in findings
                    ],
                ],
                spacing=7,
                tight=True,
            )
        )
    return controls
