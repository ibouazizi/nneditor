"""Interactive Hugging Face snapshot explorer for the Model section.

The panel keeps its search field and pane choice across inspector rebuilds.
Clicking a tensor or block asks the shell to focus that part of the graph;
expanding a config value or tokenizer preview asks the shell for a dialog.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import flet as ft

from nneditor.ir.core import Document
from nneditor.ui import overview
from nneditor.ui.hf_view import (
    HfConfigRow,
    HfContainerSummary,
    HfLayoutSegment,
    HfMemberRow,
    HfTensorRow,
    hf_config_rows,
    hf_container_summary,
    hf_layout_segments,
    hf_member_rows,
    hf_tensor_catalog,
    hf_tokenizer_rows,
    matches_query,
)
from nneditor.ui.shell_layout import ShellPalette
from nneditor.ui.viewmodel import compact_bytes

__all__ = ["HfContainerPanel"]

_MAX_ROWS = 64
ClickHandler = Callable[[ft.Event[Any]], None]
_PANES: tuple[tuple[str, str, ft.IconData], ...] = (
    ("layout", "Files", ft.Icons.FOLDER_ROUNDED),
    ("config", "Config", ft.Icons.TUNE_ROUNDED),
    ("tensors", "Tensors", ft.Icons.GRID_VIEW_ROUNDED),
    ("tokenizer", "Tokenizer", ft.Icons.TEXT_FIELDS_ROUNDED),
)
_LAYOUT_COLORS = {
    "config": "accent",
    "tokenizer": "info",
    "index": "success",
    "weights": "warning",
    "pickle": "muted",
    "other": "muted",
}


class HfContainerPanel:
    """Retained Hugging Face explorer: search, pane chips, and a catalog."""

    def __init__(
        self,
        *,
        palette: ShellPalette,
        on_select_tensor: Callable[[str], None],
        on_select_block: Callable[[int], None],
        on_expand_config: Callable[[HfConfigRow], None],
        on_preview_tokenizer: Callable[[], None],
        update: Callable[[], None],
    ) -> None:
        self.palette = palette
        self._on_select_tensor = on_select_tensor
        self._on_select_block = on_select_block
        self._on_expand_config = on_expand_config
        self._on_preview_tokenizer = on_preview_tokenizer
        self._update = update
        self.document: Document | None = None
        self._bound_hash = ""
        self.pane = "tensors"
        self.query = ""
        self.dtype_filter = ""
        self.expanded_groups: set[str] = set()
        self.row_limit = _MAX_ROWS
        self.search = ft.TextField(
            label="Search the snapshot",
            hint_text="tensor name, config key, file…",
            dense=True,
            on_change=self._on_search,
            data="hf-search",
        )
        self.list = ft.Column(spacing=4, tight=True, data="hf-catalog")
        self.layout_bar = ft.Row(spacing=3, data="hf-layout")
        self.pane_row = ft.Row(spacing=4, wrap=True, data="hf-panes")
        self.root = ft.Column(
            data="hf-explorer",
            spacing=8,
            tight=True,
            controls=[],
        )

    def set_palette(self, palette: ShellPalette) -> None:
        self.palette = palette

    def reset(self) -> None:
        self.document = None
        self._bound_hash = ""
        self.pane = "tensors"
        self.query = ""
        self.dtype_filter = ""
        self.expanded_groups.clear()
        self.row_limit = _MAX_ROWS
        self.search.value = ""
        self.list.controls = []
        self.root.controls = []

    def refresh(self, document: Document) -> None:
        """Rebuild against ``document``, resetting filters when it changes."""
        identity = document.source.content_hash
        if identity != self._bound_hash:
            self._bound_hash = identity
            self.pane = "tensors"
            self.query = ""
            self.dtype_filter = ""
            self.expanded_groups.clear()
            self.row_limit = _MAX_ROWS
            self.search.value = ""
        self.document = document
        self._rebuild()

    def _on_search(self, event: ft.Event[ft.TextField]) -> None:
        self.query = str(event.control.value or "")
        self.row_limit = _MAX_ROWS
        self._rebuild_catalog()
        self._update()

    def _set_pane(self, pane: str) -> ClickHandler:
        def handle(_event: ft.Event[ft.Control]) -> None:
            self.pane = pane
            self.row_limit = _MAX_ROWS
            if pane != "tensors":
                self.dtype_filter = ""
            self._rebuild()
            self._update()

        return handle

    def _show_more(self, _event: ft.Event[ft.Control]) -> None:
        self.row_limit += _MAX_ROWS
        self._rebuild_catalog()
        self._update()

    def _set_dtype(self, dtype: str) -> ClickHandler:
        def handle(_event: ft.Event[ft.Control]) -> None:
            self.dtype_filter = "" if self.dtype_filter == dtype else dtype
            self._rebuild_catalog()
            self._update()

        return handle

    def _click_layout(self, key: str) -> ClickHandler:
        def handle(_event: ft.Event[ft.Control]) -> None:
            self.pane = {
                "config": "config",
                "tokenizer": "tokenizer",
                "index": "layout",
                "weights": "tensors",
                "pickle": "layout",
                "other": "layout",
            }.get(key, "layout")
            self._rebuild()
            self._update()

        return handle

    def _toggle_group(self, group: str) -> ClickHandler:
        def handle(_event: ft.Event[ft.Control]) -> None:
            if group in self.expanded_groups:
                self.expanded_groups.discard(group)
            else:
                self.expanded_groups.add(group)
            self._rebuild_catalog()
            self._update()

        return handle

    def _rebuild(self) -> None:
        document = self.document
        summary = hf_container_summary(document) if document is not None else None
        if document is None or summary is None:
            self.root.controls = []
            return
        segments = hf_layout_segments(document)
        self.layout_bar.controls = _layout_controls(
            segments, self.palette, self._click_layout
        )
        self.pane_row.controls = _pane_chips(self.pane, self.palette, self._set_pane)
        trailing = summary.architecture
        if summary.n_shards > 1:
            trailing = f"{trailing} · {summary.n_shards} shards"
        heading = overview.section_heading(
            "Hugging Face snapshot",
            ft.Icons.INVENTORY_2_ROUNDED,
            palette=self.palette,
            trailing=trailing,
        )
        self._rebuild_catalog()
        self.root.controls = [
            heading,
            self.layout_bar,
            _layout_legend(segments, self.palette),
            self.pane_row,
            self.search,
            _hint(
                "Expand a group to browse its weights, or search by tensor name.",
                self.palette,
            ),
            self.list,
        ]

    def _rebuild_catalog(self) -> None:
        document = self.document
        if document is None:
            self.list.controls = []
            return
        if self.pane == "layout":
            summary = hf_container_summary(document)
            self.list.controls = (
                _layout_rows(
                    summary,
                    [
                        row
                        for row in hf_member_rows(document)
                        if matches_query(self.query, row.name, row.kind)
                    ],
                    self.palette,
                )
                if summary is not None
                else []
            )
            return
        if self.pane == "tensors":
            self.list.controls = self._tensor_controls()
            return
        rows = (
            hf_tokenizer_rows(document)
            if self.pane == "tokenizer"
            else hf_config_rows(document)
        )
        self.list.controls = self._config_controls(rows)

    def _config_controls(self, rows: Sequence[HfConfigRow]) -> list[ft.Control]:
        visible = [
            row
            for row in rows
            if matches_query(self.query, row.key, row.value_text, row.group)
        ]
        if not visible:
            empty = (
                "No tokenizer files in this snapshot."
                if self.pane == "tokenizer"
                else "No config keys match the search."
            )
            return [_hint(empty, self.palette)]
        grouped: dict[str, list[HfConfigRow]] = {}
        for row in visible:
            grouped.setdefault(row.group, []).append(row)
        controls: list[ft.Control] = []
        shown = 0
        for group, items in grouped.items():
            open_group = (
                group in self.expanded_groups
                or len(grouped) == 1
                or bool(self.query.strip())
            )
            controls.append(
                _group_header(
                    group,
                    len(items),
                    open_group,
                    self.palette,
                    self._toggle_group(group),
                )
            )
            if not open_group:
                continue
            for row in items:
                shown += 1
                if shown <= self.row_limit:
                    controls.append(self._config_card(row))
        return _with_overflow(
            controls, shown, self.palette, self.row_limit, self._show_more
        )

    def _config_card(self, row: HfConfigRow) -> ft.Control:
        actions: list[ft.Control] = []
        if self.pane == "tokenizer" and row.key == "vocab":
            actions.append(
                ft.TextButton(
                    content="Preview",
                    icon=ft.Icons.PLAYLIST_PLAY_ROUNDED,
                    data="hf-tokenizer-preview",
                    tooltip="Load the first vocabulary tokens",
                    on_click=lambda _e: self._on_preview_tokenizer(),
                )
            )
        if row.expandable:
            actions.append(
                ft.TextButton(
                    content="View",
                    icon=ft.Icons.OPEN_IN_FULL_ROUNDED,
                    data=f"hf-config-view:{row.key}",
                    on_click=lambda _e, current=row: self._on_expand_config(current),
                )
            )
        return ft.Container(
            data=f"hf-config:{row.key}",
            content=ft.Column(
                controls=[
                    ft.Text(
                        row.key,
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=self.palette.ink,
                        selectable=True,
                    ),
                    ft.Text(
                        row.value_text,
                        size=10,
                        color=self.palette.ink,
                        selectable=True,
                    ),
                    ft.Text(
                        row.detail,
                        size=9,
                        color=self.palette.muted,
                    ),
                    *actions,
                ],
                spacing=2,
                tight=True,
            ),
            padding=8,
            bgcolor=self.palette.panel,
            border=ft.Border.all(1, self.palette.border),
            border_radius=8,
        )

    def _tensor_controls(self) -> list[ft.Control]:
        document = self.document
        assert document is not None
        catalog = [
            row
            for row in hf_tensor_catalog(document)
            if matches_query(
                self.query, row.name, row.element_type, row.group, row.shape, row.shard
            )
            and (not self.dtype_filter or row.element_type == self.dtype_filter)
        ]
        types = sorted({row.element_type for row in hf_tensor_catalog(document)})
        controls: list[ft.Control] = []
        if len(types) > 1:
            controls.append(
                ft.Row(
                    controls=[
                        _filter_chip(
                            dtype,
                            self.dtype_filter == dtype,
                            self.palette,
                            self._set_dtype(dtype),
                            data=f"hf-dtype:{dtype}",
                        )
                        for dtype in types
                    ],
                    wrap=True,
                    spacing=4,
                    data="hf-dtype-filters",
                )
            )
        if not catalog:
            controls.append(
                _hint(
                    "No tensors match the search."
                    if document.tensors
                    else "No weights were found. Check the snapshot files "
                    "and findings below.",
                    self.palette,
                )
            )
            return controls
        grouped: dict[str, list[HfTensorRow]] = {}
        for row in catalog:
            grouped.setdefault(row.group, []).append(row)
        shown = 0
        for group in sorted(grouped, key=lambda name: (name != "embeddings", name)):
            items = grouped[group]
            open_group = (
                group in self.expanded_groups
                or len(grouped) == 1
                or bool(self.query.strip())
            )
            block = items[0].block
            trailing: list[ft.Control] = []
            if block is not None:
                trailing.append(
                    ft.TextButton(
                        content="Open block",
                        data=f"hf-block:{block}",
                        on_click=lambda _e, number=block: self._on_select_block(number),
                    )
                )
            controls.append(
                _group_header(
                    group,
                    len(items),
                    open_group,
                    self.palette,
                    self._toggle_group(group),
                    extra=trailing,
                )
            )
            if not open_group:
                continue
            for row in items:
                shown += 1
                if shown <= self.row_limit:
                    controls.append(self._tensor_card(row))
        return _with_overflow(
            controls, shown, self.palette, self.row_limit, self._show_more
        )

    def _tensor_card(self, row: HfTensorRow) -> ft.Control:
        offset = f" @ {row.offset:,}" if row.offset is not None else ""
        size = compact_bytes(row.n_bytes) if row.n_bytes is not None else "unlocated"
        shard = f" · {row.shard}" if row.shard else ""
        return ft.Container(
            data=f"hf-tensor:{row.tensor_id}",
            content=ft.Column(
                controls=[
                    ft.Text(
                        row.name,
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=self.palette.ink,
                        selectable=True,
                    ),
                    ft.Text(
                        f"{row.element_type} · {row.shape} · {size}{offset}{shard}",
                        size=9,
                        color=self.palette.muted,
                    ),
                    ft.TextButton(
                        content="Inspect",
                        icon=ft.Icons.SEARCH_ROUNDED,
                        data=f"hf-inspect:{row.tensor_id}",
                        tooltip="Open this tensor's bytes in the inspector",
                        on_click=lambda _e, tensor_id=row.tensor_id: (
                            self._on_select_tensor(tensor_id)
                        ),
                    ),
                ],
                spacing=2,
                tight=True,
            ),
            padding=8,
            bgcolor=self.palette.panel,
            border=ft.Border.all(1, self.palette.border),
            border_radius=8,
            on_click=lambda _e, tensor_id=row.tensor_id: self._on_select_tensor(
                tensor_id
            ),
        )


def _color(palette: ShellPalette, role: str) -> str:
    return {
        "accent": palette.accent,
        "info": palette.info,
        "success": palette.success,
        "warning": palette.warning,
        "muted": palette.muted,
    }.get(role, palette.accent)


def _layout_controls(
    segments: Sequence[HfLayoutSegment],
    palette: ShellPalette,
    on_click: Callable[[str], ClickHandler],
) -> list[ft.Control]:
    controls: list[ft.Control] = []
    for segment in segments:
        flex = max(int(segment.fraction * 100), 3)
        label = segment.name if flex >= 12 else ""
        controls.append(
            ft.Container(
                data=f"hf-layout:{segment.key}",
                expand=flex,
                height=22,
                bgcolor=_color(palette, _LAYOUT_COLORS.get(segment.key, "accent")),
                border_radius=6,
                alignment=ft.Alignment.CENTER,
                tooltip=f"{segment.name}: {compact_bytes(segment.length)}",
                content=ft.Text(
                    label,
                    size=9,
                    weight=ft.FontWeight.W_700,
                    color="#FFFFFF",
                    no_wrap=True,
                ),
                on_click=on_click(segment.key),
            )
        )
    return controls


def _layout_legend(
    segments: Sequence[HfLayoutSegment], palette: ShellPalette
) -> ft.Control:
    return ft.Row(
        data="hf-layout-legend",
        wrap=True,
        spacing=8,
        controls=[
            ft.Text(
                f"{segment.name} {compact_bytes(segment.length)}",
                size=9,
                color=palette.muted,
                data=f"hf-legend:{segment.key}",
            )
            for segment in segments
        ],
    )


def _layout_rows(
    summary: HfContainerSummary,
    members: Sequence[HfMemberRow],
    palette: ShellPalette,
) -> list[ft.Control]:
    facts = [
        ("Architecture", summary.architecture),
        ("Blocks", f"{summary.block_count:,}"),
        ("Tensors", f"{summary.n_tensors:,}"),
        ("Shards", f"{summary.n_shards:,}"),
        ("Files", f"{summary.n_members:,}"),
        ("Snapshot size", compact_bytes(summary.byte_size)),
        ("Tied embeddings", "yes" if summary.tied_embeddings else "no"),
    ]
    rows: list[ft.Control] = [
        overview.metadata_section(
            tuple(facts),
            palette=palette,
            title="Snapshot",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            role="hf-header",
        )
    ]
    for member in members:
        rows.append(
            ft.Container(
                data=f"hf-member:{member.name}",
                content=ft.Column(
                    controls=[
                        ft.Text(
                            member.name,
                            size=11,
                            weight=ft.FontWeight.W_700,
                            color=palette.ink,
                            selectable=True,
                        ),
                        ft.Text(
                            member.detail,
                            size=10,
                            color=palette.muted,
                            selectable=True,
                        ),
                    ],
                    spacing=2,
                    tight=True,
                ),
                padding=8,
                bgcolor=palette.subtle,
                border_radius=8,
            )
        )
    return rows


def _pane_chips(
    active: str,
    palette: ShellPalette,
    on_pane: Callable[[str], ClickHandler],
) -> list[ft.Control]:
    chips: list[ft.Control] = []
    for key, label, icon in _PANES:
        selected = key == active
        chips.append(
            _filter_chip(
                label,
                selected,
                palette,
                on_pane(key),
                icon=icon,
                data=f"hf-pane:{key}",
            )
        )
    return chips


def _filter_chip(
    label: str,
    selected: bool,
    palette: ShellPalette,
    on_click: ClickHandler,
    *,
    icon: ft.IconData | None = None,
    data: str,
) -> ft.Control:
    return ft.Container(
        data=data,
        content=ft.Row(
            controls=[
                *([ft.Icon(icon, size=13, color=palette.accent)] if icon else []),
                ft.Text(
                    label,
                    size=10,
                    weight=ft.FontWeight.W_700,
                    color=palette.accent if selected else palette.ink,
                ),
            ],
            spacing=4,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.symmetric(horizontal=8, vertical=5),
        bgcolor=palette.accent_soft if selected else palette.subtle,
        border=ft.Border.all(1, palette.accent if selected else palette.border),
        border_radius=16,
        on_click=on_click,
    )


def _group_header(
    group: str,
    count: int,
    expanded: bool,
    palette: ShellPalette,
    on_toggle: ClickHandler,
    extra: Sequence[ft.Control] = (),
) -> ft.Control:
    chevron = (
        ft.Icons.EXPAND_MORE_ROUNDED if expanded else ft.Icons.CHEVRON_RIGHT_ROUNDED
    )
    return ft.Row(
        data=f"hf-group:{group}",
        controls=[
            ft.TextButton(
                content=f"{group} · {count}",
                icon=chevron,
                on_click=on_toggle,
                style=ft.ButtonStyle(color=palette.ink),
            ),
            *extra,
        ],
        spacing=4,
        wrap=True,
    )


def _hint(text: str, palette: ShellPalette) -> ft.Text:
    return ft.Text(text, size=10, color=palette.muted)


def _with_overflow(
    controls: list[ft.Control],
    shown: int,
    palette: ShellPalette,
    limit: int,
    on_more: ClickHandler,
) -> list[ft.Control]:
    if shown > limit:
        controls.append(
            _hint(
                f"Showing {limit:,} of {shown:,} matching items.",
                palette,
            )
        )
        controls.append(
            ft.TextButton(content="Show more", on_click=on_more, data="hf-show-more")
        )
    return controls
