"""Stateful hierarchy explorer for the model-navigation sidebar."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from typing import Any

import flet as ft

from nneditor.analysis.hierarchy import Group, Hierarchy
from nneditor.ui import shell_layout

__all__ = ["HierarchyPanel", "natural_key"]

TextButtonHandler = Callable[[ft.Event[ft.TextButton]], Any]
CheckboxHandler = Callable[[ft.Event[ft.Checkbox]], None]


def natural_key(text: str) -> tuple[tuple[int, int, str], ...]:
    """A deterministic case-insensitive key with numeric digit runs."""
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
        for part in re.split(r"(\d+)", text)
        if part
    )


class HierarchyPanel:
    """Hierarchy tools, expansion state, and bounded tree rendering."""

    def __init__(
        self,
        *,
        palette: shell_layout.ShellPalette,
        on_multi_select: CheckboxHandler,
        on_group: TextButtonHandler,
        on_merge: TextButtonHandler,
        on_rename: TextButtonHandler,
        on_split: TextButtonHandler,
        on_lock: TextButtonHandler,
        on_reject: TextButtonHandler,
        on_reset: TextButtonHandler,
        on_navigate: Callable[[str], None],
        refresh: Callable[[], None],
        update: Callable[[], None],
    ) -> None:
        self.palette = palette
        self._on_navigate = on_navigate
        self._request_refresh = refresh
        self._update = update
        self.expanded: dict[str, bool] = {}
        self.highlight: frozenset[str] = frozenset()
        self.list = ft.ListView(expand=True, spacing=4)
        self.multi_select = ft.Checkbox(label="Multi-select", on_change=on_multi_select)
        self.label = ft.TextField(label="Group label", dense=True)
        self.group_button = ft.TextButton(content="Group", on_click=on_group)
        self.merge_button = ft.TextButton(content="Merge", on_click=on_merge)
        self.rename_button = ft.TextButton(content="Rename", on_click=on_rename)
        self.split_button = ft.TextButton(content="Split", on_click=on_split)
        self.lock_button = ft.TextButton(content="Lock/unlock", on_click=on_lock)
        self.reject_button = ft.TextButton(content="Reject", on_click=on_reject)
        self.reset_button = ft.TextButton(content="Reset", on_click=on_reset)

    @property
    def tools(self) -> ft.ExpansionTile:
        """Build the palette-neutral hierarchy command group."""
        return shell_layout.build_hierarchy_tools(
            multi_select_field=self.multi_select,
            group_label_field=self.label,
            group_button=self.group_button,
            merge_button=self.merge_button,
            rename_button=self.rename_button,
            split_button=self.split_button,
            lock_button=self.lock_button,
            reject_button=self.reject_button,
            reset_groups_button=self.reset_button,
        )

    def set_palette(self, palette: shell_layout.ShellPalette) -> None:
        self.palette = palette

    def clear(self) -> None:
        self.expanded.clear()
        self.highlight = frozenset()
        self.list.controls = []

    def reset_expansion(self) -> None:
        self.expanded.clear()
        self.highlight = frozenset()

    @staticmethod
    def selection_ids(
        hierarchy: Hierarchy, selection: Collection[str]
    ) -> frozenset[str]:
        """The deepest selected group on each hierarchy branch."""
        highlighted: set[str] = set()
        for item in selection:
            if hierarchy.has_group(item):
                highlighted.add(item)
                continue
            containers = hierarchy.groups_for_node(item)
            if containers:
                highlighted.add(containers[0].id)
        ancestors: set[str] = set()
        for group_id in highlighted:
            ancestors.update(item.id for item in hierarchy.breadcrumbs(group_id)[:-1])
        return frozenset(highlighted - ancestors)

    def refresh(
        self,
        *,
        hierarchy: Hierarchy,
        node_order: dict[str, int],
        selection: Collection[str],
        current_root_group: str | None,
        max_rows: int,
        max_depth: int,
        indent: float,
    ) -> None:
        """Rebuild the bounded tree while retaining expansion state."""
        highlight = self.selection_ids(hierarchy, selection)
        if highlight != self.highlight:
            for group_id in highlight:
                for ancestor in hierarchy.breadcrumbs(group_id)[:-1]:
                    self.expanded[ancestor.id] = True
            self.highlight = highlight
        unknown = len(node_order)

        def sort_key(group: Group) -> tuple[int, tuple[tuple[int, int, str], ...]]:
            position = min(
                (node_order.get(member, unknown) for member in group.members),
                default=unknown,
            )
            return (position, natural_key(group.label))

        rows: list[ft.Control] = []
        visible = 0

        def add(group_id: str, depth: int) -> None:
            nonlocal visible
            group = hierarchy.group(group_id)
            children = (
                sorted(hierarchy.children(group.id), key=sort_key)
                if depth + 1 < max_depth
                else []
            )
            visible += 1
            if visible <= max_rows:
                rows.append(
                    self.build_row(
                        group,
                        depth,
                        bool(children),
                        current_root_group=current_root_group,
                        indent=indent,
                    )
                )
            if children and self.expanded.get(group.id, depth == 0):
                for child in children:
                    add(child.id, depth + 1)

        for root in sorted(hierarchy.roots, key=sort_key):
            add(root.id, 0)
        if not rows:
            rows.append(ft.Text("No groups detected", size=11))
        elif visible > max_rows:
            rows.append(
                ft.Text(
                    f"{visible - max_rows:,} more… Use search or the graph to "
                    "inspect the rest.",
                    size=10,
                    color=self.palette.muted,
                )
            )
        self.list.controls = rows

    def build_row(
        self,
        group: Group,
        depth: int,
        has_children: bool,
        *,
        current_root_group: str | None,
        indent: float,
    ) -> ft.Control:
        selected = group.id in self.highlight
        active = group.id == current_root_group
        label = ft.TextButton(
            content=f"{group.label}  ·  {len(group.members)} ops",
            icon=ft.Icons.GRID_VIEW_ROUNDED,
            tooltip=group.explanation,
            data=f"hierarchy-row:{group.id}",
            style=ft.ButtonStyle(
                color=(self.palette.accent if selected or active else self.palette.ink),
                bgcolor=(
                    self.palette.accent_soft if active and not selected else "#00FFFFFF"
                ),
                shape=ft.RoundedRectangleBorder(radius=9),
                alignment=ft.Alignment.CENTER_LEFT,
            ),
            on_click=self.navigate_handler(group.id),
        )
        row: ft.Control = label
        if has_children or depth:
            controls: list[ft.Control] = []
            if depth:
                controls.append(ft.Container(width=depth * indent))
            if has_children:
                expanded = self.expanded.get(group.id, depth == 0)
                controls.append(
                    ft.IconButton(
                        icon=(
                            ft.Icons.EXPAND_MORE_ROUNDED
                            if expanded
                            else ft.Icons.CHEVRON_RIGHT_ROUNDED
                        ),
                        icon_size=16,
                        icon_color=self.palette.muted,
                        tooltip="Collapse" if expanded else "Expand",
                        data=f"hierarchy-toggle:{group.id}",
                        on_click=self.toggle_handler(group.id, depth),
                    )
                )
            controls.append(label)
            row = ft.Row(
                controls=controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        if not selected:
            return row
        return ft.Container(
            content=row,
            bgcolor=self.palette.accent_soft,
            border_radius=9,
            data=f"hierarchy-selected:{group.id}",
        )

    def toggle_handler(
        self, group_id: str, depth: int
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        def toggle(event: ft.Event[ft.IconButton]) -> None:
            expanded = self.expanded.get(group_id, depth == 0)
            self.expanded[group_id] = not expanded
            self._request_refresh()
            self._update()

        return toggle

    def navigate_handler(
        self, group_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def navigate(event: ft.Event[ft.TextButton]) -> None:
            self._on_navigate(group_id)

        return navigate
