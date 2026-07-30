"""Weight-tensor and captured-activation inspector cards.

The panel owns everything the inspector shows *below* a selection's metadata:
weight tensors with their bounded value map and hex editor, and the activations
a trace captured for the selected node, block, connection, or boundary.

It holds no reference to the shell.  Everything it needs about the current
session, graph, and trace arrives as an injected accessor, and everything it
wants the shell to do afterwards arrives as an injected callable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import flet as ft
import flet.canvas as cv

from nneditor.analysis.statistics import TensorStatistics, element_width
from nneditor.application.session import ModelSession
from nneditor.application.slices import GraphSlice
from nneditor.ir.core import Storage
from nneditor.storage.cache import BudgetedCache
from nneditor.tracing.contracts import ActivationRecord
from nneditor.tracing.visualization import ActivationVisualization
from nneditor.ui import overview, tensor_tools, viewmodel
from nneditor.ui.activation_layers import build_activation_layer_viewer
from nneditor.ui.shell_layout import ShellPalette

__all__ = ["ActivationInspector", "build_activation_plot"]

# One view tuple can hold sixteen 512x512 RGB layer PNGs plus a raster PNG
# (the tracing/visualization.py caps), so clicking through a few hundred
# traced nodes would otherwise accumulate unbounded memory. 128 MiB retains
# dozens of recent views; an evicted one costs only a click to rebuild.
_VISUALIZATION_CACHE_BUDGET = 128 * 1024 * 1024

# The session job pool runs 4 workers. A selected semantic block can span
# hundreds of boundary values; auto-building them all would queue hundreds of
# jobs behind those workers and starve every manual request, so automatic
# view-building stops here and later records keep their build button.
_AUTOLOAD_VIEW_LIMIT = 6


def _visualization_cost(views: tuple[ActivationVisualization, ...]) -> int:
    """Bytes one cached view tuple actually retains, never below one unit."""
    total = sum(
        len(view.raster_png)
        + sum(len(png) for png in view.layer_pngs)
        + 8 * len(view.values)
        for view in views
    )
    return max(1, total)


class ActivationInspector:
    """Own the tensor and activation cards, their caches, and their jobs."""

    def __init__(
        self,
        *,
        page: ft.Page,
        palette: ShellPalette,
        session: Callable[[], ModelSession | None],
        current_graph: Callable[[], str | None],
        current_slice: Callable[[], GraphSlice | None],
        active_trace_id: Callable[[], str | None],
        trace_values_for_glyph: Callable[[str], tuple[str, ...]],
        inspected_ids: Callable[[], frozenset[str]],
        selection: Callable[[], frozenset[str]],
        refresh_inspector: Callable[[frozenset[str]], None],
        refresh_edit_actions: Callable[[], None],
        show_committed_diff: Callable[[], None],
        on_error: Callable[[str], None],
        on_status: Callable[[str], None],
        watch_text_focus: Callable[[ft.TextField], ft.TextField],
        visualization_cache_budget: int = _VISUALIZATION_CACHE_BUDGET,
    ) -> None:
        self.page = page
        self.palette = palette
        self._session = session
        self._current_graph = current_graph
        self._current_slice = current_slice
        self._active_trace_id = active_trace_id
        self._trace_values_for_glyph = trace_values_for_glyph
        self._inspected_ids = inspected_ids
        self._selection = selection
        self._refresh_inspector = refresh_inspector
        self._refresh_edit_actions = refresh_edit_actions
        self._show_committed_diff = show_committed_diff
        self._on_error = on_error
        self._on_status = on_status
        self._watch_text_focus = watch_text_focus

        self.typed_preview_consent: set[str] = set()
        self.hex_offsets: dict[str, int] = {}
        self.hex_drafts: dict[tuple[str, int], str] = {}
        self.hex_errors: dict[str, str] = {}
        # Expansion state per tensor card, so a hex-pager or statistics
        # interaction does not re-collapse every card the user opened.
        self.tensor_card_expanded: dict[str, bool] = {}
        self.activation_statistics: dict[tuple[str, str], TensorStatistics] = {}
        self._visualization_cache_budget = visualization_cache_budget
        self.activation_visualizations = self._new_visualization_cache()
        self.activation_visualization_errors: dict[tuple[str, str], str] = {}
        self.activation_loading: set[tuple[str, str, str]] = set()

    def _new_visualization_cache(
        self,
    ) -> BudgetedCache[tuple[str, str], tuple[ActivationVisualization, ...]]:
        return BudgetedCache(
            "activation-visualizations",
            self._visualization_cache_budget,
            cost=_visualization_cost,
        )

    def reset(self) -> None:
        """Drop every per-session cache, consent, and draft."""
        self.typed_preview_consent.clear()
        self.hex_offsets.clear()
        self.hex_drafts.clear()
        self.hex_errors.clear()
        self.tensor_card_expanded.clear()
        self.activation_statistics.clear()
        # A fresh cache also resets its hit/miss/eviction counters, so the
        # numbers a diagnostics reader sees are always for one session.
        self.activation_visualizations = self._new_visualization_cache()
        self.activation_visualization_errors.clear()
        self.activation_loading.clear()

    # -- inspector rows ----------------------------------------------------

    def tensor_rows(self, node_id: str) -> list[ft.Control]:
        """Inspector rows for every weight tensor feeding the selected node."""
        session = self._session()
        graph_id = self._current_graph()
        assert session is not None and graph_id is not None
        try:
            tensor_ids = viewmodel.node_tensor_ids(session.document, graph_id, node_id)
        except KeyError:
            return []
        if not tensor_ids:
            return []
        rows: list[ft.Control] = [
            ft.Divider(height=8, color=self.palette.border),
            overview.section_heading(
                "Weights & tensors",
                ft.Icons.MEMORY_ROUNDED,
                trailing=str(len(tensor_ids)),
            ),
        ]
        rows.extend(
            self._tensor_card(
                tensor_id,
                node_id,
                # The first card starts open; after that the user's own
                # expand/collapse choices survive every inspector rebuild.
                expanded=self.tensor_card_expanded.get(tensor_id, position == 0),
            )
            for position, tensor_id in enumerate(tensor_ids)
        )
        return rows

    def activation_rows(self, node_id: str) -> list[ft.Control]:
        """Lazy captured input/output cards beside the node's weights."""
        session = self._session()
        trace_id = self._active_trace_id()
        if session is None or trace_id is None:
            return []
        try:
            records = session.node_activations(trace_id, node_id)
        except (KeyError, ValueError):
            records = ()
        return self._activation_record_rows(
            records,
            owner_id=node_id,
            title="Captured activations",
        )

    def group_activation_rows(
        self,
        members: frozenset[str],
        *,
        owner_id: str,
    ) -> list[ft.Control]:
        """Captured values crossing a selected block/region boundary."""
        session = self._session()
        if (
            session is None
            or self._active_trace_id() is None
            or self._current_graph() != session.document.entry_graph
        ):
            return []
        value_ids = self._value_ids_for_members(members)
        return self.activation_rows_for_values(
            value_ids,
            owner_id=owner_id,
            title="Block inputs & outputs",
        )

    def activation_rows_for_values(
        self,
        value_ids: tuple[str, ...],
        *,
        owner_id: str,
        title: str,
    ) -> list[ft.Control]:
        """Activation cards for selected connection or boundary values."""
        session = self._session()
        if session is None:
            return []
        trace_id = self._active_trace_id()
        if trace_id is None:
            return [
                ft.Divider(height=8, color=self.palette.border),
                overview.section_heading(
                    title,
                    ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
                    trailing="not run",
                ),
                ft.Container(
                    content=ft.Text(
                        "Run a trace once, then click any node, connection, input, "
                        "or output to inspect its activation here.",
                        size=10,
                        color=self.palette.info,
                    ),
                    padding=10,
                    bgcolor=self.palette.info_soft,
                    border_radius=9,
                ),
            ]
        result = session.trace(trace_id)
        records_by_value = {record.value_id: record for record in result.records}
        records = tuple(
            record
            for value_id in dict.fromkeys(value_ids)
            if (record := records_by_value.get(value_id)) is not None
        )
        return self._activation_record_rows(
            records,
            owner_id=owner_id,
            title=title,
            expand_all=True,
        )

    def autoload_views(self, ids: frozenset[str]) -> None:
        """Start bounded activation views as soon as a traced glyph is selected.

        At most ``_AUTOLOAD_VIEW_LIMIT`` jobs start per selection change, in
        the records' deterministic order; every remaining record keeps its
        manual "Build activation views now" button.
        """
        trace_id = self._active_trace_id()
        if trace_id is None or len(ids) != 1:
            return
        (owner_id,) = ids
        started = 0
        for record in self._records_for_selection(ids):
            if started >= _AUTOLOAD_VIEW_LIMIT:
                break
            if record.readable and self._request_views(
                trace_id, record.value_id, owner_id
            ):
                started += 1

    # -- trace record selection -------------------------------------------

    def _records_for_selection(
        self,
        ids: frozenset[str],
    ) -> tuple[ActivationRecord, ...]:
        """Readable trace records represented by one selected graph glyph."""
        session = self._session()
        trace_id = self._active_trace_id()
        if (
            session is None
            or trace_id is None
            or len(ids) != 1
            or self._current_graph() != session.document.entry_graph
        ):
            return ()
        (glyph_id,) = ids
        value_ids = self._trace_values_for_glyph(glyph_id)
        if value_ids:
            result = session.trace(trace_id)
            by_value = {record.value_id: record for record in result.records}
            return tuple(
                record
                for value_id in value_ids
                if (record := by_value.get(value_id)) is not None
            )
        try:
            return session.node_activations(trace_id, glyph_id)
        except (KeyError, ValueError):
            pass
        current_slice = self._current_slice()
        if current_slice is None:
            return ()
        members = current_slice.members_by_glyph.get(glyph_id, frozenset())
        value_ids = self._value_ids_for_members(members)
        if not value_ids:
            return ()
        result = session.trace(trace_id)
        by_value = {record.value_id: record for record in result.records}
        return tuple(
            record
            for value_id in value_ids
            if (record := by_value.get(value_id)) is not None
        )

    def _value_ids_for_members(self, members: frozenset[str]) -> tuple[str, ...]:
        """Values crossing the boundary of an aggregated graph glyph."""
        session = self._session()
        if (
            session is None
            or not members
            or self._current_graph() != session.document.entry_graph
        ):
            return ()
        graph = session.document.main_graph
        value_ids: set[str] = set()
        for node_id in members:
            node = graph.node(node_id)
            for value_id in node.inputs:
                producer = graph.producer(value_id)
                if value_id not in graph.initializers and (
                    producer is None or producer[0] not in members
                ):
                    value_ids.add(value_id)
            for value_id in node.outputs:
                if value_id in graph.outputs or any(
                    consumer not in members
                    for consumer, _port in graph.consumers(value_id)
                ):
                    value_ids.add(value_id)
        return tuple(sorted(value_ids))

    def _activation_record_rows(
        self,
        records: tuple[ActivationRecord, ...],
        *,
        owner_id: str,
        title: str,
        expand_all: bool = False,
    ) -> list[ft.Control]:
        trace_id = self._active_trace_id()
        assert trace_id is not None
        rows: list[ft.Control] = [
            ft.Divider(height=8, color=self.palette.border),
            overview.section_heading(
                title,
                ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
                trailing=str(len(records)),
            ),
        ]
        if not records:
            rows.append(
                ft.Text(
                    "No activation was captured for this node. The trace may "
                    "have used a narrowed selection or stopped earlier.",
                    size=10,
                    color=self.palette.muted,
                )
            )
            return rows
        rows.extend(
            self._activation_card(
                trace_id,
                record,
                owner_id,
                expanded=expand_all or record.role == "node-output",
            )
            for record in records
        )
        return rows

    # -- shared card scaffolding ------------------------------------------

    def _card(
        self,
        *,
        data: str,
        title: ft.Text,
        subtitle: str,
        metadata: tuple[tuple[str, str], ...],
        metadata_title: str,
        metadata_icon: ft.IconData,
        metadata_role: str,
        body: Sequence[ft.Control],
        expanded: bool,
        spacing: int,
        controls_padding: ft.PaddingValue,
        leading: ft.Control | None = None,
        tile_padding: ft.PaddingValue | None = None,
        on_change: Callable[[ft.Event[ft.ExpansionTile]], None] | None = None,
        shape: ft.OutlinedBorder | None = None,
        collapsed_shape: ft.OutlinedBorder | None = None,
    ) -> ft.ExpansionTile:
        """Assemble one inspector card around a kind-specific body.

        Both the weight-tensor and the activation card are a metadata header
        followed by their own content inside a collapsible tile; only the body
        and the tile's chrome differ, so both build the card from here.
        """
        return ft.ExpansionTile(
            data=data,
            title=title,
            subtitle=ft.Text(subtitle, size=10, color=self.palette.muted),
            leading=leading,
            controls=[
                ft.Column(
                    controls=[
                        overview.metadata_section(
                            metadata,
                            title=metadata_title,
                            icon=metadata_icon,
                            role=metadata_role,
                        ),
                        *body,
                    ],
                    spacing=spacing,
                    tight=True,
                )
            ],
            controls_padding=controls_padding,
            tile_padding=tile_padding,
            expanded=expanded,
            on_change=on_change,
            maintain_state=True,
            dense=True,
            shape=shape,
            collapsed_shape=collapsed_shape,
            bgcolor=self.palette.subtle,
            collapsed_bgcolor=self.palette.subtle,
        )

    def _statistics_controls(
        self,
        stats: TensorStatistics,
        *,
        role: str,
    ) -> list[ft.Control]:
        """The statistics table plus its text histogram."""
        controls: list[ft.Control] = [
            overview.metadata_section(
                viewmodel.statistics_lines(stats),
                title="Statistics",
                icon=ft.Icons.QUERY_STATS_ROUNDED,
                role=role,
            )
        ]
        for label, count, fraction in viewmodel.histogram_rows(stats):
            bar = "█" * max(1, round(fraction * 20)) if count else ""
            controls.append(
                ft.Text(
                    f"{label:>24} {bar} {count}",
                    size=10,
                    font_family="monospace",
                    color=self.palette.ink,
                )
            )
        return controls

    @staticmethod
    def _compute_statistics_button(
        on_click: Callable[[ft.Event[ft.TextButton]], None],
        *,
        data: str | None = None,
    ) -> ft.TextButton:
        return ft.TextButton(
            content="Compute statistics",
            icon=ft.Icons.QUERY_STATS_ROUNDED,
            data=data,
            on_click=on_click,
        )

    def _warning_panel(
        self,
        content: ft.Control,
        *,
        padding: int,
        border_radius: int,
        border: ft.Border | None = None,
    ) -> ft.Container:
        return ft.Container(
            content=content,
            padding=padding,
            bgcolor=self.palette.warning_soft,
            border=border,
            border_radius=border_radius,
        )

    # -- activation cards --------------------------------------------------

    def _activation_card(
        self,
        trace_id: str,
        record: ActivationRecord,
        node_id: str,
        *,
        expanded: bool,
    ) -> ft.Control:
        session = self._session()
        assert session is not None
        lines = [
            ("Value", record.value_name),
            ("Role", record.role),
            ("State", record.state.value),
            ("Dtype", record.element_type),
            ("Shape", str(list(record.shape))),
            (
                "Captured bytes",
                f"{record.stored_byte_length:,} / {record.full_byte_length:,}",
            ),
        ]
        if record.reason:
            lines.append(("Disclosure", record.reason))
        controls: list[ft.Control] = []
        key = (trace_id, record.value_id)
        if record.readable:
            try:
                view = session.activations(trace_id)
                raw = view.read(
                    record.value_id,
                    length=min(record.stored_byte_length, 64),
                )
                preview = viewmodel.preview_values_text(record.element_type, raw)
                controls.append(
                    ft.Text(
                        f"Preview: {preview}",
                        size=10,
                        font_family="monospace",
                        color=self.palette.ink,
                    )
                )
            except Exception as error:
                controls.append(
                    ft.Text(
                        f"Preview error: {error}",
                        size=10,
                        color=self.palette.danger,
                    )
                )
            stats = self.activation_statistics.get(key)
            if stats is not None:
                controls.extend(
                    self._statistics_controls(
                        stats,
                        role=f"activation-statistics:{record.value_id}",
                    )
                )
            elif (trace_id, record.value_id, "statistics") in self.activation_loading:
                controls.append(ft.ProgressRing(width=18, height=18))
                controls.append(ft.Text("Loading activation statistics…", size=10))
            else:
                controls.append(
                    self._compute_statistics_button(
                        self._activation_stats_handler(
                            trace_id, record.value_id, node_id
                        )
                    )
                )
            controls.extend(self._activation_view_controls(key, record, node_id))
        else:
            controls.append(
                self._warning_panel(
                    ft.Text(
                        record.reason
                        or f"This capture is {record.state.value} and cannot be read.",
                        size=10,
                        color=self.palette.warning,
                    ),
                    padding=8,
                    border_radius=8,
                )
            )
        return self._card(
            data=f"activation-card:{record.value_id}",
            title=ft.Text(record.value_name, size=12, weight=ft.FontWeight.W_700),
            subtitle=f"{record.role} • {record.state.value}",
            metadata=tuple(lines),
            metadata_title="Activation metadata",
            metadata_icon=ft.Icons.DATA_ARRAY_ROUNDED,
            metadata_role=f"activation-metadata:{record.value_id}",
            body=controls,
            expanded=expanded,
            spacing=8,
            controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
        )

    def _activation_view_controls(
        self,
        key: tuple[str, str],
        record: ActivationRecord,
        node_id: str,
    ) -> list[ft.Control]:
        """The bounded plots for one readable capture, or the way to build them."""
        trace_id, value_id = key
        visualizations = self.activation_visualizations.get(key)
        if visualizations is None:
            if (trace_id, value_id, "visualization") in self.activation_loading:
                return [ft.Text("Loading activation views…", size=10)]
            if view_error := self.activation_visualization_errors.get(key):
                return [
                    ft.Text(
                        f"Activation views could not be built: {view_error}",
                        size=10,
                        color=self.palette.danger,
                    ),
                    ft.TextButton(
                        content="Retry activation views",
                        icon=ft.Icons.REFRESH_ROUNDED,
                        on_click=self._views_handler(trace_id, value_id, node_id),
                    ),
                ]
            return [
                ft.TextButton(
                    content="Build activation views now",
                    icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                    on_click=self._views_handler(trace_id, value_id, node_id),
                )
            ]
        if not visualizations:
            return [
                ft.Text(
                    "This activation has no finite values to visualize.",
                    size=10,
                    color=self.palette.muted,
                )
            ]
        controls: list[ft.Control] = [
            ft.FilledButton(
                content="Open large view",
                icon=ft.Icons.OPEN_IN_FULL_ROUNDED,
                on_click=self._open_views_handler(trace_id, record),
            )
        ]
        for plot in visualizations:
            controls.append(build_activation_plot(plot, palette=self.palette))
            controls.append(
                overview.metadata_section(
                    (
                        ("Kind", plot.kind.value),
                        (
                            "Bins"
                            if plot.kind.value == "histogram"
                            else "Displayed shape",
                            (
                                str(len(plot.values))
                                if plot.kind.value == "histogram"
                                else str(list(plot.shape))
                            ),
                        ),
                        ("Colormap", plot.colormap),
                        ("Normalization", plot.normalization),
                        ("Downsampling", plot.downsampling),
                        ("Capture", "partial" if plot.partial else "complete"),
                    ),
                    title=plot.title,
                    icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                    role=f"activation-view:{value_id}:{plot.kind.value}",
                )
            )
        return controls

    def _activation_stats_handler(
        self,
        trace_id: str,
        value_id: str,
        node_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def start(event: ft.Event[ft.TextButton]) -> None:
            session = self._session()
            if session is None:
                return
            loading = (trace_id, value_id, "statistics")
            self.activation_loading.add(loading)
            job = session.compute_activation_statistics_async(trace_id, value_id)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

            def wait() -> None:
                job.wait()
                self.activation_loading.discard(loading)
                if job.state.value == "succeeded":
                    self.activation_statistics[(trace_id, value_id)] = job.result()
                elif job.state.value == "failed":
                    self._on_error(f"Activation statistics failed: {job.error}")
                if self._inspected_ids() == frozenset({node_id}):
                    self._refresh_inspector(self._inspected_ids())
                self.page.update()

            self.page.run_thread(wait)

        return start

    def _request_views(
        self,
        trace_id: str,
        value_id: str,
        owner_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """Queue one bounded visualization job unless it is already resolved."""
        session = self._session()
        if session is None:
            return False
        key = (trace_id, value_id)
        loading = (trace_id, value_id, "visualization")
        # peek keeps the presence probe out of the hit/miss counters and makes
        # a repeated request while a job is in flight a no-op.
        if (
            self.activation_visualizations.peek(key) is not None
            or loading in self.activation_loading
        ):
            return False
        if key in self.activation_visualization_errors and not force:
            return False
        self.activation_visualization_errors.pop(key, None)
        self.activation_loading.add(loading)
        try:
            job = session.activation_visualizations_async(
                trace_id,
                value_id,
                attention=self._node_is_attention(owner_id),
            )
        except Exception as error:
            self.activation_loading.discard(loading)
            self.activation_visualization_errors[key] = str(error)
            return False

        def wait() -> None:
            job.wait()
            self.activation_loading.discard(loading)
            active = self._session() is session and self._active_trace_id() == trace_id
            if active and job.state.value == "succeeded":
                self.activation_visualizations.put(key, job.result())
                self.activation_visualization_errors.pop(key, None)
            elif active and job.state.value == "failed":
                self.activation_visualization_errors[key] = str(job.error)
                self._on_status(f"Activation view failed: {job.error}")
            if active and self._inspected_ids() == frozenset({owner_id}):
                self._refresh_inspector(self._inspected_ids())
            if active:
                self.page.update()

        self.page.run_thread(wait)
        return True

    def _views_handler(
        self,
        trace_id: str,
        value_id: str,
        node_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def start(event: ft.Event[ft.TextButton]) -> None:
            if self._request_views(trace_id, value_id, node_id, force=True):
                self._refresh_inspector(frozenset({node_id}))
                self.page.update()

        return start

    def _open_views_handler(
        self,
        trace_id: str,
        record: ActivationRecord,
    ) -> Callable[[ft.Event[ft.Button]], None]:
        def open_overlay(event: ft.Event[ft.Button]) -> None:
            views = self.activation_visualizations.get((trace_id, record.value_id))
            if not views:
                self._on_status("Activation views are not ready yet")
                self.page.update()
                return
            rows: list[ft.Control] = [
                overview.metadata_section(
                    (
                        ("Value", record.value_name),
                        ("Role", record.role),
                        ("Dtype", record.element_type),
                        ("Shape", str(list(record.shape))),
                        (
                            "Captured bytes",
                            f"{record.stored_byte_length:,} / "
                            f"{record.full_byte_length:,}",
                        ),
                    ),
                    title="Captured activation",
                    icon=ft.Icons.DATA_ARRAY_ROUNDED,
                    role=f"activation-overlay-metadata:{record.value_id}",
                )
            ]
            for plot in views:
                rows.extend(
                    [
                        overview.section_heading(
                            plot.title,
                            ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                            trailing=plot.kind.value,
                        ),
                        build_activation_plot(
                            plot,
                            palette=self.palette,
                            width=780.0,
                            height=420.0,
                        ),
                        overview.metadata_section(
                            (
                                ("Source shape", str(list(plot.source_shape))),
                                (
                                    "Bins"
                                    if plot.kind.value == "histogram"
                                    else "Displayed shape",
                                    (
                                        str(len(plot.values))
                                        if plot.kind.value == "histogram"
                                        else str(list(plot.shape))
                                    ),
                                ),
                                ("Colormap", plot.colormap),
                                ("Normalization", plot.normalization),
                                ("Downsampling", plot.downsampling),
                                (
                                    "Capture",
                                    "partial" if plot.partial else "complete",
                                ),
                            ),
                            title="View details",
                            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
                            role=(
                                f"activation-overlay-view:{record.value_id}:"
                                f"{plot.kind.value}"
                            ),
                        ),
                    ]
                )
            dialog = ft.AlertDialog(
                data=f"activation-overlay:{record.value_id}",
                modal=True,
                title=ft.Text(
                    f"Activation · {record.value_name}",
                    weight=ft.FontWeight.W_700,
                ),
                content=ft.Container(
                    width=860,
                    height=620,
                    content=ft.ListView(
                        controls=rows,
                        spacing=14,
                        expand=True,
                    ),
                ),
                actions=[
                    ft.TextButton(
                        content="Close",
                        icon=ft.Icons.CLOSE_ROUNDED,
                        on_click=self._close_overlay,
                    )
                ],
                scrollable=False,
            )
            self.page.show_dialog(dialog)

        return open_overlay

    def _close_overlay(self, event: ft.Event[ft.TextButton]) -> None:
        self.page.pop_dialog()

    def _node_is_attention(self, node_id: str) -> bool:
        session = self._session()
        graph_id = self._current_graph()
        if session is None or graph_id is None:
            return False
        hierarchy = session.graph_hierarchy(graph_id)
        return any(
            node_id in group.members
            and "attention"
            in " ".join(
                (
                    group.label,
                    group.explanation,
                    *(evidence.code for evidence in group.evidence),
                )
            ).lower()
            for group in hierarchy.groups
        )

    # -- weight-tensor cards ----------------------------------------------

    def _tensor_card(
        self, tensor_id: str, node_id: str, *, expanded: bool
    ) -> ft.Control:
        session = self._session()
        assert session is not None
        tensor = session.document.tensors[tensor_id]
        materialization = (
            "generated"
            if tensor.storage is Storage.GENERATED
            else session.store.materialization(tensor_id).value
        )
        byte_length = session.editing.byte_length(tensor_id)
        requires_consent = (
            tensor.storage is not Storage.GENERATED
            and materialization == "full-parse"
            and tensor_id not in self.typed_preview_consent
        )
        tensor_controls: list[ft.Control] = []
        if requires_consent:
            tensor_controls.append(
                self._warning_panel(
                    ft.Column(
                        controls=[
                            ft.Text(
                                "Inspection requires materialization",
                                size=11,
                                weight=ft.FontWeight.W_700,
                                color=self.palette.warning,
                            ),
                            ft.Text(
                                "This tensor is stored in typed protobuf fields. "
                                "Loading values will parse the full tensor once.",
                                size=10,
                                color=self.palette.muted,
                            ),
                            self.preview_row(tensor_id),
                        ],
                        spacing=6,
                        tight=True,
                    ),
                    padding=10,
                    border_radius=10,
                    border=ft.Border.all(1, self.palette.warning_border),
                )
            )
        else:
            tensor_controls.extend(
                [
                    self._tensor_visualization(tensor_id),
                    self._hex_editor(tensor_id, node_id),
                ]
            )
        tensor_controls.extend(self._tensor_statistics_controls(tensor_id, node_id))

        short_name = tensor_id.rsplit("#", 1)[-1]
        shape = " x ".join(str(dimension) for dimension in tensor.dims) or "scalar"
        return self._card(
            data=f"tensor-card:{tensor_id}",
            title=ft.Text(
                short_name,
                size=12,
                weight=ft.FontWeight.W_700,
                color=self.palette.ink,
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
                tooltip=tensor_id,
            ),
            subtitle=f"{tensor.element_type} · {shape}",
            metadata=viewmodel.tensor_lines(
                session.document,
                tensor_id,
                materialization=materialization,
                byte_length=byte_length,
            ),
            metadata_title="Tensor metadata",
            metadata_icon=ft.Icons.DATA_OBJECT_ROUNDED,
            metadata_role=f"tensor-metadata:{tensor_id}",
            body=tensor_controls,
            expanded=expanded,
            spacing=12,
            controls_padding=ft.Padding.only(left=8, right=8, top=4, bottom=12),
            leading=ft.Icon(ft.Icons.DATA_ARRAY_ROUNDED, color=self.palette.accent),
            tile_padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            on_change=self._tensor_expansion_handler(tensor_id),
            shape=ft.RoundedRectangleBorder(radius=12),
            collapsed_shape=ft.RoundedRectangleBorder(radius=12),
        )

    def _tensor_expansion_handler(
        self, tensor_id: str
    ) -> Callable[[ft.Event[ft.ExpansionTile]], None]:
        def remember(event: ft.Event[ft.ExpansionTile]) -> None:
            data = getattr(event, "data", None)
            self.tensor_card_expanded[tensor_id] = (
                data if isinstance(data, bool) else str(data).lower() == "true"
            )

        return remember

    def _tensor_statistics_controls(
        self, tensor_id: str, node_id: str
    ) -> list[ft.Control]:
        session = self._session()
        assert session is not None
        edited = tensor_id in session.editing.edited_tensor_ids()
        stats = None if edited else session.statistics(tensor_id)
        if stats is None:
            if edited:
                return [
                    ft.Text(
                        "Statistics were cleared because this tensor has byte "
                        "revisions.",
                        size=10,
                        color=self.palette.muted,
                        italic=True,
                    )
                ]
            if session.store.readable(tensor_id):
                return [
                    self._compute_statistics_button(
                        self.statistics_handler(tensor_id, node_id),
                        data=f"tensor-compute-statistics:{tensor_id}",
                    )
                ]
            return []
        return [
            ft.Column(
                controls=self._statistics_controls(
                    stats,
                    role=f"tensor-statistics:{tensor_id}",
                ),
                spacing=5,
                tight=True,
                data=f"tensor-statistics-block:{tensor_id}",
            )
        ]

    def _tensor_visualization(self, tensor_id: str) -> ft.Control:
        """A bounded value heatmap; at most 64 elements are ever read."""
        session = self._session()
        assert session is not None
        tensor = session.document.tensors[tensor_id]
        width = element_width(tensor.element_type)
        if width is None:
            return ft.Container(
                data=f"tensor-heatmap:{tensor_id}",
                content=ft.Text(
                    f"{tensor.element_type} values cannot be decoded; "
                    "the raw bytes remain available in the hex editor.",
                    size=10,
                    color=self.palette.muted,
                ),
                padding=10,
                bgcolor=self.palette.subtle,
                border=ft.Border.all(1, self.palette.border),
                border_radius=10,
            )
        total = session.editing.byte_length(tensor_id) or 0
        try:
            raw = session.editing.read(
                tensor_id,
                offset=0,
                length=min(total, tensor_tools.HEATMAP_ELEMENTS * width),
            )
        except Exception as error:
            return ft.Text(
                f"Value visualization unavailable: {error}",
                size=10,
                color=self.palette.danger,
            )
        values = tensor_tools.sample_values(tensor.element_type, raw)
        if not values:
            return ft.Text(
                "This tensor has no values.", size=10, color=self.palette.muted
            )

        maximum_absolute = max(
            (abs(value) for value in values if math.isfinite(value)),
            default=0.0,
        )
        cell = 13.0
        gap = 2.0
        shapes: list[cv.Shape] = []
        for index, value in enumerate(values):
            column = index % tensor_tools.HEATMAP_COLUMNS
            row = index // tensor_tools.HEATMAP_COLUMNS
            shapes.append(
                cv.Rect(
                    x=column * (cell + gap),
                    y=row * (cell + gap),
                    width=cell,
                    height=cell,
                    border_radius=3,
                    paint=ft.Paint(
                        color=tensor_tools.heat_color(value, maximum_absolute),
                        style=ft.PaintingStyle.FILL,
                    ),
                )
            )
        row_count = math.ceil(len(values) / tensor_tools.HEATMAP_COLUMNS)
        canvas_width = tensor_tools.HEATMAP_COLUMNS * (cell + gap) - gap
        canvas_height = row_count * (cell + gap) - gap
        finite = tuple(value for value in values if math.isfinite(value))
        value_range = (
            f"{min(finite):.5g} to {max(finite):.5g}"
            if finite
            else "non-finite values only"
        )
        return ft.Column(
            data=f"tensor-heatmap:{tensor_id}",
            controls=[
                overview.section_heading(
                    "Value map",
                    ft.Icons.GRID_ON_ROUNDED,
                    trailing=f"FIRST {len(values)}",
                ),
                ft.Container(
                    content=cv.Canvas(
                        shapes=shapes,
                        width=canvas_width,
                        height=canvas_height,
                    ),
                    alignment=ft.Alignment.CENTER,
                    padding=10,
                    bgcolor=self.palette.panel,
                    border=ft.Border.all(1, self.palette.border),
                    border_radius=10,
                ),
                ft.Row(
                    controls=[
                        ft.Container(width=10, height=10, bgcolor="#2E90FA"),
                        ft.Text("negative", size=9, color=self.palette.muted),
                        ft.Container(width=10, height=10, bgcolor="#F2F4F7"),
                        ft.Text("zero", size=9, color=self.palette.muted),
                        ft.Container(width=10, height=10, bgcolor="#7F56D9"),
                        ft.Text("positive", size=9, color=self.palette.muted),
                    ],
                    spacing=4,
                    wrap=True,
                ),
                ft.Text(
                    f"Sample range: {value_range} · "
                    "Preview: "
                    f"{viewmodel.preview_values_text(tensor.element_type, raw)}",
                    size=10,
                    color=self.palette.muted,
                    selectable=True,
                ),
            ],
            spacing=7,
            tight=True,
        )

    # -- hex editor --------------------------------------------------------

    def _hex_editor(self, tensor_id: str, node_id: str) -> ft.Control:
        """Build a 128-byte page editor backed by reversible tensor revisions."""
        session = self._session()
        assert session is not None
        total = session.editing.byte_length(tensor_id) or 0
        last_offset = (
            max(total - 1, 0) // tensor_tools.HEX_PAGE_BYTES
        ) * tensor_tools.HEX_PAGE_BYTES
        offset = min(max(self.hex_offsets.get(tensor_id, 0), 0), last_offset)
        self.hex_offsets[tensor_id] = offset
        length = min(tensor_tools.HEX_PAGE_BYTES, max(total - offset, 0))
        try:
            raw = session.editing.read(tensor_id, offset=offset, length=length)
        except Exception as error:
            return ft.Container(
                data=f"tensor-hex-editor:{tensor_id}",
                content=ft.Text(
                    f"Raw bytes unavailable: {error}",
                    size=10,
                    color=self.palette.danger,
                ),
                padding=10,
                bgcolor=self.palette.danger_soft,
                border_radius=10,
            )

        draft_key = (tensor_id, offset)
        draft = self.hex_drafts.get(draft_key, tensor_tools.format_hex_bytes(raw))
        validation_error = self.hex_errors.get(tensor_id)
        page_end = offset + len(raw)
        return ft.Column(
            data=f"tensor-hex-editor:{tensor_id}",
            controls=[
                overview.section_heading(
                    "Hex editor",
                    ft.Icons.CODE_ROUNDED,
                    trailing=f"{len(raw)} BYTE PAGE",
                ),
                ft.Text(
                    "Edits create a reversible revision; the source model is "
                    "never overwritten.",
                    size=10,
                    color=self.palette.muted,
                ),
                ft.Row(
                    controls=[
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,
                            tooltip="Previous 128 bytes",
                            disabled=offset == 0,
                            data=f"tensor-hex-previous:{tensor_id}",
                            on_click=self.hex_page_handler(
                                tensor_id,
                                node_id,
                                -tensor_tools.HEX_PAGE_BYTES,
                            ),
                        ),
                        self._watch_text_focus(
                            ft.TextField(
                                value=f"0x{offset:X}",
                                label="Offset",
                                dense=True,
                                expand=True,
                                data=f"tensor-hex-offset:{tensor_id}",
                                on_submit=self._hex_offset_handler(tensor_id, node_id),
                            )
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_RIGHT_ROUNDED,
                            tooltip="Next 128 bytes",
                            disabled=page_end >= total,
                            data=f"tensor-hex-next:{tensor_id}",
                            on_click=self.hex_page_handler(
                                tensor_id,
                                node_id,
                                tensor_tools.HEX_PAGE_BYTES,
                            ),
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(
                    f"Bytes {offset:,}-{max(offset, page_end - 1):,} of {total:,}",
                    size=9,
                    color=self.palette.muted,
                ),
                self._watch_text_focus(
                    ft.TextField(
                        value=draft,
                        multiline=True,
                        min_lines=2,
                        max_lines=8,
                        dense=True,
                        text_style=ft.TextStyle(
                            font_family="monospace",
                            size=10,
                            color=self.palette.ink,
                        ),
                        label="Hex bytes",
                        helper=(
                            "Whitespace-separated byte pairs, for example: 00 FF 2A"
                        ),
                        error=validation_error,
                        autocorrect=False,
                        enable_suggestions=False,
                        smart_dashes_type=False,
                        smart_quotes_type=False,
                        data=f"tensor-hex-input:{tensor_id}",
                        on_change=self._hex_draft_handler(tensor_id, offset),
                    )
                ),
                ft.Container(
                    content=ft.Text(
                        tensor_tools.ascii_preview(raw) or "(empty)",
                        size=10,
                        font_family="monospace",
                        color=self.palette.muted,
                        selectable=True,
                    ),
                    padding=8,
                    bgcolor=self.palette.subtle,
                    border=ft.Border.all(1, self.palette.border),
                    border_radius=8,
                    tooltip="ASCII preview of the current persisted byte page",
                ),
                ft.Row(
                    controls=[
                        ft.FilledButton(
                            content="Apply bytes",
                            icon=ft.Icons.SAVE_ROUNDED,
                            disabled=not raw,
                            data=f"tensor-hex-apply:{tensor_id}",
                            on_click=self._hex_apply_handler(
                                tensor_id, node_id, offset, len(raw)
                            ),
                        ),
                        ft.TextButton(
                            content="Revert input",
                            data=f"tensor-hex-revert:{tensor_id}",
                            on_click=self._hex_revert_handler(
                                tensor_id, node_id, offset
                            ),
                        ),
                    ],
                    wrap=True,
                    spacing=5,
                ),
            ],
            spacing=7,
            tight=True,
        )

    def _hex_draft_handler(
        self, tensor_id: str, offset: int
    ) -> Callable[[ft.Event[ft.TextField]], None]:
        def remember(event: ft.Event[ft.TextField]) -> None:
            self.hex_drafts[(tensor_id, offset)] = event.control.value
            self.hex_errors.pop(tensor_id, None)

        return remember

    def hex_page_handler(
        self, tensor_id: str, node_id: str, delta: int
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        """Move the visible byte page without reading the whole tensor."""

        def navigate(event: ft.Event[ft.IconButton]) -> None:
            session = self._session()
            assert session is not None
            total = session.editing.byte_length(tensor_id) or 0
            maximum = (
                max(total - 1, 0) // tensor_tools.HEX_PAGE_BYTES
            ) * tensor_tools.HEX_PAGE_BYTES
            current = self.hex_offsets.get(tensor_id, 0)
            self.hex_offsets[tensor_id] = min(max(current + delta, 0), maximum)
            self.hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

        return navigate

    def _hex_offset_handler(
        self, tensor_id: str, node_id: str
    ) -> Callable[[ft.Event[ft.TextField]], None]:
        def jump(event: ft.Event[ft.TextField]) -> None:
            session = self._session()
            assert session is not None
            total = session.editing.byte_length(tensor_id) or 0
            try:
                offset = tensor_tools.parse_offset(
                    event.control.value, total_bytes=total
                )
            except ValueError as error:
                self.hex_errors[tensor_id] = str(error)
            else:
                self.hex_offsets[tensor_id] = offset
                self.hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

        return jump

    def _hex_apply_handler(
        self, tensor_id: str, node_id: str, offset: int, page_length: int
    ) -> Callable[[ft.Event[ft.Button]], None]:
        def apply(event: ft.Event[ft.Button]) -> None:
            session = self._session()
            assert session is not None
            key = (tensor_id, offset)
            try:
                current = session.editing.read(
                    tensor_id, offset=offset, length=page_length
                )
                draft = self.hex_drafts.get(key, tensor_tools.format_hex_bytes(current))
                replacement = tensor_tools.parse_hex_bytes(
                    draft, max_bytes=tensor_tools.HEX_PAGE_BYTES
                )
                if len(replacement) > page_length:
                    raise ValueError(
                        f"this page has only {page_length} editable byte(s)"
                    )
                session.editing.replace_bytes(tensor_id, offset, replacement)
            except Exception as error:
                self.hex_errors[tensor_id] = str(error)
                self._refresh_inspector(frozenset({node_id}))
                self._on_status(f"Hex edit rejected: {error}")
                self.page.update()
                return
            self.hex_drafts.pop(key, None)
            self.hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self._show_committed_diff()
            self._refresh_edit_actions()
            self._on_status(
                f"Applied {len(replacement)} byte(s) to {tensor_id} at 0x{offset:X}"
            )
            self.page.update()

        return apply

    def _hex_revert_handler(
        self, tensor_id: str, node_id: str, offset: int
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def revert(event: ft.Event[ft.TextButton]) -> None:
            self.hex_drafts.pop((tensor_id, offset), None)
            self.hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self._on_status("Hex input restored to the current tensor revision")
            self.page.update()

        return revert

    # -- typed-tensor preview consent -------------------------------------

    def preview_row(self, tensor_id: str) -> ft.Control:
        """The first few decoded values, or the reason they are unavailable."""
        session = self._session()
        assert session is not None
        tensor = session.document.tensors[tensor_id]
        if (
            tensor.storage is not Storage.GENERATED
            and session.store.materialization(tensor_id).value == "full-parse"
            and tensor_id not in self.typed_preview_consent
        ):
            return ft.TextButton(
                content="Load preview (materializes the full typed tensor)",
                on_click=self.typed_preview_handler(tensor_id),
            )
        try:
            raw = session.editing.read(
                tensor_id, offset=0, length=self._preview_bytes(tensor_id)
            )
        except Exception as error:
            return ft.Text(f"Preview unavailable: {error}", size=11, italic=True)
        return ft.Text(
            f"Preview: {viewmodel.preview_values_text(tensor.element_type, raw)}",
            size=11,
            selectable=True,
        )

    def typed_preview_handler(
        self, tensor_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        """Record consent to parse one typed protobuf tensor in full."""

        def load(event: ft.Event[ft.TextButton]) -> None:
            self.typed_preview_consent.add(tensor_id)
            self._refresh_inspector(self._selection())
            self.page.update()

        return load

    def _preview_bytes(self, tensor_id: str) -> int:
        session = self._session()
        assert session is not None
        tensor = session.document.tensors[tensor_id]
        width = element_width(tensor.element_type) or 1
        total = session.editing.byte_length(tensor_id) or 0
        return min(total, 9 * width)

    def statistics_handler(
        self, tensor_id: str, node_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        """Compute one weight tensor's statistics off the UI thread."""

        def compute(event: ft.Event[ft.TextButton]) -> None:
            session = self._session()
            assert session is not None
            self._on_status(f"Computing statistics for {tensor_id}…")
            job = session.compute_statistics_async(tensor_id)

            def wait() -> None:
                # run_thread futures are never retrieved, so anything that
                # raises here must be surfaced in the status bar itself.
                try:
                    job.wait()
                    if job.state.value == "succeeded":
                        self._on_status(f"Statistics ready for {tensor_id}")
                    elif job.state.value == "cancelled":
                        self._on_status("Statistics cancelled")
                    else:
                        self._on_status(f"Statistics failed: {job.error}")
                    # Only re-render when the model was not reopened and the
                    # inspector still shows the node whose card started the
                    # job; otherwise the completion would clobber whatever
                    # the user is looking at now.
                    if self._session() is session and self._inspected_ids() == (
                        frozenset({node_id})
                    ):
                        self._refresh_inspector(frozenset({node_id}))
                except Exception as error:
                    self._on_status(f"Statistics failed: {error}")
                finally:
                    self.page.update()

            self.page.run_thread(wait)
            self.page.update()

        return compute


def _format_count(value: float) -> str:
    """Format a histogram bar count compactly enough for a narrow axis."""
    number = round(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 10_000:
        return f"{number / 1_000:.1f}k"
    return f"{number:,}"


def _format_axis_value(value: float) -> str:
    """Format a bin-edge value without fake precision."""
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100_000 or magnitude < 0.001:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _histogram_shapes(
    view: ActivationVisualization,
    *,
    palette: ShellPalette,
    width: float,
    height: float,
) -> list[cv.Shape]:
    """Draw histogram bars with count and value axes labelled from the data.

    The x axis spans the actual bin edges the statistics pass produced, so
    the labels are real captured values, not normalized positions; the y
    axis is labelled with raw bar counts.
    """
    large = width >= 400.0
    label_size = 11.0 if large else 8.0
    axis_left = label_size * 4.0
    axis_bottom = label_size * 2.0
    plot_width = max(1.0, width - axis_left)
    plot_height = max(1.0, height - axis_bottom)
    maximum = max(view.values, default=0.0)
    label_style = ft.TextStyle(size=label_size, color=palette.muted)
    axis_paint = ft.Paint(
        color=palette.muted,
        stroke_width=1,
        style=ft.PaintingStyle.STROKE,
    )
    shapes: list[cv.Shape] = []
    bar_width = plot_width / max(len(view.values), 1)
    for index, value in enumerate(view.values):
        bar_height = 0.0 if maximum == 0 else value / maximum * plot_height
        shapes.append(
            cv.Rect(
                x=axis_left + index * bar_width,
                y=plot_height - bar_height,
                width=max(1.0, bar_width - 1.0),
                height=bar_height,
                paint=ft.Paint(
                    color=view.colors[index],
                    style=ft.PaintingStyle.FILL,
                ),
            )
        )
    shapes.append(
        cv.Line(x1=axis_left, y1=0.0, x2=axis_left, y2=plot_height, paint=axis_paint)
    )
    shapes.append(
        cv.Line(
            x1=axis_left,
            y1=plot_height,
            x2=width,
            y2=plot_height,
            paint=axis_paint,
        )
    )
    count_ticks = (0.0, maximum / 2.0, maximum) if large and maximum else (0.0, maximum)
    seen_counts: set[str] = set()
    for count in count_ticks:
        label = _format_count(count)
        if label in seen_counts:
            continue
        seen_counts.add(label)
        y = plot_height if maximum == 0 else plot_height - count / maximum * plot_height
        shapes.append(
            cv.Line(x1=axis_left - 3.0, y1=y, x2=axis_left, y2=y, paint=axis_paint)
        )
        shapes.append(
            cv.Text(
                x=axis_left - 5.0,
                # Keep the topmost label's upper half inside the canvas.
                y=max(y, label_size * 0.6),
                value=label,
                style=label_style,
                alignment=ft.Alignment.CENTER_RIGHT,
                text_align=ft.TextAlign.RIGHT,
                max_lines=1,
            )
        )
    edges = view.bin_edges
    if len(edges) >= 2:
        tick_count = 5 if large else 3
        for step in range(tick_count):
            fraction = step / (tick_count - 1)
            edge = edges[0] + (edges[-1] - edges[0]) * fraction
            x = axis_left + fraction * plot_width
            shapes.append(
                cv.Line(
                    x1=x,
                    y1=plot_height,
                    x2=x,
                    y2=plot_height + 3.0,
                    paint=axis_paint,
                )
            )
            alignment = ft.Alignment.TOP_CENTER
            if step == 0:
                alignment = ft.Alignment.TOP_LEFT
            elif step == tick_count - 1:
                alignment = ft.Alignment.TOP_RIGHT
            shapes.append(
                cv.Text(
                    x=x,
                    y=plot_height + 4.0,
                    value=_format_axis_value(edge),
                    style=label_style,
                    alignment=alignment,
                    text_align=ft.TextAlign.CENTER,
                    max_lines=1,
                )
            )
    return shapes


def build_activation_plot(
    view: ActivationVisualization,
    *,
    palette: ShellPalette,
    width: float = 180.0,
    height: float = 82.0,
) -> ft.Control:
    """Render a bounded adapter view; sampling/color choices came from core."""
    if view.layer_pngs:
        return build_activation_layer_viewer(
            view,
            width=width,
            height=max(height, 190.0),
            panel_color=palette.panel,
            border_color=palette.border,
            accent_color=palette.accent,
            muted_color=palette.muted,
        )
    if view.raster_png and view.raster_size is not None:
        raster_width, raster_height = view.raster_size
        scale = min(
            width / max(raster_width, 1),
            height / max(raster_height, 1),
        )
        rendered_width = max(1.0, raster_width * scale)
        rendered_height = max(1.0, raster_height * scale)
        return ft.Container(
            data=f"activation-plot:{view.kind.value}",
            content=ft.Image(
                src=view.raster_png,
                width=rendered_width,
                height=rendered_height,
                fit=ft.BoxFit.FILL,
                filter_quality=ft.FilterQuality.NONE,
            ),
            width=rendered_width,
            height=rendered_height,
            alignment=ft.Alignment.CENTER,
            padding=8,
            bgcolor=palette.panel,
            border=ft.Border.all(1, palette.border),
            border_radius=8,
        )
    shapes: list[cv.Shape] = []
    if view.kind.value == "feature-map-grid" and len(view.shape) == 3:
        maps, rows, columns = view.shape
        grid_columns = max(1, math.ceil(math.sqrt(maps)))
        grid_rows = max(1, math.ceil(maps / grid_columns))
        large = width >= 500
        gap = 6.0 if large else 2.0
        cell = min(
            (width - gap * (grid_columns - 1)) / max(grid_columns * columns, 1),
            (height - gap * (grid_rows - 1)) / max(grid_rows * rows, 1),
        )
        if not large:
            cell = min(cell, 3.0)
        cell = max(cell, 0.5)
        map_width = columns * cell
        map_height = rows * cell
        count = maps * rows * columns
        for index, color in enumerate(view.colors[:count]):
            map_index, within = divmod(index, rows * columns)
            row, column = divmod(within, columns)
            map_column = map_index % grid_columns
            map_row = map_index // grid_columns
            shapes.append(
                cv.Rect(
                    x=map_column * (map_width + gap) + column * cell,
                    y=map_row * (map_height + gap) + row * cell,
                    width=max(0.5, cell - 0.35),
                    height=max(0.5, cell - 0.35),
                    paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
                )
            )
        width = grid_columns * map_width + (grid_columns - 1) * gap
        height = grid_rows * map_height + (grid_rows - 1) * gap
    elif view.kind.value in {"heatmap", "attention-map"} and len(view.shape) >= 2:
        rows, columns = view.shape[-2:]
        cell = min(
            width / max(columns, 1),
            height / max(rows, 1),
        )
        if width < 500:
            cell = min(cell, 7.0)
        count = rows * columns
        for index, color in enumerate(view.colors[:count]):
            shapes.append(
                cv.Rect(
                    x=(index % columns) * cell,
                    y=(index // columns) * cell,
                    width=max(1.0, cell - 0.5),
                    height=max(1.0, cell - 0.5),
                    paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
                )
            )
        width = columns * cell
        height = rows * cell
    elif view.kind.value == "histogram":
        shapes.extend(
            _histogram_shapes(view, palette=palette, width=width, height=height)
        )
    else:
        finite = [value for value in view.values if math.isfinite(value)]
        minimum = min(finite, default=0.0)
        maximum = max(finite, default=0.0)
        span = maximum - minimum

        def point(index: int, value: float) -> tuple[float, float]:
            x = index / max(len(view.values) - 1, 1) * width
            normalized = 0.5 if span == 0 else (value - minimum) / span
            return x, height - normalized * height

        previous: tuple[float, float] | None = None
        for index, value in enumerate(view.values):
            if not math.isfinite(value):
                previous = None
                continue
            current = point(index, value)
            if previous is not None:
                shapes.append(
                    cv.Line(
                        x1=previous[0],
                        y1=previous[1],
                        x2=current[0],
                        y2=current[1],
                        paint=ft.Paint(
                            color=view.colors[index],
                            stroke_width=2,
                            style=ft.PaintingStyle.STROKE,
                        ),
                    )
                )
            previous = current
    return ft.Container(
        data=f"activation-plot:{view.kind.value}",
        content=cv.Canvas(shapes=shapes, width=width, height=height),
        alignment=ft.Alignment.CENTER,
        padding=8,
        bgcolor=palette.panel,
        border=ft.Border.all(1, palette.border),
        border_radius=8,
    )
