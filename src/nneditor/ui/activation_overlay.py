"""Tabbed, lazy inspection of one captured tensor, independent of graph selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import flet as ft

from nneditor.analysis.statistics import TensorStatistics
from nneditor.application.jobs import Job
from nneditor.application.session import ModelSession
from nneditor.detokenization import DecodedActivation, text_activation_kind
from nneditor.tensor_scripting import TensorScriptResult
from nneditor.tracing.contracts import ActivationRecord, CaptureState
from nneditor.tracing.visualization import ActivationVisualization
from nneditor.ui import overview, viewmodel
from nneditor.ui.shell_layout import ShellPalette
from nneditor.ui.tensor_page import PAGE_COLUMNS, PAGE_ROWS, read_tensor_page

Key = tuple[str, str]
Views = tuple[ActivationVisualization, ...]
DEFAULT_SCRIPT = """print("shape:", tensor.shape, "dtype:", tensor.dtype)
values = np.asarray(tensor).ravel()[:100_000]
values = values[np.isfinite(values)]
print("sample mean:", values.mean() if values.size else "no finite values")
plt.hist(values, bins=64)
plt.title("Captured tensor values (first 100,000)")
plt.tight_layout()
"""


class ActivationOverlay:
    """The overlay owns UI state; capture payloads and analysis caches stay shared."""

    def __init__(
        self,
        *,
        page: ft.Page,
        palette: ShellPalette,
        session: Callable[[], ModelSession | None],
        statistics: dict[Key, TensorStatistics],
        get_views: Callable[[Key], Views | None],
        put_views: Callable[[Key, Views], object],
        save_array: Callable[[str, ActivationRecord], Callable[..., Any]],
        save_plot: Callable[[ActivationVisualization, str], Callable[..., Any]],
        on_statistics_ready: Callable[[], None] | None = None,
    ) -> None:
        self.page, self.palette, self._session = page, palette, session
        self._statistics, self._get_views, self._put_views = (
            statistics,
            get_views,
            put_views,
        )
        self._save_array, self._save_plot = save_array, save_plot
        self._on_statistics_ready = on_statistics_ready
        self.dialog: ft.AlertDialog | None = None
        self.record: ActivationRecord | None = None
        self.trace_id = self.value_id = ""
        self._epoch = 0
        self._jobs: dict[str, Job[Any]] = {}
        self._errors: dict[str, str] = {}
        self._script_draft = DEFAULT_SCRIPT

    def open(self, trace_id: str, value_id: str, *, title: str | None = None) -> None:
        session = self._session()
        if session is None:
            return
        self.close()
        self.trace_id, self.value_id = trace_id, value_id
        self._errors.clear()
        try:
            self.record = session.trace(trace_id).record(value_id)
        except KeyError:
            self.record = None
        record = self.record
        self._width = max(
            300.0, min(1040.0, float(getattr(self.page, "width", None) or 1200) - 64)
        )
        height = max(
            280.0, min(640.0, float(getattr(self.page, "height", None) or 900) - 220)
        )
        self.leading = ft.TextField(
            label="Leading axis indices",
            value=", ".join("0" for _ in record.shape[:-2]) if record else "",
            hint_text="One index per axis, e.g. 0, 2",
            dense=True,
            width=230,
        )
        self.row = ft.TextField(label="First row", value="0", dense=True, width=110)
        self.column = ft.TextField(
            label="First column", value="0", dense=True, width=120
        )
        self.leading.visible = record is not None and len(record.shape) > 2
        kind = text_activation_kind(session.document, record) if record else None
        self.decode_batch = ft.TextField(
            label="Batch", value="0", width=100, dense=True
        )
        self.decode_position = ft.TextField(
            label="Token position",
            value="",
            width=150,
            dense=True,
            hint_text="Latest captured" if kind == "logits" else "Start at 0",
        )
        self.decode_skip = ft.Checkbox(
            label="Hide special tokens in text", value=True, width=220
        )
        self.decode_button = ft.FilledTonalButton(
            content="Inspect next-token predictions"
            if kind == "logits"
            else "Decode captured tokens",
            icon=ft.Icons.TEXT_FIELDS_ROUNDED,
            on_click=self._decode_text,
        )
        self.decode_result = ft.Column(spacing=8)
        self.decode_panel = ft.Container(
            data="activation-detokenizer",
            visible=kind is not None,
            bgcolor=self.palette.accent_soft,
            padding=12,
            border_radius=10,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text("Decoded text", weight=ft.FontWeight.W_600),
                    ft.Row(
                        wrap=True,
                        controls=[
                            self.decode_batch,
                            self.decode_position,
                            self.decode_skip,
                            self.decode_button,
                        ],
                    ),
                    self.decode_result,
                ],
            ),
        )
        self.view_area = ft.ListView(spacing=12, expand=True)
        self.statistics_area = ft.ListView(spacing=12, expand=True)
        self.visualization_area = ft.ListView(spacing=14, expand=True)
        self.script_editor = ft.TextField(
            label="Python script",
            value=self._script_draft,
            multiline=True,
            min_lines=10,
            max_lines=18,
            text_style=ft.TextStyle(font_family="monospace", size=12),
            on_change=self._on_script_change,
        )
        self.script_timeout = ft.TextField(
            label="Timeout (s)", value="120", width=120, dense=True
        )
        self.script_memory = ft.TextField(
            label="RAM (MiB)", value="4096", width=130, dense=True
        )
        self.script_run = ft.FilledButton(
            content="Run script",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            on_click=self.run_script,
        )
        self.script_cancel = ft.TextButton(
            content="Stop",
            icon=ft.Icons.STOP_ROUNDED,
            disabled=True,
            on_click=self._cancel_script,
        )
        self.script_clear = ft.TextButton(
            content="Clear", on_click=lambda _: self._set_script("")
        )
        self.script_example = ft.TextButton(
            content="Example", on_click=lambda _: self._set_script(DEFAULT_SCRIPT)
        )
        self.script_output = ft.Column(spacing=10)
        self.script_area = ft.ListView(
            spacing=12,
            expand=True,
            controls=[
                ft.Text(
                    "Preloaded: tensor, metadata, np, math, scipy, sklearn "
                    "(also scikit), matplotlib, and plt. "
                    "Print output, return a final expression, "
                    "or create Matplotlib figures. Ctrl+Enter runs the script.",
                    size=12,
                ),
                ft.Text(
                    "Only the captured prefix is available. tensor is a flat array; "
                    "metadata['shape'] contains the original shape.",
                    visible=bool(
                        record and record.stored_byte_length < record.full_byte_length
                    ),
                    color=self.palette.warning,
                ),
                self.script_editor,
                ft.Row(
                    wrap=True,
                    controls=[
                        self.script_timeout,
                        self.script_memory,
                        self.script_run,
                        self.script_cancel,
                        self.script_clear,
                        self.script_example,
                    ],
                ),
                self.script_output,
            ],
        )
        if not record or not record.readable:
            self.script_run.disabled = True
            unavailable = self._unavailable()
            if unavailable:
                self.script_output.controls = [unavailable]
        if getattr(self.page, "web", False):
            self.script_run.disabled = True
            self.script_output.controls = [
                ft.Text("Python scripting is available in the desktop app.")
            ]
        self.tabs = ft.Tabs(
            length=4,
            selected_index=0,
            expand=True,
            on_change=self._on_tab,
            content=ft.Column(
                expand=True,
                spacing=10,
                controls=[
                    ft.TabBar(
                        scrollable=True,
                        tabs=[
                            ft.Tab(label="View", icon=ft.Icons.TABLE_VIEW_ROUNDED),
                            ft.Tab(
                                label="Statistics", icon=ft.Icons.ANALYTICS_OUTLINED
                            ),
                            ft.Tab(
                                label="Visualization",
                                icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                            ),
                            ft.Tab(label="Python", icon=ft.Icons.CODE_ROUNDED),
                        ],
                    ),
                    ft.TabBarView(
                        controls=[
                            self.view_area,
                            self.statistics_area,
                            self.visualization_area,
                            self.script_area,
                        ],
                        expand=True,
                    ),
                ],
            ),
        )
        state = (
            "Preview"
            if record and record.state is CaptureState.TRUNCATED
            else record.state.value.capitalize()
            if record
            else "Not captured"
        )
        shape = " x ".join(map(str, record.shape)) if record else ""
        summary = (
            f"{record.element_type} · {shape or 'scalar'} · {state}"
            if record
            else "No payload was captured for this tensor."
        )
        actions: list[ft.Control] = []
        if record and record.readable:
            actions.append(
                ft.TextButton(
                    content="Save as .npy",
                    icon=ft.Icons.SAVE_ALT_ROUNDED,
                    on_click=self._save_array(trace_id, record),
                )
            )
        actions.append(ft.TextButton(content="Close", on_click=lambda _: self.close()))
        self.dialog = ft.AlertDialog(
            data=f"tensor-overlay:{value_id}",
            modal=True,
            on_dismiss=self._on_dismiss,
            title=ft.Text(
                title or (record.value_name if record else "Captured tensor"),
                weight=ft.FontWeight.W_700,
            ),
            content=ft.Container(
                width=self._width,
                height=height,
                content=ft.Column(
                    expand=True,
                    spacing=10,
                    controls=[
                        ft.Text(
                            record.value_name if record else value_id,
                            size=12,
                            color=self.palette.muted,
                            selectable=True,
                        ),
                        ft.Text(summary, size=12, color=self.palette.ink),
                        ft.Text(
                            f"Captured {record.stored_byte_length:,} of "
                            f"{record.full_byte_length:,} bytes. "
                            "Values beyond this prefix are unavailable."
                            if record
                            else "",
                            size=11,
                            color=self.palette.warning,
                            visible=bool(
                                record
                                and record.stored_byte_length < record.full_byte_length
                            ),
                        ),
                        self.tabs,
                    ],
                ),
            ),
            actions=actions,
        )
        self._render_view()
        self.page.show_dialog(self.dialog)

    def close(self) -> bool:
        self._epoch += 1
        for job in self._jobs.values():
            if not job.state.is_terminal:
                job.cancel()
        self._jobs.clear()
        dialog, self.dialog = self.dialog, None
        if dialog is not None and dialog.open:
            self.page.pop_dialog()
            return True
        return False

    def _on_dismiss(self, event: ft.Event[ft.DialogControl]) -> None:
        if self.dialog is event.control:
            self.dialog = None
            self.close()

    def _unavailable(self) -> ft.Control | None:
        if self.record is None:
            text = (
                "This tensor was not selected for capture. Run a trace with "
                "Preview everything or include this block."
            )
        elif not self.record.readable:
            text = (
                self.record.reason
                or f"The capture is {self.record.state.value}. "
                "Run another trace to inspect its values."
            )
        else:
            return None
        return ft.Container(
            content=ft.Text(text, selectable=True, color=self.palette.warning),
            bgcolor=self.palette.warning_soft,
            padding=16,
            border_radius=10,
        )

    def _render_view(self) -> None:
        if unavailable := self._unavailable():
            self.view_area.controls = [unavailable]
            return
        session, record = self._session(), self.record
        assert session is not None and record is not None
        controls: list[ft.Control] = [
            self.decode_panel,
            ft.Row(
                wrap=True,
                controls=[
                    self.leading,
                    self.row,
                    self.column,
                    ft.FilledTonalButton(
                        content="Apply slice", on_click=self._apply_slice
                    ),
                ],
            ),
        ]
        try:
            leading = tuple(
                int(v.strip())
                for v in (self.leading.value or "").split(",")
                if v.strip()
            )
            page = read_tensor_page(
                record,
                session.activations(self.trace_id).read,
                leading=leading,
                row=int(self.row.value or "0"),
                column=int(self.column.value or "0"),
            )
        except (OSError, ValueError) as error:
            controls.append(ft.Text(str(error), color=self.palette.danger))
            self.view_area.controls = controls
            return
        controls.append(
            ft.Text(
                f"Rows {page.rows[0] if page.rows else 0} to "
                f"{page.rows[-1] if page.rows else 0} of {page.row_count} · "
                f"columns {page.columns[0] if page.columns else 0} to "
                f"{page.columns[-1] if page.columns else 0} of {page.column_count}",
                size=12,
                color=self.palette.muted,
            )
        )
        table = ft.DataTable(
            data="tensor-value-table",
            columns=[
                ft.DataColumn(label=ft.Text("Index")),
                *(ft.DataColumn(label=ft.Text(str(c))) for c in page.columns),
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(index), color=self.palette.muted)),
                        *(
                            ft.DataCell(
                                ft.Text(
                                    self._number(v),
                                    selectable=True,
                                    font_family="monospace",
                                    color=self.palette.danger
                                    if isinstance(v, float) and not math.isfinite(v)
                                    else self.palette.ink,
                                )
                            )
                            for v in values
                        ),
                    ]
                )
                for index, values in zip(page.rows, page.values, strict=True)
            ],
            column_spacing=18,
            data_row_min_height=34,
            data_row_max_height=38,
        )
        controls.append(ft.Row(controls=[table], scroll=ft.ScrollMode.AUTO))
        controls.append(
            ft.Row(
                wrap=True,
                controls=[
                    ft.TextButton(
                        content="Previous rows",
                        disabled=not page.rows or page.rows[0] == 0,
                        on_click=lambda _: self._move(-PAGE_ROWS, 0),
                    ),
                    ft.TextButton(
                        content="Next rows",
                        disabled=not page.rows or page.rows[-1] + 1 >= page.row_count,
                        on_click=lambda _: self._move(PAGE_ROWS, 0),
                    ),
                    ft.TextButton(
                        content="Previous columns",
                        disabled=not page.columns or page.columns[0] == 0,
                        on_click=lambda _: self._move(0, -PAGE_COLUMNS),
                    ),
                    ft.TextButton(
                        content="Next columns",
                        disabled=not page.columns
                        or page.columns[-1] + 1 >= page.column_count,
                        on_click=lambda _: self._move(0, PAGE_COLUMNS),
                    ),
                ],
            )
        )
        controls.append(
            ft.Text(
                "A dash means the value is outside the captured prefix."
                if record.stored_byte_length < record.full_byte_length
                else "Values are read in small pages; "
                "slicing does not load the whole tensor.",
                size=11,
                color=self.palette.muted,
            )
        )
        self.view_area.controls = controls

    @staticmethod
    def _number(value: bool | int | float | None) -> str:
        return (
            "—"
            if value is None
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
        )

    def _decode_text(self, _: ft.Event[ft.Button]) -> None:
        session = self._session()
        if session is None or self.dialog is None or "decode" in self._jobs:
            return
        try:
            batch = int(self.decode_batch.value or "0")
            position = (
                int(self.decode_position.value) if self.decode_position.value else None
            )
        except ValueError:
            self.decode_result.controls = [
                ft.Text(
                    "Use whole numbers for batch and token position.",
                    color=self.palette.danger,
                )
            ]
            self.page.update()
            return
        job = session.decode_activation_async(
            self.trace_id,
            self.value_id,
            batch=batch,
            position=position,
            skip_special=bool(self.decode_skip.value),
        )
        self._jobs["decode"] = job
        self.decode_button.disabled = True
        self.decode_result.controls = [
            ft.ProgressRing(width=20, height=20),
            ft.Text("Decoding…"),
        ]
        epoch = self._epoch
        self.page.update()

        def wait() -> None:
            job.wait()
            if (
                epoch != self._epoch
                or self._session() is not session
                or self.dialog is None
            ):
                return
            self._jobs.pop("decode", None)
            self.decode_button.disabled = False
            if job.state.value != "succeeded":
                self.decode_result.controls = [
                    ft.Text(
                        str(job.error or "Decoding cancelled."),
                        color=self.palette.warning,
                    )
                ]
            else:
                self._show_decoded(job.result())
            self.page.update()

        self.page.run_thread(wait)

    def _show_decoded(self, result: DecodedActivation) -> None:
        controls: list[ft.Control] = [
            ft.Text(
                result.text or "No printable text for these tokens.",
                selectable=True,
                size=16,
            ),
            ft.Text(f"{result.decoder} · {result.note}", size=11, selectable=True),
        ]
        if result.candidates:
            controls.append(
                ft.DataTable(
                    columns=[
                        ft.DataColumn(label=ft.Text("Token ID")),
                        ft.DataColumn(label=ft.Text("Vocabulary piece")),
                        ft.DataColumn(label=ft.Text("Probability")),
                    ],
                    rows=[
                        ft.DataRow(
                            cells=[
                                ft.DataCell(
                                    ft.Text(str(candidate.token_id), selectable=True)
                                ),
                                ft.DataCell(
                                    ft.Text(repr(candidate.piece), selectable=True)
                                ),
                                ft.DataCell(ft.Text(f"{candidate.probability:.2%}")),
                            ]
                        )
                        for candidate in result.candidates
                    ],
                )
            )
        else:
            controls.append(
                ft.Text(
                    "Token IDs: " + ", ".join(map(str, result.ids)),
                    size=11,
                    selectable=True,
                )
            )
        self.decode_result.controls = controls

    def _on_script_change(self, _: ft.Event[ft.TextField]) -> None:
        self._script_draft = self.script_editor.value or ""

    def _set_script(self, source: str) -> None:
        self._script_draft = source
        self.script_editor.value = source
        self.page.update()
        if run_task := getattr(self.page, "run_task", None):
            run_task(self.script_editor.focus)

    def run_script(self, _: ft.Event[ft.Button] | None = None) -> None:
        session = self._session()
        if (
            session is None
            or self.dialog is None
            or self.script_run.disabled
            or "script" in self._jobs
            or getattr(self.page, "web", False)
        ):
            return
        try:
            timeout = float(self.script_timeout.value or "120")
            memory = int(self.script_memory.value or "4096") * 1024**2
            self._script_draft = self.script_editor.value or ""
            job = session.run_tensor_script_async(
                self.trace_id,
                self.value_id,
                self._script_draft,
                wall_seconds=timeout,
                memory_bytes=memory,
            )
        except ValueError as error:
            self.script_output.controls = [
                ft.Text(str(error), color=self.palette.danger)
            ]
            self.page.update()
            return
        self._jobs["script"] = job
        self.script_run.disabled, self.script_cancel.disabled = True, False
        self.script_clear.disabled = self.script_example.disabled = (
            self.script_editor.disabled
        ) = True
        self.script_output.controls = [
            ft.ProgressRing(width=22, height=22),
            ft.Text("Running Python…"),
        ]
        epoch = self._epoch
        self.page.update()

        def wait() -> None:
            job.wait()
            if (
                epoch != self._epoch
                or self._session() is not session
                or self.dialog is None
            ):
                return
            self._jobs.pop("script", None)
            self.script_run.disabled, self.script_cancel.disabled = False, True
            self.script_clear.disabled = self.script_example.disabled = (
                self.script_editor.disabled
            ) = False
            if job.state.value == "succeeded":
                self._show_script_result(job.result())
            else:
                self.script_output.controls = [
                    ft.Text(
                        str(job.error or "Script stopped."), color=self.palette.warning
                    )
                ]
            self.page.update()

        self.page.run_thread(wait)

    def _cancel_script(self, _: ft.Event[ft.TextButton]) -> None:
        if job := self._jobs.get("script"):
            job.cancel()
        self.script_cancel.disabled = True
        self.page.update()

    def _show_script_result(self, result: TensorScriptResult) -> None:
        controls: list[ft.Control] = [
            ft.Text(f"Finished in {result.seconds:.2f} s", size=11)
        ]
        for label, value, color in (
            ("Output", result.stdout, self.palette.ink),
            ("Result", result.result, self.palette.ink),
            ("Messages", result.stderr, self.palette.warning),
            ("Error", result.error, self.palette.danger),
        ):
            if value:
                controls.extend(
                    [
                        ft.Text(label, weight=ft.FontWeight.W_600),
                        ft.Text(
                            value,
                            selectable=True,
                            font_family="monospace",
                            size=12,
                            color=color,
                        ),
                    ]
                )
        controls.extend(
            ft.Image(src=png, fit=ft.BoxFit.CONTAIN) for png in result.plots
        )
        self.script_output.controls = controls

    def _move(self, rows: int, columns: int) -> None:
        try:
            self.row.value = str(max(0, int(self.row.value or "0") + rows))
            self.column.value = str(max(0, int(self.column.value or "0") + columns))
        except ValueError:
            pass
        self._apply_slice()

    def _apply_slice(self, event: ft.Event[ft.Button] | None = None) -> None:
        if self.dialog is None:
            return
        self._render_view()
        self.page.update()

    def _on_tab(self, event: ft.Event[ft.Tabs]) -> None:
        if self.dialog is None:
            return
        index = self.tabs.selected_index
        if index in (1, 2):
            kind = "statistics" if index == 1 else "visualization"
            self._render_analysis(kind)
            self._request(kind)
        self.page.update()

    def _render_analysis(self, kind: str) -> None:
        area = self.statistics_area if kind == "statistics" else self.visualization_area
        if unavailable := self._unavailable():
            area.controls = [unavailable]
            return
        key = (self.trace_id, self.value_id)
        if kind in self._errors:
            area.controls = [
                ft.Text(self._errors[kind], color=self.palette.danger),
                ft.TextButton(
                    content="Retry", on_click=lambda _: self._request(kind, force=True)
                ),
            ]
        elif kind == "statistics" and key in self._statistics:
            stats = self._statistics[key]
            area.controls = [
                overview.metadata_section(
                    viewmodel.statistics_lines(stats),
                    palette=self.palette,
                    title="Tensor statistics",
                    icon=ft.Icons.ANALYTICS_OUTLINED,
                    role="tensor-statistics",
                )
            ]
            if (
                self.record
                and self.record.stored_byte_length < self.record.full_byte_length
            ):
                area.controls.insert(
                    0,
                    ft.Text(
                        "Statistics cover the captured prefix, "
                        "not the uncaptured values.",
                        color=self.palette.warning,
                    ),
                )
        elif kind == "visualization" and (views := self._get_views(key)) is not None:
            area.controls = self._plots(views)
        else:
            area.controls = [
                ft.ProgressRing(width=24, height=24),
                ft.Text(f"Loading {kind}…", color=self.palette.muted),
            ]

    def _plots(self, views: Views) -> list[ft.Control]:
        from nneditor.ui.activation_inspector import build_activation_plot

        controls: list[ft.Control] = []
        for plot in views:
            controls.extend(
                [
                    ft.Text(plot.title, size=16, weight=ft.FontWeight.W_600),
                    build_activation_plot(
                        plot,
                        palette=self.palette,
                        width=max(260.0, self._width - 48),
                        height=360.0,
                    ),
                    ft.Text(
                        f"{plot.downsampling} · "
                        f"{'Captured prefix' if plot.partial else 'Complete capture'}",
                        size=11,
                        color=self.palette.muted,
                    ),
                ]
            )
            if plot.raster_png and self.record:
                controls.append(
                    ft.TextButton(
                        content="Save view as PNG",
                        icon=ft.Icons.IMAGE_ROUNDED,
                        on_click=self._save_plot(plot, self.record.value_name),
                    )
                )
        return controls or [ft.Text("No finite values are available to visualize.")]

    def _request(self, kind: str, *, force: bool = False) -> None:
        session, record = self._session(), self.record
        key = (self.trace_id, self.value_id)
        cached = (
            key in self._statistics
            if kind == "statistics"
            else self._get_views(key) is not None
        )
        if (
            session is None
            or record is None
            or not record.readable
            or kind in self._jobs
            or (cached and not force)
            or (kind in self._errors and not force)
        ):
            return
        self._errors.pop(kind, None)
        try:
            job = (
                session.compute_activation_statistics_async(*key)
                if kind == "statistics"
                else session.activation_visualizations_async(*key)
            )
        except (KeyError, ValueError) as error:
            self._errors[kind] = str(error)
            self._render_analysis(kind)
            return
        self._jobs[kind] = job
        epoch = self._epoch
        self._render_analysis(kind)

        def wait() -> None:
            job.wait()
            if (
                epoch != self._epoch
                or self._session() is not session
                or self.dialog is None
            ):
                return
            self._jobs.pop(kind, None)
            if job.state.value == "succeeded":
                if kind == "statistics":
                    self._statistics[key] = cast(TensorStatistics, job.result())
                    if self._on_statistics_ready:
                        self._on_statistics_ready()
                else:
                    self._put_views(key, cast(Views, job.result()))
            else:
                self._errors[kind] = str(job.error or "The request was cancelled.")
            self._render_analysis(kind)
            self.page.update()

        self.page.run_thread(wait)
