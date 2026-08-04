"""Pinnable activation watch strip.

The watch panel owns the per-session list of pinned captured values and the
strip that renders each one: name, capture state, dtype/shape, its streamed
statistics once computed, and — when a second trace of the same input
specification exists — the per-value error metrics between the two traces.

It holds no reference to the shell.  Everything it needs about the current
session and trace arrives as an injected accessor, and everything it wants
the shell to do afterwards arrives as an injected callable.  Statistics live
in the activation inspector's shared store (injected), so a value whose
statistics the inspector already computed shows them here immediately and
vice versa.
"""

from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping

import flet as ft

from nneditor.analysis.statistics import TensorStatistics
from nneditor.application.session import ModelSession
from nneditor.tracing.comparison import ActivationError, TraceComparison
from nneditor.tracing.contracts import ActivationRecord, TraceResult
from nneditor.ui import overview, viewmodel
from nneditor.ui.shell_layout import ShellPalette

__all__ = ["MAX_CONCURRENT_WATCH_JOBS", "WatchPanel", "comparison_pair"]

# The session job pool runs 4 workers. The watch strip auto-computes missing
# statistics for its pinned values, so without a cap a long pin list would
# flood the pool and starve every manual request; rows beyond the cap wait
# for a landed job to free a slot.
MAX_CONCURRENT_WATCH_JOBS = 4


def comparison_pair(
    traces: tuple[TraceResult, ...],
    active_trace_id: str | None,
) -> tuple[TraceResult, TraceResult] | None:
    """The (base, active) trace pair eligible for per-value deltas.

    Eligible means the exact comparability contract ``compare_traces``
    enforces: same source artifact and same input-specification hash. With
    several candidates the lowest trace id is chosen so the pair — and the
    cached comparison — is deterministic.
    """
    active = next((trace for trace in traces if trace.id == active_trace_id), None)
    if active is None:
        return None
    base = next(
        (
            trace
            for trace in sorted(traces, key=lambda item: item.id)
            if trace.id != active.id
            and trace.key.artifact_hash == active.key.artifact_hash
            and trace.key.input_specification_hash
            == active.key.input_specification_hash
        ),
        None,
    )
    if base is None:
        return None
    return (base, active)


def _delta_value(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:.3g}"


class WatchPanel:
    """Own the pinned-value list, its jobs, and the strip that shows them."""

    def __init__(
        self,
        *,
        page: ft.Page,
        palette: ShellPalette,
        session: Callable[[], ModelSession | None],
        active_trace_id: Callable[[], str | None],
        statistics: Callable[[], MutableMapping[tuple[str, str], TensorStatistics]],
        statistics_loading: Callable[[], set[tuple[str, str, str]]],
        on_statistics_ready: Callable[[], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        max_concurrent_jobs: int = MAX_CONCURRENT_WATCH_JOBS,
    ) -> None:
        self.page = page
        self.palette = palette
        self._session = session
        self._active_trace_id = active_trace_id
        self._statistics = statistics
        self._statistics_loading = statistics_loading
        self._on_statistics_ready = on_statistics_ready
        self._on_status = on_status
        self._on_error = on_error
        self._max_concurrent_jobs = max_concurrent_jobs

        # Pinned value id -> captured value name, in pin order.
        self._pins: dict[str, str] = {}
        self._statistics_errors: dict[tuple[str, str], str] = {}
        # One comparison per (base, active) trace pair, indexed per value at
        # render time; recomputing per row would repeat the streaming pass.
        self._comparisons: dict[tuple[str, str], TraceComparison] = {}
        self._comparison_loading: set[tuple[str, str]] = set()
        self._comparison_errors: dict[tuple[str, str], str] = {}
        self.strip = ft.Column(spacing=8, tight=True, data="watch-strip")
        self.refresh()

    @property
    def control(self) -> ft.Column:
        """The retained strip the shell mounts in the left panel's
        Activations section."""
        return self.strip

    # -- pin state ---------------------------------------------------------

    def pinned(self) -> tuple[str, ...]:
        """The pinned value ids, in pin order."""
        return tuple(self._pins)

    def pin(self, value_id: str, value_name: str) -> None:
        """Add one captured value to the watch list (idempotent)."""
        already = value_id in self._pins
        self._pins[value_id] = value_name
        self.refresh()
        self._on_status(
            f"{value_name} is already on the watch list"
            if already
            else f"Pinned {value_name} to the watch list"
        )

    def unpin(self, value_id: str) -> None:
        name = self._pins.pop(value_id, None)
        self.refresh()
        if name is not None:
            self._on_status(f"Unpinned {name} from the watch list")

    def reset(self) -> None:
        """Drop every pin and cached comparison; pins are per-session state."""
        self._pins.clear()
        self._statistics_errors.clear()
        self._comparisons.clear()
        self._comparison_loading.clear()
        self._comparison_errors.clear()
        self.refresh()

    # -- strip rendering ---------------------------------------------------

    def refresh(self) -> None:
        """Re-render every watched row from the current session and trace."""
        self.strip.controls = self._build_rows()

    def _build_rows(self) -> list[ft.Control]:
        if not self._pins:
            return [
                ft.Text(
                    "Pin any captured activation from the inspector to keep "
                    "it in view here.",
                    size=10,
                    color=self.palette.muted,
                )
            ]
        session = self._session()
        trace_id = self._active_trace_id()
        result: TraceResult | None = None
        if session is not None and trace_id is not None:
            try:
                result = session.trace(trace_id)
            except (KeyError, ValueError):
                result = None
        records = (
            {record.value_id: record for record in result.records}
            if result is not None
            else {}
        )
        pair = (
            comparison_pair(session.traces(), trace_id)
            if session is not None and result is not None
            else None
        )
        return [
            self._row(value_id, value_name, trace_id, records.get(value_id), pair)
            for value_id, value_name in self._pins.items()
        ]

    def _row(
        self,
        value_id: str,
        value_name: str,
        trace_id: str | None,
        record: ActivationRecord | None,
        pair: tuple[TraceResult, TraceResult] | None,
    ) -> ft.Control:
        body: list[ft.Control] = [
            ft.Row(
                controls=[
                    ft.Text(
                        value_name,
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=self.palette.ink,
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        tooltip=value_id,
                        expand=True,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE_ROUNDED,
                        icon_size=14,
                        tooltip="Unpin from the watch list",
                        data=f"watch-unpin:{value_id}",
                        on_click=self._unpin_handler(value_id),
                    ),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        ]
        if trace_id is None:
            body.append(
                ft.Text(
                    "Run a trace to capture this value.",
                    size=10,
                    color=self.palette.muted,
                )
            )
        elif record is None:
            body.append(
                ft.Text(
                    "Not captured by the active trace.",
                    size=10,
                    color=self.palette.warning,
                )
            )
        else:
            shape = " x ".join(str(dimension) for dimension in record.shape) or "scalar"
            body.append(
                ft.Text(
                    f"{record.state.value} · {record.element_type} · {shape}",
                    size=10,
                    color=self.palette.muted,
                )
            )
            body.extend(self._statistics_controls(trace_id, record))
            body.extend(self._delta_controls(value_id, pair))
        return ft.Container(
            data=f"watch-row:{value_id}",
            content=ft.Column(controls=body, spacing=5, tight=True),
            padding=8,
            bgcolor=self.palette.subtle,
            border=ft.Border.all(1, self.palette.border),
            border_radius=10,
        )

    def _statistics_controls(
        self,
        trace_id: str,
        record: ActivationRecord,
    ) -> list[ft.Control]:
        """The row's statistics block, sharing the inspector's job states."""
        if not record.readable:
            return [
                ft.Text(
                    record.reason
                    or f"This capture is {record.state.value} and cannot be read.",
                    size=10,
                    color=self.palette.warning,
                )
            ]
        key = (trace_id, record.value_id)
        started = True
        if (
            self._statistics().get(key) is None
            and key not in self._statistics_errors
            and (trace_id, record.value_id, "statistics")
            not in self._statistics_loading()
        ):
            started = self._start_statistics(trace_id, record.value_id)
        # Re-read after the start: the headless harness runs job waits
        # inline, so the result may already have landed by this line.
        stats = self._statistics().get(key)
        if stats is not None:
            return [
                overview.metadata_section(
                    viewmodel.statistics_lines(stats),
                    palette=self.palette,
                    title="Statistics",
                    icon=ft.Icons.QUERY_STATS_ROUNDED,
                    role=f"watch-statistics:{record.value_id}",
                )
            ]
        if error := self._statistics_errors.get(key):
            return [
                ft.Text(
                    f"Statistics failed: {error}",
                    size=10,
                    color=self.palette.danger,
                ),
                ft.TextButton(
                    content="Retry statistics",
                    icon=ft.Icons.REFRESH_ROUNDED,
                    data=f"watch-retry:{record.value_id}",
                    on_click=self._retry_handler(trace_id, record.value_id),
                ),
            ]
        if started:
            return [
                ft.Row(
                    controls=[
                        ft.ProgressRing(width=14, height=14),
                        ft.Text("Loading activation statistics…", size=10),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            ]
        return [
            ft.Text(
                "Waiting for a free statistics slot…",
                size=10,
                color=self.palette.muted,
            )
        ]

    def _delta_controls(
        self,
        value_id: str,
        pair: tuple[TraceResult, TraceResult] | None,
    ) -> list[ft.Control]:
        """Per-value error metrics against the comparable base trace."""
        if pair is None:
            return []
        base, active = pair
        key = (base.id, active.id)
        if (
            key not in self._comparisons
            and key not in self._comparison_loading
            and key not in self._comparison_errors
        ):
            self._start_comparison(base, active)
        # Re-read after the start: the headless harness runs job waits
        # inline, so the comparison may already have landed by this line.
        comparison = self._comparisons.get(key)
        if comparison is None:
            if error := self._comparison_errors.get(key):
                return [
                    ft.Text(
                        f"Trace comparison failed: {error}",
                        size=10,
                        color=self.palette.danger,
                    )
                ]
            return [
                ft.Text(
                    "Comparing against the previous same-input trace…",
                    size=10,
                    color=self.palette.info,
                )
            ]
        metric = next(
            (item for item in comparison.values if item.value_id == value_id),
            None,
        )
        if metric is None:
            return [
                ft.Text(
                    "Not comparable across the two traces.",
                    size=10,
                    color=self.palette.muted,
                )
            ]
        return [self._delta_section(value_id, metric)]

    def _delta_section(self, value_id: str, metric: ActivationError) -> ft.Control:
        lines = (
            ("Δ max abs", _delta_value(metric.max_absolute)),
            ("Δ max rel", _delta_value(metric.max_relative)),
            (
                "Cosine",
                "n/a"
                if metric.cosine_similarity is None
                else f"{metric.cosine_similarity:.6g}",
            ),
            (
                "Compared",
                f"{metric.compared_elements:,} element(s)"
                + (" · partial" if metric.partial else ""),
            ),
        )
        return overview.metadata_section(
            lines,
            palette=self.palette,
            title="Delta vs previous trace",
            icon=ft.Icons.COMPARE_ARROWS_ROUNDED,
            role=f"watch-delta:{value_id}",
        )

    # -- jobs --------------------------------------------------------------

    def _start_statistics(self, trace_id: str, value_id: str) -> bool:
        """Queue one statistics job unless the concurrency cap is reached."""
        session = self._session()
        if session is None:
            return False
        loading = self._statistics_loading()
        marker = (trace_id, value_id, "statistics")
        if marker in loading:
            return True
        if (
            sum(1 for entry in loading if entry[2] == "statistics")
            >= self._max_concurrent_jobs
        ):
            return False
        key = (trace_id, value_id)
        loading.add(marker)
        try:
            job = session.compute_activation_statistics_async(trace_id, value_id)
        except Exception as error:
            loading.discard(marker)
            self._statistics_errors[key] = str(error)
            return False

        def wait() -> None:
            job.wait()
            self._statistics_loading().discard(marker)
            if self._session() is session:
                if job.state.value == "succeeded":
                    self._statistics()[key] = job.result()
                    self._statistics_errors.pop(key, None)
                elif job.state.value == "failed":
                    self._statistics_errors[key] = str(job.error)
            self._on_statistics_ready()
            self.page.update()

        self.page.run_thread(wait)
        return True

    def _start_comparison(self, base: TraceResult, active: TraceResult) -> None:
        """Queue one trace-pair comparison; rows index it per value later."""
        session = self._session()
        key = (base.id, active.id)
        if session is None or key in self._comparison_loading:
            return
        self._comparison_loading.add(key)
        try:
            job = session.compare_traces_async(base.id, active.id)
        except Exception as error:
            self._comparison_loading.discard(key)
            self._comparison_errors[key] = str(error)
            return

        def wait() -> None:
            job.wait()
            self._comparison_loading.discard(key)
            if self._session() is session:
                if job.state.value == "succeeded":
                    self._comparisons[key] = job.result()
                    self._comparison_errors.pop(key, None)
                elif job.state.value == "failed":
                    self._comparison_errors[key] = str(job.error)
                    self._on_error(f"Trace comparison failed: {job.error}")
                self.refresh()
            self.page.update()

        self.page.run_thread(wait)

    # -- handlers ----------------------------------------------------------

    def _unpin_handler(
        self, value_id: str
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        def unpin(event: ft.Event[ft.IconButton]) -> None:
            self.unpin(value_id)
            self.page.update()

        return unpin

    def _retry_handler(
        self, trace_id: str, value_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def retry(event: ft.Event[ft.TextButton]) -> None:
            self._statistics_errors.pop((trace_id, value_id), None)
            self.refresh()
            self.page.update()

        return retry
