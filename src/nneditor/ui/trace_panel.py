"""The inference-tracing panel: consent form, run lifecycle, and glyph views.

Everything about *one approved trace* lives here: the limits form the user
approves, the job that runs it, the per-input tensor bindings, the picker
buttons drawn on top of the graph's input glyphs, and the inspector content for
a selected model boundary or dataflow connection.

Like the other extracted panels this one holds no reference to the shell.  The
current session, graph, slice, and surface size arrive as injected accessors,
and everything the shell must redraw afterwards arrives as an injected
callable.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import flet as ft

from nneditor.application.jobs import Job
from nneditor.application.session import ModelSession
from nneditor.application.slices import GraphSlice
from nneditor.ir.capabilities import Availability, Capability
from nneditor.rendering.contract import InteractiveGraphRenderer
from nneditor.tracing.comparison import TraceComparison
from nneditor.tracing.contracts import (
    InputBinding,
    TraceApproval,
    TraceLimits,
    TraceRequest,
    TraceResult,
)
from nneditor.ui import overview, viewmodel
from nneditor.ui.shell_layout import ShellPalette
from nneditor.ui.trace_graph import TraceGraphPresentation

__all__ = ["ActivationRows", "TracePanel", "uses_automatic_mask"]


def uses_automatic_mask(input_name: str) -> bool:
    """True when an input's name marks it as an all-valid attention/pixel mask."""
    normalized = input_name.lower().replace("-", "_")
    return normalized == "mask" or normalized.endswith("_mask")


class ActivationRows(Protocol):
    """The activation inspector's boundary/connection card builder."""

    def __call__(
        self,
        value_ids: tuple[str, ...],
        *,
        owner_id: str,
        title: str,
    ) -> list[ft.Control]: ...


class TracePanel:
    """Own the trace form, the approved run, and the traced-glyph inspector."""

    def __init__(
        self,
        *,
        page: ft.Page,
        picker: ft.FilePicker,
        palette: ShellPalette,
        renderer: InteractiveGraphRenderer,
        session: Callable[[], ModelSession | None],
        current_graph: Callable[[], str | None],
        current_slice: Callable[[], GraphSlice | None],
        surface_size: Callable[[], tuple[float, float]],
        inspected_ids: Callable[[], frozenset[str]],
        activation_rows: ActivationRows,
        set_heading: Callable[[str, str], None],
        refresh_inspector: Callable[[frozenset[str]], None],
        autoload_activation_views: Callable[[frozenset[str]], None],
        redraw_scene: Callable[[], None],
        rebuild_minimap: Callable[[], None],
        on_error: Callable[[str], None],
        on_status: Callable[[str], None],
        clear_error: Callable[[], None],
        watch_text_focus: Callable[[ft.TextField], ft.TextField],
    ) -> None:
        self.page = page
        self.picker = picker
        self.palette = palette
        self.renderer = renderer
        self._session = session
        self._current_graph = current_graph
        self._current_slice = current_slice
        self._surface_size = surface_size
        self._inspected_ids = inspected_ids
        self._activation_rows = activation_rows
        self._set_heading = set_heading
        self._refresh_inspector = refresh_inspector
        self._autoload_activation_views = autoload_activation_views
        self._redraw_scene = redraw_scene
        self._rebuild_minimap = rebuild_minimap
        self._on_error = on_error
        self._on_status = on_status
        self._clear_error = clear_error

        self.job: Job[TraceResult] | None = None
        self.active_trace_id: str | None = None
        self.active_comparison: TraceComparison | None = None
        self.input_bindings: dict[str, InputBinding] = {}
        self.presentation: TraceGraphPresentation | None = None
        self._preparing = False

        self.seed = ft.TextField(
            label="Deterministic seed",
            value="0",
            dense=True,
        )
        self.shapes = ft.TextField(
            label="Symbolic shapes (input=2x3; ...)",
            dense=True,
        )
        self.wall_seconds = ft.TextField(
            label="Wall limit (seconds)",
            value="30",
            dense=True,
        )
        self.memory_mib = ft.TextField(
            label="Memory limit (MiB)",
            value="2048",
            dense=True,
        )
        self.capture_mib = ft.TextField(
            label="Capture limit (MiB)",
            value="256",
            dense=True,
        )
        self.chunk_kib = ft.TextField(
            label="Write chunk (KiB)",
            value="1024",
            dense=True,
        )
        self.approval_notice = ft.Text(
            "Review the selected inputs and limits. The run button approves "
            "exactly one isolated trace.",
            size=10,
            color=palette.muted,
        )
        self.run_button = ft.FilledButton(
            content="Approve & run trace",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            on_click=self.run,
            disabled=True,
        )
        self.cancel_button = ft.TextButton(
            content="Cancel",
            on_click=self.cancel,
            disabled=True,
        )
        self.compare_with = ft.Dropdown(
            label="Compare active trace with",
            dense=True,
            options=[],
            disabled=True,
        )
        self.compare_button = ft.TextButton(
            content="Compare traces",
            icon=ft.Icons.COMPARE_ARROWS_ROUNDED,
            on_click=self.compare,
            disabled=True,
        )
        self.status = ft.Text(
            "Open an artifact to see tracing availability.",
            size=10,
            color=palette.muted,
            expand=True,
        )
        self.progress = ft.ProgressRing(
            width=18,
            height=18,
            stroke_width=2,
            visible=False,
        )
        # One picker button per visible input glyph, positioned over the graph.
        self.graph_actions = ft.Stack(controls=[], expand=True)
        for field in (
            self.seed,
            self.shapes,
            self.wall_seconds,
            self.memory_mib,
            self.capture_mib,
            self.chunk_kib,
        ):
            watch_text_focus(field)
        self.control = ft.Column(
            controls=[
                ft.Row(
                    controls=[self.progress, self.status],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.seed,
                self.shapes,
                ft.Row(
                    controls=[self.wall_seconds, self.memory_mib],
                    spacing=6,
                ),
                ft.Row(
                    controls=[self.capture_mib, self.chunk_kib],
                    spacing=6,
                ),
                self.approval_notice,
                ft.Row(
                    controls=[self.run_button, self.cancel_button],
                    wrap=True,
                    spacing=4,
                ),
                self.compare_with,
                self.compare_button,
            ],
            spacing=8,
        )

    # -- session lifecycle -------------------------------------------------

    def abandon_job(self) -> None:
        """Cancel any in-flight trace and forget it; the session is changing."""
        if self.job is not None and not self.job.state.is_terminal:
            self.job.cancel()
        self.job = None
        self._preparing = False

    def reset(self) -> None:
        """Drop the active trace, its comparison, bindings, and presentation."""
        self.active_trace_id = None
        self.active_comparison = None
        self.input_bindings.clear()
        self.presentation = None

    # -- traced glyph inspector -------------------------------------------

    def glyph_inspector(self, glyph_id: str) -> list[ft.Control] | None:
        """Inspector content for a model boundary or dataflow connection."""
        session = self._session()
        if session is None or self._current_graph() != session.document.entry_graph:
            return None
        scene = self.renderer.scene
        if scene is None:
            return None
        boundary = self.boundary_values()
        is_connection = scene.has_edge(glyph_id)
        if not is_connection and glyph_id not in boundary:
            return None
        value_ids = self.values_for_glyph(glyph_id)
        graph = session.document.main_graph
        rows: list[ft.Control] = []
        if is_connection:
            names = self.value_names(value_ids)
            self._set_heading(
                names[0] if len(names) == 1 else f"{len(names)} connections",
                "Activation on this connection · captured tensor data",
            )
            rows.append(
                overview.section_heading(
                    "Connection values",
                    ft.Icons.CABLE_ROUNDED,
                    trailing=str(len(value_ids)),
                )
            )
        else:
            value_id = boundary[glyph_id]
            value = graph.value(value_id)
            node = scene.node(glyph_id)
            is_input = node.kind == "graph-input"
            self._set_heading(
                value.name or value.id,
                "Model input · choose the tensor used for tracing"
                if is_input
                else "Model output · captured result tensor",
            )
            if is_input:
                rows.extend(self._input_binding_controls(value_id, glyph_id))
        for value_id in value_ids:
            value = graph.value(value_id)
            producer = graph.producer(value_id)
            rows.append(
                overview.metadata_section(
                    (
                        ("Value", value.name or value.id),
                        ("Dtype", value.element_type or "unknown"),
                        ("Shape", str(list(value.shape or ()))),
                        (
                            "Role",
                            "model input"
                            if value_id in graph.inputs
                            and value_id not in graph.initializers
                            else "model output"
                            if value_id in graph.outputs
                            else "intermediate activation",
                        ),
                        (
                            "Producer",
                            graph.node(producer[0]).source_name
                            or graph.node(producer[0]).op_type
                            if producer is not None
                            else "external input",
                        ),
                    ),
                    title="Tensor metadata",
                    icon=ft.Icons.DATA_ARRAY_ROUNDED,
                    role=f"trace-value-metadata:{value_id}",
                )
            )
        rows.extend(
            self._activation_rows(
                value_ids,
                owner_id=glyph_id,
                title=(
                    "Activation on this connection"
                    if is_connection
                    else "Captured tensor"
                ),
            )
        )
        return rows

    def _input_binding_controls(
        self,
        value_id: str,
        glyph_id: str,
    ) -> list[ft.Control]:
        """What the next trace will feed this input, and how to change it."""
        session = self._session()
        assert session is not None
        value = session.document.main_graph.value(value_id)
        input_name = value.name or value.id
        binding = self.input_bindings.get(input_name)
        automatic_mask = binding is None and uses_automatic_mask(input_name)
        chosen = binding is not None or automatic_mask
        source = (
            f".npy file · {Path(binding.tensor_file).name}"
            if binding is not None and binding.tensor_file is not None
            else "All-valid mask · generated automatically"
            if automatic_mask
            else "Deterministic random data"
        )
        return [
            ft.Container(
                content=ft.Text(
                    f"Next trace input: {source}",
                    size=11,
                    color=self.palette.success if chosen else self.palette.muted,
                ),
                padding=10,
                bgcolor=(self.palette.success_soft if chosen else self.palette.subtle),
                border_radius=9,
            ),
            ft.Row(
                controls=[
                    ft.FilledButton(
                        data=f"trace-input-picker:{value_id}",
                        content=(
                            "Replace tensor"
                            if binding is not None
                            else "Choose .npy tensor"
                        ),
                        icon=ft.Icons.UPLOAD_FILE_ROUNDED,
                        on_click=self.choose_tensor_handler(value_id),
                    ),
                    ft.TextButton(
                        content=(
                            "Use automatic mask"
                            if uses_automatic_mask(input_name)
                            else "Use random"
                        ),
                        visible=binding is not None,
                        on_click=self.reset_tensor_handler(input_name, glyph_id),
                    ),
                ],
                wrap=True,
                spacing=4,
            ),
        ]

    # -- trace graph presentation -----------------------------------------

    def boundary_values(self) -> dict[str, str]:
        """Glyph id to value id for every model input and output glyph."""
        presentation = self.presentation
        if presentation is None:
            return {}
        return {
            **{
                glyph_id: value_id
                for value_id, glyph_id in presentation.input_glyphs.items()
            },
            **{
                glyph_id: value_id
                for value_id, glyph_id in presentation.output_glyphs.items()
            },
        }

    def values_for_glyph(self, glyph_id: str) -> tuple[str, ...]:
        """The dataflow values one glyph or connection represents."""
        presentation = self.presentation
        if presentation is None:
            return ()
        return presentation.values_by_glyph.get(glyph_id, ())

    def value_names(self, value_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Display names for values, falling back to their ids."""
        session = self._session()
        if session is None:
            return ()
        graph = session.document.main_graph
        return tuple(
            graph.value(value_id).name or graph.value(value_id).id
            for value_id in value_ids
        )

    def refresh_graph_actions(self) -> None:
        """Position one tensor-picker button inside every visible input glyph."""
        controls: list[ft.Control] = []
        presentation = self.presentation
        viewport = self.renderer.viewport
        scene = self.renderer.scene
        session = self._session()
        if (
            presentation is None
            or viewport is None
            or scene is None
            or session is None
            or bool(getattr(self.page, "web", False))
        ):
            self.graph_actions.controls = controls
            return
        status = session.capability(Capability.TRACING)
        if status.availability is not Availability.AVAILABLE:
            self.graph_actions.controls = controls
            return
        surface_width, surface_height = self._surface_size()
        graph = session.document.main_graph
        for value_id, glyph_id in presentation.input_glyphs.items():
            if not scene.has_node(glyph_id):
                continue
            node = scene.node(glyph_id)
            left, top = viewport.to_screen(node.x, node.y)
            width = node.width * viewport.scale
            height = node.height * viewport.scale
            if (
                left + width < 0
                or top + height < 0
                or left > surface_width
                or top > surface_height
            ):
                continue
            value = graph.value(value_id)
            input_name = value.name or value.id
            selected = self.input_bindings.get(input_name)
            automatic_mask = selected is None and uses_automatic_mask(input_name)
            chosen = selected is not None or automatic_mask
            button = ft.IconButton(
                data=f"trace-input-picker:{value_id}",
                icon=(
                    ft.Icons.CHECK_CIRCLE_ROUNDED
                    if chosen
                    else ft.Icons.UPLOAD_FILE_ROUNDED
                ),
                icon_color=(self.palette.success if chosen else self.palette.accent),
                tooltip=(
                    f"Tensor: {Path(selected.tensor_file).name}"
                    if selected is not None and selected.tensor_file is not None
                    else (
                        f"{input_name} uses an all-valid mask automatically; "
                        "choose a .npy tensor to override it"
                    )
                    if automatic_mask
                    else f"Choose a .npy tensor for {input_name}"
                ),
                on_click=self.choose_tensor_handler(value_id),
            )
            controls.append(
                ft.Container(
                    data=f"trace-input-action:{value_id}",
                    content=button,
                    left=left + max(width - 38.0, 2.0),
                    top=top + max((height - 36.0) / 2.0, 0.0),
                    width=36,
                    height=36,
                    bgcolor="#F2FFFFFF",
                    border_radius=18,
                )
            )
        self.graph_actions.controls = controls

    # -- input bindings ----------------------------------------------------

    def bind_tensor(self, input_name: str, binding: InputBinding) -> None:
        """Record an already-validated tensor for one model input."""
        self.input_bindings[input_name] = binding

    def choose_tensor_handler(self, value_id: str) -> Callable[..., Any]:
        """Pick and validate a .npy tensor for one model input."""

        async def choose(event: ft.Event[ft.IconButton | ft.Button]) -> None:
            session = self._session()
            if session is None:
                return
            graph = session.document.main_graph
            value = graph.value(value_id)
            input_name = value.name or value.id
            files = await self.picker.pick_files(
                dialog_title=f"Choose a .npy tensor for {input_name}",
                allow_multiple=False,
                with_data=False,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["npy"],
            )
            if not files:
                return
            selected = files[0].path
            if selected is None:
                self._on_error("Tensor-file tracing is available in the desktop app.")
                self.page.update()
                return
            try:
                binding = session.trace_tensor_input(input_name, selected)
            except (OSError, TypeError, ValueError) as error:
                self._on_error(f"Cannot use that tensor for {input_name}: {error}")
                self.page.update()
                return
            self.input_bindings[input_name] = binding
            self._clear_error()
            self.refresh_graph_actions()
            self._refresh_inspector(self._inspected_ids())
            self.refresh_actions()
            self._on_status(
                f"{input_name} will use {Path(selected).name} on the next trace"
            )
            self.page.update()

        return choose

    def reset_tensor_handler(
        self,
        input_name: str,
        owner_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        """Return one model input to deterministic random data."""

        def reset(event: ft.Event[ft.TextButton]) -> None:
            self.input_bindings.pop(input_name, None)
            self.refresh_graph_actions()
            self._refresh_inspector(frozenset({owner_id}))
            self.refresh_actions()
            self._on_status(
                f"{input_name} will use deterministic random data on the next trace"
            )
            self.page.update()

        return reset

    # -- run lifecycle -----------------------------------------------------

    def refresh_actions(self) -> None:
        """Reflect availability, the running job, and comparable traces."""
        session = self._session()
        running = self.job is not None and not self.job.state.is_terminal
        busy = self._preparing or running
        web = bool(getattr(self.page, "web", False))
        available = False
        reason = "Open an artifact to see tracing availability."
        if session is not None:
            status = session.capability(Capability.TRACING)
            self.approval_notice.value = (
                f"Selecting Approve & run authorizes one isolated trace of "
                f"{session.title} using the graph inputs and four limits shown."
            )
            available = status.availability is Availability.AVAILABLE and not web
            reason = (
                "Web tracing awaits the Phase 8 isolated worker service."
                if web and status.availability is Availability.AVAILABLE
                else status.reason
            )
        else:
            self.approval_notice.value = (
                "Review the selected inputs and limits. The run button approves "
                "exactly one isolated trace."
            )
        self.progress.visible = busy
        self.run_button.disabled = not available or busy
        self.cancel_button.disabled = not running
        active_result = (
            session.trace(self.active_trace_id)
            if session is not None and self.active_trace_id is not None
            else None
        )
        if self.active_comparison is not None:
            self.status.value = (
                f"Compared {len(self.active_comparison.nodes)} node(s); "
                "overlay shows maximum absolute error"
            )
        elif self.active_trace_id is None:
            self.status.value = (
                f"{reason} Choose .npy tensors with the buttons on model inputs, "
                "or keep automatic all-valid masks and deterministic random data."
            )
        elif session is not None and active_result is not None:
            state = "Partial" if active_result.partial else "Complete"
            revision = active_result.key.revision_id or "base"
            stale = (
                " • not current revision"
                if active_result.key.revision_id != session.editing.current_revision_id
                else ""
            )
            self.status.value = (
                f"{state} trace • revision {revision[:18]}{stale} • "
                f"{len(active_result.captured_value_ids)} readable value(s) • "
                f"{active_result.runtime}"
            )
        traces = session.traces() if session is not None else ()
        alternatives = [
            result
            for result in traces
            if result.id != self.active_trace_id
            and active_result is not None
            and result.key.input_specification_hash
            == active_result.key.input_specification_hash
        ]
        self.compare_with.options = [
            ft.DropdownOption(
                key=result.id,
                text=(
                    "base"
                    if result.key.revision_id is None
                    else result.key.revision_id[:18]
                )
                + (" • partial" if result.partial else " • complete"),
            )
            for result in alternatives
        ]
        valid_ids = {result.id for result in alternatives}
        if self.compare_with.value not in valid_ids:
            self.compare_with.value = alternatives[-1].id if alternatives else None
        self.compare_with.disabled = not alternatives or busy
        self.compare_button.disabled = (
            self.active_trace_id is None or not alternatives or busy
        )

    def run(self, event: ft.Event[ft.Button]) -> None:
        """Approve and start exactly one isolated trace of the open model."""
        session = self._session()
        if session is None or self._preparing:
            return
        if self.job is not None and not self.job.state.is_terminal:
            return
        if getattr(self.page, "web", False):
            self._on_error(
                "Web tracing is unavailable until the Phase 8 isolated worker "
                "service is deployed."
            )
            self.page.update()
            return
        seed_text = self.seed.value or ""
        shapes_text = self.shapes.value or ""
        wall_text = self.wall_seconds.value or ""
        memory_text = self.memory_mib.value or ""
        capture_text = self.capture_mib.value or ""
        chunk_text = self.chunk_kib.value or ""
        bindings = dict(self.input_bindings)
        self._preparing = True
        self._clear_error()
        self.active_comparison = None
        self.refresh_actions()
        self.status.value = f"Preparing approved trace for {session.title}…"
        self._on_status("Preparing inputs and isolated trace worker…")
        self.page.update()

        def prepare() -> None:
            try:
                seed = int(seed_text)
                shapes = viewmodel.parse_shape_overrides(shapes_text)
                specification = session.default_trace_inputs(
                    seed=seed,
                    shapes=shapes,
                    bindings=bindings,
                )
                limits = TraceLimits(
                    wall_seconds=float(wall_text),
                    memory_bytes=int(memory_text) * 1024 * 1024,
                    capture_bytes=int(capture_text) * 1024 * 1024,
                    chunk_bytes=int(chunk_text) * 1024,
                )
                approval = TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    limits,
                )
                request = TraceRequest(specification, limits, approval)
                job = session.trace_async(request)
            except (OSError, TypeError, ValueError) as error:
                if self._session() is session:
                    self._preparing = False
                    self.refresh_actions()
                    self._on_error(f"Trace configuration is invalid: {error}")
                    self.page.update()
                return
            if self._session() is not session:
                if not job.state.is_terminal:
                    job.cancel()
                return
            self._preparing = False
            self.job = job
            self.refresh_actions()
            self.status.value = (
                f"Loading trace for {session.title} • inputs "
                f"{specification.hash[:19]} • {limits.wall_seconds:g}s / "
                f"{limits.memory_bytes // (1024 * 1024)} MiB / "
                f"{limits.capture_bytes // (1024 * 1024)} MiB capture"
            )
            self._on_status("Running approved inference trace in an isolated worker…")
            self.page.update()
            self._watch(job)

        self.page.run_thread(prepare)

    def _watch(self, job: Job[TraceResult]) -> None:
        def wait() -> None:
            try:
                job.wait()
                if job.state.value == "succeeded":
                    result = job.result()
                    self.active_trace_id = result.id
                    self.active_comparison = None
                    self._clear_error()
                    state = "partial" if result.partial else "complete"
                    self._on_status(
                        f"Trace {state}: {len(result.captured_value_ids)} "
                        "readable activation(s)"
                    )
                    if self._current_slice() is not None:
                        self._redraw_scene()
                        self.refresh_graph_actions()
                        self._rebuild_minimap()
                    self._autoload_activation_views(self._inspected_ids())
                    self._refresh_inspector(self._inspected_ids())
                elif job.state.value == "cancelled":
                    self._on_status("Trace cancelled; no partial trace was saved")
                else:
                    self._on_error(f"Trace failed with no saved state: {job.error}")
            except Exception as error:
                self._on_error(f"Could not present the trace: {error}")
            finally:
                if self.job is job:
                    self.job = None
                self.refresh_actions()
                self.page.update()

        self.page.run_thread(wait)

    def cancel(self, event: ft.Event[ft.TextButton]) -> None:
        """Stop the running trace at the next worker checkpoint."""
        if self.job is None or self.job.state.is_terminal:
            return
        self.job.cancel()
        self.cancel_button.disabled = True
        self.status.value = "Cancelling and discarding staged captures…"
        self._on_status("Cancelling trace at the next worker checkpoint…")
        self.page.update()

    def compare(self, event: ft.Event[ft.TextButton]) -> None:
        """Overlay the per-node error between the active trace and another."""
        session = self._session()
        if (
            session is None
            or self.active_trace_id is None
            or self.compare_with.value is None
        ):
            return
        try:
            job = session.compare_traces_async(
                self.compare_with.value,
                self.active_trace_id,
            )
        except ValueError as error:
            self._on_error(f"Cannot compare traces: {error}")
            self.page.update()
            return
        self.status.value = "Loading per-node trace comparison…"
        self.compare_button.disabled = True

        def wait() -> None:
            job.wait()
            if job.state.value == "succeeded":
                self.active_comparison = job.result()
                if self._current_slice() is not None:
                    self._redraw_scene()
                    self.refresh_graph_actions()
                self.status.value = (
                    f"Compared {len(self.active_comparison.nodes)} node(s); "
                    "overlay shows maximum absolute error"
                )
                self._on_status("Trace comparison overlay active")
            elif job.state.value == "cancelled":
                self._on_status("Trace comparison cancelled")
            else:
                self._on_error(f"Trace comparison failed: {job.error}")
            self.refresh_actions()
            self.page.update()

        self.page.run_thread(wait)
