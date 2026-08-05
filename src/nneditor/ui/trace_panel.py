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

import math
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
    CaptureState,
    CaptureStatus,
    InputBinding,
    TraceApproval,
    TraceBackend,
    TraceDevice,
    TraceLimits,
    TraceRequest,
    TraceResult,
    declared_byte_size,
)
from nneditor.tracing.preflight import RuntimeStatus, runtime_status
from nneditor.tracing.runner import estimated_capture_bytes
from nneditor.ui import overview, viewmodel
from nneditor.ui.shell_layout import ShellPalette
from nneditor.ui.trace_graph import TraceGraphPresentation

__all__ = ["ActivationRows", "TracePanel", "uses_automatic_mask"]

_MIB = 1024 * 1024
# One GiB of estimated decoded activations (1024 MiB). Above this, capturing
# everything is a heavyweight choice a user should make knowingly — a 1.33 GB,
# 2,199-node model decodes to roughly 8 GiB of activations — so the panel
# defaults the scope to previewing everything and leaves the greedy
# whole-value capture an explicit choice.
_FULL_CAPTURE_ADVISORY_BYTES = 1024 * _MIB

# The count trigger for the same smart preview default. The byte advisory
# alone is blind on real models: ONNX intermediates rarely declare shapes,
# so a 2,199-node model whose activations actually total ~13 GB estimated
# only 6.9 MiB of declared values, kept the greedy default, and dropped
# 2,155 of its 2,248 values. The capturable-value count is always known,
# so past this many values the session defaults to the preview scope.
_PREVIEW_DEFAULT_VALUE_COUNT = 256

# Capture-scope choices offered whenever a session is open.
# "Preview everything" trades depth for breadth: it captures the full value
# set under the worker's equal-share "preview" policy, so every value keeps
# roughly capture_bytes / N bytes and any node can be inspected after one
# trace. A later narrowed whole-value re-trace shares the same trace id and
# upgrades the previews in place.
_SCOPE_BOUNDARIES = "boundaries-plus-selection"
_SCOPE_EVERYTHING = "everything"
_SCOPE_PREVIEW = "preview-everything"

# Fallback for the per-value preview estimate while the capture-limit field
# holds unparseable text; matches the field's initial value of 256 MiB.
_DEFAULT_CAPTURE_LIMIT_BYTES = 256 * _MIB

# The finished-trace status line's leading word. "Partial" is reserved for
# missing requested data; a trace whose every value kept a readable (if
# truncated) share is a preview, and a whole capture is complete.
_STATUS_WORDS = {
    CaptureStatus.PARTIAL: "Partial",
    CaptureStatus.PREVIEW: "Preview",
    CaptureStatus.COMPLETE: "Complete",
}


def _plural(count: int, noun: str) -> str:
    """``3 values`` / ``1 value``: counts read as prose in the scope summary."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def uses_automatic_mask(input_name: str) -> bool:
    """True when an input's name marks it as an all-valid attention/pixel mask."""
    normalized = input_name.lower().replace("-", "_")
    return normalized == "mask" or normalized.endswith("_mask")


def _budget_dropped_majority(result: TraceResult) -> bool:
    """True when the capture budget dropped most of the requested values.

    A greedy trace of a large model reads as failure — "93 readable
    value(s)" against thousands requested — when the pool simply ran out,
    so the status owes the user the way out (the preview scope).
    """
    starved = sum(
        1
        for record in result.records
        if record.state is CaptureState.EVICTED
        or (
            record.state is CaptureState.DROPPED
            # The worker's reason for a budget drop, as opposed to a value
            # some backend failed to produce at all.
            and "capture byte ceiling" in (record.reason or "")
        )
    )
    return 2 * starved > len(result.records)


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
        selected_node_ids: Callable[[], frozenset[str]],
        activation_rows: ActivationRows,
        set_heading: Callable[[str, str], None],
        refresh_inspector: Callable[[frozenset[str]], None],
        autoload_activation_views: Callable[[frozenset[str]], None],
        redraw_scene: Callable[[], None],
        rebuild_minimap: Callable[[], None],
        on_error: Callable[[str], None],
        on_status: Callable[[str], None],
        on_device: Callable[[TraceDevice | None, str], None],
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
        self._selected_node_ids = selected_node_ids
        self._activation_rows = activation_rows
        self._set_heading = set_heading
        self._refresh_inspector = refresh_inspector
        self._autoload_activation_views = autoload_activation_views
        self._redraw_scene = redraw_scene
        self._rebuild_minimap = rebuild_minimap
        self._on_error = on_error
        self._on_status = on_status
        self._on_device = on_device
        self._clear_error = clear_error

        self.job: Job[TraceResult] | None = None
        self.active_trace_id: str | None = None
        self.active_comparison: TraceComparison | None = None
        self.input_bindings: dict[str, InputBinding] = {}
        self.presentation: TraceGraphPresentation | None = None
        self._preparing = False
        # The user's explicit capture-scope choice for this session; None means
        # the smart default (everything, or preview-everything once the
        # capturable-value count or the declared-byte estimate trips its
        # threshold) applies.
        self._scope_choice: str | None = None
        self._runtime_probe_started = False
        self._probing_runtime = False

        self.seed = ft.TextField(
            label="Deterministic seed",
            value="0",
            dense=True,
        )
        self.shapes = ft.TextField(
            label="Symbolic shapes (input=2x3; ...)",
            dense=True,
        )
        self.backend = ft.Dropdown(
            label="Execution backend",
            value=TraceBackend.AUTO.value,
            dense=True,
            expand=True,
            on_select=self._on_backend_changed,
            options=[
                ft.DropdownOption(
                    key=TraceBackend.AUTO.value,
                    text="Automatic (ONNX Runtime, then reference)",
                ),
                ft.DropdownOption(
                    key=TraceBackend.ONNX_RUNTIME.value,
                    text="ONNX Runtime",
                ),
                ft.DropdownOption(
                    key=TraceBackend.REFERENCE.value,
                    text="Axis-aware reference evaluator",
                ),
                ft.DropdownOption(
                    key=TraceBackend.REFERENCE_NORMALIZED.value,
                    text="Normalized reference evaluator",
                ),
            ],
        )
        self.device = ft.Dropdown(
            label="Execution device",
            value=TraceDevice.AUTO.value,
            dense=True,
            expand=True,
            options=[
                ft.DropdownOption(
                    key=TraceDevice.AUTO.value,
                    text="Automatic (accelerator, then CPU)",
                ),
                ft.DropdownOption(key=TraceDevice.CPU.value, text="CPU"),
                ft.DropdownOption(
                    key=TraceDevice.GPU.value,
                    text="GPU (CUDA, DirectML, OpenVINO)",
                ),
                ft.DropdownOption(
                    key=TraceDevice.NPU.value,
                    text="NPU (OpenVINO, QNN, Vitis AI)",
                ),
            ],
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
        self.capture_selected_only = ft.Checkbox(
            label="Capture only the selected nodes",
            value=False,
            on_change=self._on_capture_scope_changed,
        )
        # Shown whenever a session is open: the explicit widen/narrow choice
        # between boundaries + selection, previewing, and everything.
        self.capture_scope_choice = ft.Dropdown(
            label="Capture scope",
            value=_SCOPE_EVERYTHING,
            dense=True,
            visible=False,
            on_select=self._on_capture_choice_changed,
            options=[
                ft.DropdownOption(
                    key=_SCOPE_BOUNDARIES,
                    text="Boundaries + selection",
                ),
                ft.DropdownOption(key=_SCOPE_PREVIEW, text="Preview everything"),
                ft.DropdownOption(key=_SCOPE_EVERYTHING, text="Everything"),
            ],
        )
        self.capture_estimate = ft.Text(
            "",
            size=10,
            color=palette.muted,
            visible=False,
        )
        self.capture_scope = ft.Text(
            "Open an artifact to see how many values a trace captures.",
            size=10,
            color=palette.muted,
        )
        self.runtime_text = ft.Text(
            "Open an artifact to probe the ONNX Runtime.",
            size=10,
            color=palette.muted,
            expand=True,
        )
        self.runtime_refresh = ft.IconButton(
            icon=ft.Icons.REFRESH_ROUNDED,
            icon_size=14,
            tooltip="Probe the ONNX Runtime again",
            on_click=self._on_refresh_runtime,
        )
        # A finished trace's informational notes and diagnostics, rendered as
        # separately styled rows under the status line.
        self.result_annotations = ft.Column(controls=[], spacing=4, visible=False)
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
                self.result_annotations,
                self.seed,
                self.shapes,
                ft.Row(
                    controls=[self.backend, self.device],
                    spacing=6,
                ),
                ft.Row(
                    controls=[self.runtime_text, self.runtime_refresh],
                    spacing=2,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    controls=[self.wall_seconds, self.memory_mib],
                    spacing=6,
                ),
                ft.Row(
                    controls=[self.capture_mib, self.chunk_kib],
                    spacing=6,
                ),
                self.capture_estimate,
                self.capture_scope_choice,
                self.capture_selected_only,
                self.capture_scope,
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
        # The capture-scope choice persists for one session only; the next
        # model starts from the smart default again.
        self._scope_choice = None
        if self._session() is not None and not self._runtime_probe_started:
            self._runtime_probe_started = True
            self._start_runtime_probe()
        self._on_device(None, "Idle")

    # -- runtime preflight -------------------------------------------------

    def _start_runtime_probe(self, *, refresh: bool = False) -> None:
        """Ask a throwaway interpreter what ONNX Runtime a trace would find.

        The probe never runs on the UI thread and never imports the runtime
        into this process; the panel shows a quiet placeholder until the
        answer lands, and the device dropdown keeps working throughout.
        """
        if self._probing_runtime:
            return
        self._probing_runtime = True
        self.runtime_text.value = "Probing ONNX Runtime…"
        self.runtime_text.color = self.palette.muted

        def probe() -> None:
            try:
                status = runtime_status(refresh=refresh)
            finally:
                self._probing_runtime = False
            self._render_runtime_status(status)
            self.page.update()

        self.page.run_thread(probe)

    def _render_runtime_status(self, status: RuntimeStatus) -> None:
        """Show the probed providers, or the error naming nneditor[runtime]."""
        if status.error is not None:
            self.runtime_text.value = f"ONNX Runtime unavailable: {status.error}."
            self.runtime_text.color = self.palette.warning
        elif status.available:
            version = f" {status.version}" if status.version else ""
            self.runtime_text.value = (
                f"ONNX Runtime{version} providers: {', '.join(status.available)}."
            )
            self.runtime_text.color = self.palette.muted
        else:
            self.runtime_text.value = "ONNX Runtime reported no execution providers."
            self.runtime_text.color = self.palette.warning

    def _on_refresh_runtime(self, event: ft.Event[ft.IconButton]) -> None:
        self._start_runtime_probe(refresh=True)
        self.page.update()

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
                    palette=self.palette,
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
                    palette=self.palette,
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

    # -- capture scope -----------------------------------------------------

    def _capturable_value_ids(self) -> frozenset[str]:
        """Every value an unrestricted trace would capture.

        This mirrors the runner's own target set: model inputs that are not
        initializers plus every node output, minus the unnamed positional
        placeholders that stand in for omitted optional ports.
        """
        session = self._session()
        if session is None:
            return frozenset()
        graph = session.document.main_graph
        candidates = {
            value_id for value_id in graph.inputs if value_id not in graph.initializers
        }
        for node in graph.nodes:
            candidates.update(node.outputs)
        return frozenset(
            value_id for value_id in candidates if graph.value(value_id).name
        )

    def _selected_graph_node_ids(self) -> frozenset[str]:
        """Selected glyphs that are operators of the traceable graph."""
        session = self._session()
        if session is None:
            return frozenset()
        selected = self._selected_node_ids()
        return frozenset(
            node.id for node in session.document.main_graph.nodes if node.id in selected
        )

    def _selected_capture_value_ids(self) -> frozenset[str]:
        """Named output values produced by the currently selected operators.

        The runner refuses a selected value carrying no serialized name, so the
        placeholders for omitted optional outputs are dropped here rather than
        failing a trace the user already approved.
        """
        session = self._session()
        if session is None:
            return frozenset()
        graph = session.document.main_graph
        selected = self._selected_graph_node_ids()
        return frozenset(
            value_id
            for node_id in selected
            for value_id in graph.node(node_id).outputs
            if graph.value(value_id).name
        )

    def _boundary_capture_value_ids(self) -> frozenset[str]:
        """Named model inputs and outputs: the always-informative boundary."""
        session = self._session()
        if session is None:
            return frozenset()
        graph = session.document.main_graph
        candidates = {
            value_id for value_id in graph.inputs if value_id not in graph.initializers
        }
        candidates.update(graph.outputs)
        return frozenset(
            value_id for value_id in candidates if graph.value(value_id).name
        )

    def _estimated_bytes(self, value_ids: frozenset[str]) -> int:
        """Lower-bound decoded bytes of capturing these declared values."""
        session = self._session()
        if session is None:
            return 0
        graph = session.document.main_graph
        return estimated_capture_bytes(graph.value(value_id) for value_id in value_ids)

    def _declared_shape_count(self, value_ids: frozenset[str]) -> int:
        """How many of these values declare a sizeable shape and dtype."""
        session = self._session()
        if session is None:
            return 0
        graph = session.document.main_graph
        return sum(
            1
            for value_id in value_ids
            if declared_byte_size(
                graph.value(value_id).element_type,
                graph.value(value_id).shape,
            )
            is not None
        )

    def _estimate_label(self, value_ids: frozenset[str]) -> str:
        """These values' decoded-bytes estimate, honest about missing shapes.

        ``estimated_capture_bytes`` sums only values with declared shapes and
        fixed-width dtypes, so when some values declare neither the number is
        a lower bound (``≥``) and the label says how many values it covers
        rather than presenting the partial sum as the total.
        """
        estimate = viewmodel.compact_bytes(self._estimated_bytes(value_ids))
        declared = self._declared_shape_count(value_ids)
        total = len(value_ids)
        if declared == total:
            return f"~{estimate}"
        return f"≥{estimate}, {declared:,} of {total:,} values declare shapes"

    def _capture_limit_bytes(self) -> int:
        """The capture byte pool the limits form currently names."""
        try:
            return int(self.capture_mib.value or "") * _MIB
        except ValueError:
            return _DEFAULT_CAPTURE_LIMIT_BYTES

    def _effective_scope(self) -> str:
        """The active scope once the smart default and the user's explicit
        session choice are reconciled; the explicit choice always wins.

        The smart default turns to previewing on either trigger: the
        always-known capturable-value count, or the declared-byte estimate —
        a lower bound that a model with undeclared intermediate shapes never
        trips, which is exactly why the count trigger exists.
        """
        if self._scope_choice is not None:
            return self._scope_choice
        capturable = self._capturable_value_ids()
        if len(capturable) > _PREVIEW_DEFAULT_VALUE_COUNT:
            return _SCOPE_PREVIEW
        if self._estimated_bytes(capturable) > _FULL_CAPTURE_ADVISORY_BYTES:
            return _SCOPE_PREVIEW
        return _SCOPE_EVERYTHING

    def capture_value_ids(self) -> frozenset[str]:
        """The values the next trace captures; empty means capture everything."""
        if self.capture_selected_only.value:
            return self._selected_capture_value_ids()
        if self._effective_scope() == _SCOPE_BOUNDARIES:
            return (
                self._boundary_capture_value_ids() | self._selected_capture_value_ids()
            )
        return frozenset()

    def capture_policy(self) -> str:
        """How the next trace spends its capture pool: whole values, or an
        equal preview share for every value under the preview scope."""
        if self.capture_selected_only.value:
            return "greedy"
        return "preview" if self._effective_scope() == _SCOPE_PREVIEW else "greedy"

    def refresh_capture_scope(self) -> None:
        """Restate how much of the model the next approved trace would read."""
        session = self._session()
        if session is None:
            self.capture_scope.value = (
                "Open an artifact to see how many values a trace captures."
            )
            self.capture_scope_choice.visible = False
            self.capture_estimate.visible = False
            return
        all_ids = self._capturable_value_ids()
        full_estimate = self._estimated_bytes(all_ids)
        # The full-capture size keeps its lower-bound marker in the option
        # row: values without declared shapes contribute nothing to it.
        full_bound = "~" if self._declared_shape_count(all_ids) == len(all_ids) else "≥"
        preview_share = self._capture_limit_bytes() // max(1, len(all_ids))
        self.capture_scope_choice.options = [
            ft.DropdownOption(
                key=_SCOPE_BOUNDARIES,
                text="Boundaries + selection",
            ),
            ft.DropdownOption(
                key=_SCOPE_PREVIEW,
                text=(
                    f"Preview everything ({_plural(len(all_ids), 'value')}, "
                    f"~{viewmodel.compact_bytes(preview_share)} each)"
                ),
            ),
            ft.DropdownOption(
                key=_SCOPE_EVERYTHING,
                text=(
                    f"Everything ({_plural(len(all_ids), 'value')}, "
                    f"{full_bound}{viewmodel.compact_bytes(full_estimate)})"
                ),
            ),
        ]
        # Always offered while a session is open: the byte advisory cannot
        # see undeclared shapes, so the scope control must not hide behind it.
        self.capture_scope_choice.visible = True
        self.capture_scope_choice.value = self._effective_scope()
        scoped_ids = self.capture_value_ids() or all_ids
        estimate = self._estimated_bytes(scoped_ids)
        estimate_label = self._estimate_label(scoped_ids)
        suffix = f" ({estimate_label})"
        everything = f"all {_plural(len(all_ids), 'value')}"
        if self.capture_selected_only.value:
            nodes = len(self._selected_graph_node_ids())
            values = len(self._selected_capture_value_ids())
            if not nodes:
                self.capture_scope.value = (
                    f"No nodes selected; capturing {everything}{suffix}."
                )
            elif not values:
                self.capture_scope.value = (
                    f"No named values in the {_plural(nodes, 'selected node')}; "
                    f"capturing {everything}{suffix}."
                )
            else:
                self.capture_scope.value = (
                    f"Capturing {_plural(values, 'value')} from "
                    f"{_plural(nodes, 'selected node')}{suffix}."
                )
        elif self._effective_scope() == _SCOPE_BOUNDARIES:
            nodes = len(self._selected_graph_node_ids())
            detail = (
                f"model inputs and outputs plus {_plural(nodes, 'selected node')}"
                if nodes
                else "model inputs and outputs"
            )
            self.capture_scope.value = (
                f"Capturing boundaries + selection: "
                f"{_plural(len(scoped_ids), 'value')} ({detail}){suffix}."
            )
        elif self._effective_scope() == _SCOPE_PREVIEW:
            self.capture_scope.value = (
                f"Preview everything: capturing {everything} at "
                f"~{viewmodel.compact_bytes(preview_share)} per value."
            )
            # A preview stores at most the capture pool, so the inline
            # estimate — like the runner's pressure note — is capped by it,
            # and that pool ÷ count math needs no declared shapes.
            estimate = min(estimate, self._capture_limit_bytes())
            estimate_label = f"~{viewmodel.compact_bytes(estimate)}"
        else:
            self.capture_scope.value = f"Capturing {everything}{suffix}."
        self._refresh_capture_estimate(estimate, estimate_label)

    def _refresh_capture_estimate(self, estimate: int, label: str) -> None:
        """Show the scope's decoded size near the limits, and warn with the
        runner's own note wording before the run when the estimate exceeds
        half the approved memory limit."""
        self.capture_estimate.visible = True
        try:
            memory_bytes = int(self.memory_mib.value or "") * _MIB
        except ValueError:
            memory_bytes = None
        if memory_bytes is not None and estimate > memory_bytes // 2:
            estimated_mib = math.ceil(estimate / _MIB)
            limit_mib = math.ceil(memory_bytes / _MIB)
            self.capture_estimate.value = (
                f"The selected capture decodes to an estimated "
                f"{estimated_mib:,} MiB of activations, more than half the "
                f"approved {limit_mib:,} MiB memory limit; a full capture may "
                "exhaust it — capture fewer values or raise the Memory limit."
            )
            self.capture_estimate.color = self.palette.warning
        else:
            self.capture_estimate.value = (
                f"Estimated decoded activations for this scope: {label}."
            )
            self.capture_estimate.color = self.palette.muted

    def _on_capture_scope_changed(self, event: ft.Event[ft.Checkbox]) -> None:
        self.refresh_capture_scope()
        self.page.update()

    def _on_capture_choice_changed(self, event: ft.Event[ft.Dropdown]) -> None:
        choice = self.capture_scope_choice.value
        if choice in {_SCOPE_BOUNDARIES, _SCOPE_EVERYTHING, _SCOPE_PREVIEW}:
            self._scope_choice = choice
        self.refresh_capture_scope()
        self.page.update()

    def _on_backend_changed(self, event: ft.Event[ft.Dropdown]) -> None:
        if self.backend.value in {
            TraceBackend.REFERENCE.value,
            TraceBackend.REFERENCE_NORMALIZED.value,
        }:
            self.device.value = TraceDevice.CPU.value
        self.refresh_actions()
        self.page.update()

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
                f"{session.title} using the backend, device, graph inputs, and "
                "four limits shown. The memory limit covers host RAM; "
                "accelerator memory is controlled by its driver."
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
        self._render_result_annotations(active_result)
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
            # Three-way wording: "Partial" only when requested data is
            # missing; a truncated-but-all-readable capture reads "Preview".
            state = _STATUS_WORDS[active_result.capture_status]
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
            if _budget_dropped_majority(active_result):
                self.status.value += (
                    " • most values were not captured — switch Capture scope "
                    "to Preview everything to cover every node"
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
        self.capture_selected_only.disabled = busy
        self.capture_scope_choice.disabled = busy
        self.backend.disabled = busy
        reference_backend = self.backend.value in {
            TraceBackend.REFERENCE.value,
            TraceBackend.REFERENCE_NORMALIZED.value,
        }
        self.device.disabled = busy or reference_backend
        self.refresh_capture_scope()

    def _render_result_annotations(self, result: TraceResult | None) -> None:
        """Render a finished trace's diagnostics and notes as separate rows.

        Diagnostics describe defects in the captured records and keep their
        warning styling; notes are advisory provenance for a healthy run
        (skipped providers, capture pressure) and render in the info palette,
        so a complete trace with notes never reads as partial.
        """
        rows: list[ft.Control] = []
        if result is not None:
            for index, message in enumerate(result.diagnostics):
                rows.append(
                    ft.Container(
                        data=f"trace-diagnostic:{index}",
                        content=ft.Text(
                            message,
                            size=10,
                            color=self.palette.warning,
                        ),
                        padding=8,
                        bgcolor=self.palette.warning_soft,
                        border_radius=8,
                    )
                )
            for index, message in enumerate(result.notes):
                rows.append(
                    ft.Container(
                        data=f"trace-note:{index}",
                        content=ft.Text(
                            f"Note: {message}",
                            size=10,
                            color=self.palette.info,
                        ),
                        padding=8,
                        bgcolor=self.palette.info_soft,
                        border_radius=8,
                    )
                )
        self.result_annotations.controls = rows
        self.result_annotations.visible = bool(rows)

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
        backend_text = self.backend.value or TraceBackend.AUTO.value
        device_text = self.device.value or TraceDevice.AUTO.value
        if backend_text in {
            TraceBackend.REFERENCE.value,
            TraceBackend.REFERENCE_NORMALIZED.value,
        }:
            device_text = TraceDevice.CPU.value
        bindings = dict(self.input_bindings)
        value_ids = self.capture_value_ids()
        capture_policy = self.capture_policy()
        self._preparing = True
        self._clear_error()
        self.active_comparison = None
        requested_device = TraceDevice(device_text)
        self._on_device(requested_device, "Selecting provider")
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
                backend = TraceBackend(backend_text)
                device = TraceDevice(device_text)
                approval = TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    limits,
                    backend,
                    device,
                )
                request = TraceRequest(
                    specification,
                    limits,
                    approval,
                    value_ids=value_ids,
                    backend=backend,
                    device=device,
                    capture_policy=capture_policy,
                )
                job = session.trace_async(request)
            except (OSError, TypeError, ValueError) as error:
                if self._session() is session:
                    self._preparing = False
                    self.refresh_actions()
                    self._on_device(None, "Idle")
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
                f"{limits.capture_bytes // (1024 * 1024)} MiB capture • "
                f"{backend.value} / {device.value}"
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
                    self._on_device(
                        result.execution_device,
                        result.execution_provider,
                    )
                    self._clear_error()
                    self._on_status(
                        f"Trace {result.capture_status.value}: "
                        f"{len(result.captured_value_ids)} readable activation(s)"
                    )
                    if self._current_slice() is not None:
                        self._redraw_scene()
                        self.refresh_graph_actions()
                        self._rebuild_minimap()
                    self._autoload_activation_views(self._inspected_ids())
                    self._refresh_inspector(self._inspected_ids())
                elif job.state.value == "cancelled":
                    self._on_device(None, "Idle")
                    self._on_status("Trace cancelled; no partial trace was saved")
                else:
                    self._on_device(None, "Unavailable")
                    self._on_error(f"Trace failed with no saved state: {job.error}")
            except Exception as error:
                self._on_device(None, "Unavailable")
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
