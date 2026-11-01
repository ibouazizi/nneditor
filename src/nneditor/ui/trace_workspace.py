"""The model graph, with inspectable activation ports, in its own workspace."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ModelSession
from nneditor.application.slices import GraphSlice
from nneditor.rendering.contract import RendererFactory
from nneditor.rendering.scene import Scene, Viewport
from nneditor.rendering.spatial import SpatialIndex, scene_indexes
from nneditor.tracing.contracts import ActivationRecord, CaptureState
from nneditor.ui.shell_layout import ShellPalette
from nneditor.ui.trace_graph import build_trace_graph
from nneditor.ui.trace_ports import TracePort, graph_ports, port_label

_TENSOR_PAGE = 40
_PORT_LIMIT = 400


class TraceWorkspace:
    def __init__(
        self,
        *,
        page: ft.Page,
        palette: ShellPalette,
        session: Callable[[], ModelSession | None],
        active_trace: Callable[[], str | None],
        open_tensor: Callable[..., None],
        configure_trace: Callable[[], None],
        select_trace: Callable[[str], None],
        graph_slice: Callable[[], GraphSlice | None],
        graph_id: Callable[[], str | None],
        show_group: Callable[[str], None],
        show_parent: Callable[[], None],
        set_detail: Callable[[DetailLevel], None],
        renderer_factory: RendererFactory,
        watch_text_focus: Callable[[ft.TextField], ft.TextField],
    ) -> None:
        self.page, self.palette, self._session = page, palette, session
        self._active_trace, self._open_tensor = active_trace, open_tensor
        self._configure_trace, self._select_trace = configure_trace, select_trace
        self._graph_slice, self._graph_id = graph_slice, graph_id
        self._show_group, self._show_parent, self._set_detail = (
            show_group,
            show_parent,
            set_detail,
        )
        self._key: object = None
        self._records: dict[str, ActivationRecord] = {}
        self._ports: dict[str, tuple[TracePort, ...]] = {}
        self._index: SpatialIndex | None = None
        self._size = (1100.0, 550.0)
        self._measured = False
        self._view = Viewport(0, 0, *self._size)
        self._page, self._find_index = 0, -1
        self._chooser: ft.AlertDialog | None = None
        self.renderer = renderer_factory(self._on_selected)
        detector = self.renderer.control
        if not isinstance(detector, ft.GestureDetector):
            raise TypeError("the traced graph needs a Flet gesture surface")
        detector.on_pan_update, detector.on_scroll = self._on_pan, self._on_scroll
        self.port_layer = ft.Stack(expand=True, clip_behavior=ft.ClipBehavior.NONE)
        self.canvas = ft.Container(
            expand=True,
            content=ft.Stack(controls=[detector, self.port_layer], expand=True),
            on_size_change=self._on_size,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self.tensor_list = ft.ListView(expand=True, spacing=6)
        self.body = ft.Container(expand=True, content=self.canvas)
        self.trace_picker = ft.Dropdown(
            label="Captured trace", width=320, dense=True, on_select=self._on_trace
        )
        self.mode = ft.Dropdown(
            label="Show",
            width=180,
            dense=True,
            value="graph",
            options=[
                ft.DropdownOption("graph", text="Model graph"),
                ft.DropdownOption("tensors", text="Captured tensors"),
            ],
            on_select=self._on_mode,
        )
        self.detail = ft.Dropdown(
            label="Graph detail",
            width=160,
            dense=True,
            options=[
                ft.DropdownOption(d.value, text=d.value.title()) for d in DetailLevel
            ],
            on_select=self._on_detail,
        )
        self.search = ft.TextField(
            label="Find a block or tensor",
            dense=True,
            expand=True,
            prefix_icon=ft.Icons.SEARCH_ROUNDED,
            on_submit=self._find,
            on_change=self._on_search,
        )
        watch_text_focus(self.search)
        self.back = ft.TextButton(
            content="Parent block",
            icon=ft.Icons.ARROW_UPWARD_ROUNDED,
            on_click=lambda _: self._navigate_parent(),
            visible=False,
        )
        self.summary = ft.Text(size=12, color=palette.muted)
        self.context = ft.Text(size=12, color=palette.ink)
        self.hint = ft.Text(size=12, color=palette.muted)
        self.zoom_text = ft.Text(size=12)
        self.notice = ft.Container(visible=False, padding=10, border_radius=8)
        self.previous = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,
            tooltip="Previous tensors",
            on_click=lambda _: self._move(-1),
        )
        self.next = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT_ROUNDED,
            tooltip="Next tensors",
            on_click=lambda _: self._move(1),
        )
        self.page_text = ft.Text(size=12)
        self.pager = ft.Row(
            controls=[self.previous, self.page_text, self.next],
            alignment=ft.MainAxisAlignment.CENTER,
            visible=False,
        )
        self.graph_tools = ft.Row(
            wrap=True,
            controls=[
                self.back,
                self.context,
                ft.IconButton(
                    icon=ft.Icons.REMOVE_ROUNDED,
                    tooltip="Zoom out",
                    on_click=lambda _: self.zoom(0.8),
                ),
                self.zoom_text,
                ft.IconButton(
                    icon=ft.Icons.ADD_ROUNDED,
                    tooltip="Zoom in",
                    on_click=lambda _: self.zoom(1.25),
                ),
                ft.TextButton(
                    content="Fit graph",
                    icon=ft.Icons.FIT_SCREEN_ROUNDED,
                    on_click=lambda _: self.fit(),
                ),
                ft.TextButton(
                    content="Readable view",
                    icon=ft.Icons.RESTART_ALT_ROUNDED,
                    on_click=lambda _: self.fit(readable=True),
                ),
            ],
        )
        self.control = ft.Container(
            data="traced-model-workspace",
            padding=16,
            bgcolor=palette.canvas,
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=10,
                controls=[
                    ft.Row(
                        wrap=True,
                        controls=[
                            ft.Icon(
                                ft.Icons.ACCOUNT_TREE_OUTLINED, color=palette.accent
                            ),
                            ft.Text(
                                "Traced model", size=22, weight=ft.FontWeight.W_700
                            ),
                            ft.TextButton(
                                content="Trace settings",
                                icon=ft.Icons.TUNE_ROUNDED,
                                on_click=lambda _: self._configure_trace(),
                            ),
                        ],
                    ),
                    self.summary,
                    ft.Row(
                        wrap=True, controls=[self.trace_picker, self.mode, self.detail]
                    ),
                    self.notice,
                    ft.Row(
                        controls=[
                            self.search,
                            ft.IconButton(
                                icon=ft.Icons.SEARCH_ROUNDED,
                                tooltip="Find next block or tensor",
                                on_click=self._find,
                            ),
                        ]
                    ),
                    self.graph_tools,
                    self.body,
                    self.pager,
                    self.hint,
                ],
            ),
        )
        self.refresh()

    def reset(self) -> None:
        self._close_chooser()
        self.trace_picker.value = None
        self.mode.value, self.search.value = "graph", ""
        self._page, self._key, self._find_index = 0, None, -1
        self.refresh()

    def refresh(self, *, trace_id: str | None = None) -> None:
        self.control.bgcolor = self.palette.canvas
        self.summary.color = self.hint.color = self.palette.muted
        self.notice.bgcolor = self.palette.warning_soft
        session = self._session()
        results = session.traces() if session else ()
        self.trace_picker.options = [
            ft.DropdownOption(
                r.id,
                text=f"Trace {i + 1} · {r.execution_device.value.upper()} · "
                f"{r.capture_status.value} · {r.id[6:14]}",
            )
            for i, r in enumerate(results)
        ]
        ids = {r.id for r in results}
        if trace_id is not None:
            self.trace_picker.value = trace_id
        if self.trace_picker.value not in ids:
            active = self._active_trace()
            self.trace_picker.value = (
                active if active in ids else results[-1].id if results else None
            )
        ready = session is not None and self.trace_picker.value is not None
        self.trace_picker.disabled = self.mode.disabled = (
            self.search.disabled
        ) = not ready
        self.notice.visible = False
        if not ready:
            self._key, self._index, self._ports, self._records = None, None, {}, {}
            self.port_layer.controls.clear()
            self.renderer.replace_scene(Scene(()), self._view)
            self.summary.value = (
                "Run a trace to inspect captured inputs and outputs on the model graph."
            )
            self.body.content = ft.Column(
                controls=[
                    ft.Text("No trace yet", size=20),
                    ft.Text(
                        "Capture a run to explore the values flowing through blocks."
                    ),
                    ft.FilledButton(
                        content="Prepare trace",
                        on_click=lambda _: self._configure_trace(),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            self.graph_tools.visible = self.pager.visible = self.detail.visible = False
            self.hint.value = (
                "Click a small ellipse port to open View, Statistics, "
                "and Visualization."
            )
            return
        assert session is not None and self.trace_picker.value is not None
        result = session.trace(self.trace_picker.value)
        self._records = {r.value_id: r for r in result.records}
        self.summary.value = (
            f"{session.title} · {result.execution_device.value.upper()} / "
            f"{result.execution_provider} · "
            f"{len(result.captured_value_ids)} readable tensors"
        )
        stale = result.key.revision_id != session.editing.current_revision_id
        self.mode.options[0].disabled = stale
        if stale:
            self.mode.value = "tensors"
            self.notice.visible = True
            self.notice.content = ft.Text(
                "The model changed after this trace. Inspect recorded tensors here, "
                "or run a new trace to see ports on the updated graph.",
                color=self.palette.warning,
            )
        is_graph = self.mode.value == "graph"
        self.graph_tools.visible = self.detail.visible = is_graph
        if not is_graph:
            self.body.content = self.tensor_list
            self._render_tensors()
            self.hint.value = "Choose a tensor to inspect its captured values."
            return
        self.body.content = self.canvas
        self.pager.visible = False
        layout, graph_id = self._graph_slice(), self._graph_id()
        if layout is None or graph_id is None:
            return
        graph = session.document.graphs[graph_id]
        self.detail.value = layout.detail_level.value
        self.back.visible = layout.root_group is not None
        self.context.value = (
            session.graph_hierarchy(graph_id).group(layout.root_group).label
            if layout.root_group
            else "Model"
        )
        key = (
            id(session),
            id(layout),
            result.id,
            result.capture_status,
            result.records,
        )
        if key != self._key:
            self._key = key
            scene = session.trace_overlay(layout, result)
            presentation = build_trace_graph(scene, graph, layout.members_by_glyph)
            scene, self._ports = graph_ports(
                presentation,
                graph,
                layout.members_by_glyph,
                scoped=layout.root_group is not None,
            )
            self._index, _ = scene_indexes(scene)
            self.renderer.replace_scene(scene, self._view)
            self.fit(readable=True, update=False)
        else:
            self._render_ports()

    def fit(self, *, readable: bool = False, update: bool = True) -> None:
        scene = self.renderer.scene
        if scene is None or not scene.nodes:
            return
        b = scene.bounds
        width, height = self._size
        scale = max(
            0.02,
            min(1.1, (width - 80) / max(1, b.width), (height - 80) / max(1, b.height)),
        )
        if readable:
            scale = max(1.25, scale)
        x = (b.min_x + b.max_x) / 2 - width / scale / 2
        y = (
            b.min_y - 40 / scale
            if b.height * scale > height - 80
            else (b.min_y + b.max_y) / 2 - height / scale / 2
        )
        self._view = Viewport(x, y, width / scale, height / scale, scale)
        self._apply_view(update=update)

    def zoom(self, factor: float, *, anchor: tuple[float, float] | None = None) -> None:
        width, height = self._size
        sx, sy = anchor or (width / 2, height / 2)
        x, y = self._view.to_world(sx, sy)
        scale = max(0.02, min(4, self._view.scale * factor))
        self._view = Viewport(
            x - sx / scale, y - sy / scale, width / scale, height / scale, scale
        )
        self._apply_view()

    def _apply_view(self, *, update: bool = True) -> None:
        self.renderer.set_viewport(self._view)
        self._render_ports()
        if update:
            self.page.update()

    def _render_ports(self) -> None:
        self.zoom_text.value = f"{self._view.scale:.0%}"
        self.port_layer.controls.clear()
        if self._index is None or self.renderer.scene is None:
            return
        visible = sorted(self._index.query(self._view.expanded(0.03)))
        crowded = self._view.scale < 0.35
        self.hint.value = (
            "Zoom in to inspect ports. Drag to pan; scroll to zoom."
            if crowded
            else "Input ports are on top; output ports below. "
            "Click a port to inspect; click a block to open it."
        )
        if crowded:
            return
        for glyph_id in visible:
            for port in self._ports.get(glyph_id, ()):
                if len(self.port_layer.controls) >= _PORT_LIMIT:
                    self.hint.value = (
                        "Zoom in to reveal the remaining ports in this area."
                    )
                    return
                self.port_layer.controls.append(self._port_control(port))

    def _port_control(self, port: TracePort) -> ft.Container:
        session, trace_id = self._session(), self.trace_picker.value
        assert session is not None and trace_id is not None
        graph = session.document.graphs[
            self._graph_id() or session.document.entry_graph
        ]
        labels = [port_label(graph, value) for value in port.values]
        record = self._records.get(port.values[0])
        status = self._status(record)
        shape = (
            " x ".join(map(str, record.shape))
            if record and record.shape
            else "scalar"
            if record
            else ""
        )
        label = (
            f"{port.direction.title()}: {labels[0]}"
            if len(labels) == 1
            else f"{len(labels)} {port.direction} tensors"
        )
        tooltip = f"{label}\n{shape} · {status}\nClick to inspect"
        color = (
            self.palette.accent if port.direction == "input" else self.palette.success
        )
        if record is None or not record.readable:
            color = self.palette.muted
        elif record.state is CaptureState.TRUNCATED:
            color = self.palette.warning
        x, y = self._view.to_screen(port.x, port.y)
        scene = self.renderer.scene
        assert scene is not None
        node = scene.node(port.glyph_id)
        count = sum(p.direction == port.direction for p in self._ports[port.glyph_id])
        hit_width = max(8.0, min(28.0, node.width * self._view.scale / (count + 1) - 2))
        hit_height = max(8.0, min(28.0, node.height * self._view.scale * 0.9))
        width, height = min(20.0, hit_width - 4), min(12.0, hit_height - 4)
        # Keep adjacent targets separate when zoomed out or densely packed.
        return ft.Container(
            data=f"trace-port:{port.glyph_id}:{port.direction}:{port.values[0]}",
            left=x - hit_width / 2,
            top=y - hit_height / 2,
            width=hit_width,
            height=hit_height,
            tooltip=ft.Tooltip(message=tooltip, exclude_from_semantics=True),
            border_radius=14,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            ink=True,
            alignment=ft.Alignment.CENTER,
            on_click=lambda _: self._inspect(port.values, label, session, trace_id),
            content=ft.Semantics(
                label=port.direction.title(),
                hint_text=tooltip,
                button=True,
                exclude_semantics=True,
                content=ft.Container(
                    width=width,
                    height=height,
                    content=ft.Stack(
                        controls=[
                            cv.Canvas(
                                width=width,
                                height=height,
                                shapes=[
                                    cv.Oval(
                                        0, 0, width, height, paint=ft.Paint(color=color)
                                    ),
                                    cv.Oval(
                                        0,
                                        0,
                                        width,
                                        height,
                                        paint=ft.Paint(
                                            color=self.palette.panel,
                                            style=ft.PaintingStyle.STROKE,
                                            stroke_width=1.5,
                                        ),
                                    ),
                                ],
                            ),
                            ft.Text(
                                "+" if len(port.values) > 1 else "",
                                size=9,
                                color=self.palette.panel,
                            ),
                        ],
                        alignment=ft.Alignment.CENTER,
                    ),
                    alignment=ft.Alignment.CENTER,
                ),
            ),
        )

    @staticmethod
    def _status(record: ActivationRecord | None) -> str:
        if record is None:
            return "Not captured"
        return (
            "Preview"
            if record.state is CaptureState.TRUNCATED
            else record.state.value.capitalize()
        )

    def _inspect(
        self, values: tuple[str, ...], title: str, session: ModelSession, trace_id: str
    ) -> None:
        if self._session() is not session:
            return
        self._close_chooser()
        if len(values) == 1:
            self._open_tensor(trace_id, values[0], title=title)
            return
        graph = session.document.graphs[
            self._graph_id() or session.document.entry_graph
        ]
        self._chooser = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Container(
                width=440,
                height=300,
                content=ft.ListView(
                    controls=[
                        ft.TextButton(
                            content=(
                                f"{port_label(graph, value)} · "
                                f"{self._status(self._records.get(value))}"
                            ),
                            on_click=lambda _, v=value: self._inspect(
                                (v,), port_label(graph, v), session, trace_id
                            ),
                        )
                        for value in values
                    ]
                ),
            ),
            actions=[
                ft.TextButton(content="Close", on_click=lambda _: self._close_chooser())
            ],
        )
        self.page.show_dialog(self._chooser)

    def _close_chooser(self) -> bool:
        chooser = self._chooser
        self._chooser = None
        if chooser is None:
            return False
        # Setting open=False and updating the page leaves the dialog in
        # Flet's dialog stack. Each later chooser is another mounted tree,
        # and every page update walks all of them.
        if chooser.open and hasattr(self.page, "pop_dialog"):
            self.page.pop_dialog()
        else:
            chooser.open = False
        return True

    def _on_selected(self, ids: frozenset[str]) -> None:
        session, graph_id = self._session(), self._graph_id()
        if session is None or graph_id is None or not ids:
            return
        key = next(iter(ids))
        if session.graph_hierarchy(graph_id).has_group(key):
            self._show_group(key)
            self.refresh()
            self.page.update()

    def _navigate_parent(self) -> None:
        self._show_parent()
        self.refresh()
        self.page.update()

    def _on_detail(self, _: ft.Event[ft.Dropdown]) -> None:
        if self.detail.value is not None:
            self._set_detail(DetailLevel(self.detail.value))
            self.refresh()
            self.page.update()

    def _on_trace(self, _: ft.Event[ft.Dropdown]) -> None:
        self._close_chooser()
        if self.trace_picker.value:
            self._select_trace(self.trace_picker.value)
        self.refresh()
        self.page.update()

    def _on_mode(self, _: ft.Event[ft.Dropdown]) -> None:
        self._page = 0
        self.refresh()
        self.page.update()

    def _on_search(self, _: ft.Event[ft.TextField]) -> None:
        self._find_index = -1
        if self.mode.value == "tensors":
            self._page = 0
            self._render_tensors()
            self.page.update()

    def _find(self, _: ft.Event[ft.TextField] | ft.Event[ft.IconButton]) -> None:
        scene, session, graph_id = (
            self.renderer.scene,
            self._session(),
            self._graph_id(),
        )
        query = (self.search.value or "").strip().casefold()
        if (
            self.mode.value != "graph"
            or scene is None
            or not query
            or session is None
            or graph_id is None
        ):
            return
        graph = session.document.graphs[graph_id]
        matches = [
            n
            for n in scene.nodes
            if query in n.label.casefold()
            or any(
                query in port_label(graph, value).casefold()
                or query in graph.value(value).name.casefold()
                for p in self._ports.get(n.id, ())
                for value in p.values
            )
        ]
        if matches:
            self._find_index = (self._find_index + 1) % len(matches)
            node = matches[self._find_index]
            self._view = self.renderer.viewport_focused_on(
                node.id,
                width=self._size[0],
                height=self._size[1],
                scale=max(0.8, self._view.scale),
            )
            self.renderer.set_selection(frozenset({node.id}))
            self._apply_view(update=False)
            self.hint.value = (
                f"Match {self._find_index + 1} of {len(matches)} · {node.label}"
            )
        else:
            self.hint.value = "No matching blocks or tensors in this graph view."
        self.page.update()

    def _render_tensors(self) -> None:
        session, trace_id = self._session(), self.trace_picker.value
        if session is None or trace_id is None:
            return
        query = (self.search.value or "").casefold()
        records = [
            r
            for r in self._records.values()
            if query in f"{r.value_name} {r.value_id}".casefold()
        ]
        pages = max(1, (len(records) + _TENSOR_PAGE - 1) // _TENSOR_PAGE)
        self._page = min(self._page, pages - 1)
        self.tensor_list.controls = [
            ft.OutlinedButton(
                content=(
                    f"{r.value_name} · {' x '.join(map(str, r.shape)) or 'scalar'} · "
                    f"{self._status(r)}"
                ),
                data=f"trace-tensor:{r.value_id}",
                on_click=lambda _, r=r: self._inspect(
                    (r.value_id,), r.value_name, session, trace_id
                ),
            )
            for r in records[
                self._page * _TENSOR_PAGE : (self._page + 1) * _TENSOR_PAGE
            ]
        ]
        if not records:
            self.tensor_list.controls = [ft.Text("No matching captured tensors.")]
        self.pager.visible = pages > 1
        self.previous.disabled, self.next.disabled = (
            self._page == 0,
            self._page >= pages - 1,
        )
        self.page_text.value = (
            f"Page {self._page + 1} of {pages} · {len(records)} tensors"
        )

    def _move(self, delta: int) -> None:
        self._page = max(0, self._page + delta)
        self._render_tensors()
        self.page.update()

    def _on_size(self, event: ft.LayoutSizeChangeEvent[ft.LayoutControl]) -> None:
        if event.width > 0 and event.height > 0:
            first = not self._measured
            self._measured = True
            self._size = float(event.width), float(event.height)
            if first:
                self.fit(readable=True)
            else:
                center_x = self._view.x + self._view.width / 2
                center_y = self._view.y + self._view.height / 2
                width = self._size[0] / self._view.scale
                height = self._size[1] / self._view.scale
                self._view = Viewport(
                    center_x - width / 2,
                    center_y - height / 2,
                    width,
                    height,
                    self._view.scale,
                )
                self._apply_view()

    def _on_pan(self, event: ft.DragUpdateEvent[ft.GestureDetector]) -> None:
        if event.local_delta is not None:
            v = self._view
            self._view = Viewport(
                v.x - event.local_delta.x / v.scale,
                v.y - event.local_delta.y / v.scale,
                v.width,
                v.height,
                v.scale,
            )
            self._apply_view()

    def _on_scroll(self, event: ft.ScrollEvent[ft.GestureDetector]) -> None:
        if event.scroll_delta is not None and event.scroll_delta.y:
            self.zoom(
                0.9 if event.scroll_delta.y > 0 else 1.1,
                anchor=(event.local_position.x, event.local_position.y),
            )
