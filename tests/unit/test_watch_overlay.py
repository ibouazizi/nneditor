"""Headless wiring tests for the watch list and the anomaly overlay.

The watch strip, the pin action on activation cards, the ``build_shapes``
tint seam, and the shell's overlay cycle are exercised through the same stub
page and real trace worker the other shell tests use.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import flet.canvas as cv

from nneditor.analysis.statistics import TensorStatistics
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.editing.validation import RenameNodeRequest
from nneditor.rendering.flet_canvas_managed import ManagedCanvasRenderer
from nneditor.rendering.flet_shapes import BuiltShapes, build_shapes
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene, Viewport
from nneditor.rendering.spatial import scene_indexes
from nneditor.tracing.contracts import TraceKey, TraceResult
from nneditor.ui.app import Shell
from nneditor.ui.watch_panel import comparison_pair
from tests.fixtures.onnx_models import build_embedded_model
from tests.unit.test_shell import StubPage, make_shell


def _event() -> Any:
    return cast(Any, SimpleNamespace())


def open_traced_shell(
    service: ApplicationService, tmp_path: Path
) -> tuple[Shell, StubPage, ModelSession]:
    """A shell showing one opened model with one completed real trace."""
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    shell, page = make_shell(service)
    session = service.open_model(path)
    shell.show_session(session)
    shell.trace.run(_event())
    assert shell.trace.active_trace_id is not None
    return shell, page, session


def walk(root: ft.Control) -> list[ft.Control]:
    """Controls reachable via ``controls``/``content``/``leading``/``title``,
    in document order."""
    found: list[ft.Control] = []

    def visit(control: Any) -> None:
        if not isinstance(control, ft.Control):
            return
        found.append(control)
        for attribute in ("leading", "title", "content"):
            visit(getattr(control, attribute, None))
        for child in getattr(control, "controls", None) or []:
            visit(child)

    visit(root)
    return found


def find_data(root: ft.Control, prefix: str) -> list[ft.Control]:
    return [
        control
        for control in walk(root)
        if isinstance(getattr(control, "data", None), str)
        and cast(str, control.data).startswith(prefix)
    ]


def texts(root: ft.Control) -> list[str]:
    return [
        str(control.value)
        for control in walk(root)
        if isinstance(control, ft.Text) and control.value
    ]


def fake_statistics(value_id: str, *, nan_count: int = 0) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=value_id,
        element_type="float32",
        element_count=8,
        byte_size=32,
        minimum=-1.0,
        maximum=1.0,
        mean=0.0,
        std=0.5,
        zero_count=0,
        nan_count=nan_count,
        inf_count=0,
        histogram_edges=(-1.0, 1.0),
        histogram_counts=(8,),
    )


# -- (a) pin, unpin, and clear on close -----------------------------------


def test_pin_from_activation_card_unpin_and_clear_on_close(
    tmp_path: Path,
) -> None:
    with ApplicationService() as service:
        shell, _page, session = open_traced_shell(service, tmp_path)
        node_id = session.document.main_graph.nodes[0].id
        shell.current_detail = shell.current_detail.OPERATOR
        shell._show_graph(session.document.entry_graph)
        shell._refresh_inspector(frozenset({node_id}))

        pin_buttons = [
            card.leading
            for card in shell.inspector.controls
            if isinstance(card, ft.ExpansionTile)
            and isinstance(card.data, str)
            and card.data.startswith("activation-card:")
        ]
        assert pin_buttons, "every activation card carries a pin action"
        button = pin_buttons[0]
        assert isinstance(button, ft.IconButton)
        assert isinstance(button.data, str)
        value_id = button.data.removeprefix("activation-pin:")

        cast(Any, button.on_click)(_event())
        assert shell.watch.pinned() == (value_id,)
        rows = find_data(shell.watch.strip, f"watch-row:{value_id}")
        assert len(rows) == 1
        assert "Pinned" in str(shell.status_text.value)

        # Pinning again stays idempotent.
        cast(Any, button.on_click)(_event())
        assert shell.watch.pinned() == (value_id,)

        unpin = find_data(rows[0], f"watch-unpin:{value_id}")
        assert len(unpin) == 1
        cast(Any, cast(ft.IconButton, unpin[0]).on_click)(_event())
        assert shell.watch.pinned() == ()
        assert not find_data(shell.watch.strip, "watch-row:")

        cast(Any, button.on_click)(_event())
        assert shell.watch.pinned() == (value_id,)
        shell._on_close_model(cast(Any, None))
        assert shell.watch.pinned() == (), "pins are per-session state"
        assert not find_data(shell.watch.strip, "watch-row:")


def test_watch_rows_mount_under_the_activations_section(tmp_path: Path) -> None:
    with ApplicationService() as service:
        shell, _page, session = open_traced_shell(service, tmp_path)
        assert shell.info_activations_tile.expanded, "a trace opens the section"
        status = find_data(shell.info_activations_tile, "trace-status-line")
        assert len(status) == 1
        trace_id = shell.trace.active_trace_id
        assert trace_id is not None
        assert trace_id[:12] in str(cast(ft.Text, status[0]).value)

        output_id = session.document.main_graph.outputs[0]
        shell.watch.pin(output_id, "output")
        rows = find_data(shell.info_activations_tile, f"watch-row:{output_id}")
        assert len(rows) == 1, "pinned rows render inside info:activations"
        assert not find_data(shell.right_panel, "watch-strip"), (
            "the explorer no longer hosts the watch strip"
        )

        unpin = find_data(rows[0], f"watch-unpin:{output_id}")
        assert len(unpin) == 1
        cast(Any, cast(ft.IconButton, unpin[0]).on_click)(_event())
        assert shell.watch.pinned() == ()
        assert not find_data(shell.info_activations_tile, "watch-row:")


# -- (b) statistics job lifecycle -----------------------------------------


def test_watched_row_shows_statistics_after_the_job_lands(
    tmp_path: Path,
) -> None:
    with ApplicationService() as service:
        shell, page, session = open_traced_shell(service, tmp_path)
        trace_id = shell.trace.active_trace_id
        assert trace_id is not None
        output_id = session.document.main_graph.outputs[0]

        callbacks: list[Any] = []
        cast(Any, page).run_thread = callbacks.append
        shell.watch.pin(output_id, "output")

        row = find_data(shell.watch.strip, f"watch-row:{output_id}")[0]
        assert "Loading activation statistics…" in texts(row)
        assert not find_data(row, "watch-statistics:")
        assert (trace_id, output_id, "statistics") in (
            shell.activations.activation_loading
        )
        assert callbacks, "the completion wait rides the run_thread list"

        while callbacks:
            callbacks.pop(0)()

        assert (trace_id, output_id) in shell.activations.activation_statistics
        row = find_data(shell.watch.strip, f"watch-row:{output_id}")[0]
        stats_sections = find_data(row, f"watch-statistics:{output_id}")
        assert len(stats_sections) == 1
        assert "Loading activation statistics…" not in texts(row)


# -- (c) per-value deltas across two same-spec traces ----------------------


def test_watched_row_shows_deltas_with_two_same_spec_traces(
    tmp_path: Path,
) -> None:
    with ApplicationService() as service:
        shell, _page, session = open_traced_shell(service, tmp_path)
        first = shell.trace.active_trace_id

        # A committed rename creates a new revision, so the second run gets
        # its own trace identity while inputs and numerics stay identical.
        node = session.document.main_graph.nodes[0]
        shell.renderer.set_selection(frozenset({node.id}))
        shell.pending_edit = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "renamed")
        )
        shell.pending_target = node.id
        shell._on_commit_edit(cast(Any, None))
        shell.trace.run(_event())
        second = shell.trace.active_trace_id
        assert second is not None and second != first

        traces = session.traces()
        assert len(traces) == 2
        assert len({trace.key.input_specification_hash for trace in traces}) == 1

        output_id = session.document.main_graph.outputs[0]
        shell.watch.pin(output_id, "output")

        row = find_data(shell.watch.strip, f"watch-row:{output_id}")[0]
        delta_sections = find_data(row, f"watch-delta:{output_id}")
        assert len(delta_sections) == 1
        labels = texts(delta_sections[0])
        assert "Δ MAX ABS" in labels
        assert "Δ MAX REL" in labels
        # A rename does not change the math, so the delta is exactly zero.
        maximum = labels[labels.index("Δ MAX ABS") + 1]
        assert maximum == "0"


def test_comparison_pair_requires_the_same_input_specification() -> None:
    def result(revision: str | None, spec: str) -> TraceResult:
        return TraceResult(
            TraceKey("artifact", revision, spec),
            (),
            runtime="test",
        )

    base = result(None, "spec-a")
    edited = result("rev-1", "spec-a")
    other_spec = result("rev-2", "spec-b")
    traces = (base, edited, other_spec)

    assert comparison_pair(traces, edited.id) == (base, edited)
    assert comparison_pair(traces, other_spec.id) is None
    assert comparison_pair(traces, "missing") is None
    assert comparison_pair((edited,), edited.id) is None


# -- (d) the tint seam keeps untinted output identical ---------------------


def _tiny_scene() -> Scene:
    return Scene(
        (
            NodeGlyph("a", 0.0, 0.0, 100.0, 40.0, "conv", "A"),
            NodeGlyph("b", 0.0, 100.0, 100.0, 40.0, "norm", "B"),
        ),
        (EdgeGlyph("edge", "a", "b", ((50.0, 40.0), (50.0, 100.0))),),
    )


def _build_tiny(**kwargs: Any) -> BuiltShapes:
    scene = _tiny_scene()
    node_index, edge_index = scene_indexes(scene)
    return build_shapes(
        scene,
        Viewport(0.0, 0.0, 1600.0, 1000.0, scale=1.0),
        node_index,
        edge_index,
        {"a": 0, "b": 1},
        frozenset(),
        **kwargs,
    )


def _serialize(built: BuiltShapes) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for shape in built.shapes:
        if isinstance(shape, cv.Rect):
            assert shape.paint is not None
            rows.append(
                (
                    "rect",
                    shape.x,
                    shape.y,
                    shape.width,
                    shape.height,
                    shape.border_radius,
                    shape.paint.color,
                    str(shape.paint.style),
                )
            )
        elif isinstance(shape, cv.Line):
            assert shape.paint is not None
            rows.append(
                (
                    "line",
                    shape.x1,
                    shape.y1,
                    shape.x2,
                    shape.y2,
                    shape.paint.color,
                    shape.paint.stroke_width,
                )
            )
        elif isinstance(shape, cv.Text):
            assert shape.style is not None
            rows.append(("text", shape.x, shape.y, shape.value, shape.style.color))
        else:  # pragma: no cover - no other shape kinds are emitted
            raise AssertionError(f"unexpected shape {shape!r}")
    return tuple(rows)


TINY_SCENE_GOLDEN = (
    ("line", 50.0, 40.0, 50.0, 100.0, "#C5CAD4", 1.0),
    ("rect", 0.0, 0.0, 100.0, 40.0, 10.0, "#DCEBFF", "PaintingStyle.FILL"),
    ("text", 50.0, 20.0, "A\nConv", "#172033"),
    ("rect", 0.0, 100.0, 100.0, 40.0, 10.0, "#DDF7F2", "PaintingStyle.FILL"),
    ("text", 50.0, 120.0, "B\nNorm", "#172033"),
)
"""The exact untinted build of the tiny scene, pinned before the tint seam."""


def test_tint_none_keeps_build_shapes_output_byte_identical() -> None:
    default_build = _serialize(_build_tiny())
    explicit_none = _serialize(_build_tiny(tint=None))
    assert default_build == TINY_SCENE_GOLDEN
    assert explicit_none == TINY_SCENE_GOLDEN


def test_tint_overrides_only_the_named_glyph_fill() -> None:
    tinted = _serialize(
        _build_tiny(tint=lambda glyph_id: "#FF0000" if glyph_id == "a" else None)
    )
    expected = tuple(
        ("rect", 0.0, 0.0, 100.0, 40.0, 10.0, "#FF0000", "PaintingStyle.FILL")
        if row[:5] == ("rect", 0.0, 0.0, 100.0, 40.0)
        else row
        for row in TINY_SCENE_GOLDEN
    )
    assert tinted == expected


# -- (e) the shell's non-finite overlay ------------------------------------


def _danger_rects(shell: Shell) -> list[cv.Rect]:
    renderer = shell.renderer
    assert isinstance(renderer, ManagedCanvasRenderer)
    detector = renderer.control
    assert isinstance(detector, ft.GestureDetector)
    canvas = detector.content
    assert isinstance(canvas, cv.Canvas)
    return [
        shape
        for shape in canvas.shapes or []
        if isinstance(shape, cv.Rect)
        and shape.paint is not None
        and shape.paint.color == shell.palette.danger
    ]


def test_nonfinite_overlay_tints_and_legend_counts(tmp_path: Path) -> None:
    with ApplicationService() as service:
        shell, _page, session = open_traced_shell(service, tmp_path)
        trace_id = shell.trace.active_trace_id
        assert trace_id is not None
        shell.current_detail = shell.current_detail.OPERATOR
        shell._show_graph(session.document.entry_graph)

        result = session.trace(trace_id)
        record = next(item for item in result.records if item.node_id is not None)
        shell.activations.activation_statistics[(trace_id, record.value_id)] = (
            fake_statistics(record.value_id, nan_count=1)
        )

        shell._on_cycle_overlay(None)
        assert shell.overlay_mode == "nonfinite"
        assert shell.overlay_button.content == "Overlay: Non-finite"
        assert shell._overlay_tint(cast(str, record.node_id)) == (shell.palette.danger)
        assert len(_danger_rects(shell)) == 1

        scene = shell.renderer.scene
        assert scene is not None
        assert shell.overlay_legend.visible
        assert shell.overlay_legend.value == f"tinted 1 of {scene.node_count:,}"

        # Cycling to Magnitude re-tints from the same computed statistics.
        shell._on_cycle_overlay(None)
        assert shell.overlay_mode == "magnitude"
        assert shell.overlay_legend.value == f"tinted 1 of {scene.node_count:,}"
        assert not _danger_rects(shell)

        # And back to Off clears both the tint and the legend.
        shell._on_cycle_overlay(None)
        assert shell.overlay_mode == "off"
        assert not shell.overlay_legend.visible
        assert not _danger_rects(shell)


# -- (f) the overlay survives pan and zoom ---------------------------------


def test_overlay_survives_zoom_and_viewport_change(tmp_path: Path) -> None:
    with ApplicationService() as service:
        shell, _page, session = open_traced_shell(service, tmp_path)
        trace_id = shell.trace.active_trace_id
        assert trace_id is not None
        shell.current_detail = shell.current_detail.OPERATOR
        shell._show_graph(session.document.entry_graph)

        result = session.trace(trace_id)
        record = next(item for item in result.records if item.node_id is not None)
        shell.activations.activation_statistics[(trace_id, record.value_id)] = (
            fake_statistics(record.value_id, nan_count=1)
        )
        shell._on_cycle_overlay(None)
        assert len(_danger_rects(shell)) == 1

        renderer = shell.renderer
        assert isinstance(renderer, ManagedCanvasRenderer)
        rebuilds = renderer.rebuild_count
        shell._zoom_about_center(1.1)
        assert renderer.rebuild_count > rebuilds, "a zoom rebuilds the shapes"
        assert len(_danger_rects(shell)) == 1, "the tint survives the rebuild"

        # An in-margin pan takes the mutate-in-place path and keeps the
        # already tinted shape objects.
        moves = renderer.move_count
        shell.view["x"] += 5.0
        shell._request_viewport_apply()
        assert renderer.move_count > moves
        assert len(_danger_rects(shell)) == 1
