"""Tests for the tensor inspector (P3.4): viewmodel and shell wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import flet as ft
import flet.canvas as cv
import pytest

from nneditor.analysis.statistics import TensorStatistics, compute_statistics
from nneditor.application.session import ApplicationService
from nneditor.storage.store import TensorStore
from nneditor.ui import tensor_tools, viewmodel
from nneditor.ui.app import Shell
from tests.fixtures.onnx_models import build_embedded_model, build_external_model
from tests.unit.test_shell import StubPage
from tests.unit.test_tensor_store import embedded_document


class TestViewmodel:
    def test_node_tensor_ids_finds_weights_in_port_order(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        scale = document.main_graph.nodes[0]
        shift = document.main_graph.nodes[1]
        assert viewmodel.node_tensor_ids(document, document.entry_graph, scale.id) == (
            document.main_graph.initializers[0],
        )
        assert viewmodel.node_tensor_ids(document, document.entry_graph, shift.id) == (
            document.main_graph.initializers[1],
        )

    def test_tensor_lines_disclose_storage_and_access(self, tmp_path: Path) -> None:
        path, _values = build_external_model(tmp_path, elements=16)
        from nneditor.adapters.onnx import index_model, index_to_document

        document = index_to_document(index_model(path))
        tensor_id = document.main_graph.initializers[0]
        lines = dict(
            viewmodel.tensor_lines(
                document, tensor_id, materialization="range", byte_length=64
            )
        )
        assert lines["Dtype"] == "float32"
        assert lines["Shape"] == "[16]"
        assert lines["Storage"] == "external file"
        assert lines["Access"] == "range"
        assert lines["Bytes"] == "64"
        assert lines["Location"] == "weights.bin"
        assert lines["Quantization"] == "none recorded"
        assert lines["Provenance"].startswith("import of sha256:")

    def test_preview_values_text(self) -> None:
        import struct

        raw = struct.pack("<4f", 1.0, 2.5, -3.0, 4.0)
        assert viewmodel.preview_values_text("float32", raw) == "[1, 2.5, -3, 4]"
        long = struct.pack("<10f", *range(10))
        assert viewmodel.preview_values_text("float32", long).endswith(", …]")
        assert "not decodable" in viewmodel.preview_values_text("string", b"ab")

    def test_statistics_and_histogram_rows(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            stats = compute_statistics(store, weight_id)
        lines = dict(viewmodel.statistics_lines(stats))
        assert set(lines) == {"Min", "Max", "Mean", "Std", "Sparsity", "Non-finite"}
        rows = viewmodel.histogram_rows(stats, max_rows=16)
        assert 0 < len(rows) <= 16
        assert sum(count for _label, count, _fraction in rows) == stats.element_count
        assert max(fraction for _label, count, fraction in rows) == 1.0

    def test_histogram_rows_merge_down_to_max(self) -> None:
        stats = TensorStatistics(
            tensor_id="t",
            element_type="float32",
            element_count=8,
            byte_size=32,
            minimum=0.0,
            maximum=8.0,
            mean=4.0,
            std=1.0,
            zero_count=0,
            nan_count=0,
            inf_count=0,
            histogram_edges=tuple(float(index) for index in range(9)),
            histogram_counts=(1, 1, 1, 1, 1, 1, 1, 1),
        )
        rows = viewmodel.histogram_rows(stats, max_rows=4)
        assert len(rows) == 4
        assert all(count == 2 for _label, count, _fraction in rows)
        empty = TensorStatistics(
            tensor_id="t",
            element_type="float32",
            element_count=0,
            byte_size=0,
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
            zero_count=0,
            nan_count=0,
            inf_count=0,
            histogram_edges=(),
            histogram_counts=(),
        )
        assert viewmodel.histogram_rows(empty) == ()


class TestTensorTools:
    def test_hex_format_parse_and_ascii_preview(self) -> None:
        raw = b"ABC\x00\xff"
        formatted = tensor_tools.format_hex_bytes(raw, columns=4)
        assert formatted == "41 42 43 00\nFF"
        assert tensor_tools.parse_hex_bytes("41 0x42\nff") == b"AB\xff"
        assert tensor_tools.ascii_preview(raw, columns=4) == "ABC.\n."

    def test_hex_parser_rejects_ambiguous_or_oversized_input(self) -> None:
        for invalid in ("", "0", "GG", "123"):
            with pytest.raises(ValueError):
                tensor_tools.parse_hex_bytes(invalid)
        with pytest.raises(ValueError, match="at most 2"):
            tensor_tools.parse_hex_bytes("00 01 02", max_bytes=2)

    def test_offsets_accept_decimal_and_hex_and_are_bounded(self) -> None:
        assert tensor_tools.parse_offset("15", total_bytes=16) == 15
        assert tensor_tools.parse_offset("0x0F", total_bytes=16) == 15
        assert tensor_tools.parse_offset("0", total_bytes=0) == 0
        with pytest.raises(ValueError, match="below"):
            tensor_tools.parse_offset("16", total_bytes=16)

    def test_value_samples_and_diverging_colors(self) -> None:
        import struct

        raw = struct.pack("<4f", -2.0, 0.0, 1.0, float("nan"))
        assert tensor_tools.sample_values("float32", raw, limit=3) == (
            -2.0,
            0.0,
            1.0,
        )
        assert tensor_tools.sample_values("string", raw) is None
        assert tensor_tools.heat_color(-2.0, 2.0) == "#2E90FA"
        assert tensor_tools.heat_color(0.0, 2.0) == "#F2F4F7"
        assert tensor_tools.heat_color(2.0, 2.0) == "#7F56D9"
        assert tensor_tools.heat_color(float("nan"), 2.0) == "#98A2B3"


class TestShellIntegration:
    def make_shell(
        self, tmp_path: Path, service: ApplicationService
    ) -> tuple[Shell, str, str]:
        model = tmp_path / "model.onnx"
        build_embedded_model(model, elements=64)
        page = StubPage()
        shell = Shell(cast(ft.Page, page), service)
        shell.build()
        session = service.open_model(model)
        shell.show_session(session)
        node_id = session.document.main_graph.nodes[0].id
        tensor_id = session.document.main_graph.initializers[0]
        return shell, node_id, tensor_id

    def controls(self, shell: Shell) -> tuple[ft.Control, ...]:
        found: list[ft.Control] = []

        def visit(control: ft.Control) -> None:
            found.append(control)
            children = getattr(control, "controls", None)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, ft.Control):
                        visit(child)
            content = getattr(control, "content", None)
            if isinstance(content, ft.Control):
                visit(content)
            for attribute in ("title", "subtitle", "leading", "trailing"):
                child = getattr(control, attribute, None)
                if isinstance(child, ft.Control):
                    visit(child)

        for control in shell.inspector.controls:
            visit(control)
        return tuple(found)

    def inspector_text(self, shell: Shell) -> str:
        return "\n".join(
            str(control.value or "")
            for control in self.controls(shell)
            if isinstance(control, ft.Text)
        )

    def find_data(self, shell: Shell, data: str) -> ft.Control:
        return next(control for control in self.controls(shell) if control.data == data)

    def test_selecting_a_node_shows_its_tensor_with_preview(
        self, tmp_path: Path
    ) -> None:
        with ApplicationService() as service:
            shell, node_id, tensor_id = self.make_shell(tmp_path, service)
            shell._refresh_inspector(frozenset({node_id}))
            text = self.inspector_text(shell)
            assert "Tensor" in text and tensor_id in text
            assert "ACCESS" in text and "range" in text
            assert "Preview: [" in text
            buttons = [
                control
                for control in self.controls(shell)
                if isinstance(control, ft.TextButton)
                and control.data == f"tensor-compute-statistics:{tensor_id}"
            ]
            assert buttons, "the compute-statistics affordance is offered"
            heatmap = self.find_data(shell, f"tensor-heatmap:{tensor_id}")
            assert isinstance(heatmap, ft.Column)
            canvas = next(
                control
                for control in self.controls(shell)
                if isinstance(control, cv.Canvas)
            )
            assert len(canvas.shapes) == tensor_tools.HEATMAP_ELEMENTS
            assert self.find_data(shell, f"tensor-hex-editor:{tensor_id}")

    def test_computing_statistics_updates_the_inspector(self, tmp_path: Path) -> None:
        with ApplicationService() as service:
            shell, node_id, tensor_id = self.make_shell(tmp_path, service)
            shell._refresh_inspector(frozenset({node_id}))
            button = self.find_data(shell, f"tensor-compute-statistics:{tensor_id}")
            assert isinstance(button, ft.TextButton)
            handler = button.on_click
            assert handler is not None
            from collections.abc import Callable

            cast(Callable[[object], None], handler)(None)
            # The stub page runs threads synchronously, so the job is done.
            assert shell.session is not None
            assert shell.session.statistics(tensor_id) is not None
            text = self.inspector_text(shell)
            assert "Statistics" in text
            assert "SPARSITY" in text
            assert "█" in text, "histogram bars are rendered"
            assert "ready" in (shell.status_text.value or "")

    def test_unreadable_tensor_shows_error_state(self, tmp_path: Path) -> None:
        with ApplicationService() as service:
            shell, node_id, tensor_id = self.make_shell(tmp_path, service)
            # Sabotage the payload so the preview read fails.
            tensor = shell.session.document.tensors[tensor_id]  # type: ignore[union-attr]
            object.__setattr__(tensor, "element_type", "string")
            shell._refresh_inspector(frozenset({node_id}))
            text = self.inspector_text(shell)
            assert "values cannot be decoded" in text
            assert self.find_data(shell, f"tensor-hex-editor:{tensor_id}")

    def test_hex_edit_creates_a_reversible_revision(self, tmp_path: Path) -> None:
        with ApplicationService() as service:
            shell, node_id, tensor_id = self.make_shell(tmp_path, service)
            shell._refresh_inspector(frozenset({node_id}))
            assert shell.session is not None
            before = shell.session.editing.read(tensor_id, offset=0, length=1)
            replacement = b"\xff" if before != b"\xff" else b"\x00"
            shell._hex_drafts[(tensor_id, 0)] = replacement.hex()
            apply = self.find_data(shell, f"tensor-hex-apply:{tensor_id}")
            assert isinstance(apply, ft.FilledButton)
            assert apply.on_click is not None

            cast(Any, apply.on_click)(None)

            assert (
                shell.session.editing.read(tensor_id, offset=0, length=1) == replacement
            )
            assert shell.session.editing.can_undo
            assert "Applied 1 byte" in (shell.status_text.value or "")
            shell.session.editing.undo()
            assert shell.session.editing.read(tensor_id, offset=0, length=1) == before

    def test_hex_editor_pages_without_reading_the_whole_tensor(
        self, tmp_path: Path
    ) -> None:
        with ApplicationService() as service:
            shell, node_id, tensor_id = self.make_shell(tmp_path, service)
            shell._refresh_inspector(frozenset({node_id}))
            next_button = self.find_data(shell, f"tensor-hex-next:{tensor_id}")
            assert isinstance(next_button, ft.IconButton)
            assert next_button.on_click is not None and not next_button.disabled

            cast(Any, next_button.on_click)(None)

            assert shell._hex_offsets[tensor_id] == tensor_tools.HEX_PAGE_BYTES
            offset_field = self.find_data(shell, f"tensor-hex-offset:{tensor_id}")
            assert isinstance(offset_field, ft.TextField)
            assert offset_field.value == "0x80"
