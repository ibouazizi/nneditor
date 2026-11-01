"""Decoded text and Python processing are directly accessible in tensor overlays."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import numpy as np

from nneditor.application.session import ApplicationService
from tests.fixtures.model_families import write_decodable_gguf
from tests.unit.test_shell import make_shell, press_key
from tests.unit.test_watch_overlay import open_traced_shell, texts, walk


def test_python_tab_runs_and_keeps_capture_and_draft(tmp_path: Path) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _, session = open_traced_shell(service, tmp_path)
        trace_id = shell.trace.active_trace_id or ""
        value_id = session.document.main_graph.outputs[0]
        original = session.activations(trace_id).read(value_id)
        shell.activations.open_tensor(trace_id, value_id)
        overlay = shell.activations.value_overlay
        assert overlay.tabs.length == 4
        overlay._set_script("")
        assert overlay.script_editor.value == ""
        overlay._set_script("result = tensor.shape")
        assert overlay.script_editor.value == "result = tensor.shape"
        overlay.tabs.selected_index = 3
        source = (
            "tensor += 1\nprint('script ran',math.sqrt(16))\n"
            "plt.plot(tensor.ravel())\nfloat(tensor.mean())"
        )
        overlay.script_editor.value = source
        press_key(shell, "Enter", ctrl=True)
        assert "script ran 4.0" in " ".join(texts(overlay.script_output))
        assert any(
            isinstance(control, ft.Image) for control in walk(overlay.script_output)
        )
        assert session.activations(trace_id).read(value_id) == original
        assert not overlay.script_run.disabled and overlay.script_cancel.disabled
        assert not overlay.script_editor.disabled and not overlay.script_clear.disabled
        overlay.script_editor.value = (
            "print('before failure')\nraise RuntimeError('example error')"
        )
        overlay.run_script()
        assert "example error" in " ".join(texts(overlay.script_output))
        assert "before failure" in " ".join(texts(overlay.script_output))
        press_key(shell, "Escape")
        assert overlay.dialog is None
        shell.activations.open_tensor(trace_id, value_id)
        assert (
            overlay.script_editor.value
            and "example error" in overlay.script_editor.value
        )


def test_decoded_tokens_and_predictions_use_the_gguf_vocabulary(tmp_path: Path) -> None:
    path = write_decodable_gguf(tmp_path / "llama.gguf")
    np.save(tmp_path / "ids.npy", np.array([[3, 4, 5]], np.int64))
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        shell.trace.bind_tensor(
            "tokens", session.trace_tensor_input("tokens", tmp_path / "ids.npy")
        )
        shell.trace.run(cast(Any, SimpleNamespace()))
        trace_id = shell.trace.active_trace_id or ""
        graph = session.document.main_graph
        shell.activations.open_tensor(trace_id, graph.inputs[0])
        overlay = shell.activations.value_overlay
        assert overlay.decode_panel.visible
        overlay._decode_text(cast(Any, SimpleNamespace()))
        assert "Hello world!" in texts(overlay.decode_result)
        shell.activations.open_tensor(trace_id, graph.outputs[0])
        overlay._decode_text(cast(Any, SimpleNamespace()))
        assert "Greedy next-token" in " ".join(texts(overlay.decode_result))
        table = next(
            c for c in walk(overlay.decode_result) if isinstance(c, ft.DataTable)
        )
        assert len(table.rows) == 8
