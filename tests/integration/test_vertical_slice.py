"""The Phase 1 vertical slice, end to end (P1.10).

The first test is the phase exit gate: a representative model whose weights
dominate the file and live in an external data file is opened, navigated,
searched, and inspected — and the parser instrumentation plus the tensor
store's file accounting prove that no tensor payload was materialized by any
of it. Import does stream source components once for cryptographic identity;
explicit inspection is the only operation that decodes or retains weight
bytes, and only retains the requested slice.

Boot smoke tests for the real desktop window and the Flet web server are
opt-in via ``NNEDITOR_SMOKE=1`` because they start real processes; the rest
of the suite must stay fast and displayless.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from nneditor.application.session import ApplicationService
from nneditor.cancellation import OperationCancelled
from nneditor.ir.capabilities import Capability
from tests.fixtures.onnx_models import LARGE_TENSOR_ELEMENTS, build_external_model

SMOKE = os.environ.get("NNEDITOR_SMOKE") == "1"


def test_exit_gate_large_external_model_opens_lazily(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path, values = build_external_model(model_dir, elements=LARGE_TENSOR_ELEMENTS)
    weights_file = model_dir / "weights.bin"
    payload_bytes = weights_file.stat().st_size
    assert payload_bytes == 4 * LARGE_TENSOR_ELEMENTS, "2 MiB of weights"

    with ApplicationService() as service:
        session = service.open_model(path)

        # Navigate: full scene, search, neighborhood, subgraph listing.
        layout = session.scene()
        assert layout.scene.node_count == 1
        assert session.search("scale")
        node_id = layout.scene.nodes[0].id
        assert session.neighborhood(session.document.entry_graph, node_id) == {node_id}

        # Inspect metadata: capabilities, tensor shape/dtype/storage.
        assert session.capability(Capability.TOPOLOGY) is not None
        weight_id = session.document.main_graph.initializers[0]
        reference = session.store.metadata(weight_id)
        assert reference.dims == (LARGE_TENSOR_ELEMENTS,)
        assert session.store.byte_length(weight_id) == payload_bytes

        # The session's tensor store has not opened a payload. Import already
        # streamed it through SHA-256 for identity, without decoding/caching it.
        assert session.store.open_file_count == 0

        # Explicit inspection is the first semantic read and retains only the
        # slice asked for.
        slice_bytes = session.store.read(weight_id, offset=0, length=64)
        assert slice_bytes == values.tobytes()[:64]
        assert session.store.open_file_count == 1


def test_import_reads_stay_within_the_structural_budget(tmp_path: Path) -> None:
    """Reconfirm the P0.4 laziness budget through the full import pipeline.

    Uses the *embedded* model, where the payload dominates the file, so the
    budget is meaningful. These counters measure the structural parser, not
    the separate streaming identity hash.
    """
    from nneditor.adapters.onnx import index_model
    from tests.fixtures.onnx_models import build_embedded_model

    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=LARGE_TENSOR_ELEMENTS)
    index = index_model(path)
    model_bytes = path.stat().st_size
    assert index.stats.logical_bytes < model_bytes * 0.01, (
        "structural reads stay under 1% of a payload-dominated file"
    )
    for tensor in index.iter_tensors():
        if tensor.payload is not None:
            assert not index.stats.touched_logically(tensor.payload)


def test_repeated_opens_and_closes_release_resources(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    from tests.fixtures.onnx_models import build_embedded_model

    build_embedded_model(path, elements=64)
    with ApplicationService() as service:
        for _ in range(5):
            session = service.open_model(path)
            session.scene()
            weight_id = session.document.main_graph.initializers[0]
            session.store.read(weight_id)
            service.close_session(session.id)
            assert session.store.closed
        assert service.open_sessions == ()


def test_cancelling_an_open_leaves_no_session(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path, _values = build_external_model(model_dir, elements=LARGE_TENSOR_ELEMENTS)
    with ApplicationService() as service:
        job = service.open_model_async(path)
        job.cancel()
        job.wait(timeout=10)
        assert job.state.is_terminal
        if job.state.value == "cancelled":
            assert service.open_sessions == ()
            with pytest.raises(OperationCancelled):
                job.result(timeout=1)
        else:
            # The import won the race; the session must then be fully usable.
            assert job.result(timeout=1).document.main_graph.nodes


@pytest.mark.skipif(not SMOKE, reason="set NNEDITOR_SMOKE=1 to run boot smokes")
def test_web_server_boot_smoke() -> None:
    """The Flet web server boots and serves the app shell."""
    port = 8571
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import flet as ft\n"
            "import uvicorn\n"
            "from nneditor.ui.app import main\n"
            "app = ft.run(main, export_asgi_app=True)\n"
            f"uvicorn.run(app, host='127.0.0.1', port={port})\n",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=2
                ) as response:
                    assert response.status == 200
                    return
            except Exception as error:
                last_error = error
                time.sleep(1.0)
        raise AssertionError(f"web server never answered: {last_error}")
    finally:
        process.kill()
