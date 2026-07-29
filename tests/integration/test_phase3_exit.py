"""Phase 3 exit gate and cross-cutting storage/revision hardening (P3.8).

The gate: inspect slices and statistics lazily, apply one copy-on-write
weight replacement, preview the diff, and undo/redo it — without altering the
source artifact or exceeding the configured cache budget.
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import pytest

from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.storage.reader import hash_file
from nneditor.storage.store import TensorStore
from tests.fixtures.onnx_models import (
    LARGE_TENSOR_ELEMENTS,
    build_embedded_model,
    build_external_model,
)
from tests.unit.test_tensor_store import embedded_document

ELEMENTS = 256


def test_phase3_exit_gate(tmp_path: Path) -> None:
    """The whole cycle, end to end, against an external-weights model."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path, values = build_external_model(model_dir, elements=ELEMENTS)
    weights_file = model_dir / "weights.bin"
    source_hashes = (hash_file(path), hash_file(weights_file))
    state_dir = tmp_path / "state"

    with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]

        # Lazy inspection: metadata and disclosure read nothing; the first
        # slice read opens exactly one file; statistics stream on a job.
        assert session.store.open_file_count == 0
        assert session.store.materialization(weight_id).value == "range"
        preview = session.store.read(weight_id, offset=0, length=32)
        assert preview == values.tobytes()[:32]
        assert session.store.open_file_count == 1
        stats = session.compute_statistics_async(weight_id).result(timeout=30)
        assert stats.element_count == ELEMENTS

        # One copy-on-write weight replacement.
        new_value = struct.pack("<f", 123.5)
        session.editing.replace_bytes(weight_id, 8, new_value)
        edited = session.editing.read(weight_id)
        assert edited[8:12] == new_value
        assert edited[:8] == values.tobytes()[:8]

        # The diff preview names the change at element level.
        diff = session.editing.preview()
        (tensor_diff,) = diff.tensors
        assert tensor_diff.tensor_id == weight_id
        assert tensor_diff.elements_changed == 1
        (span,) = tensor_diff.spans
        assert span.after_values == (123.5,)

        # Undo and redo are deterministic and reversible.
        session.editing.undo()
        assert session.editing.read(weight_id) == values.tobytes()
        session.editing.redo()
        assert session.editing.read(weight_id)[8:12] == new_value

        # The source artifacts are byte-identical throughout.
        assert (hash_file(path), hash_file(weights_file)) == source_hashes

        # Every cache respects its configured budget.
        for snapshot in session.store.cache_snapshots():
            assert snapshot.current_cost <= snapshot.budget

    # A fresh service recovers the committed edit from the sidecar alone.
    with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        assert session.editing.is_dirty
        assert session.editing.read(weight_id)[8:12] == new_value
        # The persisted base statistics survived the restart, but they no
        # longer masquerade as summaries of the *edited* bytes; they resurface
        # as soon as the chain returns to base.
        assert session.statistics(weight_id) is None, "edited bytes have no stats yet"
        session.editing.undo()
        assert session.statistics(weight_id) is not None, "stats persisted too"
        session.editing.redo()
        assert (hash_file(path), hash_file(weights_file)) == source_hashes


def test_concurrent_reads_are_consistent_under_cache_pressure(
    tmp_path: Path,
) -> None:
    document, values = embedded_document(tmp_path)
    weight_id = document.main_graph.initializers[0]
    expected = values.tobytes()
    errors: list[BaseException] = []

    with TensorStore(document, cache_budget=512) as store:

        def worker(worker_id: int) -> None:
            try:
                for index in range(50):
                    offset = ((worker_id * 50 + index) % 32) * 8
                    data = store.read(weight_id, offset=offset, length=8)
                    assert data == expected[offset : offset + 8]
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        (slices, _typed) = store.cache_snapshots()
        assert slices.current_cost <= 512, "the memory ceiling held"
        assert slices.hits > 0


def test_mid_stream_cancellation_persists_nothing(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=LARGE_TENSOR_ELEMENTS)
    state_dir = tmp_path / "state"
    with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
        session = service.open_model(model)
        weight_id = session.document.main_graph.initializers[0]
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            session.compute_statistics(weight_id, token=token)
        assert session.statistics(weight_id) is None

    with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
        session = service.open_model(model)
        weight_id = session.document.main_graph.initializers[0]
        assert session.statistics(weight_id) is None, "nothing was persisted"


def test_editing_a_typed_tensor_goes_through_materialization(
    tmp_path: Path,
) -> None:
    """The COW overlay composes with full-parse tensors transparently."""
    document, _values = embedded_document(tmp_path)
    bias_id = document.main_graph.initializers[1]
    with ApplicationService() as service:
        session = service.open_model(Path(document.source.path))
        bias_id = session.document.main_graph.initializers[1]
        replacement = struct.pack("<f", -9.5)
        session.editing.replace_bytes(bias_id, 0, replacement)
        assert session.editing.read(bias_id) == replacement
        session.editing.undo()
        assert session.editing.read(bias_id) == struct.pack("<f", 0.5)


def test_undo_chain_depth(tmp_path: Path) -> None:
    """A long alternating edit history stays exact through full unwind."""
    document, values = embedded_document(tmp_path)
    with ApplicationService() as service:
        session = service.open_model(Path(document.source.path))
        weight_id = session.document.main_graph.initializers[0]
        for step in range(20):
            session.editing.replace_bytes(weight_id, step * 4, bytes([step + 1] * 4))
        assert len(session.editing.applied_revisions) == 20
        for _ in range(20):
            session.editing.undo()
        assert session.editing.read(weight_id) == values.tobytes()
        for _ in range(20):
            session.editing.redo()
        view = session.editing.read(weight_id)
        for step in range(20):
            assert view[step * 4 : step * 4 + 4] == bytes([step + 1] * 4)
