"""Long jobs run against captured snapshots, never under session locks.

Statistics passes and trace-model exports capture the document and revision
id briefly under the session lock, stream without any lock, and re-acquire
only to commit. These tests pin the observable contract: a mid-pass commit
discards the stale result, UI reads never wait behind a pass, the trace
export serves the captured revision's bytes, and closing the session under
an in-flight pass fails the job instead of deadlocking.
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path
from typing import Any

import pytest

import nneditor.application.session as session_module
from nneditor.application.jobs import JobState
from nneditor.application.session import ApplicationService, SessionError
from nneditor.cancellation import CancellationToken
from nneditor.storage.store import TensorStore, TensorUnavailableError
from nneditor.tracing.contracts import TraceApproval, TraceLimits, TraceRequest
from tests.fixtures.onnx_models import build_embedded_model

ELEMENTS = 32

FIVE = struct.pack("<f", 5.0)
NINE = struct.pack("<f", 9.0)


def _gated_store_read(entered: threading.Event, release: threading.Event) -> Any:
    """A ``TensorStore.read`` wrapper that parks the first caller on events."""
    original = TensorStore.read

    def gated(
        store: TensorStore,
        tensor_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
        token: CancellationToken | None = None,
    ) -> bytes:
        entered.set()
        assert release.wait(timeout=10), "the test must release the pass"
        return original(store, tensor_id, offset=offset, length=length, token=token)

    return gated


def test_mid_pass_commit_discards_stale_statistics_and_rerun_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        session.editing.replace_bytes(weight_id, 0, FIVE)
        first_revision = session.editing.current_revision_id
        assert first_revision is not None

        # Deterministic interleave: the first chunk read commits a second
        # revision from inside the pass, so the pass provably straddles a
        # revision change without any sleeping or thread races.
        original_read = session_module._EditedBytesView.read
        state = {"committed": False}

        def commit_mid_pass(
            view: object,
            tensor_id: str,
            *,
            offset: int = 0,
            length: int | None = None,
            token: CancellationToken | None = None,
        ) -> bytes:
            data = original_read(
                view,  # type: ignore[arg-type]
                tensor_id,
                offset=offset,
                length=length,
                token=token,
            )
            if not state["committed"]:
                state["committed"] = True
                session.editing.replace_bytes(weight_id, 0, NINE)
            return data

        monkeypatch.setattr(session_module._EditedBytesView, "read", commit_mid_pass)

        with pytest.raises(SessionError, match="stale"):
            session.compute_statistics(weight_id)
        assert session._edited_stats.get((first_revision, weight_id)) is None, (
            "the stale result must not be cached under the captured revision"
        )
        assert session.statistics(weight_id) is None, (
            "nothing may be cached for the new revision either"
        )

        rerun = session.compute_statistics(weight_id)
        assert rerun.element_count == ELEMENTS
        assert session.statistics(weight_id) is rerun, (
            "the rerun on the new revision commits normally"
        )


def test_document_and_scene_reads_are_not_blocked_by_a_statistics_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(TensorStore, "read", _gated_store_read(entered, release))

        job = session.compute_statistics_async(weight_id)
        try:
            assert entered.wait(timeout=10), "the pass never reached its first chunk"
            finished = threading.Event()

            def read_ui_state() -> None:
                assert session.document.main_graph.nodes
                assert session.scene().scene.node_count == 2
                finished.set()

            reader = threading.Thread(target=read_ui_state, daemon=True)
            reader.start()
            assert finished.wait(timeout=5), (
                "document/scene reads must not wait behind a streaming pass"
            )
            reader.join(timeout=5)
        finally:
            release.set()
        stats = job.result(timeout=30)
        assert stats.element_count == ELEMENTS


def test_trace_export_holds_no_lock_and_serves_the_captured_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        session.editing.replace_bytes(weight_id, 0, FIVE)
        first_revision = session.editing.current_revision_id

        specification = session.default_trace_inputs(seed=29)
        limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=1024 * 1024 * 1024,
            capture_bytes=1024 * 1024,
            chunk_bytes=64,
        )
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            specification,
            limits,
        )
        request = TraceRequest(specification, limits, approval)

        # The worker subprocess is out of scope here: the runner boundary is
        # replaced so the test exercises exactly the export-under-no-lock
        # property that session.py owns.
        entered = threading.Event()
        release = threading.Event()
        exported: list[bytes] = []
        keys: list[Any] = []
        sentinel = object()

        def gated_export(
            document: object,
            revisions: object,
            destination: object,
            *,
            tensor_reader: Any,
            token: object = None,
            **kwargs: object,
        ) -> None:
            entered.set()
            assert release.wait(timeout=10), "the test must release the export"
            # The reader is pinned to the captured revisions; after the
            # mid-export commit below it must still serve the snapshot.
            exported.append(tensor_reader(weight_id))

        def fake_run(
            document: object,
            key: object,
            req: object,
            store: object,
            build_model: Any,
            *,
            token: CancellationToken,
        ) -> object:
            keys.append(key)
            build_model(tmp_path / "trace-model.onnx", token)
            return sentinel

        monkeypatch.setattr(session_module, "export_revision", gated_export)
        monkeypatch.setattr(session_module, "run_onnx_trace", fake_run)

        job = session.trace_async(request)
        try:
            assert entered.wait(timeout=10), "the export never started"
            finished = threading.Event()

            def read_and_commit() -> None:
                assert session.document.main_graph.nodes
                assert session.scene().scene.node_count == 2
                session.editing.replace_bytes(weight_id, 0, NINE)
                finished.set()

            worker = threading.Thread(target=read_and_commit, daemon=True)
            worker.start()
            assert finished.wait(timeout=5), (
                "reads and commits must not wait behind the trace export"
            )
            worker.join(timeout=5)
        finally:
            release.set()
        assert job.result(timeout=10) is sentinel
        assert exported[0][:4] == FIVE, (
            "the export must serve the captured revision, not the newer commit"
        )
        assert keys[0].revision_id == first_revision, (
            "the trace result stays keyed by the captured revision id"
        )


def test_closing_the_session_during_a_pass_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(TensorStore, "read", _gated_store_read(entered, release))

        job = session.compute_statistics_async(weight_id)
        try:
            assert entered.wait(timeout=10), "the pass never reached its first chunk"
            closed = threading.Event()

            def close_session() -> None:
                session.close()
                closed.set()

            closer = threading.Thread(target=close_session, daemon=True)
            closer.start()
            assert closed.wait(timeout=5), "close() must not wait behind the pass"
            closer.join(timeout=5)
        finally:
            release.set()
        assert job.wait(timeout=10), "the interrupted job must reach a terminal state"
        assert job.state in (JobState.FAILED, JobState.CANCELLED)
        if job.state is JobState.FAILED:
            assert isinstance(job.error, TensorUnavailableError), (
                "a closed-store read fails through the ordinary job-error path"
            )
