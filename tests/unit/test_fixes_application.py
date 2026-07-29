"""Regression tests for verified application-layer defects.

Each test pins one fix in ``nneditor.application``: job lifecycle robustness,
revision-aware layout caching, sidecar write coordination on Windows,
non-destructive sidecar failure handling, manual-hierarchy collision safety,
persistence-failure reporting, and session lifecycle hygiene.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.analysis.detectors import DetectionPipeline
from nneditor.analysis.layout import layout_graph
from nneditor.application.editing import EditingController, SidecarPersistenceError
from nneditor.application.edits_store import RevisionSidecarStore
from nneditor.application.hierarchy import HierarchyController, HierarchySidecarStore
from nneditor.application.jobs import Job, JobManager, JobState
from nneditor.application.navigation import Direction, NavigationModel
from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.application.slices import GraphSlicer
from nneditor.application.statistics_store import StatisticsSidecarStore
from nneditor.cancellation import CancellationToken
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import ArtifactRef, Document, Graph, Node, Value
from nneditor.ir.identity import NodeIdStability
from nneditor.storage.store import TensorStore
from tests.fixtures.onnx_models import build_embedded_model

# -- shared builders --------------------------------------------------------


def make_document(graph: Graph) -> Document:
    return Document(
        source=ArtifactRef(
            path="model.onnx",
            content_hash="sha256:" + "1" * 64,
            byte_size=100,
        ),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=contract_for(ArtifactKind.ONNX_MODEL).statuses,
        graphs=(graph,),
        entry_graph=graph.id,
    )


def linear_graph(ops: tuple[str, ...], *, graph_id: str = "g:main") -> Graph:
    values = tuple(
        Value(id=f"v{index}", name=f"v{index}", element_type="float32", shape=(4, 4))
        for index in range(len(ops))
    )
    nodes = tuple(
        Node(
            id=f"n{index}",
            id_stability=NodeIdStability.NAMED,
            op_type=op,
            inputs=() if index == 0 else (f"v{index - 1}",),
            outputs=(f"v{index}",),
        )
        for index, op in enumerate(ops)
    )
    return Graph(id=graph_id, name="main", nodes=nodes, values=values)


def embedded_document(tmp_path: Path) -> tuple[Document, np.ndarray]:
    path = tmp_path / "model.onnx"
    values = build_embedded_model(path, elements=32)
    return index_to_document(index_model(path)), values


def run_concurrently(
    worker: Callable[[], None], *, threads: int = 4
) -> list[BaseException]:
    errors: list[BaseException] = []

    def call() -> None:
        try:
            worker()
        except BaseException as error:
            errors.append(error)

    pool = [threading.Thread(target=call, daemon=True) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(timeout=30)
    return errors


class FailingSidecar:
    """A revision sidecar whose save always fails (full disk, locked file)."""

    def load(self, content_hash: str) -> None:
        return None

    def save(self, content_hash: str, revisions: list[object], cursor: int) -> None:
        raise OSError("disk full")


# -- finding 1: a raising listener must not wedge a job ---------------------


def test_a_listener_raising_on_running_does_not_wedge_the_job() -> None:
    def listener(job: Job[object]) -> None:
        if job.state is JobState.RUNNING:
            raise RuntimeError("listener bug")

    manager = JobManager(listener=listener)
    try:
        job = manager.submit("work", lambda token: 21 * 2)
        assert job.result(timeout=5) == 42
        assert job.state is JobState.SUCCEEDED
    finally:
        manager.close()


# -- finding 2: close() must not hold the lock across listeners -------------


def test_close_with_a_reentrant_listener_does_not_deadlock() -> None:
    holder: dict[str, JobManager] = {}
    reentered: list[int] = []

    def listener(job: Job[object]) -> None:
        manager = holder.get("manager")
        if manager is not None and job.state is JobState.CANCELLED:
            reentered.append(len(manager.active_jobs))

    manager = JobManager(max_workers=1, listener=listener)
    holder["manager"] = manager
    started = threading.Event()
    release = threading.Event()

    def blocked(token: CancellationToken) -> None:
        started.set()
        release.wait(timeout=5)
        token.raise_if_cancelled()

    try:
        manager.submit("running", blocked)
        queued = manager.submit("queued", lambda token: None)
        assert started.wait(timeout=5)
        closer = threading.Thread(target=lambda: manager.close(wait=False), daemon=True)
        closer.start()
        closer.join(timeout=10)
        assert not closer.is_alive(), "close() deadlocked with a reentrant listener"
        assert queued.state is JobState.CANCELLED
        assert reentered, "the listener observed the close-time cancellation"
    finally:
        release.set()
        manager.close()


def test_terminal_jobs_are_pruned_beyond_the_retention_cap() -> None:
    manager = JobManager(max_workers=2, retained_jobs=5)
    try:
        for index in range(20):
            manager.submit(f"job-{index}", lambda token: None).result(timeout=5)
        manager.submit("final", lambda token: None).result(timeout=5)
        assert len(manager._jobs) <= 6, "terminal jobs must not accumulate"
    finally:
        manager.close()


# -- finding 3: layout caches must discriminate editing revisions -----------


def test_slice_cache_discriminates_editing_revisions(tmp_path: Path) -> None:
    document, _values = embedded_document(tmp_path)
    slicer = GraphSlicer()
    base = slicer.slice_graph(document)
    edited = slicer.slice_graph(document, revision_id="rev-1")
    assert edited is not base, "revisions must not share slice cache entries"
    assert slicer.slice_graph(document) is base
    assert slicer.slice_graph(document, revision_id="rev-1") is edited
    snapshots = slicer.cache_snapshots()
    assert snapshots[1].entry_count == 2, "base layouts are revision-keyed too"


def test_sessions_at_different_revisions_stop_sharing_scenes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService() as service:
        first = service.open_model(path)
        second = service.open_model(path)
        assert first.scene() is second.scene(), "same revision still shares"
        weight_id = second.document.main_graph.initializers[0]
        second.editing.replace_bytes(weight_id, 0, b"\x01\x02\x03\x04")
        assert second.editing.current_revision_id is not None
        assert first.scene() is not second.scene(), (
            "an edited session must not serve its scene to a base session"
        )
        assert second.scene() is second.scene(), "the edited key is stable"


# -- finding 4: invalidate must honour the slicer lock ----------------------


def test_invalidate_waits_for_an_in_flight_slice() -> None:
    slicer = GraphSlicer()
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with slicer._lock:
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert entered.wait(timeout=5)
    finished = threading.Event()

    def drop() -> None:
        slicer.invalidate()
        finished.set()

    dropper = threading.Thread(target=drop, daemon=True)
    dropper.start()
    assert not finished.wait(timeout=0.25), "invalidate must wait for the lock"
    release.set()
    assert finished.wait(timeout=5)
    holder.join(timeout=5)
    dropper.join(timeout=5)


# -- finding 5: concurrent sidecar saves must not collide -------------------


def test_concurrent_session_state_saves_do_not_collide(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path)
    store.state.record_open("model.onnx", "sha256:" + "a" * 64, "model")

    def worker() -> None:
        for _ in range(10):
            store.save()

    assert run_concurrently(worker) == []
    assert SessionStateStore(tmp_path).state.recent
    assert not list(tmp_path.glob("*.partial"))


def test_concurrent_statistics_saves_do_not_collide(tmp_path: Path) -> None:
    store = StatisticsSidecarStore(tmp_path)
    content = "sha256:" + "b" * 64

    def worker() -> None:
        for _ in range(10):
            store.save(content, {})

    assert run_concurrently(worker) == []
    assert store.load(content) == {}
    assert not list(store.directory.glob("*.partial"))


def test_concurrent_revision_saves_do_not_collide(tmp_path: Path) -> None:
    store = RevisionSidecarStore(tmp_path)
    content = "sha256:" + "c" * 64

    def worker() -> None:
        for _ in range(10):
            store.save(content, [], 0)

    assert run_concurrently(worker) == []
    assert store.load(content) == ([], 0)
    assert not list(store.directory.glob("*.partial"))


def test_hierarchy_store_save_and_for_artifact_do_not_race(
    tmp_path: Path,
) -> None:
    store = HierarchySidecarStore(tmp_path)
    counter = itertools.count()

    def worker() -> None:
        for _ in range(10):
            store.for_artifact(f"sha256:{next(counter):064d}")
            store.save()

    errors = run_concurrently(worker)
    assert errors == []
    assert not list(tmp_path.glob("*.partial"))


# -- finding 6: sidecar load failures must never destroy the chain ----------


def test_transient_read_errors_do_not_quarantine_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RevisionSidecarStore(tmp_path)
    content = "sha256:" + "d" * 64
    store.save(content, [], 0)
    target = store._path_for(content)
    original = Path.read_text

    def flaky(self: Path, *args: object, **kwargs: object) -> str:
        if self == target:
            raise PermissionError("sharing violation")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", flaky)
    assert store.load(content) is None, "a transient failure reads as absent"
    monkeypatch.undo()
    assert target.exists(), "the user's chain must survive a transient error"
    assert list(target.parent.glob("*.corrupt")) == []
    assert store.load(content) == ([], 0), "the chain loads once the error clears"


def test_corrupt_chain_is_quarantined_by_rename_not_deletion(
    tmp_path: Path,
) -> None:
    store = RevisionSidecarStore(tmp_path)
    content = "sha256:" + "e" * 64
    target = store._path_for(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{broken", encoding="utf-8")
    assert store.load(content) is None
    assert not target.exists()
    corrupt = target.with_name(target.name + ".corrupt")
    assert corrupt.read_text(encoding="utf-8") == "{broken"


def test_failed_quarantine_neither_deletes_nor_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RevisionSidecarStore(tmp_path)
    content = "sha256:" + "f" * 64
    target = store._path_for(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{broken", encoding="utf-8")
    original_replace = os.replace

    def failing_replace(src: object, dst: object, **kwargs: object) -> None:
        if str(dst).endswith(".corrupt"):
            raise PermissionError("locked by a scanner")
        original_replace(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("nneditor.application.edits_store.os.replace", failing_replace)
    assert store.load(content) is None, "a failing quarantine must not raise"
    assert target.exists(), "the fallback must never delete the user's chain"


# -- finding 7: manual-group collisions must fail without destruction -------


class TestManualGroupCollisions:
    @staticmethod
    def controller() -> tuple[Graph, HierarchyController]:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm", "Mul", "Sub"))
        return graph, HierarchyController(
            make_document(graph), pipeline=DetectionPipeline(())
        )

    def test_rename_collision_preserves_the_group(self) -> None:
        graph, controller = self.controller()
        alpha = controller.group(graph.id, "Alpha", frozenset(("n0", "n1")))
        controller.group(graph.id, "Beta", frozenset(("n2", "n3")))
        with pytest.raises(ValueError, match="already used"):
            controller.rename(graph.id, alpha.id, "Beta")
        hierarchy = controller.hierarchy(graph.id)
        assert {group.label for group in hierarchy.groups} == {"Alpha", "Beta"}
        assert hierarchy.group(alpha.id).members == {"n0", "n1"}

    def test_rename_to_the_same_label_is_allowed(self) -> None:
        graph, controller = self.controller()
        alpha = controller.group(graph.id, "Alpha", frozenset(("n0", "n1")))
        assert controller.rename(graph.id, alpha.id, "Alpha").label == "Alpha"

    def test_merge_does_not_overwrite_an_existing_label(self) -> None:
        graph, controller = self.controller()
        left = controller.group(graph.id, "Left", frozenset(("n0", "n1")))
        right = controller.group(graph.id, "Right", frozenset(("n2", "n3")))
        taken = controller.group(graph.id, "Taken", frozenset(("n4", "n5")))
        with pytest.raises(ValueError, match="already used"):
            controller.merge(graph.id, (left.id, right.id), "Taken")
        hierarchy = controller.hierarchy(graph.id)
        assert hierarchy.group(taken.id).members == {"n4", "n5"}
        assert hierarchy.group(left.id).members == {"n0", "n1"}
        assert hierarchy.group(right.id).members == {"n2", "n3"}

    def test_merge_may_reuse_a_consumed_groups_label(self) -> None:
        graph, controller = self.controller()
        left = controller.group(graph.id, "Left", frozenset(("n0", "n1")))
        right = controller.group(graph.id, "Right", frozenset(("n2", "n3")))
        merged = controller.merge(graph.id, (left.id, right.id), "Left")
        assert merged.members == {"n0", "n1", "n2", "n3"}

    def test_split_does_not_overwrite_an_existing_label(self) -> None:
        graph, controller = self.controller()
        block = controller.group(graph.id, "Block", frozenset(("n0", "n1", "n2", "n3")))
        controller.group(graph.id, "Block 1", frozenset(("n4", "n5")))
        with pytest.raises(ValueError, match="already used"):
            controller.split(
                graph.id,
                block.id,
                (frozenset(("n0", "n1")), frozenset(("n2", "n3"))),
            )
        hierarchy = controller.hierarchy(graph.id)
        assert hierarchy.group(block.id).members == {"n0", "n1", "n2", "n3"}
        assert {group.label for group in hierarchy.groups} == {"Block", "Block 1"}


# -- finding 8: persistence failure must not masquerade as a failed edit ----


def test_persistence_failure_reports_but_keeps_the_commit(tmp_path: Path) -> None:
    document, _values = embedded_document(tmp_path)
    weight_id = document.main_graph.initializers[0]
    with TensorStore(document) as store:
        controller = EditingController(
            document,
            store,
            sidecar=FailingSidecar(),  # type: ignore[arg-type]
        )
        with pytest.raises(SidecarPersistenceError) as info:
            controller.replace_bytes(weight_id, 0, b"\x01\x02\x03\x04")
        assert controller.is_dirty, "the revision applied despite the warning"
        assert controller.applied_revisions[-1] is info.value.revision
        assert controller.read(weight_id)[:4] == b"\x01\x02\x03\x04"


def test_session_refreshes_caches_when_only_persistence_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService(
        edits_store=FailingSidecar()  # type: ignore[arg-type]
    ) as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        with pytest.raises(SidecarPersistenceError):
            session.editing.replace_bytes(weight_id, 0, b"\x09\x08\x07\x06")
        before = session.hierarchy
        with pytest.raises(SidecarPersistenceError):
            session.undo_edit()
        assert not session.editing.can_undo, "the undo itself applied"
        assert session.editing.can_redo
        assert session.hierarchy is not before, (
            "caches must refresh even when only persistence failed"
        )


# -- finding 9: a failed session init must not leak the tensor store --------


def test_model_session_init_failure_closes_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _values = embedded_document(tmp_path)
    created: list[TensorStore] = []

    class RecordingStore(TensorStore):
        def __init__(self, doc: Document) -> None:
            super().__init__(doc)
            created.append(self)

    class ExplodingSidecar:
        def load(self, content_hash: str) -> None:
            raise RuntimeError("recovery boom")

    monkeypatch.setattr("nneditor.application.session.TensorStore", RecordingStore)
    with pytest.raises(RuntimeError, match="recovery boom"):
        ModelSession(
            1,
            document,
            GraphSlicer(),
            HierarchyController(document, pipeline=DetectionPipeline(())),
            edits_sidecar=ExplodingSidecar(),  # type: ignore[arg-type]
        )
    assert len(created) == 1
    assert created[0].closed, "the store must close when init fails"


# -- finding 11: stale selections navigate to nowhere, not to a crash -------


def test_directional_with_a_stale_selection_returns_none() -> None:
    scene = layout_graph(linear_graph(("Conv", "Relu", "Add"))).scene
    model = NavigationModel()
    assert model.directional(scene, "ghost", Direction.DOWN) is None
    assert model.directional(scene, "n0", Direction.DOWN) == "n1"


# -- finding 12: statistics must reflect committed edits --------------------


def test_statistics_follow_committed_edits(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        base = session.compute_statistics(weight_id)
        session.editing.replace_bytes(
            weight_id, 0, np.full(32, 7.0, dtype=np.float32).tobytes()
        )
        edited = session.compute_statistics(weight_id)
        assert edited is not base, "edited bytes must not reuse base statistics"
        assert edited.minimum == pytest.approx(7.0)
        assert edited.maximum == pytest.approx(7.0)
        assert session.statistics(weight_id) is edited
        session.editing.undo()
        assert session.compute_statistics(weight_id) is base, (
            "back at base, the shared base statistics apply again"
        )


# -- finding 13: shutdown must be complete, not fail-fast -------------------


def test_service_close_survives_a_failing_session(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    service = ApplicationService()

    class ExplodingSession:
        closed = False

        def close(self) -> None:
            raise RuntimeError("cannot close")

    service._sessions[0] = ExplodingSession()  # type: ignore[assignment]
    session = service.open_model(path)
    upload_root = service.stage_upload("probe.onnx", b"x").parent.parent
    service.close()
    assert session.closed, "later sessions must still close"
    assert not upload_root.exists(), "uploads must be cleaned up regardless"
