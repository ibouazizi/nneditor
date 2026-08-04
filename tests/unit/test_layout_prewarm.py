"""First-switch latency: deeper detail levels are prewarmed after open.

The first switch to LAYER or OPERATOR on a large model pays the full
base-layout-plus-semantic-slice cost cold, past the interaction budget. The
application therefore schedules one background job after a session opens that
walks BLOCK -> LAYER -> OPERATOR for the entry graph through the same slicer
path :meth:`ModelSession.scene` uses. These tests pin the observable
contract: the walk populates exactly the keys the shell later requests
(cache hits, identical cached objects), disabling the flag creates no job at
all, and closing the session with the walk mid-flight cancels it cleanly
without deadlock.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from nneditor.analysis.lod import DetailLevel
from nneditor.application.jobs import JobState
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.application.slices import GraphSlicer
from tests.fixtures.onnx_models import build_embedded_model

ELEMENTS = 32


def _open_model(tmp_path: Path, service: ApplicationService) -> ModelSession:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    return service.open_model(path)


def test_prewarm_makes_first_deep_scene_requests_cache_hits(tmp_path: Path) -> None:
    with ApplicationService() as service:
        session = _open_model(tmp_path, service)
        job = session._prewarm_job
        assert job is not None, "opening must schedule the prewarm job"
        assert "prewarm layouts" in job.name
        job.result(timeout=30)

        semantic_before, base_before = session._slicer.cache_snapshots()
        layer_slice = session.scene(detail_level=DetailLevel.LAYER)
        operator_slice = session.scene(detail_level=DetailLevel.OPERATOR)
        semantic_after, base_after = session._slicer.cache_snapshots()

        assert layer_slice.detail_level is DetailLevel.LAYER
        assert operator_slice.detail_level is DetailLevel.OPERATOR
        assert semantic_after.hits == semantic_before.hits + 2, (
            "the first LAYER and OPERATOR requests must both be cache hits"
        )
        assert semantic_after.misses == semantic_before.misses, (
            "a prewarm that populates the wrong key would register misses here"
        )
        assert base_after.misses == base_before.misses, (
            "warm semantic slices must not recompute base layouts either"
        )


def test_disabled_prewarm_creates_no_job(tmp_path: Path) -> None:
    observed: list[str] = []
    with ApplicationService(
        prewarm_layouts=False,
        job_listener=lambda job: observed.append(job.name),
    ) as service:
        session = _open_model(tmp_path, service)
        assert session._prewarm_job is None
        assert service.jobs._jobs == [], (
            "a synchronous open with prewarm disabled submits no job at all"
        )
    assert observed == []


def test_closing_the_session_mid_prewarm_cancels_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic parking: the walk's first slicer call blocks on an event
    # while holding no session lock, so close() must be able to run past it.
    entered = threading.Event()
    release = threading.Event()
    original = GraphSlicer.slice_graph

    def gated(slicer: GraphSlicer, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(timeout=10), "the test must release the walk"
        return original(slicer, *args, **kwargs)

    monkeypatch.setattr(GraphSlicer, "slice_graph", gated)
    with ApplicationService() as service:
        session = _open_model(tmp_path, service)
        job = session._prewarm_job
        assert job is not None
        try:
            assert entered.wait(timeout=10), "the walk never reached a level"
            closed = threading.Event()

            def close_session() -> None:
                service.close_session(session.id)
                closed.set()

            closer = threading.Thread(target=close_session, daemon=True)
            closer.start()
            assert closed.wait(timeout=5), "close() must not wait behind the walk"
            closer.join(timeout=5)
        finally:
            release.set()
        assert job.wait(timeout=10), "the abandoned job must reach a terminal state"
        assert job.state is JobState.CANCELLED
        assert job.error is None, "abandonment is cancellation, not failure"


def test_prewarmed_operator_scene_is_the_on_demand_scene(tmp_path: Path) -> None:
    with ApplicationService() as service:
        session = _open_model(tmp_path, service)
        job = session._prewarm_job
        assert job is not None
        warmed = job.result(timeout=30)
        assert [item.detail_level for item in warmed] == [
            DetailLevel.BLOCK,
            DetailLevel.LAYER,
            DetailLevel.OPERATOR,
        ]
        on_demand = session.scene(detail_level=DetailLevel.OPERATOR)
        assert on_demand is warmed[-1], (
            "identity proves the on-demand request was served from the "
            "prewarmed cache entry, i.e. the keys matched exactly"
        )
