"""Tests for background jobs (P1.5)."""

from __future__ import annotations

import threading

import pytest

from nneditor.application.jobs import Job, JobManager, JobState
from nneditor.cancellation import CancellationToken, OperationCancelled


def test_a_job_runs_and_returns_its_result() -> None:
    with_manager = JobManager(max_workers=2)
    try:
        job = with_manager.submit("double", lambda token: 21 * 2)
        assert job.result(timeout=5) == 42
        assert job.state is JobState.SUCCEEDED
        assert job.error is None
    finally:
        with_manager.close()


def test_a_failing_job_reports_its_error() -> None:
    manager = JobManager()
    try:

        def explode(token: CancellationToken) -> None:
            raise RuntimeError("boom")

        job = manager.submit("explode", explode)
        job.wait(timeout=5)
        assert job.state is JobState.FAILED
        with pytest.raises(RuntimeError, match="boom"):
            job.result(timeout=5)
    finally:
        manager.close()


def test_cancellation_stops_work_at_a_checkpoint() -> None:
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()
    try:

        def work(token: CancellationToken) -> str:
            started.set()
            release.wait(timeout=5)
            token.raise_if_cancelled()
            return "never"

        job = manager.submit("slow", work)
        assert started.wait(timeout=5)
        job.cancel()
        release.set()
        job.wait(timeout=5)
        assert job.state is JobState.CANCELLED
        with pytest.raises(OperationCancelled):
            job.result(timeout=5)
    finally:
        manager.close()


def test_a_job_cancelled_before_running_never_starts() -> None:
    manager = JobManager(max_workers=1)
    block = threading.Event()
    try:
        manager.submit("blocker", lambda token: block.wait(timeout=5))
        queued = manager.submit("queued", lambda token: "ran")
        queued.cancel()
        block.set()
        queued.wait(timeout=5)
        assert queued.state is JobState.CANCELLED
    finally:
        manager.close()


def test_listener_observes_state_changes() -> None:
    seen: list[tuple[str, JobState]] = []

    def listener(job: Job[object]) -> None:
        seen.append((job.name, job.state))

    manager = JobManager(listener=listener)
    try:
        job = manager.submit("observed", lambda token: 1)
        job.wait(timeout=5)
    finally:
        manager.close()
    assert ("observed", JobState.RUNNING) in seen
    assert ("observed", JobState.SUCCEEDED) in seen


def test_active_jobs_and_close_semantics() -> None:
    manager = JobManager(max_workers=1)
    block = threading.Event()

    def blocked(token: CancellationToken) -> None:
        block.wait(timeout=5)
        token.raise_if_cancelled()

    running = manager.submit("running", blocked)
    queued = manager.submit("queued", lambda token: None)
    assert {job.name for job in manager.active_jobs} == {"running", "queued"}
    block.set()
    manager.close()
    assert manager.active_jobs == ()
    assert queued.state.is_terminal
    assert running.state.is_terminal
    with pytest.raises(RuntimeError, match="closed"):
        manager.submit("late", lambda token: None)


def test_result_timeout() -> None:
    manager = JobManager()
    block = threading.Event()
    try:
        job = manager.submit("slow", lambda token: block.wait(timeout=5))
        with pytest.raises(TimeoutError):
            job.result(timeout=0.05)
    finally:
        block.set()
        manager.close()
