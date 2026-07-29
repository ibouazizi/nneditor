"""Background jobs with cancellation and observable state (task P1.5).

The UI must never block on imports, layouts, or reads, so anything slower
than a frame runs as a :class:`Job`. Jobs are plain threads behind an
executor: Flet event handlers already run off the UI loop, and none of the
current workloads release meaningful work to more than a few cores. Process
isolation arrives with the Phase 8 worker split; the *interface* — submit,
observe, cancel — is what the rest of the code binds to.

State changes are pushed to an optional listener rather than polled, so the
shell can update a status bar without a timer.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from typing import Any

from nneditor.cancellation import CancellationToken, OperationCancelled

__all__ = ["Job", "JobManager", "JobState"]

JobListener = Callable[["Job[Any]"], None]


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


class Job[T]:
    """One unit of background work and its observable lifecycle."""

    __slots__ = (
        "_done",
        "_error",
        "_future",
        "_listener",
        "_lock",
        "_result",
        "_state",
        "id",
        "name",
        "token",
    )

    def __init__(self, job_id: int, name: str, listener: JobListener | None) -> None:
        self.id = job_id
        self.name = name
        self.token = CancellationToken()
        self._state = JobState.PENDING
        self._result: T | None = None
        self._error: BaseException | None = None
        self._future: Future[None] | None = None
        self._listener = listener
        self._lock = threading.Lock()
        self._done = threading.Event()

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def error(self) -> BaseException | None:
        return self._error

    def result(self, timeout: float | None = None) -> T:
        """Wait for completion and return the result or raise the failure."""
        if not self._done.wait(timeout):
            raise TimeoutError(f"job {self.name!r} did not finish in time")
        if self._state is JobState.SUCCEEDED:
            assert self._result is not None or True
            return self._result  # type: ignore[return-value]
        if self._state is JobState.CANCELLED:
            raise OperationCancelled
        assert self._error is not None
        raise self._error

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for completion without consuming the result."""
        return self._done.wait(timeout)

    def cancel(self) -> None:
        """Request cancellation; the job stops at its next checkpoint."""
        self.token.cancel()

    # -- manager-side transitions -----------------------------------------

    def _transition(self, state: JobState) -> None:
        with self._lock:
            if self._state.is_terminal:
                return
            self._state = state
        if state.is_terminal:
            self._done.set()
        if self._listener is not None:
            try:
                self._listener(self)
            except Exception:
                # A raising listener would otherwise skip the work entirely
                # and leave the job in RUNNING forever; state transitions are
                # authoritative, notification is best-effort.
                pass

    def _run(self, work: Callable[[CancellationToken], T]) -> None:
        try:
            if self.token.cancelled:
                self._transition(JobState.CANCELLED)
                return
            self._transition(JobState.RUNNING)
            result = work(self.token)
        except OperationCancelled:
            self._transition(JobState.CANCELLED)
        except BaseException as error:
            self._error = error
            self._transition(JobState.FAILED)
        else:
            self._result = result
            self._transition(JobState.SUCCEEDED)


class JobManager:
    """Owns the worker pool and every job submitted through it."""

    __slots__ = (
        "_closed",
        "_counter",
        "_executor",
        "_jobs",
        "_listener",
        "_lock",
        "_retention",
    )

    def __init__(
        self,
        *,
        max_workers: int = 4,
        listener: JobListener | None = None,
        retained_jobs: int = 64,
    ) -> None:
        if retained_jobs < 0:
            raise ValueError("retained_jobs must be non-negative")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="nneditor-job"
        )
        self._jobs: list[Job[Any]] = []
        self._counter = itertools.count(1)
        self._listener = listener
        self._lock = threading.Lock()
        self._closed = False
        self._retention = retained_jobs

    def submit[T](self, name: str, work: Callable[[CancellationToken], T]) -> Job[T]:
        """Schedule ``work``; it receives the job's cancellation token."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the job manager is closed")
            self._prune_terminal_locked()
            job: Job[T] = Job(next(self._counter), name, self._listener)
            try:
                # Submitting under the lock makes the closed check and the
                # executor hand-off atomic against close(); a shutdown racing
                # in anyway surfaces as the same user-facing error.
                job._future = self._executor.submit(job._run, work)
            except RuntimeError as error:
                raise RuntimeError("the job manager is closed") from error
            self._jobs.append(job)
        return job

    def _prune_terminal_locked(self) -> None:
        """Drop the oldest terminal jobs beyond the retention cap."""
        excess = sum(1 for job in self._jobs if job.state.is_terminal) - self._retention
        if excess <= 0:
            return
        kept: list[Job[Any]] = []
        for job in self._jobs:
            if excess > 0 and job.state.is_terminal:
                excess -= 1
                continue
            kept.append(job)
        self._jobs = kept

    @property
    def active_jobs(self) -> tuple[Job[Any], ...]:
        """Jobs that have not reached a terminal state."""
        with self._lock:
            return tuple(job for job in self._jobs if not job.state.is_terminal)

    def cancel_all(self) -> None:
        for job in self.active_jobs:
            job.cancel()

    def close(self, *, wait: bool = True) -> None:
        """Cancel outstanding work and shut the pool down."""
        with self._lock:
            self._closed = True
        self.cancel_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)
        # Queued-but-never-started futures skip _run entirely; without this
        # sweep their jobs would report PENDING forever. The transitions run
        # after the lock is released: they invoke listeners, and a listener
        # calling back into the manager must not deadlock.
        with self._lock:
            stuck = [job for job in self._jobs if not job.state.is_terminal]
        for job in stuck:
            job._transition(JobState.CANCELLED)
