"""Cooperative cancellation (task P1.4).

Long-running work — tensor reads, statistics, layout — periodically asks its
token whether to continue. Cancellation is cooperative because the work runs
in the same process as the UI: there is nothing safe to kill, only loops that
agree to stop at their next checkpoint.
"""

from __future__ import annotations

import threading

__all__ = ["CancellationToken", "OperationCancelled"]


class OperationCancelled(Exception):
    """Raised at a checkpoint after the token was cancelled."""


class CancellationToken:
    """A thread-safe, one-way cancellation flag.

    Once cancelled, a token stays cancelled: work that observes the flag late
    must still stop, or a caller who cancelled during a cache hit would see
    the next expensive step run anyway.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request that work holding this token stop at its next checkpoint."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Checkpoint: raise :class:`OperationCancelled` when cancelled."""
        if self._event.is_set():
            raise OperationCancelled
