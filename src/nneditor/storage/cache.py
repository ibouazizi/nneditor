"""Size-bounded caches with observable behavior (task P3.1).

Every cache in the product — tensor slices, typed-tensor materializations,
statistics, layouts — is an instance of one :class:`BudgetedCache`, so budget
enforcement, LRU eviction, thread safety, and the hit/miss/eviction counters
that tests and the diagnostics UI read are implemented exactly once. A cache
differs from its siblings only in its cost function: byte-sized caches charge
``len(value)``, entry-counted caches charge one per entry.

Two properties are deliberate:

* **An oversized value is never admitted by default.** Caching something larger
  than the whole budget would evict everything else for a value that cannot be
  kept anyway; the caller still gets its value, it just is not retained. A
  cache built with ``admit_oversized=True`` opts out: it keeps a single
  oversized entry (displacing everything else) for callers who promised a
  one-time cost that must not silently repeat. When a rejected oversized value
  replaces an existing entry, the old entry's disappearance is counted as an
  eviction rather than hidden.
* **Snapshots are cheap and consistent.** ``snapshot()`` reads every counter
  under the same lock the mutations take, so an observer never sees a
  half-updated cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

__all__ = ["BudgetedCache", "CacheSnapshot", "CacheStats"]


class CacheStats:
    """Mutable observable counters; reset together with the cache."""

    __slots__ = ("evictions", "hits", "misses")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions})"
        )


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """A consistent point-in-time view of one cache, for display and tests."""

    name: str
    hits: int
    misses: int
    evictions: int
    entry_count: int
    current_cost: int
    budget: int


class BudgetedCache[K: Hashable, V]:
    """A thread-safe LRU cache bounded by a total cost budget."""

    __slots__ = (
        "_admit_oversized",
        "_budget",
        "_cost",
        "_current",
        "_entries",
        "_lock",
        "name",
        "stats",
    )

    def __init__(
        self,
        name: str,
        budget: int,
        *,
        cost: Callable[[V], int] | None = None,
        admit_oversized: bool = False,
    ) -> None:
        if budget < 0:
            raise ValueError(f"cache budget must be non-negative, got {budget}")
        self.name = name
        self._budget = budget
        self._cost = cost if cost is not None else (lambda _value: 1)
        self._admit_oversized = admit_oversized
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._current = 0
        self._lock = threading.RLock()
        self.stats = CacheStats()

    @property
    def budget(self) -> int:
        return self._budget

    def get(self, key: K) -> V | None:
        """The cached value, counting a hit or miss."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return entry[0]

    def peek(self, key: K) -> V | None:
        """The cached value without touching the hit/miss counters.

        For double-checked locking: a caller that already counted its miss
        re-checks under its own lock without inflating the observable miss
        rate. Recency is still refreshed, because an entry served this way is
        in active use and must not be evicted as idle.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: K, value: V) -> None:
        """Retain ``value`` if it fits, evicting least-recent entries.

        A value costing more than the whole budget is retained only when the
        cache was built with ``admit_oversized=True`` (it then stands alone,
        displacing every other entry). Otherwise it is not retained — and if
        the key already held an entry, that entry is dropped and counted as an
        eviction, so a key never disappears without the counters saying so.
        """
        cost = self._cost(value)
        if cost < 0:
            raise ValueError(f"cache cost must be non-negative, got {cost}")
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._current -= existing[1]
            if cost > self._budget and not self._admit_oversized:
                if existing is not None:
                    self.stats.evictions += 1
                return
            self._entries[key] = (value, cost)
            self._current += cost
            # The just-inserted entry is most recent, so it is never popped
            # here; the length guard only matters for an admitted oversized
            # entry, which is allowed to remain alone over budget.
            while self._current > self._budget and len(self._entries) > 1:
                _, (_evicted, evicted_cost) = self._entries.popitem(last=False)
                self._current -= evicted_cost
                self.stats.evictions += 1

    def invalidate(self, predicate: Callable[[K], bool] | None = None) -> int:
        """Drop matching entries (all of them by default); returns the count."""
        with self._lock:
            if predicate is None:
                dropped = len(self._entries)
                self._entries.clear()
                self._current = 0
                return dropped
            doomed = [key for key in self._entries if predicate(key)]
            for key in doomed:
                _value, cost = self._entries.pop(key)
                self._current -= cost
            return len(doomed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def current_cost(self) -> int:
        with self._lock:
            return self._current

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            return CacheSnapshot(
                name=self.name,
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                entry_count=len(self._entries),
                current_cost=self._current,
                budget=self._budget,
            )
