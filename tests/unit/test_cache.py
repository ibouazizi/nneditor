"""Tests for the shared budgeted cache (P3.1)."""

from __future__ import annotations

import threading

import pytest

from nneditor.storage.cache import BudgetedCache


class TestBasics:
    def test_get_counts_hits_and_misses(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 100, cost=len)
        assert cache.get("a") is None
        cache.put("a", b"1234")
        assert cache.get("a") == b"1234"
        snapshot = cache.snapshot()
        assert (snapshot.hits, snapshot.misses) == (1, 1)
        assert snapshot.current_cost == 4
        assert snapshot.entry_count == 1
        assert snapshot.budget == 100
        assert snapshot.name == "t"

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BudgetedCache("t", -1)

    def test_negative_cost_is_rejected(self) -> None:
        cache: BudgetedCache[str, int] = BudgetedCache("t", 10, cost=lambda v: v)
        with pytest.raises(ValueError, match="non-negative"):
            cache.put("a", -1)


class TestEviction:
    def test_lru_eviction_under_budget_pressure(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 10, cost=len)
        cache.put("a", b"aaaa")
        cache.put("b", b"bbbb")
        assert cache.get("a") == b"aaaa", "touch a so b is least recent"
        cache.put("c", b"cccc")
        assert cache.get("b") is None, "b was evicted"
        assert cache.get("a") is not None
        assert cache.snapshot().evictions == 1

    def test_oversized_values_are_never_admitted(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 4, cost=len)
        cache.put("big", b"12345")
        assert len(cache) == 0
        assert cache.snapshot().evictions == 0

    def test_replacing_a_key_adjusts_cost(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 10, cost=len)
        cache.put("a", b"aaaaaaaa")
        cache.put("a", b"aa")
        assert cache.current_cost == 2
        assert len(cache) == 1

    def test_entry_counted_cache_defaults_to_cost_one(self) -> None:
        cache: BudgetedCache[int, str] = BudgetedCache("t", 2)
        cache.put(1, "one")
        cache.put(2, "two")
        cache.put(3, "three")
        assert len(cache) == 2
        assert cache.get(1) is None


class TestInvalidation:
    def test_invalidate_all(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 100, cost=len)
        cache.put("a", b"1")
        cache.put("b", b"2")
        assert cache.invalidate() == 2
        assert len(cache) == 0
        assert cache.current_cost == 0

    def test_invalidate_by_predicate(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 100, cost=len)
        cache.put("keep:x", b"11")
        cache.put("drop:y", b"22")
        assert cache.invalidate(lambda key: key.startswith("drop:")) == 1
        assert cache.get("keep:x") is not None
        assert cache.current_cost == 2


class TestConcurrency:
    def test_parallel_readers_and_writers_stay_consistent(self) -> None:
        cache: BudgetedCache[int, bytes] = BudgetedCache("t", 1024, cost=len)
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for index in range(200):
                    key = (worker_id * 200 + index) % 64
                    cache.put(key, bytes(16))
                    cache.get(key)
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        snapshot = cache.snapshot()
        assert snapshot.current_cost <= 1024
        assert snapshot.entry_count == snapshot.current_cost // 16
