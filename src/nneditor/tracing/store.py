"""Disk-budgeted, range-readable activation capture storage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import cast

from nneditor.analysis.statistics import TensorStatistics, compute_statistics
from nneditor.cancellation import CancellationToken
from nneditor.storage.cache import BudgetedCache, CacheSnapshot
from nneditor.storage.store import TensorStore, TensorUnavailableError
from nneditor.tracing.contracts import (
    ActivationRecord,
    CaptureState,
    TraceKey,
    TraceResult,
)

__all__ = ["ActivationStore", "ActivationView"]


class ActivationView:
    """One trace exposed through the tensor-store read/statistics discipline."""

    __slots__ = ("_store", "trace_id")

    def __init__(self, store: ActivationStore, trace_id: str) -> None:
        self._store = store
        self.trace_id = trace_id

    def metadata(self, value_id: str) -> ActivationRecord:
        return self._store.result(self.trace_id).record(value_id)

    def readable(self, value_id: str) -> bool:
        return self.metadata(value_id).readable

    def byte_length(self, value_id: str) -> int | None:
        record = self.metadata(value_id)
        return record.stored_byte_length if record.readable else None

    def read(
        self,
        value_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
        token: CancellationToken | None = None,
    ) -> bytes:
        return self._store.read(
            self.trace_id,
            value_id,
            offset=offset,
            length=length,
            token=token,
        )

    def statistics(
        self,
        value_id: str,
        *,
        token: CancellationToken | None = None,
    ) -> TensorStatistics:
        return compute_statistics(
            cast(TensorStore, self),
            value_id,
            token=token,
        )


class ActivationStore:
    """Own trace directories, tombstones, LRU eviction, and slice caching."""

    __slots__ = (
        "_available",
        "_budget",
        "_current",
        "_lock",
        "_results",
        "_slice_cache",
        "root",
    )

    def __init__(
        self,
        root: Path,
        *,
        byte_budget: int = 512 * 1024 * 1024,
        slice_cache_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if byte_budget <= 0:
            raise ValueError("activation-store byte budget must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._budget = byte_budget
        self._current = 0
        self._lock = threading.RLock()
        self._results: dict[str, TraceResult] = {}
        self._available: OrderedDict[str, int] = OrderedDict()
        self._slice_cache: BudgetedCache[tuple[str, str, int, int], bytes] = (
            BudgetedCache(
                "activation-slices",
                slice_cache_bytes,
                cost=len,
            )
        )
        self._load()

    @property
    def byte_budget(self) -> int:
        return self._budget

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current

    def cache_snapshot(self) -> CacheSnapshot:
        return self._slice_cache.snapshot()

    def _trace_directory(self, trace_id: str) -> Path:
        digest = trace_id.split(":", 1)[-1]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("invalid trace identity")
        return self.root / f"trace-{digest}"

    def _load(self) -> None:
        for directory in sorted(self.root.glob("trace-*")):
            if not directory.is_dir():
                continue
            try:
                result = TraceResult.from_json(
                    json.loads((directory / "trace.json").read_text(encoding="utf-8"))
                )
                if directory != self._trace_directory(result.id):
                    raise ValueError
                cost = sum(
                    record.stored_byte_length
                    for record in result.records
                    if record.readable
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            self._results[result.id] = result
            if cost:
                self._available[result.id] = cost
                self._current += cost
        self._evict_to_budget()

    def begin_staging(self) -> Path:
        return Path(tempfile.mkdtemp(prefix=".trace-staging-", dir=self.root))

    @staticmethod
    def _write_manifest(directory: Path, result: TraceResult) -> None:
        manifest = directory / "trace.json"
        partial = directory / f".trace-{uuid.uuid4().hex}.partial"
        try:
            with open(partial, "w", encoding="utf-8") as handle:
                json.dump(result.to_json(), handle, indent=1, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, manifest)
        finally:
            partial.unlink(missing_ok=True)

    def discard_staging(self, directory: Path) -> None:
        target = directory.resolve()
        if target.parent != self.root or not target.name.startswith(".trace-staging-"):
            raise ValueError("refusing to discard a non-staging trace directory")
        shutil.rmtree(target, ignore_errors=True)

    def commit(
        self,
        key: TraceKey,
        records: tuple[ActivationRecord, ...],
        runtime: str,
        diagnostics: tuple[str, ...],
        staging: Path,
    ) -> TraceResult:
        target_staging = staging.resolve()
        if target_staging.parent != self.root or not target_staging.name.startswith(
            ".trace-staging-"
        ):
            raise ValueError("trace staging directory is outside the store")
        for record in records:
            if not record.readable:
                continue
            assert record.file_name is not None
            path = (target_staging / record.file_name).resolve()
            if not path.is_relative_to(target_staging) or not path.is_file():
                raise ValueError(f"capture file for {record.value_id!r} is missing")
            if path.stat().st_size != record.stored_byte_length:
                raise ValueError(
                    f"capture file for {record.value_id!r} has the wrong length"
                )
        trace_id = key.id
        target = self._trace_directory(trace_id)
        old = self.root / f".trace-old-{uuid.uuid4().hex}"
        with self._lock:
            merged = {record.value_id: record for record in records}
            retained_after_drop: list[str] = []
            previous_result = self._results.get(trace_id)
            if previous_result is not None and target.is_dir():
                for previous_record in previous_result.records:
                    current = merged.get(previous_record.value_id)
                    keep_previous = current is None or (
                        previous_record.readable
                        and (
                            not current.readable
                            or previous_record.stored_byte_length
                            > current.stored_byte_length
                        )
                    )
                    if not keep_previous:
                        continue
                    if previous_record.readable:
                        assert previous_record.file_name is not None
                        source = (target / previous_record.file_name).resolve()
                        destination = (
                            target_staging / previous_record.file_name
                        ).resolve()
                        if (
                            not source.is_relative_to(target)
                            or not destination.is_relative_to(target_staging)
                            or not source.is_file()
                        ):
                            raise ValueError(
                                "an existing activation capture is missing or unsafe"
                            )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                    if current is not None:
                        retained_after_drop.append(previous_record.value_id)
                    merged[previous_record.value_id] = previous_record
            merged_diagnostics = diagnostics
            if retained_after_drop:
                merged_diagnostics = (
                    *diagnostics,
                    "this run produced a shorter or unavailable capture for "
                    f"{len(retained_after_drop)} value(s); an earlier capture "
                    "for the identical artifact/revision/input key was retained",
                )
            result = TraceResult(
                key,
                tuple(sorted(merged.values(), key=lambda item: item.value_id)),
                runtime,
                merged_diagnostics,
            )
            self._write_manifest(target_staging, result)
            replaced = target.exists()
            try:
                if replaced:
                    os.replace(target, old)
                os.replace(target_staging, target)
            except BaseException:
                if replaced and old.exists() and not target.exists():
                    os.replace(old, target)
                raise
            finally:
                if old.exists():
                    shutil.rmtree(old, ignore_errors=True)
            previous = self._available.pop(result.id, 0)
            self._current -= previous
            self._results[result.id] = result
            cost = sum(
                record.stored_byte_length for record in records if record.readable
            )
            if cost:
                self._available[result.id] = cost
                self._current += cost
            self._slice_cache.invalidate(lambda item: item[0] == result.id)
            self._evict_to_budget()
            return self._results[result.id]

    def _evict_to_budget(self) -> None:
        while self._current > self._budget and self._available:
            trace_id, cost = self._available.popitem(last=False)
            self._current -= cost
            result = self._results[trace_id]
            evicted = TraceResult(
                result.key,
                tuple(
                    ActivationRecord(
                        value_id=record.value_id,
                        value_name=record.value_name,
                        node_id=record.node_id,
                        role=record.role,
                        element_type=record.element_type,
                        numpy_dtype=record.numpy_dtype,
                        shape=record.shape,
                        state=(
                            CaptureState.EVICTED if record.readable else record.state
                        ),
                        full_byte_length=record.full_byte_length,
                        stored_byte_length=(
                            0 if record.readable else record.stored_byte_length
                        ),
                        file_name=None,
                        reason=(
                            "capture was evicted by the activation-store byte budget"
                            if record.readable
                            else record.reason
                        ),
                    )
                    for record in result.records
                ),
                result.runtime,
                (*result.diagnostics, "capture payloads were evicted by budget"),
            )
            directory = self._trace_directory(trace_id)
            captures = directory / "captures"
            if captures.exists():
                shutil.rmtree(captures)
            self._write_manifest(directory, evicted)
            self._results[trace_id] = evicted
            self._slice_cache.invalidate(self._trace_predicate(trace_id))

    @staticmethod
    def _trace_predicate(
        trace_id: str,
    ) -> Callable[[tuple[str, str, int, int]], bool]:
        def matches(item: tuple[str, str, int, int]) -> bool:
            return item[0] == trace_id

        return matches

    def result(self, trace_id: str) -> TraceResult:
        with self._lock:
            try:
                result = self._results[trace_id]
            except KeyError:
                raise KeyError(trace_id) from None
            if trace_id in self._available:
                self._available.move_to_end(trace_id)
            return result

    def results(self, artifact_hash: str | None = None) -> tuple[TraceResult, ...]:
        with self._lock:
            values = [
                result
                for result in self._results.values()
                if artifact_hash is None or result.key.artifact_hash == artifact_hash
            ]
            return tuple(sorted(values, key=lambda item: item.id))

    def open(self, trace_id: str) -> ActivationView:
        self.result(trace_id)
        return ActivationView(self, trace_id)

    def read(
        self,
        trace_id: str,
        value_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
        token: CancellationToken | None = None,
    ) -> bytes:
        if token is not None:
            token.raise_if_cancelled()
        # Keep eviction/replacement from removing the payload between the
        # manifest lookup and the range read. The lock is store-local and
        # reads are bounded by the caller's requested slice.
        with self._lock:
            record = self.result(trace_id).record(value_id)
            if not record.readable:
                raise TensorUnavailableError(
                    record.reason
                    or f"activation {value_id!r} is {record.state.value}, not readable"
                )
            total = record.stored_byte_length
            resolved_length = total - offset if length is None else length
            if offset < 0 or resolved_length < 0 or offset + resolved_length > total:
                raise TensorUnavailableError(
                    f"activation range offset={offset} length={resolved_length} "
                    f"is outside {total} captured bytes"
                )
            key = (trace_id, value_id, offset, resolved_length)
            cached = self._slice_cache.get(key)
            if cached is not None:
                return cached
            assert record.file_name is not None
            directory = self._trace_directory(trace_id)
            path = (directory / record.file_name).resolve()
            if not path.is_relative_to(directory):
                raise TensorUnavailableError(
                    "activation capture path escaped its trace"
                )
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(resolved_length)
            if len(data) != resolved_length:
                raise TensorUnavailableError(
                    f"activation {value_id!r} ended before its recorded range"
                )
            if token is not None:
                token.raise_if_cancelled()
            self._slice_cache.put(key, data)
            return data
