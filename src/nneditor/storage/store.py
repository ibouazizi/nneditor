"""The tensor store: metadata-only access and bounded, cancellable reads.

Task P1.4, extended by P3.1/P3.2. The store is the *only* component allowed
to turn a :class:`~nneditor.ir.core.TensorRef` into bytes, and it does so
under three disciplines the architecture demands:

* **Safety.** Every read is range-checked against the real file, and external
  locations are resolved inside an approved root — a document cannot make the
  store read outside the model's directory.
* **Laziness stays observable and disclosed.** Metadata access (`metadata`,
  `readable`, `byte_length`, `materialization`) never opens a file; only
  `read` does, in chunks with cancellation checkpoints. Tensors stored in
  typed protobuf fields cannot be range-read — reading any part of one
  materializes the whole tensor, and :meth:`materialization` says so *before*
  the caller pays that cost.
* **Bounded memory.** Slice results and typed materializations are cached in
  separate :class:`~nneditor.storage.cache.BudgetedCache` instances with
  observable hit/miss/eviction counters and configurable budgets.

The store is thread-safe: statistics jobs read tensors concurrently with the
UI. It owns its file handles; closing it (or leaving its context) releases
every reader it opened.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from nneditor.cancellation import CancellationToken
from nneditor.ir.core import Document, Storage, TensorRef
from nneditor.storage.cache import BudgetedCache, CacheSnapshot, CacheStats
from nneditor.storage.paths import UnsafePathError, resolve_within
from nneditor.storage.reader import ArtifactReader, RangeReadError

__all__ = [
    "CacheStats",
    "Materialization",
    "TensorStore",
    "TensorUnavailableError",
]

_READ_CHUNK: Final = 4 * 1024 * 1024
"""Bytes fetched between cancellation checkpoints."""

DEFAULT_CACHE_BUDGET: Final = 64 * 1024 * 1024
DEFAULT_TYPED_BUDGET: Final = 32 * 1024 * 1024


class TensorUnavailableError(ValueError):
    """Raised when a tensor's bytes cannot be provided, with the reason."""


class Materialization(StrEnum):
    """What providing a tensor's bytes costs, disclosed before any read."""

    RANGE = "range"
    """Any slice is fetched by reading exactly that byte range."""

    FULL_PARSE = "full-parse"
    """The container stores values in typed protobuf fields; reading any
    slice materializes the whole tensor once (then slices are cache hits).
    The one-time promise holds even when the materialization exceeds the
    typed-cache budget: such a tensor is still retained, at the price of
    displacing every other typed materialization."""

    UNAVAILABLE = "unavailable"
    """The tensor declares no readable data."""


class TensorStore:
    """Bytes-on-demand access to one document's tensors."""

    __slots__ = (
        "_closed",
        "_lock",
        "_path_locks",
        "_readers",
        "_root",
        "_slice_cache",
        "_source_path",
        "_source_resolved",
        "_typed_cache",
        "_verified_paths",
        "document",
    )

    def __init__(
        self,
        document: Document,
        *,
        external_root: Path | None = None,
        cache_budget: int = DEFAULT_CACHE_BUDGET,
        typed_budget: int = DEFAULT_TYPED_BUDGET,
    ) -> None:
        self.document = document
        self._source_path = Path(document.source.path)
        # Resolved once here rather than once per read: resolution is a
        # syscall, and reads used to pay it while holding the store lock.
        self._source_resolved = self._source_path.resolve()
        self._root = external_root if external_root else self._source_path.parent
        self._slice_cache: BudgetedCache[tuple[str, int, int], bytes] = BudgetedCache(
            "tensor-slices", cache_budget, cost=len
        )
        # ``admit_oversized`` keeps FULL_PARSE honest: a materialization larger
        # than the budget is retained anyway, so slices stay one-time cost.
        self._typed_cache: BudgetedCache[str, bytes] = BudgetedCache(
            "typed-materializations", typed_budget, cost=len, admit_oversized=True
        )
        self._readers: dict[Path, ArtifactReader] = {}
        self._path_locks: dict[Path, threading.Lock] = {}
        self._verified_paths: set[Path] = set()
        self._lock = threading.RLock()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release every file handle and drop the caches."""
        with self._lock:
            self._closed = True
            doomed = [
                (self._path_locks[path], reader)
                for path, reader in self._readers.items()
            ]
            self._readers.clear()
            self._verified_paths.clear()
            self._slice_cache.invalidate()
            self._typed_cache.invalidate()
        # A descriptor is only ever used under its path lock, so acquiring the
        # lock before closing means no in-flight read loses its file mid-loop.
        for path_lock, reader in doomed:
            with path_lock:
                reader.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def open_file_count(self) -> int:
        """How many artifact files the store currently holds open."""
        with self._lock:
            return len(self._readers)

    @property
    def stats(self) -> CacheStats:
        """The slice cache's counters (kept for the Phase 1 test surface)."""
        return self._slice_cache.stats

    def cache_snapshots(self) -> tuple[CacheSnapshot, ...]:
        """Observable state of every cache this store owns."""
        return (self._slice_cache.snapshot(), self._typed_cache.snapshot())

    # -- metadata-only access ---------------------------------------------

    def metadata(self, tensor_id: str) -> TensorRef:
        """The tensor's reference; never touches a file."""
        try:
            return self.document.tensors[tensor_id]
        except KeyError:
            raise TensorUnavailableError(
                f"tensor {tensor_id!r} is not in the document"
            ) from None

    def materialization(self, tensor_id: str) -> Materialization:
        """How :meth:`read` would obtain this tensor's bytes."""
        tensor = self.metadata(tensor_id)
        if tensor.storage in (Storage.EMBEDDED_RAW, Storage.EXTERNAL):
            return Materialization.RANGE
        if tensor.storage is Storage.EMBEDDED_TYPED and tensor.typed_span is not None:
            return Materialization.FULL_PARSE
        if tensor.storage is Storage.GENERATED:
            return Materialization.UNAVAILABLE
        return Materialization.UNAVAILABLE

    def readable(self, tensor_id: str) -> bool:
        """Whether :meth:`read` can provide bytes for this tensor."""
        return self.materialization(tensor_id) is not Materialization.UNAVAILABLE

    def byte_length(self, tensor_id: str) -> int | None:
        """The tensor's payload length in bytes, when declared or derivable."""
        tensor = self.metadata(tensor_id)
        if tensor.storage is Storage.EMBEDDED_RAW:
            assert tensor.payload is not None
            return tensor.payload.length
        if tensor.storage is Storage.EXTERNAL:
            assert tensor.external is not None
            return tensor.external.length
        if tensor.storage is Storage.EMBEDDED_TYPED and tensor.typed_span is not None:
            from nneditor.adapters.onnx.typed_data import packed_length_for

            return packed_length_for(tensor)
        return None

    # -- reads -------------------------------------------------------------

    def read(
        self,
        tensor_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
        token: CancellationToken | None = None,
    ) -> bytes:
        """Return ``length`` payload bytes starting at ``offset``.

        ``offset`` and ``length`` are relative to the tensor's packed payload,
        not the file; ``length=None`` reads to the end of the payload. For
        typed-field tensors the first read materializes the whole tensor —
        check :meth:`materialization` to know in advance.
        """
        if self._closed:
            raise TensorUnavailableError("the tensor store is closed")
        if token is not None:
            token.raise_if_cancelled()
        tensor = self.metadata(tensor_id)
        if offset < 0 or (length is not None and length < 0):
            raise TensorUnavailableError(
                f"negative range for tensor {tensor_id!r}: "
                f"offset={offset} length={length}"
            )
        if tensor.storage is Storage.EMBEDDED_TYPED:
            return self._read_typed(tensor, offset, length, token)

        file_path, base_offset, total = self._locate(tensor)
        if length is None:
            length = max(0, total - offset)
        if offset + length > total:
            raise TensorUnavailableError(
                f"range [{offset}, {offset + length}) exceeds the "
                f"{total}-byte payload of tensor {tensor_id!r}"
            )

        key = (tensor_id, offset, length)
        cached = self._slice_cache.get(key)
        if cached is not None:
            return cached

        chunks: list[bytes] = []
        position = 0
        # Readers are serialized per file, not store-wide: verifying or
        # reading one multi-gigabyte component must not stall readers of
        # every other file behind an uncancellable lock.
        with self._path_lock(file_path):
            # Re-checked (without recounting the miss): another thread may
            # have produced this exact slice while we waited for the lock,
            # and its physical read must not be repeated.
            cached = self._slice_cache.peek(key)
            if cached is not None:
                return cached
            reader = self._reader(file_path, token=token)
            while position < length:
                if token is not None:
                    token.raise_if_cancelled()
                step = min(_READ_CHUNK, length - position)
                try:
                    chunks.append(reader.read(base_offset + offset + position, step))
                except RangeReadError as error:
                    raise TensorUnavailableError(
                        f"tensor {tensor_id!r} reads past the end of "
                        f"{file_path.name}: {error}"
                    ) from error
                position += step
        data = b"".join(chunks) if len(chunks) != 1 else chunks[0]
        self._slice_cache.put(key, data)
        return data

    def _read_typed(
        self,
        tensor: TensorRef,
        offset: int,
        length: int | None,
        token: CancellationToken | None,
    ) -> bytes:
        """Serve a slice of a typed-field tensor via full materialization."""
        if tensor.typed_span is None:
            raise TensorUnavailableError(
                f"tensor {tensor.id!r} has embedded_typed storage without a "
                "recorded message span; its bytes are not readable "
                "(reopen the artifact with a current build to index it)"
            )
        packed = self._typed_cache.get(tensor.id)
        if packed is None:
            from nneditor.adapters.onnx.typed_data import materialize_typed_tensor

            # The full parse runs under the source file's lock rather than the
            # store-wide one, so readers of other files are never stalled by it.
            with self._path_lock(self._source_resolved):
                packed = self._typed_cache.peek(tensor.id)
                if packed is None:
                    # Re-checked under the lock (without recounting the miss):
                    # a thread that lost the race reuses the winner's
                    # materialization instead of parsing the whole tensor again.
                    reader = self._reader(self._source_resolved, token=token)
                    try:
                        packed = materialize_typed_tensor(reader, tensor, token=token)
                    except TensorUnavailableError:
                        raise
                    except Exception as error:
                        raise TensorUnavailableError(
                            f"tensor {tensor.id!r} typed data could not be "
                            f"materialized: {error}"
                        ) from error
                    self._typed_cache.put(tensor.id, packed)
        total = len(packed)
        if length is None:
            length = max(0, total - offset)
        if offset + length > total:
            raise TensorUnavailableError(
                f"range [{offset}, {offset + length}) exceeds the "
                f"{total}-byte materialized payload of tensor {tensor.id!r}"
            )
        return packed[offset : offset + length]

    # -- internals ---------------------------------------------------------

    def _locate(self, tensor: TensorRef) -> tuple[Path, int, int]:
        """Resolve where a tensor's payload lives and how long it is.

        Every returned path is already canonical (the source path is resolved
        once at construction; :func:`resolve_within` canonicalizes external
        locations), so readers of the same file always share one dict key.
        """
        if tensor.storage is Storage.EMBEDDED_RAW:
            assert tensor.payload is not None
            return self._source_resolved, tensor.payload.offset, tensor.payload.length
        if tensor.storage is Storage.EXTERNAL:
            assert tensor.external is not None
            reference = tensor.external
            try:
                target = resolve_within(self._root, reference.location)
            except UnsafePathError as error:
                raise TensorUnavailableError(
                    f"tensor {tensor.id!r} references an unsafe external "
                    f"location: {error}"
                ) from error
            if reference.length is not None:
                return target, reference.offset, reference.length
            try:
                file_size = target.stat().st_size
            except OSError as error:
                raise TensorUnavailableError(
                    f"tensor {tensor.id!r} external file is unreadable: {error}"
                ) from error
            return target, reference.offset, max(0, file_size - reference.offset)
        raise TensorUnavailableError(
            f"tensor {tensor.id!r} has {tensor.storage.value} storage; its "
            "bytes are not range-readable"
        )

    def _path_lock(self, path: Path) -> threading.Lock:
        """The lock serializing all descriptor use for one artifact file."""
        with self._lock:
            lock = self._path_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._path_locks[path] = lock
            return lock

    def _reader(
        self, path: Path, *, token: CancellationToken | None = None
    ) -> ArtifactReader:
        """The open, verified reader for an already-canonical ``path``.

        The caller must hold the per-path lock for ``path``. The store-wide
        lock is taken only for dictionary bookkeeping: first-open verification
        hashes a possibly multi-gigabyte component, and doing that under the
        store lock would stall every other reader thread at a point where none
        of them can see their cancellation tokens. Hashing goes through the
        reader's own descriptor, so the verified bytes are exactly the bytes
        later reads will see — a path swapped between hash and open cannot
        slip through.
        """
        with self._lock:
            if self._closed:
                raise TensorUnavailableError("the tensor store is closed")
            reader = self._readers.get(path)
            if reader is not None:
                return reader
            verified = path in self._verified_paths
        expected = None if verified else self._expected_hash(path)
        try:
            reader = ArtifactReader(path)
        except OSError as error:
            raise TensorUnavailableError(f"cannot open {path.name}: {error}") from error
        try:
            if expected is not None:
                actual = reader.content_hash(token=token)
                if actual != expected:
                    raise TensorUnavailableError(
                        f"{path.name} changed after the artifact was opened: "
                        f"expected {expected}, computed {actual}"
                    )
            with self._lock:
                if self._closed:
                    raise TensorUnavailableError("the tensor store is closed")
                self._verified_paths.add(path)
                self._readers[path] = reader
        except BaseException:
            reader.close()
            raise
        return reader

    def _expected_hash(self, path: Path) -> str | None:
        """Return the import-time digest for one source component."""
        model: object | None = None
        for namespace, value in self.document.extensions:
            if namespace == "x-onnx.model":
                model = value
                break
        if path == self._source_resolved:
            if isinstance(model, dict):
                value = model.get("model_hash")
                if isinstance(value, str):
                    return value
            # Every adapter records the source artifact's digest. ONNX uses a
            # component-specific hash above because its document identity may
            # also cover external data; single-file formats use this fallback.
            return self.document.source.content_hash
        if not isinstance(model, dict):
            return None
        try:
            location = path.relative_to(self._root.resolve()).as_posix()
        except ValueError:
            return None
        entries = model.get("external_hashes")
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if (
                isinstance(entry, dict)
                and entry.get("location") == location
                and isinstance(entry.get("content_hash"), str)
            ):
                return str(entry["content_hash"])
        return None
