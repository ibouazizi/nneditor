"""Instrumented, bounds-checked range reads over local artifacts.

Two things make this reader worth having over a plain file object:

* **Bounds are checked before allocation.** A malformed offset or length is
  rejected against the real file size rather than turning into a huge
  ``read`` call.
* **Reads are observable.** Every request is recorded, so a test can assert
  that opening a model never touched the byte ranges holding tensor payloads.

Two kinds of range are recorded because they answer different questions:

``logical``
    Bytes the parser consumed as content. This is the measure of laziness: if a
    logical range never overlaps a tensor payload, nothing in the payload was
    interpreted. Look-ahead that a decoder needs but then discards — a varint is
    self-terminating, so its length is unknown until it is examined — is fetched
    with ``record=False`` and reported only as physical traffic.
``physical``
    What actually reached the file, in whole blocks. Block alignment means a
    physical read next to a tensor payload can pull in part of it, so this is
    tracked as a byte budget rather than as an exact-overlap assertion.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from nneditor.cancellation import CancellationToken

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "MAX_RECORDED_RANGES",
    "MAX_SINGLE_READ",
    "ArtifactReader",
    "ByteRange",
    "RangeReadError",
    "ReadStats",
    "hash_file",
]

DEFAULT_BLOCK_SIZE: Final = 64 * 1024
MAX_SINGLE_READ: Final = 1 << 30
_HASH_CHUNK: Final = 1 << 20

MAX_RECORDED_RANGES: Final = 65_536
"""Entries kept per range list before overflow counting takes over.

Generous enough for every laziness assertion a test makes, small enough that a
session doing gigabytes of I/O does not grow its stats without bound."""

# Windows opens files in text mode by default, which would corrupt binary
# reads; POSIX has no O_BINARY, so fall back to zero there.
_OPEN_FLAGS: Final = os.O_RDONLY | getattr(os, "O_BINARY", 0)


def _read_at(fd: int, length: int, offset: int) -> bytes:
    """Positional read that works on both POSIX and Windows.

    ``os.pread`` does not exist on Windows, so seek-then-read is the fallback.
    An :class:`ArtifactReader` owns its descriptor exclusively, so the seek
    position is not shared state.
    """
    if hasattr(os, "pread"):
        data: bytes = os.pread(fd, length, offset)
        return data
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, length)


class RangeReadError(ValueError):
    """Raised when a requested byte range is not valid for the artifact."""


@dataclass(frozen=True, slots=True, order=True)
class ByteRange:
    """A half-open ``[offset, offset + length)`` span of an artifact."""

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 0:
            raise RangeReadError(
                f"byte range must be non-negative, got offset={self.offset} "
                f"length={self.length}"
            )

    @property
    def end(self) -> int:
        """The first offset past this range."""
        return self.offset + self.length

    def overlaps(self, other: ByteRange) -> bool:
        """Whether the two ranges share at least one byte."""
        return self.offset < other.end and other.offset < self.end

    def contains(self, other: ByteRange) -> bool:
        """Whether ``other`` lies entirely inside this range."""
        return self.offset <= other.offset and other.end <= self.end


def _coalesce(ranges: list[ByteRange]) -> tuple[ByteRange, ...]:
    merged: list[ByteRange] = []
    for current in sorted(ranges):
        if current.length == 0:
            continue
        if merged and current.offset <= merged[-1].end:
            previous = merged[-1]
            end = max(previous.end, current.end)
            merged[-1] = ByteRange(previous.offset, end - previous.offset)
        else:
            merged.append(current)
    return tuple(merged)


@dataclass(slots=True)
class ReadStats:
    """Everything a test needs to prove what an import touched.

    The two range lists are capped at ``max_recorded_ranges`` entries each so a
    long-lived session cannot leak memory through its own instrumentation.
    Ranges recorded past the cap are tallied in ``dropped_ranges`` instead of
    stored; the byte totals stay exact regardless, because dropped ranges still
    contribute their lengths.
    """

    logical_ranges: list[ByteRange] = field(default_factory=list)
    physical_ranges: list[ByteRange] = field(default_factory=list)
    max_recorded_ranges: int = MAX_RECORDED_RANGES
    dropped_ranges: int = 0
    _dropped_logical_bytes: int = field(default=0, repr=False)
    _dropped_physical_bytes: int = field(default=0, repr=False)

    def record_logical(self, span: ByteRange) -> None:
        """Record a parser request, dropping the range itself once capped."""
        if len(self.logical_ranges) < self.max_recorded_ranges:
            self.logical_ranges.append(span)
        else:
            self.dropped_ranges += 1
            self._dropped_logical_bytes += span.length

    def record_physical(self, span: ByteRange) -> None:
        """Record a file fetch, dropping the range itself once capped."""
        if len(self.physical_ranges) < self.max_recorded_ranges:
            self.physical_ranges.append(span)
        else:
            self.dropped_ranges += 1
            self._dropped_physical_bytes += span.length

    @property
    def logical_bytes(self) -> int:
        """Bytes the parser asked for, counting repeats; exact despite the cap."""
        return (
            sum(item.length for item in self.logical_ranges)
            + self._dropped_logical_bytes
        )

    @property
    def physical_bytes(self) -> int:
        """Bytes actually fetched from the file, counting repeats; exact."""
        return (
            sum(item.length for item in self.physical_ranges)
            + self._dropped_physical_bytes
        )

    def coalesced_logical(self) -> tuple[ByteRange, ...]:
        """Merged logical ranges, useful for readable assertions."""
        return _coalesce(self.logical_ranges)

    def coalesced_physical(self) -> tuple[ByteRange, ...]:
        """Merged physical ranges."""
        return _coalesce(self.physical_ranges)

    def touched_logically(self, target: ByteRange) -> bool:
        """Whether any parser request overlapped ``target``."""
        return any(item.overlaps(target) for item in self.logical_ranges)

    def reset(self) -> None:
        """Forget everything recorded so far."""
        self.logical_ranges.clear()
        self.physical_ranges.clear()
        self.dropped_ranges = 0
        self._dropped_logical_bytes = 0
        self._dropped_physical_bytes = 0


class ArtifactReader:
    """A read-only, block-cached, instrumented view of one local artifact.

    The source file is never opened for writing: architectural rule 1 makes
    source artifacts immutable.
    """

    __slots__ = ("_blocks", "_fd", "_max_blocks", "block_size", "path", "size", "stats")

    def __init__(
        self,
        path: Path | str,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_cached_blocks: int = 64,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_cached_blocks <= 0:
            raise ValueError("max_cached_blocks must be positive")
        self.path = Path(path)
        self.block_size = block_size
        self._max_blocks = max_cached_blocks
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self.stats = ReadStats()
        self._fd = os.open(self.path, _OPEN_FLAGS)
        try:
            self.size = os.fstat(self._fd).st_size
        except BaseException:  # pragma: no cover - fstat failure on an open fd
            os.close(self._fd)
            raise

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
        """Release the file descriptor and drop cached blocks."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._blocks.clear()

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has already run."""
        return self._fd < 0

    def validate(self, offset: int, length: int) -> ByteRange:
        """Check a range against the artifact without reading anything."""
        if isinstance(offset, bool) or isinstance(length, bool):
            raise RangeReadError("byte ranges must be integers, not booleans")
        if not isinstance(offset, int) or not isinstance(length, int):
            raise RangeReadError("byte ranges must be integers")
        if offset < 0 or length < 0:
            raise RangeReadError(
                f"negative byte range: offset={offset} length={length}"
            )
        if length > MAX_SINGLE_READ:
            raise RangeReadError(
                f"refusing a {length} byte read; the limit is {MAX_SINGLE_READ}"
            )
        if offset + length > self.size:
            raise RangeReadError(
                f"range [{offset}, {offset + length}) exceeds artifact size "
                f"{self.size} for {self.path.name}"
            )
        return ByteRange(offset, length)

    def record_logical(self, offset: int, length: int) -> None:
        """Record that the parser consumed ``length`` bytes at ``offset``.

        Used by decoders that must fetch a look-ahead window and only afterwards
        know how much of it was content.
        """
        self.stats.record_logical(self.validate(offset, length))

    def read(self, offset: int, length: int, *, record: bool = True) -> bytes:
        """Return ``length`` bytes at ``offset``.

        ``record=False`` fetches without counting the bytes as consumed; the
        caller is then responsible for calling :meth:`record_logical` with the
        part it actually used.
        """
        if self.closed:
            raise RangeReadError(f"reader for {self.path.name} is closed")
        span = self.validate(offset, length)
        if record:
            self.stats.record_logical(span)
        if length == 0:
            return b""
        first = offset // self.block_size
        last = (offset + length - 1) // self.block_size
        if first == last:
            block = self._block(first)
            start = offset - first * self.block_size
            return block[start : start + length]
        chunks = [self._block(index) for index in range(first, last + 1)]
        joined = b"".join(chunks)
        start = offset - first * self.block_size
        return joined[start : start + length]

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            self._blocks.move_to_end(index)
            return cached
        start = index * self.block_size
        length = min(self.block_size, self.size - start)
        data = _read_at(self._fd, length, start)
        if len(data) != length:
            # Windows opens artifacts with FILE_SHARE_WRITE, so a concurrent
            # truncation surfaces as a silently short read rather than an error.
            raise RangeReadError(
                f"short read from {self.path.name}: expected {length} bytes at "
                f"offset {start}, got {len(data)} (the file shrank while open)"
            )
        self.stats.record_physical(ByteRange(start, len(data)))
        self._blocks[index] = data
        while len(self._blocks) > self._max_blocks:
            self._blocks.popitem(last=False)
        return data

    def content_hash(
        self,
        *,
        algorithm: str = "sha256",
        token: CancellationToken | None = None,
    ) -> str:
        """Hash the whole artifact through this reader's own descriptor.

        Hashing the already-open descriptor closes the gap between hashing a
        path and opening it: the bytes that are verified are exactly the bytes
        later reads will see. The traffic bypasses the block cache and the
        instrumentation — hashing is not parsing, so it must not pollute the
        laziness stats — and checkpoints the token between chunks so a
        multi-gigabyte verification stays cancellable.
        """
        if self.closed:
            raise RangeReadError(f"reader for {self.path.name} is closed")
        digest = hashlib.new(algorithm)
        position = 0
        while position < self.size:
            if token is not None:
                token.raise_if_cancelled()
            step = min(_HASH_CHUNK, self.size - position)
            chunk = _read_at(self._fd, step, position)
            if not chunk:
                raise RangeReadError(
                    f"short read from {self.path.name} while hashing: expected "
                    f"{step} bytes at offset {position}, got none "
                    "(the file shrank while open)"
                )
            digest.update(chunk)
            position += len(chunk)
        return f"{algorithm}:{digest.hexdigest()}"


def hash_file(
    path: Path | str,
    *,
    algorithm: str = "sha256",
    token: CancellationToken | None = None,
) -> str:
    """Return the content hash that identifies an immutable source artifact.

    Deliberately separate from :class:`ArtifactReader` so that hashing a whole
    file never pollutes the laziness instrumentation.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            if token is not None:
                token.raise_if_cancelled()
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"
