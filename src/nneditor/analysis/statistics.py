"""Streaming, cancellable tensor statistics (task P3.3).

Statistics are computed from the packed bytes the tensor store serves, in
fixed-size chunks with a cancellation checkpoint between every chunk — a
multi-gigabyte tensor can be summarized without ever holding more than one
chunk of decoded values, and a cancelled job stops within one chunk.

Two passes over the data: the first collects extrema, the sum, sparsity, and
non-finite counts; the second bins the histogram, whose edges need the
extrema, and accumulates squared deviations from the first pass's mean —
the one-pass ``E[x²] - mean²`` form cancels catastrophically for data with
a large offset. Non-finite values are counted but excluded from extrema,
moments, and the histogram — a single NaN must not blank the entire summary.

Byte decoding is delegated to the dtype registry
(:mod:`nneditor.ir.dtypes`), the single authority on element widths and
decoders; everything else here is pure Python. The dependency policy keeps
numpy test-only; if Phase 5's numeric work changes that policy, this module
is the seam where it lands.

``STATISTICS_VERSION`` stamps every result. Persisted statistics whose
version differs from the running build are discarded and recomputed, never
migrated — they are derived data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from nneditor.cancellation import CancellationToken
from nneditor.ir.dtypes import dtype_info
from nneditor.storage.store import TensorStore, TensorUnavailableError

__all__ = [
    "STATISTICS_VERSION",
    "TensorStatistics",
    "compute_statistics",
    "decode_packed",
    "decode_unavailability",
    "element_width",
    "statistics_from_json",
    "statistics_to_json",
]

STATISTICS_VERSION: Final = 2
"""Version 2: two-pass variance replaced the cancellation-prone one-pass
form, so version-1 sidecar results are discarded and recomputed."""

DEFAULT_BINS: Final = 32
_CHUNK_ELEMENTS: Final = 262_144
"""Elements decoded per chunk; the checkpoint and memory granularity."""


def _decoder_entry(
    element_type: str,
) -> tuple[int, Callable[[bytes], Sequence[float]], bool] | None:
    """``(byte width, decoder, is floating point)`` from the dtype registry."""
    info = dtype_info(element_type)
    if info is None or info.decoder is None or info.byte_width is None:
        return None
    return info.byte_width, info.decoder, info.is_float


def element_width(element_type: str) -> int | None:
    """Bytes per element for dtypes whose packed bytes can be decoded."""
    entry = _decoder_entry(element_type)
    return entry[0] if entry is not None else None


def decode_packed(element_type: str, raw: bytes) -> Sequence[float] | None:
    """Decode packed bytes into numbers, or ``None`` for unsupported dtypes.

    Shared by statistics, diff previews, and the inspector's value previews so
    every feature interprets tensor bytes identically.
    """
    entry = _decoder_entry(element_type)
    if entry is None:
        return None
    width, decode, _is_float = entry
    usable = len(raw) - len(raw) % width
    return decode(raw[:usable])


def decode_unavailability(element_type: str) -> str | None:
    """Why this dtype's values cannot be decoded, or ``None`` when they can.

    UI surfaces show this reason instead of a silent blank when statistics,
    previews, or diffs cannot interpret a tensor's bytes.
    """
    info = dtype_info(element_type)
    if info is None:
        return f"element type {element_type!r} is not a recognized dtype"
    if info.decoder is not None and info.byte_width is not None:
        return None
    return info.unavailable_reason


def _unsupported(element_type: str) -> TensorUnavailableError:
    """The disclosure raised when a dtype's bytes cannot be summarized."""
    reason = decode_unavailability(element_type)
    detail = f": {reason}" if reason else ""
    return TensorUnavailableError(
        f"statistics for element type {element_type!r} are not supported{detail}"
    )


@dataclass(frozen=True, slots=True)
class TensorStatistics:
    """A complete streamed summary of one tensor's values."""

    tensor_id: str
    element_type: str
    element_count: int
    byte_size: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    zero_count: int
    nan_count: int
    inf_count: int
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    version: int = STATISTICS_VERSION

    @property
    def sparsity(self) -> float:
        """Fraction of exactly-zero elements."""
        if self.element_count == 0:
            return 0.0
        return self.zero_count / self.element_count

    @property
    def non_finite_count(self) -> int:
        return self.nan_count + self.inf_count


def compute_statistics(
    store: TensorStore,
    tensor_id: str,
    *,
    bins: int = DEFAULT_BINS,
    token: CancellationToken | None = None,
) -> TensorStatistics:
    """Stream one tensor's statistics through the store."""
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    tensor = store.metadata(tensor_id)
    decoder = _decoder_entry(tensor.element_type)
    if decoder is None:
        raise _unsupported(tensor.element_type)
    width, decode, is_float = decoder
    total_bytes = store.byte_length(tensor_id)
    if total_bytes is None:
        raise TensorUnavailableError(
            f"tensor {tensor_id!r} has no readable payload to summarize"
        )
    if total_bytes % width:
        raise TensorUnavailableError(
            f"tensor {tensor_id!r} has a malformed payload: {total_bytes} "
            f"bytes is not a multiple of the {width}-byte "
            f"{tensor.element_type} element size"
        )
    chunk_bytes = _CHUNK_ELEMENTS * width

    def chunks() -> Iterator[tuple[int, int]]:
        return (
            (offset, min(chunk_bytes, total_bytes - offset))
            for offset in range(0, total_bytes, chunk_bytes)
        )

    minimum: float | None = None
    maximum: float | None = None
    total = 0.0
    finite_count = 0
    zero_count = 0
    nan_count = 0
    inf_count = 0
    element_count = 0

    for offset, length in chunks():
        if token is not None:
            token.raise_if_cancelled()
        values = decode(store.read(tensor_id, offset=offset, length=length))
        element_count += len(values)
        if is_float:
            for value in values:
                if math.isnan(value):
                    nan_count += 1
                    continue
                if math.isinf(value):
                    inf_count += 1
                    continue
                finite_count += 1
                total += value
                if value == 0.0:
                    zero_count += 1
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
        else:
            finite_count += len(values)
            zero_count += sum(1 for value in values if value == 0)
            chunk_min = min(values, default=None)
            chunk_max = max(values, default=None)
            if chunk_min is not None:
                minimum = (
                    float(chunk_min)
                    if minimum is None
                    else min(minimum, float(chunk_min))
                )
            if chunk_max is not None:
                maximum = (
                    float(chunk_max)
                    if maximum is None
                    else max(maximum, float(chunk_max))
                )
            total += float(sum(values))

    mean: float | None = None
    if finite_count:
        mean = total / finite_count

    # The second pass, which the histogram needs anyway, accumulates squared
    # deviations from the settled mean: the one-pass E[x²] - mean² form
    # cancels catastrophically when the data sits on a large offset. A
    # constant tensor (minimum == maximum) skips the pass; its variance is
    # exactly zero.
    squared_error = 0.0
    edges: tuple[float, ...] = ()
    counts: tuple[int, ...] = ()
    if minimum is not None and maximum is not None:
        if minimum == maximum:
            edges = (minimum, maximum)
            counts = (finite_count,)
        else:
            assert mean is not None
            span = maximum - minimum
            edges = tuple(minimum + span * index / bins for index in range(bins + 1))
            histogram = [0] * bins
            for offset, length in chunks():
                if token is not None:
                    token.raise_if_cancelled()
                values = decode(store.read(tensor_id, offset=offset, length=length))
                for value in values:
                    if is_float and not math.isfinite(value):
                        continue
                    deviation = value - mean
                    squared_error += deviation * deviation
                    slot = int((value - minimum) / span * bins)
                    histogram[min(slot, bins - 1)] += 1
            counts = tuple(histogram)

    std: float | None = None
    if finite_count:
        std = math.sqrt(squared_error / finite_count)

    return TensorStatistics(
        tensor_id=tensor_id,
        element_type=tensor.element_type,
        element_count=element_count,
        byte_size=total_bytes,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        zero_count=zero_count,
        nan_count=nan_count,
        inf_count=inf_count,
        histogram_edges=edges,
        histogram_counts=counts,
    )


def statistics_to_json(stats: TensorStatistics) -> dict[str, object]:
    """A JSON-compatible form for the sidecar cache."""
    return {
        "tensor_id": stats.tensor_id,
        "element_type": stats.element_type,
        "element_count": stats.element_count,
        "byte_size": stats.byte_size,
        "minimum": stats.minimum,
        "maximum": stats.maximum,
        "mean": stats.mean,
        "std": stats.std,
        "zero_count": stats.zero_count,
        "nan_count": stats.nan_count,
        "inf_count": stats.inf_count,
        "histogram_edges": list(stats.histogram_edges),
        "histogram_counts": list(stats.histogram_counts),
        "version": stats.version,
    }


def statistics_from_json(data: dict[str, object]) -> TensorStatistics | None:
    """Rebuild persisted statistics; ``None`` for stale or malformed data."""
    try:
        if data.get("version") != STATISTICS_VERSION:
            return None
        edges = data.get("histogram_edges")
        counts = data.get("histogram_counts")
        assert isinstance(edges, list) and isinstance(counts, list)
        return TensorStatistics(
            tensor_id=str(data["tensor_id"]),
            element_type=str(data["element_type"]),
            element_count=int(data["element_count"]),  # type: ignore[call-overload]
            byte_size=int(data["byte_size"]),  # type: ignore[call-overload]
            minimum=None if data["minimum"] is None else float(data["minimum"]),  # type: ignore[arg-type]
            maximum=None if data["maximum"] is None else float(data["maximum"]),  # type: ignore[arg-type]
            mean=None if data["mean"] is None else float(data["mean"]),  # type: ignore[arg-type]
            std=None if data["std"] is None else float(data["std"]),  # type: ignore[arg-type]
            zero_count=int(data["zero_count"]),  # type: ignore[call-overload]
            nan_count=int(data["nan_count"]),  # type: ignore[call-overload]
            inf_count=int(data["inf_count"]),  # type: ignore[call-overload]
            histogram_edges=tuple(float(item) for item in edges),
            histogram_counts=tuple(int(item) for item in counts),
        )
    except (KeyError, TypeError, ValueError, AssertionError):
        return None
