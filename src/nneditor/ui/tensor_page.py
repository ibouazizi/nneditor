"""Bounded numeric windows of captured tensors, including partial captures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nneditor.tracing.contracts import ActivationRecord

PAGE_ROWS = 12
PAGE_COLUMNS = 12
Number = bool | int | float


@dataclass(frozen=True, slots=True)
class TensorPage:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    values: tuple[tuple[Number | None, ...], ...]
    row_count: int
    column_count: int
    leading: tuple[int, ...]


def read_tensor_page(
    record: ActivationRecord,
    read: Callable[..., bytes],
    *,
    leading: tuple[int, ...] = (),
    row: int = 0,
    column: int = 0,
) -> TensorPage:
    """Read at most 12x12 elements. None denotes bytes absent from a preview."""
    try:
        dtype: np.dtype[Any] = np.dtype(record.numpy_dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported capture dtype: {record.numpy_dtype}") from error
    if dtype.kind not in "biuf":
        raise ValueError("This dtype cannot be displayed as numeric values.")
    shape = record.shape
    rows, columns = shape[-2:] if len(shape) >= 2 else (1, shape[-1] if shape else 1)
    extents = shape[:-2]
    indices = leading or (0,) * len(extents)
    if len(indices) != len(extents) or any(
        i < 0 or i >= size for i, size in zip(indices, extents, strict=True)
    ):
        raise ValueError(f"Choose one valid index for each leading axis: {extents}")
    if row < 0 or column < 0 or row >= max(1, rows) or column >= max(1, columns):
        raise ValueError(f"Row must be below {rows}; column must be below {columns}.")
    prefix = 0
    for index, size in zip(indices, extents, strict=True):
        prefix = prefix * size + index
    row_ids = tuple(range(row, min(rows, row + PAGE_ROWS)))
    column_ids = tuple(range(column, min(columns, column + PAGE_COLUMNS)))
    values: list[tuple[Number | None, ...]] = []
    itemsize = dtype.itemsize
    for row_id in row_ids:
        offset = ((prefix * rows + row_id) * columns + column) * itemsize
        count = min(
            len(column_ids), max(0, (record.stored_byte_length - offset) // itemsize)
        )
        raw = (
            read(record.value_id, offset=offset, length=count * itemsize)
            if count
            else b""
        )
        array = np.frombuffer(raw, dtype=dtype, count=count)
        numbers: tuple[Number | None, ...] = tuple(
            bool(v) if dtype.kind == "b" else int(v) if dtype.kind in "iu" else float(v)
            for v in array
        )
        values.append(numbers + (None,) * (len(column_ids) - count))
    return TensorPage(row_ids, column_ids, tuple(values), rows, columns, indices)
