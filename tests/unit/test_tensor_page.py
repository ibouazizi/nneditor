"""Captured tensor paging must never materialize a whole large activation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nneditor.tracing.contracts import ActivationRecord, CaptureState
from nneditor.ui.tensor_page import PAGE_COLUMNS, PAGE_ROWS, read_tensor_page


def record_for(array: np.ndarray, stored: int | None = None) -> ActivationRecord:
    size = array.nbytes if stored is None else stored
    return ActivationRecord(
        "v:test",
        "tensor",
        None,
        "node-output",
        str(array.dtype),
        str(array.dtype),
        array.shape,
        CaptureState.DROPPED
        if size == 0
        else CaptureState.COMPLETE
        if size == array.nbytes
        else CaptureState.TRUNCATED,
        array.nbytes,
        size,
        "value.bin" if size else None,
    )


@pytest.mark.parametrize("shape", [(), (29,), (23, 31), (2, 3, 23, 31)])
def test_exact_bounded_window_and_leading_axes(shape: tuple[int, ...]) -> None:
    array = np.arange(
        np.prod(shape, dtype=np.int64) if shape else 1, dtype=np.float32
    ).reshape(shape)
    record = record_for(array)
    payload = array.tobytes()
    calls: list[tuple[int, int]] = []

    def read(_id: str, *, offset: int, length: int) -> bytes:
        calls.append((offset, length))
        return payload[offset : offset + length]

    leading = tuple(size - 1 for size in shape[:-2])
    row = 3 if len(shape) >= 2 else 0
    column = 5 if shape else 0
    page = read_tensor_page(record, read, leading=leading, row=row, column=column)
    matrix = array[leading] if leading else array
    if matrix.ndim < 2:
        matrix = matrix.reshape(1, -1)
    expected = matrix[row : row + PAGE_ROWS, column : column + PAGE_COLUMNS]
    np.testing.assert_array_equal(page.values, expected)
    assert (
        sum(length for _, length in calls) <= PAGE_ROWS * PAGE_COLUMNS * array.itemsize
    )
    assert all(length <= PAGE_COLUMNS * array.itemsize for _, length in calls)


def test_partial_prefix_uses_gaps_instead_of_fabricating_zeros() -> None:
    array = np.arange(60, dtype=np.float32).reshape(3, 20)
    raw = array.ravel()[:30].tobytes()
    record = record_for(array, len(raw))

    def read(_id: str, *, offset: int, length: int) -> bytes:
        assert offset + length <= len(raw)
        return raw[offset : offset + length]

    page = read_tensor_page(record, read)
    assert page.values[0][0] == 0
    assert page.values[1][:10] == tuple(range(20, 30))
    assert page.values[1][10:] == (None, None)
    assert page.values[2] == (None,) * PAGE_COLUMNS
    later = read_tensor_page(record, read, row=2, column=12)
    assert later.values == ((None,) * 8,)


@pytest.mark.parametrize(
    "indices",
    [
        {"leading": (2, 0)},
        {"leading": (0,)},
        {"leading": (-1, 0)},
        {"row": -1},
        {"column": -1},
        {"row": 3},
        {"column": 4},
    ],
)
def test_invalid_indices_are_rejected(indices: dict[str, object]) -> None:
    record = record_for(np.zeros((2, 2, 3, 4), np.float32))
    with pytest.raises(ValueError):
        read_tensor_page(
            record,
            lambda *_a, **_k: pytest.fail("must not read invalid slice"),
            **indices,  # type: ignore[arg-type]
        )


def test_integer_precision_boolean_and_invalid_dtype() -> None:
    for array in (np.array([2**62 + 1], np.int64), np.array([True, False])):
        record = record_for(array)
        page = read_tensor_page(
            record,
            lambda _id, array=array, **kw: array.tobytes()[
                kw["offset"] : kw["offset"] + kw["length"]
            ],
        )
        assert page.values == (tuple(array.tolist()),)
    for dtype in ("object", "complex64", "invalid_dtype"):
        bad = replace(record_for(np.ones(1, np.float32)), numpy_dtype=dtype)
        with pytest.raises(ValueError):
            read_tensor_page(bad, lambda *_a, **_k: b"")


def test_empty_tensor_does_not_read() -> None:
    record = record_for(np.empty((0, 4), np.float32))
    page = read_tensor_page(record, lambda *_a, **_k: pytest.fail("empty read"))
    assert not page.values
