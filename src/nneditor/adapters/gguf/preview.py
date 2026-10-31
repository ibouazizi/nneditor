"""On-demand previews of GGUF metadata arrays that were elided at open.

Tokenizer vocabularies and other large KV arrays stay byte ranges in the
index. This reader decodes a bounded prefix so the inspector can show the
first tokens without parsing the rest of the array.
"""

from __future__ import annotations

import struct
from pathlib import Path

from nneditor.adapters.gguf.constants import (
    GGUF_VALUE_TYPE_WIDTH,
    KV_TYPE_ARRAY,
    KV_TYPE_BOOL,
    KV_TYPE_FLOAT32,
    KV_TYPE_FLOAT64,
    KV_TYPE_INT8,
    KV_TYPE_INT16,
    KV_TYPE_INT32,
    KV_TYPE_INT64,
    KV_TYPE_STRING,
    KV_TYPE_UINT8,
    KV_TYPE_UINT16,
    KV_TYPE_UINT32,
    KV_TYPE_UINT64,
    MAX_GGUF_STRING,
    GgufWireLayout,
    layout_for,
)
from nneditor.adapters.gguf.indexer import GgufIndexError
from nneditor.ir.core import JsonValue
from nneditor.storage.reader import ArtifactReader, RangeReadError

__all__ = ["preview_kv_array"]

_SCALAR_FORMAT: dict[int, str] = {
    KV_TYPE_UINT8: "B",
    KV_TYPE_INT8: "b",
    KV_TYPE_UINT16: "H",
    KV_TYPE_INT16: "h",
    KV_TYPE_UINT32: "I",
    KV_TYPE_INT32: "i",
    KV_TYPE_FLOAT32: "f",
    KV_TYPE_UINT64: "Q",
    KV_TYPE_INT64: "q",
    KV_TYPE_FLOAT64: "d",
}


class _Window:
    """A bounded random-access reader over one KV value's bytes."""

    def __init__(
        self,
        reader: ArtifactReader,
        start: int,
        length: int,
        layout: GgufWireLayout,
    ) -> None:
        self.reader = reader
        self.start = start
        self.end = start + length
        self.pos = start
        self.layout = layout

    def read(self, size: int) -> bytes:
        if size < 0:
            raise GgufIndexError("negative GGUF preview read")
        if self.pos + size > self.end:
            raise GgufIndexError("GGUF preview overran the recorded value range")
        try:
            data = self.reader.read(self.pos, size)
        except RangeReadError as error:
            raise GgufIndexError(f"truncated GGUF preview: {error}") from error
        self.pos += size
        return data

    def unpack(self, fmt: str) -> tuple[int | float, ...]:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def u32(self) -> int:
        return int(self.unpack(self.layout.struct_endian + "I")[0])

    def count(self) -> int:
        return int(self.unpack(self.layout.struct_endian + self.layout.count_code)[0])


def _decode_item(window: _Window, type_id: int) -> JsonValue:
    if type_id == KV_TYPE_STRING:
        length = window.count()
        if length > MAX_GGUF_STRING:
            raise GgufIndexError(
                f"GGUF preview string is {length} bytes; the limit is {MAX_GGUF_STRING}"
            )
        raw = window.read(length).rstrip(b"\x00")
        return raw.decode("utf-8", errors="replace")
    if type_id == KV_TYPE_BOOL:
        return window.read(1)[0] != 0
    if type_id == KV_TYPE_ARRAY:
        raise GgufIndexError("nested GGUF arrays are not previewed")
    fmt_code = _SCALAR_FORMAT.get(type_id)
    width = GGUF_VALUE_TYPE_WIDTH.get(type_id)
    if fmt_code is None or width is None:
        raise GgufIndexError(f"unsupported GGUF preview type {type_id}")
    value = window.unpack(window.layout.struct_endian + fmt_code)[0]
    if type_id in {KV_TYPE_FLOAT32, KV_TYPE_FLOAT64}:
        return float(value)
    return int(value)


def preview_kv_array(
    path: Path | str,
    *,
    offset: int,
    length: int,
    version: int,
    endian: str,
    limit: int = 32,
) -> tuple[tuple[JsonValue, ...], int]:
    """Decode the first ``limit`` items of a GGUF array at ``offset``.

    Returns ``(items, total_count)``. Nested arrays are refused; the caller
    should only offer this for elided scalar or string arrays.
    """
    if limit <= 0:
        raise ValueError("preview limit must be positive")
    if offset < 0 or length < 0:
        raise GgufIndexError("GGUF preview range must be non-negative")
    layout = layout_for(version, endian)
    with ArtifactReader(path) as reader:
        window = _Window(reader, offset, length, layout)
        elem_type = window.u32()
        count = window.count()
        take = min(limit, count)
        items = tuple(_decode_item(window, elem_type) for _ in range(take))
    return items, count
