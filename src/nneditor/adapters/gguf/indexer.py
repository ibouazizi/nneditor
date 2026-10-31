"""Native GGUF indexer: header + KV + tensor table, never the payloads.

Supports every published GGUF structural version:

* **v1** — uint32 tensor/KV counts, string lengths, array lengths, and dims.
* **v2** — those countable fields widened to uint64.
* **v3** — v2 layout plus big-endian files, inferred from the version word.

Identity is ``gguf-index:``: SHA-256 of bytes ``[0, data_start)`` plus a
little-endian uint64 file size. Tensor payloads are recorded as ranges and
are not hashed or decoded at open.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nneditor.adapters.gguf.constants import (
    GGML_MAX_DIMS,
    GGML_QUANT_SIZES,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC_BYTES,
    GGUF_SUPPORTED_VERSIONS,
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
    MAX_GGUF_KV_ENTRIES,
    MAX_GGUF_KV_VALUE_BYTES,
    MAX_GGUF_STRING,
    MAX_GGUF_TENSORS,
    MAX_RETAINED_ARRAY_BYTES,
    MAX_RETAINED_STRING_ARRAY,
    MAX_TENSOR_NAME,
    MAX_VALUE_DEPTH,
    REMOVED_GGML_TYPE_IDS,
    GgufWireLayout,
    element_type_name,
    layout_for,
    packed_nbytes,
)
from nneditor.adapters.gguf.index import GgufIndex, GgufKvEntry, GgufTensorIndex
from nneditor.cancellation import CancellationToken
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.core import JsonValue, Storage
from nneditor.ir.identity import ROOT_GRAPH_ID, initializer_id
from nneditor.storage.reader import ArtifactReader, ByteRange, RangeReadError

__all__ = ["GgufIndexError", "index_gguf"]

_IMPLAUSIBLE_ELEMS: Final = 1 << 48
_SWAPPED_VERSIONS: Final = {1: 0x01000000, 2: 0x02000000, 3: 0x03000000}
_SCALAR_FORMAT: Final[dict[int, str]] = {
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


class GgufIndexError(ValueError):
    """Raised when a GGUF artifact cannot be indexed at all."""


class _Cursor:
    """Sequential hashed reader over GGUF metadata."""

    __slots__ = ("_ops", "digest", "layout", "pos", "reader", "token")

    def __init__(self, reader: ArtifactReader, token: CancellationToken | None) -> None:
        self.reader = reader
        self.token = token
        self.pos = 0
        self.digest = hashlib.sha256()
        self.layout: GgufWireLayout | None = None
        self._ops = 0

    def read(self, length: int) -> bytes:
        if length < 0:
            raise GgufIndexError("negative GGUF read")
        if length == 0:
            return b""
        self._ops += 1
        if self.token is not None and self._ops % 64 == 1:
            self.token.raise_if_cancelled()
        try:
            data = self.reader.read(self.pos, length)
        except RangeReadError as error:
            raise GgufIndexError(f"truncated GGUF: {error}") from error
        self.digest.update(data)
        self.pos += length
        return data

    def unpack(self, fmt: str) -> tuple[int | float, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))

    def u32(self) -> int:
        endian = "<" if self.layout is None else self.layout.struct_endian
        return int(self.unpack(endian + "I")[0])

    def u64(self) -> int:
        assert self.layout is not None
        return int(self.unpack(self.layout.struct_endian + "Q")[0])

    def count(self) -> int:
        assert self.layout is not None
        return int(self.unpack(self.layout.struct_endian + self.layout.count_code)[0])

    def dim(self) -> int:
        assert self.layout is not None
        return int(self.unpack(self.layout.struct_endian + self.layout.dim_code)[0])

    def finish_identity(self, data_start: int, file_size: int) -> str:
        if self.pos > data_start:
            raise GgufIndexError("GGUF tensor table overran the aligned data section")
        if self.pos < data_start:
            try:
                padding = self.reader.read(
                    self.pos, data_start - self.pos, record=False
                )
            except RangeReadError as error:
                raise GgufIndexError(f"truncated GGUF padding: {error}") from error
            self.digest.update(padding)
            self.pos = data_start
        self.digest.update(struct.pack("<Q", file_size))
        return f"gguf-index:{self.digest.hexdigest()}"


def _read_header(cursor: _Cursor) -> tuple[GgufWireLayout, int, int]:
    magic = cursor.read(4)
    if magic != GGUF_MAGIC_BYTES:
        raise GgufIndexError(
            f"not a GGUF file: magic {magic!r}, expected {GGUF_MAGIC_BYTES!r}"
        )
    raw_version = cursor.read(4)
    version_le = int(struct.unpack("<I", raw_version)[0])
    if version_le in GGUF_SUPPORTED_VERSIONS:
        layout = layout_for(version_le, "little")
    elif version_le in _SWAPPED_VERSIONS.values():
        version = int(struct.unpack(">I", raw_version)[0])
        layout = layout_for(version, "big")
    else:
        raise GgufIndexError(f"unsupported GGUF version {version_le}")
    cursor.layout = layout
    n_tensors = cursor.count()
    n_kv = cursor.count()
    if n_tensors > MAX_GGUF_TENSORS:
        raise GgufIndexError(
            f"GGUF declares {n_tensors} tensors; the limit is {MAX_GGUF_TENSORS}"
        )
    if n_kv > MAX_GGUF_KV_ENTRIES:
        raise GgufIndexError(
            f"GGUF declares {n_kv} key-value pairs; the limit is {MAX_GGUF_KV_ENTRIES}"
        )
    return layout, n_tensors, n_kv


def _decode_utf8(raw: bytes) -> str:
    stripped = raw.rstrip(b"\x00")
    return stripped.decode("utf-8", errors="replace")


def _read_string(cursor: _Cursor, *, limit: int = MAX_GGUF_STRING) -> str:
    length = cursor.count()
    if length > limit:
        raise GgufIndexError(
            f"GGUF string length {length} exceeds the {limit}-byte ceiling"
        )
    return _decode_utf8(cursor.read(length))


def _skip_string(cursor: _Cursor) -> int:
    start = cursor.pos
    length = cursor.count()
    if length > MAX_GGUF_STRING:
        raise GgufIndexError(
            f"GGUF string length {length} exceeds the {MAX_GGUF_STRING}-byte ceiling"
        )
    cursor.read(length)
    return cursor.pos - start


def _read_scalar(cursor: _Cursor, type_id: int) -> JsonValue:
    if type_id == KV_TYPE_BOOL:
        raw = cursor.read(1)[0]
        return raw != 0
    fmt_code = _SCALAR_FORMAT.get(type_id)
    if fmt_code is None:
        raise GgufIndexError(f"unsupported GGUF value type {type_id}")
    assert cursor.layout is not None
    value = cursor.unpack(cursor.layout.struct_endian + fmt_code)[0]
    if type_id in {KV_TYPE_FLOAT32, KV_TYPE_FLOAT64}:
        return float(value)
    return int(value)


def _scalar_width(type_id: int) -> int:
    width = GGUF_VALUE_TYPE_WIDTH.get(type_id)
    if width is None:
        raise GgufIndexError(f"unsupported GGUF value type {type_id}")
    return width


def _skip_value(cursor: _Cursor, type_id: int, depth: int) -> int:
    start = cursor.pos
    if type_id == KV_TYPE_STRING:
        _skip_string(cursor)
        return cursor.pos - start
    if type_id == KV_TYPE_ARRAY:
        if depth >= MAX_VALUE_DEPTH:
            raise GgufIndexError("GGUF array nesting exceeds the depth ceiling")
        elem_type = cursor.u32()
        count = cursor.count()
        if count > MAX_GGUF_KV_ENTRIES:
            raise GgufIndexError(
                f"GGUF array length {count} exceeds the {MAX_GGUF_KV_ENTRIES} ceiling"
            )
        if elem_type == KV_TYPE_STRING or elem_type == KV_TYPE_ARRAY:
            for _ in range(count):
                _skip_value(cursor, elem_type, depth + 1)
        else:
            width = _scalar_width(elem_type)
            cursor.read(count * width)
        return cursor.pos - start
    cursor.read(_scalar_width(type_id))
    return cursor.pos - start


@dataclass(slots=True)
class _ValueRead:
    value: JsonValue
    elided: bool


def _read_value(
    cursor: _Cursor, type_id: int, log: DiagnosticLog, key: str, depth: int
) -> _ValueRead:
    if type_id == KV_TYPE_STRING:
        start = cursor.pos
        length = cursor.count()
        if length > MAX_GGUF_STRING or length > MAX_GGUF_KV_VALUE_BYTES:
            if length > MAX_GGUF_STRING:
                raise GgufIndexError(
                    f"GGUF string {key!r} is {length} bytes; the limit is "
                    f"{MAX_GGUF_STRING}"
                )
            cursor.read(length)
            log.add(
                "gguf.kv-value-elided",
                Severity.INFO,
                f"key {key!r} holds a {length}-byte string; it is kept as a "
                "byte range rather than parsed at open",
                key,
            )
            return _ValueRead(
                {
                    "type": "string",
                    "offset": start,
                    "length": cursor.pos - start,
                },
                True,
            )
        text = _decode_utf8(cursor.read(length))
        return _ValueRead(text, False)
    if type_id != KV_TYPE_ARRAY:
        return _ValueRead(_read_scalar(cursor, type_id), False)
    return _read_array(cursor, log, key, depth)


def _read_array(
    cursor: _Cursor, log: DiagnosticLog, key: str, depth: int
) -> _ValueRead:
    if depth >= MAX_VALUE_DEPTH:
        start = cursor.pos
        _skip_value(cursor, KV_TYPE_ARRAY, depth)
        log.add(
            "gguf.kv-value-elided",
            Severity.INFO,
            f"key {key!r} nests arrays deeper than {MAX_VALUE_DEPTH}; the "
            "value is kept as a byte range",
            key,
        )
        return _ValueRead(
            {
                "type": "array",
                "offset": start,
                "length": cursor.pos - start,
            },
            True,
        )
    start = cursor.pos
    elem_type = cursor.u32()
    count = cursor.count()
    if count > MAX_GGUF_KV_ENTRIES:
        raise GgufIndexError(
            f"GGUF array {key!r} has {count} elements; the limit is "
            f"{MAX_GGUF_KV_ENTRIES}"
        )
    elide = _should_elide_array(elem_type, count)
    if elide:
        for _ in range(count) if elem_type in {KV_TYPE_STRING, KV_TYPE_ARRAY} else ():
            _skip_value(cursor, elem_type, depth + 1)
        if elem_type not in {KV_TYPE_STRING, KV_TYPE_ARRAY}:
            cursor.read(count * _scalar_width(elem_type))
        log.add(
            "gguf.kv-value-elided",
            Severity.INFO,
            f"key {key!r} holds a {count}-element array; it is kept as a "
            "byte range rather than parsed at open",
            key,
        )
        return _ValueRead(
            {
                "type": "array",
                "element_type": elem_type,
                "count": count,
                "offset": start,
                "length": cursor.pos - start,
            },
            True,
        )
    items: list[JsonValue] = []
    for _ in range(count):
        item = _read_value(cursor, elem_type, log, key, depth + 1)
        items.append(item.value)
    return _ValueRead(items, False)


def _should_elide_array(elem_type: int, count: int) -> bool:
    if elem_type == KV_TYPE_STRING:
        return count > MAX_RETAINED_STRING_ARRAY
    if elem_type == KV_TYPE_ARRAY:
        return count > 16
    width = GGUF_VALUE_TYPE_WIDTH.get(elem_type)
    if width is None:
        return True
    nbytes = count * width
    return nbytes > MAX_RETAINED_ARRAY_BYTES or nbytes > MAX_GGUF_KV_VALUE_BYTES


def _read_kv_entries(
    cursor: _Cursor,
    n_kv: int,
    token: CancellationToken | None,
    log: DiagnosticLog,
) -> list[GgufKvEntry]:
    entries: list[GgufKvEntry] = []
    seen: set[str] = set()
    for _ in range(n_kv):
        if token is not None:
            token.raise_if_cancelled()
        start = cursor.pos
        key = _read_string(cursor, limit=65_535)
        type_id = cursor.u32()
        parsed = _read_value(cursor, type_id, log, key, 0)
        if key in seen:
            log.add(
                "gguf.duplicate-kv-key",
                Severity.WARNING,
                f"duplicate metadata key {key!r}; the later value is kept",
                key,
            )
            entries = [item for item in entries if item.key != key]
        seen.add(key)
        entries.append(
            GgufKvEntry(
                key=key,
                type_id=type_id,
                value=parsed.value,
                elided=parsed.elided,
                offset=start,
                length=cursor.pos - start,
            )
        )
    return entries


def _alignment_from_kv(entries: list[GgufKvEntry], log: DiagnosticLog) -> int:
    matching = [item for item in entries if item.key == "general.alignment"]
    if not matching:
        return GGUF_DEFAULT_ALIGNMENT
    value = matching[-1].value
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        log.add(
            "gguf.bad-alignment",
            Severity.WARNING,
            "general.alignment is missing or not a positive integer; "
            f"{GGUF_DEFAULT_ALIGNMENT}-byte alignment is used",
            "general.alignment",
        )
        return GGUF_DEFAULT_ALIGNMENT
    if value & (value - 1) != 0:
        log.add(
            "gguf.bad-alignment",
            Severity.WARNING,
            f"general.alignment is {value}, which is not a power of two; "
            f"{GGUF_DEFAULT_ALIGNMENT}-byte alignment is used",
            "general.alignment",
        )
        return GGUF_DEFAULT_ALIGNMENT
    return value


@dataclass(frozen=True, slots=True)
class _RawTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    relative_offset: int


def _read_tensor_table(
    cursor: _Cursor,
    n_tensors: int,
    token: CancellationToken | None,
    log: DiagnosticLog,
) -> list[_RawTensor]:
    tensors: list[_RawTensor] = []
    seen: set[str] = set()
    for _ in range(n_tensors):
        if token is not None:
            token.raise_if_cancelled()
        name = _read_string(cursor, limit=MAX_TENSOR_NAME)
        n_dims = cursor.u32()
        if n_dims > GGML_MAX_DIMS:
            raise GgufIndexError(
                f"tensor {name!r} declares {n_dims} dimensions; "
                f"GGML_MAX_DIMS is {GGML_MAX_DIMS}"
            )
        dims = tuple(cursor.dim() for _ in range(n_dims))
        ggml_type = cursor.u32()
        relative_offset = cursor.u64()
        if name in seen:
            log.add(
                "gguf.duplicate-tensor-name",
                Severity.ERROR,
                f"duplicate tensor name {name!r}; the later entry is ignored",
                name,
            )
            continue
        seen.add(name)
        tensors.append(_RawTensor(name, dims, ggml_type, relative_offset))
    return tensors


def _n_elems(dims: tuple[int, ...]) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


def _payload_length(
    raw: _RawTensor,
    n_elems: int,
    log: DiagnosticLog,
    tensor_id: str,
) -> int | None:
    sizes = GGML_QUANT_SIZES.get(raw.ggml_type)
    if sizes is None:
        reason = (
            "removed from GGUF"
            if raw.ggml_type in REMOVED_GGML_TYPE_IDS
            else "undocumented GGUF type; byte length is inferred from offsets"
        )
        log.add(
            "gguf.unknown-tensor-type",
            Severity.WARNING,
            f"tensor {raw.name!r} has ggml type {raw.ggml_type}; {reason}",
            tensor_id,
        )
        return None
    block_size, _type_size = sizes
    if n_elems % block_size != 0:
        log.add(
            "gguf.tensor-length-mismatch",
            Severity.WARNING,
            f"tensor {raw.name!r} has {n_elems} elements, which is not a "
            f"multiple of block size {block_size}",
            tensor_id,
        )
    return packed_nbytes(raw.ggml_type, n_elems)


def _gap_lengths(
    raw_tensors: list[_RawTensor],
    data_start: int,
    file_size: int,
) -> dict[int, int]:
    ordered = sorted(
        range(len(raw_tensors)),
        key=lambda index: (raw_tensors[index].relative_offset, index),
    )
    gaps: dict[int, int] = {}
    for position, index in enumerate(ordered):
        start = data_start + raw_tensors[index].relative_offset
        if position + 1 < len(ordered):
            nxt = raw_tensors[ordered[position + 1]]
            end = data_start + nxt.relative_offset
        else:
            end = file_size
        gaps[index] = max(0, end - start)
    return gaps


def _absent_tensor(raw: _RawTensor, tensor_id: str, n_elems: int) -> GgufTensorIndex:
    return GgufTensorIndex(
        tensor_id,
        raw.name,
        raw.ggml_type,
        element_type_name(raw.ggml_type),
        raw.dims,
        n_elems,
        raw.relative_offset,
        Storage.ABSENT,
        None,
        None,
    )


def _describe_raw_tensor(
    raw: _RawTensor, log: DiagnosticLog
) -> tuple[GgufTensorIndex, int | None, bool]:
    """Return (partial tensor, known nbytes, refuse_payload)."""
    tensor_id = initializer_id(ROOT_GRAPH_ID, raw.name)
    if any(dim < 0 for dim in raw.dims):
        log.add(
            "gguf.implausible-shape",
            Severity.ERROR,
            f"tensor {raw.name!r} has a negative dimension",
            tensor_id,
        )
        return _absent_tensor(raw, tensor_id, 0), None, True
    n_elems = _n_elems(raw.dims)
    if n_elems > _IMPLAUSIBLE_ELEMS:
        log.add(
            "gguf.implausible-shape",
            Severity.ERROR,
            f"tensor {raw.name!r} declares {n_elems} elements",
            tensor_id,
        )
        return _absent_tensor(raw, tensor_id, n_elems), None, True
    n_bytes = _payload_length(raw, n_elems, log, tensor_id)
    return (
        GgufTensorIndex(
            tensor_id,
            raw.name,
            raw.ggml_type,
            element_type_name(raw.ggml_type),
            raw.dims,
            n_elems,
            raw.relative_offset,
            Storage.ABSENT,
            None,
            n_bytes,
        ),
        n_bytes,
        False,
    )


def _with_payload(tensor: GgufTensorIndex, payload: ByteRange) -> GgufTensorIndex:
    return GgufTensorIndex(
        tensor.id,
        tensor.name,
        tensor.ggml_type,
        tensor.element_type,
        tensor.dims,
        tensor.n_elems,
        tensor.relative_offset,
        Storage.EMBEDDED_RAW,
        payload,
        payload.length,
    )


def _locate_payloads(
    raw_tensors: list[_RawTensor],
    data_start: int,
    file_size: int,
    log: DiagnosticLog,
) -> list[GgufTensorIndex]:
    described = [_describe_raw_tensor(raw, log) for raw in raw_tensors]
    gaps = _gap_lengths(raw_tensors, data_start, file_size)
    located: list[GgufTensorIndex] = []
    ranges: list[tuple[int, int, int]] = []
    for index, (tensor, n_bytes, refuse) in enumerate(described):
        if refuse:
            located.append(tensor)
            continue
        length = n_bytes if n_bytes is not None else gaps.get(index, 0)
        offset = data_start + tensor.relative_offset
        if offset < 0 or offset + length > file_size:
            log.add(
                "gguf.tensor-range-out-of-bounds",
                Severity.ERROR,
                f"tensor {tensor.name!r} occupies "
                f"[{offset}, {offset + length}) which exceeds the "
                f"{file_size}-byte file",
                tensor.id,
            )
            located.append(tensor)
            continue
        ranges.append((offset, length, index))
        located.append(tensor)
        located[-1] = _with_payload(tensor, ByteRange(offset, length))
    overlapping = _overlapping_indices(ranges)
    if not overlapping:
        return located
    revised: list[GgufTensorIndex] = []
    for index, tensor in enumerate(located):
        if index not in overlapping or tensor.payload is None:
            revised.append(tensor)
            continue
        log.add(
            "gguf.tensor-range-overlap",
            Severity.ERROR,
            f"tensor {tensor.name!r} overlaps another tensor's payload; its "
            "bytes are not exposed",
            tensor.id,
        )
        revised.append(
            GgufTensorIndex(
                tensor.id,
                tensor.name,
                tensor.ggml_type,
                tensor.element_type,
                tensor.dims,
                tensor.n_elems,
                tensor.relative_offset,
                Storage.ABSENT,
                None,
                tensor.n_bytes,
            )
        )
    return revised


def _overlapping_indices(ranges: list[tuple[int, int, int]]) -> set[int]:
    overlapping: set[int] = set()
    ordered = sorted(ranges)
    previous_end = 0
    previous_index = -1
    for offset, length, index in ordered:
        end = offset + length
        if offset < previous_end:
            overlapping.add(index)
            if previous_index >= 0:
                overlapping.add(previous_index)
        if end > previous_end:
            previous_end = end
            previous_index = index
    return overlapping


def _note_shards(path: Path, entries: list[GgufKvEntry], log: DiagnosticLog) -> None:
    values = {item.key: item.value for item in entries}
    count = values.get("split.count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 1:
        return
    present = 0
    parent = path.parent
    stem = path.name
    # Typical names: model-00001-of-00008.gguf
    for index in range(1, count + 1):
        marker = f"-{index:05d}-of-{count:05d}.gguf"
        if any(child.name.endswith(marker) for child in parent.glob("*.gguf")):
            present += 1
            continue
        if marker in stem:
            present += 1
    if present >= count:
        return
    log.add(
        "gguf.shard-incomplete",
        Severity.WARNING,
        f"GGUF split.count is {count} but only {present} shard file(s) were "
        "found next to the opened artifact; weight_bytes covers this file only",
        path.name,
    )


def index_gguf(
    path: Path | str,
    *,
    token: CancellationToken | None = None,
) -> GgufIndex:
    """Walk a GGUF header, KV block, and tensor table without reading payloads."""
    source = Path(path)
    try:
        with ArtifactReader(source) as reader:
            return _index_open(reader, token)
    except GgufIndexError:
        raise
    except RangeReadError as error:
        raise GgufIndexError(str(error)) from error


def _index_open(reader: ArtifactReader, token: CancellationToken | None) -> GgufIndex:
    log = DiagnosticLog()
    cursor = _Cursor(reader, token)
    layout, n_tensors, n_kv = _read_header(cursor)
    if layout.endian == "big":
        log.add(
            "gguf.big-endian",
            Severity.INFO,
            f"GGUF version {layout.version} is stored big-endian; multi-byte "
            "fields are decoded with swapped byte order",
            reader.path.name,
        )
    kv_entries = _read_kv_entries(cursor, n_kv, token, log)
    alignment = _alignment_from_kv(kv_entries, log)
    raw_tensors = _read_tensor_table(cursor, n_tensors, token, log)
    remainder = cursor.pos % alignment
    data_start = cursor.pos if remainder == 0 else cursor.pos + (alignment - remainder)
    if data_start > reader.size:
        raise GgufIndexError(
            f"aligned GGUF data section starts at {data_start}, past the "
            f"{reader.size}-byte file"
        )
    content_hash = cursor.finish_identity(data_start, reader.size)
    tensors = _locate_payloads(raw_tensors, data_start, reader.size, log)
    _note_shards(reader.path, kv_entries, log)
    stats = reader.stats
    return GgufIndex(
        path=reader.path,
        version=layout.version,
        endian=layout.endian,
        alignment=alignment,
        n_tensors=n_tensors,
        n_kv=n_kv,
        data_start=data_start,
        file_size=reader.size,
        content_hash=content_hash,
        kv=tuple(kv_entries),
        tensors=tuple(tensors),
        diagnostics=log,
        stats=stats,
    )
