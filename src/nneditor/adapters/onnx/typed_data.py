"""Materializes embedded-typed tensor values on demand (task P3.2).

ONNX can store tensor values in typed protobuf fields (``float_data``,
``int32_data``, ...) instead of raw bytes. Such a tensor has no contiguous
payload to range-read: providing *any* of its bytes means decoding the whole
field — a full parse of that one ``TensorProto`` message, whose span the
indexer records without reading it (``TensorRef.typed_span``).

The output is the tensor's *packed* little-endian representation — exactly
the bytes ``raw_data`` would have held — so slices, statistics, and edits
treat typed tensors identically to raw ones downstream. Narrow types ride in
wider proto fields (float16 bit patterns and int8 values both travel in
``int32_data``), which is why each element type carries its own source field
and packer.

Decoding checkpoints its cancellation token between chunks: a 100-million
element typed tensor must not wedge the job that asked for it.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from nneditor.adapters.onnx.wire import MessageCursor, ScanLimits, WireField, WireType
from nneditor.cancellation import CancellationToken
from nneditor.ir.core import TensorRef
from nneditor.storage.reader import ArtifactReader
from nneditor.storage.store import TensorUnavailableError

__all__ = ["materialize_typed_tensor", "packed_length_for"]

_TENSOR_FLOAT_DATA: Final = 4
_TENSOR_INT32_DATA: Final = 5
_TENSOR_INT64_DATA: Final = 7
_TENSOR_DOUBLE_DATA: Final = 10
_TENSOR_UINT64_DATA: Final = 11

_CHECKPOINT_EVERY: Final = 65536
"""Elements decoded between cancellation checkpoints."""

_DECODE_CHUNK: Final = 1 << 20
"""Bytes fetched between cancellation checkpoints while reading one field."""

_MAX_VARINT_WIRE_BYTES: Final = 10
"""Longest legal wire encoding of one 64-bit varint element."""

_UINT64_MASK: Final = (1 << 64) - 1
_INT64_SIGN_BIT: Final = 1 << 63


@dataclass(frozen=True, slots=True)
class _TypedLayout:
    """How one element type travels in typed fields and packs into bytes."""

    field_number: int
    bytes_per_element: int
    pack: Callable[[Sequence[int | float]], bytes]


def _packer(fmt: str) -> Callable[[Sequence[int | float]], bytes]:
    def pack(values: Sequence[int | float]) -> bytes:
        return struct.pack(f"<{len(values)}{fmt}", *values)

    return pack


def _mask_packer(fmt: str, bits: int) -> Callable[[Sequence[int | float]], bytes]:
    """Packs narrow integers that ride sign-extended inside int32 entries."""
    mask = (1 << bits) - 1

    def pack(values: Sequence[int | float]) -> bytes:
        return struct.pack(f"<{len(values)}{fmt}", *(int(v) & mask for v in values))

    return pack


_LAYOUTS: Final[dict[str, _TypedLayout]] = {
    "float32": _TypedLayout(_TENSOR_FLOAT_DATA, 4, _packer("f")),
    "float64": _TypedLayout(_TENSOR_DOUBLE_DATA, 8, _packer("d")),
    "int64": _TypedLayout(_TENSOR_INT64_DATA, 8, _packer("q")),
    # Unsigned fields travel as varints that the wire layer sign-extends;
    # masking restores the unsigned bit pattern.
    "uint64": _TypedLayout(_TENSOR_UINT64_DATA, 8, _mask_packer("Q", 64)),
    "uint32": _TypedLayout(_TENSOR_UINT64_DATA, 4, _mask_packer("I", 32)),
    "int32": _TypedLayout(_TENSOR_INT32_DATA, 4, _packer("i")),
    "int16": _TypedLayout(_TENSOR_INT32_DATA, 2, _packer("h")),
    "uint16": _TypedLayout(_TENSOR_INT32_DATA, 2, _mask_packer("H", 16)),
    "int8": _TypedLayout(_TENSOR_INT32_DATA, 1, _packer("b")),
    "uint8": _TypedLayout(_TENSOR_INT32_DATA, 1, _mask_packer("B", 8)),
    "bool": _TypedLayout(_TENSOR_INT32_DATA, 1, _mask_packer("B", 8)),
    # float16/bfloat16 travel as raw bit patterns in int32_data.
    "float16": _TypedLayout(_TENSOR_INT32_DATA, 2, _mask_packer("H", 16)),
    "bfloat16": _TypedLayout(_TENSOR_INT32_DATA, 2, _mask_packer("H", 16)),
}


def packed_length_for(tensor: TensorRef) -> int | None:
    """Bytes the materialized tensor will hold, or ``None`` if unsupported."""
    layout = _LAYOUTS.get(tensor.element_type)
    if layout is None:
        return None
    count = 1
    for dim in tensor.dims:
        count *= dim
    return count * layout.bytes_per_element


def _wire_bytes_per_element(layout: _TypedLayout) -> int:
    """Most wire bytes one element may occupy inside the typed field."""
    if layout.field_number == _TENSOR_FLOAT_DATA:
        return 4
    if layout.field_number == _TENSOR_DOUBLE_DATA:
        return 8
    return _MAX_VARINT_WIRE_BYTES


def _read_field_payload(
    reader: ArtifactReader,
    item: WireField,
    token: CancellationToken | None,
) -> bytes:
    """Fetch one field payload in bounded chunks with cancellation checkpoints."""
    if item.length <= _DECODE_CHUNK:
        return reader.read(item.offset, item.length)
    chunks: list[bytes] = []
    position = 0
    while position < item.length:
        if token is not None:
            token.raise_if_cancelled()
        step = min(_DECODE_CHUNK, item.length - position)
        chunks.append(reader.read(item.offset + position, step))
        position += step
    return b"".join(chunks)


def _decode_fixed(
    reader: ArtifactReader,
    item: WireField,
    fmt: str,
    token: CancellationToken | None,
) -> list[float]:
    width = struct.calcsize(f"<{fmt}")
    if item.length % width:
        raise TensorUnavailableError(
            f"typed field payload of {item.length} bytes is not a multiple of "
            f"its {width}-byte element width"
        )
    raw = _read_field_payload(reader, item, token)
    return [value[0] for value in struct.iter_unpack(f"<{fmt}", raw)]


def _decode_packed_varints(raw: bytes, token: CancellationToken | None) -> list[int]:
    """Decode packed varints as signed 64-bit values, checkpointing regularly.

    This is the large-payload path :meth:`MessageCursor.packed_varints` cannot
    serve: the cursor's ``payload`` read is capped for hostile scans, while a
    typed tensor legitimately packs hundreds of megabytes into one field.
    """
    values: list[int] = []
    result = 0
    shift = 0
    started = False
    for byte in raw:
        started = True
        result |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift >= _MAX_VARINT_WIRE_BYTES * 7:
                raise TensorUnavailableError(
                    "typed field holds a varint longer than ten bytes"
                )
            continue
        if result > _UINT64_MASK:
            raise TensorUnavailableError("typed field holds a varint beyond 64 bits")
        values.append(result - (1 << 64) if result & _INT64_SIGN_BIT else result)
        result = 0
        shift = 0
        started = False
        if token is not None and not len(values) % _CHECKPOINT_EVERY:
            token.raise_if_cancelled()
    if started:
        raise TensorUnavailableError("typed field ends mid-varint")
    return values


def materialize_typed_tensor(
    reader: ArtifactReader,
    tensor: TensorRef,
    *,
    token: CancellationToken | None = None,
) -> bytes:
    """Decode a typed-field tensor into its packed byte representation.

    ``reader`` must be open on the model file that ``tensor.typed_span``
    indexes. Raises :class:`TensorUnavailableError` for element types whose
    typed layout is not modelled (strings, unknown codes).
    """
    layout = _LAYOUTS.get(tensor.element_type)
    if layout is None:
        raise TensorUnavailableError(
            f"tensor {tensor.id!r} has element type {tensor.element_type!r}, "
            "whose typed-field layout is not supported; only raw or external "
            "storage can provide its bytes"
        )
    assert tensor.typed_span is not None
    span = tensor.typed_span

    # The scan bounds scale with the *declared* tensor rather than a global
    # constant: a legitimate hundred-megabyte typed initializer decodes
    # normally, while a hostile field larger than the shape can justify is
    # refused before a byte of it is buffered.
    count = 1
    for dim in tensor.dims:
        count *= dim
    defaults = ScanLimits()
    field_ceiling = max(
        defaults.max_string_bytes, count * _wire_bytes_per_element(layout)
    )
    limits = ScanLimits(
        max_fields_per_message=max(defaults.max_fields_per_message, count + 16),
        max_string_bytes=field_ceiling,
    )
    cursor = MessageCursor(
        reader, span.offset, span.offset + span.length, limits, token=token
    )

    values: list[int | float] = []
    chunks: list[bytes] = []

    def flush() -> None:
        if values:
            chunks.append(layout.pack(values))
            values.clear()

    for item in cursor:
        if item.number != layout.field_number:
            continue
        if token is not None:
            token.raise_if_cancelled()
        if item.wire_type is WireType.LEN and item.length > field_ceiling:
            raise TensorUnavailableError(
                f"tensor {tensor.id!r} typed field holds {item.length} bytes "
                f"but its declared shape and type justify at most {field_ceiling}"
            )
        if layout.field_number == _TENSOR_FLOAT_DATA:
            if item.wire_type is WireType.I32:
                raw = reader.read(item.offset, 4)
                values.append(struct.unpack("<f", raw)[0])
            elif item.wire_type is WireType.LEN:
                values.extend(_decode_fixed(reader, item, "f", token))
        elif layout.field_number == _TENSOR_DOUBLE_DATA:
            if item.wire_type is WireType.I64:
                raw = reader.read(item.offset, 8)
                values.append(struct.unpack("<d", raw)[0])
            elif item.wire_type is WireType.LEN:
                values.extend(_decode_fixed(reader, item, "d", token))
        else:  # varint-backed integer fields
            if item.wire_type is WireType.VARINT:
                values.append(item.as_int64)
            elif item.wire_type is WireType.LEN:
                raw = _read_field_payload(reader, item, token)
                values.extend(_decode_packed_varints(raw, token))
        if len(values) >= _CHECKPOINT_EVERY:
            if token is not None:
                token.raise_if_cancelled()
            flush()
    flush()

    packed = b"".join(chunks)
    expected = packed_length_for(tensor)
    if expected is not None and len(packed) != expected:
        raise TensorUnavailableError(
            f"tensor {tensor.id!r} typed data holds {len(packed)} packed bytes "
            f"but its shape and type imply {expected}"
        )
    return packed
