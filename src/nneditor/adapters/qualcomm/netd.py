"""Qualcomm DLC ``NETD``/``NETP`` stream decoding.

The DLC ``model`` member is a FlatBuffers buffer (file identifier ``NETD``)
behind an 8-byte prefix, and ``model.params`` is its payload directory
(``NETP``) mapping tensor names to byte ranges inside ``model.params.bin``.
Qualcomm publishes no schema; the field indices used here were
reverse-engineered and validated against QAIRT converter 2.38 and 2.45
outputs, where every op edge resolved and every static payload matched its
declared shape and dtype. A stream that does not match this layout raises
:class:`NetdError`, and the caller falls back to container-only inspection —
decoding never guesses.

Layout summary (field indices per table):

- root: ``0`` graph vector — graph: ``0`` name, ``1`` op vector, ``2``
  tensor vector.
- op: ``0`` name, ``1`` type, ``2`` input names, ``3`` output names, ``4``
  attribute vector.
- attribute: ``0`` name, ``3`` scalar value table (``0`` QNN dtype code,
  ``1`` numeric, ``3`` string), ``4`` static parameter tensor table.
- tensor: ``0`` id, ``1`` name, ``2`` role (0 input, 1 output, 3 native,
  4 static), ``3`` shape, ``5`` quantization table, ``6`` stored dtype
  code, ``7`` source dtype code.
- quantization: ``0`` encoding (absent/0 scale-offset, 1 per-axis,
  0x7FFFFFFF undefined), ``2`` scale-offset table (``0`` bitwidth, ``1``
  min, ``2`` max, ``3`` scale, ``4`` offset), ``3`` axis table (``0`` axis,
  ``1`` scale-offset vector).
- NETP root: ``0`` graph vector — graph: ``0`` name, ``1`` static records,
  ``2`` per-op records (op order mirrors NETD); record: ``0`` tensor name,
  ``3`` location (``0`` offset, ``1`` size).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "DlcAttribute",
    "DlcGraph",
    "DlcOp",
    "DlcTensor",
    "NetdError",
    "PayloadLocation",
    "QuantInfo",
    "decode_netd",
    "decode_netp",
    "qnn_dtype_name",
]

_STREAM_PREFIX_MAGIC: Final = 0x00040AD5
_STREAM_PREFIX_SIZE: Final = 8
_NETD_IDENTIFIER: Final = b"NETD"
_NETP_IDENTIFIER: Final = b"NETP"

_MAX_VECTOR_ELEMENTS: Final = 1_000_000
_MAX_STRING_BYTES: Final = 65_536

_TENSOR_ROLE_INPUT: Final = 0
_TENSOR_ROLE_OUTPUT: Final = 1
_TENSOR_ROLE_STATIC: Final = 4

_ENCODING_UNDEFINED: Final = 0x7FFFFFFF
_ENCODING_PER_AXIS: Final = 1

# QNN_DATATYPE codes: the low byte is the width in decimal-as-hex, the high
# byte the family (0 signed int, 1 unsigned int, 2 float, 3 signed
# fixed-point, 4 unsigned fixed-point, 5 bool). Fixed-point storage maps to
# the integer type of the stored bytes.
_QNN_DTYPES: Final[dict[int, str]] = {
    0x0008: "int8",
    0x0016: "int16",
    0x0032: "int32",
    0x0064: "int64",
    0x0108: "uint8",
    0x0116: "uint16",
    0x0132: "uint32",
    0x0164: "uint64",
    0x0216: "float16",
    0x0232: "float32",
    0x0308: "int8",
    0x0316: "int16",
    0x0332: "int32",
    0x0408: "uint8",
    0x0416: "uint16",
    0x0432: "uint32",
    0x0508: "bool",
}
_QNN_STRING_CODE: Final = 0x0608
_QNN_FLOAT_CODES: Final = frozenset({0x0216, 0x0232})


class NetdError(ValueError):
    """Raised when a stream does not match the validated NETD/NETP layout."""


def qnn_dtype_name(code: int) -> str | None:
    """The canonical dtype name for a QNN datatype code, or ``None``."""
    return _QNN_DTYPES.get(code)


# -- bounds-checked FlatBuffers access ---------------------------------------


class _Buffer:
    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        self.data = data

    def _check(self, offset: int, length: int) -> None:
        if offset < 0 or offset + length > len(self.data):
            raise NetdError(f"read of {length} bytes at {offset} leaves the buffer")

    def u8(self, offset: int) -> int:
        self._check(offset, 1)
        return self.data[offset]

    def u16(self, offset: int) -> int:
        self._check(offset, 2)
        return int(struct.unpack_from("<H", self.data, offset)[0])

    def u32(self, offset: int) -> int:
        self._check(offset, 4)
        return int(struct.unpack_from("<I", self.data, offset)[0])

    def i32(self, offset: int) -> int:
        self._check(offset, 4)
        return int(struct.unpack_from("<i", self.data, offset)[0])

    def f32(self, offset: int) -> float:
        self._check(offset, 4)
        return float(struct.unpack_from("<f", self.data, offset)[0])


class _Table:
    """One FlatBuffers table: bounds-checked vtable-indexed field access."""

    __slots__ = ("buffer", "position")

    def __init__(self, buffer: _Buffer, position: int) -> None:
        self.buffer = buffer
        self.position = position

    @classmethod
    def root(cls, data: bytes, identifier: bytes) -> _Table:
        buffer = _Buffer(data)
        if data[4:8] != identifier:
            raise NetdError(f"buffer identifier {data[4:8]!r} is not {identifier!r}")
        return cls(buffer, buffer.u32(0))

    def _field_position(self, index: int) -> int | None:
        vtable = self.position - self.buffer.i32(self.position)
        vtable_length = self.buffer.u16(vtable)
        slot = 4 + 2 * index
        if slot + 2 > vtable_length:
            return None
        relative = self.buffer.u16(vtable + slot)
        if relative == 0:
            return None
        return self.position + relative

    def u8_field(self, index: int, default: int = 0) -> int:
        position = self._field_position(index)
        return default if position is None else self.buffer.u8(position)

    def u32_field(self, index: int, default: int = 0) -> int:
        position = self._field_position(index)
        return default if position is None else self.buffer.u32(position)

    def f32_field(self, index: int, default: float = 0.0) -> float:
        position = self._field_position(index)
        return default if position is None else self.buffer.f32(position)

    def i32_field(self, index: int, default: int = 0) -> int:
        position = self._field_position(index)
        return default if position is None else self.buffer.i32(position)

    def _indirect(self, position: int) -> int:
        return position + self.buffer.u32(position)

    def table_field(self, index: int) -> _Table | None:
        position = self._field_position(index)
        if position is None:
            return None
        return _Table(self.buffer, self._indirect(position))

    def string_field(self, index: int) -> str | None:
        position = self._field_position(index)
        if position is None:
            return None
        start = self._indirect(position)
        length = self.buffer.u32(start)
        if length > _MAX_STRING_BYTES:
            raise NetdError(f"string of {length} bytes exceeds the ceiling")
        self.buffer._check(start + 4, length)
        return self.buffer.data[start + 4 : start + 4 + length].decode(
            "utf-8", errors="replace"
        )

    def _vector(self, index: int) -> tuple[int, int] | None:
        position = self._field_position(index)
        if position is None:
            return None
        start = self._indirect(position)
        count = self.buffer.u32(start)
        if count > _MAX_VECTOR_ELEMENTS:
            raise NetdError(f"vector of {count} elements exceeds the ceiling")
        return start + 4, count

    def table_vector(self, index: int) -> list[_Table]:
        located = self._vector(index)
        if located is None:
            return []
        start, count = located
        return [
            _Table(self.buffer, self._indirect(start + 4 * item))
            for item in range(count)
        ]

    def string_vector(self, index: int) -> list[str]:
        located = self._vector(index)
        if located is None:
            return []
        start, count = located
        strings: list[str] = []
        for item in range(count):
            position = self._indirect(start + 4 * item)
            length = self.buffer.u32(position)
            if length > _MAX_STRING_BYTES:
                raise NetdError(f"string of {length} bytes exceeds the ceiling")
            self.buffer._check(position + 4, length)
            strings.append(
                self.buffer.data[position + 4 : position + 4 + length].decode(
                    "utf-8", errors="replace"
                )
            )
        return strings

    def u32_vector(self, index: int) -> list[int]:
        located = self._vector(index)
        if located is None:
            return []
        start, count = located
        self.buffer._check(start, 4 * count)
        return list(struct.unpack_from(f"<{count}I", self.buffer.data, start))


def _strip_prefix(data: bytes, identifier: bytes) -> bytes:
    if len(data) < _STREAM_PREFIX_SIZE + 8:
        raise NetdError("stream is shorter than its prefix and root")
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != _STREAM_PREFIX_MAGIC:
        raise NetdError(f"stream prefix magic {magic:#x} is not recognized")
    payload = data[_STREAM_PREFIX_SIZE:]
    if payload[4:8] != identifier:
        raise NetdError(f"stream identifier {payload[4:8]!r} is not {identifier!r}")
    return payload


# -- decoded model -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuantInfo:
    """One tensor's quantization encoding, per-tensor or per-axis."""

    per_axis: bool
    bitwidth: int
    scale: float
    offset: int
    minimum: float
    maximum: float
    axis: int | None = None
    channel_count: int | None = None


@dataclass(frozen=True, slots=True)
class DlcTensor:
    name: str
    role: int
    shape: tuple[int, ...]
    stored_dtype_code: int
    source_dtype_code: int
    quantization: QuantInfo | None

    @property
    def is_static(self) -> bool:
        return self.role == _TENSOR_ROLE_STATIC

    @property
    def is_input(self) -> bool:
        return self.role == _TENSOR_ROLE_INPUT

    @property
    def is_output(self) -> bool:
        return self.role == _TENSOR_ROLE_OUTPUT


@dataclass(frozen=True, slots=True)
class DlcAttribute:
    name: str
    string_value: str | None = None
    int_value: int | None = None
    float_value: float | None = None
    tensor: DlcTensor | None = None


@dataclass(frozen=True, slots=True)
class DlcOp:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[DlcAttribute, ...]


@dataclass(frozen=True, slots=True)
class PayloadLocation:
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class DlcGraph:
    name: str
    ops: tuple[DlcOp, ...]
    tensors: tuple[DlcTensor, ...]
    static_payloads: dict[str, PayloadLocation] = field(default_factory=dict)
    parameter_payloads: tuple[dict[str, PayloadLocation], ...] = ()


def _decode_quantization(table: _Table | None) -> QuantInfo | None:
    if table is None:
        return None
    encoding = table.u32_field(0, default=0)
    if encoding == _ENCODING_UNDEFINED:
        return None
    if encoding == _ENCODING_PER_AXIS:
        axis_table = table.table_field(3)
        if axis_table is None:
            return None
        entries = axis_table.table_vector(1)
        if not entries:
            return None
        first = entries[0]
        return QuantInfo(
            per_axis=True,
            bitwidth=first.u32_field(0),
            scale=first.f32_field(3),
            offset=first.i32_field(4),
            minimum=first.f32_field(1),
            maximum=first.f32_field(2),
            axis=axis_table.u32_field(0),
            channel_count=len(entries),
        )
    scale_offset = table.table_field(2)
    if scale_offset is None:
        return None
    return QuantInfo(
        per_axis=False,
        bitwidth=scale_offset.u32_field(0),
        scale=scale_offset.f32_field(3),
        offset=scale_offset.i32_field(4),
        minimum=scale_offset.f32_field(1),
        maximum=scale_offset.f32_field(2),
    )


def _decode_tensor(table: _Table) -> DlcTensor:
    name = table.string_field(1)
    if not name:
        raise NetdError("a tensor table declares no name")
    return DlcTensor(
        name=name,
        role=table.u32_field(2, default=0),
        shape=tuple(table.u32_vector(3)),
        stored_dtype_code=table.u32_field(6),
        source_dtype_code=table.u32_field(7),
        quantization=_decode_quantization(table.table_field(5)),
    )


def _decode_attribute(table: _Table) -> DlcAttribute:
    name = table.string_field(0) or ""
    tensor_table = table.table_field(4)
    if tensor_table is not None:
        return DlcAttribute(name=name, tensor=_decode_tensor(tensor_table))
    scalar = table.table_field(3)
    if scalar is None:
        return DlcAttribute(name=name)
    code = scalar.u32_field(0)
    if code == _QNN_STRING_CODE:
        return DlcAttribute(name=name, string_value=scalar.string_field(3))
    if code in _QNN_FLOAT_CODES:
        return DlcAttribute(name=name, float_value=scalar.f32_field(1))
    return DlcAttribute(name=name, int_value=scalar.u32_field(1))


def _decode_op(table: _Table) -> DlcOp:
    name = table.string_field(0)
    op_type = table.string_field(1)
    if not name or not op_type:
        raise NetdError("an op table declares no name or type")
    return DlcOp(
        name=name,
        op_type=op_type,
        inputs=tuple(table.string_vector(2)),
        outputs=tuple(table.string_vector(3)),
        attributes=tuple(
            _decode_attribute(attribute) for attribute in table.table_vector(4)
        ),
    )


def decode_netd(model_stream: bytes) -> list[DlcGraph]:
    """Decode a DLC ``model`` member into graphs of ops and tensors.

    Raises :class:`NetdError` when the stream does not match the validated
    layout; the caller is expected to fall back rather than guess.
    """
    payload = _strip_prefix(model_stream, _NETD_IDENTIFIER)
    root = _Table.root(payload, _NETD_IDENTIFIER)
    graphs: list[DlcGraph] = []
    for graph_table in root.table_vector(0):
        name = graph_table.string_field(0) or "graph"
        ops = tuple(_decode_op(item) for item in graph_table.table_vector(1))
        tensors = tuple(_decode_tensor(item) for item in graph_table.table_vector(2))
        graphs.append(DlcGraph(name=name, ops=ops, tensors=tensors))
    if not graphs:
        raise NetdError("the stream declares no graphs")
    return graphs


def _decode_location(record: _Table) -> tuple[str, PayloadLocation] | None:
    name = record.string_field(0)
    location = record.table_field(3)
    if not name or location is None:
        return None
    return name, PayloadLocation(
        offset=location.u32_field(0), size=location.u32_field(1)
    )


def decode_netp(
    params_stream: bytes,
) -> dict[str, tuple[dict[str, PayloadLocation], list[dict[str, PayloadLocation]]]]:
    """Decode ``model.params`` into per-graph payload directories.

    Returns ``{graph name: (static payloads, per-op parameter payloads)}``
    where the per-op list mirrors the NETD op order.
    """
    payload = _strip_prefix(params_stream, _NETP_IDENTIFIER)
    root = _Table.root(payload, _NETP_IDENTIFIER)
    directory: dict[
        str, tuple[dict[str, PayloadLocation], list[dict[str, PayloadLocation]]]
    ] = {}
    for graph_table in root.table_vector(0):
        name = graph_table.string_field(0) or "graph"
        statics: dict[str, PayloadLocation] = {}
        for record in graph_table.table_vector(1):
            decoded = _decode_location(record)
            if decoded is not None:
                statics[decoded[0]] = decoded[1]
        per_op: list[dict[str, PayloadLocation]] = []
        for op_record in graph_table.table_vector(2):
            parameters: dict[str, PayloadLocation] = {}
            for record in op_record.table_vector(0):
                decoded = _decode_location(record)
                if decoded is not None:
                    parameters[decoded[0]] = decoded[1]
            per_op.append(parameters)
        directory[name] = (statics, per_op)
    return directory
