"""Hand-built protocol-buffer bytes.

Some cases cannot be produced through the reference ``onnx`` package because
they are exactly the encodings it does not emit — packed repeated fields,
truncated varints, deprecated group markers. Building them by hand keeps those
paths under test.
"""

from __future__ import annotations

from nneditor.adapters.onnx.wire import WireType

__all__ = ["length_field", "tag", "varint", "varint_field"]


def varint(value: int) -> bytes:
    """Encode ``value`` as a base-128 varint."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def tag(number: int, wire_type: WireType) -> bytes:
    """Encode a field tag."""
    return varint(number << 3 | wire_type.value)


def length_field(number: int, payload: bytes) -> bytes:
    """Encode a length-delimited field."""
    return tag(number, WireType.LEN) + varint(len(payload)) + payload


def varint_field(number: int, value: int) -> bytes:
    """Encode a varint field."""
    return tag(number, WireType.VARINT) + varint(value)
