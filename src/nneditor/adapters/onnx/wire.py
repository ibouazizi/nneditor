"""A minimal protocol-buffer wire reader driven by byte ranges.

The generated ONNX Python bindings parse an entire ``ModelProto`` into memory,
which copies every embedded tensor payload before the viewer has decided it
needs one. This reader walks the same bytes lazily instead: it decodes the small
scalar fields it is asked for and, for length-delimited fields, records where
the payload *is* without reading it.

Only what the indexer needs is implemented. Groups (wire types 3 and 4) are
deprecated in protocol buffers and are rejected rather than skipped, because
silently ignoring them would misreport the structure of a hostile file.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Protocol

from nneditor.cancellation import CancellationToken

__all__ = [
    "MalformedProtobufError",
    "MessageCursor",
    "RangeSource",
    "ScanLimits",
    "WireField",
    "WireType",
]

_MAX_VARINT_BYTES: Final = 10
_UINT64_MASK: Final = (1 << 64) - 1
_INT64_SIGN_BIT: Final = 1 << 63


class MalformedProtobufError(ValueError):
    """Raised when the byte stream is not a well-formed protocol buffer."""


class RangeSource(Protocol):
    """The slice of :class:`~nneditor.storage.reader.ArtifactReader` used here."""

    size: int

    def read(self, offset: int, length: int, *, record: bool = True) -> bytes:
        """Return ``length`` bytes at ``offset``."""

    def record_logical(self, offset: int, length: int) -> None:
        """Record that ``length`` bytes at ``offset`` were consumed."""


class WireType(IntEnum):
    """Protocol-buffer wire types."""

    VARINT = 0
    I64 = 1
    LEN = 2
    SGROUP = 3
    EGROUP = 4
    I32 = 5


@dataclass(frozen=True, slots=True)
class ScanLimits:
    """Bounds that stop a hostile artifact from exhausting the process."""

    max_depth: int = 64
    max_fields_per_message: int = 1 << 22
    max_string_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.max_fields_per_message < 1:
            raise ValueError("max_fields_per_message must be at least 1")
        if self.max_string_bytes < 1:
            raise ValueError("max_string_bytes must be at least 1")


@dataclass(frozen=True, slots=True)
class WireField:
    """One decoded field header.

    For length-delimited fields ``offset`` and ``length`` locate the payload in
    the artifact and the payload itself has *not* been read.
    """

    number: int
    wire_type: WireType
    value: int = 0
    offset: int = 0
    length: int = 0

    @property
    def as_int64(self) -> int:
        """Interpret a varint payload as a signed 64-bit integer."""
        if self.value & _INT64_SIGN_BIT:
            return self.value - (1 << 64)
        return self.value

    @property
    def as_int32(self) -> int:
        """Interpret a varint payload as a signed 32-bit integer."""
        masked = self.value & 0xFFFFFFFF
        return masked - (1 << 32) if masked & (1 << 31) else masked


class MessageCursor:
    """Iterates the fields of one message occupying ``[start, end)``."""

    __slots__ = (
        "_depth",
        "_end",
        "_limits",
        "_pos",
        "_source",
        "_start",
        "_token",
    )

    def __init__(
        self,
        source: RangeSource,
        start: int,
        end: int,
        limits: ScanLimits | None = None,
        depth: int = 0,
        token: CancellationToken | None = None,
    ) -> None:
        self._limits = limits or ScanLimits()
        if depth >= self._limits.max_depth:
            raise MalformedProtobufError(
                f"message nesting deeper than {self._limits.max_depth} levels"
            )
        if start < 0 or end < start or end > source.size:
            raise MalformedProtobufError(
                f"message bounds [{start}, {end}) are outside the artifact"
            )
        self._source = source
        self._start = start
        self._end = end
        self._pos = start
        self._depth = depth
        self._token = token

    @property
    def limits(self) -> ScanLimits:
        """The limits this cursor and its children enforce."""
        return self._limits

    @property
    def depth(self) -> int:
        """How deeply nested this message is."""
        return self._depth

    def __iter__(self) -> Iterator[WireField]:
        self._pos = self._start
        seen = 0
        while self._pos < self._end:
            if self._token is not None:
                self._token.raise_if_cancelled()
            seen += 1
            if seen > self._limits.max_fields_per_message:
                raise MalformedProtobufError(
                    f"message holds more than {self._limits.max_fields_per_message} "
                    "fields"
                )
            yield self._read_field()

    def _read_field(self) -> WireField:
        tag = self._read_varint()
        number = tag >> 3
        if number == 0:
            raise MalformedProtobufError("field number 0 is not valid")
        try:
            wire_type = WireType(tag & 0x07)
        except ValueError as error:
            raise MalformedProtobufError(
                f"unknown wire type {tag & 0x07} for field {number}"
            ) from error

        if wire_type is WireType.VARINT:
            return WireField(number, wire_type, value=self._read_varint())
        if wire_type is WireType.I64:
            offset = self._advance(8)
            return WireField(number, wire_type, offset=offset, length=8)
        if wire_type is WireType.I32:
            offset = self._advance(4)
            return WireField(number, wire_type, offset=offset, length=4)
        if wire_type is WireType.LEN:
            length = self._read_varint()
            if length > self._end - self._pos:
                raise MalformedProtobufError(
                    f"field {number} claims {length} bytes but only "
                    f"{self._end - self._pos} remain"
                )
            offset = self._advance(length)
            return WireField(number, wire_type, offset=offset, length=length)
        raise MalformedProtobufError(
            f"group wire type {wire_type.value} in field {number} is not supported"
        )

    def _advance(self, length: int) -> int:
        if length > self._end - self._pos:
            raise MalformedProtobufError(
                f"truncated field: {length} bytes needed, "
                f"{self._end - self._pos} remain"
            )
        offset = self._pos
        self._pos += length
        return offset

    def _read_varint(self) -> int:
        # A varint is self-terminating, so its length is unknown until it is
        # examined. The look-ahead window is fetched without being counted as
        # consumed, and only the bytes that made up the value are recorded;
        # otherwise a header ending next to a tensor payload would look like a
        # read of that payload.
        window = self._source.read(
            self._pos, min(_MAX_VARINT_BYTES, self._end - self._pos), record=False
        )
        result = 0
        shift = 0
        for index, byte in enumerate(window):
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                if result > _UINT64_MASK:
                    raise MalformedProtobufError("varint exceeds 64 bits")
                self._source.record_logical(self._pos, index + 1)
                self._pos += index + 1
                return result
            shift += 7
        raise MalformedProtobufError(
            "truncated or over-long varint" if window else "unexpected end of message"
        )

    def submessage(self, wire_field: WireField) -> MessageCursor:
        """Return a cursor over a nested message's payload."""
        if wire_field.wire_type is not WireType.LEN:
            raise MalformedProtobufError(
                f"field {wire_field.number} is not length delimited"
            )
        return MessageCursor(
            self._source,
            wire_field.offset,
            wire_field.offset + wire_field.length,
            self._limits,
            self._depth + 1,
            self._token,
        )

    def payload(self, wire_field: WireField) -> bytes:
        """Read a length-delimited payload. Only call this for small fields."""
        if wire_field.length > self._limits.max_string_bytes:
            raise MalformedProtobufError(
                f"field {wire_field.number} payload of {wire_field.length} bytes "
                f"exceeds the {self._limits.max_string_bytes} byte limit"
            )
        return self._source.read(wire_field.offset, wire_field.length)

    def text(self, wire_field: WireField) -> str:
        """Read a length-delimited payload as UTF-8 text."""
        raw = self.payload(wire_field)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MalformedProtobufError(
                f"field {wire_field.number} is not valid UTF-8"
            ) from error

    def packed_varints(self, wire_field: WireField) -> tuple[int, ...]:
        """Decode a packed repeated varint field into signed 64-bit values."""
        raw = self.payload(wire_field)
        values: list[int] = []
        result = 0
        shift = 0
        started = False
        for byte in raw:
            started = True
            result |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                if shift >= _MAX_VARINT_BYTES * 7:
                    raise MalformedProtobufError("packed varint exceeds 64 bits")
                continue
            if result > _UINT64_MASK:
                # A ten-byte varint can carry up to 70 bits; reject the excess
                # exactly like _read_varint does for standalone fields.
                raise MalformedProtobufError("packed varint exceeds 64 bits")
            values.append(result - (1 << 64) if result & _INT64_SIGN_BIT else result)
            result = 0
            shift = 0
            started = False
        if started:
            raise MalformedProtobufError("packed varint field ends mid-value")
        return tuple(values)
