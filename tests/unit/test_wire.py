"""Tests for the protocol-buffer wire reader (P0.4)."""

from __future__ import annotations

import pytest

from nneditor.adapters.onnx.wire import (
    MalformedProtobufError,
    MessageCursor,
    ScanLimits,
    WireType,
)
from tests.fixtures.protobuf import length_field, tag, varint, varint_field


class MemorySource:
    """An in-memory :class:`~nneditor.adapters.onnx.wire.RangeSource`."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.size = len(data)
        self.reads: list[tuple[int, int]] = []
        self.consumed: list[tuple[int, int]] = []

    def read(self, offset: int, length: int, *, record: bool = True) -> bytes:
        self.reads.append((offset, length))
        if record:
            self.consumed.append((offset, length))
        return self._data[offset : offset + length]

    def record_logical(self, offset: int, length: int) -> None:
        self.consumed.append((offset, length))


def cursor(data: bytes, limits: ScanLimits | None = None) -> MessageCursor:
    return MessageCursor(MemorySource(data), 0, len(data), limits)


def test_varint_and_length_fields_are_decoded() -> None:
    data = varint_field(1, 300) + length_field(2, b"hello")
    fields = list(cursor(data))
    assert fields[0].number == 1
    assert fields[0].value == 300
    assert fields[1].number == 2
    assert fields[1].length == 5


def test_length_payloads_are_located_but_not_read() -> None:
    source = MemorySource(length_field(9, b"x" * 4096))
    reader = MessageCursor(source, 0, source.size, None)
    fields = list(reader)
    assert fields[0].length == 4096
    # Only the tag and length varints count as consumed.
    assert sum(length for _, length in source.consumed) == 3
    payload_start = fields[0].offset
    assert all(offset < payload_start for offset, _ in source.consumed)


def test_payload_is_read_only_when_asked_for() -> None:
    data = length_field(1, b"payload")
    reader = cursor(data)
    field = next(iter(reader))
    assert reader.payload(field) == b"payload"
    assert reader.text(field) == "payload"


def test_negative_int64_round_trips_through_two_complement() -> None:
    fields = list(cursor(varint_field(1, (1 << 64) - 5)))
    assert fields[0].as_int64 == -5


def test_int32_interpretation_masks_to_thirty_two_bits() -> None:
    fields = list(cursor(varint_field(1, (1 << 64) - 1)))
    assert fields[0].as_int32 == -1


def test_fixed_width_fields_are_skipped_without_reading() -> None:
    data = tag(1, WireType.I64) + b"\x00" * 8 + tag(2, WireType.I32) + b"\x00" * 4
    fields = list(cursor(data))
    assert [item.length for item in fields] == [8, 4]


def test_submessages_are_walked_lazily() -> None:
    inner = varint_field(1, 7) + length_field(2, b"name")
    data = length_field(3, inner)
    outer = cursor(data)
    field = next(iter(outer))
    nested = list(outer.submessage(field))
    assert nested[0].value == 7
    assert outer.submessage(field).depth == 1


def test_packed_varints_are_decoded() -> None:
    packed = varint(2) + varint(3) + varint(5)
    reader = cursor(length_field(1, packed))
    field = next(iter(reader))
    assert reader.packed_varints(field) == (2, 3, 5)


def test_packed_negative_values_are_signed() -> None:
    packed = varint((1 << 64) - 1)
    reader = cursor(length_field(1, packed))
    assert reader.packed_varints(next(iter(reader))) == (-1,)


def test_a_truncated_packed_field_is_rejected() -> None:
    reader = cursor(length_field(1, b"\x80"))
    with pytest.raises(MalformedProtobufError, match="ends mid-value"):
        reader.packed_varints(next(iter(reader)))


def test_an_over_long_packed_varint_is_rejected() -> None:
    reader = cursor(length_field(1, b"\x80" * 12 + b"\x01"))
    with pytest.raises(MalformedProtobufError, match="exceeds 64 bits"):
        reader.packed_varints(next(iter(reader)))


def test_field_number_zero_is_rejected() -> None:
    with pytest.raises(MalformedProtobufError, match="field number 0"):
        list(cursor(varint(0) + varint(1)))


def test_group_wire_types_are_rejected_rather_than_skipped() -> None:
    with pytest.raises(MalformedProtobufError, match="group wire type"):
        list(cursor(tag(1, WireType.SGROUP)))


def test_unknown_wire_types_are_rejected() -> None:
    with pytest.raises(MalformedProtobufError, match="unknown wire type"):
        list(cursor(varint(1 << 3 | 6)))


def test_a_length_beyond_the_message_is_rejected() -> None:
    data = tag(1, WireType.LEN) + varint(100) + b"short"
    with pytest.raises(MalformedProtobufError, match="claims 100 bytes"):
        list(cursor(data))


def test_a_truncated_fixed_field_is_rejected() -> None:
    with pytest.raises(MalformedProtobufError, match="truncated field"):
        list(cursor(tag(1, WireType.I64) + b"\x00" * 3))


def test_an_over_long_varint_is_rejected() -> None:
    with pytest.raises(MalformedProtobufError, match="truncated or over-long"):
        list(cursor(tag(1, WireType.VARINT) + b"\x80" * 11))


def test_a_varint_beyond_sixty_four_bits_is_rejected() -> None:
    with pytest.raises(MalformedProtobufError, match="exceeds 64 bits"):
        list(cursor(tag(1, WireType.VARINT) + b"\xff" * 9 + b"\x7f"))


def test_bounds_outside_the_source_are_rejected() -> None:
    source = MemorySource(b"abc")
    with pytest.raises(MalformedProtobufError, match="outside the artifact"):
        MessageCursor(source, 0, 99)


def test_nesting_deeper_than_the_limit_is_rejected() -> None:
    limits = ScanLimits(max_depth=2)
    data = length_field(1, length_field(1, length_field(1, b"deep")))
    outer = cursor(data, limits)
    level_one = outer.submessage(next(iter(outer)))
    with pytest.raises(MalformedProtobufError, match="nesting deeper"):
        level_one.submessage(next(iter(level_one)))


def test_too_many_fields_in_one_message_is_rejected() -> None:
    limits = ScanLimits(max_fields_per_message=3)
    data = b"".join(varint_field(1, index) for index in range(5))
    with pytest.raises(MalformedProtobufError, match="more than 3"):
        list(cursor(data, limits))


def test_an_oversized_string_payload_is_refused() -> None:
    limits = ScanLimits(max_string_bytes=4)
    reader = cursor(length_field(1, b"far too long"), limits)
    with pytest.raises(MalformedProtobufError, match="exceeds the 4 byte limit"):
        reader.payload(next(iter(reader)))


def test_invalid_utf8_text_is_reported() -> None:
    reader = cursor(length_field(1, b"\xff\xfe"))
    with pytest.raises(MalformedProtobufError, match="not valid UTF-8"):
        reader.text(next(iter(reader)))


def test_submessage_requires_a_length_delimited_field() -> None:
    reader = cursor(varint_field(1, 5))
    with pytest.raises(MalformedProtobufError, match="not length delimited"):
        reader.submessage(next(iter(reader)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": 0},
        {"max_fields_per_message": 0},
        {"max_string_bytes": 0},
    ],
)
def test_scan_limits_validate_their_own_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ScanLimits(**kwargs)


def test_limits_are_inherited_by_submessages() -> None:
    limits = ScanLimits(max_string_bytes=7)
    reader = cursor(length_field(1, length_field(1, b"payload")), limits)
    nested = reader.submessage(next(iter(reader)))
    assert nested.limits.max_string_bytes == 7


def test_iterating_twice_restarts_from_the_beginning() -> None:
    reader = cursor(varint_field(1, 1) + varint_field(2, 2))
    assert [item.number for item in reader] == [1, 2]
    assert [item.number for item in reader] == [1, 2]


def test_an_empty_message_yields_nothing() -> None:
    assert list(cursor(b"")) == []
