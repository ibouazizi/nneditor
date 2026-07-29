"""Tests for ONNX element-type metadata."""

from __future__ import annotations

import pytest

from nneditor.adapters.onnx.dtypes import (
    ELEMENT_TYPES,
    element_type_for,
    packed_byte_length,
)


@pytest.mark.parametrize(
    ("code", "name", "bits"),
    [(1, "FLOAT", 32), (7, "INT64", 64), (10, "FLOAT16", 16), (22, "INT4", 4)],
)
def test_known_types_report_their_width(code: int, name: str, bits: int) -> None:
    element_type = element_type_for(code)
    assert element_type.name == name
    assert element_type.bits == bits
    assert element_type.is_fixed_width


def test_every_registered_code_matches_its_entry() -> None:
    assert all(code == item.code for code, item in ELEMENT_TYPES.items())


def test_strings_have_no_fixed_width() -> None:
    assert not element_type_for(8).is_fixed_width
    assert packed_byte_length(element_type_for(8), 4) is None


def test_an_unknown_code_is_described_rather_than_rejected() -> None:
    element_type = element_type_for(250)
    assert element_type.name == "UNKNOWN(250)"
    assert not element_type.is_fixed_width


@pytest.mark.parametrize(
    ("code", "count", "expected"),
    [(1, 4, 16), (7, 3, 24), (9, 5, 5), (22, 3, 2), (22, 4, 2), (21, 1, 1)],
)
def test_packed_byte_length_rounds_sub_byte_types_up(
    code: int, count: int, expected: int
) -> None:
    assert packed_byte_length(element_type_for(code), count) == expected


def test_a_negative_element_count_has_no_length() -> None:
    assert packed_byte_length(element_type_for(1), -1) is None


def test_a_scalar_has_one_element_worth_of_bytes() -> None:
    assert packed_byte_length(element_type_for(1), 1) == 4
