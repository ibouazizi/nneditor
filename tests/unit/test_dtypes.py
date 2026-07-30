"""Tests for element-type metadata.

Covers the ONNX ``TensorProto`` table, the shared IR dtype registry that is
the single width/decoder authority, and the consistency of every adapter's
private dtype table with that authority.
"""

from __future__ import annotations

import math

import pytest

from nneditor.adapters.onnx.dtypes import (
    ELEMENT_TYPES,
    element_type_for,
    packed_byte_length,
)
from nneditor.analysis.statistics import (
    decode_packed,
    decode_unavailability,
    element_width,
)
from nneditor.ir.dtypes import DTYPES, dtype_info


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


def _decode_one(name: str, code: int) -> float:
    info = DTYPES[name]
    assert info.decoder is not None
    (value,) = tuple(info.decoder(bytes([code])))
    return value


class TestRegistryInvariants:
    def test_names_are_canonical_lowercase(self) -> None:
        assert all(name == name.lower() for name in DTYPES)
        assert all(info.name == name for name, info in DTYPES.items())

    def test_every_dtype_decodes_or_explains_itself(self) -> None:
        for info in DTYPES.values():
            assert (info.decoder is None) != (info.unavailable_reason is None), (
                info.name
            )

    def test_bits_and_byte_width_agree(self) -> None:
        for info in DTYPES.values():
            if info.byte_width is not None:
                assert info.byte_width * 8 == info.bits, info.name
            if info.decoder is not None:
                assert info.byte_width is not None, info.name

    def test_unknown_names_have_no_record(self) -> None:
        assert dtype_info("float128") is None
        assert dtype_info("FLOAT32") is None


class TestFloat8Decoding:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0x38, 1.0),
            (0x30, 0.5),
            (0x40, 2.0),
            (0x7E, 448.0),
            (0x01, 2.0**-9),
            (0xB8, -1.0),
        ],
    )
    def test_e4m3fn_exact_values(self, code: int, expected: float) -> None:
        assert _decode_one("float8e4m3fn", code) == expected

    def test_e4m3fn_specials(self) -> None:
        assert math.isnan(_decode_one("float8e4m3fn", 0x7F))
        assert math.isnan(_decode_one("float8e4m3fn", 0xFF))
        assert _decode_one("float8e4m3fn", 0x00) == 0.0
        negative_zero = _decode_one("float8e4m3fn", 0x80)
        assert negative_zero == 0.0 and math.copysign(1.0, negative_zero) == -1.0

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0x3C, 1.0),
            (0x7B, 57344.0),
            (0x01, 2.0**-16),
            (0x04, 2.0**-14),
            (0xBC, -1.0),
        ],
    )
    def test_e5m2_exact_values(self, code: int, expected: float) -> None:
        assert _decode_one("float8e5m2", code) == expected

    def test_e5m2_specials(self) -> None:
        assert _decode_one("float8e5m2", 0x7C) == math.inf
        assert _decode_one("float8e5m2", 0xFC) == -math.inf
        assert math.isnan(_decode_one("float8e5m2", 0x7D))
        assert math.isnan(_decode_one("float8e5m2", 0xFF))

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(0x40, 1.0), (0x7F, 240.0), (0x01, 2.0**-10), (0xC0, -1.0)],
    )
    def test_e4m3fnuz_exact_values(self, code: int, expected: float) -> None:
        assert _decode_one("float8e4m3fnuz", code) == expected

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(0x40, 1.0), (0x7F, 57344.0), (0x01, 2.0**-17), (0xC0, -1.0)],
    )
    def test_e5m2fnuz_exact_values(self, code: int, expected: float) -> None:
        assert _decode_one("float8e5m2fnuz", code) == expected

    def test_fnuz_formats_reserve_only_0x80_for_nan(self) -> None:
        for name in ("float8e4m3fnuz", "float8e5m2fnuz"):
            assert math.isnan(_decode_one(name, 0x80)), name
            non_finite = [
                code
                for code in range(256)
                if not math.isfinite(_decode_one(name, code))
            ]
            assert non_finite == [0x80], name

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(0x7F, 1.0), (0x00, 2.0**-127), (0xFE, 2.0**127)],
    )
    def test_e8m0_exact_values(self, code: int, expected: float) -> None:
        assert _decode_one("float8e8m0", code) == expected

    def test_e8m0_nan(self) -> None:
        assert math.isnan(_decode_one("float8e8m0", 0xFF))

    def test_packed_sequences_decode_elementwise(self) -> None:
        raw = bytes([0x38, 0x40, 0xB8, 0x00])
        decoded = decode_packed("float8e4m3fn", raw)
        assert decoded is not None
        assert list(decoded) == [1.0, 2.0, -1.0, 0.0]


class TestConsistencyWithAdapterTables:
    def test_onnx_widths_agree_with_the_registry(self) -> None:
        from nneditor.adapters.onnx.to_ir import _element_name

        for element_type in ELEMENT_TYPES.values():
            name = _element_name(element_type)
            assert name is not None
            info = dtype_info(name)
            assert info is not None, name
            assert info.bits == element_type.bits, name
            if info.byte_width is not None:
                assert packed_byte_length(element_type, 1) == info.byte_width

    def test_pytorch_names_agree_with_the_registry(self) -> None:
        from nneditor.adapters.pytorch.scalar_types import (
            SERDE_SCALAR_TYPES,
            STORAGE_CLASS_TYPES,
            TORCH_DTYPE_NAMES,
            element_width_bytes,
        )

        names = (
            set(SERDE_SCALAR_TYPES.values())
            | set(STORAGE_CLASS_TYPES.values())
            | set(TORCH_DTYPE_NAMES.values())
        )
        for name in names:
            info = dtype_info(name)
            assert info is not None, name
            assert info.byte_width is not None, name
            assert element_width_bytes(name) == info.byte_width

    def test_typed_layouts_agree_with_the_registry(self) -> None:
        from nneditor.adapters.onnx.typed_data import _LAYOUTS

        for name, layout in _LAYOUTS.items():
            info = dtype_info(name)
            assert info is not None, name
            assert layout.bytes_per_element == info.byte_width, name

    def test_safetensors_dtypes_agree_with_the_registry(self) -> None:
        from nneditor.adapters.pytorch.safetensors import _DTYPE_TO_IR

        for name in _DTYPE_TO_IR.values():
            info = dtype_info(name)
            assert info is not None, name
            assert info.byte_width is not None and info.decoder is not None, name

    def test_jax_names_agree_with_the_registry(self) -> None:
        from nneditor.adapters.jax.mlir_text import _ELEMENT_TYPES

        for name in _ELEMENT_TYPES.values():
            assert dtype_info(name) is not None, name

    def test_statistics_delegates_to_the_registry(self) -> None:
        for name, info in DTYPES.items():
            if info.decoder is not None:
                assert element_width(name) == info.byte_width, name
                assert decode_unavailability(name) is None, name
            else:
                assert element_width(name) is None, name
                assert decode_packed(name, b"\x00" * 16) is None, name
                assert decode_unavailability(name) == info.unavailable_reason, name

    def test_unknown_dtypes_still_get_a_reason(self) -> None:
        reason = decode_unavailability("unknown(250)")
        assert reason is not None and "not a recognized dtype" in reason

    def test_sub_byte_reason_mentions_the_packing(self) -> None:
        reason = decode_unavailability("int4")
        assert reason is not None and "packed two per byte" in reason
