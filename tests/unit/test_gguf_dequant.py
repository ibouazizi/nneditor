"""Decode public ggml packed types used by the numpy LLM worker."""

from __future__ import annotations

import numpy as np
import pytest

from nneditor.adapters.gguf.dequant import GgufDequantError, dequantize


def test_float32_round_trip_respects_endian() -> None:
    values = np.array([[1.0, -2.5], [0.25, 8.0]], dtype="<f4")
    decoded = dequantize("float32", (2, 2), values.tobytes(), endian="little")
    np.testing.assert_array_equal(decoded, values)
    swapped = dequantize("float32", (2, 2), values.byteswap().tobytes(), endian="big")
    np.testing.assert_array_equal(swapped, values)


def test_bfloat16_shifts_into_float32() -> None:
    # 1.0 in bfloat16 is 0x3f80; 2.0 is 0x4000.
    payload = np.array([0x3F80, 0x4000], dtype="<u2").tobytes()
    decoded = dequantize("bfloat16", (2,), payload)
    np.testing.assert_array_equal(decoded, np.array([1.0, 2.0], dtype=np.float32))


def test_q8_0_scales_int8_blocks() -> None:
    scale = np.float16(0.5)
    qs = np.arange(32, dtype=np.int8)
    payload = np.asarray([scale], dtype="<f2").tobytes() + qs.tobytes()
    decoded = dequantize("q8_0", (32,), payload)
    np.testing.assert_allclose(decoded, qs.astype(np.float32) * 0.5)


def test_q4_0_uses_nibble_offset() -> None:
    scale = np.float16(0.5)
    # Low nibbles 0, high nibbles 0 → (0 - 8) * 0.5 = -4 for every element.
    payload = np.asarray([scale], dtype="<f2").tobytes() + bytes(16)
    decoded = dequantize("q4_0", (32,), payload)
    np.testing.assert_allclose(decoded, np.full(32, -4.0, dtype=np.float32))


def test_q4_k_applies_superblock_scale_and_min() -> None:
    scale = np.float16(1.0)
    minimum = np.float16(0.0)
    # sc=1, min=0 for all eight 6-bit pairs. See get_scale_min_k4.
    scales = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    qs = bytes([0x01] * 128)  # low nibble 1, high nibble 0
    payload = (
        np.asarray([scale], dtype="<f2").tobytes()
        + np.asarray([minimum], dtype="<f2").tobytes()
        + scales
        + qs
    )
    decoded = dequantize("q4_k", (256,), payload)
    expected = np.zeros(256, dtype=np.float32)
    for group in range(4):
        expected[group * 64 : group * 64 + 32] = 1.0
    np.testing.assert_allclose(decoded, expected)


def test_q4_1_adds_the_block_minimum() -> None:
    scale = np.float16(0.5)
    minimum = np.float16(1.0)
    payload = (
        np.asarray([scale], dtype="<f2").tobytes()
        + np.asarray([minimum], dtype="<f2").tobytes()
        + bytes(16)
    )
    decoded = dequantize("q4_1", (32,), payload)
    np.testing.assert_allclose(decoded, np.full(32, 1.0, dtype=np.float32))


def test_q8_1_scales_int8_and_ignores_the_sum() -> None:
    scale = np.float16(0.25)
    total = np.float16(3.0)
    qs = np.arange(32, dtype=np.int8)
    payload = (
        np.asarray([scale], dtype="<f2").tobytes()
        + np.asarray([total], dtype="<f2").tobytes()
        + qs.tobytes()
    )
    decoded = dequantize("q8_1", (32,), payload)
    np.testing.assert_allclose(decoded, qs.astype(np.float32) * 0.25)


def test_q8_k_scales_the_superblock() -> None:
    scale = np.float32(0.5)
    qs = np.full(256, 4, dtype=np.int8)
    sums = np.zeros(16, dtype="<i2")
    payload = scale.tobytes() + qs.tobytes() + sums.tobytes()
    decoded = dequantize("q8_k", (256,), payload)
    np.testing.assert_allclose(decoded, np.full(256, 2.0, dtype=np.float32))


def test_unknown_packed_type_is_refused() -> None:
    with pytest.raises(GgufDequantError, match="not decoded"):
        dequantize("gguf.type_143", (4,), bytes(16))


def test_short_payload_is_refused() -> None:
    with pytest.raises(GgufDequantError, match="expected"):
        dequantize("float32", (4,), bytes(4))


@pytest.mark.parametrize(
    "element_type,code",
    [
        ("float16", "f2"),
        ("float64", "f8"),
        ("int8", "i1"),
        ("int16", "i2"),
        ("int32", "i4"),
        ("int64", "i8"),
        ("uint8", "u1"),
    ],
)
@pytest.mark.parametrize("endian", ["little", "big"])
def test_native_weight_types(element_type: str, code: str, endian: str) -> None:
    values = np.array([0, 1, 7, 12], dtype=("<" if endian == "little" else ">") + code)
    actual = dequantize(element_type, (2, 2), values.tobytes(), endian=endian)
    np.testing.assert_array_equal(actual, values.astype(np.float32).reshape(2, 2))


def _scale_bytes(scales: np.ndarray, minima: np.ndarray) -> bytes:
    packed = np.empty(12, dtype=np.uint8)
    packed[:4] = scales[:4] | ((scales[4:] >> 4) << 6)
    packed[4:8] = minima[:4] | ((minima[4:] >> 4) << 6)
    packed[8:] = (scales[4:] & 15) | ((minima[4:] & 15) << 4)
    return packed.tobytes()


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("endian", ["little", "big"])
def test_k_quant_scales_minima_and_high_bits(bits: int, endian: str) -> None:
    # Encode an independently specified set of integer weights, then recover
    # its affine dequantization. Nonzero high scale/min bits exercise all
    # twelve packed scale bytes rather than only the easy first four groups.
    scales = np.array([3, 12, 45, 63, 17, 34, 51, 62], dtype=np.uint8)
    minima = np.array([7, 44, 11, 32, 61, 29, 48, 35], dtype=np.uint8)
    values = ((np.arange(256) * 7 + 3) % (1 << bits)).astype(np.uint8)
    groups = values.reshape(4, 2, 32)
    low = (groups[:, 0] & 15) | ((groups[:, 1] & 15) << 4)
    high = np.zeros(32, dtype=np.uint8)
    for group in range(4):
        high |= (groups[group, 0] >> 4) << (2 * group)
        high |= (groups[group, 1] >> 4) << (2 * group + 1)
    prefix = np.array(
        [0.125, 0.25], dtype=("<" if endian == "little" else ">") + "f2"
    ).tobytes()
    payload = (
        prefix
        + _scale_bytes(scales, minima)
        + (high.tobytes() if bits == 5 else b"")
        + low.tobytes()
    )
    expected = 0.125 * np.repeat(
        scales.astype(np.float32), 32
    ) * values - 0.25 * np.repeat(minima.astype(np.float32), 32)
    actual = dequantize(f"q{bits}_k", (256,), payload, endian=endian)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("endian", ["little", "big"])
def test_q6_k_signed_scales_and_all_six_quant_bits(endian: str) -> None:
    values = (((np.arange(256) * 13) % 64) - 32).astype(np.int8)
    unsigned = (values.astype(np.int16) + 32).astype(np.uint8).reshape(2, 4, 32)
    low = np.empty((2, 2, 32), dtype=np.uint8)
    low[:, 0] = (unsigned[:, 0] & 15) | ((unsigned[:, 2] & 15) << 4)
    low[:, 1] = (unsigned[:, 1] & 15) | ((unsigned[:, 3] & 15) << 4)
    high = np.zeros((2, 32), dtype=np.uint8)
    for quarter in range(4):
        high |= (unsigned[:, quarter] >> 4) << (quarter * 2)
    scales = np.arange(-8, 8, dtype=np.int8)
    payload = (
        low.tobytes()
        + high.tobytes()
        + scales.tobytes()
        + np.array([0.125], dtype=("<" if endian == "little" else ">") + "f2").tobytes()
    )
    expected = (
        values.astype(np.float32) * np.repeat(scales.astype(np.float32), 16) * 0.125
    )
    np.testing.assert_array_equal(
        dequantize("q6_k", (16, 16), payload, endian=endian), expected.reshape(16, 16)
    )


@pytest.mark.parametrize(
    "element_type,dims,payload,message",
    [
        ("q4_k", (31,), b"", "multiple"),
        ("float32", (-1,), b"", "negative"),
    ],
)
def test_invalid_quantized_shapes_are_refused(
    element_type: str, dims: tuple[int, ...], payload: bytes, message: str
) -> None:
    with pytest.raises(GgufDequantError, match=message):
        dequantize(element_type, dims, payload)


def test_invalid_endian_is_refused() -> None:
    with pytest.raises(GgufDequantError, match="endian"):
        dequantize("float32", (1,), bytes(4), endian="unknown")
