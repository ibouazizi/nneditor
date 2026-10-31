"""Decode public ggml packed types into float32 arrays.

GGUF stores some tensors as native IEEE values and others as block-quantized
ggml layouts. The LLM worker needs float32 weights; this module is the one
decoder for both. Prism names are assigned separately from public ggml IDs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from nneditor.adapters.gguf.constants import GGML_IR_DTYPES, GGML_QUANT_SIZES, QK_K
from nneditor.adapters.gguf.prism import decode_ternary

__all__ = ["GgufDequantError", "dequantize", "supports_dequantization"]

Float32Array = NDArray[np.float32]
_Decoder = Callable[[bytes, int, str], Float32Array]


class GgufDequantError(ValueError):
    """A GGUF tensor payload could not be decoded into float32 weights."""


def dequantize(
    element_type: str,
    dims: tuple[int, ...],
    payload: bytes,
    *,
    endian: str = "little",
) -> Float32Array:
    """Return ``payload`` as a float32 array of ``dims``.

    ``endian`` applies to GGUF v3 big-endian files. Hugging Face safetensors
    payloads are little-endian.
    """
    if endian not in {"little", "big"}:
        raise GgufDequantError(f"unsupported tensor endian {endian!r}")
    n_elems = 1
    for dim in dims:
        if dim < 0:
            raise GgufDequantError(f"negative tensor extent in {dims}")
        n_elems *= dim
    decoder = _DECODERS.get(element_type)
    if decoder is None:
        raise GgufDequantError(
            f"packed GGUF type {element_type!r} is not decoded in this build"
        )
    values = decoder(payload, n_elems, endian)
    if values.size != n_elems:
        raise GgufDequantError(
            f"{element_type} payload decoded {values.size} values, expected {n_elems}"
        )
    return np.reshape(values, dims)


def _need(payload: bytes, nbytes: int, element_type: str) -> None:
    if len(payload) < nbytes:
        raise GgufDequantError(
            f"{element_type} payload is {len(payload)} bytes, expected {nbytes}"
        )


def _native(
    payload: bytes, n_elems: int, endian: str, code: str, width: int
) -> Float32Array:
    _need(payload, n_elems * width, code)
    prefix = "<" if endian == "little" else ">"
    values = np.frombuffer(payload, dtype=np.dtype(prefix + code), count=n_elems)
    return np.asarray(values, dtype=np.float32)


def _decode_f32(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "f4", 4)


def _decode_f16(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "f2", 2)


def _decode_f64(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "f8", 8)


def _decode_i8(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "i1", 1)


def _decode_i16(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "i2", 2)


def _decode_i32(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "i4", 4)


def _decode_i64(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "i8", 8)


def _decode_u8(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    return _native(payload, n_elems, endian, "u1", 1)


def _decode_bf16(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    _need(payload, n_elems * 2, "bfloat16")
    prefix = "<" if endian == "little" else ">"
    bits = np.frombuffer(payload, dtype=np.dtype(prefix + "u2"), count=n_elems)
    out = np.empty(n_elems, dtype=np.float32)
    out.view(np.uint32)[:] = bits.astype(np.uint32) << 16
    return out


def _decode_pq2(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    try:
        return decode_ternary(payload, n_elems, endian, dense=False)
    except ValueError as error:
        raise GgufDequantError(str(error)) from error


def _decode_ptq1(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    try:
        return decode_ternary(payload, n_elems, endian, dense=True)
    except ValueError as error:
        raise GgufDequantError(str(error)) from error


def _fp16_at(payload: bytes, offset: int, endian: str) -> float:
    prefix = "<" if endian == "little" else ">"
    value = np.frombuffer(
        payload, dtype=np.dtype(prefix + "f2"), count=1, offset=offset
    )
    return float(value[0])


def _block_count(n_elems: int, block: int, type_id: int, name: str) -> tuple[int, int]:
    if n_elems % block != 0:
        raise GgufDequantError(
            f"{name} tensor has {n_elems} elements, not a multiple of {block}"
        )
    sizes = GGML_QUANT_SIZES[type_id]
    return n_elems // block, sizes[1]


def _decode_q8_0(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    n_blocks, type_size = _block_count(n_elems, 32, 8, "q8_0")
    _need(payload, n_blocks * type_size, "q8_0")
    out = np.empty(n_elems, dtype=np.float32)
    for index in range(n_blocks):
        base = index * type_size
        scale = _fp16_at(payload, base, endian)
        qs = np.frombuffer(payload, dtype=np.int8, count=32, offset=base + 2)
        start = index * 32
        out[start : start + 32] = qs.astype(np.float32) * scale
    return out


def _decode_q4_0(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    n_blocks, type_size = _block_count(n_elems, 32, 2, "q4_0")
    _need(payload, n_blocks * type_size, "q4_0")
    out = np.empty(n_elems, dtype=np.float32)
    for index in range(n_blocks):
        base = index * type_size
        scale = _fp16_at(payload, base, endian)
        qs = np.frombuffer(payload, dtype=np.uint8, count=16, offset=base + 2)
        start = index * 32
        out[start : start + 16] = (qs & 0x0F).astype(np.float32) - 8.0
        out[start + 16 : start + 32] = (qs >> 4).astype(np.float32) - 8.0
        out[start : start + 32] *= scale
    return out


def _scale_min_k4(scales: bytes, index: int) -> tuple[int, int]:
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    scale = (scales[index + 4] & 0xF) | ((scales[index - 4] >> 6) << 4)
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)
    return scale, minimum


def _decode_q4_k(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    n_blocks, type_size = _block_count(n_elems, QK_K, 12, "q4_k")
    _need(payload, n_blocks * type_size, "q4_k")
    out = np.empty(n_elems, dtype=np.float32)
    for index in range(n_blocks):
        base = index * type_size
        scale = _fp16_at(payload, base, endian)
        minimum = _fp16_at(payload, base + 2, endian)
        scales = payload[base + 4 : base + 16]
        qs = payload[base + 16 : base + 16 + QK_K // 2]
        dest = out[index * QK_K : (index + 1) * QK_K]
        _fill_q4_k_block(dest, qs, scales, scale, minimum)
    return out


def _fill_q4_k_block(
    dest: Float32Array,
    qs: bytes,
    scales: bytes,
    scale: float,
    minimum: float,
) -> None:
    cursor = 0
    pair = 0
    for _group in range(QK_K // 64):
        sc_a, min_a = _scale_min_k4(scales, pair)
        sc_b, min_b = _scale_min_k4(scales, pair + 1)
        d1, m1 = scale * sc_a, minimum * min_a
        d2, m2 = scale * sc_b, minimum * min_b
        chunk = qs[cursor : cursor + 32]
        for offset, byte in enumerate(chunk):
            dest[_group * 64 + offset] = d1 * (byte & 0xF) - m1
            dest[_group * 64 + 32 + offset] = d2 * (byte >> 4) - m2
        cursor += 32
        pair += 2


def _decode_q5_k(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    n_blocks, type_size = _block_count(n_elems, QK_K, 13, "q5_k")
    _need(payload, n_blocks * type_size, "q5_k")
    out = np.empty(n_elems, dtype=np.float32)
    for index in range(n_blocks):
        base = index * type_size
        scale = _fp16_at(payload, base, endian)
        minimum = _fp16_at(payload, base + 2, endian)
        scales = payload[base + 4 : base + 16]
        qh = payload[base + 16 : base + 16 + QK_K // 8]
        qs = payload[base + 16 + QK_K // 8 : base + type_size]
        dest = out[index * QK_K : (index + 1) * QK_K]
        _fill_q5_k_block(dest, qs, qh, scales, scale, minimum)
    return out


def _fill_q5_k_block(
    dest: Float32Array,
    qs: bytes,
    qh: bytes,
    scales: bytes,
    scale: float,
    minimum: float,
) -> None:
    cursor = 0
    pair = 0
    high_a, high_b = 1, 2
    for group in range(QK_K // 64):
        sc_a, min_a = _scale_min_k4(scales, pair)
        sc_b, min_b = _scale_min_k4(scales, pair + 1)
        d1, m1 = scale * sc_a, minimum * min_a
        d2, m2 = scale * sc_b, minimum * min_b
        chunk = qs[cursor : cursor + 32]
        for offset, byte in enumerate(chunk):
            extra_a = 16.0 if qh[offset] & high_a else 0.0
            extra_b = 16.0 if qh[offset] & high_b else 0.0
            dest[group * 64 + offset] = d1 * ((byte & 0xF) + extra_a) - m1
            dest[group * 64 + 32 + offset] = d2 * ((byte >> 4) + extra_b) - m2
        cursor += 32
        pair += 2
        high_a <<= 2
        high_b <<= 2


def _decode_q6_k(payload: bytes, n_elems: int, endian: str) -> Float32Array:
    n_blocks, type_size = _block_count(n_elems, QK_K, 14, "q6_k")
    _need(payload, n_blocks * type_size, "q6_k")
    out = np.empty(n_elems, dtype=np.float32)
    for index in range(n_blocks):
        base = index * type_size
        ql = payload[base : base + QK_K // 2]
        qh = payload[base + QK_K // 2 : base + QK_K // 2 + QK_K // 4]
        scales = np.frombuffer(
            payload,
            dtype=np.int8,
            count=QK_K // 16,
            offset=base + QK_K // 2 + QK_K // 4,
        )
        d_offset = base + type_size - 2
        scale = _fp16_at(payload, d_offset, endian)
        dest = out[index * QK_K : (index + 1) * QK_K]
        _fill_q6_k_block(dest, ql, qh, scales, scale)
    return out


def _fill_q6_k_block(
    dest: Float32Array,
    ql: bytes,
    qh: bytes,
    scales: NDArray[np.int8],
    scale: float,
) -> None:
    for half in range(2):
        ql_off = half * 64
        qh_off = half * 32
        sc_off = half * 8
        y_off = half * 128
        for index in range(32):
            slot = index // 16
            high = qh[qh_off + index]
            q1 = (ql[ql_off + index] & 0xF) | ((high & 3) << 4)
            q2 = (ql[ql_off + index + 32] & 0xF) | (((high >> 2) & 3) << 4)
            q3 = (ql[ql_off + index] >> 4) | (((high >> 4) & 3) << 4)
            q4 = (ql[ql_off + index + 32] >> 4) | (((high >> 6) & 3) << 4)
            dest[y_off + index] = scale * int(scales[sc_off + slot]) * (q1 - 32)
            dest[y_off + 32 + index] = (
                scale * int(scales[sc_off + slot + 2]) * (q2 - 32)
            )
            dest[y_off + 64 + index] = (
                scale * int(scales[sc_off + slot + 4]) * (q3 - 32)
            )
            dest[y_off + 96 + index] = (
                scale * int(scales[sc_off + slot + 6]) * (q4 - 32)
            )


_DECODERS: dict[str, _Decoder] = {
    "prism.pq2_0": _decode_pq2,
    "prism.ptq1_0": _decode_ptq1,
    "float32": _decode_f32,
    "float16": _decode_f16,
    "float64": _decode_f64,
    "bfloat16": _decode_bf16,
    "int8": _decode_i8,
    "int16": _decode_i16,
    "int32": _decode_i32,
    "int64": _decode_i64,
    "uint8": _decode_u8,
    "q8_0": _decode_q8_0,
    "q4_0": _decode_q4_0,
    "q4_k": _decode_q4_k,
    "q5_k": _decode_q5_k,
    "q6_k": _decode_q6_k,
}


def supports_dequantization(element_type: str) -> bool:
    """Whether the worker can decode this weight type without reading it."""
    return element_type in _DECODERS


def _assert_known_sizes() -> None:
    # Import-time sanity: IR names we decode must match the ggml size table.
    for type_id, name in GGML_IR_DTYPES.items():
        if name in _DECODERS and type_id in GGML_QUANT_SIZES:
            block, size = GGML_QUANT_SIZES[type_id]
            if block <= 0 or size <= 0 or math.isnan(size):
                raise RuntimeError(f"invalid ggml size for {name}")


_assert_known_sizes()
