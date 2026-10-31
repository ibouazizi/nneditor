"""Tiny GGUF files covering v1, v2, v3 little-endian, and v3 big-endian."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nneditor.adapters.gguf.constants import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC_BYTES,
    KV_TYPE_ARRAY,
    KV_TYPE_BOOL,
    KV_TYPE_FLOAT32,
    KV_TYPE_STRING,
    KV_TYPE_UINT32,
    layout_for,
)

__all__ = ["GgufFixtureTensor", "write_gguf"]


@dataclass(frozen=True, slots=True)
class GgufFixtureTensor:
    """One tensor in a generated GGUF, dims in numpy/PyTorch order."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    payload: bytes


def write_gguf(
    path: Path,
    *,
    version: int,
    endian: str = "little",
    kv: tuple[tuple[str, int, Any], ...] = (),
    tensors: tuple[GgufFixtureTensor, ...] = (),
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
) -> Path:
    """Write a minimal well-formed GGUF file and return ``path``."""
    layout = layout_for(version, endian)
    payload_plan = _payload_layout(tensors, alignment)
    table = bytearray()
    table += GGUF_MAGIC_BYTES
    table += _pack_version(version, endian)
    table += _pack_count(layout, len(tensors))
    table += _pack_count(layout, len(kv))
    for key, type_id, value in kv:
        table += _pack_kv(layout, key, type_id, value)
    for tensor, relative in zip(tensors, payload_plan.offsets, strict=True):
        table += _pack_tensor(layout, tensor, relative)
    remainder = len(table) % alignment
    if remainder:
        table += bytes(alignment - remainder)
    data = bytearray(payload_plan.span)
    for tensor, relative in zip(tensors, payload_plan.offsets, strict=True):
        data[relative : relative + len(tensor.payload)] = tensor.payload
    path.write_bytes(bytes(table) + bytes(data))
    return path


@dataclass(frozen=True, slots=True)
class _PayloadLayout:
    offsets: tuple[int, ...]
    span: int


def _payload_layout(
    tensors: tuple[GgufFixtureTensor, ...], alignment: int
) -> _PayloadLayout:
    offsets: list[int] = []
    cursor = 0
    for tensor in tensors:
        remainder = cursor % alignment
        if remainder:
            cursor += alignment - remainder
        offsets.append(cursor)
        cursor += len(tensor.payload)
    return _PayloadLayout(tuple(offsets), cursor)


def _pack_version(version: int, endian: str) -> bytes:
    code = "<I" if endian == "little" else ">I"
    return struct.pack(code, version)


def _pack_count(layout: Any, value: int) -> bytes:
    return struct.pack(layout.struct_endian + layout.count_code, value)


def _pack_u32(layout: Any, value: int) -> bytes:
    return struct.pack(layout.struct_endian + "I", value)


def _pack_u64(layout: Any, value: int) -> bytes:
    return struct.pack(layout.struct_endian + "Q", value)


def _pack_dim(layout: Any, value: int) -> bytes:
    return struct.pack(layout.struct_endian + layout.dim_code, value)


def _pack_string(layout: Any, text: str) -> bytes:
    raw = text.encode("utf-8")
    return _pack_count(layout, len(raw)) + raw


def _pack_kv(layout: Any, key: str, type_id: int, value: Any) -> bytes:
    packed = _pack_string(layout, key) + _pack_u32(layout, type_id)
    return packed + _pack_value(layout, type_id, value)


def _pack_value(layout: Any, type_id: int, value: Any) -> bytes:
    if type_id == KV_TYPE_STRING:
        return _pack_string(layout, str(value))
    if type_id == KV_TYPE_BOOL:
        return bytes((1 if value else 0,))
    if type_id == KV_TYPE_UINT32:
        return _pack_u32(layout, int(value))
    if type_id == KV_TYPE_FLOAT32:
        return struct.pack(layout.struct_endian + "f", float(value))
    if type_id == KV_TYPE_ARRAY:
        elem_type, items = value
        packed = _pack_u32(layout, elem_type) + _pack_count(layout, len(items))
        for item in items:
            packed += _pack_value(layout, elem_type, item)
        return packed
    raise ValueError(f"fixture writer does not pack GGUF value type {type_id}")


def _pack_tensor(layout: Any, tensor: GgufFixtureTensor, relative: int) -> bytes:
    packed = _pack_string(layout, tensor.name)
    packed += _pack_u32(layout, len(tensor.dims))
    for dim in reversed(tensor.dims):
        packed += _pack_dim(layout, dim)
    packed += _pack_u32(layout, tensor.ggml_type)
    packed += _pack_u64(layout, relative)
    return packed
