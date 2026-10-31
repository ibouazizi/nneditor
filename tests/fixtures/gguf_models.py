"""Tiny GGUF files covering v1, v2, v3 little-endian, and v3 big-endian."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

__all__ = ["GgufFixtureTensor", "write_gguf", "write_traceable_llama_gguf"]

_F32 = 0
_N_EMBD = 4
_N_HEAD = 2
_N_FF = 8
_VOCAB = 8


def _f32_payload(shape: tuple[int, ...], seed: int, scale: float = 0.02) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(shape).astype("<f4") * np.float32(scale)
    return array.tobytes()


def _ones_payload(shape: tuple[int, ...]) -> bytes:
    return np.ones(shape, dtype="<f4").tobytes()


def write_traceable_llama_gguf(path: Path) -> Path:
    """A one-block Llama GGUF with every weight a numpy trace needs."""
    tensors = (
        GgufFixtureTensor(
            "token_embd.weight",
            (_VOCAB, _N_EMBD),
            _F32,
            _f32_payload((_VOCAB, _N_EMBD), 1),
        ),
        GgufFixtureTensor(
            "blk.0.attn_norm.weight", (_N_EMBD,), _F32, _ones_payload((_N_EMBD,))
        ),
        GgufFixtureTensor(
            "blk.0.attn_q.weight",
            (_N_EMBD, _N_EMBD),
            _F32,
            _f32_payload((_N_EMBD, _N_EMBD), 2),
        ),
        GgufFixtureTensor(
            "blk.0.attn_k.weight",
            (_N_EMBD, _N_EMBD),
            _F32,
            _f32_payload((_N_EMBD, _N_EMBD), 3),
        ),
        GgufFixtureTensor(
            "blk.0.attn_v.weight",
            (_N_EMBD, _N_EMBD),
            _F32,
            _f32_payload((_N_EMBD, _N_EMBD), 4),
        ),
        GgufFixtureTensor(
            "blk.0.attn_output.weight",
            (_N_EMBD, _N_EMBD),
            _F32,
            _f32_payload((_N_EMBD, _N_EMBD), 5),
        ),
        GgufFixtureTensor(
            "blk.0.ffn_norm.weight", (_N_EMBD,), _F32, _ones_payload((_N_EMBD,))
        ),
        GgufFixtureTensor(
            "blk.0.ffn_gate.weight",
            (_N_FF, _N_EMBD),
            _F32,
            _f32_payload((_N_FF, _N_EMBD), 6),
        ),
        GgufFixtureTensor(
            "blk.0.ffn_up.weight",
            (_N_FF, _N_EMBD),
            _F32,
            _f32_payload((_N_FF, _N_EMBD), 7),
        ),
        GgufFixtureTensor(
            "blk.0.ffn_down.weight",
            (_N_EMBD, _N_FF),
            _F32,
            _f32_payload((_N_EMBD, _N_FF), 8),
        ),
        GgufFixtureTensor(
            "output_norm.weight", (_N_EMBD,), _F32, _ones_payload((_N_EMBD,))
        ),
        GgufFixtureTensor(
            "output.weight", (_VOCAB, _N_EMBD), _F32, _f32_payload((_VOCAB, _N_EMBD), 9)
        ),
    )
    return write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", KV_TYPE_STRING, "llama"),
            ("general.name", KV_TYPE_STRING, "tiny-trace"),
            ("llama.block_count", KV_TYPE_UINT32, 1),
            ("llama.embedding_length", KV_TYPE_UINT32, _N_EMBD),
            ("llama.feed_forward_length", KV_TYPE_UINT32, _N_FF),
            ("llama.attention.head_count", KV_TYPE_UINT32, _N_HEAD),
            ("llama.attention.head_count_kv", KV_TYPE_UINT32, _N_HEAD),
            ("llama.attention.layer_norm_rms_eps", KV_TYPE_FLOAT32, 1e-5),
            ("llama.rope.freq_base", KV_TYPE_FLOAT32, 10000.0),
            ("llama.context_length", KV_TYPE_UINT32, 32),
            ("llama.vocab_size", KV_TYPE_UINT32, _VOCAB),
            ("tokenizer.ggml.bos_token_id", KV_TYPE_UINT32, 1),
            ("tokenizer.ggml.eos_token_id", KV_TYPE_UINT32, 2),
        ),
        tensors=tensors,
    )


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
    if type_id == 5:  # INT32, used by Prism's explicit sign vectors
        return struct.pack(layout.struct_endian + "i", int(value))
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
