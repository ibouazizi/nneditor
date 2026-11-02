"""Public GGUF / ggml type tables used by the native indexer.

Copied from the ggml-org GGUF specification (magic, versions, KV types) and
the live ``ggml_type`` / type-traits table. Fork-only ids such as PTQ1_0 and
PQ2_0 are intentionally absent: they surface as ``gguf.type_<n>`` unless
Prism extension metadata identifies the verified private layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

__all__ = [
    "GGML_IR_DTYPES",
    "GGML_MAX_DIMS",
    "GGML_QUANT_SIZES",
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_MAGIC",
    "GGUF_MAGIC_BYTES",
    "GGUF_SPEC_URL",
    "GGUF_SUPPORTED_VERSIONS",
    "GGUF_VALUE_TYPE_WIDTH",
    "KV_TYPE_ARRAY",
    "KV_TYPE_NAMES",
    "KV_TYPE_STRING",
    "MAX_GGUF_KV_ENTRIES",
    "MAX_GGUF_KV_VALUE_BYTES",
    "MAX_GGUF_STRING",
    "MAX_GGUF_TENSORS",
    "MAX_RETAINED_ARRAY_BYTES",
    "MAX_RETAINED_STRING_ARRAY",
    "MAX_TENSOR_NAME",
    "MAX_VALUE_DEPTH",
    "QK_K",
    "REMOVED_GGML_TYPE_IDS",
    "GgufWireLayout",
    "element_type_name",
    "header_nbytes",
    "kv_type_name",
    "layout_for",
    "packed_nbytes",
]

GGUF_MAGIC: Final = 0x46554747
GGUF_MAGIC_BYTES: Final = b"GGUF"
GGUF_SUPPORTED_VERSIONS: Final = frozenset({1, 2, 3})
GGUF_DEFAULT_ALIGNMENT: Final = 32
MAX_VALUE_DEPTH: Final = 8
"""Nested ARRAY depth the KV parser will walk before eliding."""


@dataclass(frozen=True, slots=True)
class GgufWireLayout:
    """Per-version, per-endian sizes of GGUF countable fields.

    v1 stored tensor/KV counts, string lengths, array lengths, and tensor
    dimensions as uint32. v2 widened those to uint64 so larger models fit.
    v3 keeps the v2 layout and adds big-endian files, detected from the
    version word rather than a flag.
    """

    version: int
    endian: str
    count_size: int
    dim_size: int

    def __post_init__(self) -> None:
        if self.endian not in {"little", "big"}:
            raise ValueError(f"endian must be little or big, got {self.endian!r}")
        if self.count_size not in {4, 8} or self.dim_size not in {4, 8}:
            raise ValueError("GGUF count and dim sizes must be 4 or 8 bytes")

    @property
    def struct_endian(self) -> str:
        return "<" if self.endian == "little" else ">"

    @property
    def count_code(self) -> str:
        return "I" if self.count_size == 4 else "Q"

    @property
    def dim_code(self) -> str:
        return "I" if self.dim_size == 4 else "Q"


def layout_for(version: int, endian: str) -> GgufWireLayout:
    """Wire sizes for a GGUF version. v1 is 32-bit counts; v2 and v3 are 64-bit."""
    if version not in GGUF_SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported GGUF version {version}")
    width = 4 if version == 1 else 8
    return GgufWireLayout(version, endian, width, width)


def header_nbytes(version: int) -> int:
    """Bytes in the GGUF magic, version word, and the two count fields."""
    layout = layout_for(version, "little")
    return 8 + 2 * layout.count_size


GGML_MAX_DIMS: Final = 4
MAX_TENSOR_NAME: Final = 64
"""ggml tensor names, including the trailing NUL in the C layout."""

GGUF_SPEC_URL: Final = "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md"

MAX_GGUF_TENSORS: Final = 1_048_576
MAX_GGUF_KV_ENTRIES: Final = 1_048_576
MAX_GGUF_STRING: Final = 1 * 1024 * 1024
MAX_GGUF_KV_VALUE_BYTES: Final = 32 * 1024 * 1024
MAX_RETAINED_ARRAY_BYTES: Final = 16 * 1024
MAX_RETAINED_STRING_ARRAY: Final = 256

# GGUFValueType
KV_TYPE_UINT8: Final = 0
KV_TYPE_INT8: Final = 1
KV_TYPE_UINT16: Final = 2
KV_TYPE_INT16: Final = 3
KV_TYPE_UINT32: Final = 4
KV_TYPE_INT32: Final = 5
KV_TYPE_FLOAT32: Final = 6
KV_TYPE_BOOL: Final = 7
KV_TYPE_STRING: Final = 8
KV_TYPE_ARRAY: Final = 9
KV_TYPE_UINT64: Final = 10
KV_TYPE_INT64: Final = 11
KV_TYPE_FLOAT64: Final = 12

GGUF_VALUE_TYPE_WIDTH: Final[MappingProxyType[int, int]] = MappingProxyType(
    {
        KV_TYPE_UINT8: 1,
        KV_TYPE_INT8: 1,
        KV_TYPE_UINT16: 2,
        KV_TYPE_INT16: 2,
        KV_TYPE_UINT32: 4,
        KV_TYPE_INT32: 4,
        KV_TYPE_FLOAT32: 4,
        KV_TYPE_BOOL: 1,
        KV_TYPE_UINT64: 8,
        KV_TYPE_INT64: 8,
        KV_TYPE_FLOAT64: 8,
    }
)

KV_TYPE_NAMES: Final[MappingProxyType[int, str]] = MappingProxyType(
    {
        KV_TYPE_UINT8: "uint8",
        KV_TYPE_INT8: "int8",
        KV_TYPE_UINT16: "uint16",
        KV_TYPE_INT16: "int16",
        KV_TYPE_UINT32: "uint32",
        KV_TYPE_INT32: "int32",
        KV_TYPE_FLOAT32: "float32",
        KV_TYPE_BOOL: "bool",
        KV_TYPE_STRING: "string",
        KV_TYPE_ARRAY: "array",
        KV_TYPE_UINT64: "uint64",
        KV_TYPE_INT64: "int64",
        KV_TYPE_FLOAT64: "float64",
    }
)


def kv_type_name(type_id: int) -> str:
    """Human name for a GGUF value-type id, or ``type_<n>`` when unknown."""
    return KV_TYPE_NAMES.get(type_id, f"type_{type_id}")


# Historical ggml_type ids that GGUF no longer writes.
REMOVED_GGML_TYPE_IDS: Final = frozenset({4, 5, 31, 32, 33, 36, 37, 38})

QK_K: Final = 256

# (block_size, type_size) for public live ggml types. nbytes =
# n_elems * type_size // block_size.
GGML_QUANT_SIZES: Final[MappingProxyType[int, tuple[int, int]]] = MappingProxyType(
    {
        0: (1, 4),  # F32
        1: (1, 2),  # F16
        2: (32, 18),  # Q4_0
        3: (32, 20),  # Q4_1
        6: (32, 22),  # Q5_0
        7: (32, 24),  # Q5_1
        8: (32, 34),  # Q8_0
        9: (32, 36),  # Q8_1: two fp16 values plus 32 int8 quants
        10: (256, 2 + 2 + QK_K // 16 + QK_K // 4),  # Q2_K
        11: (256, 2 + QK_K // 4 + QK_K // 8 + 12),  # Q3_K
        12: (256, 2 + 2 + QK_K // 2 + 12),  # Q4_K
        13: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),  # Q5_K
        14: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),  # Q6_K
        15: (256, 4 + QK_K + QK_K // 8),  # Q8_K
        16: (256, 2 + QK_K // 4),  # IQ2_XXS
        17: (256, 2 + QK_K // 4 + QK_K // 32),  # IQ2_XS
        18: (256, 2 + QK_K // 4 + QK_K // 8),  # IQ3_XXS
        19: (256, 2 + QK_K // 8 + QK_K // 16),  # IQ1_S
        20: (32, 18),  # IQ4_NL
        21: (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),  # IQ3_S
        22: (256, 2 + QK_K // 4 + QK_K // 16),  # IQ2_S
        23: (256, 2 + 2 + QK_K // 2 + QK_K // 64),  # IQ4_XS
        24: (1, 1),  # I8
        25: (1, 2),  # I16
        26: (1, 4),  # I32
        27: (1, 8),  # I64
        28: (1, 8),  # F64
        29: (256, QK_K // 8 + QK_K // 16 + QK_K // 32),  # IQ1_M
        30: (1, 2),  # BF16
        34: (256, 2 + 4 * 13),  # TQ1_0
        35: (256, 2 + 64),  # TQ2_0
        39: (32, 17),  # MXFP4
        40: (64, 36),  # NVFP4
        41: (128, 18),  # Q1_0
        42: (64, 18),  # Q2_0
    }
)

# Canonical IR names for public ggml types. Packed quants keep a ggml-style
# spelling so the inspector can name them.
GGML_IR_DTYPES: Final[MappingProxyType[int, str]] = MappingProxyType(
    {
        0: "float32",
        1: "float16",
        2: "q4_0",
        3: "q4_1",
        6: "q5_0",
        7: "q5_1",
        8: "q8_0",
        9: "q8_1",
        10: "q2_k",
        11: "q3_k",
        12: "q4_k",
        13: "q5_k",
        14: "q6_k",
        15: "q8_k",
        16: "iq2_xxs",
        17: "iq2_xs",
        18: "iq3_xxs",
        19: "iq1_s",
        20: "iq4_nl",
        21: "iq3_s",
        22: "iq2_s",
        23: "iq4_xs",
        24: "int8",
        25: "int16",
        26: "int32",
        27: "int64",
        28: "float64",
        29: "iq1_m",
        30: "bfloat16",
        34: "tq1_0",
        35: "tq2_0",
        39: "mxfp4",
        40: "nvfp4",
        41: "q1_0",
        42: "q2_0",
    }
)

_PACKED_QUANT_REASON: Final = (
    "packed GGUF block quantization; values are not decoded in this build"
)


def element_type_name(type_id: int) -> str:
    """Canonical IR element-type name for a ggml type id."""
    return GGML_IR_DTYPES.get(type_id, f"gguf.type_{type_id}")


def packed_nbytes(type_id: int, n_elems: int) -> int | None:
    """Payload bytes for ``n_elems`` of a known ggml type, or ``None``."""
    sizes = GGML_QUANT_SIZES.get(type_id)
    if sizes is None or n_elems < 0:
        return None
    block_size, type_size = sizes
    return n_elems * type_size // block_size


def packed_quant_reason() -> str:
    """Shared unavailable-reason for every packed ggml quant in the dtype registry."""
    return _PACKED_QUANT_REASON
