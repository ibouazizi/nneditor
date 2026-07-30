"""The single authority for tensor element-type metadata.

Adapters and analysis passes used to keep private dtype tables that drifted
apart: an artifact with float8 weights imported and validated cleanly, yet
every statistic, preview, and diff silently returned nothing. This registry
is the one place that records, for every canonical lowercase IR element-type
name ("float32", "bfloat16", "float8e4m3fn", ...), how wide an element is,
how to decode its packed bytes into Python numbers, and — when that is not
possible — the reason to disclose instead of silence.

Decoders are pure functions from packed little-endian bytes to a sequence of
Python floats or ints, built on the C-speed ``array``/``struct`` modules.
The float8 formats decode through 256-entry lookup tables computed once at
import from their bit-level definitions, so the module needs nothing beyond
the standard library.
"""

from __future__ import annotations

import math
import struct
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

__all__ = ["DTYPES", "DTypeInfo", "dtype_info"]


@dataclass(frozen=True, slots=True)
class DTypeInfo:
    """One canonical element type: its width and how (or why not) to decode."""

    name: str
    bits: int
    """Bits per element; zero when the width is not fixed, as for strings."""
    byte_width: int | None
    """Whole bytes per element; ``None`` for sub-byte and non-fixed widths."""
    is_float: bool
    """Whether values are floating point, so decoded output can hold NaN or
    infinities that statistics must filter."""
    decoder: Callable[[bytes], Sequence[float]] | None = None
    """Packed little-endian bytes to Python numbers; ``None`` when the type
    cannot be decoded."""
    unavailable_reason: str | None = None
    """Why the type cannot be decoded; set exactly when ``decoder`` is not."""

    def __post_init__(self) -> None:
        # Exactly one of decoder/unavailable_reason: a dtype either decodes
        # or explains itself — silence is not a permitted state.
        if (self.decoder is None) == (self.unavailable_reason is None):
            raise ValueError(
                f"dtype {self.name!r} must carry exactly one of a decoder "
                "or an unavailable_reason"
            )
        if self.decoder is not None and self.byte_width is None:
            raise ValueError(f"dtype {self.name!r} has a decoder but no byte width")
        if self.byte_width is not None and self.byte_width * 8 != self.bits:
            raise ValueError(
                f"dtype {self.name!r} declares {self.bits} bits but a "
                f"{self.byte_width}-byte width"
            )


def _array_decoder(typecode: str) -> Callable[[bytes], Sequence[float]]:
    def decode(raw: bytes) -> Sequence[float]:
        values = array(typecode)
        values.frombytes(raw)
        return values

    return decode


def _half_decoder(raw: bytes) -> Sequence[float]:
    return [value[0] for value in struct.iter_unpack("<e", raw)]


def _bfloat16_decoder(raw: bytes) -> Sequence[float]:
    """Decode packed bfloat16 bit patterns.

    A bfloat16 value is exactly the high 16 bits of the float32 with the
    same value, so each little-endian 2-byte pattern widens to a float32 bit
    pattern with two zero low bytes — NaN, infinity, and denormal patterns
    all survive the widening unchanged.
    """
    widened = bytearray(len(raw) * 2)
    widened[2::4] = raw[0::2]
    widened[3::4] = raw[1::2]
    values = array("f")
    values.frombytes(bytes(widened))
    return values


def _float8_table(
    exponent_bits: int, mantissa_bits: int, bias: int, variant: str
) -> tuple[float, ...]:
    """The 256 values of one sign+exponent+mantissa float8 format.

    ``variant`` selects the special-value rules: ``"ieee"`` reserves the
    all-ones exponent for infinities (mantissa zero) and NaNs (e5m2);
    ``"fn"`` has no infinities and one NaN pattern per sign at exponent and
    mantissa all-ones (e4m3fn); ``"fnuz"`` has no infinities, no negative
    zero, and a single NaN at 0x80.
    """
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    values: list[float] = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> mantissa_bits) & exponent_mask
        mantissa = code & mantissa_mask
        if variant == "fnuz" and code == 0x80:
            values.append(math.nan)
        elif variant == "ieee" and exponent == exponent_mask:
            values.append(sign * math.inf if mantissa == 0 else math.nan)
        elif (
            variant == "fn" and exponent == exponent_mask and mantissa == mantissa_mask
        ):
            values.append(math.nan)
        elif exponent == 0:
            # Subnormal: no implicit leading one, minimum exponent.
            values.append(sign * math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias))
        else:
            values.append(
                sign
                * math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
            )
    return tuple(values)


# OCP MX scale format: unsigned power of two, bias 127, NaN at 0xFF.
_E8M0_TABLE: Final[tuple[float, ...]] = tuple(
    math.nan if code == 0xFF else math.ldexp(1.0, code - 127) for code in range(256)
)


def _table_decoder(table: tuple[float, ...]) -> Callable[[bytes], Sequence[float]]:
    def decode(raw: bytes) -> Sequence[float]:
        return [table[code] for code in raw]

    return decode


_SUB_BYTE_REASON: Final = (
    "4-bit elements are packed two per byte and are not yet decodable"
)
_COMPLEX_REASON: Final = (
    "complex elements do not reduce to single real values and are not yet decodable"
)

_REGISTRY: Final[dict[str, DTypeInfo]] = {
    info.name: info
    for info in (
        DTypeInfo("bool", 8, 1, False, _array_decoder("B")),
        DTypeInfo("int8", 8, 1, False, _array_decoder("b")),
        DTypeInfo("uint8", 8, 1, False, _array_decoder("B")),
        DTypeInfo("int16", 16, 2, False, _array_decoder("h")),
        DTypeInfo("uint16", 16, 2, False, _array_decoder("H")),
        DTypeInfo("int32", 32, 4, False, _array_decoder("i")),
        DTypeInfo("uint32", 32, 4, False, _array_decoder("I")),
        DTypeInfo("int64", 64, 8, False, _array_decoder("q")),
        DTypeInfo("uint64", 64, 8, False, _array_decoder("Q")),
        DTypeInfo("float16", 16, 2, True, _half_decoder),
        DTypeInfo("bfloat16", 16, 2, True, _bfloat16_decoder),
        DTypeInfo("float32", 32, 4, True, _array_decoder("f")),
        DTypeInfo("float64", 64, 8, True, _array_decoder("d")),
        DTypeInfo(
            "float8e4m3fn", 8, 1, True, _table_decoder(_float8_table(4, 3, 7, "fn"))
        ),
        DTypeInfo(
            "float8e4m3fnuz",
            8,
            1,
            True,
            _table_decoder(_float8_table(4, 3, 8, "fnuz")),
        ),
        DTypeInfo(
            "float8e5m2", 8, 1, True, _table_decoder(_float8_table(5, 2, 15, "ieee"))
        ),
        DTypeInfo(
            "float8e5m2fnuz",
            8,
            1,
            True,
            _table_decoder(_float8_table(5, 2, 16, "fnuz")),
        ),
        DTypeInfo("float8e8m0", 8, 1, True, _table_decoder(_E8M0_TABLE)),
        DTypeInfo("int4", 4, None, False, None, _SUB_BYTE_REASON),
        DTypeInfo("uint4", 4, None, False, None, _SUB_BYTE_REASON),
        DTypeInfo("float4e2m1", 4, None, True, None, _SUB_BYTE_REASON),
        DTypeInfo("complex32", 32, 4, False, None, _COMPLEX_REASON),
        DTypeInfo("complex64", 64, 8, False, None, _COMPLEX_REASON),
        DTypeInfo("complex128", 128, 16, False, None, _COMPLEX_REASON),
        DTypeInfo(
            "string",
            0,
            None,
            False,
            None,
            "string elements have no fixed width and no numeric value",
        ),
        DTypeInfo(
            "undefined",
            0,
            None,
            False,
            None,
            "the artifact does not declare an element type",
        ),
    )
}

DTYPES: Final[Mapping[str, DTypeInfo]] = MappingProxyType(_REGISTRY)
"""Every canonical element-type name any adapter or analysis pass knows."""


def dtype_info(name: str) -> DTypeInfo | None:
    """The registry record for a canonical element-type name, or ``None``."""
    return DTYPES.get(name)
