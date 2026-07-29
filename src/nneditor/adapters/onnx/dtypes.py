"""ONNX ``TensorProto.DataType`` metadata.

Only the information the indexer needs is modelled: the name to show and the
element width used to check a declared external length against the declared
shape. Widths are in bits because ONNX defines four-bit types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["ELEMENT_TYPES", "ElementType", "element_type_for", "packed_byte_length"]


@dataclass(frozen=True, slots=True)
class ElementType:
    """One ONNX element type."""

    code: int
    name: str
    bits: int
    """Bits per element; zero when the width is not fixed, as for strings."""

    @property
    def is_fixed_width(self) -> bool:
        """Whether a byte length can be derived from an element count."""
        return self.bits > 0


_TYPES: tuple[ElementType, ...] = (
    ElementType(0, "UNDEFINED", 0),
    ElementType(1, "FLOAT", 32),
    ElementType(2, "UINT8", 8),
    ElementType(3, "INT8", 8),
    ElementType(4, "UINT16", 16),
    ElementType(5, "INT16", 16),
    ElementType(6, "INT32", 32),
    ElementType(7, "INT64", 64),
    ElementType(8, "STRING", 0),
    ElementType(9, "BOOL", 8),
    ElementType(10, "FLOAT16", 16),
    ElementType(11, "DOUBLE", 64),
    ElementType(12, "UINT32", 32),
    ElementType(13, "UINT64", 64),
    ElementType(14, "COMPLEX64", 64),
    ElementType(15, "COMPLEX128", 128),
    ElementType(16, "BFLOAT16", 16),
    ElementType(17, "FLOAT8E4M3FN", 8),
    ElementType(18, "FLOAT8E4M3FNUZ", 8),
    ElementType(19, "FLOAT8E5M2", 8),
    ElementType(20, "FLOAT8E5M2FNUZ", 8),
    ElementType(21, "UINT4", 4),
    ElementType(22, "INT4", 4),
    ElementType(23, "FLOAT4E2M1", 4),
    ElementType(24, "FLOAT8E8M0", 8),
)

ELEMENT_TYPES: Mapping[int, ElementType] = MappingProxyType(
    {item.code: item for item in _TYPES}
)


def element_type_for(code: int) -> ElementType:
    """Return the element type for ``code``, or a placeholder for unknown codes.

    Unknown codes are expected: new ONNX releases add types. The indexer reports
    them as a diagnostic instead of refusing to open the model.
    """
    known = ELEMENT_TYPES.get(code)
    if known is not None:
        return known
    return ElementType(code, f"UNKNOWN({code})", 0)


def packed_byte_length(element_type: ElementType, element_count: int) -> int | None:
    """Bytes needed for ``element_count`` values, or ``None`` if not derivable.

    Sub-byte types are packed two per byte with the final element padded, which
    matches how ONNX stores four-bit tensors.
    """
    if not element_type.is_fixed_width or element_count < 0:
        return None
    return (element_count * element_type.bits + 7) // 8
