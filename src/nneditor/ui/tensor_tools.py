"""Small, bounded helpers for the tensor visualizer and hex editor.

The UI intentionally works on a page of bytes supplied by the editing
controller.  Nothing in this module can request or materialize a whole tensor.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Final

from nneditor.analysis.statistics import decode_packed

__all__ = [
    "HEATMAP_COLUMNS",
    "HEATMAP_ELEMENTS",
    "HEX_COLUMNS",
    "HEX_PAGE_BYTES",
    "ascii_preview",
    "format_hex_bytes",
    "heat_color",
    "parse_hex_bytes",
    "parse_offset",
    "sample_values",
]

HEX_PAGE_BYTES: Final = 128
HEX_COLUMNS: Final = 16
HEATMAP_ELEMENTS: Final = 64
HEATMAP_COLUMNS: Final = 16

_HEX_TOKEN = re.compile(r"(?:0[xX])?([0-9a-fA-F]{2})\Z")


def format_hex_bytes(raw: bytes, *, columns: int = HEX_COLUMNS) -> str:
    """Format a bounded byte page as editable, whitespace-separated hex."""
    if columns <= 0:
        raise ValueError("columns must be positive")
    return "\n".join(
        " ".join(f"{value:02X}" for value in raw[start : start + columns])
        for start in range(0, len(raw), columns)
    )


def parse_hex_bytes(text: str, *, max_bytes: int = HEX_PAGE_BYTES) -> bytes:
    """Parse strict byte tokens while accepting spaces and line breaks."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    tokens = text.split()
    if not tokens:
        raise ValueError("enter at least one byte")
    if len(tokens) > max_bytes:
        raise ValueError(f"this editor accepts at most {max_bytes} bytes per change")
    values: list[int] = []
    for position, token in enumerate(tokens, start=1):
        match = _HEX_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(
                f"byte {position} must contain exactly two hexadecimal digits"
            )
        values.append(int(match.group(1), 16))
    return bytes(values)


def ascii_preview(raw: bytes, *, columns: int = HEX_COLUMNS) -> str:
    """Render the byte page as printable ASCII with dots for other bytes."""
    if columns <= 0:
        raise ValueError("columns must be positive")
    decoded = "".join(chr(value) if 32 <= value <= 126 else "." for value in raw)
    return "\n".join(
        decoded[start : start + columns] for start in range(0, len(decoded), columns)
    )


def parse_offset(text: str, *, total_bytes: int) -> int:
    """Parse a decimal or ``0x`` offset and validate it against the tensor."""
    if total_bytes < 0:
        raise ValueError("total byte length cannot be negative")
    stripped = text.strip()
    # Parse exactly the two forms the error message promises; ``int(_, 0)``
    # would also accept octal and binary prefixes the UI never mentions.
    try:
        offset = (
            int(stripped, 16)
            if stripped.lower().startswith("0x")
            else int(stripped, 10)
        )
    except ValueError as error:
        raise ValueError("offset must be a decimal integer or start with 0x") from error
    if offset < 0:
        raise ValueError("offset cannot be negative")
    if total_bytes == 0:
        if offset == 0:
            return 0
        raise ValueError("an empty tensor only has offset 0")
    if offset >= total_bytes:
        raise ValueError(f"offset must be below {total_bytes:,} bytes")
    return offset


def sample_values(
    element_type: str, raw: bytes, *, limit: int = HEATMAP_ELEMENTS
) -> tuple[float, ...] | None:
    """Decode at most ``limit`` already-read values for a heatmap."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    decoded: Sequence[float] | None = decode_packed(element_type, raw)
    if decoded is None:
        return None
    return tuple(float(value) for value in decoded[:limit])


def heat_color(value: float, maximum_absolute: float) -> str:
    """Map a sample to a modern diverging blue/neutral/purple palette."""
    if not math.isfinite(value):
        return "#98A2B3"
    if maximum_absolute <= 0 or value == 0:
        return "#F2F4F7"
    strength = min(abs(value) / maximum_absolute, 1.0)
    neutral = (242, 244, 247)
    endpoint = (46, 144, 250) if value < 0 else (127, 86, 217)
    # A little color at low magnitudes keeps near-zero structure perceptible.
    blend = 0.18 + strength * 0.82
    rgb = tuple(
        round(start + (end - start) * blend)
        for start, end in zip(neutral, endpoint, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in rgb)
