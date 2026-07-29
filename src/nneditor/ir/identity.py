"""Stable identifier rules for the IR (task P0.3).

Identifiers must survive three things: reopening the same artifact, collapsing
or expanding hierarchy, and editing an unrelated part of the graph. Positional
indices satisfy none of those, so they are only ever a last resort and are
labelled as such through :class:`NodeIdStability`.

Every identifier is a printable string with a short kind prefix. Segments that
come from a model are percent-encoded so that a name containing ``/``, ``#``,
``:``, or ``%`` cannot forge a different identifier.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from typing import Final

__all__ = [
    "GROUP_ID_DIGEST_LENGTH",
    "ROOT_GRAPH_ID",
    "NodeIdStability",
    "attribute_subgraph_id",
    "auto_group_id",
    "decode_segment",
    "encode_segment",
    "initializer_id",
    "missing_value_id",
    "node_id",
    "user_group_id",
    "value_id",
]

_RESERVED: Final = {"%": "%25", "/": "%2F", "#": "%23", ":": "%3A"}

_HEX_DIGITS: Final = frozenset("0123456789abcdefABCDEF")

ROOT_GRAPH_ID: Final = "g:main"
"""The identifier of a document's entry graph."""

GROUP_ID_DIGEST_LENGTH: Final = 16
"""Hex characters kept from the group member digest."""


class NodeIdStability(StrEnum):
    """How trustworthy a derived node identifier is across reopens and edits."""

    NAMED = "named"
    """Derived from an explicit, unique node name in the source artifact."""

    OUTPUT_DERIVED = "output-derived"
    """Derived from the node's first output value name, unique within a graph."""

    POSITIONAL = "positional"
    """Derived from the node's index; changes when neighbouring nodes change."""


def encode_segment(text: str) -> str:
    """Percent-encode the characters that structure an identifier.

    Non-printable characters are escaped by code point with a width chosen by
    range: ``%XX`` below U+0100, ``%uXXXX`` for the rest of the Basic
    Multilingual Plane, and ``%UXXXXXXXX`` for astral code points. Fixed
    widths per marker keep the encoding injective — an astral code point can
    never be spelled as a BMP escape followed by literal hex digits.
    """
    out: list[str] = []
    for char in text:
        replacement = _RESERVED.get(char)
        if replacement is not None:
            out.append(replacement)
        elif char.isprintable():
            out.append(char)
        else:
            point = ord(char)
            if point < 0x100:
                out.append(f"%{point:02X}")
            elif point < 0x10000:
                out.append(f"%u{point:04X}")
            else:
                out.append(f"%U{point:08X}")
    return "".join(out)


def _decode_escape(text: str, index: int, prefix: int, width: int) -> str:
    """Decode one fixed-width escape starting at ``text[index]`` (a ``%``)."""
    digits = text[index + prefix : index + prefix + width]
    if len(digits) != width or any(digit not in _HEX_DIGITS for digit in digits):
        raise ValueError(
            f"malformed identifier segment {text!r}: the escape at position "
            f"{index} needs {width} hexadecimal digits"
        )
    point = int(digits, 16)
    if point > 0x10FFFF:
        raise ValueError(
            f"malformed identifier segment {text!r}: the escape at position "
            f"{index} names an impossible code point U+{point:X}"
        )
    return chr(point)


def decode_segment(text: str) -> str:
    """Reverse :func:`encode_segment`.

    Strict: a truncated, non-hexadecimal, or out-of-range escape raises
    :class:`ValueError` with an explanation instead of mis-decoding.
    """
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "%":
            out.append(char)
            index += 1
            continue
        marker = text[index + 1 : index + 2]
        if marker == "u":
            out.append(_decode_escape(text, index, 2, 4))
            index += 6
        elif marker == "U":
            out.append(_decode_escape(text, index, 2, 8))
            index += 10
        else:
            out.append(_decode_escape(text, index, 1, 2))
            index += 3
    return "".join(out)


def attribute_subgraph_id(
    parent_graph_id: str,
    parent_node_id: str,
    attribute_name: str,
    index: int | None = None,
) -> str:
    """Identify a control-flow or function subgraph by where it is attached.

    Nesting is expressed by the attachment path rather than by a counter, so an
    ``If`` branch keeps its identifier when a sibling node is inserted.
    """
    suffix = "" if index is None else f"[{index}]"
    return (
        f"{parent_graph_id}/{encode_segment(parent_node_id)}"
        f"/{encode_segment(attribute_name)}{suffix}"
    )


def node_id(
    graph_id: str,
    *,
    name: str = "",
    first_output: str = "",
    ordinal: int,
) -> tuple[str, NodeIdStability]:
    """Derive a node identifier and report how stable it is.

    Preference order: the explicit node name, then the first output value name
    (unique within a graph in ONNX and StableHLO), then the positional index.
    """
    if name:
        return f"n:{graph_id}#name:{encode_segment(name)}", NodeIdStability.NAMED
    if first_output:
        return (
            f"n:{graph_id}#out:{encode_segment(first_output)}",
            NodeIdStability.OUTPUT_DERIVED,
        )
    return f"n:{graph_id}#idx:{ordinal}", NodeIdStability.POSITIONAL


def value_id(graph_id: str, name: str) -> str:
    """Identify a named value within its owning graph scope."""
    return f"v:{graph_id}#{encode_segment(name)}"


def missing_value_id(owner_node_id: str, port_index: int) -> str:
    """Identify the placeholder for an omitted optional input."""
    return f"v:{encode_segment(owner_node_id)}#empty:{port_index}"


def initializer_id(graph_id: str, name: str) -> str:
    """Identify an initializer tensor within its owning graph scope."""
    return f"t:{graph_id}#{encode_segment(name)}"


def auto_group_id(detector: str, version: int, member_ids: Iterable[str]) -> str:
    """Identify an automatically detected group by its membership.

    The digest is order independent so that a detector emitting members in a
    different order still produces the same group, which keeps user overrides
    attached across reopens.
    """
    digest = hashlib.blake2b(digest_size=GROUP_ID_DIGEST_LENGTH // 2)
    for member in sorted(set(member_ids)):
        digest.update(member.encode("utf-8"))
        digest.update(b"\x00")
    return f"grp:auto:{encode_segment(detector)}:v{version}:{digest.hexdigest()}"


def user_group_id(graph_id: str, label: str) -> str:
    """Identify a group a user created by hand."""
    return f"grp:user:{graph_id}#{encode_segment(label)}"
