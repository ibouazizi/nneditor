"""Headless GGUF container presentation: layout, metadata, and tensor catalog.

The shell renders these rows; this module never imports Flet and never reads
tensor payloads. Large tokenizer arrays stay descriptions of byte ranges
until the user asks for a bounded preview.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from nneditor.adapters.gguf.blocks import tensor_source_name
from nneditor.adapters.gguf.constants import header_nbytes, kv_type_name
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document, JsonValue
from nneditor.ui.viewmodel import compact_bytes, node_tensor_ids

__all__ = [
    "GgufContainerSummary",
    "GgufKvRow",
    "GgufLayoutSegment",
    "GgufTensorRow",
    "gguf_container_summary",
    "gguf_kv_rows",
    "gguf_layout_segments",
    "gguf_tensor_catalog",
    "gguf_tokenizer_rows",
    "matches_query",
    "tensor_owner_node_id",
]

_BLOCK_NAME: Final = re.compile(r"^blk\.(\d+)\.")
_EMBED_PREFIXES: Final = ("token_embd", "position_embd", "token_types")
_SUMMARY_KEYS: Final = (
    ("context_length", "Context"),
    ("embedding_length", "Embedding"),
    ("feed_forward_length", "FFN width"),
    ("attention.head_count", "Heads"),
    ("attention.head_count_kv", "KV heads"),
    ("expert_count", "Experts"),
    ("ssm.state_size", "SSM state"),
)
_MAX_INLINE_ITEMS: Final = 6
_MAX_INLINE_CHARS: Final = 96


@dataclass(frozen=True, slots=True)
class GgufLayoutSegment:
    """One contiguous region of the GGUF file."""

    key: str
    name: str
    offset: int
    length: int
    fraction: float


@dataclass(frozen=True, slots=True)
class GgufKvRow:
    """One metadata key, ready to render."""

    key: str
    group: str
    type_name: str
    type_id: int
    value_text: str
    detail: str
    offset: int
    length: int
    elided: bool
    previewable: bool
    expandable: bool
    value: JsonValue


@dataclass(frozen=True, slots=True)
class GgufTensorRow:
    """One tensor in the container catalog."""

    tensor_id: str
    name: str
    group: str
    block: int | None
    element_type: str
    shape: str
    offset: int | None
    n_bytes: int | None
    owner_node_id: str | None


@dataclass(frozen=True, slots=True)
class GgufContainerSummary:
    """At-a-glance facts for the Model section."""

    architecture: str
    name: str
    version: int
    endian: str
    alignment: int
    block_count: int
    vocab_size: int | None
    embedding_length: int | None
    context_length: int | None
    n_tensors: int
    n_kv: int
    data_start: int
    file_size: int
    payload_bytes: int
    tied_embeddings: bool
    extras: tuple[tuple[str, str], ...]


def _extension_map(document: Document, namespace: str) -> dict[str, JsonValue]:
    value = dict(document.extensions).get(namespace)
    return dict(value) if isinstance(value, dict) else {}


def _extension_list(document: Document, namespace: str) -> list[JsonValue]:
    value = dict(document.extensions).get(namespace)
    return list(value) if isinstance(value, list) else []


def _int_field(data: Mapping[str, JsonValue], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _str_field(data: Mapping[str, JsonValue], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _lookup_arch_int(
    metadata: Mapping[str, JsonValue], architecture: str, suffix: str
) -> int | None:
    keys = (f"{architecture}.{suffix}", suffix) if architecture else (suffix,)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None


def gguf_container_summary(document: Document) -> GgufContainerSummary | None:
    """Facts from ``x-gguf.model`` plus a few well-known architecture keys."""
    if document.artifact_kind is not ArtifactKind.GGUF_MODEL:
        return None
    model = _extension_map(document, "x-gguf.model")
    metadata = _extension_map(document, "x-gguf.metadata")
    architecture = _str_field(model, "architecture")
    file_size = document.source.byte_size
    data_start = _int_field(model, "data_start") or 0
    extras = tuple(
        (label, f"{value:,}")
        for suffix, label in _SUMMARY_KEYS
        if (value := _lookup_arch_int(metadata, architecture, suffix)) is not None
        and suffix not in {"embedding_length"}
    )
    return GgufContainerSummary(
        architecture=architecture or "gguf",
        name=_str_field(model, "name"),
        version=_int_field(model, "version") or 0,
        endian=_str_field(model, "endian") or "little",
        alignment=_int_field(model, "alignment") or 32,
        block_count=_int_field(model, "block_count") or 0,
        vocab_size=_int_field(model, "vocab_size"),
        embedding_length=_int_field(model, "embedding_length"),
        context_length=_lookup_arch_int(metadata, architecture, "context_length"),
        n_tensors=_int_field(model, "n_tensors") or len(document.tensors),
        n_kv=_int_field(model, "n_kv") or 0,
        data_start=data_start,
        file_size=file_size,
        payload_bytes=max(0, file_size - data_start),
        tied_embeddings=model.get("tied_embeddings") is True,
        extras=extras,
    )


def _segment(
    key: str, name: str, offset: int, length: int, file_size: int
) -> GgufLayoutSegment | None:
    if length <= 0:
        return None
    fraction = length / file_size if file_size else 0.0
    return GgufLayoutSegment(key, name, offset, length, fraction)


def gguf_layout_segments(document: Document) -> tuple[GgufLayoutSegment, ...]:
    """Header, metadata, tensor table, padding, and weight payloads."""
    summary = gguf_container_summary(document)
    if summary is None:
        return ()
    model = _extension_map(document, "x-gguf.model")
    header_end = header_nbytes(summary.version) if summary.version else 8
    kv_end = _int_field(model, "kv_end") or header_end
    table_end = _int_field(model, "tensor_table_end") or kv_end
    data_start = summary.data_start
    file_size = summary.file_size
    parts = (
        _segment("header", "Header", 0, header_end, file_size),
        _segment("metadata", "Metadata", header_end, kv_end - header_end, file_size),
        _segment("tensors", "Tensor table", kv_end, table_end - kv_end, file_size),
        _segment("padding", "Padding", table_end, data_start - table_end, file_size),
        _segment("weights", "Weights", data_start, file_size - data_start, file_size),
    )
    return tuple(part for part in parts if part is not None)


def _kv_group(key: str, architecture: str) -> str:
    prefix, separator, _rest = key.partition(".")
    if not separator:
        return "other"
    if architecture and prefix == architecture:
        return "architecture"
    if prefix in {"general", "tokenizer"}:
        return prefix
    return prefix


def _format_scalar(value: JsonValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        rendered = f"{value:.6g}"
        return rendered
    if isinstance(value, str):
        collapsed = " ".join(value.split())
        if len(collapsed) > _MAX_INLINE_CHARS:
            return collapsed[: _MAX_INLINE_CHARS - 1] + "…"
        return collapsed
    return ""


def _format_list(items: Sequence[JsonValue]) -> str:
    shown = [_format_scalar(item) or str(item) for item in items[:_MAX_INLINE_ITEMS]]
    suffix = ", …" if len(items) > _MAX_INLINE_ITEMS else ""
    return "[" + ", ".join(shown) + suffix + "]"


def _elided_detail(value: Mapping[str, JsonValue]) -> tuple[str, str, bool]:
    kind = value.get("type")
    count = value.get("count")
    length = value.get("length")
    elem = value.get("element_type")
    type_label = kv_type_name(elem) if isinstance(elem, int) else "value"
    if kind == "array" and isinstance(count, int) and not isinstance(count, bool):
        noun = type_label if count == 1 else f"{type_label}s"
        text = f"{count:,} {noun}"
    elif kind == "string":
        text = "string"
    else:
        text = "byte range"
    size = compact_bytes(length) if isinstance(length, int) else ""
    detail = "elided" if not size else f"elided · {size}"
    previewable = kind == "array" and type_label != "array"
    return text, detail, previewable


def _kv_row(record: Mapping[str, JsonValue], architecture: str) -> GgufKvRow | None:
    key = record.get("key")
    type_id = record.get("type_id")
    offset = record.get("offset")
    length = record.get("length")
    if not isinstance(key, str):
        return None
    if isinstance(type_id, bool) or not isinstance(type_id, int):
        return None
    if isinstance(offset, bool) or not isinstance(offset, int):
        return None
    if isinstance(length, bool) or not isinstance(length, int):
        return None
    elided = record.get("elided") is True
    raw = record.get("value")
    expandable = False
    previewable = False
    detail = f"@ {offset:,}"
    if elided and isinstance(raw, dict):
        value_text, elided_detail, previewable = _elided_detail(raw)
        detail = f"{elided_detail} · @ {offset:,}"
    elif isinstance(raw, list):
        value_text = _format_list(raw)
        detail = f"{len(raw):,} items · @ {offset:,}"
    elif isinstance(raw, str) and len(raw) > _MAX_INLINE_CHARS:
        value_text = _format_scalar(raw)
        expandable = True
        detail = f"{len(raw):,} chars · @ {offset:,}"
    else:
        value_text = _format_scalar(raw) or compact_bytes(length)
    return GgufKvRow(
        key=key,
        group=_kv_group(key, architecture),
        type_name=kv_type_name(type_id),
        type_id=type_id,
        value_text=value_text,
        detail=detail,
        offset=offset,
        length=length,
        elided=elided,
        previewable=previewable,
        expandable=expandable,
        value=raw,
    )


def gguf_kv_rows(document: Document) -> tuple[GgufKvRow, ...]:
    """Every GGUF metadata key, grouped by the first path segment."""
    summary = gguf_container_summary(document)
    architecture = summary.architecture if summary is not None else ""
    if architecture == "gguf":
        architecture = ""
    rows: list[GgufKvRow] = []
    for item in _extension_list(document, "x-gguf.kv"):
        if not isinstance(item, dict):
            continue
        row = _kv_row(item, architecture)
        if row is not None:
            rows.append(row)
    return tuple(rows)


def gguf_tokenizer_rows(document: Document) -> tuple[GgufKvRow, ...]:
    """The ``tokenizer.*`` subset of the metadata catalog."""
    return tuple(row for row in gguf_kv_rows(document) if row.group == "tokenizer")


def _tensor_group(name: str) -> tuple[str, int | None]:
    match = _BLOCK_NAME.match(name)
    if match is not None:
        block = int(match.group(1))
        return f"block-{block:03d}", block
    if any(
        name == prefix or name.startswith(prefix + ".") for prefix in _EMBED_PREFIXES
    ):
        return "embeddings", None
    if name == "output.weight" or name.startswith("output_norm"):
        return "output", None
    return "other", None


def _shape_text(dims: Sequence[int]) -> str:
    if not dims:
        return "scalar"
    return " x ".join(str(dim) for dim in dims)


def _owner_map(document: Document) -> dict[str, str]:
    owners: dict[str, str] = {}
    for graph in document.graphs.values():
        for node in graph.nodes:
            try:
                tensor_ids = node_tensor_ids(document, graph.id, node.id)
            except KeyError:
                continue
            for tensor_id in tensor_ids:
                owners.setdefault(tensor_id, node.id)
    return owners


def tensor_owner_node_id(document: Document, tensor_id: str) -> str | None:
    """The synthesized operator that binds ``tensor_id``, if any."""
    return _owner_map(document).get(tensor_id)


def gguf_tensor_catalog(document: Document) -> tuple[GgufTensorRow, ...]:
    """Every tensor in file order, grouped for the container explorer."""
    owners = _owner_map(document)
    rows: list[GgufTensorRow] = []
    for tensor_id, tensor in document.tensors.items():
        name = tensor_source_name(tensor_id)
        group, block = _tensor_group(name)
        payload = tensor.payload
        rows.append(
            GgufTensorRow(
                tensor_id=tensor_id,
                name=name,
                group=group,
                block=block,
                element_type=tensor.element_type,
                shape=_shape_text(tensor.dims),
                offset=payload.offset if payload is not None else None,
                n_bytes=payload.length if payload is not None else None,
                owner_node_id=owners.get(tensor_id),
            )
        )
    return tuple(rows)


def matches_query(query: str, *fields: str) -> bool:
    """Case-insensitive substring match across already-rendered fields."""
    needle = query.strip().casefold()
    if not needle:
        return True
    return any(needle in field.casefold() for field in fields)
