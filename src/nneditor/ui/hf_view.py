"""Headless Hugging Face snapshot presentation: files, config, tensors.

The shell renders these rows; this module never imports Flet and never reads
tensor payloads. Large tokenizer vocabularies stay file descriptions until
the user asks for a bounded preview.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nneditor.adapters.huggingface.architecture import (
    is_embedding_name,
    is_output_name,
    layer_index,
)
from nneditor.adapters.huggingface.snapshot import tensor_source_name
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document, JsonValue, Storage
from nneditor.ui.viewmodel import compact_bytes, node_tensor_ids

__all__ = [
    "HfConfigRow",
    "HfContainerSummary",
    "HfLayoutSegment",
    "HfMemberRow",
    "HfTensorRow",
    "hf_config_rows",
    "hf_container_summary",
    "hf_layout_segments",
    "hf_member_rows",
    "hf_tensor_catalog",
    "hf_tokenizer_rows",
    "matches_query",
    "tensor_owner_node_id",
]

_MAX_INLINE_ITEMS = 6
_MAX_INLINE_CHARS = 96
_ARCH_KEYS = frozenset(
    {
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "hidden_act",
        "rms_norm_eps",
        "layer_norm_eps",
        "rope_theta",
        "torch_dtype",
        "tie_word_embeddings",
        "n_embd",
        "n_layer",
        "n_head",
        "n_inner",
        "n_positions",
        "d_model",
        "num_layers",
        "head_dim",
        "sliding_window",
        "max_position_embeddings",
        "num_experts",
        "num_experts_per_tok",
        "transformers_version",
    }
)
_TOKEN_KEYS = frozenset(
    {
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "tokenizer_class",
        "model_max_length",
    }
)


@dataclass(frozen=True, slots=True)
class HfLayoutSegment:
    """One kind of file in the snapshot, stacked by total size."""

    key: str
    name: str
    length: int
    fraction: float


@dataclass(frozen=True, slots=True)
class HfMemberRow:
    """One file at the snapshot root."""

    name: str
    kind: str
    size: int
    detail: str


@dataclass(frozen=True, slots=True)
class HfConfigRow:
    """One flattened config.json key."""

    key: str
    group: str
    value_text: str
    detail: str
    expandable: bool
    value: JsonValue


@dataclass(frozen=True, slots=True)
class HfTensorRow:
    """One tensor in the snapshot catalog."""

    tensor_id: str
    name: str
    group: str
    block: int | None
    element_type: str
    shape: str
    shard: str
    offset: int | None
    n_bytes: int | None
    owner_node_id: str | None


@dataclass(frozen=True, slots=True)
class HfContainerSummary:
    """At-a-glance facts for the Model section."""

    architecture: str
    name: str
    block_count: int
    vocab_size: int | None
    embedding_length: int | None
    context_length: int | None
    n_tensors: int
    n_shards: int
    n_members: int
    byte_size: int
    payload_bytes: int
    tied_embeddings: bool
    tokenizer_type: str


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


def hf_container_summary(document: Document) -> HfContainerSummary | None:
    """Facts from ``x-huggingface.model`` plus tokenizer type."""
    if document.artifact_kind is not ArtifactKind.HUGGINGFACE_MODEL:
        return None
    model = _extension_map(document, "x-huggingface.model")
    tokenizer = _extension_map(document, "x-huggingface.tokenizer")
    payload = 0
    for tensor in document.tensors.values():
        if tensor.storage is Storage.EXTERNAL and tensor.external is not None:
            payload += tensor.external.length or 0
    return HfContainerSummary(
        architecture=_str_field(model, "architecture") or "huggingface",
        name=_str_field(model, "name"),
        block_count=_int_field(model, "block_count") or 0,
        vocab_size=_int_field(model, "vocab_size"),
        embedding_length=_int_field(model, "embedding_length"),
        context_length=_int_field(model, "context_length"),
        n_tensors=_int_field(model, "n_tensors") or len(document.tensors),
        n_shards=_int_field(model, "n_shards") or 0,
        n_members=_int_field(model, "n_members") or 0,
        byte_size=_int_field(model, "byte_size") or document.source.byte_size,
        payload_bytes=payload,
        tied_embeddings=model.get("tied_embeddings") is True,
        tokenizer_type=_str_field(tokenizer, "model_type"),
    )


def hf_layout_segments(document: Document) -> tuple[HfLayoutSegment, ...]:
    """Stacked snapshot regions by member kind."""
    summary = hf_container_summary(document)
    if summary is None:
        return ()
    totals: dict[str, int] = {}
    labels = {
        "config": "Config",
        "tokenizer": "Tokenizer",
        "index": "Index",
        "weights": "Weights",
        "pickle": "Pickle",
        "other": "Other",
    }
    for item in _extension_list(document, "x-huggingface.members"):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        size = item.get("size")
        if not isinstance(kind, str):
            continue
        if isinstance(size, bool) or not isinstance(size, int):
            continue
        totals[kind] = totals.get(kind, 0) + size
    total = summary.byte_size or sum(totals.values())
    segments: list[HfLayoutSegment] = []
    for key in ("config", "tokenizer", "index", "weights", "pickle", "other"):
        length = totals.get(key, 0)
        if length <= 0:
            continue
        fraction = length / total if total else 0.0
        segments.append(HfLayoutSegment(key, labels[key], length, fraction))
    return tuple(segments)


def hf_member_rows(document: Document) -> tuple[HfMemberRow, ...]:
    """Every file at the snapshot root, largest first within each kind."""
    rows: list[HfMemberRow] = []
    for item in _extension_list(document, "x-huggingface.members"):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        kind = item.get("kind")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(kind, str):
            continue
        if isinstance(size, bool) or not isinstance(size, int):
            continue
        rows.append(HfMemberRow(name, kind, size, f"{kind} · {compact_bytes(size)}"))
    order = {"config": 0, "index": 1, "weights": 2, "tokenizer": 3, "pickle": 4}
    rows.sort(key=lambda row: (order.get(row.kind, 9), -row.size, row.name))
    return tuple(rows)


def _format_scalar(value: JsonValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
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


def _flatten(value: JsonValue, prefix: str) -> list[tuple[str, JsonValue]]:
    if isinstance(value, dict):
        rows: list[tuple[str, JsonValue]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_flatten(item, path))
        return rows
    return [(prefix, value)]


def _config_group(key: str) -> str:
    root = key.split(".", 1)[0]
    if root == "quantization_config":
        return "quantization"
    if root in _ARCH_KEYS or key in _ARCH_KEYS:
        return "architecture"
    if root in _TOKEN_KEYS or key in _TOKEN_KEYS:
        return "tokenizer"
    return "other"


def hf_config_rows(document: Document) -> tuple[HfConfigRow, ...]:
    """Flattened ``config.json`` keys, grouped for the explorer."""
    config = _extension_map(document, "x-huggingface.config")
    rows: list[HfConfigRow] = []
    for key, value in _flatten(config, ""):
        expandable = False
        if isinstance(value, list):
            text = _format_list(value)
            detail = f"{len(value):,} items"
        elif isinstance(value, str) and len(value) > _MAX_INLINE_CHARS:
            text = _format_scalar(value)
            expandable = True
            detail = f"{len(value):,} chars"
        else:
            text = _format_scalar(value)
            detail = ""
        rows.append(
            HfConfigRow(
                key=key,
                group=_config_group(key),
                value_text=text,
                detail=detail,
                expandable=expandable or isinstance(value, list),
                value=value,
            )
        )
    return tuple(rows)


def hf_tokenizer_rows(document: Document) -> tuple[HfConfigRow, ...]:
    """Tokenizer files and a short vocabulary preview."""
    tokenizer = _extension_map(document, "x-huggingface.tokenizer")
    rows: list[HfConfigRow] = []
    files = tokenizer.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            size = item.get("size")
            if not isinstance(name, str):
                continue
            size_text = (
                compact_bytes(size)
                if isinstance(size, int) and not isinstance(size, bool)
                else ""
            )
            rows.append(
                HfConfigRow(
                    key=name,
                    group="files",
                    value_text=size_text,
                    detail="tokenizer file",
                    expandable=False,
                    value=name,
                )
            )
    model_type = tokenizer.get("model_type")
    if isinstance(model_type, str) and model_type:
        rows.append(
            HfConfigRow(
                key="model_type",
                group="model",
                value_text=model_type,
                detail="",
                expandable=False,
                value=model_type,
            )
        )
    vocab_size = tokenizer.get("vocab_size")
    if isinstance(vocab_size, int) and not isinstance(vocab_size, bool):
        rows.append(
            HfConfigRow(
                key="vocab_size",
                group="model",
                value_text=f"{vocab_size:,}",
                detail="",
                expandable=False,
                value=vocab_size,
            )
        )
    preview = tokenizer.get("preview")
    if isinstance(preview, list) and preview:
        rows.append(
            HfConfigRow(
                key="vocab",
                group="model",
                value_text=_format_list(preview),
                detail="preview",
                expandable=True,
                value=preview,
            )
        )
    special = tokenizer.get("special_tokens")
    if isinstance(special, dict):
        for key, value in special.items():
            rows.append(
                HfConfigRow(
                    key=key,
                    group="special",
                    value_text=_format_scalar(value) or str(value),
                    detail="",
                    expandable=isinstance(value, str)
                    and len(value) > _MAX_INLINE_CHARS,
                    value=value,
                )
            )
    return tuple(rows)


def _tensor_group(name: str) -> tuple[str, int | None]:
    index = layer_index(name)
    if index is not None:
        return f"block-{index:03d}", index
    if is_embedding_name(name):
        return "embeddings", None
    if is_output_name(name):
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


def hf_tensor_catalog(document: Document) -> tuple[HfTensorRow, ...]:
    """Every tensor in snapshot order, grouped for the container explorer."""
    owners = _owner_map(document)
    rows: list[HfTensorRow] = []
    for tensor_id, tensor in document.tensors.items():
        name = tensor_source_name(tensor_id)
        group, block = _tensor_group(name)
        external = tensor.external
        rows.append(
            HfTensorRow(
                tensor_id=tensor_id,
                name=name,
                group=group,
                block=block,
                element_type=tensor.element_type,
                shape=_shape_text(tensor.dims),
                shard=external.location if external is not None else "",
                offset=external.offset if external is not None else None,
                n_bytes=external.length if external is not None else None,
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
