"""Index a Hugging Face snapshot directory without reading weight payloads.

The container is a directory: ``config.json`` (or ``adapter_config.json``)
next to safetensors shards, an optional ``model.safetensors.index.json``
weight map, and tokenizer files. Identity is ``hf-index:`` over small JSON
members plus each weight file's name and size — payloads are never hashed.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nneditor.adapters.pytorch.safetensors import (
    SafetensorEntry,
    SafetensorsError,
    index_safetensors_file,
)
from nneditor.cancellation import CancellationToken
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.core import JsonValue, Storage
from nneditor.ir.identity import ROOT_GRAPH_ID, decode_segment, initializer_id
from nneditor.storage.paths import UnsafePathError, resolve_snapshot_member

__all__ = [
    "HfIndex",
    "HfMember",
    "HfTensorIndex",
    "HuggingFaceError",
    "index_huggingface",
    "is_huggingface_snapshot",
    "preview_tokenizer_vocab",
    "tensor_source_name",
]

MAX_JSON_BYTES: Final = 8 * 1024 * 1024
MAX_INDEX_BYTES: Final = 32 * 1024 * 1024
MAX_TOKENIZER_PARSE: Final = 8 * 1024 * 1024
MAX_IDENTITY_FILE: Final = 32 * 1024 * 1024

_CONFIG_NAMES: Final = ("config.json", "adapter_config.json")
_INDEX_PRIORITY: Final = (
    "model.safetensors.index.json",
    "adapter_model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_SINGLE_WEIGHTS: Final = (
    "model.safetensors",
    "adapter_model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "model.fp16.safetensors",
    "model.bf16.safetensors",
)
_PICKLE_SUFFIXES: Final = (".bin", ".pt", ".pth", ".ckpt")
_TOKENIZER_NAMES: Final = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "tokenizer.model",
        "chat_template.jinja",
        "chat_template.json",
    }
)
_SKIP_DIR_NAMES: Final = {".git", "__pycache__"}


class HuggingFaceError(ValueError):
    """Raised when a Hugging Face snapshot cannot be indexed."""


@dataclass(frozen=True, slots=True)
class HfMember:
    """One file at the snapshot root."""

    name: str
    size: int
    kind: str


@dataclass(frozen=True, slots=True)
class HfTensorIndex:
    """One weight located in a shard, or listed without bytes."""

    id: str
    name: str
    element_type: str
    dims: tuple[int, ...]
    shard: str
    offset: int
    length: int
    storage: Storage


@dataclass(frozen=True, slots=True)
class HfIndex:
    """Header-only view of a Hugging Face snapshot directory."""

    path: Path
    content_hash: str
    byte_size: int
    members: tuple[HfMember, ...]
    tensors: tuple[HfTensorIndex, ...]
    config: dict[str, JsonValue]
    tokenizer: dict[str, JsonValue]
    shards: tuple[str, ...]
    weight_map: dict[str, str]
    diagnostics: DiagnosticLog


def is_huggingface_snapshot(target: Path) -> bool:
    """Whether ``target`` looks like a Transformers/PEFT snapshot directory."""
    if not target.is_dir():
        return False
    try:
        names = {path.name for path in target.iterdir() if path.is_file()}
    except OSError:
        return False
    if not any(name in names for name in _CONFIG_NAMES):
        return False
    if any(name.endswith(".safetensors") for name in names):
        return True
    if any(name.endswith(".safetensors.index.json") for name in names):
        return True
    if any(name.endswith(".bin.index.json") for name in names):
        return True
    return "pytorch_model.bin" in names or "adapter_model.bin" in names


def tensor_source_name(tensor_id: str) -> str:
    """The snapshot tensor name encoded in an initializer id."""
    _, separator, encoded = tensor_id.partition("#")
    if not separator:
        return tensor_id
    return decode_segment(encoded)


def preview_tokenizer_vocab(
    path: Path | str, limit: int = 32
) -> tuple[tuple[str, ...], int]:
    """The first ``limit`` vocabulary tokens from ``tokenizer.json``."""
    source = Path(path)
    tokenizer = source / "tokenizer.json" if source.is_dir() else source
    if not tokenizer.is_file():
        raise HuggingFaceError(f"{source.name} holds no tokenizer.json")
    parsed = _read_json_object(tokenizer, MAX_TOKENIZER_PARSE)
    vocab = _vocab_from_tokenizer(parsed)
    items = tuple(vocab[: max(0, limit)])
    return items, len(vocab)


def index_huggingface(
    path: Path | str,
    token: CancellationToken | None = None,
) -> HfIndex:
    """Index a snapshot directory without reading weight payloads."""
    target = Path(path).resolve()
    if not target.is_dir():
        raise HuggingFaceError(f"{target} is not a snapshot directory")
    if token is not None:
        token.raise_if_cancelled()
    members = _list_members(target)
    if not members:
        raise HuggingFaceError(f"{target.name} holds no snapshot files")
    log = DiagnosticLog()
    config = _load_config(target, members, log)
    tensors, shards, weight_map = _index_weights(target, members, log, token)
    identity = _identity(target, members, tensors)
    tokenizer = _tokenizer_summary(target, members, log)
    return HfIndex(
        path=target,
        content_hash=identity,
        byte_size=sum(member.size for member in members),
        members=members,
        tensors=tensors,
        config=config,
        tokenizer=tokenizer,
        shards=shards,
        weight_map=weight_map,
        diagnostics=log,
    )


def _list_members(target: Path) -> tuple[HfMember, ...]:
    members: list[HfMember] = []
    try:
        children = sorted(target.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise HuggingFaceError(f"{target.name} could not be listed: {error}") from error
    for child in children:
        if child.name in _SKIP_DIR_NAMES or not child.is_file():
            continue
        try:
            size = child.stat().st_size
        except OSError:
            continue
        members.append(HfMember(child.name, size, _member_kind(child.name)))
    return tuple(members)


def _member_kind(name: str) -> str:
    if name in _CONFIG_NAMES:
        return "config"
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return "index"
    if name.endswith(".safetensors"):
        return "weights"
    if name in _TOKENIZER_NAMES:
        return "tokenizer"
    if name.endswith(_PICKLE_SUFFIXES):
        return "pickle"
    return "other"


def _identity(
    target: Path, members: tuple[HfMember, ...], tensors: tuple[HfTensorIndex, ...]
) -> str:
    digest = hashlib.sha256()
    for member in members:
        digest.update(member.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<Q", member.size))
        digest.update(b"\0")
        digest.update(member.kind.encode("ascii"))
        digest.update(b"\n")
        if member.kind in {"weights", "pickle"}:
            continue
        if member.size > MAX_IDENTITY_FILE:
            continue
        try:
            digest.update((target / member.name).read_bytes())
        except OSError:
            continue
    # The catalog covers the bounded safetensors headers, including names,
    # shapes, dtypes, locations and readability. Payloads stay unread.
    for tensor in tensors:
        digest.update(
            json.dumps(
                [
                    tensor.name,
                    tensor.element_type,
                    tensor.dims,
                    tensor.shard,
                    tensor.offset,
                    tensor.length,
                    tensor.storage.value,
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return f"hf-index:{digest.hexdigest()}"


def _as_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_as_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    return str(value)


def _read_json_object(path: Path, limit: int) -> dict[str, JsonValue]:
    size = path.stat().st_size
    if size > limit:
        raise HuggingFaceError(
            f"{path.name} is {size} bytes; the JSON limit is {limit}"
        )
    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError) as error:
        raise HuggingFaceError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise HuggingFaceError(f"{path.name} must be a JSON object")
    return {str(key): _as_json(value) for key, value in parsed.items()}


def _load_config(
    target: Path, members: tuple[HfMember, ...], log: DiagnosticLog
) -> dict[str, JsonValue]:
    names = {member.name for member in members}
    for name in _CONFIG_NAMES:
        if name not in names:
            continue
        try:
            return _read_json_object(target / name, MAX_JSON_BYTES)
        except HuggingFaceError as error:
            log.add(
                "huggingface.config-unreadable",
                Severity.WARNING,
                str(error),
                name,
            )
            return {}
    log.add(
        "huggingface.config-unreadable",
        Severity.WARNING,
        "the snapshot holds no config.json",
        "config.json",
    )
    return {}


def _index_name(members: tuple[HfMember, ...]) -> str | None:
    names = {member.name for member in members}
    for name in _INDEX_PRIORITY:
        if name in names:
            return name
    for member in members:
        if member.kind == "index":
            return member.name
    return None


def _single_weight(members: tuple[HfMember, ...]) -> str | None:
    names = {member.name for member in members}
    for name in _SINGLE_WEIGHTS:
        if name in names:
            return name
    safetensors = [member.name for member in members if member.kind == "weights"]
    if len(safetensors) == 1:
        return safetensors[0]
    return None


def _all_safetensors(members: tuple[HfMember, ...]) -> tuple[str, ...]:
    return tuple(member.name for member in members if member.kind == "weights")


def _index_weights(
    target: Path,
    members: tuple[HfMember, ...],
    log: DiagnosticLog,
    token: CancellationToken | None,
) -> tuple[tuple[HfTensorIndex, ...], tuple[str, ...], dict[str, str]]:
    index_name = _index_name(members)
    if index_name is not None:
        weight_map = _load_weight_map(target / index_name, log)
        if weight_map:
            tensors, shards = _tensors_from_weight_map(target, weight_map, log, token)
            return tensors, shards, weight_map
    single = _single_weight(members)
    shards = (single,) if single is not None else _all_safetensors(members)
    if shards:
        tensors = _tensors_from_shards(target, shards, log, token)
        weight_map = {tensor.name: tensor.shard for tensor in tensors}
        return tensors, shards, weight_map
    pickle_names = tuple(member.name for member in members if member.kind == "pickle")
    if pickle_names:
        log.add(
            "huggingface.pickle-weights",
            Severity.WARNING,
            "pickle weight shards are not unpickled in safe artifact mode",
            pickle_names[0],
        )
    return (), pickle_names, {}


def _load_weight_map(path: Path, log: DiagnosticLog) -> dict[str, str]:
    try:
        parsed = _read_json_object(path, MAX_INDEX_BYTES)
    except HuggingFaceError as error:
        log.add(
            "huggingface.index-unreadable",
            Severity.WARNING,
            str(error),
            path.name,
        )
        return {}
    raw = parsed.get("weight_map")
    if not isinstance(raw, dict):
        log.add(
            "huggingface.index-unreadable",
            Severity.WARNING,
            f"{path.name} has no weight_map object",
            path.name,
        )
        return {}
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value:
            mapping[str(key)] = value
    return mapping


def _tensors_from_weight_map(
    target: Path,
    weight_map: dict[str, str],
    log: DiagnosticLog,
    token: CancellationToken | None,
) -> tuple[tuple[HfTensorIndex, ...], tuple[str, ...]]:
    shards = tuple(dict.fromkeys(weight_map.values()))
    catalogs: dict[str, dict[str, SafetensorEntry]] = {}
    for shard in shards:
        if token is not None:
            token.raise_if_cancelled()
        catalogs[shard] = _catalog_for_shard(target, shard, log)
    seen: set[str] = set()
    tensors: list[HfTensorIndex] = []
    for name, shard in weight_map.items():
        if name in seen:
            log.add(
                "huggingface.duplicate-tensor-name",
                Severity.WARNING,
                f"tensor {name!r} appears more than once in the weight map",
                name,
            )
            continue
        seen.add(name)
        entry = catalogs.get(shard, {}).get(name)
        tensors.append(_mapped_tensor(name, shard, entry))
    return tuple(tensors), shards


def _catalog_for_shard(
    target: Path, shard: str, log: DiagnosticLog
) -> dict[str, SafetensorEntry]:
    try:
        resolved = resolve_snapshot_member(target, shard)
    except UnsafePathError as error:
        log.add(
            "huggingface.unsafe-shard-path",
            Severity.ERROR,
            f"shard {shard!r} is not a safe snapshot path: {error}",
            shard,
        )
        return {}
    if not resolved.is_file():
        log.add(
            "huggingface.missing-shard",
            Severity.ERROR,
            f"shard {shard!r} is not next to config.json",
            shard,
        )
        return {}
    if Path(shard).suffix.casefold() != ".safetensors":
        log.add(
            "huggingface.pickle-weights",
            Severity.WARNING,
            f"shard {shard!r} is not safetensors and is not unpickled",
            shard,
        )
        return {}
    try:
        entries, _metadata, _size = index_safetensors_file(resolved, log)
    except SafetensorsError as error:
        log.add(
            "huggingface.missing-shard",
            Severity.ERROR,
            f"shard {shard!r} could not be indexed: {error}",
            shard,
        )
        return {}
    return {entry.name: entry for entry in entries}


def _mapped_tensor(
    name: str, shard: str, entry: SafetensorEntry | None
) -> HfTensorIndex:
    tensor_id = initializer_id(ROOT_GRAPH_ID, name)
    if entry is None:
        return HfTensorIndex(tensor_id, name, "uint8", (), shard, 0, 0, Storage.ABSENT)
    storage = Storage.EXTERNAL if entry.readable else Storage.ABSENT
    return HfTensorIndex(
        tensor_id,
        name,
        entry.element_type,
        entry.dims,
        shard,
        entry.offset,
        entry.length,
        storage,
    )


def _tensors_from_shards(
    target: Path,
    shards: tuple[str, ...],
    log: DiagnosticLog,
    token: CancellationToken | None,
) -> tuple[HfTensorIndex, ...]:
    tensors: list[HfTensorIndex] = []
    seen: set[str] = set()
    for shard in shards:
        if token is not None:
            token.raise_if_cancelled()
        catalog = _catalog_for_shard(target, shard, log)
        for entry in catalog.values():
            if entry.name in seen:
                log.add(
                    "huggingface.duplicate-tensor-name",
                    Severity.WARNING,
                    f"tensor {entry.name!r} is repeated in {shard}",
                    entry.name,
                )
                continue
            seen.add(entry.name)
            tensors.append(_mapped_tensor(entry.name, shard, entry))
    return tuple(tensors)


def _vocab_from_tokenizer(parsed: Mapping[str, JsonValue]) -> tuple[str, ...]:
    model = parsed.get("model")
    if not isinstance(model, dict):
        return ()
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return tuple(str(key) for key in vocab)
    if isinstance(vocab, list):
        return tuple(str(item) for item in vocab)
    return ()


def _tokenizer_summary(
    target: Path, members: tuple[HfMember, ...], log: DiagnosticLog
) -> dict[str, JsonValue]:
    files: list[JsonValue] = [
        {"name": member.name, "size": member.size}
        for member in members
        if member.kind == "tokenizer"
    ]
    summary: dict[str, JsonValue] = {"files": files}
    tokenizer_json = target / "tokenizer.json"
    if tokenizer_json.is_file():
        size = tokenizer_json.stat().st_size
        if size > MAX_TOKENIZER_PARSE:
            log.add(
                "huggingface.tokenizer-elided",
                Severity.INFO,
                f"tokenizer.json is {size} bytes; vocabulary is not parsed at open",
                "tokenizer.json",
            )
            summary["elided"] = True
            summary["size"] = size
        else:
            try:
                parsed = _read_json_object(tokenizer_json, MAX_TOKENIZER_PARSE)
            except HuggingFaceError:
                summary["elided"] = True
            else:
                _fill_tokenizer_model(summary, parsed)
    config_path = target / "tokenizer_config.json"
    if config_path.is_file() and config_path.stat().st_size <= MAX_JSON_BYTES:
        try:
            parsed_config = _read_json_object(config_path, MAX_JSON_BYTES)
        except HuggingFaceError:
            parsed_config = {}
        special = {
            key: parsed_config[key]
            for key in (
                "bos_token",
                "eos_token",
                "pad_token",
                "unk_token",
                "cls_token",
                "sep_token",
                "mask_token",
                "model_max_length",
                "tokenizer_class",
            )
            if key in parsed_config
        }
        if special:
            summary["special_tokens"] = special
    return summary


def _fill_tokenizer_model(
    summary: dict[str, JsonValue], parsed: dict[str, JsonValue]
) -> None:
    model = parsed.get("model")
    vocab = _vocab_from_tokenizer(parsed)
    if isinstance(model, dict):
        model_type = model.get("type")
        if isinstance(model_type, str):
            summary["model_type"] = model_type
    if vocab:
        summary["vocab_size"] = len(vocab)
        summary["preview"] = list(vocab[:32])
    added = parsed.get("added_tokens")
    if isinstance(added, list):
        summary["added_tokens"] = len(added)
    summary["elided"] = False
