"""The compact index produced by the native GGUF indexer.

Nothing here holds tensor values. A tensor is a name, a ggml type id, file
order dimensions, and a byte range in the data section. Identity is
``gguf-index:`` — a digest of the header, KV block, tensor table, padding up
to the data section, and the file size — so opening a 27B-class artifact does
not hash the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from nneditor.diagnostics import DiagnosticLog
from nneditor.ir.core import JsonValue, Storage
from nneditor.storage.reader import ByteRange, ReadStats

__all__ = ["GgufIndex", "GgufKvEntry", "GgufTensorIndex"]


@dataclass(frozen=True, slots=True)
class GgufKvEntry:
    """One GGUF metadata key, either parsed or retained as a byte range."""

    key: str
    type_id: int
    value: JsonValue
    elided: bool
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class GgufTensorIndex:
    """One GGUF tensor, described by location only."""

    id: str
    name: str
    ggml_type: int
    element_type: str
    dims: tuple[int, ...]
    n_elems: int
    relative_offset: int
    storage: Storage
    payload: ByteRange | None
    n_bytes: int | None


@dataclass(frozen=True, slots=True)
class GgufIndex:
    """Everything the IR converter needs after a header-only GGUF walk."""

    path: Path
    version: int
    endian: str
    alignment: int
    n_tensors: int
    n_kv: int
    kv_end: int
    tensor_table_end: int
    data_start: int
    file_size: int
    content_hash: str
    kv: tuple[GgufKvEntry, ...]
    tensors: tuple[GgufTensorIndex, ...]
    diagnostics: DiagnosticLog
    stats: ReadStats

    @property
    def kv_map(self) -> MappingProxyType[str, GgufKvEntry]:
        return MappingProxyType({entry.key: entry for entry in self.kv})

    def kv_value(self, key: str) -> JsonValue:
        entry = self.kv_map.get(key)
        if entry is None:
            return None
        return entry.value

    @property
    def payload_ranges(self) -> tuple[ByteRange, ...]:
        return tuple(
            tensor.payload for tensor in self.tensors if tensor.payload is not None
        )
