"""On-demand GGUF block loads through coalesced tensor-store reads.

A transformer block's tensors sit next to each other in the GGUF data
section. Reading them as one bounded span (gaps up to the 32-byte default
alignment, span capped at 64 MiB) is how a 27B-class artifact stays
inspectable without pulling neighbouring blocks.
"""

from __future__ import annotations

from collections.abc import Sequence

from nneditor.cancellation import CancellationToken
from nneditor.ir.core import Document
from nneditor.ir.identity import decode_segment
from nneditor.storage.store import TensorStore

__all__ = ["block_tensor_ids", "load_block_tensors", "tensor_source_name"]

_DEFAULT_MAX_GAP = 32
_DEFAULT_MAX_SPAN = 64 * 1024 * 1024


def tensor_source_name(tensor_id: str) -> str:
    """The GGUF tensor name encoded in an initializer id."""
    _, separator, encoded = tensor_id.partition("#")
    if not separator:
        return tensor_id
    return decode_segment(encoded)


def block_tensor_ids(document: Document, block: int) -> tuple[str, ...]:
    """Initializer ids whose GGUF names belong to ``blk.{block}.*``."""
    prefix = f"blk.{block}."
    return tuple(
        tensor_id
        for tensor_id in document.tensors
        if tensor_source_name(tensor_id).startswith(prefix)
    )


def load_block_tensors(
    store: TensorStore,
    block: int,
    *,
    token: CancellationToken | None = None,
    max_gap: int = _DEFAULT_MAX_GAP,
    max_span: int = _DEFAULT_MAX_SPAN,
) -> dict[str, bytes]:
    """Read every tensor in one GGUF block, coalescing adjacent ranges."""
    tensor_ids: Sequence[str] = block_tensor_ids(store.document, block)
    return store.read_group(tensor_ids, max_gap=max_gap, max_span=max_span, token=token)
