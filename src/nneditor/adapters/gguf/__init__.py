"""Native GGUF adapter: v1/v2/v3 indexer, synthetic LLM graph, block loads.

The parser is local. It does not import gguf-py. Unknown ggml type ids,
including fork-only codes, surface as ``gguf.type_<n>``.
"""

from nneditor.adapters.gguf.blocks import (
    block_tensor_ids,
    load_block_tensors,
    tensor_source_name,
)
from nneditor.adapters.gguf.index import GgufIndex, GgufKvEntry, GgufTensorIndex
from nneditor.adapters.gguf.indexer import GgufIndexError, index_gguf
from nneditor.adapters.gguf.to_ir import index_to_document, open_gguf

__all__ = [
    "GgufIndex",
    "GgufIndexError",
    "GgufKvEntry",
    "GgufTensorIndex",
    "block_tensor_ids",
    "index_gguf",
    "index_to_document",
    "load_block_tensors",
    "open_gguf",
    "tensor_source_name",
]
