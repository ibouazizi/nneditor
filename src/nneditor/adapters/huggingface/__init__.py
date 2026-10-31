"""Native Hugging Face snapshot adapter: config, shards, synthetic LLM graph.

The reader is local. It does not import transformers. Weight payloads stay
as TensorRefs in their safetensors shards until a block is requested.
"""

from nneditor.adapters.huggingface.snapshot import (
    HfIndex,
    HuggingFaceError,
    index_huggingface,
    is_huggingface_snapshot,
    preview_tokenizer_vocab,
    tensor_source_name,
)
from nneditor.adapters.huggingface.to_ir import index_to_document, open_huggingface

__all__ = [
    "HfIndex",
    "HuggingFaceError",
    "index_huggingface",
    "index_to_document",
    "is_huggingface_snapshot",
    "open_huggingface",
    "preview_tokenizer_vocab",
    "tensor_source_name",
]
