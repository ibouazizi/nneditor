"""A real traceable GGUF with its own complete byte-BPE tokenizer."""

from pathlib import Path

import numpy as np

from nneditor.adapters.gguf.indexer import index_gguf
from nneditor.tokenization import _byte_encoder
from tests.fixtures.gguf_models import (
    GgufFixtureTensor,
    write_gguf,
    write_traceable_llama_gguf,
)


def write_tokenized_gguf(path: Path, *, pre: str = "qwen35") -> Path:
    write_traceable_llama_gguf(path)
    index = index_gguf(path)
    source = path.read_bytes()
    vocabulary = (
        *_byte_encoder().values(),
        "He",
        "Hel",
        "Hell",
        "Hello",
        "Ġw",
        "Ġwo",
        "Ġwor",
        "Ġworl",
        "Ġworld",
        "<|endoftext|>",
        "<|im_end|>",
    )
    merges = (
        "H e",
        "He l",
        "Hel l",
        "Hell o",
        "Ġ w",
        "Ġw o",
        "Ġwo r",
        "Ġwor l",
        "Ġworl d",
    )
    tensors = []
    for tensor in index.tensors:
        assert tensor.payload is not None
        shape = tuple(reversed(tensor.dims))
        payload = source[
            tensor.payload.offset : tensor.payload.offset + tensor.payload.length
        ]
        if tensor.name in {"token_embd.weight", "output.weight"}:
            shape = (len(vocabulary), shape[-1])
            payload = np.resize(
                np.frombuffer(payload, dtype=np.float32), shape
            ).tobytes()
        tensors.append(GgufFixtureTensor(tensor.name, shape, tensor.ggml_type, payload))
    overridden = {
        "llama.vocab_size",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    }
    metadata = (
        *(
            (entry.key, entry.type_id, entry.value)
            for entry in index.kv
            if entry.key not in overridden
        ),
        ("llama.vocab_size", 4, len(vocabulary)),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.pre", 8, pre),
        ("tokenizer.ggml.tokens", 9, (8, vocabulary)),
        ("tokenizer.ggml.merges", 9, (8, merges)),
        ("tokenizer.ggml.add_bos_token", 7, False),
        ("tokenizer.ggml.add_eos_token", 7, False),
        ("tokenizer.ggml.bos_token_id", 4, len(vocabulary) - 2),
        ("tokenizer.ggml.eos_token_id", 4, len(vocabulary) - 1),
        ("tokenizer.ggml.padding_token_id", 4, len(vocabulary) - 2),
    )
    return write_gguf(path, version=3, kv=metadata, tensors=tuple(tensors))
