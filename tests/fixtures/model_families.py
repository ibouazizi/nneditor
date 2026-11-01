"""Small genuine Mamba and Stable Diffusion modules for numerical oracles."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def write_mamba(root: Path, *, second: bool = False) -> Any:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if second:
        config = transformers.Mamba2Config(
            hidden_size=16,
            num_heads=4,
            head_dim=8,
            state_size=4,
            num_hidden_layers=2,
            vocab_size=32,
            n_groups=1,
            chunk_size=4,
        )
        model = transformers.Mamba2ForCausalLM(config).eval()
    else:
        config = transformers.MambaConfig(
            hidden_size=16,
            state_size=4,
            num_hidden_layers=2,
            vocab_size=32,
            time_step_rank=2,
        )
        model = transformers.MambaForCausalLM(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_diffusion(root: Path, *, xl: bool = False) -> Any:
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    options: dict[str, Any] = dict(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    if xl:
        options.update(
            addition_embed_type="text_time",
            addition_time_embed_dim=2,
            projection_class_embeddings_input_dim=16,
        )
    model = diffusers.UNet2DConditionModel(**options).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_decodable_gguf(path: Path) -> Path:
    from nneditor.adapters.gguf.indexer import index_gguf
    from tests.fixtures.gguf_models import (
        GgufFixtureTensor,
        write_gguf,
        write_traceable_llama_gguf,
    )

    write_traceable_llama_gguf(path)
    index = index_gguf(path)
    source = path.read_bytes()
    tensors = tuple(
        GgufFixtureTensor(
            t.name,
            tuple(reversed(t.dims)),
            t.ggml_type,
            source[t.payload.offset : t.payload.offset + t.payload.length],
        )
        for t in index.tensors
        if t.payload is not None
    )
    metadata = (
        *((entry.key, entry.type_id, entry.value) for entry in index.kv),
        ("tokenizer.ggml.model", 8, "gpt2"),
        (
            "tokenizer.ggml.tokens",
            9,
            (8, ("<pad>", "<s>", "</s>", "Hello", "Ġworld", "!", "Ċ", ".")),
        ),
    )
    return write_gguf(path, version=3, kv=metadata, tensors=tensors)
