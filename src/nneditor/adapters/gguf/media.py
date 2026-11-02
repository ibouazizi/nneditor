"""GGUF media tensor names and metadata-only graph construction."""

from __future__ import annotations

import re
from collections.abc import Mapping

from nneditor.adapters.gguf.index import GgufIndex
from nneditor.adapters.media_graphs import clip_graph, whisper_graph
from nneditor.ir.core import Graph, JsonValue

_CLIP_BLOCK = re.compile(r"v\.blk\.(\d+)\.([^.]+)\.(weight|bias)\Z")
_AUDIO_BLOCK = re.compile(r"a\.blk\.(\d+)\.([^.]+)\.(weight|bias)\Z")
_WHISPER_BLOCK = re.compile(r"(encoder|decoder)\.blocks\.(\d+)\.(.+)\.(weight|bias)\Z")
_CLIP_LEAVES = {
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
}
_AUDIO_LEAVES = {
    "ln1": "self_attn_layer_norm",
    "ln2": "final_layer_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_up": "fc1",
    "ffn_down": "fc2",
}


def clip_state_key(name: str) -> str | None:
    """Hugging Face CLIP vision parameter for one llama.cpp vision tensor."""
    fixed = {
        "v.class_embd": "embeddings.class_embedding",
        "v.patch_embd.weight": "embeddings.patch_embedding.weight",
        "v.patch_embd.bias": "embeddings.patch_embedding.bias",
        "v.position_embd.weight": "embeddings.position_embedding.weight",
        "v.pre_ln.weight": "pre_layrnorm.weight",
        "v.pre_ln.bias": "pre_layrnorm.bias",
        "v.post_ln.weight": "post_layernorm.weight",
        "v.post_ln.bias": "post_layernorm.bias",
    }
    if name in fixed:
        return fixed[name]
    match = _CLIP_BLOCK.match(name)
    if match is None:
        return None
    leaf = _CLIP_LEAVES.get(match.group(2))
    if leaf is None:
        return None
    return f"encoder.layers.{match.group(1)}.{leaf}.{match.group(3)}"


def whisper_state_key(name: str) -> str | None:
    """Hugging Face Whisper parameter for one GGUF audio tensor."""
    fixed = {
        "a.conv1d.1.weight": "model.encoder.conv1.weight",
        "a.conv1d.1.bias": "model.encoder.conv1.bias",
        "a.conv1d.2.weight": "model.encoder.conv2.weight",
        "a.conv1d.2.bias": "model.encoder.conv2.bias",
        "a.position_embd.weight": "model.encoder.embed_positions.weight",
        "a.post_ln.weight": "model.encoder.layer_norm.weight",
        "a.post_ln.bias": "model.encoder.layer_norm.bias",
        "encoder.conv1.weight": "model.encoder.conv1.weight",
        "encoder.conv1.bias": "model.encoder.conv1.bias",
        "encoder.conv2.weight": "model.encoder.conv2.weight",
        "encoder.conv2.bias": "model.encoder.conv2.bias",
        "encoder.positional_embedding": "model.encoder.embed_positions.weight",
        "encoder.ln_post.weight": "model.encoder.layer_norm.weight",
        "encoder.ln_post.bias": "model.encoder.layer_norm.bias",
        "decoder.token_embedding.weight": "model.decoder.embed_tokens.weight",
        "decoder.positional_embedding": "model.decoder.embed_positions.weight",
        "decoder.ln.weight": "model.decoder.layer_norm.weight",
        "decoder.ln.bias": "model.decoder.layer_norm.bias",
        "proj_out.weight": "proj_out.weight",
    }
    if name in fixed:
        return fixed[name]
    audio = _AUDIO_BLOCK.match(name)
    if audio is not None:
        leaf = _AUDIO_LEAVES.get(audio.group(2))
        if leaf is None:
            return None
        return f"model.encoder.layers.{audio.group(1)}.{leaf}.{audio.group(3)}"
    block = _WHISPER_BLOCK.match(name)
    if block is None:
        return None
    side, index, leaf, suffix = block.groups()
    translated = _whisper_leaf(leaf)
    if translated is None:
        return None
    return f"model.{side}.layers.{index}.{translated}.{suffix}"


def _whisper_leaf(leaf: str) -> str | None:
    table = {
        "attn.query": "self_attn.q_proj",
        "attn.key": "self_attn.k_proj",
        "attn.value": "self_attn.v_proj",
        "attn.out": "self_attn.out_proj",
        "cross_attn.query": "encoder_attn.q_proj",
        "cross_attn.key": "encoder_attn.k_proj",
        "cross_attn.value": "encoder_attn.v_proj",
        "cross_attn.out": "encoder_attn.out_proj",
        "attn_ln": "self_attn_layer_norm",
        "cross_attn_ln": "encoder_attn_layer_norm",
        "mlp_ln": "final_layer_norm",
        "mlp.0": "fc1",
        "mlp.2": "fc2",
    }
    return table.get(leaf)


def media_kind(names: set[str]) -> str:
    if any(name.startswith(("v.blk.", "v.patch_embd")) for name in names):
        return "vision"
    if any(
        name.startswith(
            ("a.blk.", "a.conv1d.", "encoder.conv", "encoder.blocks.", "encoder.blk.")
        )
        for name in names
    ):
        return "whisper"
    raise ValueError("This GGUF has neither a CLIP vision tower nor a Whisper encoder.")


def full_whisper(names: set[str]) -> bool:
    return any(
        name.startswith(("decoder.blocks.", "decoder.blk.", "decoder.token_embedding"))
        for name in names
    )


def media_config(
    metadata: Mapping[str, JsonValue], kind: str, *, full: bool
) -> dict[str, JsonValue]:
    if kind == "vision":
        keys = {
            "hidden_size": "embedding_length",
            "intermediate_size": "feed_forward_length",
            "num_hidden_layers": "block_count",
            "num_attention_heads": "attention.head_count",
            "image_size": "image_size",
            "patch_size": "patch_size",
            "num_channels": "num_channels",
        }
        result = {
            key: metadata.get("clip.vision." + suffix) for key, suffix in keys.items()
        }
        if result["num_channels"] is None:
            result["num_channels"] = 3
        return result
    keys = {
        "d_model": "embedding_length",
        "encoder_layers": "block_count",
        "encoder_attention_heads": "attention.head_count",
        "encoder_ffn_dim": "feed_forward_length",
        "num_mel_bins": "num_mel_bins",
        "max_source_positions": "max_pos_emb",
    }
    result = {
        key: metadata.get("whisper." + key, metadata.get("clip.audio." + suffix))
        for key, suffix in keys.items()
    }
    for key, default in {
        "decoder_layers": 1,
        "decoder_attention_heads": 1,
        "decoder_ffn_dim": 32,
        "vocab_size": 32,
        "max_target_positions": 16,
    }.items():
        result[key] = metadata.get("whisper." + key) if full else default
    return result


def media_graph(index: GgufIndex) -> Graph | None:
    if index.kv_value("general.architecture") not in {"clip", "whisper"}:
        return None
    names = {tensor.name for tensor in index.tensors}
    kind = media_kind(names)
    full = kind == "whisper" and full_whisper(names)
    config = media_config(
        {entry.key: entry.value for entry in index.kv}, kind, full=full
    )
    translate = clip_state_key if kind == "vision" else whisper_state_key
    weights = {
        key: tensor.id
        for tensor in index.tensors
        if (key := translate(tensor.name)) is not None
    }
    # Tied output projections still belong to the output stage.
    if (
        full
        and "proj_out.weight" not in weights
        and "model.decoder.embed_tokens.weight" in weights
    ):
        weights["proj_out.weight"] = weights["model.decoder.embed_tokens.weight"]
    return (
        clip_graph(weights, config)
        if kind == "vision"
        else whisper_graph(weights, config, full=full)
    )
