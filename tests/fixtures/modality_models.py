"""Tiny genuine library modules for the modality tracers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _seed() -> Any:
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return torch


def write_llava(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    vision = transformers.CLIPVisionConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=8,
        patch_size=8,
        num_channels=3,
    )
    text = transformers.LlamaConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        vocab_size=32,
    )
    config = transformers.LlavaConfig(
        vision_config=vision, text_config=text, image_token_index=1, vocab_size=32
    )
    model = transformers.LlavaForConditionalGeneration(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_whisper(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    config = transformers.WhisperConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=32,
        num_mel_bins=8,
        max_source_positions=32,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = transformers.WhisperForConditionalGeneration(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_t5(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    config = transformers.T5Config(
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        vocab_size=32,
        relative_attention_num_buckets=4,
        dropout_rate=0.0,
    )
    model = transformers.T5ForConditionalGeneration(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_rwkv(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    config = transformers.RwkvConfig(
        hidden_size=16,
        num_hidden_layers=2,
        vocab_size=32,
        intermediate_size=32,
        attention_hidden_size=16,
    )
    model = transformers.RwkvForCausalLM(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_sd3(root: Path) -> Any:
    _seed()
    diffusers = importlib.import_module("diffusers")
    model = diffusers.SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=1,
        attention_head_dim=4,
        num_attention_heads=2,
        joint_attention_dim=8,
        caption_projection_dim=8,
        pooled_projection_dim=8,
        out_channels=4,
    ).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_flux(root: Path) -> Any:
    _seed()
    diffusers = importlib.import_module("diffusers")
    model = diffusers.FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 2, 2),
    ).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_cogvideox(root: Path) -> Any:
    _seed()
    diffusers = importlib.import_module("diffusers")
    model = diffusers.CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        patch_size=2,
        sample_height=8,
        sample_width=8,
        sample_frames=2,
        text_embed_dim=16,
        time_embed_dim=16,
        temporal_compression_ratio=1,
        max_text_seq_length=4,
    ).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_qwen2_vl(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    vision = transformers.Qwen2VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=1,
        temporal_patch_size=2,
        tokens_per_second=1,
        window_size=2,
        out_hidden_size=16,
    )
    text = transformers.Qwen2VLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        bos_token_id=5,
        eos_token_id=6,
    )
    config = transformers.Qwen2VLConfig(
        vision_config=vision,
        text_config=text,
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        vision_end_token_id=4,
        vocab_size=32,
        bos_token_id=5,
        eos_token_id=6,
    )
    model = transformers.Qwen2VLForConditionalGeneration(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_gemma3(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    text = transformers.Gemma3TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        bos_token_id=5,
        eos_token_id=6,
    )
    vision = transformers.SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
        num_channels=3,
    )
    config = transformers.Gemma3Config(
        text_config=text,
        vision_config=vision,
        image_token_index=1,
        mm_tokens_per_image=4,
        vocab_size=32,
        bos_token_id=5,
        eos_token_id=6,
    )
    model = transformers.Gemma3ForConditionalGeneration(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_bitnet(root: Path) -> Any:
    _seed()
    transformers = importlib.import_module("transformers")
    config = transformers.BitNetConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = transformers.BitNetForCausalLM(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model


def write_bitnet_gguf(path: Path, model: Any) -> Path:
    from nneditor.tracing.gguf_bitnet import _state_key
    from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

    reverse = {}
    for name in (
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_sub_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_sub_norm.weight",
        "blk.0.ffn_norm.weight",
    ):
        key = _state_key(name)
        assert key is not None
        reverse[key] = name
    tensors = []
    for key, array in model.state_dict().items():
        tensors.append(
            GgufFixtureTensor(
                reverse[key],
                tuple(array.shape),
                0,
                array.detach().float().cpu().numpy().astype("<f4").tobytes(),
            )
        )
    return write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", 8, "bitnet"),
            ("bitnet.block_count", 4, 1),
            ("bitnet.embedding_length", 4, 16),
            ("bitnet.feed_forward_length", 4, 32),
            ("bitnet.attention.head_count", 4, 2),
            ("bitnet.attention.head_count_kv", 4, 2),
            ("bitnet.vocab_size", 4, 32),
        ),
        tensors=tuple(tensors),
    )


def write_wan(root: Path) -> Any:
    _seed()
    diffusers = importlib.import_module("diffusers")
    model = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    ).eval()
    model.save_pretrained(root, safe_serialization=True)
    return model
