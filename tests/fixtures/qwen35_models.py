"""Small dense hybrid decoder with Prism's actual g128 packing geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf


def write_qwen35(
    path: Path,
    *,
    packing: int = 0,
    endian: str = "little",
    grouped: bool = True,
    tied: bool = False,
    hadamard_version: int = 1,
) -> tuple[Path, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(54)
    h, ff, nk, nv, dim, vocab = 128, 128, 2, 8, 16, 9
    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, h),
        "output_norm.weight": (h,),
        "output.weight": (vocab, h),
    }
    for i in range(2):
        shapes.update(
            {
                f"blk.{i}.{n}": s
                for n, s in {
                    "attn_norm.weight": (h,),
                    "post_attention_norm.weight": (h,),
                    "ffn_gate.weight": (ff, h),
                    "ffn_up.weight": (ff, h),
                    "ffn_down.weight": (h, ff),
                }.items()
            }
        )
    shapes.update(
        {
            "blk.0." + n: s
            for n, s in {
                "attn_qkv.weight": (2 * nk * dim + h, h),
                "attn_gate.weight": (h, h),
                "ssm_conv1d.weight": (2 * nk * dim + h, 4),
                "ssm_dt.bias": (nv,),
                "ssm_a": (nv,),
                "ssm_alpha.weight": (nv, h),
                "ssm_beta.weight": (nv, h),
                "ssm_norm.weight": (dim,),
                "ssm_out.weight": (h, h),
            }.items()
        }
    )
    shapes.update(
        {
            "blk.1." + n: s
            for n, s in {
                "attn_q.weight": (2 * 4 * dim, h),
                "attn_k.weight": (2 * dim, h),
                "attn_v.weight": (2 * dim, h),
                "attn_output.weight": (h, 4 * dim),
                "attn_q_norm.weight": (dim,),
                "attn_k_norm.weight": (dim,),
            }.items()
        }
    )
    if tied:
        shapes.pop("output.weight")
    weights = {
        name: rng.normal(0, 0.08, shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    for name in weights:
        if "norm.weight" in name:
            weights[name] = rng.uniform(0.7, 1.3, shapes[name]).astype(np.float32)
        if name.endswith("ssm_a"):
            weights[name] = -np.exp(weights[name])
    prefix = "<" if endian == "little" else ">"
    tensors = []
    folded = []
    for name, array in weights.items():
        # Only widths divisible by g128 are ternary in this small fixture.
        ternary = packing and array.ndim == 2 and array.shape[-1] == h
        if ternary:
            codes = rng.integers(0, 3, (array.size // 128, 128), dtype=np.uint8)
            scales = np.full((len(codes), 1), 0.0625, dtype=prefix + "f2")
            if packing == 142:
                packed = np.sum(
                    codes.reshape(-1, 32, 4).astype(np.uint16) << np.arange(0, 8, 2),
                    axis=-1,
                ).astype(np.uint8)
                raw = np.concatenate([scales.view(np.uint8), packed], axis=1)
                decoded = codes.astype(np.float32) - 1
            else:
                # Encode the inverse of the fork's fractional base-3 digit map.
                packed = np.empty((len(codes), 26), dtype=np.uint8)
                out = 0
                for start, width, digits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
                    digits_array = codes[:, out : out + width * digits].reshape(
                        -1, digits, width
                    )
                    integer = sum(
                        digits_array[:, n].astype(np.uint16) * 3 ** (digits - 1 - n)
                        for n in range(digits)
                    )
                    packed[:, start : start + width] = (
                        (integer * 256 + 3**digits - 1) // 3**digits
                    ).astype(np.uint8)
                    out += width * digits
                raw = np.concatenate([packed, scales.view(np.uint8)], axis=1)
                decoded = codes.astype(np.float32) - 1
            weights[name] = (decoded * scales.astype(np.float32)).reshape(array.shape)
            payload, dtype = raw.tobytes(), packing
            if name != "token_embd.weight":
                folded.append(name)
        else:
            payload, dtype = array.astype(prefix + "f4").tobytes(), 0
        tensors.append(GgufFixtureTensor(name, array.shape, dtype, payload))
    kv: list[tuple[str, int, Any]] = [
        ("general.architecture", 8, "qwen35"),
        ("qwen35.block_count", 4, 2),
        ("qwen35.full_attention_interval", 4, 2),
        ("qwen35.embedding_length", 4, h),
        ("qwen35.feed_forward_length", 4, ff),
        ("qwen35.attention.head_count", 4, 4),
        ("qwen35.attention.head_count_kv", 4, 2),
        ("qwen35.attention.key_length", 4, dim),
        ("qwen35.rope.dimension_count", 4, 8),
        ("qwen35.attention.layer_norm_rms_epsilon", 6, 1e-6),
        ("qwen35.rope.freq_base", 6, 10000.0),
        ("qwen35.context_length", 4, 32),
        ("qwen35.ssm.conv_kernel", 4, 4),
        ("qwen35.ssm.state_size", 4, dim),
        ("qwen35.ssm.group_count", 4, nk),
        ("qwen35.ssm.time_step_rank", 4, nv),
        ("qwen35.ssm.inner_size", 4, h),
        ("qwen35.vocab_size", 4, vocab),
    ]
    signs = rng.choice([-1, 1], h).astype(np.int32)
    if packing:
        if hadamard_version not in (1, 2):
            raise ValueError("fixture Hadamard version must be 1 or 2")
        if hadamard_version == 2 and not tied:
            raise ValueError("Hadamard version 2 fixture requires tied=True")
        kv += [
            ("prism.hadamard.version", 4, hadamard_version),
            ("prism.hadamard.block_size", 4, 128),
            ("prism.hadamard.transform", 8, "normalized-sylvester-walsh-hadamard"),
            ("prism.hadamard.axis", 8, "input-last-dimension"),
            ("prism.hadamard.sign_mode", 8, "explicit"),
            ("prism.hadamard.weight_names", 9, (8, folded)),
            ("prism.hadamard.inverse_weight_names", 9, (8, ["token_embd.weight"])),
            ("prism.hadamard.sign_widths", 9, (5, [h])),
            ("prism.hadamard.sign_values", 9, (5, signs.tolist())),
            ("prism.hadamard.gdn_v_grouped", 7, grouped),
        ]
        if hadamard_version == 2:
            kv.append(("prism.hadamard.tied_output", 7, True))
    return (
        write_gguf(
            path, version=3, endian=endian, kv=tuple(kv), tensors=tuple(tensors)
        ),
        weights,
        signs,
    )
