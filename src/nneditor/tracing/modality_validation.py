"""Check media/token relationships before loading a modality model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from nneditor.adapters.huggingface.modalities import modality_kind
from nneditor.adapters.media_graphs import positive
from nneditor.ir.core import Document, JsonValue


def validate_modality_inputs(
    document: Document,
    feeds: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    extensions = dict(document.extensions)
    raw = extensions.get("x-huggingface.config")
    config = dict(raw) if isinstance(raw, dict) else {}
    family = modality_kind(config)
    metadata = extensions.get("x-gguf.metadata")
    if isinstance(metadata, dict) and "decoder-tokens" in feeds:
        family = "whisper"
        config = {
            key.removeprefix("whisper."): value
            for key, value in metadata.items()
            if key.startswith("whisper.")
        }
    if family not in {"whisper", "llava", "qwen2_vl", "gemma3"}:
        return
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Missing text configuration")
    vocab = positive(text, "vocab_size")
    tokens = feeds["decoder-tokens" if family == "whisper" else "tokens"]
    if not tokens.size or np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError(f"Token IDs must be inside the model vocabulary [0, {vocab})")
    if family == "whisper":
        if tokens.shape[-1] > positive(config, "max_target_positions", 448):
            raise ValueError("Decoder tokens exceed Whisper's position limit")
        return
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("Missing vision configuration")
    if family == "qwen2_vl":
        count = _qwen_images(vision, feeds)
        image_id = config.get("image_token_id")
        if not np.array_equal(
            feeds["token-types"], (tokens == image_id).astype(np.int64)
        ):
            raise ValueError("token-types must mark exactly the image-token positions")
    else:
        image_id = config.get("image_token_index")
        count = (
            positive(config, "mm_tokens_per_image")
            if family == "gemma3"
            else (
                (positive(vision, "image_size") // positive(vision, "patch_size")) ** 2
                + int(config.get("vision_feature_select_strategy", "default") == "full")
            )
        )
        count *= feeds["pixel-values"].shape[0]
    if int(np.count_nonzero(tokens == image_id)) != count:
        raise ValueError(
            f"The image tensors require exactly {count} image placeholder tokens"
        )


def _qwen_images(
    vision: dict[str, JsonValue],
    feeds: Mapping[str, np.ndarray[Any, Any]],
) -> int:
    grid = feeds["image-grid"]
    merge = positive(vision, "spatial_merge_size", 2)
    if grid.ndim != 2 or grid.shape[1] != 3 or np.any(grid <= 0):
        raise ValueError(
            "image-grid must contain positive temporal, height, width triples"
        )
    if np.any(grid[:, 1:] % merge):
        raise ValueError(
            "image-grid height and width must be divisible by spatial_merge_size"
        )
    # Python integers avoid overflow when validating untrusted input arrays.
    patches = sum(math.prod(int(x) for x in row) for row in grid)
    if patches != feeds["pixel-values"].shape[0]:
        raise ValueError("image-grid does not match the number of image patches")
    return patches // merge**2
