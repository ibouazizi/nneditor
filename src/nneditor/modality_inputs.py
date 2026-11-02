"""Coordinated diagnostic inputs for speech and vision-language stage graphs.

These presets exercise the model with synthetic media. They do not encode a
user's recording, image, or prompt. Real inputs must use the model's processor.
"""

from __future__ import annotations

from dataclasses import dataclass

from nneditor.adapters.huggingface.modalities import modality_kind
from nneditor.adapters.media_graphs import positive
from nneditor.ir.core import Document, JsonValue


@dataclass(frozen=True, slots=True)
class ModalityInputDefault:
    shape: tuple[int, ...]
    kind: str = "tensor"
    values: tuple[int, ...] | None = None
    note: str = "Synthetic diagnostic input, not processed media."
    unavailable_reason: str | None = None


def _object(config: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing {key} configuration")
    return value


def _token(
    config: dict[str, JsonValue], key: str, vocab: int, default: int | None = None
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab:
        raise ValueError(f"{key} must identify a token in the model vocabulary")
    return value


def _whisper(
    document: Document, config: dict[str, JsonValue]
) -> dict[str, ModalityInputDefault]:
    graph = document.main_graph
    features = next(
        graph.value(key)
        for key in graph.inputs
        if graph.value(key).name == "input-features"
    )
    shape = features.shape
    if (
        shape is None
        or len(shape) != 3
        or not all(isinstance(d, int) and d > 0 for d in shape[1:])
    ):
        raise ValueError("Whisper needs its configured mel count and frame count")
    assert isinstance(shape[1], int) and isinstance(shape[2], int)
    result = {
        "input-features": ModalityInputDefault(
            (1, int(shape[1]), int(shape[2])), "audio-features"
        )
    }
    if any(graph.value(key).name == "decoder-tokens" for key in graph.inputs):
        vocab = positive(config, "vocab_size")
        start = _token(config, "decoder_start_token_id", vocab)
        result["decoder-tokens"] = ModalityInputDefault(
            (1, 1),
            "decoder-tokens",
            (start,),
            "One decoder start token for a single forward pass.",
        )
    return result


def _vision(
    config: dict[str, JsonValue], family: str
) -> dict[str, ModalityInputDefault]:
    vision = _object(config, "vision_config")
    text = _object(config, "text_config")
    vocab = positive(text, "vocab_size")
    result: dict[str, ModalityInputDefault] = {}
    tokens: tuple[int, ...]
    if family == "qwen2_vl":
        merge = positive(vision, "spatial_merge_size", 2)
        if merge > 64:
            raise ValueError("spatial_merge_size exceeds the diagnostic preset limit")
        patch = positive(vision, "patch_size")
        packed = (
            positive(vision, "in_channels", 3)
            * positive(vision, "temporal_patch_size", 2)
            * patch**2
        )
        result["pixel-values"] = ModalityInputDefault(
            (merge**2, packed), "image-patches"
        )
        result["image-grid"] = ModalityInputDefault(
            (1, 3),
            "image-grid",
            (1, merge, merge),
            "Grid matching the generated image patches.",
        )
        image_id = _token(config, "image_token_id", vocab)
        tokens = (
            _token(config, "vision_start_token_id", vocab),
            image_id,
            _token(config, "vision_end_token_id", vocab),
        )
        result["token-types"] = ModalityInputDefault(
            (1, 3),
            "token-types",
            (0, 1, 0),
            "Image positions matching the generated token sequence.",
        )
    else:
        size = positive(vision, "image_size")
        result["pixel-values"] = ModalityInputDefault(
            (1, positive(vision, "num_channels", 3), size, size), "image"
        )
        image_id = _token(config, "image_token_index", vocab)
        if family == "gemma3":
            count = positive(config, "mm_tokens_per_image")
        else:
            patch = positive(vision, "patch_size")
            if size % patch:
                raise ValueError("image_size must be divisible by patch_size")
            strategy = config.get("vision_feature_select_strategy", "default")
            if not isinstance(strategy, str) or strategy not in {"default", "full"}:
                raise ValueError("Unsupported vision feature selection strategy")
            count = (size // patch) ** 2 + int(strategy == "full")
        if count > 65536:
            raise ValueError("Image token count exceeds the diagnostic preset limit")
        text_id = next((i for i in range(min(vocab, 4)) if i != image_id), None)
        if text_id is None:
            raise ValueError("The vocabulary needs a non-image token")
        tokens = (image_id,) * count + (text_id,)
    result["tokens"] = ModalityInputDefault(
        (1, len(tokens)),
        "multimodal-tokens",
        tokens,
        "Diagnostic image placeholders, not an encoded user prompt.",
    )
    return result


def modality_input_defaults(document: Document) -> dict[str, ModalityInputDefault]:
    extensions = dict(document.extensions)
    raw = extensions.get("x-huggingface.config")
    config = dict(raw) if isinstance(raw, dict) else {}
    family = modality_kind(config)
    metadata = extensions.get("x-gguf.metadata")
    if isinstance(metadata, dict) and any(
        document.main_graph.value(k).name == "input-features"
        for k in document.main_graph.inputs
    ):
        family = "whisper"
        config = {
            key.removeprefix("whisper."): value
            for key, value in metadata.items()
            if key.startswith("whisper.")
        }
        # GGUF encoder-only models do not need decoder settings. A full model
        # without start-token metadata requires an explicitly supplied prefix.
        if "decoder_start_token_id" not in config:
            config["decoder_start_token_id"] = metadata.get(
                "tokenizer.ggml.bos_token_id"
            )
    if family not in {"whisper", "llava", "qwen2_vl", "gemma3"}:
        return {}
    try:
        return (
            _whisper(document, config)
            if family == "whisper"
            else _vision(config, family)
        )
    except (ValueError, StopIteration) as error:
        return {
            document.main_graph.value(key).name: ModalityInputDefault(
                (),
                unavailable_reason=(
                    f"{error}. Assign model-processed tensors explicitly"
                ),
            )
            for key in document.main_graph.inputs
        }
