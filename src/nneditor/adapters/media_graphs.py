"""Whisper and CLIP stage graphs shared by safetensors and GGUF readers."""

from __future__ import annotations

from collections.abc import Mapping

from nneditor.adapters.stages import StageBuilder
from nneditor.ir.core import Graph, JsonValue


def positive(
    config: Mapping[str, JsonValue], key: str, default: int | None = None
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"This checkpoint needs a positive {key}.")
    return value


def whisper_graph(
    weights: Mapping[str, str],
    config: Mapping[str, JsonValue],
    *,
    full: bool = True,
) -> Graph:
    built = StageBuilder(weights, "Whisper")
    domain = "nneditor.audio"
    source = built.input(
        "input-features",
        (
            "batch",
            positive(config, "num_mel_bins"),
            2 * positive(config, "max_source_positions", 1500),
        ),
    )
    hidden = built.step(
        "conv1", "Conv", (source,), "model.encoder.conv1", domain=domain
    )
    hidden = built.step(
        "conv1-gelu",
        "GELU",
        (hidden,),
        "model.encoder.conv2",
        domain=domain,
        capture_input="input",
    )
    hidden = built.step(
        "conv2", "Conv", (hidden,), "model.encoder.conv2", domain=domain
    )
    positions = built.step(
        "encoder-positions",
        "PositionEmbedding",
        (source,),
        "model.encoder.embed_positions",
        domain=domain,
    )
    hidden = built.step(
        "encoder-embedding",
        "GELUTransposeAdd",
        (hidden, positions),
        "model.encoder.layers.0",
        domain=domain,
        capture_input="hidden_states",
    )
    hidden = built.blocks(
        "encoder-block",
        positive(config, "encoder_layers"),
        "EncoderBlock",
        "model.encoder.layers.{index}",
        hidden,
        domain=domain,
    )
    encoded = built.step(
        "encoder-norm",
        "LayerNorm",
        (hidden,),
        "model.encoder.layer_norm",
        domain=domain,
    )
    if not full:
        return built.graph(encoded)
    tokens = built.input("decoder-tokens", ("batch", "target"), "int64")
    hidden = built.step(
        "decoder-embed",
        "Embedding",
        (tokens,),
        "model.decoder.embed_tokens",
        domain=domain,
    )
    positions = built.step(
        "decoder-positions",
        "PositionEmbedding",
        (tokens,),
        "model.decoder.embed_positions",
        domain=domain,
    )
    hidden = built.step(
        "decoder-embedding",
        "Add",
        (hidden, positions),
        "model.decoder.layers.0",
        domain=domain,
        capture_input="hidden_states",
    )
    hidden = built.blocks(
        "decoder-block",
        positive(config, "decoder_layers"),
        "DecoderBlock",
        "model.decoder.layers.{index}",
        hidden,
        domain=domain,
        also=(encoded,),
    )
    return built.graph(
        built.finish(
            hidden,
            ("model.decoder.layer_norm", "proj_out"),
            positive(config, "vocab_size"),
            domain,
            sequence_dimension="target",
        )
    )


def clip_graph(weights: Mapping[str, str], config: Mapping[str, JsonValue]) -> Graph:
    built = StageBuilder(weights, "CLIP vision")
    domain = "nneditor.vision"
    size = positive(config, "image_size")
    pixels = built.input(
        "pixel-values", ("batch", positive(config, "num_channels", 3), size, size)
    )
    hidden = built.step(
        "vision-embed", "VisionEmbed", (pixels,), "embeddings", domain=domain
    )
    hidden = built.step(
        "vision-pre-norm", "LayerNorm", (hidden,), "pre_layrnorm", domain=domain
    )
    hidden = built.blocks(
        "vision-block",
        positive(config, "num_hidden_layers"),
        "VisionBlock",
        "encoder.layers.{index}",
        hidden,
        domain=domain,
    )
    pooled = built.step(
        "cls-token",
        "SelectCLS",
        (hidden,),
        "post_layernorm",
        domain=domain,
        capture_input="input",
    )
    built.step("pooled-output", "LayerNorm", (pooled,), "post_layernorm", domain=domain)
    output = built.step("vision-output", "Output", (hidden,), "output", domain=domain)
    return built.graph(output)
