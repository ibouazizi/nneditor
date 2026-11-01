"""Stable Diffusion UNet stages with real conditioning and skip connections."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from nneditor.adapters.huggingface.snapshot import HfIndex, HuggingFaceError
from nneditor.ir.core import Attribute, AttrKind, Graph, JsonValue, Node, Value
from nneditor.ir.identity import ROOT_GRAPH_ID, node_id, value_id

DIFFUSION_CLASSES = frozenset({"UNet2DConditionModel", "UNet2DModel"})


def _conditioning_inputs(
    config: dict[str, JsonValue],
    input_value: Callable[[str, tuple[int | str, ...], str], str],
    integer: Callable[[str, int], int],
    timestep: str,
) -> tuple[list[str], tuple[str, ...]]:
    feeds = [timestep]
    if config.get("class_embed_type") not in (None, "timestep"):
        raise HuggingFaceError(
            "This UNet class-conditioning projection is not supported."
        )
    if config.get("encoder_hid_dim_type") is not None:
        raise HuggingFaceError("This UNet encoder projection is not supported.")
    context: tuple[str, ...] = ()
    if config.get("_class_name") == "UNet2DConditionModel":
        cross = config.get("cross_attention_dim", 1280)
        if isinstance(cross, list) and cross:
            cross = cross[0]
        if not isinstance(cross, int) or cross <= 0:
            raise HuggingFaceError("UNet cross_attention_dim must be positive.")
        encoder = config.get("encoder_hid_dim", cross)
        context = (
            input_value(
                "encoder_hidden_states",
                (1, 77, encoder if isinstance(encoder, int) else cross),
                "float32",
            ),
        )
        addition = config.get("addition_embed_type")
        if addition == "text_time":
            time_dim = integer("addition_time_embed_dim", 256)
            projection = integer("projection_class_embeddings_input_dim", 2816)
            time_count = 5 if cross == 1280 and projection == 2560 else 6
            text_dim = projection - time_dim * time_count
            if text_dim <= 0:
                raise HuggingFaceError(
                    "Invalid SDXL pooled-text conditioning dimensions."
                )
            feeds += [
                input_value("text_embeds", (1, text_dim), "float32"),
                input_value("time_ids", (1, time_count), "float32"),
            ]
        elif addition not in (None, "text"):
            raise HuggingFaceError(
                f"Unsupported UNet addition_embed_type {addition!r}."
            )
        elif addition == "text":
            feeds.extend(context)
    if (
        config.get("num_class_embeds") is not None
        or config.get("class_embed_type") is not None
    ):
        feeds.append(input_value("class_labels", (1,), "int64"))
    return feeds, context


def diffusion_graph(index: HfIndex) -> Graph:
    config = index.config
    values: dict[str, Value] = {}
    nodes: list[Node] = []
    inputs: list[str] = []

    def integer(key: str, default: int) -> int:
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise HuggingFaceError(f"Diffusion config needs a positive {key}.")
        return value

    def value(
        name: str, shape: tuple[int | str, ...] | None = None, dtype: str = "float32"
    ) -> str:
        key = value_id(ROOT_GRAPH_ID, name)
        values[key] = Value(key, name=name, element_type=dtype, shape=shape)
        return key

    def input_value(
        name: str, shape: tuple[int | str, ...], dtype: str = "float32"
    ) -> str:
        key = value(name, shape, dtype)
        inputs.append(key)
        return key

    def stage(
        name: str,
        label: str,
        operation: str,
        feeds: tuple[str, ...],
        outputs: tuple[str, ...],
        module: str,
        prefixes: tuple[str, ...] | None = None,
    ) -> None:
        identifier, stability = node_id(ROOT_GRAPH_ID, name=name, ordinal=len(nodes))
        weights = tuple(
            Attribute(tensor.name, AttrKind.TENSOR, tensor.id)
            for tensor in index.tensors
            if any(
                tensor.name.startswith(prefix + ".")
                for prefix in (prefixes or (module,))
            )
        )
        nodes.append(
            Node(
                identifier,
                stability,
                operation,
                feeds,
                outputs,
                domain="nneditor.diffusion",
                source_name=label,
                attributes=(Attribute("module", AttrKind.STRING, module), *weights),
            )
        )

    size = config.get("sample_size", 64)
    if isinstance(size, int) and size > 0:
        height, width = size, size
    elif (
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(i, int) and i > 0 for i in size)
    ):
        height, width = cast(int, size[0]), cast(int, size[1])
    else:
        raise HuggingFaceError(
            "UNet sample_size must describe positive spatial dimensions."
        )
    sample = input_value("sample", (1, integer("in_channels", 4), height, width))
    timestep = input_value("timestep", (1,), "int64")
    conditioning_inputs, context = _conditioning_inputs(
        config, input_value, integer, timestep
    )
    embedding = value("conditioning")
    stage(
        "conditioning",
        "Time and auxiliary conditioning",
        "Conditioning",
        tuple(conditioning_inputs),
        (embedding,),
        "conditioning",
        ("time_embedding", "add_embedding", "class_embedding"),
    )
    previous = value("conv-in")
    stage(
        "conv-in", "Latent input projection", "Conv", (sample,), (previous,), "conv_in"
    )
    residuals = [previous]
    down = config.get("down_block_types", ["DownBlock2D"])
    up = config.get("up_block_types", ["UpBlock2D"])
    if (
        not isinstance(down, list)
        or not isinstance(up, list)
        or not down
        or len(down) != len(up)
    ):
        raise HuggingFaceError(
            "UNet needs matching nonempty down_block_types and up_block_types."
        )
    counts = config.get("layers_per_block", 2)
    layers = [counts] * len(down) if isinstance(counts, int) else counts
    if (
        not isinstance(layers, list)
        or len(layers) != len(down)
        or not all(isinstance(i, int) and i > 0 for i in layers)
    ):
        raise HuggingFaceError("UNet layers_per_block must match its blocks.")
    layer_counts = cast(list[int], layers)
    for index_value, block_type in enumerate(down):
        count = layer_counts[index_value] + (index_value < len(down) - 1)
        outputs = (
            *(value(f"down-{index_value:03d}-skip-{i}") for i in range(count - 1)),
            value(f"down-{index_value:03d}"),
        )
        feeds = (
            previous,
            embedding,
            *(context if "CrossAttn" in str(block_type) else ()),
        )
        stage(
            f"down-{index_value:03d}",
            f"Down block {index_value}",
            "UNetDown",
            feeds,
            outputs,
            f"down_blocks.{index_value}",
        )
        previous = outputs[-1]
        residuals.extend(outputs)
    if config.get("mid_block_type", "UNetMidBlock2D") is not None:
        output = value("mid")
        stage(
            "mid",
            "Bottleneck",
            "UNetMid",
            (previous, embedding, *context),
            (output,),
            "mid_block",
        )
        previous = output
    for index_value, block_type in enumerate(up):
        count = layer_counts[-index_value - 1] + 1
        if len(residuals) < count:
            raise HuggingFaceError("UNet skip connection counts are inconsistent.")
        skips = tuple(residuals[-count:])
        del residuals[-count:]
        output = value(f"up-{index_value:03d}")
        feeds = (
            previous,
            *skips,
            embedding,
            *(context if "CrossAttn" in str(block_type) else ()),
        )
        stage(
            f"up-{index_value:03d}",
            f"Up block {index_value}",
            "UNetUp",
            feeds,
            (output,),
            f"up_blocks.{index_value}",
        )
        previous = output
    output = value("noise-prediction", (1, integer("out_channels", 4), height, width))
    stage(
        "noise-prediction",
        "Noise prediction",
        "Conv",
        (previous,),
        (output,),
        "conv_out",
        ("conv_norm_out", "conv_out"),
    )
    return Graph(
        id=ROOT_GRAPH_ID,
        name="Stable Diffusion denoiser",
        nodes=nodes,
        values=values.values(),
        inputs=inputs,
        outputs=(output,),
        initializers=[tensor.id for tensor in index.tensors],
    )
