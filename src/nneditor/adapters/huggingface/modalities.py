"""Vision, DiT, video, speech, sequence-to-sequence, and RWKV graphs.

Each stage names the library module a tracer hooks. A stage whose module is
``output`` is the model return, including unpatchifying and tied heads.
"""

from __future__ import annotations

from nneditor.adapters.huggingface.snapshot import HfIndex, HuggingFaceError
from nneditor.adapters.stages import StageBuilder
from nneditor.ir.core import Graph, JsonValue

MODALITY_ALIASES: dict[str, frozenset[str]] = {
    "llava": frozenset({"llava", "LlavaForConditionalGeneration"}),
    "qwen2_vl": frozenset({"qwen2_vl", "Qwen2VLForConditionalGeneration"}),
    "gemma3": frozenset({"gemma3", "Gemma3ForConditionalGeneration"}),
    "bitnet": frozenset({"bitnet", "BitNetForCausalLM"}),
    "whisper": frozenset({"whisper", "WhisperForConditionalGeneration"}),
    "t5": frozenset({"t5", "T5ForConditionalGeneration"}),
    "rwkv": frozenset({"rwkv", "RwkvForCausalLM"}),
    "sd3": frozenset({"SD3Transformer2DModel"}),
    "flux": frozenset({"FluxTransformer2DModel"}),
    "cogvideox": frozenset({"CogVideoXTransformer3DModel"}),
    "wan": frozenset({"WanTransformer3DModel"}),
}
BLOCK_OPS = frozenset(
    {"VisionBlock", "DecoderBlock", "EncoderBlock", "RecurrentBlock", "DiTBlock"}
)


def modality_kind(config: dict[str, JsonValue]) -> str | None:
    """The supported family declared by a config, if it names one."""
    names: set[str] = set()
    for key in ("_class_name", "model_type"):
        value = config.get(key)
        if isinstance(value, str) and value:
            names.add(value)
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        names.update(item for item in architectures if isinstance(item, str) and item)
    for kind, aliases in MODALITY_ALIASES.items():
        if names & aliases:
            return kind
    return None


class _Builder(StageBuilder):
    def __init__(self, index: HfIndex, title: str) -> None:
        super().__init__({tensor.name: tensor.id for tensor in index.tensors}, title)


def _mapping(config: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    raise HuggingFaceError(f"This checkpoint needs a {key} object.")


def _positive(config: dict[str, JsonValue], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HuggingFaceError(f"This checkpoint needs a positive {key}.")
    return value


def _optional_positive(config: dict[str, JsonValue], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HuggingFaceError(f"This checkpoint needs a positive {key}.")
    return value


def _vision_selection(
    config: dict[str, JsonValue], states: list[str]
) -> tuple[str, ...]:
    layers = config.get("vision_feature_layer", -2)
    indices = layers if isinstance(layers, list) else [layers]
    if not indices or any(
        isinstance(i, bool)
        or not isinstance(i, int)
        or not -len(states) <= i < len(states)
        for i in indices
    ):
        raise HuggingFaceError("vision_feature_layer is outside the vision tower.")
    return tuple(states[i] for i in indices if isinstance(i, int))


def _image_text_merge(
    built: _Builder, hidden: str, projected: str, *, also: tuple[str, ...] = ()
) -> str:
    return built.step(
        "image-text-merge",
        "ImageTextMerge",
        (hidden, projected, *also),
        "model.language_model.layers.0",
        domain="nneditor.vision",
        capture_input="hidden_states",
    )


def _llava(index: HfIndex) -> Graph:
    vision = _mapping(index.config, "vision_config")
    text = _mapping(index.config, "text_config")
    built = _Builder(index, "LLaVA")
    image = _positive(vision, "image_size")
    pixels = built.input(
        "pixel-values", (1, _positive(vision, "num_channels"), image, image)
    )
    tokens = built.input("tokens", ("batch", "seq"), "int64")
    visual = built.value("vision-embed")
    built.stage(
        "vision-embed",
        "Vision patch embedding",
        "VisionEmbed",
        (pixels,),
        (visual,),
        "model.vision_tower.embeddings",
        domain="nneditor.vision",
    )
    visual = built.step(
        "vision-pre-norm",
        "LayerNorm",
        (visual,),
        "model.vision_tower.pre_layrnorm",
        domain="nneditor.vision",
    )
    vision_states = [visual]
    visual = built.blocks(
        "vision-block",
        _positive(vision, "num_hidden_layers"),
        "VisionBlock",
        "model.vision_tower.encoder.layers.{index}",
        visual,
        domain="nneditor.vision",
    )
    vision_states.extend(
        built.values[key].id
        for key in built.values
        if built.values[key].name.startswith("vision-block-")
    )
    cls_token = built.step(
        "vision-cls",
        "SelectCLS",
        (visual,),
        "model.vision_tower.post_layernorm",
        domain="nneditor.vision",
        capture_input="input",
    )
    built.step(
        "vision-norm",
        "LayerNorm",
        (cls_token,),
        "model.vision_tower.post_layernorm",
        domain="nneditor.vision",
    )
    selected = _vision_selection(index.config, vision_states)
    features = built.step(
        "vision-features",
        "VisionFeatureSelection",
        selected,
        "model.multi_modal_projector",
        domain="nneditor.vision",
        capture_input="image_features",
    )
    projected = built.value("projector")
    built.stage(
        "projector",
        "Vision projector",
        "Projector",
        (features,),
        (projected,),
        "model.multi_modal_projector",
        domain="nneditor.vision",
    )
    hidden = built.value("text-embed")
    built.stage(
        "text-embed",
        "Token embedding",
        "Embedding",
        (tokens,),
        (hidden,),
        "model.language_model.embed_tokens",
        domain="nneditor.vision",
    )
    hidden = _image_text_merge(built, hidden, projected)
    hidden = built.blocks(
        "text-block",
        _positive(text, "num_hidden_layers"),
        "DecoderBlock",
        "model.language_model.layers.{index}",
        hidden,
        domain="nneditor.vision",
    )
    logits = built.finish(
        hidden,
        ("model.language_model.norm", "lm_head"),
        _positive(text, "vocab_size"),
        "nneditor.vision",
    )
    return built.graph(logits)


def _encoder_decoder(
    index: HfIndex,
    *,
    title: str,
    domain: str,
    source_name: str,
    source_shape: tuple[int | str, ...],
    source_dtype: str,
    encoder_count: int,
    encoder_module: str,
    encoder_norm: str,
    decoder_count: int,
    decoder_embed: str,
    decoder_module: str,
    decoder_norm: str,
    head: str,
    vocab: int,
) -> Graph:
    built = _Builder(index, title)
    source = built.input(source_name, source_shape, source_dtype)
    tokens = built.input("decoder-tokens", ("batch", "target"), "int64")
    encoded = built.blocks(
        "encoder-block",
        encoder_count,
        "EncoderBlock",
        encoder_module,
        source,
        domain=domain,
    )
    normalized = built.value("encoder-norm")
    built.stage(
        "encoder-norm",
        "Encoder normalization",
        "LayerNorm",
        (encoded,),
        (normalized,),
        encoder_norm,
        domain=domain,
    )
    encoded = normalized
    decoded = built.value("decoder-embed")
    built.stage(
        "decoder-embed",
        "Decoder embedding",
        "Embedding",
        (tokens,),
        (decoded,),
        decoder_embed,
        domain=domain,
    )
    decoded = built.blocks(
        "decoder-block",
        decoder_count,
        "DecoderBlock",
        decoder_module,
        decoded,
        domain=domain,
        also=(encoded,),
    )
    logits = built.finish(decoded, (decoder_norm, head), vocab, domain)
    return built.graph(logits)


def _whisper(index: HfIndex) -> Graph:
    from nneditor.adapters.media_graphs import whisper_graph

    try:
        return whisper_graph(
            {tensor.name: tensor.id for tensor in index.tensors}, index.config
        )
    except ValueError as error:
        raise HuggingFaceError(str(error)) from error


def _t5(index: HfIndex) -> Graph:
    config = index.config
    return _encoder_decoder(
        index,
        title="T5",
        domain="nneditor.seq2seq",
        source_name="source-tokens",
        source_shape=("batch", "source"),
        source_dtype="int64",
        encoder_count=_positive(config, "num_layers"),
        encoder_module="encoder.block.{index}",
        encoder_norm="encoder.final_layer_norm",
        decoder_count=_optional_positive(
            config, "num_decoder_layers", _positive(config, "num_layers")
        ),
        decoder_embed="decoder.embed_tokens",
        decoder_module="decoder.block.{index}",
        decoder_norm="decoder.final_layer_norm",
        head="lm_head",
        vocab=_positive(config, "vocab_size"),
    )


def _rwkv(index: HfIndex) -> Graph:
    config = index.config
    built = _Builder(index, "RWKV")
    tokens = built.input("tokens", ("batch", "seq"), "int64")
    hidden = built.value("embed")
    built.stage(
        "embed",
        "Token embedding",
        "Embedding",
        (tokens,),
        (hidden,),
        "rwkv.embeddings",
        domain="nneditor.recurrent",
    )
    hidden = built.blocks(
        "block",
        _positive(config, "num_hidden_layers"),
        "RecurrentBlock",
        "rwkv.blocks.{index}",
        hidden,
        domain="nneditor.recurrent",
    )
    logits = built.finish(
        hidden,
        ("rwkv.ln_out", "head"),
        _positive(config, "vocab_size"),
        "nneditor.recurrent",
    )
    return built.graph(logits)


def _prediction(built: _Builder, hidden: str, domain: str) -> str:
    output = built.value("noise-prediction")
    built.stage(
        "noise-prediction",
        "Denoised sample",
        "Output",
        (hidden,),
        (output,),
        "output",
        domain=domain,
        weights_from="proj_out",
    )
    return output


def _sd3(index: HfIndex) -> Graph:
    config = index.config
    size = _positive(config, "sample_size")
    patch = _positive(config, "patch_size")
    if size % patch:
        raise HuggingFaceError("SD3 sample_size must be divisible by patch_size.")
    channels = _positive(config, "in_channels")
    text = _positive(config, "joint_attention_dim")
    pooled = _positive(config, "pooled_projection_dim")
    built = _Builder(index, "SD3")
    sample = built.input("sample", (1, channels, size, size))
    timestep = built.input("timestep", (1,), "int64")
    context = built.input("encoder-hidden-states", ("batch", "text", text))
    pooled_input = built.input("pooled-projections", ("batch", pooled))
    hidden = built.value("patch-embed")
    built.stage(
        "patch-embed",
        "Patch embedding",
        "PatchEmbed",
        (sample,),
        (hidden,),
        "pos_embed",
        domain="nneditor.diffusion",
    )
    conditioning = built.value("conditioning")
    built.stage(
        "conditioning",
        "Time and pooled-text conditioning",
        "Conditioning",
        (timestep, pooled_input),
        (conditioning,),
        "time_text_embed",
        domain="nneditor.diffusion",
    )
    text_hidden = built.value("context")
    built.stage(
        "context",
        "Text context",
        "Context",
        (context,),
        (text_hidden,),
        "context_embedder",
        domain="nneditor.diffusion",
    )
    hidden = built.blocks(
        "block",
        _positive(config, "num_layers"),
        "DiTBlock",
        "transformer_blocks.{index}",
        hidden,
        domain="nneditor.diffusion",
        also=(conditioning, text_hidden),
    )
    return built.graph(_prediction(built, hidden, "nneditor.diffusion"))


def _flux(index: HfIndex) -> Graph:
    config = index.config
    channels = _positive(config, "in_channels")
    text = _positive(config, "joint_attention_dim")
    pooled = _positive(config, "pooled_projection_dim")
    built = _Builder(index, "Flux")
    sample = built.input("sample", ("batch", "image_seq", channels))
    timestep = built.input("timestep", (1,), "int64")
    context = built.input("encoder-hidden-states", ("batch", "text_seq", text))
    pooled_input = built.input("pooled-projections", ("batch", pooled))
    built.input("img-ids", ("image_seq", 3))
    built.input("txt-ids", ("text_seq", 3))
    if config.get("guidance_embeds") is True:
        built.input("guidance", (1,))
    hidden = built.value("image-embed")
    built.stage(
        "image-embed",
        "Image token embedding",
        "PatchEmbed",
        (sample,),
        (hidden,),
        "x_embedder",
        domain="nneditor.diffusion",
    )
    conditioning = built.value("conditioning")
    built.stage(
        "conditioning",
        "Time and pooled-text conditioning",
        "Conditioning",
        (timestep, pooled_input),
        (conditioning,),
        "time_text_embed",
        domain="nneditor.diffusion",
    )
    text_hidden = built.value("context")
    built.stage(
        "context",
        "Text context",
        "Context",
        (context,),
        (text_hidden,),
        "context_embedder",
        domain="nneditor.diffusion",
    )
    hidden = built.blocks(
        "double",
        _positive(config, "num_layers"),
        "DiTBlock",
        "transformer_blocks.{index}",
        hidden,
        domain="nneditor.diffusion",
        also=(conditioning, text_hidden),
    )
    hidden = built.blocks(
        "single",
        _positive(config, "num_single_layers"),
        "DiTBlock",
        "single_transformer_blocks.{index}",
        hidden,
        domain="nneditor.diffusion",
        also=(conditioning,),
    )
    return built.graph(_prediction(built, hidden, "nneditor.diffusion"))


def _cogvideox(index: HfIndex) -> Graph:
    config = index.config
    built = _Builder(index, "CogVideoX")
    sample = built.input(
        "sample",
        (
            "batch",
            _positive(config, "sample_frames"),
            _positive(config, "in_channels"),
            _positive(config, "sample_height"),
            _positive(config, "sample_width"),
        ),
    )
    timestep = built.input("timestep", (1,), "int64")
    context = built.input(
        "encoder-hidden-states",
        ("batch", "text", _positive(config, "text_embed_dim")),
    )
    hidden = built.value("patch-embed")
    built.stage(
        "patch-embed",
        "Spatiotemporal patch embedding",
        "PatchEmbed",
        (sample, context),
        (hidden,),
        "patch_embed",
        domain="nneditor.diffusion",
    )
    conditioning = built.value("conditioning")
    built.stage(
        "conditioning",
        "Time conditioning",
        "Conditioning",
        (timestep,),
        (conditioning,),
        "time_embedding",
        domain="nneditor.diffusion",
    )
    hidden = built.blocks(
        "block",
        _positive(config, "num_layers"),
        "DiTBlock",
        "transformer_blocks.{index}",
        hidden,
        domain="nneditor.diffusion",
        also=(conditioning,),
    )
    return built.graph(_prediction(built, hidden, "nneditor.diffusion"))


def _wan(index: HfIndex) -> Graph:
    config = index.config
    built = _Builder(index, "Wan")
    sample = built.input(
        "sample",
        (
            "batch",
            _positive(config, "in_channels"),
            "frames",
            "height",
            "width",
        ),
    )
    timestep = built.input("timestep", (1,), "int64")
    context = built.input(
        "encoder-hidden-states",
        ("batch", "text", _positive(config, "text_dim")),
    )
    hidden = built.value("patch-embed")
    built.stage(
        "patch-embed",
        "Spatiotemporal patch embedding",
        "PatchEmbed",
        (sample,),
        (hidden,),
        "patch_embedding",
        domain="nneditor.diffusion",
    )
    conditioning = built.value("conditioning")
    built.stage(
        "conditioning",
        "Time and text conditioning",
        "Conditioning",
        (timestep, context),
        (conditioning,),
        "condition_embedder",
        domain="nneditor.diffusion",
    )
    hidden = built.blocks(
        "block",
        _positive(config, "num_layers"),
        "DiTBlock",
        "blocks.{index}",
        hidden,
        domain="nneditor.diffusion",
        also=(conditioning,),
    )
    return built.graph(_prediction(built, hidden, "nneditor.diffusion"))


def _qwen2_vl(index: HfIndex) -> Graph:
    vision = _mapping(index.config, "vision_config")
    text = _mapping(index.config, "text_config")
    patch = _positive(vision, "patch_size")
    packed = (
        _positive(vision, "in_channels")
        * _positive(vision, "temporal_patch_size")
        * patch
        * patch
    )
    built = _Builder(index, "Qwen2-VL")
    pixels = built.input("pixel-values", ("patches", packed))
    grid = built.input("image-grid", ("images", 3), "int64")
    types = built.input("token-types", ("batch", "seq"), "int64")
    tokens = built.input("tokens", ("batch", "seq"), "int64")
    visual = built.value("vision-embed")
    built.stage(
        "vision-embed",
        "Vision patch embedding",
        "VisionEmbed",
        (pixels, grid),
        (visual,),
        "model.visual.patch_embed",
        domain="nneditor.vision",
    )
    visual = built.blocks(
        "vision-block",
        _positive(vision, "depth"),
        "VisionBlock",
        "model.visual.blocks.{index}",
        visual,
        domain="nneditor.vision",
        also=(grid,),
    )
    projected = built.value("projector")
    built.stage(
        "projector",
        "Vision merger",
        "Projector",
        (visual,),
        (projected,),
        "model.visual.merger",
        domain="nneditor.vision",
    )
    hidden = built.value("text-embed")
    built.stage(
        "text-embed",
        "Token embedding",
        "Embedding",
        (tokens,),
        (hidden,),
        "model.language_model.embed_tokens",
        domain="nneditor.vision",
    )
    hidden = _image_text_merge(built, hidden, projected, also=(grid, types))
    hidden = built.blocks(
        "text-block",
        _positive(text, "num_hidden_layers"),
        "DecoderBlock",
        "model.language_model.layers.{index}",
        hidden,
        domain="nneditor.vision",
    )
    logits = built.finish(
        hidden,
        ("model.language_model.norm", "lm_head"),
        _positive(text, "vocab_size"),
        "nneditor.vision",
    )
    return built.graph(logits)


def _gemma3(index: HfIndex) -> Graph:
    if "vision_config" not in index.config:
        raise HuggingFaceError("Gemma 3 text checkpoints use the decoder graph.")
    vision = _mapping(index.config, "vision_config")
    text = _mapping(index.config, "text_config")
    image = _positive(vision, "image_size")
    built = _Builder(index, "Gemma 3")
    pixels = built.input(
        "pixel-values", (1, _positive(vision, "num_channels"), image, image)
    )
    tokens = built.input("tokens", ("batch", "seq"), "int64")
    visual = built.value("vision-embed")
    built.stage(
        "vision-embed",
        "Vision patch embedding",
        "VisionEmbed",
        (pixels,),
        (visual,),
        "model.vision_tower.embeddings",
        domain="nneditor.vision",
    )
    visual = built.blocks(
        "vision-block",
        _positive(vision, "num_hidden_layers"),
        "VisionBlock",
        "model.vision_tower.encoder.layers.{index}",
        visual,
        domain="nneditor.vision",
    )
    normalized = built.value("vision-norm")
    built.stage(
        "vision-norm",
        "Vision normalization",
        "LayerNorm",
        (visual,),
        (normalized,),
        "model.vision_tower.post_layernorm",
        domain="nneditor.vision",
    )
    projected = built.value("projector")
    built.stage(
        "projector",
        "Vision projector",
        "Projector",
        (normalized,),
        (projected,),
        "model.multi_modal_projector",
        domain="nneditor.vision",
    )
    hidden = built.value("text-embed")
    built.stage(
        "text-embed",
        "Token embedding",
        "Embedding",
        (tokens,),
        (hidden,),
        "model.language_model.embed_tokens",
        domain="nneditor.vision",
    )
    hidden = _image_text_merge(built, hidden, projected)
    hidden = built.blocks(
        "text-block",
        _positive(text, "num_hidden_layers"),
        "DecoderBlock",
        "model.language_model.layers.{index}",
        hidden,
        domain="nneditor.vision",
    )
    logits = built.finish(
        hidden,
        ("model.language_model.norm", "lm_head"),
        _positive(text, "vocab_size"),
        "nneditor.vision",
    )
    return built.graph(logits)


def _bitnet(index: HfIndex) -> Graph:
    config = index.config
    built = _Builder(index, "BitNet")
    tokens = built.input("tokens", ("batch", "seq"), "int64")
    hidden = built.value("embed")
    built.stage(
        "embed",
        "Token embedding",
        "Embedding",
        (tokens,),
        (hidden,),
        "model.embed_tokens",
        domain="nneditor.llm",
    )
    hidden = built.blocks(
        "block",
        _positive(config, "num_hidden_layers"),
        "DecoderBlock",
        "model.layers.{index}",
        hidden,
        domain="nneditor.llm",
    )
    logits = built.finish(
        hidden,
        ("model.norm", "lm_head"),
        _positive(config, "vocab_size"),
        "nneditor.llm",
    )
    return built.graph(logits)


_BUILDERS = {
    "llava": _llava,
    "whisper": _whisper,
    "t5": _t5,
    "rwkv": _rwkv,
    "sd3": _sd3,
    "flux": _flux,
    "cogvideox": _cogvideox,
    "wan": _wan,
    "qwen2_vl": _qwen2_vl,
    "gemma3": _gemma3,
    "bitnet": _bitnet,
}


def modality_graph(index: HfIndex) -> Graph | None:
    """A family graph, or None when this snapshot is not one of these families."""
    kind = modality_kind(index.config)
    if kind is None:
        return None
    if kind == "gemma3" and "vision_config" not in index.config:
        return None
    return _BUILDERS[kind](index)
