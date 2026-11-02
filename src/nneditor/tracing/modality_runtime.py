"""Run a modality graph by hooking the library modules the reader named."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError
from nneditor.tracing.stage_hooks import capture_stages

_CLASSES: dict[str, tuple[str, str]] = {
    "llava": ("transformers", "LlavaForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("transformers", "LlavaForConditionalGeneration"),
    "qwen2_vl": ("transformers", "Qwen2VLForConditionalGeneration"),
    "Qwen2VLForConditionalGeneration": (
        "transformers",
        "Qwen2VLForConditionalGeneration",
    ),
    "gemma3": ("transformers", "Gemma3ForConditionalGeneration"),
    "Gemma3ForConditionalGeneration": (
        "transformers",
        "Gemma3ForConditionalGeneration",
    ),
    "bitnet": ("transformers", "BitNetForCausalLM"),
    "BitNetForCausalLM": ("transformers", "BitNetForCausalLM"),
    "whisper": ("transformers", "WhisperForConditionalGeneration"),
    "WhisperForConditionalGeneration": (
        "transformers",
        "WhisperForConditionalGeneration",
    ),
    "t5": ("transformers", "T5ForConditionalGeneration"),
    "T5ForConditionalGeneration": ("transformers", "T5ForConditionalGeneration"),
    "rwkv": ("transformers", "RwkvForCausalLM"),
    "RwkvForCausalLM": ("transformers", "RwkvForCausalLM"),
    "SD3Transformer2DModel": ("diffusers", "SD3Transformer2DModel"),
    "FluxTransformer2DModel": ("diffusers", "FluxTransformer2DModel"),
    "CogVideoXTransformer3DModel": ("diffusers", "CogVideoXTransformer3DModel"),
    "WanTransformer3DModel": ("diffusers", "WanTransformer3DModel"),
}
_ARGUMENTS = {
    "tokens": "input_ids",
    "pixel-values": "pixel_values",
    "input-features": "input_features",
    "decoder-tokens": "decoder_input_ids",
    "source-tokens": "input_ids",
    "sample": "hidden_states",
    "encoder-hidden-states": "encoder_hidden_states",
    "pooled-projections": "pooled_projections",
    "img-ids": "img_ids",
    "txt-ids": "txt_ids",
    "guidance": "guidance",
    "timestep": "timestep",
    "image-grid": "image_grid_thw",
    "token-types": "mm_token_type_ids",
}
_CACHE = frozenset(
    {
        "llava",
        "qwen2_vl",
        "gemma3",
        "bitnet",
        "whisper",
        "t5",
        "rwkv",
        "LlavaForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Gemma3ForConditionalGeneration",
        "BitNetForCausalLM",
        "WhisperForConditionalGeneration",
        "T5ForConditionalGeneration",
        "RwkvForCausalLM",
    }
)


def _family(context: TraceContext) -> tuple[str, str]:
    metadata = context.metadata
    for key in ("model_type", "_class_name"):
        name = metadata.get(key)
        if isinstance(name, str) and name in _CLASSES:
            return name, _CLASSES[name][0]
    raise TracePluginError(
        "This tracer supports LLaVA, Qwen2-VL, Gemma 3, BitNet, Whisper, "
        "T5, RWKV, SD3, Flux, CogVideoX, and Wan."
    )


def _load(context: TraceContext, library: str, class_name: str) -> Any:
    torch = importlib.import_module("torch")
    module = importlib.import_module(library)
    model_class = getattr(module, class_name)
    options: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
        "output_loading_info": True,
    }
    if library == "transformers":
        options["dtype"] = torch.float32
    else:
        options["torch_dtype"] = torch.float32
    model, information = model_class.from_pretrained(
        context.document.source.path, **options
    )
    if any(
        information.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
    ):
        raise TracePluginError(
            f"{class_name} weights do not match its configuration; "
            "refusing missing, unused, or randomly initialized weights."
        )
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


def _kwargs(context: TraceContext, family: str) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    graph = context.document.main_graph
    required = {graph.value(key).name for key in graph.inputs}
    missing = sorted(name for name in required if name not in context.inputs)
    if missing:
        raise TracePluginError("Missing inputs: " + ", ".join(missing))
    arguments = {
        _ARGUMENTS[name]: torch.from_numpy(np.array(array, copy=True))
        for name, array in context.inputs.items()
        if name in _ARGUMENTS
    }
    unknown = sorted(set(context.inputs) - set(_ARGUMENTS))
    if unknown:
        raise TracePluginError("Unexpected inputs: " + ", ".join(unknown))
    if family in _CACHE:
        arguments["use_cache"] = False
    return arguments


def trace_modality(context: TraceContext) -> TraceOutcome:
    """Execute one approved modality checkpoint and capture every graph port."""
    try:
        importlib.import_module("torch")
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-modalities] to run this tracer."
        ) from error
    family, library = _family(context)
    class_name = _CLASSES[family][1]
    try:
        importlib.import_module(library)
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-modalities] to run this tracer."
        ) from error
    model = _load(context, library, class_name)
    with capture_stages(model, context) as capture_result:
        context.capture_inputs()
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            result = model(**_kwargs(context, family), return_dict=True)
        capture_result(result)
    return TraceOutcome(
        provider="pytorch-cpu",
        notes=(
            f"{class_name}: one CPU forward pass captured at the graph stages.",
            "The result is that pass, not a generated continuation or a decoded video.",
        ),
    )
