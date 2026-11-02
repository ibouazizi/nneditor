"""Run a CLIP vision GGUF or a Whisper GGUF on CPU."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from nneditor.adapters.gguf.media import (
    clip_state_key,
    full_whisper,
    media_config,
    media_kind,
    whisper_state_key,
)
from nneditor.adapters.media_graphs import positive
from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError
from nneditor.tracing.stage_hooks import capture_stages

__all__ = ["clip_state_key", "trace_gguf_media", "whisper_state_key"]


def _names(context: TraceContext) -> set[str]:
    from nneditor.tracing.llm_runtime import _tensor_name

    return {_tensor_name(tensor.id) for tensor in context.document.tensors.values()}


def _load_arrays(context: TraceContext, translate: Any) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    from nneditor.tracing.llm_runtime import _tensor_name

    mapped: dict[str, Any] = {}
    for tensor in context.document.tensors.values():
        key = translate(_tensor_name(tensor.id))
        if key is None:
            continue
        array = context.load_weight(tensor.id)
        mapped[key] = torch.from_numpy(np.array(array, copy=True))
    return mapped


def _vision_model(context: TraceContext, mapped: dict[str, Any]) -> Any:
    transformers = importlib.import_module("transformers")
    options = media_config(context.metadata, "vision", full=False)
    config = transformers.CLIPVisionConfig(
        **{key: positive(options, key) for key in options}
    )
    model = transformers.CLIPVisionModel(config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise TracePluginError(
            "Vision GGUF weights do not match the CLIP vision module layout."
        )
    model.eval()
    return model


def _whisper_model(context: TraceContext, mapped: dict[str, Any], *, full: bool) -> Any:
    transformers = importlib.import_module("transformers")
    options = media_config(context.metadata, "whisper", full=full)
    config = transformers.WhisperConfig(
        **{key: positive(options, key) for key in options},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = transformers.WhisperForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if (
        full
        and "proj_out.weight" not in mapped
        and "model.decoder.embed_tokens.weight" in mapped
    ):
        model.proj_out.weight = torch_param(mapped["model.decoder.embed_tokens.weight"])
        missing = [key for key in missing if key != "proj_out.weight"]
    rejected = [
        key
        for key in missing
        if full or not (key.startswith("model.decoder.") or key.startswith("proj_out"))
    ]
    if rejected or unexpected:
        raise TracePluginError(
            "Whisper GGUF weights do not match the Whisper module layout."
        )
    model.eval()
    model.config.use_cache = False
    return model


def torch_param(array: Any) -> Any:
    torch = importlib.import_module("torch")
    return torch.nn.Parameter(array)


def _run(model: Any, kind: str, context: TraceContext, *, full: bool) -> Any:
    torch = importlib.import_module("torch")

    def tensor(name: str) -> Any:
        array = context.inputs.get(name)
        if array is None:
            raise TracePluginError(f"Missing input {name!r}.")
        return torch.from_numpy(np.array(array, copy=True))

    with torch.inference_mode():
        if kind == "vision":
            return model(pixel_values=tensor("pixel-values")).last_hidden_state
        features = tensor("input-features")
        if not full:
            return model.model.encoder(features).last_hidden_state
        return model(
            input_features=features,
            decoder_input_ids=tensor("decoder-tokens"),
            use_cache=False,
            return_dict=True,
        ).logits


def trace_gguf_media(context: TraceContext) -> TraceOutcome:
    """Trace one vision tower or one Whisper checkpoint stored as GGUF."""
    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-modalities] to run this tracer."
        ) from error
    names = _names(context)
    kind = media_kind(names)
    full = kind == "whisper" and full_whisper(names)
    translate = clip_state_key if kind == "vision" else whisper_state_key
    mapped = _load_arrays(context, translate)
    model = (
        _vision_model(context, mapped)
        if kind == "vision"
        else _whisper_model(context, mapped, full=full)
    )
    del mapped
    with capture_stages(model, context) as capture_result:
        context.capture_inputs()
        capture_result(_run(model, kind, context, full=full))
    label = "CLIP vision tower" if kind == "vision" else "Whisper"
    return TraceOutcome(
        provider="pytorch-cpu",
        notes=(f"{label} GGUF: one CPU forward pass.",),
    )
