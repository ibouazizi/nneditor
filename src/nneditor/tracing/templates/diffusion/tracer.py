"""Stable Diffusion denoiser tracing, including actual UNet skip tensors."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np

from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError


def _capture(context: TraceContext, key: str, tensor: Any) -> None:
    context.capture(
        key, cast(np.ndarray[Any, Any], tensor.detach().float().cpu().numpy())
    )


def _hook(context: TraceContext, outputs: tuple[str, ...], down: bool) -> Any:
    def save(_module: Any, _arguments: Any, output: Any) -> None:
        if down:
            hidden, skips = output
            if len(skips) != len(outputs):
                raise TracePluginError("Runtime UNet skips differ from the graph.")
            for key, tensor in zip(outputs, skips, strict=True):
                _capture(context, key, tensor)
            if hidden.shape != skips[-1].shape:
                raise TracePluginError(
                    "UNet hidden state differs from its last residual."
                )
        else:
            _capture(context, outputs[0], output)

    return save


def trace(context: TraceContext) -> TraceOutcome:
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-diffusion] to run this tracer."
        ) from error
    class_name = context.metadata.get("_class_name")
    if class_name not in {"UNet2DConditionModel", "UNet2DModel"}:
        raise TracePluginError(
            "Open a Diffusers UNet component folder, "
            "such as a Stable Diffusion pipeline's unet folder."
        )
    model_class = getattr(diffusers, str(class_name))
    model, information = model_class.from_pretrained(
        context.document.source.path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        information.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
    ):
        raise TracePluginError(
            "UNet weights do not match its configuration; "
            "refusing missing, unused, or randomly initialized weights."
        )
    model.eval()
    modules = dict(model.named_modules())
    handles: list[Any] = []
    graph = context.document.main_graph

    def capture(key: str, tensor: Any) -> None:
        context.capture(
            key, cast(np.ndarray[Any, Any], tensor.detach().float().cpu().numpy())
        )

    def conditioning(_module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        tensor = kwargs.get("temb")
        if tensor is None and len(args) > 1:
            tensor = args[1]
        if tensor is None:
            raise TracePluginError("UNet did not supply a time-conditioning tensor.")
        capture("conditioning", tensor)

    feeds = {
        name: torch.from_numpy(np.array(array, copy=True))
        for name, array in context.inputs.items()
    }
    required = {graph.value(key).name for key in graph.inputs}
    if required - feeds.keys():
        raise TracePluginError(
            "Missing UNet inputs: " + ", ".join(sorted(required - feeds.keys()))
        )
    kwargs = {
        name: feeds[name]
        for name in ("sample", "timestep", "encoder_hidden_states", "class_labels")
        if name in feeds
    }
    if "text_embeds" in feeds:
        kwargs["added_cond_kwargs"] = {
            "text_embeds": feeds["text_embeds"],
            "time_ids": feeds["time_ids"],
        }
    try:
        for node in graph.nodes:
            module = next(
                (
                    attribute.value
                    for attribute in node.attributes
                    if attribute.name == "module"
                ),
                None,
            )
            if module == "conditioning":
                handles.append(
                    model.down_blocks[0].register_forward_pre_hook(
                        conditioning, with_kwargs=True
                    )
                )
                continue
            if module == "conv_out":
                # The final stage includes any model-level output scaling.
                continue
            if not isinstance(module, str) or module not in modules:
                raise TracePluginError(f"UNet has no runtime stage {module!r}.")
            handles.append(
                modules[module].register_forward_hook(
                    _hook(context, node.outputs, node.op_type == "UNetDown")
                )
            )
        context.capture_inputs()
        with torch.inference_mode():
            prediction = model(**kwargs, return_dict=True)
            capture("noise-prediction", prediction.sample)
    finally:
        for handle in handles:
            handle.remove()
    return TraceOutcome(
        provider="pytorch-cpu",
        notes=(
            f"{class_name}: one CPU denoising pass, including skip connections.",
            "The result is a latent/noise prediction, "
            "not a decoded or iteratively sampled image.",
        ),
    )
