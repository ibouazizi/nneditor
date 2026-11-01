"""Mamba-family forward tracing with CPU PyTorch kernels and fresh state."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np

from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError


def trace(context: TraceContext) -> TraceOutcome:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise TracePluginError(
            "Install nneditor[tracer-ssm] to run the Mamba tracer."
        ) from error
    family = context.metadata.get("model_type")
    class_name = {"mamba": "MambaForCausalLM", "mamba2": "Mamba2ForCausalLM"}.get(
        str(family)
    )
    if class_name is None:
        raise TracePluginError("This extension supports Mamba and Mamba2 checkpoints.")
    tokens = context.inputs.get("tokens")
    if tokens is None or tokens.ndim != 2 or tokens.dtype.kind not in "iu":
        raise TracePluginError("Mamba needs integer tokens shaped [batch, sequence].")
    model_class = getattr(transformers, class_name)
    model, information = model_class.from_pretrained(
        context.document.source.path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if information.get("missing_keys") or information.get("unexpected_keys"):
        raise TracePluginError(
            "Checkpoint weights do not match the Mamba configuration; "
            "refusing randomly initialized or unused weights."
        )
    model.eval()
    model.config.use_cache = False
    if np.any(tokens < 0) or np.any(tokens >= model.config.vocab_size):
        raise TracePluginError("A token ID is outside the Mamba vocabulary.")
    modules = dict(model.named_modules())
    bindings = {
        "embed": "backbone.embeddings",
        "output-norm": "backbone.norm_f",
        "lm-head": "lm_head",
    }
    bindings.update(
        {
            f"block-{i:03d}-ssm": f"backbone.layers.{i}"
            for i in range(model.config.num_hidden_layers)
        }
    )
    names = {value.name for value in context.document.main_graph.values}
    handles: list[Any] = []

    def hook(name: str) -> Any:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            array = cast(np.ndarray[Any, Any], value.detach().float().cpu().numpy())
            context.capture(name, array)

        return capture

    try:
        for name, module in bindings.items():
            if name not in names or module not in modules:
                raise TracePluginError(
                    f"The graph or runtime has no Mamba stage {name!r}."
                )
            handles.append(modules[module].register_forward_hook(hook(name)))
        context.capture_inputs()
        with torch.inference_mode():
            model(
                input_ids=torch.from_numpy(np.array(tokens, dtype=np.int64, copy=True)),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return TraceOutcome(
        provider="pytorch-cpu",
        notes=(
            f"{class_name}: CPU forward pass with fresh recurrent state; "
            "no generated continuation.",
        ),
    )
