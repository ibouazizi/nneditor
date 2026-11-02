"""Capture explicit stage outputs and functional boundaries at module inputs."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np

from nneditor.tracing.sdk import TraceContext, TracePluginError


def _activation(output: Any) -> Any:
    torch = importlib.import_module("torch")
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        tensors = [item for item in output if torch.is_tensor(item)]
        if tensors:
            return tensors[-1]
    for name in ("sample", "logits", "last_hidden_state"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TracePluginError("A hooked stage did not return a numeric tensor.")


def _capture(context: TraceContext, key: str, output: Any) -> None:
    tensor = _activation(output)
    context.capture(
        key, cast(np.ndarray[Any, Any], tensor.detach().float().cpu().numpy())
    )


@contextmanager
def capture_stages(
    model: Any, context: TraceContext
) -> Iterator[Callable[[Any], None]]:
    """Fail on unresolved or unexecuted stages and always remove installed hooks."""
    modules = dict(model.named_modules())
    handles: list[Any] = []
    expected: set[str] = set()
    seen: set[str] = set()
    returns: list[str] = []

    def save(key: str, output: Any) -> None:
        _capture(context, key, output)
        seen.add(key)

    def result(output: Any) -> None:
        for key in returns:
            save(key, output)

    def prehook(key: str, argument: str) -> Callable[..., None]:
        def capture(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            value = kwargs.get(argument, args[0] if args else None)
            save(key, value)

        return capture

    try:
        for node in context.document.main_graph.nodes:
            attrs = {attribute.name: attribute.value for attribute in node.attributes}
            module = attrs.get("module")
            key = node.outputs[0]
            expected.add(key)
            if module == "output":
                returns.append(key)
                continue
            if not isinstance(module, str) or module not in modules:
                raise TracePluginError(
                    f"{type(model).__name__} has no module {module!r}."
                )
            argument = attrs.get("capture_input")
            if isinstance(argument, str):
                handle = modules[module].register_forward_pre_hook(
                    prehook(key, argument), with_kwargs=True
                )
            else:
                handle = modules[module].register_forward_hook(
                    lambda _module, _inputs, output, key=key: save(key, output)
                )
            handles.append(handle)
        yield result
        missing = expected - seen
        if missing:
            names = sorted(
                context.document.main_graph.value(key).name for key in missing
            )
            raise TracePluginError(
                "Model did not execute graph stages: " + ", ".join(names)
            )
    finally:
        for handle in handles:
            handle.remove()
