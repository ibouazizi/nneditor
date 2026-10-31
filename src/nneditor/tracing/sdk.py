"""Public version 1 API for tracing extensions. Import this module in plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from nneditor.ir.core import Document, JsonValue
from nneditor.storage.store import TensorStore
from nneditor.tracing.contracts import TraceDevice, TraceLimits

TRACE_API_VERSION = 1
__all__ = ["TRACE_API_VERSION", "TraceContext", "TraceOutcome", "TracePluginError"]


class TracePluginError(ValueError):
    """An extension's configuration, compatibility check, or execution failed."""


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    """Optional execution details returned by ``trace(context)``."""

    notes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    provider: str = ""


class TraceContext:
    """One approved worker execution. Capture arrays as soon as they are ready.

    ``capture`` accepts a graph value ID or its unique display name. NNEditor
    owns capture selection, byte limits, disk writes, cancellation, and cleanup.
    Plugins should stream weights through ``tensors`` and avoid retaining
    activations after capture. The worker terminates on cancellation or timeout.
    """

    def __init__(
        self,
        document: Document,
        inputs: Mapping[str, np.ndarray[Any, Any]],
        tensors: TensorStore,
        capture: Callable[[str, np.ndarray[Any, Any]], None],
        *,
        limits: TraceLimits,
        device: TraceDevice = TraceDevice.CPU,
    ) -> None:
        self.document = document
        self.inputs = MappingProxyType(dict(inputs))
        self.tensors = tensors
        self.limits = limits
        self.device = device
        self._capture = capture
        values = document.main_graph.values
        self._ids = {v.id for v in values}
        self._names: dict[str, list[str]] = {}
        for value in values:
            if value.name:
                self._names.setdefault(value.name, []).append(value.id)

    @property
    def metadata(self) -> Mapping[str, JsonValue]:
        """GGUF metadata or Hugging Face config, without materializing weights."""
        extensions = dict(self.document.extensions)
        raw = extensions.get("x-gguf.metadata", extensions.get("x-huggingface.config"))
        return MappingProxyType(raw if isinstance(raw, dict) else {})

    def capture(self, name_or_id: str, array: np.ndarray[Any, Any]) -> None:
        """Publish one graph activation; reject typos and ambiguous names."""
        value_id = name_or_id
        if value_id not in self._ids:
            matches = self._names.get(name_or_id, [])
            if len(matches) != 1:
                raise TracePluginError(
                    f"Capture {name_or_id!r} must identify one graph value; "
                    "use its value ID when names are ambiguous."
                )
            value_id = matches[0]
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
            raise TracePluginError("Captured activations must be real numeric arrays")
        self._capture(value_id, array)

    def capture_inputs(self) -> None:
        """Capture every declared input using its feed name or value ID."""
        graph = self.document.main_graph
        for value_id in graph.inputs:
            value = graph.value(value_id)
            name = value.name or value.id
            array = self.inputs.get(name, self.inputs.get(value_id))
            if array is None:
                raise TracePluginError(f"Missing input {name!r}")
            self.capture(value_id, array)

    def load_weight(self, tensor_id: str) -> np.ndarray[Any, Any]:
        """Decode a weight by ID or unique name; use read slices for large weights."""
        from nneditor.tracing.llm_runtime import _tensor_name, load_weight

        if tensor_id not in self.document.tensors:
            matches = [
                t.id
                for t in self.document.tensors.values()
                if _tensor_name(t.id) == tensor_id
            ]
            if len(matches) != 1:
                raise TracePluginError(f"Weight {tensor_id!r} must identify one tensor")
            tensor_id = matches[0]

        model = dict(self.document.extensions).get("x-gguf.model")
        endian = (
            str(model.get("endian", "little")) if isinstance(model, dict) else "little"
        )
        return load_weight(self.tensors, tensor_id, endian)

    def trace_numpy_decoder(self) -> TraceOutcome:
        """Reuse NNEditor's built-in LLM decoder and its streaming capture path."""
        from nneditor.tracing.llm_runtime import trace_document

        _, notes, diagnostics = trace_document(
            self.document, dict(self.inputs), self.tensors, capture=self.capture
        )
        return TraceOutcome(notes=notes, diagnostics=diagnostics, provider="numpy")
