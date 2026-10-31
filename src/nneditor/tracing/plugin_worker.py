"""Versioned tracing extension host, launched only in a capped worker."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

from nneditor.adapters.registry import default_artifact_adapter_registry
from nneditor.storage.store import TensorStore
from nneditor.tracing.contracts import (
    CAPTURE_POLICIES,
    TraceBackend,
    TraceDevice,
    TraceLimits,
)
from nneditor.tracing.llm_worker import CaptureWriter, _limit_process, _load_feeds
from nneditor.tracing.plugin_manifest import TracePlugin, load_manifest
from nneditor.tracing.plugins import TraceRegistry
from nneditor.tracing.sdk import TraceContext, TraceOutcome, TracePluginError


def _load_function(
    plugin: TracePlugin,
) -> Callable[[TraceContext], TraceOutcome | None]:
    if os.environ.get("NNEDITOR_TRACE_WORKER") != "1":
        raise TracePluginError("Extension code can only load inside the trace worker")
    if plugin.manifest is None:
        raise TracePluginError("Built-in tracers do not load external Python files")
    root = plugin.manifest.parent
    filename, _, function = plugin.entry_point.partition(":")
    source = root / filename
    namespace = "_nneditor_tracer_" + plugin.fingerprint
    package = ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    # Sibling imports work in a private package; absolute helper imports are
    # also convenient for a folder copied from an existing runtime project.
    sys.path.insert(0, str(root))
    parts = Path(filename).parts[:-1]
    parent = root
    for part in parts:
        namespace += "." + part
        parent /= part
        package = ModuleType(namespace)
        package.__path__ = [str(parent)]
        sys.modules[namespace] = package
    name = namespace + "." + source.stem
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise TracePluginError(f"Cannot load tracer entry point {plugin.entry_point}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    callback = getattr(module, function, None)
    if not callable(callback):
        raise TracePluginError(
            f"Tracer entry point {plugin.entry_point} is not callable"
        )
    return cast(Callable[[TraceContext], TraceOutcome | None], callback)


def _execute(plugin: TracePlugin, context: TraceContext) -> TraceOutcome:
    if plugin.id == "builtin.numpy-llm":
        return context.trace_numpy_decoder()
    result = _load_function(plugin)(context)
    if result is None:
        return TraceOutcome()
    if not isinstance(result, TraceOutcome):
        raise TracePluginError("trace(context) must return TraceOutcome or None")
    return result


def run(request_path: Path) -> None:
    if os.environ.get("NNEDITOR_TRACE_WORKER") != "1":
        raise TracePluginError("Trace worker must be launched by NNEditor")
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("targets"), list):
        raise TracePluginError("Trace request is malformed")
    _limit_process(raw)
    policy = raw.get("capture_policy", "greedy")
    if policy not in CAPTURE_POLICIES:
        raise TracePluginError("Unknown capture policy")
    document = default_artifact_adapter_registry().open(Path(raw["artifact"]))
    if document.source.content_hash != raw.get("artifact_hash"):
        raise TracePluginError(
            "The model changed after approval. Reopen it before tracing."
        )
    if document.artifact_kind.value != raw.get("kind") or raw.get("revision_id"):
        raise TracePluginError(
            "Extension tracing requires the approved unedited source artifact"
        )
    manifest = raw.get("tracer_manifest")
    plugin = (
        load_manifest(Path(manifest))
        if isinstance(manifest, str)
        else TraceRegistry().resolve(document, raw.get("tracer"))
    )
    if plugin.identity != raw.get("tracer"):
        raise TracePluginError(
            "Tracer files changed after approval. Refresh tracers and approve again."
        )
    if reason := plugin.incompatibility(document):
        raise TracePluginError(reason)
    if plugin.id == "builtin.onnx":
        raise TracePluginError("The ONNX built-in uses the exported model worker")
    backend = TraceBackend(raw.get("backend", "auto"))
    if backend is TraceBackend.ONNX_RUNTIME:
        raise TracePluginError("A tracer extension chooses its own execution runtime")
    device = TraceDevice(raw.get("device", "auto"))
    if device is TraceDevice.AUTO:
        device = plugin.devices[0]
    if device not in plugin.devices:
        raise TracePluginError(f"{plugin.name} does not support device {device.value}")
    writer = CaptureWriter(
        Path(raw["output"]),
        raw["targets"],
        capture_bytes=int(raw["capture_bytes"]),
        chunk_bytes=int(raw["chunk_bytes"]),
        capture_policy=policy,
    )
    feeds = _load_feeds(Path(raw["inputs"]))
    with TensorStore(document) as tensors:
        context = TraceContext(
            document,
            feeds,
            tensors,
            writer.capture,
            limits=TraceLimits.from_json(raw),
            device=device,
        )
        outcome = _execute(plugin, context)
    response = {
        "runtime": "numpy-llm" if plugin.builtin else plugin.id,
        "execution_device": device.value,
        "execution_provider": outcome.provider or plugin.name,
        "records": writer.records(),
        "diagnostics": list(outcome.diagnostics),
        "notes": [
            *outcome.notes,
            f"Tracer: {plugin.name} {plugin.version}",
        ],
    }
    Path(raw["response"]).write_text(
        json.dumps(response, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nneditor.tracing.plugin_worker REQUEST")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
