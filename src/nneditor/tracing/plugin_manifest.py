"""Declarative tracing manifests. Discovery never imports extension code."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from nneditor import __version__
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document
from nneditor.tracing.contracts import TraceDevice
from nneditor.tracing.sdk import TRACE_API_VERSION, TracePluginError

MANIFEST_NAME = "nneditor-tracer.json"
_MAX_BYTES = 32 * 1024 * 1024
_ID = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")


def _strings(
    raw: dict[str, object], key: str, *, required: bool = False
) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(v, str) or not v for v in value
    ):
        raise TracePluginError(f"{key} must be an array of nonempty strings")
    if required and not value:
        raise TracePluginError(f"{key} must not be empty")
    return tuple(str(v) for v in value)


def plugin_files(root: Path) -> tuple[Path, ...]:
    """Bound and validate a plugin tree, including local helper modules/data."""
    total = 0
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".venv"}]
        for name in (*dirs, *names):
            path = Path(directory) / name
            if path.is_symlink():
                raise TracePluginError(
                    f"Plugin files must not be symlinks: {path.relative_to(root)}"
                )
            if not path.is_file():
                continue
            total += path.stat().st_size
            files.append(path)
            if total > _MAX_BYTES or len(files) > 1024:
                raise TracePluginError(
                    "Plugin exceeds 32 MiB or 1024 files; "
                    "install dependencies separately"
                )
    return tuple(sorted(files))


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    for path in plugin_files(root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                total += len(block)
                if total > _MAX_BYTES:
                    raise TracePluginError(
                        "Plugin files grew beyond 32 MiB during discovery"
                    )
                digest.update(block)
    return digest.hexdigest()


def architecture(document: Document) -> str:
    for key, field in (
        ("x-gguf.model", "architecture"),
        ("x-huggingface.config", "model_type"),
    ):
        value = dict(document.extensions).get(key)
        if isinstance(value, dict):
            name = value.get(field, value.get("_class_name"))
            if isinstance(name, str):
                return name
    return ""


@dataclass(frozen=True, slots=True)
class TracePlugin:
    id: str
    name: str
    version: str
    artifact_kinds: tuple[ArtifactKind, ...]
    architectures: tuple[str, ...] = ()
    element_types: tuple[str, ...] = ()
    devices: tuple[TraceDevice, ...] = (TraceDevice.CPU,)
    description: str = ""
    manifest: Path | None = None
    entry_point: str = ""
    fingerprint: str = ""
    operators: tuple[str, ...] = ()
    python_modules: tuple[str, ...] = ()
    install_extra: str = ""
    streaming_weights: bool = False

    @property
    def builtin(self) -> bool:
        return self.manifest is None

    @property
    def identity(self) -> str:
        return f"{self.id}@{self.version}" + (
            f"+{self.fingerprint}/nneditor-{__version__}" if self.fingerprint else ""
        )

    def incompatibility(self, document: Document) -> str | None:
        if document.artifact_kind not in self.artifact_kinds:
            return f"{self.name} does not support {document.artifact_kind.value}"
        if self.architectures and architecture(document) not in self.architectures:
            return (
                f"{self.name} supports architectures: {', '.join(self.architectures)}"
            )
        extensions = dict(document.extensions)
        graph_error = extensions.get(
            "x-huggingface.trace-graph-error"
        ) or extensions.get("x-gguf.trace-graph-error")
        if isinstance(graph_error, str):
            return graph_error
        if self.element_types:
            unsupported = {t.element_type for t in document.tensors.values()} - set(
                self.element_types
            )
            if unsupported:
                return f"{self.name} does not decode: {', '.join(sorted(unsupported))}"
        if not document.main_graph.inputs:
            return "This tracer needs a graph with declared inputs."
        if self.operators:
            unsupported_ops = {n.op_type for n in document.main_graph.nodes} - set(
                self.operators
            )
            if unsupported_ops:
                return (
                    f"{self.name} does not implement: "
                    f"{', '.join(sorted(unsupported_ops))}"
                )
        return None


def load_manifest(path: Path) -> TracePlugin:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise TracePluginError(
            "Tracer manifest must be a regular JSON file below 64 KiB"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("api_version") != TRACE_API_VERSION:
        raise TracePluginError(
            "Unsupported tracer API version; expected api_version: 1"
        )
    for key in ("id", "name", "version", "entry_point"):
        if not isinstance(raw.get(key), str) or not raw[key] or len(raw[key]) > 128:
            raise TracePluginError(
                f"Tracer manifest needs a {key} string of 1-128 characters"
            )
    if not _ID.fullmatch(raw["id"]) or raw["id"].startswith("builtin."):
        raise TracePluginError("Tracer id must be lowercase; builtin.* is reserved")
    source, separator, function = raw["entry_point"].partition(":")
    relative = Path(source)
    if (
        not separator
        or not function.isidentifier()
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix != ".py"
        or any(p in {".venv", ".git", "__pycache__"} for p in relative.parts)
    ):
        raise TracePluginError(
            "entry_point must be a relative Python file and function: tracer.py:trace"
        )
    root = path.parent.resolve()
    entry = (root / relative).resolve()
    if not entry.is_relative_to(root) or not entry.is_file():
        raise TracePluginError(
            "Tracer entry point must be a file inside the plugin folder"
        )
    try:
        kinds = tuple(
            ArtifactKind(v) for v in _strings(raw, "artifact_kinds", required=True)
        )
        devices = tuple(TraceDevice(v) for v in _strings(raw, "devices")) or (
            TraceDevice.CPU,
        )
    except ValueError as error:
        raise TracePluginError(str(error)) from error
    if TraceDevice.AUTO in devices:
        raise TracePluginError("Plugin devices must name concrete devices")
    modules = _strings(raw, "python_modules")
    if len(modules) > 32 or any(
        not re.fullmatch(r"[A-Za-z_]\w{0,63}", module) for module in modules
    ):
        raise TracePluginError(
            "python_modules must name at most 32 top-level Python modules"
        )
    extra = raw.get("install_extra", "")
    if not isinstance(extra, str) or (
        extra and not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", extra)
    ):
        raise TracePluginError("install_extra must be a Python package extra name")
    streaming = raw.get("streaming_weights", False)
    if not isinstance(streaming, bool):
        raise TracePluginError("streaming_weights must be a boolean")
    return TracePlugin(
        id=raw["id"],
        name=raw["name"],
        version=raw["version"],
        artifact_kinds=kinds,
        architectures=_strings(raw, "architectures"),
        element_types=_strings(raw, "element_types"),
        devices=devices,
        description=str(raw.get("description", "")),
        manifest=path.resolve(),
        entry_point=raw["entry_point"],
        fingerprint=_fingerprint(root),
        operators=_strings(raw, "operators"),
        python_modules=modules,
        install_extra=extra,
        streaming_weights=streaming,
    )
