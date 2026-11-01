"""Discover, select, and install tracing extensions without importing them."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from importlib import resources
from pathlib import Path

from nneditor import __version__
from nneditor.application.persistence import default_state_directory
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import Document
from nneditor.tracing.contracts import TraceDevice
from nneditor.tracing.plugin_manifest import (
    MANIFEST_NAME,
    TracePlugin,
    load_manifest,
    plugin_files,
)
from nneditor.tracing.sdk import TracePluginError

__all__ = [
    "TracePlugin",
    "TracePluginError",
    "TraceRegistry",
    "create_trace_plugin",
    "default_trace_registry",
    "install_trace_plugin",
    "trace_plugin_directory",
]


def trace_plugin_directory() -> Path:
    return default_state_directory() / "tracers"


def _builtins() -> tuple[TracePlugin, ...]:
    return (
        TracePlugin(
            "builtin.onnx",
            "ONNX tracing",
            __version__,
            (ArtifactKind.ONNX_MODEL,),
            devices=(TraceDevice.CPU, TraceDevice.GPU, TraceDevice.NPU),
            description="ONNX Runtime or reference evaluator",
        ),
        TracePlugin(
            "builtin.numpy-llm",
            "NumPy LLM decoder",
            __version__,
            (ArtifactKind.GGUF_MODEL, ArtifactKind.HUGGINGFACE_MODEL),
            description="Llama, GPT-2, Qwen and Bonsai decoders",
            streaming_weights=True,
        ),
    )


class TraceRegistry:
    """A discovery snapshot; selection is deterministic and explicitly refreshable."""

    def __init__(
        self,
        plugins: Iterable[TracePlugin] = (),
        *,
        errors: Iterable[str] = (),
        roots: tuple[Path, ...] = (),
    ) -> None:
        all_plugins = (*_builtins(), *plugins)
        counts = Counter(plugin.id for plugin in all_plugins)
        duplicates = {key for key, count in counts.items() if count > 1}
        self.plugins = tuple(p for p in all_plugins if p.id not in duplicates)
        self.errors = (
            *errors,
            *(
                f"Duplicate tracer id {key}; conflicting extensions disabled"
                for key in sorted(duplicates)
            ),
        )
        self.roots = roots

    def reason(self, plugin: TracePlugin, document: Document) -> str | None:
        if reason := plugin.incompatibility(document):
            return reason
        missing = [
            module
            for module in plugin.python_modules
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            return (
                f"Install nneditor[{plugin.install_extra}] to use {plugin.name}."
                if plugin.install_extra
                else "Install missing tracer dependencies: " + ", ".join(missing)
            )
        if plugin.id == "builtin.numpy-llm":
            from nneditor.tracing.llm_support import trace_unavailable_reason

            return trace_unavailable_reason(document)
        if plugin.id == "builtin.onnx":
            status = document.capability(Capability.TRACING)
            if status.availability is not Availability.AVAILABLE:
                return status.reason
        return None

    def candidates(self, document: Document) -> tuple[TracePlugin, ...]:
        return tuple(
            p for p in self.plugins if document.artifact_kind in p.artifact_kinds
        )

    def preferred(self, document: Document) -> TracePlugin | None:
        return next(
            (p for p in self.candidates(document) if self.reason(p, document) is None),
            None,
        )

    def resolve(self, document: Document, identity: str | None) -> TracePlugin:
        if identity is None:
            # Legacy requests use only built-in implementations. Running third-party
            # code requires an approval bound to that plugin's exact identity.
            plugin = next(
                (
                    p
                    for p in self.candidates(document)
                    if p.builtin and self.reason(p, document) is None
                ),
                None,
            )
            if plugin is None:
                raise TracePluginError(
                    "Select a compatible tracer extension and approve its "
                    "identity before running."
                )
        else:
            plugin = next((p for p in self.plugins if p.identity == identity), None)
            if plugin is None:
                raise TracePluginError(
                    "The selected tracer changed or was removed. "
                    "Refresh tracers and approve again."
                )
        if reason := self.reason(plugin, document):
            raise TracePluginError(reason)
        return plugin

    def refreshed(self) -> TraceRegistry:
        return default_trace_registry(self.roots) if self.roots else self


def default_trace_registry(roots: Iterable[Path] | None = None) -> TraceRegistry:
    if roots is None:
        extra = os.environ.get("NNEDITOR_TRACE_PLUGIN_PATH", "")
        roots = (
            trace_plugin_directory(),
            *(Path(p) for p in extra.split(os.pathsep) if p),
        )
    paths: list[Path] = []
    plugins: list[TracePlugin] = []
    errors: list[str] = []
    manifests: set[Path] = set()
    for family in ("ssm", "diffusion"):
        bundled = (
            resources.files("nneditor.tracing") / "templates" / family / MANIFEST_NAME
        )
        if bundled.is_file():
            manifests.add(Path(str(bundled)))
    for supplied in roots:
        try:
            root = Path(supplied).expanduser().resolve()
            paths.append(root)
            if (root / MANIFEST_NAME).is_file():
                manifests.add(root / MANIFEST_NAME)
            elif root.is_dir():
                manifests.update(
                    p / MANIFEST_NAME
                    for p in root.iterdir()
                    if p.is_dir()
                    and not p.name.startswith(".")
                    and (p / MANIFEST_NAME).exists()
                )
        except (OSError, ValueError) as error:
            errors.append(f"{supplied}: {error}")
    for manifest in sorted(manifests):
        try:
            plugins.append(load_manifest(manifest))
        except (OSError, ValueError) as error:
            errors.append(f"{manifest.parent.name}: {error}")
    return TraceRegistry(plugins, errors=errors, roots=tuple(paths))


def snapshot_plugin(plugin: TracePlugin, destination: Path) -> TracePlugin:
    """Freeze the approved code before starting a worker, excluding bytecode caches."""
    if plugin.manifest is None:
        return plugin
    root = plugin.manifest.parent
    destination.mkdir(parents=True)
    for source in plugin_files(root):
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied = load_manifest(destination / MANIFEST_NAME)
    if copied.identity != plugin.identity:
        raise TracePluginError(
            "Tracer files changed after approval. Refresh tracers and approve again."
        )
    return copied


def install_trace_plugin(source: Path, *, directory: Path | None = None) -> TracePlugin:
    """Install a validated folder atomically, without executing or overwriting code."""
    plugin = load_manifest(source / MANIFEST_NAME)
    root = directory or trace_plugin_directory()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / plugin.id
    if destination.exists():
        raise TracePluginError(
            f"Tracer {plugin.id} is already installed at {destination}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=root))
    try:
        snapshot_plugin(plugin, temporary / "plugin")
        (temporary / "plugin").rename(destination)
    finally:
        shutil.rmtree(temporary)
    return load_manifest(destination / MANIFEST_NAME)


def create_trace_plugin(destination: Path, *, template: str = "llm") -> TracePlugin:
    """Create an editable working example without replacing an existing folder."""
    if template not in {"llm", "ssm", "diffusion"}:
        raise TracePluginError("Choose an llm, ssm, or diffusion tracer template.")
    source = resources.files("nneditor.tracing") / "templates" / template
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in (MANIFEST_NAME, "tracer.py"):
            (destination / name).write_bytes((source / name).read_bytes())
        if template != "llm":
            manifest = destination / MANIFEST_NAME
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["id"] = raw["id"].replace("nneditor.", "example.", 1)
            manifest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        return load_manifest(destination / MANIFEST_NAME)
    except BaseException:
        shutil.rmtree(destination)
        raise
