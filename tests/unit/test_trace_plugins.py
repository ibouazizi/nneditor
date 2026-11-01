"""Declarative discovery, compatibility, fingerprinting, and installation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.tracing.plugin_manifest import MANIFEST_NAME, load_manifest
from nneditor.tracing.plugins import (
    TraceRegistry,
    create_trace_plugin,
    default_trace_registry,
    install_trace_plugin,
    snapshot_plugin,
)
from nneditor.tracing.sdk import TracePluginError
from tests.fixtures.tracing_plugins import write_linear_gguf, write_trace_plugin


def test_discovery_is_declarative_and_supports_development_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "imported"
    root = write_trace_plugin(
        tmp_path / "plugins" / "test", code=f"raise RuntimeError({str(marker)!r})\n"
    )
    monkeypatch.setenv("NNEDITOR_TRACE_PLUGIN_PATH", str(root))
    registry = default_trace_registry()
    assert any(p.id == "test.linear" for p in registry.plugins)
    assert not marker.exists()
    document = open_gguf(write_linear_gguf(tmp_path / "model.gguf"))
    assert registry.preferred(document).id == "test.linear"  # type: ignore[union-attr]
    plugin = load_manifest(root / MANIFEST_NAME)
    assert registry.resolve(document, plugin.identity) == plugin
    with pytest.raises(TracePluginError, match="identity before running"):
        registry.resolve(document, None)
    assert len(TraceRegistry().plugins) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"api_version": 2},
        {"name": ""},
        {"id": "builtin.fake"},
        {"id": "../outside"},
        {"entry_point": "../outside.py:trace"},
        {"entry_point": "/outside.py:trace"},
        {"entry_point": "missing.py:trace"},
        {"entry_point": "tracer.py:not-valid"},
        {"entry_point": "tracer.py"},
        {"devices": ["auto"]},
        {"devices": ["invalid"]},
        {"artifact_kinds": []},
        {"artifact_kinds": ["unknown"]},
        {"architectures": "wrong"},
        {"element_types": [1]},
        {"operators": [""]},
        {"python_modules": ["parent.child"]},
        {"python_modules": ["../unsafe"]},
        {"python_modules": "torch"},
        {"install_extra": "unsafe]"},
        {"streaming_weights": "yes"},
    ],
)
def test_malformed_manifests_do_not_disable_other_tracers(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    root = write_trace_plugin(tmp_path / "bad", **changes)
    with pytest.raises(TracePluginError):
        load_manifest(root / MANIFEST_NAME)
    registry = default_trace_registry([tmp_path])
    assert {p.id for p in registry.plugins} == {
        "builtin.onnx",
        "builtin.numpy-llm",
        "nneditor.mamba",
        "nneditor.stable-diffusion",
    }
    assert len(registry.errors) == 1


def test_invalid_json_duplicate_ids_and_missing_plugins(tmp_path: Path) -> None:
    first = write_trace_plugin(tmp_path / "one")
    write_trace_plugin(tmp_path / "two")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MANIFEST_NAME).write_text("{")
    registry = default_trace_registry([tmp_path])
    assert not any(p.id == "test.linear" for p in registry.plugins)
    assert any("Duplicate" in e for e in registry.errors)
    document = open_gguf(write_linear_gguf(tmp_path / "model.gguf"))
    with pytest.raises(TracePluginError, match="changed or was removed"):
        registry.resolve(document, load_manifest(first / MANIFEST_NAME).identity)


def test_compatibility_explains_architecture_types_and_operators(
    tmp_path: Path,
) -> None:
    document = open_gguf(write_linear_gguf(tmp_path / "model.gguf"))
    for key, value, reason in (
        ("artifact_kinds", ["onnx_model"], "does not support"),
        ("architectures", ["different"], "supports architectures"),
        ("element_types", ["float16"], "does not decode"),
        ("operators", ["Relu"], "does not implement"),
    ):
        plugin = load_manifest(
            write_trace_plugin(tmp_path / key, **{key: value}) / MANIFEST_NAME
        )
        assert reason in str(plugin.incompatibility(document))
        with pytest.raises(TracePluginError, match=reason):
            TraceRegistry([plugin]).resolve(document, plugin.identity)


def test_hashes_helpers_and_data_but_ignores_caches(tmp_path: Path) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    before = load_manifest(root / MANIFEST_NAME)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"cache")
    assert load_manifest(root / MANIFEST_NAME).identity == before.identity
    (root / "helper.py").write_text("SCALE = 2\n")
    after = load_manifest(root / MANIFEST_NAME)
    assert after.id == before.id and after.version == before.version
    assert after.identity != before.identity
    with pytest.raises(TracePluginError, match="changed after approval"):
        snapshot_plugin(before, tmp_path / "stale-snapshot")
    frozen = snapshot_plugin(after, tmp_path / "frozen")
    assert frozen.identity == after.identity
    assert not (tmp_path / "frozen" / "__pycache__").exists()
    (root / "helper.py").write_text("SCALE = 3\n")
    assert load_manifest(frozen.manifest).identity == after.identity  # type: ignore[arg-type]


def test_plugin_tree_limits_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nneditor.tracing import plugin_manifest

    root = write_trace_plugin(tmp_path / "plugin")
    linked = root / "link.py"
    linked.symlink_to(root / "helper.py")
    with pytest.raises(TracePluginError, match="symlinks"):
        load_manifest(root / MANIFEST_NAME)
    linked.unlink()
    monkeypatch.setattr(plugin_manifest, "_MAX_BYTES", 10)
    with pytest.raises(TracePluginError, match="exceeds"):
        load_manifest(root / MANIFEST_NAME)


def test_create_install_refresh_and_duplicate_install(tmp_path: Path) -> None:
    created = create_trace_plugin(tmp_path / "source")
    registry = default_trace_registry([tmp_path / "installed"])
    assert not registry.errors
    assert not any(p.id == created.id for p in registry.plugins)
    installed = install_trace_plugin(
        tmp_path / "source", directory=tmp_path / "installed"
    )
    assert installed.identity == created.identity
    assert any(
        plugin.identity == created.identity for plugin in registry.refreshed().plugins
    )
    with pytest.raises(TracePluginError, match="already installed"):
        install_trace_plugin(tmp_path / "source", directory=tmp_path / "installed")
    with pytest.raises(FileExistsError):
        create_trace_plugin(tmp_path / "source")
    assert TraceRegistry().refreshed().plugins == TraceRegistry().plugins


def test_manifest_size_and_entry_point_symlink(tmp_path: Path) -> None:
    root = write_trace_plugin(tmp_path / "plugin")
    manifest = root / MANIFEST_NAME
    original = manifest.read_text()
    manifest.write_text(" " * (64 * 1024 + 1))
    with pytest.raises(TracePluginError, match="64 KiB"):
        load_manifest(manifest)
    manifest.write_text(original)
    (root / "tracer.py").unlink()
    (root / "tracer.py").symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("pass")
    with pytest.raises(TracePluginError, match="inside the plugin folder"):
        load_manifest(manifest)


def test_manifest_requires_an_object(tmp_path: Path) -> None:
    manifest = tmp_path / MANIFEST_NAME
    manifest.write_text(json.dumps([]))
    with pytest.raises(TracePluginError, match="API version"):
        load_manifest(manifest)


def test_unreadable_plugin_directory_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.iterdir

    def iterdir(path: Path) -> Any:
        if path == tmp_path:
            raise PermissionError("cannot read tracer directory")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    registry = default_trace_registry([tmp_path])
    assert {p.id for p in registry.plugins} == {
        "builtin.onnx",
        "builtin.numpy-llm",
        "nneditor.mamba",
        "nneditor.stable-diffusion",
    }
    assert "cannot read tracer directory" in registry.errors[0]
