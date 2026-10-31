"""Exercise catalog navigation as a user searches, filters and inspects weights."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import pytest

from nneditor.adapters.gguf.constants import (
    KV_TYPE_ARRAY,
    KV_TYPE_STRING,
    KV_TYPE_UINT32,
)
from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.application.session import ApplicationService
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf
from tests.unit.test_huggingface_ui import _walk
from tests.unit.test_shell import make_shell


def _click(root: ft.Control, data: str) -> None:
    control = next(item for item in _walk(root) if item.data == data)
    cast(Any, control).on_click(None)


def _texts(root: ft.Control) -> str:
    return " ".join(
        str(item.value) for item in _walk(root) if isinstance(item, ft.Text)
    )


def _bundle(tmp_path: Path, kind: str) -> Path:
    if kind == "gguf":
        return write_gguf(
            tmp_path / "model.gguf",
            version=3,
            kv=(
                ("general.architecture", KV_TYPE_STRING, "llama"),
                ("general.name", KV_TYPE_STRING, "Inspection " * 30),
                ("llama.block_count", KV_TYPE_UINT32, 1),
                (
                    "tokenizer.ggml.tokens",
                    KV_TYPE_ARRAY,
                    (KV_TYPE_STRING, tuple(f"token{i}" for i in range(300))),
                ),
            ),
            tensors=tuple(
                GgufFixtureTensor(
                    f"blk.0.attn_q.extra_{i:03}.weight",
                    (4,),
                    1 if i == 0 else 0,
                    bytes(8 if i == 0 else 16),
                )
                for i in range(80)
            ),
        )
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 4,
                "notes": "Inspection " * 30,
                "rope_scaling": {"factor": 2.0},
                "labels": list(range(12)),
            }
        )
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": {f"token{i}": i for i in range(300)}},
                "added_tokens": [],
            }
        )
    )
    write_safetensors(
        root / "model.safetensors",
        (
            (
                f"model.layers.0.self_attn.q_proj.extra_{i:03}.weight",
                "float16" if i == 0 else "float32",
                (4,),
                bytes(8 if i == 0 else 16),
            )
            for i in range(80)
        ),
    )
    return root


@pytest.mark.parametrize("kind", ["hf", "gguf"])
def test_search_filter_paging_and_reopen_keep_catalog_usable(
    tmp_path: Path, kind: str
) -> None:
    path = _bundle(tmp_path, kind)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page = make_shell(service)
        shell.show_session(service.open_model(path))
        panel = shell.hf_panel if kind == "hf" else shell.gguf_panel

        def cards() -> list[ft.Control]:
            return [
                item
                for item in _walk(panel.root)
                if str(item.data).startswith(f"{kind}-tensor:")
            ]

        assert len(cards()) == 64
        _click(panel.root, f"{kind}-show-more")
        assert len(cards()) == 80
        _click(panel.root, f"{kind}-dtype:float16")
        assert len(cards()) == 1
        _click(panel.root, f"{kind}-dtype:float16")
        assert len(cards()) == 80
        panel.search.value = "extra_079"
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        assert len(cards()) == 1
        panel.search.value = "no-such-weight"
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        assert "No tensors" in _texts(panel.root)
        panel.search.value = ""
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        _click(panel.root, f"{kind}-pane:layout")
        assert (
            "Snapshot" in _texts(panel.root)
            if kind == "hf"
            else "GGUF" in _texts(panel.root)
        )
        _click(panel.root, f"{kind}-layout:weights")
        assert panel.pane == "tensors"
        # Return to a fresh session: filters and expanded pagination must reset.
        shell.show_session(service.open_model(path))
        assert panel.query == "" and panel.dtype_filter == ""
        assert len(cards()) == 64
        assert page.updates > 0


@pytest.mark.parametrize("kind", ["hf", "gguf"])
def test_metadata_values_and_bounded_vocab_preview_open_in_dialogs(
    tmp_path: Path, kind: str
) -> None:
    path = _bundle(tmp_path, kind)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, page = make_shell(service)
        shell.show_session(service.open_model(path))
        panel = shell.hf_panel if kind == "hf" else shell.gguf_panel
        pane = "config" if kind == "hf" else "metadata"
        _click(panel.root, f"{kind}-pane:{pane}")
        panel.search.value = "notes" if kind == "hf" else "general.name"
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        _click(
            panel.root,
            "hf-config-view:notes" if kind == "hf" else "gguf-kv-view:general.name",
        )
        assert "Inspection" in _texts(page.dialogs[-1])
        shell._close_gguf_dialog()
        panel.search.value = ""
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        _click(panel.root, f"{kind}-pane:tokenizer")
        panel.search.value = "vocab" if kind == "hf" else "tokens"
        cast(Any, panel.search.on_change)(SimpleNamespace(control=panel.search))
        _click(
            panel.root,
            "hf-tokenizer-preview"
            if kind == "hf"
            else "gguf-kv-preview:tokenizer.ggml.tokens",
        )
        body = _texts(page.dialogs[-1])
        assert "token0" in body and "token31" in body
        assert "token32" not in body
        assert "268 more not loaded" in body
