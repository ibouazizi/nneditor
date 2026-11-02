"""Family graphs and plugin availability do not import heavyweight runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.huggingface.to_ir import open_huggingface
from nneditor.adapters.pytorch.safetensors import write_safetensors
from nneditor.application.session import ApplicationService
from nneditor.detokenization import text_activation_kind
from nneditor.tracing.plugins import create_trace_plugin, default_trace_registry
from tests.unit.test_shell import make_shell
from tests.unit.test_tensor_page import record_for


def _header(root: Path, **options: Any) -> Path:
    root.mkdir()
    config = dict(
        _class_name="UNet2DConditionModel",
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        down_block_types=["CrossAttnDownBlock2D", "DownBlock2D"],
        up_block_types=["UpBlock2D", "CrossAttnUpBlock2D"],
        cross_attention_dim=8,
        **options,
    )
    (root / "config.json").write_text(json.dumps(config))
    write_safetensors(
        root / "diffusion_pytorch_model.safetensors",
        [("conv_in.weight", "float32", (1,), b"\0" * 4)],
    )
    return root


def test_diffusion_inputs_and_skip_connections_are_real_graph_ports(
    tmp_path: Path,
) -> None:
    document = open_huggingface(_header(tmp_path / "unet"))
    graph = document.main_graph
    inputs = {graph.value(key).name: graph.value(key) for key in graph.inputs}
    assert inputs["sample"].shape == (1, 4, 8, 8)
    assert inputs["timestep"].element_type == "int64"
    assert inputs["encoder_hidden_states"].shape == (1, 77, 8)
    assert graph.value(graph.outputs[0]).name == "noise-prediction"
    initial = next(v for v in graph.values if v.name == "conv-in")
    assert len(graph.consumers(initial.id)) == 2
    assert all(n.domain == "nneditor.diffusion" for n in graph.nodes)
    from dataclasses import replace

    timestep = replace(record_for(np.ones(1, np.int64)), value_id=inputs["timestep"].id)
    assert text_activation_kind(document, timestep) is None
    registry = default_trace_registry([])
    plugin = next(p for p in registry.plugins if p.id == "nneditor.stable-diffusion")
    assert plugin.incompatibility(document) is None


@pytest.mark.parametrize("refiner", [False, True])
def test_sdxl_text_and_time_inputs(tmp_path: Path, refiner: bool) -> None:
    path = _header(
        tmp_path / "unet",
        addition_embed_type="text_time",
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=2560 if refiner else 2816,
    )
    config = json.loads((path / "config.json").read_text())
    config["cross_attention_dim"] = 1280 if refiner else 2048
    (path / "config.json").write_text(json.dumps(config))
    graph = open_huggingface(path).main_graph
    inputs = {graph.value(key).name: graph.value(key) for key in graph.inputs}
    assert inputs["text_embeds"].shape == (1, 1280)
    assert inputs["time_ids"].shape == (1, 5 if refiner else 6)


def test_unsupported_unet_remains_inspectable_and_reports_tracing_reason(
    tmp_path: Path,
) -> None:
    document = open_huggingface(
        _header(tmp_path / "unet", addition_embed_type="image_hint")
    )
    assert document.tensors
    error = dict(document.extensions)["x-huggingface.trace-graph-error"]
    assert isinstance(error, str) and "addition_embed_type" in error
    registry = default_trace_registry([])
    plugin = next(p for p in registry.plugins if p.id == "nneditor.stable-diffusion")
    assert registry.reason(plugin, document) == error


@pytest.mark.parametrize(
    "family",
    [
        "ssm",
        "diffusion",
        "vision",
        "dit",
        "video",
        "audio",
        "seq2seq",
        "rwkv",
        "bitnet",
        "gguf-media",
    ],
)
def test_packaged_plugins_have_editable_starters_and_explicit_dependencies(
    tmp_path: Path, family: str
) -> None:
    created = create_trace_plugin(tmp_path / "plugin", template=family)
    assert created.id.startswith("example.") and created.python_modules
    assert created.install_extra.startswith("tracer-")
    assert (tmp_path / "plugin" / "tracer.py").is_file()


def test_missing_plugin_runtime_is_reported_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = open_huggingface(_header(tmp_path / "unet"))
    registry = default_trace_registry([])
    plugin = next(p for p in registry.plugins if p.id == "nneditor.stable-diffusion")
    monkeypatch.setattr(
        "nneditor.tracing.plugins.importlib.util.find_spec", lambda _: None
    )
    assert "nneditor[tracer-diffusion]" in (registry.reason(plugin, document) or "")


def test_trace_panel_selects_missing_family_runtime_with_install_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _header(tmp_path / "unet")
    monkeypatch.setattr(
        "nneditor.tracing.plugins.importlib.util.find_spec", lambda _: None
    )
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        plugin = next(
            p
            for p in session.trace_registry.plugins
            if p.id == "nneditor.stable-diffusion"
        )
        assert shell.trace.tracer.value == plugin.identity
        assert (
            shell.trace.run_button.disabled and not shell.trace.prepare_prompt.visible
        )
        assert "nneditor[tracer-diffusion]" in str(shell.trace.status.value)


def test_whole_model_plugins_do_not_get_streaming_weight_memory_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nneditor.tracing.runner import recommended_trace_limits
    from nneditor.ui import app
    from tests.fixtures.huggingface_models import write_traceable_llama_snapshot

    flags: list[bool] = []

    def recommend(size: int, *, streaming_weights: bool = False) -> Any:
        flags.append(streaming_weights)
        return recommended_trace_limits(size, streaming_weights=streaming_weights)

    monkeypatch.setattr(app, "recommended_trace_limits", recommend)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        shell.show_session(service.open_model(_header(tmp_path / "unet")))
        shell.show_session(
            service.open_model(write_traceable_llama_snapshot(tmp_path / "llama"))
        )
    assert flags == [False, True]
