"""The packaged extensions execute real models and capture numerical outputs."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from nneditor.application.session import ApplicationService
from nneditor.tracing.contracts import TraceApproval, TraceLimits, TraceRequest
from tests.fixtures.model_families import write_diffusion, write_mamba


@pytest.mark.parametrize("family", ["mamba", "mamba2", "sd", "sdxl"])
def test_family_plugin_matches_native_forward_and_captures_all_ports(
    tmp_path: Path,
    family: str,
) -> None:
    pytest.importorskip("transformers" if family.startswith("mamba") else "diffusers")
    torch = importlib.import_module("torch")
    path = tmp_path / "model"
    if family.startswith("mamba"):
        model = write_mamba(path, second=family == "mamba2")
        feeds = {"tokens": np.array([[1, 3, 5, 7, 9]], dtype=np.int64)}
        with torch.inference_mode():
            expected = model(
                input_ids=torch.from_numpy(feeds["tokens"]), use_cache=False
            ).logits.numpy()
        plugin_id = "nneditor.mamba"
    else:
        model = write_diffusion(path, xl=family == "sdxl")
        rng = np.random.default_rng(19)
        feeds = {
            "sample": rng.normal(size=(1, 4, 8, 8)).astype(np.float32),
            "timestep": np.array([20], dtype=np.int64),
            "encoder_hidden_states": rng.normal(size=(1, 77, 8)).astype(np.float32),
        }
        if family == "sdxl":
            feeds.update(
                text_embeds=rng.normal(size=(1, 4)).astype(np.float32),
                time_ids=np.array([[64, 64, 0, 0, 64, 64]], dtype=np.float32),
            )
        kwargs = {
            name: torch.from_numpy(array)
            for name, array in feeds.items()
            if name not in {"text_embeds", "time_ids"}
        }
        if family == "sdxl":
            kwargs["added_cond_kwargs"] = {
                name: torch.from_numpy(feeds[name])
                for name in ("text_embeds", "time_ids")
            }
        with torch.inference_mode():
            expected = model(**kwargs).sample.numpy()
        plugin_id = "nneditor.stable-diffusion"
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        plugin = next(p for p in session.trace_registry.plugins if p.id == plugin_id)
        assert session.tracer_reason(plugin) is None
        bindings = {}
        for name, array in feeds.items():
            file = tmp_path / f"{name}.npy"
            np.save(file, array)
            bindings[name] = session.trace_tensor_input(name, file)
        inputs = session.default_trace_inputs(bindings=bindings)
        limits = TraceLimits(wall_seconds=120, memory_bytes=4 * 1024**3)
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            inputs,
            limits,
            tracer=plugin.identity,
        )
        result = session.trace_async(
            TraceRequest(inputs, limits, approval, tracer=plugin.identity)
        ).result(timeout=135)
        graph = session.document.main_graph
        assert set(result.captured_value_ids) == {value.id for value in graph.values}
        record = result.record(graph.outputs[0])
        actual = np.frombuffer(
            session.activations(result.id).read(record.value_id),
            dtype=record.numpy_dtype,
        ).reshape(record.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
        if family.startswith("sd"):
            skips = [v for v in graph.values if "skip" in v.name]
            assert skips and all(result.record(v.id).readable for v in skips)
            assert "nneditor.diffusion" in {n.domain for n in graph.nodes}
