"""Modality plugins match a native forward and capture every graph port."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from nneditor.application.session import ApplicationService
from nneditor.tracing.contracts import TraceApproval, TraceLimits, TraceRequest
from nneditor.tracing.inputs import bind_inputs
from tests.fixtures import modality_models


@pytest.mark.parametrize("family", ["whisper", "llava", "qwen2_vl", "gemma3"])
def test_automatic_media_inputs_run_a_real_forward(tmp_path: Path, family: str) -> None:
    pytest.importorskip("transformers")
    from tests.unit.test_shell import make_shell

    model = getattr(modality_models, f"write_{family}")(tmp_path / "model")
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        session = service.open_model(tmp_path / "model")
        shell.show_session(session)
        generator = shell.input_generator
        generator.output_format.value = "npy"
        generated = generator.generate_configured_inputs(tmp_path)
        bindings = {
            name: session.trace_tensor_input(name, tensor.path)
            for name, tensor in generated
        }
        feeds = bind_inputs(
            session.document, session.default_trace_inputs(bindings=bindings)
        )
        assert np.isfinite(_native(model, family, feeds)).all()
        if family == "whisper":
            assert generator.input_presets["input-features"].kind == "audio-features"
            assert feeds["input-features"].shape == (1, 8, 64)
            np.testing.assert_array_equal(
                feeds["decoder-tokens"], [[model.config.decoder_start_token_id]]
            )
        if family == "qwen2_vl":
            assert feeds["image-grid"].shape == (1, 3)
            assert set(feeds["token-types"].flat) <= {0, 1}
            assert generator.input_presets["image-grid"].kind != "text-tokens"
        if family != "whisper":
            prompt = generator.input_presets["tokens"].prompt
            assert prompt is not None and prompt.vocab_size == 32
            assert "Diagnostic" in generator.input_presets["tokens"].assumptions[0]


def test_llava_feature_selection_and_merge_match_module_inputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from nneditor.adapters.huggingface.to_ir import open_huggingface
    from nneditor.storage.store import TensorStore
    from nneditor.tracing.modality_runtime import trace_modality
    from nneditor.tracing.sdk import TraceContext

    model = modality_models.write_llava(tmp_path / "model")
    feeds = _feeds("llava")
    expected = {}

    def remember(name: str) -> Any:
        def save(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            tensor = args[0] if args else kwargs["hidden_states"]
            expected[name] = tensor.detach().numpy().copy()

        return save

    handles = [
        module.register_forward_pre_hook(remember(name), with_kwargs=True)
        for name, module in (
            ("vision-features", model.model.multi_modal_projector),
            ("image-text-merge", model.model.language_model.layers[0]),
        )
    ]
    with torch.inference_mode():
        _native(model, "llava", feeds)
    for handle in handles:
        handle.remove()
    document = open_huggingface(tmp_path / "model")
    captured = {}
    with TensorStore(document) as tensors:
        trace_modality(
            TraceContext(
                document,
                feeds,
                tensors,
                lambda key, array: captured.update(
                    {document.main_graph.value(key).name: array.copy()}
                ),
                limits=TraceLimits(),
            )
        )
    for name, array in expected.items():
        np.testing.assert_allclose(captured[name], array, rtol=1e-4, atol=2e-5)
    graph = document.main_graph
    selection = next(
        node for node in graph.nodes if node.op_type == "VisionFeatureSelection"
    )
    # With one vision layer, -2 selects the pre-normalized embedding sequence.
    assert graph.value(selection.inputs[0]).name == "vision-pre-norm"
    assert captured["vision-features"].shape == (1, 1, 8)
    assert captured["vision-norm"].shape == (1, 8)


def _feeds(family: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)
    if family == "llava":
        return {
            "tokens": np.array([[1, 2, 3]], dtype=np.int64),
            "pixel-values": rng.normal(size=(1, 3, 8, 8)).astype(np.float32),
        }
    if family == "whisper":
        return {
            "input-features": rng.normal(size=(1, 8, 64)).astype(np.float32),
            "decoder-tokens": np.array([[1, 2]], dtype=np.int64),
        }
    if family == "t5":
        return {
            "source-tokens": np.array([[1, 2, 3]], dtype=np.int64),
            "decoder-tokens": np.array([[1, 2]], dtype=np.int64),
        }
    if family == "rwkv":
        return {"tokens": np.array([[1, 2, 3, 4]], dtype=np.int64)}
    if family == "qwen2_vl":
        return {
            "tokens": np.array([[3, 1, 1, 1, 1, 4]], dtype=np.int64),
            "pixel-values": rng.normal(size=(4, 3 * 2 * 14 * 14)).astype(np.float32),
            "image-grid": np.array([[1, 2, 2]], dtype=np.int64),
            "token-types": np.array([[0, 1, 1, 1, 1, 0]], dtype=np.int64),
        }
    if family == "gemma3":
        return {
            "tokens": np.array([[1, 1, 1, 1, 2]], dtype=np.int64),
            "pixel-values": rng.normal(size=(1, 3, 16, 16)).astype(np.float32),
        }
    if family == "bitnet":
        return {"tokens": np.array([[1, 2, 3]], dtype=np.int64)}
    if family == "sd3":
        return {
            "sample": rng.normal(size=(1, 4, 8, 8)).astype(np.float32),
            "timestep": np.array([10], dtype=np.int64),
            "encoder-hidden-states": rng.normal(size=(1, 4, 8)).astype(np.float32),
            "pooled-projections": rng.normal(size=(1, 8)).astype(np.float32),
        }
    if family == "flux":
        return {
            "sample": rng.normal(size=(1, 16, 4)).astype(np.float32),
            "timestep": np.array([10], dtype=np.int64),
            "encoder-hidden-states": rng.normal(size=(1, 4, 8)).astype(np.float32),
            "pooled-projections": rng.normal(size=(1, 8)).astype(np.float32),
            "img-ids": np.zeros((16, 3), dtype=np.float32),
            "txt-ids": np.zeros((4, 3), dtype=np.float32),
        }
    if family == "cogvideox":
        return {
            "sample": rng.normal(size=(1, 2, 4, 8, 8)).astype(np.float32),
            "timestep": np.array([10], dtype=np.int64),
            "encoder-hidden-states": rng.normal(size=(1, 4, 16)).astype(np.float32),
        }
    return {
        "sample": rng.normal(size=(1, 4, 2, 4, 4)).astype(np.float32),
        "timestep": np.array([10], dtype=np.int64),
        "encoder-hidden-states": rng.normal(size=(1, 4, 16)).astype(np.float32),
    }


def _native(model: Any, family: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    torch = importlib.import_module("torch")
    arguments = {
        {
            "tokens": "input_ids",
            "pixel-values": "pixel_values",
            "image-grid": "image_grid_thw",
            "token-types": "mm_token_type_ids",
            "input-features": "input_features",
            "decoder-tokens": "decoder_input_ids",
            "source-tokens": "input_ids",
            "sample": "hidden_states",
            "encoder-hidden-states": "encoder_hidden_states",
            "pooled-projections": "pooled_projections",
            "img-ids": "img_ids",
            "txt-ids": "txt_ids",
            "timestep": "timestep",
        }[name]: torch.from_numpy(np.array(array, copy=True))
        for name, array in feeds.items()
    }
    if family in {"llava", "whisper", "t5", "rwkv", "qwen2_vl", "gemma3", "bitnet"}:
        arguments["use_cache"] = False
    with torch.inference_mode():
        result = model(**arguments, return_dict=True)
    tensor = (
        result.logits
        if family in {"llava", "whisper", "t5", "rwkv", "qwen2_vl", "gemma3", "bitnet"}
        else result.sample
    )
    return cast(np.ndarray[Any, Any], tensor.detach().cpu().numpy())


@pytest.mark.parametrize(
    "family,plugin_id,library",
    [
        ("llava", "nneditor.vision", "transformers"),
        ("whisper", "nneditor.audio", "transformers"),
        ("t5", "nneditor.seq2seq", "transformers"),
        ("rwkv", "nneditor.rwkv", "transformers"),
        ("sd3", "nneditor.dit", "diffusers"),
        ("flux", "nneditor.dit", "diffusers"),
        ("cogvideox", "nneditor.video", "diffusers"),
        ("wan", "nneditor.video", "diffusers"),
        ("qwen2_vl", "nneditor.vision", "transformers"),
        ("gemma3", "nneditor.vision", "transformers"),
        ("bitnet", "nneditor.bitnet", "transformers"),
    ],
)
def test_modality_plugin_matches_native_forward(
    tmp_path: Path, family: str, plugin_id: str, library: str
) -> None:
    pytest.importorskip(library)
    writer = getattr(modality_models, f"write_{family}")
    model = writer(tmp_path / "model")
    feeds = _feeds(family)
    expected = _native(model, family, feeds)
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(tmp_path / "model")
        plugin = next(p for p in session.trace_registry.plugins if p.id == plugin_id)
        assert session.tracer_reason(plugin) is None
        bindings = {}
        for name, array in feeds.items():
            file = tmp_path / f"{name}.npy"
            np.save(file, array)
            bindings[name] = session.trace_tensor_input(name, file)
        inputs = session.default_trace_inputs(bindings=bindings)
        limits = TraceLimits(wall_seconds=180, memory_bytes=4 * 1024**3)
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            inputs,
            limits,
            tracer=plugin.identity,
        )
        result = session.trace_async(
            TraceRequest(inputs, limits, approval, tracer=plugin.identity)
        ).result(timeout=200)
        graph = session.document.main_graph
        assert set(result.captured_value_ids) == {value.id for value in graph.values}
        record = result.record(graph.outputs[0])
        actual = np.frombuffer(
            session.activations(result.id).read(record.value_id),
            dtype=record.numpy_dtype,
        ).reshape(record.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_bitnet_gguf_matches_the_hugging_face_forward(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    model = modality_models.write_bitnet(tmp_path / "hf")
    path = modality_models.write_bitnet_gguf(tmp_path / "bitnet.gguf", model)
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    expected = _native(model, "bitnet", {"tokens": tokens})
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        plugin = next(
            p for p in session.trace_registry.plugins if p.id == "nneditor.bitnet"
        )
        assert session.tracer_reason(plugin) is None
        file = tmp_path / "tokens.npy"
        np.save(file, tokens)
        binding = session.trace_tensor_input("tokens", file)
        inputs = session.default_trace_inputs(bindings={"tokens": binding})
        limits = TraceLimits(wall_seconds=180, memory_bytes=4 * 1024**3)
        approval = TraceApproval.approve(
            session.title,
            session.document.source.content_hash,
            inputs,
            limits,
            tracer=plugin.identity,
        )
        result = session.trace_async(
            TraceRequest(inputs, limits, approval, tracer=plugin.identity)
        ).result(timeout=200)
        graph = session.document.main_graph
        record = result.record(graph.outputs[0])
        actual = np.frombuffer(
            session.activations(result.id).read(record.value_id),
            dtype=record.numpy_dtype,
        ).reshape(record.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
