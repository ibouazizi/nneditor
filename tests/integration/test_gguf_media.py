"""CLIP vision and Whisper GGUF files match the Hugging Face forward."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.gguf.dequant import dequantize
from nneditor.application.session import ApplicationService
from nneditor.tracing.contracts import TraceApproval, TraceLimits, TraceRequest
from nneditor.tracing.gguf_media import clip_state_key, whisper_state_key
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf


def _seed() -> None:
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.manual_seed(7)


def _export(
    path: Path,
    architecture: str,
    kv: tuple[tuple[str, int, Any], ...],
    model: Any,
    translate: Any,
    candidates: tuple[str, ...],
) -> Path:
    state = {
        key: array.detach().float().cpu().numpy()
        for key, array in model.state_dict().items()
    }
    found: dict[str, str] = {}
    for name in candidates:
        key = translate(name)
        if key is not None and key in state:
            found[key] = name
    missing = set(state) - set(found)
    assert not missing, sorted(missing)
    tensors = tuple(
        GgufFixtureTensor(
            found[key],
            tuple(array.shape),
            0,
            np.ascontiguousarray(array).astype("<f4").tobytes(),
        )
        for key, array in state.items()
    )
    return write_gguf(
        path,
        version=3,
        kv=(("general.architecture", 8, architecture), *kv),
        tensors=tensors,
    )


def _clip_candidates() -> tuple[str, ...]:
    names = [
        "v.class_embd",
        "v.patch_embd.weight",
        "v.position_embd.weight",
        "v.pre_ln.weight",
        "v.pre_ln.bias",
        "v.post_ln.weight",
        "v.post_ln.bias",
    ]
    for leaf in (
        "ln1",
        "ln2",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "ffn_up",
        "ffn_down",
    ):
        for suffix in ("weight", "bias"):
            names.append(f"v.blk.0.{leaf}.{suffix}")
    return tuple(names)


def _whisper_candidates() -> tuple[str, ...]:
    names = [
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        "encoder.conv2.weight",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
        "encoder.ln_post.weight",
        "encoder.ln_post.bias",
        "decoder.token_embedding.weight",
        "decoder.positional_embedding",
        "decoder.ln.weight",
        "decoder.ln.bias",
        "proj_out.weight",
    ]
    for side, parts in (
        (
            "encoder",
            (
                "attn.query",
                "attn.key",
                "attn.value",
                "attn.out",
                "attn_ln",
                "mlp_ln",
                "mlp.0",
                "mlp.2",
            ),
        ),
        (
            "decoder",
            (
                "attn.query",
                "attn.key",
                "attn.value",
                "attn.out",
                "cross_attn.query",
                "cross_attn.key",
                "cross_attn.value",
                "cross_attn.out",
                "attn_ln",
                "cross_attn_ln",
                "mlp_ln",
                "mlp.0",
                "mlp.2",
            ),
        ),
    ):
        for leaf in parts:
            for suffix in ("weight", "bias"):
                names.append(f"{side}.blocks.0.{leaf}.{suffix}")
    return tuple(names)


def _trace(
    path: Path,
    feeds: dict[str, np.ndarray],
    expected_stages: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        plugin = next(
            item
            for item in session.trace_registry.plugins
            if item.id == "nneditor.gguf-media"
        )
        assert session.tracer_reason(plugin) is None
        bindings = {}
        for name, array in feeds.items():
            file = path.parent / f"{name}.npy"
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
        for name, expected in (expected_stages or {}).items():
            value = next(value for value in graph.values if value.name == name)
            stage = result.record(value.id)
            actual = np.frombuffer(
                session.activations(result.id).read(value.id),
                dtype=stage.numpy_dtype,
            ).reshape(stage.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
        record = result.record(graph.outputs[0])
        return np.frombuffer(
            session.activations(result.id).read(record.value_id),
            dtype=record.numpy_dtype,
        ).reshape(record.shape)


def _quantize(array: np.ndarray, kind: str) -> tuple[bytes, int]:
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    block = 32
    pieces: list[bytes] = []
    for start in range(0, flat.size, block):
        chunk = flat[start : start + block]
        peak = float(np.max(np.abs(chunk)))
        if kind == "q8_0":
            scale = np.float16(peak / 127.0) if peak else np.float16(0)
            width = float(scale)
            quants = (
                np.zeros(block, dtype=np.int8)
                if width == 0
                else np.clip(np.rint(chunk / width), -128, 127).astype(np.int8)
            )
            pieces.append(np.asarray([scale], dtype="<f2").tobytes() + quants.tobytes())
        else:
            scale = np.float16(peak / 7.0) if peak else np.float16(0)
            width = float(scale)
            levels = (
                np.zeros(block, dtype=np.uint8)
                if width == 0
                else np.clip(np.rint(chunk / width) + 8, 0, 15).astype(np.uint8)
            )
            nibbles = levels[:16] | (levels[16:] << 4)
            pieces.append(
                np.asarray([scale], dtype="<f2").tobytes() + nibbles.tobytes()
            )
    code = 8 if kind == "q8_0" else 2
    return b"".join(pieces), code


def test_clip_vision_gguf_matches_the_tower(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    _seed()
    config = transformers.CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=8,
        num_channels=3,
    )
    model = transformers.CLIPVisionModel(config).eval()
    path = _export(
        tmp_path / "vision.gguf",
        "clip",
        (
            ("clip.vision.embedding_length", 4, 8),
            ("clip.vision.feed_forward_length", 4, 16),
            ("clip.vision.block_count", 4, 1),
            ("clip.vision.attention.head_count", 4, 2),
            ("clip.vision.image_size", 4, 8),
            ("clip.vision.patch_size", 4, 8),
        ),
        model,
        clip_state_key,
        _clip_candidates(),
    )
    pixels = torch.randn(1, 3, 8, 8)
    stages = {}
    handle = model.encoder.layers[0].register_forward_hook(
        lambda _m, _i, output: stages.update(
            {"vision-block-000": output.detach().numpy().copy()}
        )
    )
    with torch.inference_mode():
        native = model(pixel_values=pixels)
        expected = native.last_hidden_state.numpy()
        stages["pooled-output"] = native.pooler_output.numpy()
    handle.remove()
    actual = _trace(path, {"pixel-values": pixels.numpy()}, stages)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("kind", ["q4_0", "q8_0"])
def test_packed_vision_gguf_matches_the_dequantized_tower(
    tmp_path: Path, kind: str
) -> None:
    pytest.importorskip("transformers")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    _seed()
    config = transformers.CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=8,
        patch_size=8,
        num_channels=3,
    )
    model = transformers.CLIPVisionModel(config).eval()
    state = {
        key: array.detach().float().cpu().numpy()
        for key, array in model.state_dict().items()
    }
    found: dict[str, str] = {}
    for name in _clip_candidates():
        key = clip_state_key(name)
        if key is not None and key in state:
            found[key] = name
    assert set(found) == set(state)
    restored = {}
    tensors = []
    for key, array in state.items():
        if array.size % 32 == 0:
            payload, code = _quantize(array, kind)
            restored[key] = dequantize(kind, array.shape, payload)
        else:
            payload, code = np.ascontiguousarray(array).astype("<f4").tobytes(), 0
            restored[key] = array
        tensors.append(GgufFixtureTensor(found[key], tuple(array.shape), code, payload))
    path = write_gguf(
        tmp_path / "packed-vision.gguf",
        version=3,
        kv=(
            ("general.architecture", 8, "clip"),
            ("clip.vision.embedding_length", 4, 32),
            ("clip.vision.feed_forward_length", 4, 64),
            ("clip.vision.block_count", 4, 1),
            ("clip.vision.attention.head_count", 4, 4),
            ("clip.vision.image_size", 4, 8),
            ("clip.vision.patch_size", 4, 8),
        ),
        tensors=tuple(tensors),
    )
    reference = transformers.CLIPVisionModel(config).eval()
    reference.load_state_dict(
        {key: torch.from_numpy(array.copy()) for key, array in restored.items()}
    )
    pixels = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        expected = reference(pixel_values=pixels).last_hidden_state.numpy()
    actual = _trace(path, {"pixel-values": pixels.numpy()})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_whisper_gguf_matches_the_decoder(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    _seed()
    config = transformers.WhisperConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=32,
        num_mel_bins=8,
        max_source_positions=64,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = transformers.WhisperForConditionalGeneration(config).eval()
    path = _export(
        tmp_path / "whisper.gguf",
        "whisper",
        (
            ("whisper.d_model", 4, 16),
            ("whisper.encoder_layers", 4, 1),
            ("whisper.decoder_layers", 4, 1),
            ("whisper.encoder_attention_heads", 4, 2),
            ("whisper.decoder_attention_heads", 4, 2),
            ("whisper.encoder_ffn_dim", 4, 32),
            ("whisper.decoder_ffn_dim", 4, 32),
            ("whisper.vocab_size", 4, 32),
            ("whisper.num_mel_bins", 4, 8),
            ("whisper.max_source_positions", 4, 64),
            ("whisper.max_target_positions", 4, 16),
        ),
        model,
        whisper_state_key,
        _whisper_candidates(),
    )
    features = torch.randn(1, 8, 128)
    tokens = torch.tensor([[1, 2]])
    stages = {}
    handles = [
        module.register_forward_hook(
            lambda _m, _i, output, name=name: stages.update(
                {name: output.detach().numpy().copy()}
            )
        )
        for name, module in (
            ("encoder-block-000", model.model.encoder.layers[0]),
            ("decoder-block-000", model.model.decoder.layers[0]),
        )
    ]
    with torch.inference_mode():
        expected = model(
            input_features=features, decoder_input_ids=tokens, use_cache=False
        ).logits.numpy()
    for handle in handles:
        handle.remove()
    actual = _trace(
        path,
        {"input-features": features.numpy(), "decoder-tokens": tokens.numpy()},
        stages,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_mtmd_audio_encoder_gguf_matches_the_encoder(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    _seed()
    config = transformers.WhisperConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=32,
        num_mel_bins=8,
        max_source_positions=64,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = transformers.WhisperForConditionalGeneration(config).eval()
    names = [
        "a.conv1d.1.weight",
        "a.conv1d.1.bias",
        "a.conv1d.2.weight",
        "a.conv1d.2.bias",
        "a.position_embd.weight",
        "a.post_ln.weight",
        "a.post_ln.bias",
    ]
    leaves = (
        "ln1",
        "ln2",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "ffn_up",
        "ffn_down",
    )
    for leaf in leaves:
        for suffix in ("weight", "bias"):
            names.append(f"a.blk.0.{leaf}.{suffix}")
    state = model.state_dict()
    encoder_keys = {key for key in state if key.startswith("model.encoder.")}
    found = {
        whisper_state_key(name): name
        for name in names
        if whisper_state_key(name) in encoder_keys
    }
    assert set(found) == encoder_keys
    tensors = tuple(
        GgufFixtureTensor(
            found[key],
            tuple(array.shape),
            0,
            array.detach().float().cpu().numpy().astype("<f4").tobytes(),
        )
        for key, array in state.items()
        if key in found
    )
    path = write_gguf(
        tmp_path / "audio.gguf",
        version=3,
        kv=(
            ("general.architecture", 8, "clip"),
            ("clip.audio.embedding_length", 4, 16),
            ("clip.audio.block_count", 4, 1),
            ("clip.audio.attention.head_count", 4, 2),
            ("clip.audio.feed_forward_length", 4, 32),
            ("clip.audio.num_mel_bins", 4, 8),
            ("clip.audio.max_pos_emb", 4, 64),
        ),
        tensors=tensors,
    )
    features = torch.randn(1, 8, 128)
    with torch.inference_mode():
        expected = model.model.encoder(features).last_hidden_state.numpy()
    actual = _trace(path, {"input-features": features.numpy()})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
