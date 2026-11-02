"""Independent PyTorch parity for hybrid layers, packings, and transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.gguf.dequant import GgufDequantError, dequantize
from nneditor.ir.capabilities import Availability, Capability
from nneditor.ir.core import Document, JsonValue
from nneditor.ir.serialize import document_from_json, document_to_json
from nneditor.storage.store import TensorStore
from nneditor.tracing.llm_runtime import LlmTraceError, trace_document
from nneditor.tracing.llm_support import trace_unavailable_reason
from nneditor.tracing.llm_weights import GgufWeights, hadamard
from tests.fixtures.qwen35_models import write_qwen35

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _reference(
    weights: dict[str, np.ndarray],
    ids: np.ndarray,
    signs: np.ndarray,
    *,
    prism: bool,
    grouped: bool,
) -> dict[str, np.ndarray]:
    w = {name: torch.from_numpy(array.copy()) for name, array in weights.items()}
    H = torch.ones(1, 1)
    for _ in range(7):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H /= np.sqrt(128)
    sign = torch.from_numpy(signs.astype(np.float32))

    def linear(name: str, x: Any) -> Any:
        if prism and w[name].shape[-1] == 128:
            if grouped and name.endswith("ssm_out.weight"):
                # Qwen3.5 cycles keys [0,1,0,1,...], Prism groups [0,0,...,1,1,...].
                x = x.reshape(*x.shape[:-1], 8, 16)[
                    ..., [0, 2, 4, 6, 1, 3, 5, 7], :
                ].flatten(-2)
            x = (x * sign) @ H
        return F.linear(x, w[name])

    def norm(x: Any, name: str) -> Any:
        return F.rms_norm(x, (x.shape[-1],), w[name], 1e-6)

    x = F.embedding(torch.from_numpy(ids), w["token_embd.weight"])
    if prism:
        x = (x @ H) * sign
    output = {"embed": x.numpy()}
    b, seq = ids.shape
    for block in range(2):
        p = f"blk.{block}."
        h = norm(x, p + "attn_norm.weight")
        if block == 0:
            qkv = linear(p + "attn_qkv.weight", h)
            conv = F.conv1d(
                F.pad(qkv.transpose(1, 2), (3, 0)),
                w[p + "ssm_conv1d.weight"][:, None, :],
                groups=192,
            ).transpose(1, 2)
            q, k, v = torch.split(F.silu(conv), [32, 32, 128], dim=-1)
            q = F.normalize(q.reshape(b, seq, 2, 16), dim=-1, eps=1e-6)
            k = F.normalize(k.reshape(b, seq, 2, 16), dim=-1, eps=1e-6)
            v = v.reshape(b, seq, 8, 16)
            beta = torch.sigmoid(linear(p + "ssm_beta.weight", h))
            decay = (
                F.softplus(linear(p + "ssm_alpha.weight", h) + w[p + "ssm_dt.bias"])
                * w[p + "ssm_a"]
            )
            # Use scalar sequence/head/token loops and transposed state storage.
            core = torch.zeros_like(v)
            for batch in range(b):
                for head in range(8):
                    state = torch.zeros(16, 16)  # [key, value]
                    for t in range(seq):
                        key = k[batch, t, head % 2]
                        state = state * torch.exp(decay[batch, t, head])
                        delta = (v[batch, t, head] - key @ state) * beta[batch, t, head]
                        state += torch.outer(key, delta)
                        core[batch, t, head] = (q[batch, t, head % 2] @ state) / 4
            gate = linear(p + "attn_gate.weight", h).reshape(core.shape)
            mixed = norm(core, p + "ssm_norm.weight") * F.silu(gate)
            x = x + linear(p + "ssm_out.weight", mixed.flatten(-2))
            output["block-000-ssm"] = x.numpy()
        else:
            q, gate = linear(p + "attn_q.weight", h).reshape(b, seq, 4, 32).chunk(2, -1)
            q = norm(q, p + "attn_q_norm.weight")
            k = norm(
                linear(p + "attn_k.weight", h).reshape(b, seq, 2, 16),
                p + "attn_k_norm.weight",
            )
            v = linear(p + "attn_v.weight", h).reshape(b, seq, 2, 16)
            angles = torch.arange(seq)[:, None] * (
                10000.0 ** (-torch.arange(0, 8, 2) / 8)
            )
            for t in (q, k):
                a, c = t[..., :4].clone(), t[..., 4:8].clone()
                t[..., :4] = (
                    a * angles.cos()[None, :, None] - c * angles.sin()[None, :, None]
                )
                t[..., 4:8] = (
                    c * angles.cos()[None, :, None] + a * angles.sin()[None, :, None]
                )
            mixed = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.repeat_interleave(2, 2).transpose(1, 2),
                v.repeat_interleave(2, 2).transpose(1, 2),
                is_causal=True,
            ).transpose(1, 2)
            x = x + linear(
                p + "attn_output.weight", (mixed * gate.sigmoid()).flatten(-2)
            )
            output["block-001-attn"] = x.numpy()
        h = norm(x, p + "post_attention_norm.weight")
        x = x + linear(
            p + "ffn_down.weight",
            F.silu(linear(p + "ffn_gate.weight", h)) * linear(p + "ffn_up.weight", h),
        )
        output[f"block-{block:03d}-ffn"] = x.numpy()
    x = norm(x, "output_norm.weight")
    output["output-norm"] = x.numpy()
    output["lm-head"] = linear(
        "output.weight" if "output.weight" in w else "token_embd.weight", x
    ).numpy()
    return output


@pytest.mark.parametrize(
    "packing,endian,grouped",
    [
        (0, "little", False),
        (142, "little", True),
        (143, "little", True),
        (142, "big", False),
        (143, "big", False),
    ],
)
def test_hybrid_trace_matches_torch(
    tmp_path: Path, packing: int, endian: str, grouped: bool
) -> None:
    path, weights, signs = write_qwen35(
        tmp_path / "hybrid.gguf", packing=packing, endian=endian, grouped=grouped
    )
    document = open_gguf(path)
    assert (
        document.capability(Capability.TRACING).availability is Availability.AVAILABLE
    )
    assert [n.op_type for n in document.main_graph.nodes] == [
        "Embedding",
        "SSM",
        "FFN",
        "Attention",
        "FFN",
        "RMSNorm",
        "Linear",
    ]
    ids = np.array([[1, 4, 2, 6, 3], [3, 2, 7, 2, 0]], dtype=np.int64)
    expected = _reference(weights, ids, signs, prism=bool(packing), grouped=grouped)
    with TensorStore(document) as store:
        actual, _, _ = trace_document(document, {"tokens": ids}, store)
        again, _, _ = trace_document(document, {"tokens": ids[:, :2]}, store)
    for value_id, value in actual.items():
        name = document.main_graph.value(value_id).name
        if name == "tokens":
            continue
        np.testing.assert_allclose(value, expected[str(name)], rtol=4e-4, atol=2e-5)
        # Prefix invariance checks convolution padding, causality, and reset state.
        np.testing.assert_allclose(value[:, :2], again[value_id], rtol=4e-4, atol=2e-5)


def test_hadamard_block_transform_matches_explicit_matrix() -> None:
    x = np.random.default_rng(2).normal(size=(2, 3, 24)).astype(np.float32)
    matrix = np.array(
        [[(-1) ** ((i & j).bit_count()) for i in range(8)] for j in range(8)],
        dtype=np.float32,
    ) / np.sqrt(np.float32(8))
    expected = (x.reshape(-1, 8) @ matrix).reshape(x.shape)
    np.testing.assert_allclose(hadamard(x, 8), expected, atol=5e-7)
    np.testing.assert_allclose(hadamard(hadamard(x, 8), 8), x, atol=5e-7)
    for block in (0, 3, 16):
        with pytest.raises(LlmTraceError):
            hadamard(x, block)


@pytest.mark.parametrize("packing,size", [(142, 34), (143, 28)])
def test_ternary_decoders_known_digits_and_truncated_payloads(
    packing: int, size: int
) -> None:
    raw = bytearray(size)
    start = 0 if packing == 142 else 26
    raw[start : start + 2] = np.float16(0.5).tobytes()
    expected = np.full((128,), -0.5, dtype=np.float32)
    name = "prism.pq2_0" if packing == 142 else "prism.ptq1_0"
    np.testing.assert_array_equal(dequantize(name, (128,), bytes(raw)), expected)
    with pytest.raises(GgufDequantError):
        dequantize(name, (128,), bytes(raw[:-1]))
    with pytest.raises(GgufDequantError):
        dequantize(name, (127,), bytes(raw))


def test_metadata_arrays_remain_lazy_until_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nneditor.adapters.gguf import indexer

    monkeypatch.setattr(indexer, "MAX_RETAINED_ARRAY_BYTES", 16)
    path, _, _ = write_qwen35(tmp_path / "elided.gguf", packing=143)
    document = open_gguf(path)
    assert (
        document.capability(Capability.TRACING).availability is Availability.AVAILABLE
    )
    with TensorStore(document) as store:
        weights = GgufWeights(document, store, "little")
        assert weights.signs[128].shape == (128,)
        assert weights.folded
        # Matrix reads stay below the configured decoded chunk size.
        calls: list[int] = []
        original = store.read

        def read(tensor_id: str, **kwargs: Any) -> bytes:
            calls.append(kwargs.get("length", 0))
            return original(tensor_id, **kwargs)

        monkeypatch.setattr(
            TensorStore, "read", lambda _self, tid, **kw: read(tid, **kw)
        )
        weights.linear("output.weight", np.ones((1, 128), dtype=np.float32))
        assert 0 < max(calls) <= 8 * 1024 * 1024


def _changed_metadata(document: Document, changes: dict[str, JsonValue]) -> Document:
    data = document_to_json(document)
    extensions = data["extensions"]
    assert isinstance(extensions, dict)
    meta = extensions["x-gguf.metadata"]
    assert isinstance(meta, dict)
    meta.update(changes)
    return document_from_json(data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("qwen35.block_count", 0),
        ("qwen35.block_count", 1000000),
        ("qwen35.ssm.group_count", 3),
        ("qwen35.ssm.inner_size", 129),
        ("qwen35.rope.dimension_count", 3),
        ("qwen35.rope.dimension_count", 18),
        ("qwen35.attention.value_length", 32),
        ("qwen35.rope.scaling.type", "linear"),
        ("prism.hadamard.version", 3),
        ("prism.hadamard.transform", "unknown"),
        ("prism.hadamard.axis", "output"),
        ("prism.hadamard.block_size", 3),
        ("prism.hadamard.block_size", 0),
        ("prism.hadamard.sign_mode", []),
        ("prism.hadamard.weight_names", []),
        ("prism.hadamard.weight_names", None),
        ("prism.hadamard.weight_names", {"type": "array", "count": -1}),
        ("prism.hadamard.sign_widths", []),
        ("prism.hadamard.sign_widths", [127]),
        ("prism.hadamard.sign_values", []),
    ],
)
def test_invalid_metadata_is_rejected_before_weight_reads(
    tmp_path: Path, key: str, value: JsonValue, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=142)
    document = _changed_metadata(open_gguf(path), {key: value})
    monkeypatch.setattr(
        TensorStore, "read", lambda *a, **k: pytest.fail("Preflight read weights")
    )
    assert trace_unavailable_reason(document)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"prism.hadamard.weight_names": ["missing.weight"]}, "unknown weight"),
        ({"prism.hadamard.weight_names": ["token_embd.weight"]}, "folded weights"),
        (
            {"prism.hadamard.weight_names": ["blk.0.ssm_conv1d.weight"]},
            "folded weights",
        ),
        ({"prism.hadamard.inverse_weight_names": ["output.weight"]}, "inverse weight"),
        ({"prism.hadamard.sign_values": [0] * 128}, "signs must be"),
        (
            {
                "prism.hadamard.sign_widths": [128, 128],
                "prism.hadamard.sign_values": [1] * 256,
            },
            "Duplicate",
        ),
        (
            {
                "prism.hadamard.sign_widths": [256],
                "prism.hadamard.sign_values": [1] * 256,
            },
            "Missing Hadamard signs",
        ),
        ({"prism.hadamard.weight_names": ["output_norm.weight"]}, "weight dimensions"),
    ],
)
def test_invalid_transform_payload_is_refused(
    tmp_path: Path, changes: dict[str, JsonValue], match: str
) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=142)
    document = _changed_metadata(open_gguf(path), changes)
    with TensorStore(document) as store, pytest.raises(LlmTraceError, match=match):
        GgufWeights(document, store, "little")


def test_prompt_bounds_and_streamed_capture(tmp_path: Path) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=143)
    document = open_gguf(path)
    with TensorStore(document) as store:
        for ids in (
            np.array([[-1]]),
            np.array([[9]]),
            np.ones((1, 33), dtype=np.int64),
        ):
            with pytest.raises(LlmTraceError, match="limits"):
                trace_document(document, {"tokens": ids}, store)
        emitted: list[str] = []
        retained, _, _ = trace_document(
            document,
            {"tokens": np.array([[1, 2]])},
            store,
            capture=lambda key, _value: emitted.append(key),
        )
        assert not retained
        assert len(emitted) == 8


@pytest.mark.parametrize("packing", [142, 143])
def test_isolated_bonsai_capture(tmp_path: Path, packing: int) -> None:
    from nneditor.application.session import ApplicationService
    from tests.unit.test_llm_trace import _approved_request

    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=packing)
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, np.array([[1, 2, 3]], dtype=np.int64))
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        binding = session.trace_tensor_input("tokens", tokens)
        spec = session.default_trace_inputs(bindings={"tokens": binding})
        result = session.trace_async(_approved_request(session, spec)).result(
            timeout=20
        )
        assert len(result.records) == 8
        assert all(r.readable for r in result.records)
        head = next(r for r in result.records if r.value_name == "lm-head")
        actual = np.frombuffer(
            session.activations(result.id).read(head.value_id), np.float32
        ).reshape(head.shape)
        with TensorStore(session.document) as store:
            direct, _, _ = trace_document(
                session.document, {"tokens": np.array([[1, 2, 3]])}, store
            )
        np.testing.assert_allclose(actual, direct[head.value_id], atol=2e-5)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("dims", [8, 128], "expected shape"),
        ("element_type", "gguf.type_999", "unsupported"),
    ],
)
def test_bad_weight_catalog_is_rejected(
    tmp_path: Path, field: str, value: JsonValue, match: str
) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf")
    document = open_gguf(path)
    data = document_to_json(document)
    tensors = data["tensors"]
    assert isinstance(tensors, list) and isinstance(tensors[0], dict)
    tensors[0][field] = value
    reason = trace_unavailable_reason(document_from_json(data))
    assert reason is not None and match in reason
    missing = _changed_metadata(document, {"qwen35.block_count": 3})
    assert "missing required weight" in str(trace_unavailable_reason(missing))


def test_extra_weights_are_not_silently_ignored(tmp_path: Path) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf")
    data = document_to_json(open_gguf(path))
    tensors = data["tensors"]
    assert isinstance(tensors, list) and isinstance(tensors[0], dict)
    tensors.append({**tensors[0], "id": "t:g:main#extra.weight"})
    assert "Unsupported Qwen3.5 weight" in str(
        trace_unavailable_reason(document_from_json(data))
    )


def test_identity_sign_mode_and_optional_inverse_list(tmp_path: Path) -> None:
    path, _, _ = write_qwen35(tmp_path / "model.gguf", packing=142)
    document = _changed_metadata(
        open_gguf(path), {"prism.hadamard.sign_mode": "identity"}
    )
    data = document_to_json(document)
    extensions = data["extensions"]
    assert isinstance(extensions, dict)
    meta = extensions["x-gguf.metadata"]
    assert isinstance(meta, dict)
    meta.pop("prism.hadamard.inverse_weight_names")
    document = document_from_json(data)
    with TensorStore(document) as store:
        w = GgufWeights(document, store, "little")
        assert not w.signs and not w.inverses
        captures, _, _ = trace_document(document, {"tokens": np.array([[1, 2]])}, store)
        assert all(np.isfinite(v).all() for v in captures.values())


def test_ternary_type_ids_are_scoped_to_prism_metadata(tmp_path: Path) -> None:
    from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

    path = write_gguf(
        tmp_path / "opaque.gguf",
        version=3,
        tensors=(GgufFixtureTensor("unknown", (128,), 142, bytes(34)),),
    )
    opaque = open_gguf(path)
    assert next(iter(opaque.tensors.values())).element_type == "gguf.type_142"
    assert any(d.code == "gguf.unknown-tensor-type" for d in opaque.diagnostics)
    prism, _, _ = write_qwen35(tmp_path / "prism.gguf", packing=142)
    known = open_gguf(prism)
    assert not any(d.code == "gguf.unknown-tensor-type" for d in known.diagnostics)
    tensor = next(iter(known.tensors.values()))
    assert tensor.payload is not None and tensor.payload.length == 9 * 34


def test_tied_output_and_unsupported_prism_tying(tmp_path: Path) -> None:
    path, weights, signs = write_qwen35(tmp_path / "tied.gguf", tied=True)
    document = open_gguf(path)
    tokens = np.array([[1, 2]], dtype=np.int64)
    with TensorStore(document) as store:
        captures, _, _ = trace_document(document, {"tokens": tokens}, store)
    head = captures[document.main_graph.outputs[0]]
    reference = _reference(weights, tokens, signs, prism=False, grouped=False)
    np.testing.assert_allclose(head, reference["lm-head"], rtol=4e-4, atol=2e-5)
    path, _, _ = write_qwen35(tmp_path / "prism-tied.gguf", packing=142, tied=True)
    reason = open_gguf(path).capability(Capability.TRACING).reason
    assert "explicit output projection" in reason


@pytest.mark.parametrize("packing", [142, 143])
def test_prism_version_two_tied_head_matches_torch(
    tmp_path: Path, packing: int
) -> None:
    path, weights, signs = write_qwen35(
        tmp_path / "v2.gguf",
        packing=packing,
        tied=True,
        hadamard_version=2,
    )
    document = open_gguf(path)
    assert (
        document.capability(Capability.TRACING).availability is Availability.AVAILABLE
    )
    tokens = np.array([[1, 4, 2, 6], [3, 2, 7, 0]], dtype=np.int64)
    expected = _reference(weights, tokens, signs, prism=True, grouped=True)
    with TensorStore(document) as store:
        actual, _, _ = trace_document(document, {"tokens": tokens}, store)
    for value_id, value in actual.items():
        name = document.main_graph.value(value_id).name
        if name == "tokens":
            continue
        np.testing.assert_allclose(value, expected[str(name)], rtol=4e-4, atol=2e-5)


def test_hadamard_version_two_contract(tmp_path: Path) -> None:
    path, _, _ = write_qwen35(tmp_path / "v1.gguf", packing=142)

    def reason(changes: dict[str, JsonValue]) -> str | None:
        return trace_unavailable_reason(_changed_metadata(open_gguf(path), changes))

    untied = reason({"prism.hadamard.version": 2})
    assert untied is not None and "tied_output=true" in untied
    forbidden = reason({"prism.hadamard.tied_output": True})
    assert forbidden is not None and "version 1 forbids it" in forbidden
    separate_head = reason(
        {
            "prism.hadamard.version": 2,
            "prism.hadamard.tied_output": True,
        }
    )
    assert separate_head is not None and "output.weight to be absent" in separate_head
    tied_path, _, _ = write_qwen35(
        tmp_path / "v2.gguf", packing=142, tied=True, hadamard_version=2
    )
    missing_latent = trace_unavailable_reason(
        _changed_metadata(
            open_gguf(tied_path),
            {"prism.hadamard.inverse_weight_names": ["blk.0.attn_qkv.weight"]},
        )
    )
    assert missing_latent is not None and "latent token embedding" in missing_latent
