"""Metadata preflight and prompt preparation for Prism/Bonsai tracing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.gguf.constants import KV_TYPE_STRING, KV_TYPE_UINT32
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import Availability, Capability
from nneditor.storage.store import TensorStore
from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf
from tests.fixtures.huggingface_models import write_traceable_llama_snapshot
from tests.fixtures.qwen35_models import write_qwen35
from tests.unit.test_shell import make_shell


def _bonsai(path: Path, *, architecture: str = "qwen35", type_id: int = 142) -> Path:
    return write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", KV_TYPE_STRING, architecture),
            ("prism.hadamard.version", KV_TYPE_UINT32, 1),
            ("prism.hadamard.block_size", KV_TYPE_UINT32, 1024),
        ),
        tensors=(GgufFixtureTensor("token_embd.weight", (1, 128), type_id, bytes(34)),),
    )


def test_bonsai_declares_execution_without_reading_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_read(*_args: Any, **_kwargs: Any) -> bytes:
        pytest.fail("availability must not read model weights")

    monkeypatch.setattr(TensorStore, "read", unexpected_read)
    path, _, _ = write_qwen35(tmp_path / "Bonsai-PQ2_0.gguf", packing=142)
    document = open_gguf(path)
    status = document.capability(Capability.TRACING)
    assert status.availability is Availability.AVAILABLE
    assert (
        document.capability(Capability.WEIGHTS).availability is Availability.AVAILABLE
    )
    assert next(iter(document.tensors.values())).element_type == "prism.pq2_0"


@pytest.mark.parametrize("type_id,name", [(142, "PQ2_0"), (143, "PTQ1_0")])
def test_prism_requirements_also_apply_to_supported_decoder_families(
    tmp_path: Path,
    type_id: int,
    name: str,
) -> None:
    document = open_gguf(
        _bonsai(tmp_path / "model.gguf", architecture="llama", type_id=type_id)
    )
    reason = document.capability(Capability.TRACING).reason
    assert name in reason
    assert "Hadamard" in reason


def test_incomplete_bonsai_keeps_a_visible_explanation(
    tmp_path: Path,
) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(_bonsai(tmp_path / "model.gguf")))
        trace = shell.trace
        assert trace.run_button.disabled
        assert trace.blocker_notice.visible
        assert not trace.run_settings.visible
        assert not trace.prepare_prompt.visible
        assert trace.blocker_title.value == "Tracing unavailable"
        assert "metadata" in str(trace.blocker_text.value)
        assert trace.run_button.tooltip
        assert trace.blocker_notice in trace.control.controls
        assert not trace.approval_notice.visible


@pytest.mark.parametrize("packing", [142, 143])
def test_bonsai_run_enables_after_assigning_prompt(
    tmp_path: Path, packing: int
) -> None:
    import numpy as np

    path, _, _ = write_qwen35(tmp_path / "Bonsai.gguf", packing=packing)
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _ = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        trace = shell.trace
        assert trace.run_settings.visible
        assert trace.prepare_prompt.visible and not trace.prepare_prompt.disabled
        assert trace.blocker_title.value == "Prompt input required"
        tokens = tmp_path / "tokens.npy"
        np.save(tokens, np.array([[1, 2]], dtype=np.int64))
        trace.bind_tensor("tokens", session.trace_tensor_input("tokens", tokens))
        trace.refresh_actions()
        assert not trace.run_button.disabled
        assert not trace.blocker_notice.visible
        assert trace.approval_notice.visible


def test_prompt_prerequisite_is_distinct_from_runtime_support(tmp_path: Path) -> None:
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        session = service.open_model(write_traceable_llama_snapshot(tmp_path / "hf"))
        shell.show_session(session)
        trace = shell.trace
        assert trace.run_button.disabled
        assert trace.blocker_title.value == "Prompt input required"
        assert "Generate, save & assign" in str(trace.blocker_text.value)
        assert not trace.prepare_prompt.disabled
        assert trace.run_settings.visible
        from nneditor.prompt_inputs import generate_prompt_token_tensor

        generated = generate_prompt_token_tensor(
            tmp_path / "prompt.npy", document=session.document
        )
        trace.bind_tensor(
            "tokens", session.trace_tensor_input("tokens", generated.path)
        )
        trace.refresh_actions()
        assert not trace.run_button.disabled
        assert not trace.blocker_notice.visible
        assert trace.approval_notice.visible
