"""Module lookup and capture failures must not leave stale hooks behind."""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.huggingface.to_ir import open_huggingface
from nneditor.adapters.stages import StageBuilder
from nneditor.ir.core import Document
from nneditor.storage.store import TensorStore
from nneditor.tracing.contracts import TraceLimits
from nneditor.tracing.sdk import TraceContext, TracePluginError
from nneditor.tracing.stage_hooks import capture_stages
from tests.unit.test_modality_graphs import _snapshot


@pytest.mark.parametrize("missing_module", [False, True])
def test_unresolved_or_unexecuted_stage_fails_and_removes_hooks(
    tmp_path: Path,
    missing_module: bool,
) -> None:
    torch = pytest.importorskip("torch")
    source = open_huggingface(_snapshot(tmp_path / "model", {}))
    built = StageBuilder({}, "Hook validation")
    feed = built.input("input", (1, 2))
    first = built.step("first", "Identity", (feed,), "0", domain="test")
    output = built.step(
        "second", "Identity", (first,), "000" if missing_module else "1", domain="test"
    )
    document = Document(
        source=source.source,
        artifact_kind=source.artifact_kind,
        capabilities=source.capabilities.values(),
        graphs=[built.graph(output)],
        tensors=source.tensors.values(),
    )
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    with TensorStore(document) as tensors:
        context = TraceContext(
            document, {}, tensors, lambda _key, _array: None, limits=TraceLimits()
        )
        match = "has no module" if missing_module else "did not execute"
        with pytest.raises(TracePluginError, match=match):
            with capture_stages(model, context):
                model[0](torch.ones(1, 2))
    assert not model[0]._forward_hooks
    assert not model[1]._forward_hooks
