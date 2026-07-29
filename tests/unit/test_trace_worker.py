"""Direct coverage of the minimal subprocess trace entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

from nneditor.tracing import trace_worker
from tests.fixtures.onnx_models import (
    build_custom_domain_model,
    build_embedded_model,
)


def _target(
    value_id: str,
    value_name: str,
    *,
    node_id: str | None = "node",
    graph_output: bool = False,
) -> dict[str, object]:
    return {
        "value_id": value_id,
        "value_name": value_name,
        "node_id": node_id,
        "role": "node-output" if node_id else "graph-input",
        "graph_output": graph_output,
        "element_type": "float32",
        "shape": [4],
    }


def test_worker_run_captures_augmented_outputs_in_bounded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    build_embedded_model(model_path, elements=4)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, input=np.arange(4, dtype=np.float32))
    output = tmp_path / "output"
    output.mkdir()
    response = tmp_path / "response.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "inputs": str(inputs),
                "output": str(output),
                "response": str(response),
                "targets": [
                    _target("input-id", "input", node_id=None),
                    _target("scaled-id", "scaled"),
                    _target("output-id", "output"),
                ],
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                "capture_bytes": 1024,
                "chunk_bytes": 7,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    # ``run`` normally executes in a fresh subprocess. Applying its hard POSIX
    # address-space limit to pytest itself is irreversible and can kill the
    # coverage process after large optional runtimes have been imported.
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["runtime"].startswith("onnx.reference")
    assert [record["state"] for record in payload["records"]] == [
        "complete",
        "complete",
        "complete",
    ]
    assert all(
        (output / record["file_name"]).stat().st_size == record["stored_byte_length"]
        for record in payload["records"]
    )


def test_worker_prioritizes_boundaries_and_shares_remaining_capture_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    build_embedded_model(model_path, elements=4)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, input=np.arange(4, dtype=np.float32))
    output = tmp_path / "output"
    output.mkdir()
    response = tmp_path / "response.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "inputs": str(inputs),
                "output": str(output),
                "response": str(response),
                "targets": [
                    _target("input-id", "input", node_id=None),
                    _target("scaled-id", "scaled"),
                    _target("output-id", "output", graph_output=True),
                ],
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                "capture_bytes": 40,
                "chunk_bytes": 8,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    records = json.loads(response.read_text(encoding="utf-8"))["records"]
    by_name = {record["value_name"]: record for record in records}
    assert by_name["input"]["state"] == "complete"
    assert by_name["output"]["state"] == "complete"
    assert by_name["scaled"]["state"] == "truncated"
    assert by_name["scaled"]["stored_byte_length"] == 8


def test_worker_capture_states_are_explicit(tmp_path: Path) -> None:
    target = _target("value", "value")
    missing, remaining = trace_worker._write_capture(
        tmp_path,
        target,
        None,
        remaining=8,
        chunk_bytes=4,
        reason="unsupported operator",
    )
    assert missing["state"] == "dropped"
    assert missing["reason"] == "unsupported operator"
    assert remaining == 8

    array = np.arange(4, dtype=np.float32)
    dropped, remaining = trace_worker._write_capture(
        tmp_path,
        target,
        array,
        remaining=0,
        chunk_bytes=4,
        reason=None,
    )
    assert dropped["state"] == "dropped"
    assert remaining == 0

    truncated, remaining = trace_worker._write_capture(
        tmp_path,
        target,
        array,
        remaining=8,
        chunk_bytes=3,
        reason=None,
    )
    assert truncated["state"] == "truncated"
    assert truncated["stored_byte_length"] == 8
    assert remaining == 0


def test_worker_unsupported_operator_returns_inputs_and_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.onnx"
    build_custom_domain_model(path)
    model = onnx.load_model(path)
    feed = np.arange(4, dtype=np.float32)
    captures, diagnostics = trace_worker._evaluate(
        model,
        {"input": feed},
        [
            _target("input-id", "input", node_id=None),
            _target("output-id", "output"),
        ],
    )
    assert np.array_equal(captures["input-id"], feed)
    assert "output-id" not in captures
    assert diagnostics
    assert "degraded" in diagnostics[0]


def test_worker_prefix_fallback_restores_model_without_copying_weights() -> None:
    input_info = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [4]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [4]
    )
    scale = onnx.helper.make_tensor(
        "scale",
        onnx.TensorProto.FLOAT,
        [1],
        [2.0],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scale"], ["scaled"]),
                onnx.helper.make_node(
                    "Unsupported",
                    ["scaled"],
                    ["output"],
                    domain="nneditor.test",
                ),
            ],
            "prefix-fallback",
            [input_info],
            [output_info],
            [scale],
        ),
        opset_imports=[
            onnx.helper.make_operatorsetid("", 18),
            onnx.helper.make_operatorsetid("nneditor.test", 1),
        ],
    )
    before = model.SerializeToString()

    captures, diagnostics = trace_worker._evaluate(
        model,
        {"input": np.arange(4, dtype=np.float32)},
        [
            _target("scaled-id", "scaled"),
            _target("output-id", "output"),
        ],
    )

    assert np.array_equal(
        captures["scaled-id"],
        np.arange(4, dtype=np.float32) * 2,
    )
    assert "output-id" not in captures
    assert diagnostics
    assert model.SerializeToString() == before


def test_worker_memory_exhaustion_does_not_retry_every_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    build_embedded_model(model_path, elements=4)
    model = onnx.load_model(model_path)
    calls = 0

    class ExhaustedEvaluator:
        def __init__(self, _model: onnx.ModelProto) -> None:
            nonlocal calls
            calls += 1

        def run(
            self,
            _names: object,
            _feeds: object,
        ) -> list[object]:
            raise MemoryError("allocation failed")

    monkeypatch.setattr(trace_worker, "ReferenceEvaluator", ExhaustedEvaluator)
    captures, diagnostics = trace_worker._evaluate(
        model,
        {"input": np.arange(4, dtype=np.float32)},
        [
            _target("scaled-id", "scaled"),
            _target("output-id", "output"),
        ],
    )

    assert captures == {}
    assert calls == 1
    assert "increase the Memory limit" in diagnostics[0]


def test_worker_rejects_bad_requests_and_direct_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    with pytest.raises(ValueError, match="malformed"):
        trace_worker.run(request)

    monkeypatch.delenv("NNEDITOR_TRACE_WORKER")
    with pytest.raises(RuntimeError, match="launched by NNEditor"):
        trace_worker.run(request)

    monkeypatch.setattr(sys, "argv", ["trace_worker"])
    with pytest.raises(SystemExit, match="usage"):
        trace_worker.main()
