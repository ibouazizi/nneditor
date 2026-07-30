"""Direct coverage of the minimal subprocess trace entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    captures, diagnostics, _failures = trace_worker._evaluate(
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
    # The first diagnostic explains that whole-model evaluation gave way to
    # per-value capture, and names the operator responsible.
    assert "capturing per value" in diagnostics[0]
    assert "FusedGelu" in diagnostics[0]


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

    captures, diagnostics, _failures = trace_worker._evaluate(
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
    captures, diagnostics, _failures = trace_worker._evaluate(
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


def test_evaluate_attributes_failures_to_the_value_that_failed(
    tmp_path: Path,
) -> None:
    """A real value must not inherit a diagnostic that merely mentions it.

    Failures used to be matched by searching diagnostic text for the value's
    repr, so the generic "full trace degraded" message — which embeds the
    runtime's own error text — could be blamed on an unrelated value.
    """
    model_path = tmp_path / "model.onnx"
    build_embedded_model(model_path, elements=4)
    model = onnx.load(model_path)
    feeds = {"input": np.arange(4, dtype=np.float32)}

    captures, diagnostics, failures = trace_worker._evaluate(
        model,
        feeds,
        [
            _target("ghost-id", "ghost"),
            _target("scaled-id", "scaled"),
        ],
    )

    # Requesting a value no node produces degrades the whole run to the
    # per-value fallback, whose generic diagnostic names 'ghost'.
    assert any("degraded" in diagnostic for diagnostic in diagnostics)
    assert "ghost" in failures
    assert "scaled" not in failures
    assert "scaled-id" in captures


def test_unbuildable_runtime_still_captures_and_names_the_real_cause(
    tmp_path: Path,
) -> None:
    """An operator the runtime cannot build must name itself, not MemoryError.

    Reproduces a real xLSTM export whose Scan uses input axis 1, which the
    reference runtime refuses outright.
    """
    body = onnx.helper.make_graph(
        nodes=[onnx.helper.make_node("Identity", ["state_in"], ["state_out"])],
        name="scan_body",
        inputs=[
            onnx.helper.make_tensor_value_info("state_in", onnx.TensorProto.FLOAT, [1])
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("state_out", onnx.TensorProto.FLOAT, [1])
        ],
    )
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "Scan",
                ["state", "series"],
                ["final"],
                name="scan",
                body=body,
                num_scan_inputs=1,
                scan_input_axes=[1],
            )
        ],
        name="unbuildable",
        inputs=[
            onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info(
                "series", onnx.TensorProto.FLOAT, [1, 4]
            ),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("final", onnx.TensorProto.FLOAT, [1])
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 20)]
    )
    feeds = {
        "state": np.zeros((1,), dtype=np.float32),
        "series": np.zeros((1, 4), dtype=np.float32),
    }

    _captures, diagnostics, failures = trace_worker._evaluate(
        model, feeds, [_target("final-id", "final")]
    )

    joined = " ".join(diagnostics)
    assert "cannot execute the whole model" in joined
    assert "Scan" in joined
    # The reason the model will not run must survive into the value's own
    # failure text rather than being replaced by a memory error.
    assert "Scan" in failures["final"] or "not captured" in failures["final"]
    assert not any("MemoryError" in item for item in diagnostics)


def test_prefix_capture_stops_once_memory_is_exhausted() -> None:
    """One memory failure ends the pass; it cannot recover by trying again."""
    calls: list[str] = []

    def exploding(
        model: onnx.ModelProto,
        feeds: dict[str, np.ndarray[Any, Any]],
        name: str,
        *,
        producer_limit: int,
    ) -> np.ndarray[Any, Any]:
        calls.append(name)
        raise MemoryError("cap reached")

    targets = [_target(f"v{i}-id", f"v{i}") for i in range(5)]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("Identity", ["x"], [f"v{i}"], name=f"n{i}")
                for i in range(5)
            ],
            name="chain",
            inputs=[
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("v0", onnx.TensorProto.FLOAT, [1])
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )

    original = trace_worker._evaluate_prefix
    trace_worker._evaluate_prefix = exploding
    try:
        captures, diagnostics, failures = trace_worker._evaluate_by_prefix(
            model, {}, targets, {}, [], {}
        )
    finally:
        trace_worker._evaluate_prefix = original

    assert not captures
    # Only the first value was attempted; the rest inherit the same reason.
    assert calls == ["v0"]
    assert len(failures) == 5
    assert all("memory limit" in reason for reason in failures.values())
    assert sum("ran out of memory" in item for item in diagnostics) == 1
