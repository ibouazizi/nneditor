"""Direct coverage of the minimal subprocess trace entry point."""

from __future__ import annotations

import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import onnx
import pytest

from nneditor.tracing import trace_worker
from nneditor.tracing.contracts import TraceBackend, TraceDevice
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
    shape: list[int] | None = None,
    element_type: str = "float32",
) -> dict[str, object]:
    return {
        "value_id": value_id,
        "value_name": value_name,
        "node_id": node_id,
        "role": "node-output" if node_id else "graph-input",
        "graph_output": graph_output,
        "element_type": element_type,
        "shape": [4] if shape is None else shape,
    }


def _capture_all(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
    *,
    evaluator_factory: trace_worker.EvaluatorFactory | None = None,
) -> tuple[dict[str, np.ndarray[Any, Any]], list[str], dict[str, str]]:
    """Drain a capture source eagerly, the way the worker drains it lazily."""
    source = trace_worker.open_captures(
        model,
        feeds,
        targets,
        evaluator_factory=evaluator_factory,
    )
    captures: dict[str, np.ndarray[Any, Any]] = {}
    failures: dict[str, str] = {}
    for target in targets:
        array, reason = source.take(target)
        if array is not None:
            captures[str(target["value_id"])] = array
        elif reason is not None:
            failures[str(target["value_name"])] = reason
    return captures, source.diagnostics, failures


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
    assert payload["runtime"].startswith("onnxruntime")
    assert payload["execution_device"] == "cpu"
    assert payload["execution_provider"] == "CPUExecutionProvider"
    assert payload["notes"] == []
    assert [record["state"] for record in payload["records"]] == [
        "complete",
        "complete",
        "complete",
    ]
    assert all(
        (output / record["file_name"]).stat().st_size == record["stored_byte_length"]
        for record in payload["records"]
    )


def test_gpu_provider_candidates_are_strict_and_prioritized() -> None:
    candidates = trace_worker._provider_candidates(
        [
            "CPUExecutionProvider",
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "OpenVINOExecutionProvider",
        ],
        TraceDevice.GPU,
    )

    assert [candidate.label for candidate in candidates] == [
        "TensorRT / CUDA",
        "CUDA",
        "DirectML",
        "OpenVINO GPU",
    ]
    assert all(candidate.device is TraceDevice.GPU for candidate in candidates)
    assert all(
        "CPUExecutionProvider" not in candidate.providers for candidate in candidates
    )
    assert candidates[-1].provider_options == ({"device_type": "GPU"},)


def test_npu_provider_candidates_apply_provider_specific_options() -> None:
    candidates = trace_worker._provider_candidates(
        [
            "CPUExecutionProvider",
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
        ],
        TraceDevice.NPU,
    )

    assert [candidate.label for candidate in candidates] == [
        "OpenVINO NPU",
        "Qualcomm QNN HTP",
        "AMD Vitis AI",
    ]
    assert candidates[0].provider_options == ({"device_type": "NPU"},)
    assert candidates[1].provider_options == ({"backend_type": "htp"},)
    assert all(candidate.device is TraceDevice.NPU for candidate in candidates)


def test_auto_device_falls_back_to_cpu_only_when_it_is_installed() -> None:
    candidates = trace_worker._provider_candidates(
        ["AzureExecutionProvider", "CPUExecutionProvider"],
        TraceDevice.AUTO,
    )

    assert len(candidates) == 1
    assert candidates[0].device is TraceDevice.CPU
    assert candidates[0].provider == "CPUExecutionProvider"


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


def test_worker_does_not_truncate_known_captures_when_the_total_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_info = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [100],
    )
    output_info = onnx.helper.make_tensor_value_info(
        "scalar19",
        onnx.TensorProto.FLOAT,
        [],
    )
    index = onnx.helper.make_tensor(
        "index",
        onnx.TensorProto.INT64,
        [],
        [0],
    )
    nodes = [
        onnx.helper.make_node("Identity", ["input"], ["big"]),
        onnx.helper.make_node("Gather", ["input", "index"], ["scalar0"], axis=0),
        *[
            onnx.helper.make_node(
                "Identity",
                [f"scalar{position - 1}"],
                [f"scalar{position}"],
            )
            for position in range(1, 20)
        ],
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "known-sizes",
            [input_info],
            [output_info],
            [index],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, input=np.arange(100, dtype=np.float32))
    output = tmp_path / "output"
    output.mkdir()
    response = tmp_path / "response.json"
    request = tmp_path / "request.json"
    # Declared shapes must be truthful: the lazily fetching runtime source
    # budgets from them before any batch has materialized.
    targets = [
        _target("input-id", "input", node_id=None, shape=[100]),
        _target("big-id", "big", shape=[100]),
        *[
            _target(f"scalar{position}-id", f"scalar{position}", shape=[])
            for position in range(20)
        ],
    ]
    request.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "inputs": str(inputs),
                "output": str(output),
                "response": str(response),
                "targets": targets,
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                # input + big + twenty scalars = 400 + 400 + 20 * 4
                "capture_bytes": 880,
                "chunk_bytes": 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    records = json.loads(response.read_text(encoding="utf-8"))["records"]
    assert {record["state"] for record in records} == {"complete"}
    assert sum(record["stored_byte_length"] for record in records) == 880


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
    captures, diagnostics, _failures = _capture_all(
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

    captures, diagnostics, _failures = _capture_all(
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
        def __init__(self, _model: onnx.ModelProto, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1

        def run(
            self,
            _names: object,
            _feeds: object,
        ) -> list[object]:
            raise MemoryError("allocation failed")

    monkeypatch.setattr(trace_worker, "ReferenceEvaluator", ExhaustedEvaluator)
    captures, diagnostics, _failures = _capture_all(
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

    captures, diagnostics, failures = _capture_all(
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


def test_axis_aware_reference_runtime_executes_scan_axis_one(
    tmp_path: Path,
) -> None:
    """The reference backend supports the xLSTM export's nonzero Scan axis."""
    body = onnx.helper.make_graph(
        nodes=[onnx.helper.make_node("Identity", ["state_in"], ["state_out"])],
        name="scan_body",
        inputs=[
            onnx.helper.make_tensor_value_info("state_in", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1]),
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

    captures, diagnostics, failures = _capture_all(
        model, feeds, [_target("final-id", "final")]
    )

    assert np.array_equal(captures["final-id"], np.zeros((1,), dtype=np.float32))
    assert diagnostics == []
    assert failures == {}


def test_prefix_capture_stops_once_memory_is_exhausted() -> None:
    """One memory failure ends the pass; it cannot recover by trying again."""
    calls: list[str] = []

    def exploding(
        model: onnx.ModelProto,
        feeds: dict[str, np.ndarray[Any, Any]],
        name: str,
        *,
        producer_limit: int,
        evaluator_factory: trace_worker.EvaluatorFactory | None = None,
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

    source = trace_worker.CaptureSource(
        model, {}, None, [], len(model.graph.node), None
    )
    original = trace_worker._evaluate_prefix
    trace_worker._evaluate_prefix = exploding
    try:
        results = [source.take(target) for target in targets]
    finally:
        trace_worker._evaluate_prefix = original

    assert all(array is None for array, _reason in results)
    # Only the first value was attempted; the rest inherit the same reason.
    assert calls == ["v0"]
    reasons = [reason for _array, reason in results]
    assert all(reason is not None and "memory limit" in reason for reason in reasons)
    assert sum("ran out of memory" in item for item in source.diagnostics) == 1


def _scan_axis_one_model(leading: int) -> onnx.ModelProto:
    """`leading` buildable Identity nodes, then a Scan the runtime refuses."""
    body = onnx.helper.make_graph(
        nodes=[onnx.helper.make_node("Identity", ["s_in"], ["s_out"])],
        name="body",
        inputs=[
            onnx.helper.make_tensor_value_info("s_in", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("x_in", onnx.TensorProto.FLOAT, [1]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("s_out", onnx.TensorProto.FLOAT, [1])
        ],
    )
    nodes = [
        onnx.helper.make_node("Identity", ["x" if i == 0 else f"v{i - 1}"], [f"v{i}"])
        for i in range(leading)
    ]
    nodes.append(
        onnx.helper.make_node(
            "Scan",
            ["state", "series"],
            ["final"],
            body=body,
            num_scan_inputs=1,
            scan_input_axes=[1],
        )
    )
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="frontier",
        inputs=[
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info(
                "series", onnx.TensorProto.FLOAT, [1, 2]
            ),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("final", onnx.TensorProto.FLOAT, [1])
        ],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 20)]
    )


def test_buildable_frontier_finds_the_first_refused_node() -> None:
    """The frontier is found by binary search, not by probing every node."""
    model = _scan_axis_one_model(leading=6)
    assert trace_worker._buildable_frontier(model) == 7
    assert (
        trace_worker._buildable_frontier(
            model,
            trace_worker._standard_reference_evaluator,
        )
        == 6
    )
    # Probing must leave the model exactly as it was.
    assert len(model.graph.node) == 7
    assert model.graph.node[6].op_type == "Scan"

    # A model the runtime accepts end to end has no frontier.
    clean = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("Identity", ["x"], ["y"])],
            name="clean",
            inputs=[
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    assert trace_worker._buildable_frontier(clean) == 1


def test_values_beyond_the_frontier_are_reported_without_being_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuilding an evaluator for a value that cannot build only wastes time."""
    model = _scan_axis_one_model(leading=3)
    feeds = {
        "x": np.zeros((1,), dtype=np.float32),
        "state": np.zeros((1,), dtype=np.float32),
        "series": np.zeros((1, 2), dtype=np.float32),
    }
    targets = [
        _target("v0-id", "v0"),
        _target("v2-id", "v2"),
        _target("final-id", "final", graph_output=True),
    ]

    attempted: list[str] = []
    original = trace_worker._evaluate_prefix

    def counting(
        model_: onnx.ModelProto,
        feeds_: dict[str, np.ndarray[Any, Any]],
        name: str,
        *,
        producer_limit: int,
        evaluator_factory: trace_worker.EvaluatorFactory | None = None,
    ) -> np.ndarray[Any, Any]:
        attempted.append(name)
        return original(
            model_,
            feeds_,
            name,
            producer_limit=producer_limit,
            evaluator_factory=evaluator_factory,
        )

    monkeypatch.setattr(trace_worker, "_evaluate_prefix", counting)
    captures, diagnostics, failures = _capture_all(
        model,
        feeds,
        targets,
        evaluator_factory=trace_worker._standard_reference_evaluator,
    )

    # Values before the Scan are captured; the one that needs it is not
    # attempted at all, and says why.
    assert attempted == ["v0", "v2"]
    assert set(captures) == {"v0-id", "v2-id"}
    assert "Scan" in failures["final"]
    assert "unavailable" in failures["final"]
    assert any("cannot build Scan" in item for item in diagnostics)


def test_missing_onnxruntime_auto_falls_back_with_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    model = onnx.load_model(str(path))
    feeds = {"input": np.ones(4, dtype=np.float32)}
    targets = [_target("val:scaled", "scaled")]
    # A None entry makes `import onnxruntime` raise ImportError, exactly as
    # in an install without any runtime extra.
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, None))

    source, runtime, device, provider = trace_worker._open_backend(
        model,
        path,
        feeds,
        targets,
        TraceBackend.AUTO,
    )

    assert provider == "ReferenceEvaluator"
    assert device is TraceDevice.CPU
    assert runtime.startswith("onnx.reference")
    assert "nneditor[runtime]" in source.diagnostics[0]


def test_missing_onnxruntime_gpu_request_names_accelerator_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    model = onnx.load_model(str(path))
    feeds = {"input": np.ones(4, dtype=np.float32)}
    targets = [_target("val:scaled", "scaled")]
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, None))

    source, runtime, device, provider = trace_worker._open_backend(
        model,
        path,
        feeds,
        targets,
        TraceBackend.AUTO,
        TraceDevice.GPU,
    )

    assert provider == "Unavailable"
    assert device is TraceDevice.GPU
    assert runtime == "onnxruntime accelerator unavailable"
    assert "nneditor[runtime-gpu]" in source.diagnostics[0]
    assert "nneditor[runtime-directml]" in source.diagnostics[0]


def test_onnxruntime_reads_a_staged_capture_file_not_serialized_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    model = onnx.load_model(str(path))
    original_bytes = path.read_bytes()
    feeds = {"input": np.ones(4, dtype=np.float32)}
    targets = [
        _target("val:input", "input", node_id=None),
        _target("val:scaled", "scaled"),
        _target("val:output", "output", graph_output=True),
    ]
    observed: dict[str, Any] = {}

    class FakeSessionOptions:
        def __init__(self) -> None:
            self.enable_cpu_mem_arena = True
            self.enable_mem_pattern = True
            self.execution_mode: Any = None
            self.graph_optimization_level: Any = None
            self.inter_op_num_threads = 0
            self.intra_op_num_threads = 0
            self.log_severity_level = 0

        def add_session_config_entry(self, key: str, value: str) -> None:
            observed.setdefault("config", {})[key] = value

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            # The whole point of the staged file: the runtime receives a
            # path, never a serialized copy of the loaded model.
            assert isinstance(model_source, str)
            staged = Path(model_source)
            observed["staged_path"] = staged
            observed["staged_exists"] = staged.exists()
            loaded = onnx.load_model(model_source)
            observed["staged_outputs"] = [value.name for value in loaded.graph.output]

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            observed["run_names"] = list(names)
            return [np.full(4, 7.0, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))

    source, runtime, device, provider = trace_worker._open_backend(
        model,
        path,
        feeds,
        targets,
        TraceBackend.ONNX_RUNTIME,
    )

    assert observed["staged_exists"] is True
    assert observed["staged_path"].name == "model.onnx.capture.onnx"
    # The staged file was augmented with the intermediate value; the
    # in-memory model and the source artifact were left untouched.
    assert "scaled" in observed["staged_outputs"]
    assert [value.name for value in model.graph.output] == ["output"]
    assert path.read_bytes() == original_bytes
    assert provider == "CPUExecutionProvider"
    assert device is TraceDevice.CPU
    assert "0-test" in runtime
    # Fetches are lazy: nothing has run yet and the session still maps the
    # staged file, so it must survive until the captures are spent.
    assert "run_names" not in observed
    assert observed["staged_path"].exists()
    array, reason = source.take(targets[1])
    assert reason is None
    assert array is not None
    assert float(array[0]) == 7.0
    # The batch was fetched in the worker's consumption order — the graph
    # output ahead of the ordinary intermediate — and the in-graph input is
    # served from feeds, never fetched from the runtime.
    assert observed["run_names"] == ["output", "scaled"]
    assert observed["staged_path"].exists()
    out_array, out_reason = source.take(targets[2])
    assert out_reason is None
    assert out_array is not None
    # Every batched target was taken, so the staged copies are now spent
    # and deleted.
    assert not observed["staged_path"].exists()


def _fake_session_options() -> type:
    class FakeSessionOptions:
        def __init__(self) -> None:
            self.enable_cpu_mem_arena = True
            self.enable_mem_pattern = True
            self.execution_mode: Any = None
            self.graph_optimization_level: Any = None
            self.inter_op_num_threads = 0
            self.intra_op_num_threads = 0
            self.log_severity_level = 0

        def add_session_config_entry(self, key: str, value: str) -> None:
            pass

    return FakeSessionOptions


def test_nvidia_dlls_are_preloaded_and_skipped_providers_disclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA wheels live in site-packages, so the worker must ask ONNX
    Runtime to preload them; a candidate that still fails must stay
    visible in the notes of the candidate that finally served — as
    context, not as a diagnostic that would mark a complete trace
    partial."""
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    model = onnx.load_model(str(path))
    feeds = {"input": np.ones(4, dtype=np.float32)}
    targets = [_target("val:scaled", "scaled")]
    calls: list[str] = []

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            if "CUDAExecutionProvider" in (providers or []):
                raise RuntimeError(
                    "[ONNXRuntimeError] : 1 : FAIL : This session contains "
                    "graph nodes that are assigned to the default CPU EP, "
                    "but fallback to CPU EP has been explicitly disabled "
                    "by the user."
                )

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            return [np.zeros(4, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        preload_dlls=lambda: calls.append("preload"),
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))

    source, _runtime, device, provider = trace_worker._open_backend(
        model,
        path,
        feeds,
        targets,
        TraceBackend.ONNX_RUNTIME,
    )

    assert calls == ["preload"]
    assert provider == "CPUExecutionProvider"
    assert device is TraceDevice.CPU
    # Missing accelerators are the normal state of most machines: the
    # disclosure must not ride the diagnostics channel, which would make
    # every complete CPU trace claim to be partial.
    assert source.diagnostics == []
    skipped = source.notes[0]
    assert "skipped" in skipped
    assert "CUDA" in skipped
    assert "native libraries likely failed to load" in skipped


def test_batches_follow_consumption_order_and_bound_estimated_bytes() -> None:
    """Fetch planning mirrors the run loop's priority phases exactly."""
    out = _target("out-id", "out", graph_output=True)
    a = _target("a-id", "a")
    b = _target("b-id", "b")
    c = _target("c-id", "c")
    sym = _target("sym-id", "sym", shape=[-1])
    ordered = trace_worker._capture_order([a, b, out, c, sym])

    # The graph output moves ahead of ordinary values; order is stable
    # within each phase, so budget exhaustion drops the same trailing
    # values a single fetch would have dropped.
    assert [target["value_id"] for target in ordered] == [
        "out-id",
        "a-id",
        "b-id",
        "c-id",
        "sym-id",
    ]

    # 16-byte float32 targets against a 32-byte ceiling: two per batch, and
    # the symbolic-dim target is isolated because its size is unknowable.
    batches = trace_worker._plan_batches(ordered, 32)
    assert [[target["value_id"] for target in batch] for batch in batches] == [
        ["out-id", "a-id"],
        ["b-id", "c-id"],
        ["sym-id"],
    ]

    # A single target beyond the ceiling still forms its own batch, and an
    # undecodable dtype is isolated like a symbolic dimension.
    oversized = _target("big-id", "big", shape=[64])
    unknown = _target("odd-id", "odd", element_type="unknown")
    assert trace_worker._plan_batches([oversized, a], 32) == [
        (oversized,),
        (a,),
    ]
    assert trace_worker._plan_batches([a, unknown, b], 64) == [
        (a,),
        (unknown,),
        (b,),
    ]


class _RecordingSession:
    """A fake ONNX Runtime session that records and answers each fetch."""

    def __init__(self, failing_names: frozenset[str] = frozenset()) -> None:
        self.calls: list[list[str]] = []
        self._failing_names = failing_names

    def run(
        self,
        names: list[str],
        feeds: dict[str, Any],
    ) -> list[Any]:
        self.calls.append(list(names))
        if self._failing_names & set(names):
            raise RuntimeError("bad allocation")
        return [np.full(4, float(len(name)), dtype=np.float32) for name in names]


def _staged_capture(tmp_path: Path) -> Path:
    capture = tmp_path / "model.onnx.capture.onnx"
    capture.write_bytes(b"staged")
    capture.with_name(capture.name + ".data").write_bytes(b"weights")
    return capture


def test_ort_capture_source_fetches_lazily_and_releases_per_batch(
    tmp_path: Path,
) -> None:
    targets = [_target(f"v{index}-id", f"v{index}") for index in range(4)]
    session = _RecordingSession()
    capture = _staged_capture(tmp_path)
    source = trace_worker.OrtCaptureSource(
        session,
        {},
        trace_worker._plan_batches(targets, 32),
        capture,
    )

    # Nothing runs until the first take, and a take from the first batch
    # does not touch the second.
    assert session.calls == []
    first, reason = source.take(targets[0])
    assert reason is None
    assert first is not None
    assert session.calls == [["v0", "v1"]]

    # Arrays are dropped as they are taken: once the caller releases its
    # reference nothing in the source keeps the activation alive.
    released = weakref.ref(first)
    del first
    assert released() is None

    second, _reason = source.take(targets[1])
    assert second is not None
    assert 0 not in source._materialized

    source.take(targets[2])
    assert session.calls == [["v0", "v1"], ["v2", "v3"]]
    assert capture.exists()
    source.take(targets[3])
    # Every batch is spent: the session and the staged copies are released.
    assert source._session is None
    assert not capture.exists()
    assert not capture.with_name(capture.name + ".data").exists()


def test_failed_batch_marks_only_its_own_targets(tmp_path: Path) -> None:
    targets = [_target(f"v{index}-id", f"v{index}") for index in range(4)]
    session = _RecordingSession(failing_names=frozenset({"v2"}))
    capture = _staged_capture(tmp_path)
    source = trace_worker.OrtCaptureSource(
        session,
        {},
        trace_worker._plan_batches(targets, 32),
        capture,
    )

    served = [source.take(target) for target in targets[:2]]
    assert all(array is not None for array, _reason in served)

    # The failed fetch degrades only its own batch, with one diagnostic and
    # a per-target reason — the trace itself survives.
    failed = [source.take(target) for target in targets[2:]]
    assert all(array is None for array, _reason in failed)
    assert all(
        reason is not None and "not captured" in reason for _array, reason in failed
    )
    assert session.calls == [["v0", "v1"], ["v2", "v3"]]
    assert len(source.diagnostics) == 1
    assert "bad allocation" in source.diagnostics[0]
    assert "other fetches were still attempted" in source.diagnostics[0]
    # Exhaustion still releases the staged file even after a failure.
    assert not capture.exists()


def test_open_backend_fetches_in_multiple_batches_under_a_small_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    model = onnx.load_model(str(path))
    feeds = {"input": np.ones(4, dtype=np.float32)}
    targets = [
        _target("val:input", "input", node_id=None),
        _target("val:scaled", "scaled"),
        _target("val:output", "output", graph_output=True),
    ]
    calls: list[list[str]] = []

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            pass

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            calls.append(list(names))
            return [np.zeros(4, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))
    # A ceiling below one 16-byte tensor forces one fetch per target.
    monkeypatch.setattr(trace_worker, "_ORT_BATCH_BYTES", 8)

    source, _runtime, _device, _provider = trace_worker._open_backend(
        model,
        path,
        feeds,
        targets,
        TraceBackend.ONNX_RUNTIME,
    )
    for target in (targets[2], targets[1]):
        array, reason = source.take(target)
        assert reason is None
        assert array is not None

    assert calls == [["output"], ["scaled"]]


def test_worker_response_moves_skipped_provider_disclosure_to_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full worker entry point reports notes beside clean diagnostics."""
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
                "backend": "onnxruntime",
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                "capture_bytes": 1024,
                "chunk_bytes": 64,
            }
        ),
        encoding="utf-8",
    )

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            if "CUDAExecutionProvider" in (providers or []):
                raise RuntimeError("CUDA libraries could not be loaded")

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            return [np.arange(4, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["diagnostics"] == []
    assert len(payload["notes"]) == 1
    assert "skipped" in payload["notes"][0]
    assert "CUDA" in payload["notes"][0]
    assert [record["state"] for record in payload["records"]] == [
        "complete",
        "complete",
        "complete",
    ]
    # The worker released the staged capture copies before exiting.
    assert not model_path.with_name("model.onnx.capture.onnx").exists()


def test_spent_capture_budget_skips_runtime_fetches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values past the byte ceiling are dropped without a runtime pass.

    On a large model the capture budget covers a fraction of the declared
    activations; fetching the rest anyway is what made a full-graph trace
    pay for ~8 GB of transfers to keep 256 MiB.
    """
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
                "backend": "onnxruntime",
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                # Exactly the input plus the prioritized output: the pool is
                # spent before the ordinary 'scaled' value is reached.
                "capture_bytes": 32,
                "chunk_bytes": 16,
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            pass

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            calls.append(list(names))
            return [np.arange(4, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))
    # One 16-byte tensor per batch, so an unfetched batch is observable.
    monkeypatch.setattr(trace_worker, "_ORT_BATCH_BYTES", 8)
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    payload = json.loads(response.read_text(encoding="utf-8"))
    by_name = {record["value_name"]: record for record in payload["records"]}
    # Only the prioritized output's batch ever reached the runtime.
    assert calls == [["output"]]
    assert by_name["input"]["state"] == "complete"
    assert by_name["output"]["state"] == "complete"
    assert by_name["scaled"]["state"] == "dropped"
    assert "exhausted" in by_name["scaled"]["reason"]
    # Skipping the last batch still released the staged capture copies.
    assert not model_path.with_name("model.onnx.capture.onnx").exists()


def test_batched_budgets_are_greedy_whole_values_not_equal_shares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batched source spends the pool on whole values in consume order.

    Equal shares would fetch every batch to keep a sliver of each value —
    on a large model that pays full runtime passes for activations the
    budget can never hold.
    """
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
                "backend": "onnxruntime",
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                # Eight bytes past the boundaries: greedy budgeting gives the
                # first ordinary value the whole remainder as a prefix.
                "capture_bytes": 40,
                "chunk_bytes": 8,
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            pass

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            calls.append(list(names))
            return [np.arange(4, dtype=np.float32) for _ in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))
    monkeypatch.setattr(trace_worker, "_ORT_BATCH_BYTES", 8)
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    payload = json.loads(response.read_text(encoding="utf-8"))
    by_name = {record["value_name"]: record for record in payload["records"]}
    assert calls == [["output"], ["scaled"]]
    assert by_name["output"]["state"] == "complete"
    assert by_name["scaled"]["state"] == "truncated"
    assert by_name["scaled"]["stored_byte_length"] == 8


def test_preview_policy_stages_everything_and_shares_the_budget_equally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under "preview", breadth wins: no value is pruned from the staged
    outputs and every ordinary value keeps an equal share of the pool.

    The same request under greedy budgeting would drop the trailing value
    from the staged model entirely and spend the whole remainder on the
    first ordinary value; here both keep a 4-byte sliver instead, boundaries
    still complete via the exact-budget path, and every batch is fetched.
    """
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
                    _target("a-id", "a", shape=[8]),
                    _target("b-id", "b"),
                    _target("output-id", "output", graph_output=True),
                ],
                "backend": "onnxruntime",
                "capture_policy": "preview",
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                # The prioritized input and output fit exactly (32 bytes);
                # the ordinary 'a' (32 B) and 'b' (16 B) share the last 8.
                "capture_bytes": 40,
                "chunk_bytes": 4,
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    element_counts = {"output": 4, "a": 8, "b": 4}
    observed: dict[str, Any] = {}

    class FakeInferenceSession:
        def __init__(
            self,
            model_source: object,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
        ) -> None:
            assert isinstance(model_source, str)
            loaded = onnx.load_model(model_source, load_external_data=False)
            observed["staged_outputs"] = [value.name for value in loaded.graph.output]

        def run(
            self,
            names: list[str],
            run_feeds: dict[str, Any],
        ) -> list[Any]:
            calls.append(list(names))
            return [np.arange(element_counts[name], dtype=np.float32) for name in names]

    fake = SimpleNamespace(
        SessionOptions=_fake_session_options(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=FakeInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        __version__="0-test",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, fake))
    monkeypatch.setattr(trace_worker, "_ORT_BATCH_BYTES", 8)
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    trace_worker.run(request)

    payload = json.loads(response.read_text(encoding="utf-8"))
    by_name = {record["value_name"]: record for record in payload["records"]}
    # A greedy 40-byte budget would prune 'b' from the staged outputs
    # (output + a already exceed it); preview stages every requested value.
    assert set(observed["staged_outputs"]) >= {"output", "a", "b"}
    # Every batch was fetched — no runtime pass was skipped for breadth.
    assert calls == [["output"], ["a"], ["b"]]
    assert by_name["input"]["state"] == "complete"
    assert by_name["output"]["state"] == "complete"
    # Both ordinary values keep an equal truncated share; under greedy 'a'
    # would take the whole 8-byte remainder and 'b' would be dropped.
    assert by_name["a"]["state"] == "truncated"
    assert by_name["a"]["stored_byte_length"] == 4
    assert by_name["b"]["state"] == "truncated"
    assert by_name["b"]["stored_byte_length"] == 4
    assert all(record["state"] != "dropped" for record in payload["records"])


def test_worker_rejects_unknown_capture_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    build_embedded_model(model_path, elements=4)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, input=np.arange(4, dtype=np.float32))
    output = tmp_path / "output"
    output.mkdir()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "inputs": str(inputs),
                "output": str(output),
                "response": str(tmp_path / "response.json"),
                "targets": [_target("scaled-id", "scaled")],
                "capture_policy": "eager",
                "wall_seconds": 10,
                "memory_bytes": 1024 * 1024 * 1024,
                "capture_bytes": 1024,
                "chunk_bytes": 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(trace_worker, "_limit_process", lambda raw: None)

    with pytest.raises(ValueError, match="capture policy"):
        trace_worker.run(request)
