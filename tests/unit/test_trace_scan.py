"""ONNX Runtime, axis-aware Scan, and Scan-axis normalization coverage."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from nneditor.tracing.contracts import TraceBackend
from nneditor.tracing.scan import (
    AxisAwareScan,
    ScanNormalizationError,
    normalize_scan_axes,
)
from nneditor.tracing.trace_worker import _open_backend


def _scan_model(
    *,
    input_axis: int = 1,
    output_axis: int = 1,
    input_direction: int = 0,
    output_direction: int = 0,
    typed_series: bool = True,
) -> onnx.ModelProto:
    body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "x"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["y"]),
        ],
        "body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
        ],
    )
    series = (
        helper.make_tensor_value_info("series", TensorProto.FLOAT, [2, 3])
        if typed_series
        else onnx.ValueInfoProto(name="series")
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Scan",
                ["state", "series"],
                ["final", "ys"],
                name="axis-one-scan",
                body=body,
                num_scan_inputs=1,
                scan_input_axes=[input_axis],
                scan_input_directions=[input_direction],
                scan_output_axes=[output_axis],
                scan_output_directions=[output_direction],
            )
        ],
        "axis-one",
        [
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [2]),
            series,
        ],
        [
            helper.make_tensor_value_info("final", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("ys", TensorProto.FLOAT, [2, 3]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],
    )


def _feeds() -> dict[str, np.ndarray]:
    return {
        "state": np.zeros(2, dtype=np.float32),
        "series": np.array(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float32,
        ),
    }


def test_axis_aware_scan_and_normalized_model_match_expected_values() -> None:
    model = _scan_model()
    before = model.SerializeToString()
    with pytest.raises(RuntimeError, match="input axes"):
        ReferenceEvaluator(model)

    direct = cast(
        list[np.ndarray[Any, Any]],
        ReferenceEvaluator(model, new_ops=[AxisAwareScan]).run(None, _feeds()),
    )
    expected_final = np.array([6.0, 60.0], dtype=np.float32)
    expected_scan = np.array(
        [[1.0, 3.0, 6.0], [10.0, 30.0, 60.0]],
        dtype=np.float32,
    )
    assert np.array_equal(direct[0], expected_final)
    assert np.array_equal(direct[1], expected_scan)

    normalized, report = normalize_scan_axes(model)
    assert model.SerializeToString() == before
    assert report.rewritten_scans == 1
    assert report.inserted_transposes == 2
    assert [node.op_type for node in normalized.graph.node] == [
        "Transpose",
        "Scan",
        "Transpose",
    ]
    scan = normalized.graph.node[1]
    attributes = {attribute.name: attribute for attribute in scan.attribute}
    assert tuple(attributes["scan_input_axes"].ints) == (0,)
    assert tuple(attributes["scan_output_axes"].ints) == (0,)
    onnx.checker.check_model(normalized, full_check=True)
    rewritten = ReferenceEvaluator(normalized).run(None, _feeds())
    assert all(
        np.array_equal(left, right)
        for left, right in zip(direct, rewritten, strict=True)
    )


@pytest.mark.parametrize(
    "backend",
    [
        TraceBackend.AUTO,
        TraceBackend.ONNX_RUNTIME,
        TraceBackend.REFERENCE,
        TraceBackend.REFERENCE_NORMALIZED,
    ],
)
def test_every_trace_backend_executes_nonzero_scan_axes(
    backend: TraceBackend,
) -> None:
    targets: list[dict[str, object]] = [
        {"value_id": "final-id", "value_name": "final"},
        {"value_id": "ys-id", "value_name": "ys"},
    ]
    source, runtime = _open_backend(_scan_model(), _feeds(), targets, backend)
    final, final_error = source.take(targets[0])
    values, values_error = source.take(targets[1])

    assert final_error is None
    assert values_error is None
    assert final is not None and np.array_equal(
        final,
        np.array([6.0, 60.0], dtype=np.float32),
    )
    assert values is not None and np.array_equal(
        values,
        np.array([[1.0, 3.0, 6.0], [10.0, 30.0, 60.0]], dtype=np.float32),
    )
    assert "failed" not in runtime


def test_axis_aware_scan_supports_negative_axes_and_reverse_directions() -> None:
    model = _scan_model(
        input_axis=-1,
        output_axis=-1,
        input_direction=1,
        output_direction=1,
    )
    final, values = ReferenceEvaluator(model, new_ops=[AxisAwareScan]).run(
        None,
        _feeds(),
    )

    assert np.array_equal(final, np.array([6.0, 60.0], dtype=np.float32))
    assert np.array_equal(
        values,
        np.array([[6.0, 5.0, 3.0], [60.0, 50.0, 30.0]], dtype=np.float32),
    )


def test_scan_normalization_refuses_to_guess_an_unknown_rank() -> None:
    model = _scan_model(typed_series=False)
    with pytest.raises(ScanNormalizationError, match="rank is unknown"):
        normalize_scan_axes(model, check=False)
