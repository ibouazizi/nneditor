"""Structural consistency checks run during ONNX import."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.diagnostics import Diagnostic, Severity, describe
from tests.fixtures.onnx_models import build_embedded_model

_OPSET = 20


def _codes(path: Path) -> list[str]:
    document = index_to_document(index_model(path))
    return [item.code for item in document.diagnostics]


def _finding(path: Path, code: str) -> Diagnostic:
    document = index_to_document(index_model(path))
    matches = [item for item in document.diagnostics if item.code == code]
    assert matches, f"{code} not reported; got {[i.code for i in document.diagnostics]}"
    return matches[0]


def _save(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, path)
    return path


def _annotated_model(
    *,
    annotation_type: int,
    annotation_dims: list[int] | None,
    initializer_type: int = TensorProto.FLOAT,
    initializer_dims: list[int] | None = None,
) -> onnx.ModelProto:
    """A model whose value_info annotates its own initializer."""
    dims = [1] if initializer_dims is None else initializer_dims
    weight = helper.make_tensor(
        name="weight",
        data_type=initializer_type,
        dims=dims,
        vals=[1.0],
        raw=False,
    )
    graph = helper.make_graph(
        nodes=[helper.make_node("Mul", ["input", "weight"], ["output"], name="mul")],
        name="annotated",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=[weight],
        value_info=[
            helper.make_tensor_value_info("weight", annotation_type, annotation_dims)
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="nneditor"
    )
    model.producer_version = "test"
    return model


def test_type_annotation_conflict_is_reported_with_both_types(tmp_path: Path) -> None:
    """The defect that makes ONNX inference reject an otherwise fine model."""
    path = _save(
        _annotated_model(annotation_type=TensorProto.INT64, annotation_dims=[1]),
        tmp_path / "conflict.onnx",
    )
    finding = _finding(path, "onnx.type-annotation-conflict")
    assert finding.severity is Severity.WARNING
    # Both sides are named, so the reader knows which one to correct.
    assert "INT64" in finding.message
    assert "FLOAT" in finding.message
    assert "weight" in finding.message
    # And the consequence is stated rather than left to be discovered later.
    assert "export and tracing" in finding.message
    assert describe(finding.code).guidance.strip()

    # ONNX itself agrees this model is broken, which is the point of the check.
    with pytest.raises(Exception, match="elem type"):
        onnx.checker.check_model(onnx.load(path), full_check=True)


def test_matching_annotation_is_not_reported(tmp_path: Path) -> None:
    path = _save(
        _annotated_model(annotation_type=TensorProto.FLOAT, annotation_dims=[1]),
        tmp_path / "fine.onnx",
    )
    assert "onnx.type-annotation-conflict" not in _codes(path)
    assert "onnx.shape-annotation-conflict" not in _codes(path)
    onnx.checker.check_model(onnx.load(path), full_check=True)


def test_shape_annotation_conflict_is_reported(tmp_path: Path) -> None:
    path = _save(
        _annotated_model(annotation_type=TensorProto.FLOAT, annotation_dims=[1, 1]),
        tmp_path / "rank.onnx",
    )
    finding = _finding(path, "onnx.shape-annotation-conflict")
    assert "rank" in finding.message


def test_healthy_model_reports_no_structural_conflicts(tmp_path: Path) -> None:
    """A clean fixture must not trip any of the new checks."""
    path = tmp_path / "clean.onnx"
    build_embedded_model(path, elements=4)
    codes = set(_codes(path))
    assert not codes & {
        "onnx.type-annotation-conflict",
        "onnx.shape-annotation-conflict",
        "onnx.duplicate-initializer",
        "onnx.unresolved-input",
    }


def test_unresolved_input_is_an_error(tmp_path: Path) -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Mul", ["input", "ghost"], ["output"], name="mul")],
        name="dangling",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="nneditor"
    )
    model.producer_version = "test"
    path = _save(model, tmp_path / "dangling.onnx")
    finding = _finding(path, "onnx.unresolved-input")
    assert finding.severity is Severity.ERROR
    assert "ghost" in finding.message


def test_subgraph_reading_an_outer_value_is_not_reported(tmp_path: Path) -> None:
    """Control-flow bodies legitimately read from enclosing scopes.

    Resolving only local names would report every such body as incomplete.
    """
    body = helper.make_graph(
        nodes=[helper.make_node("Identity", ["outer"], ["branch_out"], name="pass")],
        name="then_body",
        inputs=[],
        outputs=[helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [1])],
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["input"], ["outer"], name="produce"),
            helper.make_node(
                "If",
                ["cond"],
                ["output"],
                name="branch",
                then_branch=body,
                else_branch=body,
            ),
        ],
        name="withsub",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="nneditor"
    )
    model.producer_version = "test"
    path = _save(model, tmp_path / "subgraph.onnx")
    assert "onnx.unresolved-input" not in _codes(path)
