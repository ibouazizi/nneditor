"""Tests for the shell's presentation logic (P1.8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.ir.core import Document
from nneditor.transformations.engine import (
    GraphQuantizationRequest,
    LogicalPruningRequest,
    StructuredPruningRequest,
    WeightQuantizationRequest,
)
from nneditor.transformations.schema import Granularity, PruningMode
from nneditor.ui.viewmodel import (
    capability_lines,
    compact_bytes,
    diagnostic_lines,
    graph_entries,
    humanize_identifier,
    model_summary,
    node_details,
    transformation_request,
)
from tests.fixtures.onnx_models import (
    build_control_flow_model,
    build_custom_domain_model,
    build_embedded_model,
    build_function_model,
    build_optional_input_model,
)


def open_document(tmp_path: Path, builder: object) -> Document:
    path = tmp_path / "model.onnx"
    builder(path)  # type: ignore[operator]
    return index_to_document(index_model(path))


def test_graph_entries_nest_subgraphs(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_control_flow_model)
    entries = graph_entries(document)
    assert entries[0].graph_id == document.entry_graph
    assert entries[0].depth == 0
    children = [entry for entry in entries if entry.depth == 1]
    assert len(children) == 2, "then and else branches nest under main"
    assert {entry.node_count for entry in children} == {1}


def test_graph_entries_include_function_bodies(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_function_model)
    entries = graph_entries(document)
    labels = [entry.label for entry in entries]
    assert "Scale" in labels


def test_model_summary_lines(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    document = index_to_document(index_model(path))
    summary = dict(model_summary(document))
    assert summary["Kind"] == "onnx_model"
    assert summary["Graphs"] == "1"
    assert summary["Nodes"] == "2"
    assert summary["Tensors"] == "2"
    assert summary["Producer"].startswith("nneditor")
    assert summary["Content hash"].startswith("sha256:")


def test_capability_lines_cover_all_seven(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_custom_domain_model)
    lines = capability_lines(document)
    assert len(lines) == 7
    assert all(reason for _name, _availability, reason in lines)


def test_diagnostic_lines_use_registry_titles(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_custom_domain_model)
    lines = diagnostic_lines(document)
    assert any(title == "Operator from a custom domain" for _sev, title, _msg in lines)


def test_node_details_for_a_plain_node(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    document = index_to_document(index_model(path))
    node = document.main_graph.nodes[0]
    details = dict(node_details(document, document.entry_graph, node.id))
    assert details["Operator"] == "Mul"
    assert details["Name"] == "scale"
    assert "Input 0" in details and "Output 0" in details
    assert "Weight weight" in details
    assert "embedded" in details["Weight weight"]


def test_node_details_show_attributes_and_notes(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_custom_domain_model)
    node = document.main_graph.nodes[0]
    details = node_details(document, document.entry_graph, node.id)
    keyed = dict(details)
    assert keyed["Operator"] == "com.example.ops::FusedGelu"
    assert keyed["Attr approximation"] == "tanh"
    assert any("editing" in key for key, _value in details), (
        "the capability note is surfaced on the node"
    )


def test_node_details_mark_omitted_inputs(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_optional_input_model)
    node = document.main_graph.nodes[0]
    details = dict(node_details(document, document.entry_graph, node.id))
    assert details["Input 1"].startswith("(omitted)")


def test_node_details_for_control_flow_list_subgraphs(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_control_flow_model)
    node = document.main_graph.nodes[0]
    details = node_details(document, document.entry_graph, node.id)
    subgraph_rows = [value for key, value in details if key == "Subgraph"]
    assert len(subgraph_rows) == 2


def test_unknown_node_raises(tmp_path: Path) -> None:
    document = open_document(tmp_path, build_control_flow_model)
    with pytest.raises(KeyError):
        node_details(document, document.entry_graph, "ghost")


def test_overview_formatting_is_headless_and_consistent() -> None:
    assert humanize_identifier("pytorch_exported_program") == (
        "PyTorch Exported Program"
    )
    assert compact_bytes(999) == "999 B"
    assert compact_bytes(1024) == "1 KiB"
    assert compact_bytes(1536) == "1.5 KiB"


@pytest.mark.parametrize(
    ("kind", "parameter", "expected_type"),
    (
        ("weight-quantization", "", WeightQuantizationRequest),
        ("graph-quantization", "", GraphQuantizationRequest),
        ("threshold-pruning", "0.25", LogicalPruningRequest),
        ("mask-pruning", "keep,drop,true,0", LogicalPruningRequest),
        ("nm-pruning", "", LogicalPruningRequest),
        ("structured-pruning", "0, 2, 4", StructuredPruningRequest),
    ),
)
def test_transformation_form_parsing(
    kind: str,
    parameter: str,
    expected_type: type[object],
) -> None:
    request = transformation_request(
        kind=kind,
        graph_id="main",
        node_id="node",
        tensor_id="weight",
        granularity_value=Granularity.PER_CHANNEL.value,
        axis_value="1",
        parameter=parameter,
    )
    assert isinstance(request, expected_type)
    if isinstance(request, WeightQuantizationRequest | GraphQuantizationRequest):
        assert request.granularity is Granularity.PER_CHANNEL
        assert request.axis == 1
    if isinstance(request, LogicalPruningRequest) and kind == "mask-pruning":
        assert request.mode is PruningMode.MASK
        assert request.mask == (True, False, True, False)


def test_transformation_form_errors_are_product_facing() -> None:
    with pytest.raises(ValueError, match="no initializer"):
        transformation_request(
            kind="weight-quantization",
            graph_id="main",
            node_id="node",
            tensor_id=None,
            granularity_value=Granularity.PER_TENSOR.value,
            axis_value="0",
            parameter="",
        )
    with pytest.raises(ValueError, match="comma-separated integers"):
        transformation_request(
            kind="structured-pruning",
            graph_id="main",
            node_id="node",
            tensor_id=None,
            granularity_value=Granularity.PER_TENSOR.value,
            axis_value="0",
            parameter="one",
        )
    with pytest.raises(ValueError, match="mask entries"):
        transformation_request(
            kind="mask-pruning",
            graph_id="main",
            node_id="node",
            tensor_id="weight",
            granularity_value=Granularity.PER_TENSOR.value,
            axis_value="0",
            parameter="maybe",
        )
