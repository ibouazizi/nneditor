"""ONNX-index-to-IR conversion (P1.2).

Every fixture model is indexed lazily, converted, and — because Document
construction *is* validation — the conversion itself asserts internal
consistency. The tests then check the mapping decisions: value synthesis,
placeholder ports, attribute typing, subgraph attachment, capability
narrowing, extensions, and serialization round trips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import AttrKind, Document, Storage
from nneditor.ir.serialize import document_from_bytes, document_to_bytes
from tests.fixtures.onnx_models import (
    build_attribute_variety_model,
    build_control_flow_model,
    build_custom_domain_model,
    build_embedded_model,
    build_external_model,
    build_function_model,
    build_optional_input_model,
    build_symbolic_shape_model,
)

ELEMENTS = 64


def convert(path: Path) -> Document:
    return index_to_document(index_model(path))


def test_embedded_model_maps_topology_and_tensors(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    document = convert(path)

    assert document.artifact_kind is ArtifactKind.ONNX_MODEL
    assert document.source.content_hash.startswith("sha256:")
    main = document.main_graph
    assert [node.op_type for node in main.nodes] == ["Mul", "Add"]
    assert len(main.initializers) == 2

    weight = document.tensors[main.initializers[0]]
    assert weight.storage is Storage.EMBEDDED_RAW
    assert weight.payload is not None and weight.payload.length == 4 * ELEMENTS
    assert weight.element_type == "float32"
    bias = document.tensors[main.initializers[1]]
    assert bias.storage is Storage.EMBEDDED_TYPED

    # The intermediate value between the two nodes was synthesized and its
    # dataflow links are derivable.
    scale, add = main.nodes
    intermediate = scale.outputs[0]
    assert main.producer(intermediate) == (scale.id, 0)
    assert main.consumers(intermediate) == ((add.id, 0),)

    # Initializer-backed values carry the tensor's type and shape.
    weight_value = main.value(scale.inputs[1])
    assert weight_value.element_type == "float32"
    assert weight_value.shape == (ELEMENTS,)


def test_external_model_maps_external_references(tmp_path: Path) -> None:
    path, _values = build_external_model(tmp_path, elements=ELEMENTS)
    document = convert(path)
    tensor = document.tensors[document.main_graph.initializers[0]]
    assert tensor.storage is Storage.EXTERNAL
    assert tensor.external is not None
    assert tensor.external.location == "weights.bin"
    assert tensor.external.length == 4 * ELEMENTS
    assert document.capability(Capability.WEIGHTS).availability is not None


def test_missing_external_file_narrows_weights(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    path, _values = build_external_model(directory, elements=ELEMENTS)
    (directory / "weights.bin").unlink()
    document = convert(path)

    status = document.capability(Capability.WEIGHTS)
    assert status.availability is Availability.PARTIAL
    tensor_id = document.main_graph.initializers[0]
    notes = document.notes_for(tensor_id)
    assert len(notes) == 1
    assert notes[0].capability is Capability.WEIGHTS
    assert notes[0].availability is Availability.UNAVAILABLE


def test_symbolic_shapes_survive(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_symbolic_shape_model(path)
    document = convert(path)
    main = document.main_graph
    entry = main.value(main.inputs[0])
    assert entry.shape == ("batch", 3, "height", 224)
    assert main.symbolic_dimensions == ("batch", "height")


def test_custom_domain_notes_editing_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_custom_domain_model(path)
    document = convert(path)
    node = document.main_graph.nodes[0]
    assert node.domain == "com.example.ops"
    assert node.attribute("approximation").value == "tanh"
    notes = document.notes_for(node.id)
    assert any(
        note.capability is Capability.EDITING
        and note.availability is Availability.UNAVAILABLE
        for note in notes
    )


def test_attribute_kinds_are_typed(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_attribute_variety_model(path)
    document = convert(path)
    node = document.main_graph.nodes[0]

    axis = node.attribute("axis")
    assert axis.kind is AttrKind.INT and axis.value == -2
    alpha = node.attribute("alpha")
    assert alpha.kind is AttrKind.FLOAT and alpha.value == 1.5
    mode = node.attribute("mode")
    assert mode.kind is AttrKind.STRING and mode.value == "reflect"
    pads = node.attribute("pads")
    assert pads.kind is AttrKind.INTS and pads.value == (0, 1, 2)
    scales = node.attribute("scales")
    assert scales.kind is AttrKind.FLOATS and scales.value == (0.5, 2.0)
    labels = node.attribute("labels")
    assert labels.kind is AttrKind.STRINGS and labels.value == ("a", "b")


def test_control_flow_subgraphs_attach_to_their_node(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_control_flow_model(path)
    document = convert(path)
    branch = document.main_graph.nodes[0]
    assert branch.op_type == "If"
    assert len(branch.subgraphs) == 2

    then_attr = branch.attribute("then_branch")
    assert then_attr.kind is AttrKind.GRAPH
    then_graph = document.graphs[str(then_attr.value)]
    assert then_graph.parent_node == branch.id

    # The constant tensor inside each branch is addressable at the document
    # level through the branch node's Constant attribute.
    constant = then_graph.nodes[0]
    tensor_ref = constant.attribute("value")
    assert tensor_ref.kind is AttrKind.TENSOR
    assert str(tensor_ref.value) in document.tensors


def test_function_bodies_become_graphs_with_signatures(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_function_model(path)
    document = convert(path)

    extensions = dict(document.extensions)
    functions = extensions["x-onnx.functions"]
    assert isinstance(functions, list)
    record = functions[0]
    assert isinstance(record, dict)
    assert record["name"] == "Scale"
    body = document.graphs[str(record["graph_id"])]
    assert [node.op_type for node in body.nodes] == ["Mul"]
    assert [body.value(item).name for item in body.inputs] == ["x"]
    assert [body.value(item).name for item in body.outputs] == ["y"]


def test_omitted_optional_input_keeps_its_position(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_optional_input_model(path)
    document = convert(path)
    clip = document.main_graph.nodes[0]
    assert len(clip.inputs) == 3, "the empty min input keeps position 1"
    placeholder = document.main_graph.value(clip.inputs[1])
    assert placeholder.name == ""
    assert "#empty:" in placeholder.id
    assert document.main_graph.value(clip.inputs[2]).name == "limit"


def test_custom_domain_and_incomplete_shape_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_custom_domain_model(path)
    document = convert(path)
    codes = [item.code for item in document.diagnostics]
    assert "onnx.custom-domain" in codes

    symbolic = tmp_path / "symbolic.onnx"
    build_symbolic_shape_model(symbolic)
    loose = convert(symbolic)
    incomplete = [
        item for item in loose.diagnostics if item.code == "onnx.incomplete-io-shape"
    ]
    assert len(incomplete) == 2, "one finding per underspecified input and output"
    assert not loose.has_errors


def test_model_metadata_rides_in_extensions(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    document = convert(path)
    model_extension = dict(document.extensions)["x-onnx.model"]
    assert isinstance(model_extension, dict)
    assert model_extension["producer_name"] == "nneditor"
    opsets = model_extension["opset_imports"]
    assert isinstance(opsets, list) and {"domain": "", "version": 18} in opsets


def test_import_provenance_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    document = convert(path)
    entry = document.provenance[0]
    assert entry.operation == "import"
    assert entry.tool_version.startswith("nneditor ")
    assert entry.source_artifact == document.source.content_hash
    assert ("loading_mode", "safe artifact") in entry.parameters


@pytest.mark.parametrize(
    "builder",
    [
        build_embedded_model,
        build_symbolic_shape_model,
        build_custom_domain_model,
        build_control_flow_model,
        build_function_model,
        build_optional_input_model,
        build_attribute_variety_model,
    ],
)
def test_documents_round_trip_deterministically(
    tmp_path: Path, builder: object
) -> None:
    path = tmp_path / "model.onnx"
    builder(path)  # type: ignore[operator]
    document = convert(path)
    payload = document_to_bytes(document)
    assert document_to_bytes(document_from_bytes(payload)) == payload
    # Converting the same artifact twice is byte-identical.
    assert document_to_bytes(convert(path)) == payload
