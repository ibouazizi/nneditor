"""Edge cases and hostile encodings for the ONNX lazy indexer (P0.4)."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.adapters.onnx import (
    OnnxIndexError,
    TensorStorage,
    index_model,
    read_tensor_bytes,
)
from nneditor.adapters.onnx.wire import WireType
from nneditor.diagnostics import Severity
from nneditor.ir.identity import NodeIdStability
from tests.fixtures.onnx_models import external_tensor_model
from tests.fixtures.protobuf import length_field, tag, varint, varint_field

_OPSET = 18

# Field numbers used to hand-encode a minimal model; see onnx.proto.
_MODEL_IR_VERSION = 1
_MODEL_GRAPH = 7
_GRAPH_INITIALIZER = 5
_GRAPH_NAME = 2
_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_NAME = 8
_TENSOR_RAW_DATA = 9


def _errors(path: Path, **kwargs: object) -> tuple[str, ...]:
    index = index_model(path, **kwargs)  # type: ignore[arg-type]
    return tuple(item.code for item in index.diagnostics.of_severity(Severity.ERROR))


# --------------------------------------------------------------------------
# Model-level metadata
# --------------------------------------------------------------------------


def test_model_metadata_properties_and_domain_are_indexed(tmp_path: Path) -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Relu", ["input"], ["output"], name="activation")],
        name="metadata",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        value_info=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.domain = "com.example"
    model.model_version = 42
    entry = model.metadata_props.add()
    entry.key = "author"
    entry.value = "nneditor tests"
    path = tmp_path / "metadata.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    assert index.domain == "com.example"
    assert index.model_version == 42
    assert index.metadata_props == (("author", "nneditor tests"),)
    assert [value.name for value in index.main_graph.value_info] == ["output"]


def test_a_non_tensor_typed_value_leaves_its_type_unresolved(tmp_path: Path) -> None:
    """Sequence and map types are not modelled yet; they must not crash."""
    sequence = helper.make_tensor_sequence_value_info(
        "sequence", TensorProto.FLOAT, [2]
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("SequenceEmpty", [], ["sequence"], name="empty", dtype=1)
        ],
        name="sequence_output",
        inputs=[],
        outputs=[sequence],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "sequence.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    value = index.main_graph.outputs[0]
    assert value.name == "sequence"
    assert value.element_type is None
    assert value.shape is None
    assert not value.is_fully_specified
    assert not value.has_symbolic_dimensions


def test_sparse_initializers_are_reported_as_unmodelled(tmp_path: Path) -> None:
    values = helper.make_tensor("values", TensorProto.FLOAT, [1], [1.0])
    indices = helper.make_tensor("indices", TensorProto.INT64, [1, 1], [0])
    sparse = helper.make_sparse_tensor(values, indices, [4])
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["sparse"], ["output"], name="pass")],
        name="sparse",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        sparse_initializer=[sparse],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "sparse.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    assert index.main_graph.sparse_initializer_count == 1
    assert "onnx.sparse-initializer-unsupported" in index.diagnostics.codes()
    assert not index.diagnostics.has_errors


# --------------------------------------------------------------------------
# Node identity fallbacks
# --------------------------------------------------------------------------


def test_unnamed_nodes_fall_back_to_their_first_output(tmp_path: Path) -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Relu", ["input"], ["hidden"]),
            helper.make_node("Neg", ["hidden"], ["output"]),
        ],
        name="unnamed",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "unnamed.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    nodes = index.main_graph.nodes
    assert all(node.id_stability is NodeIdStability.OUTPUT_DERIVED for node in nodes)
    assert nodes[0].id.endswith("#out:hidden")


def test_a_node_with_neither_a_name_nor_outputs_is_flagged(tmp_path: Path) -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Relu", ["input"], ["output"], name="activation")],
        name="anonymous",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "anonymous.onnx"
    onnx.save_model(model, path)

    stripped = onnx.load(path)
    stripped.graph.node[0].ClearField("name")
    stripped.graph.node[0].ClearField("output")
    onnx.save_model(stripped, path)

    index = index_model(path)
    node = index.main_graph.nodes[0]
    assert node.id_stability is NodeIdStability.POSITIONAL
    assert "onnx.positional-node-id" in index.diagnostics.codes()


def test_two_nodes_sharing_a_name_fall_back_to_position(tmp_path: Path) -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Relu", ["input"], ["hidden"], name="same"),
            helper.make_node("Neg", ["hidden"], ["output"], name="same"),
        ],
        name="duplicate",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "duplicate.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    nodes = index.main_graph.nodes
    assert len({node.id for node in nodes}) == 2, "identifiers must stay unique"
    assert nodes[1].id_stability is NodeIdStability.POSITIONAL
    assert "onnx.duplicate-node-id" in index.diagnostics.codes()


# --------------------------------------------------------------------------
# Repeated attribute payloads
# --------------------------------------------------------------------------


def test_repeated_graph_and_tensor_attributes_are_indexed(tmp_path: Path) -> None:
    body = helper.make_graph(
        nodes=[helper.make_node("Identity", ["x"], ["y"], name="inner")],
        name="body",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    tensors = [
        helper.make_tensor(
            f"const{index}", TensorProto.FLOAT, [1], struct.pack("<f", index), raw=True
        )
        for index in range(2)
    ]
    node = helper.make_node(
        "CustomMulti",
        ["input"],
        ["output"],
        name="multi",
        domain="com.example.ops",
        graphs=[body, body],
        tensors=tensors,
    )
    graph = helper.make_graph(
        nodes=[node],
        name="repeated",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example.ops", 1),
        ],
    )
    path = tmp_path / "repeated.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    indexed = index.main_graph.nodes[0]
    assert len(indexed.subgraph_ids) == 2
    assert indexed.subgraph_ids[0].endswith("[0]")
    assert indexed.subgraph_ids[1].endswith("[1]")
    assert len(set(indexed.subgraph_ids)) == 2
    assert len(index.main_graph.attribute_tensors) == 2
    for tensor in index.main_graph.attribute_tensors:
        assert tensor.payload is not None
        assert not index.stats.touched_logically(tensor.payload)


def test_a_function_overload_is_part_of_its_identifier(tmp_path: Path) -> None:
    function = helper.make_function(
        domain="com.example.fn",
        fname="Scale",
        inputs=["x"],
        outputs=["y"],
        nodes=[helper.make_node("Mul", ["x", "x"], ["y"], name="square")],
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    function.overload = "float"
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Scale", ["input"], ["output"], name="call", domain="com.example.fn"
            )
        ],
        name="overloaded",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example.fn", 1),
        ],
        functions=[function],
    )
    path = tmp_path / "overload.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    assert index.functions[0].overload == "float"
    assert index.functions[0].graph_id.endswith("/float")
    assert index.main_graph.nodes[0].overload == ""


# --------------------------------------------------------------------------
# Encodings the reference writer does not emit
# --------------------------------------------------------------------------


def _hand_built_model(tensor_body: bytes, graph_name: bytes = b"packed") -> bytes:
    graph = length_field(_GRAPH_NAME, graph_name) + length_field(
        _GRAPH_INITIALIZER, tensor_body
    )
    return varint_field(_MODEL_IR_VERSION, 9) + length_field(_MODEL_GRAPH, graph)


def test_packed_dimensions_are_decoded(tmp_path: Path) -> None:
    """A producer using proto3 packed encoding must index identically."""
    payload = struct.pack("<6f", *range(6))
    tensor = (
        length_field(_TENSOR_DIMS, varint(2) + varint(3))
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.FLOAT)
        + length_field(_TENSOR_NAME, b"weight")
        + length_field(_TENSOR_RAW_DATA, payload)
    )
    path = tmp_path / "packed.onnx"
    path.write_bytes(_hand_built_model(tensor))

    index = index_model(path)
    weight = index.main_graph.initializers[0]
    assert weight.dims == (2, 3)
    assert weight.element_count == 6
    assert weight.storage is TensorStorage.EMBEDDED_RAW
    assert read_tensor_bytes(index, weight) == payload


def test_an_implausible_element_count_is_refused_before_allocation(
    tmp_path: Path,
) -> None:
    huge = (1 << 40) - 1
    tensor = (
        length_field(_TENSOR_DIMS, varint(huge) + varint(huge))
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.FLOAT)
        + length_field(_TENSOR_NAME, b"weight")
        + length_field(_TENSOR_RAW_DATA, b"\x00\x00\x00\x00")
    )
    path = tmp_path / "huge.onnx"
    path.write_bytes(_hand_built_model(tensor))

    index = index_model(path)
    assert "onnx.implausible-shape" in index.diagnostics.codes()
    assert not index.main_graph.initializers[0].is_readable


def test_an_unnamed_initializer_still_gets_an_identifier(tmp_path: Path) -> None:
    tensor = (
        length_field(_TENSOR_DIMS, varint(1))
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.FLOAT)
        + length_field(_TENSOR_RAW_DATA, struct.pack("<f", 1.0))
    )
    path = tmp_path / "unnamed_tensor.onnx"
    path.write_bytes(_hand_built_model(tensor))

    index = index_model(path)
    weight = index.main_graph.initializers[0]
    assert weight.name == ""
    # The synthetic label is percent-encoded like any other name segment.
    assert weight.id == "t:g:main#initializer%3A0"
    assert index.tensor(weight.id) is weight


def test_a_group_wire_type_in_the_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "grouped.onnx"
    path.write_bytes(tag(1, WireType.SGROUP))
    with pytest.raises(OnnxIndexError, match="not a readable ONNX model"):
        index_model(path)


# --------------------------------------------------------------------------
# External data: sharing, checksums, and explicit roots
# --------------------------------------------------------------------------


def test_two_tensors_sharing_one_external_file_are_both_resolved(
    tmp_path: Path,
) -> None:
    payload = struct.pack("<8f", *range(8))
    (tmp_path / "weights.bin").write_bytes(payload)
    path = external_tensor_model(
        tmp_path, location="weights.bin", offset=0, length=16, dims=[4]
    )
    model = onnx.load(path, load_external_data=False)
    second = model.graph.initializer.add()
    second.CopyFrom(model.graph.initializer[0])
    second.name = "weight2"
    for entry in second.external_data:
        if entry.key == "offset":
            entry.value = "16"
    onnx.save_model(model, path)

    index = index_model(path)
    assert not index.diagnostics.has_errors
    assert index.external_files == (tmp_path / "weights.bin",)
    assert index.declared_tensor_bytes == 32
    first, other = index.main_graph.initializers
    assert read_tensor_bytes(index, first) == payload[:16]
    assert read_tensor_bytes(index, other) == payload[16:]


def test_a_correct_checksum_passes_verification(tmp_path: Path) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=16,
        dims=[4],
        payload=payload,
    )
    model = onnx.load(path, load_external_data=False)
    entry = model.graph.initializer[0].external_data.add()
    entry.key = "checksum"
    entry.value = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    onnx.save_model(model, path)

    index = index_model(path, verify_checksums=True)
    assert not index.diagnostics.has_errors
    assert index.main_graph.initializers[0].is_readable


def test_an_unsupported_checksum_algorithm_is_reported(tmp_path: Path) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=16,
        dims=[4],
        payload=payload,
    )
    model = onnx.load(path, load_external_data=False)
    entry = model.graph.initializer[0].external_data.add()
    entry.key = "checksum"
    entry.value = "not-a-real-algorithm:00"
    onnx.save_model(model, path)

    assert _errors(path, verify_checksums=True) == ("onnx.checksum-mismatch",)


def test_reading_through_an_explicit_root_revalidates_the_location(
    tmp_path: Path,
) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    other = tmp_path / "other"
    other.mkdir()
    (other / "weights.bin").write_bytes(payload[::-1])
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=16,
        dims=[4],
        payload=payload,
    )

    index = index_model(path)
    tensor = index.main_graph.initializers[0]
    assert read_tensor_bytes(index, tensor) == payload
    assert read_tensor_bytes(index, tensor, external_root=other) == payload[::-1]


def test_declared_bytes_ignore_an_unresolved_external_tensor(tmp_path: Path) -> None:
    path = external_tensor_model(
        tmp_path, location="absent.bin", offset=0, length=16, dims=[4]
    )
    index = index_model(path)
    tensor = index.main_graph.initializers[0]
    assert tensor.storage is TensorStorage.EXTERNAL
    assert not tensor.is_readable
    with pytest.raises(OnnxIndexError, match="not readable as a byte range"):
        read_tensor_bytes(index, tensor)
