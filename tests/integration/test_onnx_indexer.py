"""End-to-end tests for the ONNX lazy indexer (P0.4).

The central claim under test is that structural parsing never requests tensor
payload ranges or materializes values. Artifact identity hashing separately
streams every source byte without decoding or retaining it. Two parser
measurements back up the structural claim:

* ``stats.logical_ranges`` — what the parser asked for. No request may overlap a
  tensor payload.
* ``stats.physical_bytes`` — what actually reached the file, in whole blocks.
  Block alignment can pull in bytes next to a payload, so this is checked as a
  budget rather than as an exact-overlap assertion.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest

from nneditor.adapters.onnx import (
    ModelIndex,
    OnnxIndexError,
    ScanLimits,
    TensorStorage,
    index_model,
    read_tensor_bytes,
)
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.diagnostics import Severity
from nneditor.ir.identity import NodeIdStability
from nneditor.storage.reader import ByteRange, hash_file
from tests.fixtures.onnx_models import (
    LARGE_TENSOR_ELEMENTS,
    build_control_flow_model,
    build_custom_domain_model,
    build_embedded_model,
    build_external_model,
    build_function_model,
    build_symbolic_shape_model,
    build_tensor_only_model,
    external_tensor_model,
)

# --------------------------------------------------------------------------
# Embedded tensors
# --------------------------------------------------------------------------


@pytest.fixture
def embedded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "embedded.onnx"
    values = build_embedded_model(path)
    return path, values


def test_topology_is_indexed(embedded: tuple[Path, np.ndarray]) -> None:
    path, _ = embedded
    index = index_model(path)

    graph = index.main_graph
    assert [node.op_type for node in graph.nodes] == ["Mul", "Add"]
    assert [node.name for node in graph.nodes] == ["scale", "shift"]
    assert graph.nodes[0].inputs == ("input", "weight")
    assert graph.nodes[1].outputs == ("output",)
    assert [value.name for value in graph.inputs] == ["input"]
    assert [value.name for value in graph.outputs] == ["output"]
    assert index.producer_name == "nneditor"
    assert index.opset_imports[0].is_default_domain
    assert not index.diagnostics.has_errors


def test_node_ids_are_name_derived_and_unique(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _ = embedded
    index = index_model(path)
    nodes = index.main_graph.nodes
    assert all(node.id_stability is NodeIdStability.NAMED for node in nodes)
    assert len({node.id for node in nodes}) == len(nodes)


def test_the_content_hash_identifies_the_source(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _ = embedded
    index = index_model(path)
    assert index.content_hash == hash_file(path)
    assert index.byte_size == path.stat().st_size


def test_external_payloads_participate_in_artifact_identity(tmp_path: Path) -> None:
    path, _values = build_external_model(tmp_path, elements=16)
    first = index_model(path)
    weights = tmp_path / "weights.bin"

    changed = bytearray(weights.read_bytes())
    changed[0] ^= 0xFF
    weights.write_bytes(changed)
    second = index_model(path)

    assert first.model_hash == second.model_hash == hash_file(path)
    assert first.external_hashes != second.external_hashes
    assert first.content_hash != second.content_hash


def test_cancelled_index_stops_during_artifact_hashing(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _values = embedded
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        index_model(path, token=token)


def test_embedded_tensor_payloads_are_located_not_read(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, values = embedded
    index = index_model(path)

    weight = next(t for t in index.main_graph.initializers if t.name == "weight")
    assert weight.storage is TensorStorage.EMBEDDED_RAW
    assert weight.payload is not None
    assert weight.payload.length == values.nbytes
    assert weight.dims == (LARGE_TENSOR_ELEMENTS,)
    assert weight.expected_byte_length == values.nbytes
    assert weight.is_readable

    assert not index.stats.touched_logically(weight.payload), (
        "indexing requested bytes inside the tensor payload"
    )


def test_indexing_reads_a_small_fraction_of_a_weight_heavy_model(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _ = embedded
    index = index_model(path)
    size = path.stat().st_size

    assert size > 2_000_000, "fixture is too small for this budget to mean anything"
    assert index.stats.logical_bytes < size // 100
    assert index.stats.physical_bytes < size // 10


def test_the_located_range_holds_the_real_tensor_bytes(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, values = embedded
    index = index_model(path)
    weight = next(t for t in index.main_graph.initializers if t.name == "weight")

    raw = read_tensor_bytes(index, weight)
    assert np.frombuffer(raw, dtype=np.float32).tolist() == values.tolist()


def test_a_tensor_can_be_read_by_identifier(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, values = embedded
    index = index_model(path)
    weight = next(t for t in index.main_graph.initializers if t.name == "weight")
    assert read_tensor_bytes(index, weight.id) == values.tobytes()


def test_typed_tensor_fields_are_flagged_as_not_range_readable(
    embedded: tuple[Path, np.ndarray],
) -> None:
    """``bias`` is written as ``float_data``, not ``raw_data``."""
    path, _ = embedded
    index = index_model(path)
    bias = next(t for t in index.main_graph.initializers if t.name == "bias")
    assert bias.storage is TensorStorage.EMBEDDED_TYPED
    assert not bias.is_readable
    assert "onnx.typed-tensor-field" in index.diagnostics.codes()
    with pytest.raises(OnnxIndexError, match="not readable as a byte range"):
        read_tensor_bytes(index, bias)


def test_declared_tensor_bytes_dwarf_the_bytes_read(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _ = embedded
    index = index_model(path)
    assert index.declared_tensor_bytes > 100 * index.stats.logical_bytes


# --------------------------------------------------------------------------
# External tensors
# --------------------------------------------------------------------------


def test_external_tensors_are_indexed_by_file_offset_and_length(
    tmp_path: Path,
) -> None:
    path, values = build_external_model(tmp_path)
    index = index_model(path)

    weight = index.main_graph.initializers[0]
    assert weight.storage is TensorStorage.EXTERNAL
    assert weight.external is not None
    assert weight.external.is_usable
    assert weight.external.offset == 0
    assert weight.external.length == values.nbytes
    assert weight.external.resolved_path == (tmp_path / "weights.bin")
    assert index.external_files == (tmp_path / "weights.bin",)
    assert not index.diagnostics.has_errors


def test_opening_an_external_model_does_not_read_the_weights_file(
    tmp_path: Path,
) -> None:
    path, values = build_external_model(tmp_path)
    index = index_model(path)

    # The model file holds topology only; the weights file is stat'ed for its
    # size and never read.
    assert index.byte_size < values.nbytes // 100
    assert index.stats.logical_bytes < values.nbytes // 1000
    assert index.stats.coalesced_physical() == (ByteRange(0, index.byte_size),)


def test_external_tensor_bytes_are_readable_on_demand(tmp_path: Path) -> None:
    path, values = build_external_model(tmp_path)
    index = index_model(path)
    raw = read_tensor_bytes(index, index.main_graph.initializers[0])
    assert np.frombuffer(raw, dtype=np.float32).tolist() == values.tolist()


def test_an_omitted_length_means_the_rest_of_the_file(tmp_path: Path) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=None,
        dims=[4],
        payload=payload,
    )
    index = index_model(path)
    weight = index.main_graph.initializers[0]
    assert weight.external is not None
    assert weight.external.length == len(payload)
    assert read_tensor_bytes(index, weight) == payload


def test_a_non_zero_offset_is_honoured(tmp_path: Path) -> None:
    payload = b"\x00" * 16 + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=16,
        length=16,
        dims=[4],
        payload=payload,
    )
    index = index_model(path)
    assert read_tensor_bytes(index, index.main_graph.initializers[0]) == payload[16:]


# --------------------------------------------------------------------------
# Malformed and hostile external references
# --------------------------------------------------------------------------


def _first_error(index: ModelIndex) -> str:
    errors = index.diagnostics.of_severity(Severity.ERROR)
    assert errors, "expected an error diagnostic"
    return errors[0].code


def test_a_traversing_external_location_is_rejected(tmp_path: Path) -> None:
    (tmp_path.parent / "secret.bin").write_bytes(b"secret")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = external_tensor_model(
        model_dir,
        location="../secret.bin",
        offset=0,
        length=6,
        dims=[6],
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.external-path-unsafe"
    assert not index.main_graph.initializers[0].is_readable


def test_an_absolute_external_location_is_rejected(tmp_path: Path) -> None:
    path = external_tensor_model(
        tmp_path,
        location="/etc/passwd",
        offset=0,
        length=4,
        dims=[1],
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.external-path-unsafe"


def test_a_missing_external_file_is_reported(tmp_path: Path) -> None:
    path = external_tensor_model(
        tmp_path,
        location="absent.bin",
        offset=0,
        length=4,
        dims=[1],
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.external-file-missing"


def test_a_range_beyond_the_external_file_is_reported(tmp_path: Path) -> None:
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=8,
        length=4,
        dims=[1],
        payload=b"\x00" * 8,
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.external-range-out-of-bounds"
    assert not index.main_graph.initializers[0].is_readable


def test_a_length_disagreeing_with_the_shape_is_reported(tmp_path: Path) -> None:
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=8,
        dims=[4],
        payload=b"\x00" * 8,
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.external-length-mismatch"


@pytest.mark.parametrize("value", ["not-a-number", "-16", "0x10"])
def test_non_numeric_external_metadata_is_reported(tmp_path: Path, value: str) -> None:
    path = external_tensor_model(
        tmp_path,
        location="weights.bin",
        offset=0,
        length=16,
        dims=[4],
        payload=b"\x00" * 16,
    )
    model = onnx.load(path, load_external_data=False)
    for entry in model.graph.initializer[0].external_data:
        if entry.key == "offset":
            entry.value = value
    onnx.save_model(model, path)

    index = index_model(path)
    assert _first_error(index) == "onnx.external-metadata-invalid"


def test_an_embedded_payload_disagreeing_with_the_shape_is_reported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.onnx"
    build_tensor_only_model(
        path, dims=[4], raw_bytes=b"\x00" * 8, data_type=onnx.TensorProto.FLOAT
    )
    index = index_model(path)
    assert _first_error(index) == "onnx.payload-length-mismatch"
    assert not index.main_graph.initializers[0].is_readable


def test_a_negative_dimension_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "negative.onnx"
    build_tensor_only_model(
        path, dims=[4], raw_bytes=b"\x00" * 16, data_type=onnx.TensorProto.FLOAT
    )
    model = onnx.load(path)
    model.graph.initializer[0].dims[0] = -4
    onnx.save_model(model, path)

    index = index_model(path)
    assert _first_error(index) == "onnx.negative-dimension"


def test_an_unknown_element_type_is_reported_without_refusing_the_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "future.onnx"
    build_tensor_only_model(
        path, dims=[4], raw_bytes=b"\x00" * 16, data_type=onnx.TensorProto.FLOAT
    )
    model = onnx.load(path)
    model.graph.initializer[0].data_type = 250
    onnx.save_model(model, path)

    index = index_model(path)
    assert "onnx.unknown-element-type" in index.diagnostics.codes()
    tensor = index.main_graph.initializers[0]
    assert tensor.expected_byte_length is None
    assert tensor.storage is TensorStorage.EMBEDDED_RAW


def test_checksum_verification_is_opt_in_and_detects_tampering(
    tmp_path: Path,
) -> None:
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
    entry.value = "sha256:" + "0" * 64
    onnx.save_model(model, path)

    lazy = index_model(path)
    assert not lazy.diagnostics.has_errors, "checksums must not be verified by default"

    checked = index_model(path, verify_checksums=True)
    assert _first_error(checked) == "onnx.checksum-mismatch"


def test_a_file_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "garbage.onnx"
    path.write_bytes(b"\x08\x07\x12\x04test")  # valid protobuf, no graph
    with pytest.raises(OnnxIndexError, match="has no graph"):
        index_model(path)


def test_a_corrupt_file_is_refused_with_a_clear_message(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.onnx"
    path.write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff")
    with pytest.raises(OnnxIndexError, match="not a readable ONNX model"):
        index_model(path)


def test_scan_limits_stop_a_deeply_nested_artifact(tmp_path: Path) -> None:
    path = tmp_path / "nested.onnx"
    build_control_flow_model(path)
    with pytest.raises(OnnxIndexError, match="not a readable ONNX model"):
        index_model(path, limits=ScanLimits(max_depth=2))


# --------------------------------------------------------------------------
# Shapes, custom domains, control flow, and functions
# --------------------------------------------------------------------------


def test_symbolic_dimensions_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "symbolic.onnx"
    build_symbolic_shape_model(path)
    index = index_model(path)

    value = index.main_graph.inputs[0]
    assert value.shape == ("batch", 3, "height", 224)
    assert value.has_symbolic_dimensions
    assert not value.is_fully_specified
    assert value.element_type is not None
    assert value.element_type.name == "FLOAT"


def test_custom_domains_are_visible(tmp_path: Path) -> None:
    path = tmp_path / "custom.onnx"
    build_custom_domain_model(path)
    index = index_model(path)

    node = index.main_graph.nodes[0]
    assert node.domain == "com.example.ops"
    assert node.qualified_op_type == "com.example.ops::FusedGelu"
    assert node.attribute_names == ("approximation",)
    assert index.custom_domains == ("com.example.ops",)


def test_control_flow_subgraphs_are_indexed_as_their_own_graphs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control_flow.onnx"
    build_control_flow_model(path)
    index = index_model(path)

    branch = index.main_graph.nodes[0]
    assert branch.op_type == "If"
    assert len(branch.subgraph_ids) == 2
    for subgraph_id in branch.subgraph_ids:
        subgraph = index.graphs[subgraph_id]
        assert subgraph.parent_node_id == branch.id
        assert len(subgraph.nodes) == 1
    assert index.node_count == 3


def test_constant_attribute_tensors_are_located_not_read(tmp_path: Path) -> None:
    path = tmp_path / "control_flow.onnx"
    build_control_flow_model(path)
    index = index_model(path)

    attribute_tensors = [
        tensor for graph in index.graphs.values() for tensor in graph.attribute_tensors
    ]
    assert len(attribute_tensors) == 2
    for tensor in attribute_tensors:
        assert tensor.storage is TensorStorage.EMBEDDED_RAW
        assert tensor.payload is not None
        assert not index.stats.touched_logically(tensor.payload)
    assert {read_tensor_bytes(index, tensor) for tensor in attribute_tensors} == {
        struct.pack("<f", 1.0),
        struct.pack("<f", -1.0),
    }


def test_model_local_functions_are_indexed(tmp_path: Path) -> None:
    path = tmp_path / "function.onnx"
    build_function_model(path)
    index = index_model(path)

    assert len(index.functions) == 1
    function = index.functions[0]
    assert function.name == "Scale"
    assert function.domain == "com.example.fn"
    assert function.inputs == ("x",)
    body = index.graphs[function.graph_id]
    assert [node.op_type for node in body.nodes] == ["Mul"]


# --------------------------------------------------------------------------
# Differential validation against the reference parser
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    ["embedded", "symbolic", "custom", "control_flow", "function"],
)
def test_the_index_agrees_with_the_reference_parser(
    tmp_path: Path, builder: str
) -> None:
    path = tmp_path / f"{builder}.onnx"
    builders: dict[str, Callable[[], object]] = {
        "embedded": lambda: build_embedded_model(path, elements=16),
        "symbolic": lambda: build_symbolic_shape_model(path),
        "custom": lambda: build_custom_domain_model(path),
        "control_flow": lambda: build_control_flow_model(path),
        "function": lambda: build_function_model(path),
    }
    builders[builder]()

    index = index_model(path)
    reference = onnx.load(path, load_external_data=False)

    assert index.ir_version == reference.ir_version
    assert index.producer_name == reference.producer_name
    assert [(item.domain, item.version) for item in index.opset_imports] == [
        (item.domain, item.version) for item in reference.opset_import
    ]

    graph = index.main_graph
    assert graph.name == reference.graph.name
    assert [node.op_type for node in graph.nodes] == [
        node.op_type for node in reference.graph.node
    ]
    assert [node.name for node in graph.nodes] == [
        node.name for node in reference.graph.node
    ]
    assert [node.domain for node in graph.nodes] == [
        node.domain for node in reference.graph.node
    ]
    assert [tuple(node.inputs) for node in graph.nodes] == [
        tuple(node.input) for node in reference.graph.node
    ]
    assert [tuple(node.outputs) for node in graph.nodes] == [
        tuple(node.output) for node in reference.graph.node
    ]
    assert [(t.name, t.dims) for t in graph.initializers] == [
        (t.name, tuple(t.dims)) for t in reference.graph.initializer
    ]
    assert [(t.element_type.code) for t in graph.initializers] == [
        t.data_type for t in reference.graph.initializer
    ]


def test_embedded_ranges_match_the_reference_tensor_bytes(tmp_path: Path) -> None:
    path = tmp_path / "embedded.onnx"
    build_embedded_model(path, elements=1024)
    index = index_model(path)
    reference = onnx.load(path)

    by_name = {tensor.name: tensor for tensor in reference.graph.initializer}
    for tensor in index.main_graph.initializers:
        if tensor.storage is TensorStorage.EMBEDDED_RAW:
            assert read_tensor_bytes(index, tensor) == by_name[tensor.name].raw_data


def test_repeated_opens_produce_identical_indexes(tmp_path: Path) -> None:
    path = tmp_path / "embedded.onnx"
    build_embedded_model(path, elements=64)

    first = index_model(path)
    second = index_model(path)
    assert first.content_hash == second.content_hash
    assert [n.id for n in first.main_graph.nodes] == [
        n.id for n in second.main_graph.nodes
    ]
    assert [t.payload for t in first.main_graph.initializers] == [
        t.payload for t in second.main_graph.initializers
    ]


def test_a_smaller_block_size_reads_strictly_fewer_bytes(tmp_path: Path) -> None:
    path = tmp_path / "embedded.onnx"
    build_embedded_model(path)
    coarse = index_model(path, block_size=64 * 1024)
    fine = index_model(path, block_size=4 * 1024)
    assert fine.stats.physical_bytes < coarse.stats.physical_bytes
    assert fine.stats.logical_bytes == coarse.stats.logical_bytes


def test_an_explicit_external_root_confines_reads(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    (weights_dir / "weights.bin").write_bytes(payload)
    path = external_tensor_model(
        model_dir,
        location="weights.bin",
        offset=0,
        length=16,
        dims=[4],
    )

    unresolved = index_model(path)
    assert _first_error(unresolved) == "onnx.external-file-missing"

    resolved = index_model(path, external_root=weights_dir)
    assert not resolved.diagnostics.has_errors
    assert read_tensor_bytes(resolved, resolved.main_graph.initializers[0]) == payload


def test_tensor_lookup_covers_every_indexed_tensor(tmp_path: Path) -> None:
    path = tmp_path / "control_flow.onnx"
    build_control_flow_model(path)
    index = index_model(path)
    for tensor in index.iter_tensors():
        assert index.tensor(tensor.id) is tensor
    assert index.tensor_count == 2


def test_payload_ranges_stay_inside_the_artifact(
    embedded: tuple[Path, np.ndarray],
) -> None:
    path, _ = embedded
    index = index_model(path)
    whole = ByteRange(0, index.byte_size)
    for tensor in index.iter_tensors():
        if tensor.payload is not None:
            assert whole.contains(tensor.payload)
