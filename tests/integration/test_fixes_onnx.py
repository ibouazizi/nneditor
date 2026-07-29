"""Integration regression tests for the verified ONNX adapter defect fixes."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data

from nneditor.adapters.onnx import (
    ExportError,
    SpliceExportError,
    export_revision,
    export_with_edits,
    index_model,
    index_to_document,
    read_tensor_bytes,
)
from nneditor.adapters.onnx.exporter import (
    _relocate_external_references,
    _unsupported_operators,
)
from nneditor.adapters.onnx.numerical import compare_numerically
from nneditor.storage.store import TensorStore
from tests.fixtures.onnx_models import build_embedded_model, build_external_model

_OPSET = 18


# --------------------------------------------------------------------------
# Finding 1: >1 MiB typed integer tensors materialize
# --------------------------------------------------------------------------


def test_a_large_typed_int32_initializer_materializes(tmp_path: Path) -> None:
    """The verified failure: 300k int32 elements exceed the old 1 MiB cap."""
    elements = 300_000
    values = [2**30] * elements  # five wire bytes each: a 1.5 MB packed field
    weight = helper.make_tensor(
        "weight", TensorProto.INT32, [elements], values, raw=False
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"], name="passthrough")],
        "typed_large",
        [],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [elements])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "typed_large.onnx"
    onnx.save_model(model, path)

    document = index_to_document(index_model(path))
    tensor_id = document.main_graph.initializers[0]
    expected = np.full(elements, 2**30, dtype=np.int32).tobytes()
    assert len(expected) > 1 << 20
    with TensorStore(document) as store:
        assert store.read(tensor_id) == expected


# --------------------------------------------------------------------------
# Finding 2: oversized attributes degrade to a diagnostic, not a refusal
# --------------------------------------------------------------------------


def test_an_oversized_attribute_is_diagnosed_without_failing_the_import(
    tmp_path: Path,
) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Probe",
                ["input"],
                ["output"],
                name="probe",
                domain="com.example",
                blob="x" * (2 * 1024 * 1024),
                axis=3,
            )
        ],
        "big_attribute",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example", 1),
        ],
    )
    path = tmp_path / "big_attribute.onnx"
    onnx.save_model(model, path)

    index = index_model(path)  # previously OnnxIndexError refused the model
    assert "onnx.attribute-too-large" in index.diagnostics.codes()
    assert not index.diagnostics.has_errors
    node = index.main_graph.nodes[0]
    assert "blob" in node.unsupported_attributes
    assert all(attribute.name != "blob" for attribute in node.attributes)
    # Attributes under the limit on the same node still import normally.
    assert any(attribute.name == "axis" for attribute in node.attributes)


# --------------------------------------------------------------------------
# Finding 3: destination collisions fail loudly; relative paths export
# --------------------------------------------------------------------------


def test_a_destination_colliding_with_an_external_file_fails_loudly(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    path, _values = build_external_model(source_dir, elements=8)
    index = index_model(path)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(SpliceExportError, match="twice"):
        export_with_edits(index, (), out / "weights.bin")
    assert list(out.iterdir()) == []


def test_a_relatively_indexed_model_still_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    path, _values = build_external_model(source_dir, elements=8)
    monkeypatch.chdir(tmp_path)

    index = index_model(Path("src") / "model.onnx")
    destination = tmp_path / "out" / "model.onnx"
    report = export_with_edits(index, (), destination)

    assert destination.read_bytes() == path.read_bytes()
    assert (tmp_path / "out" / "weights.bin").read_bytes() == (
        source_dir / "weights.bin"
    ).read_bytes()
    assert len(report.written_files) == 2


# --------------------------------------------------------------------------
# Finding 7: relocation rewrites external locations in every graph
# --------------------------------------------------------------------------


def _external_initializer(name: str, location: str, length: int) -> TensorProto:
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend([4])
    tensor.raw_data = b""
    set_external_data(tensor, location=location, offset=0, length=length)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    return tensor


def test_relocation_rewrites_subgraph_initializer_locations(tmp_path: Path) -> None:
    payload = np.arange(4, dtype=np.float32).tobytes()
    (tmp_path / "weights.bin").write_bytes(payload)
    then_graph = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["branch_weight"], ["branch_output"], name="then_identity"
            )
        ],
        "then_body",
        [],
        [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [4])],
        initializer=[_external_initializer("branch_weight", "weights.bin", 16)],
    )
    else_constant = helper.make_tensor(
        "else_value", TensorProto.FLOAT, [4], struct.pack("<4f", 0, 0, 0, 0), raw=True
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["branch_output"],
                name="else_const",
                value=else_constant,
            )
        ],
        "else_body",
        [],
        [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [4])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["output"],
                name="branch",
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        "control_flow",
        [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    proto = onnx.load_model(path, load_external_data=False)
    destination = tmp_path / "exported.onnx"
    mapping = {
        (tmp_path / "weights.bin").resolve(): tmp_path
        / "exported.onnx.data"
        / "weights.bin"
    }
    _relocate_external_references(proto, index, mapping, destination)

    then_attr = next(
        item for item in proto.graph.node[0].attribute if item.name == "then_branch"
    )
    entry = next(
        item
        for item in then_attr.g.initializer[0].external_data
        if item.key == "location"
    )
    assert entry.value == "exported.onnx.data/weights.bin"


# --------------------------------------------------------------------------
# Finding 8: duplicate tensor identifiers are diagnosed
# --------------------------------------------------------------------------


def test_duplicate_initializer_names_are_diagnosed(tmp_path: Path) -> None:
    first = helper.make_tensor(
        "weight", TensorProto.FLOAT, [1], struct.pack("<f", 1.0), raw=True
    )
    second = helper.make_tensor(
        "weight", TensorProto.FLOAT, [1], struct.pack("<f", 2.0), raw=True
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"], name="passthrough")],
        "duplicates",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=[first, second],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = tmp_path / "duplicates.onnx"
    onnx.save_model(model, path)

    index = index_model(path)
    assert "onnx.duplicate-tensor-id" in index.diagnostics.codes()
    assert len(index.main_graph.initializers) == 2
    tensor_id = index.main_graph.initializers[0].id
    assert index.main_graph.initializers[1].id == tensor_id
    # Lookups resolve to the later definition — now disclosed, not silent.
    assert read_tensor_bytes(index, index.tensor(tensor_id)) == struct.pack("<f", 2.0)


# --------------------------------------------------------------------------
# Finding 9: pre-staging failures surface as ExportError
# --------------------------------------------------------------------------


class _FakeDecodeError(Exception):
    """Stands in for the raw protobuf DecodeError the bindings can raise."""


def test_pre_staging_failures_surface_as_export_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=4)
    index = index_model(source)
    document = index_to_document(index)

    def boom(*args: object, **kwargs: object) -> object:
        raise _FakeDecodeError("Error parsing message")

    monkeypatch.setattr("nneditor.adapters.onnx.exporter.onnx.load_model", boom)
    with pytest.raises(ExportError, match="export preparation failed"):
        export_revision(
            document,
            (),
            tmp_path / "out.onnx",
            tensor_reader=lambda tensor_id: read_tensor_bytes(index, tensor_id),
        )
    assert not (tmp_path / "out.onnx").exists()


# --------------------------------------------------------------------------
# Finding 10: the "ai.onnx" opset alias is normalized at import
# --------------------------------------------------------------------------


def test_the_ai_onnx_opset_alias_is_normalized_at_import(tmp_path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"], name="activation")],
        "alias",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("ai.onnx", _OPSET)]
    )
    path = tmp_path / "alias.onnx"
    onnx.save_model(model, path)

    document = index_to_document(index_model(path))
    extension = dict(document.extensions)["x-onnx.model"]
    assert isinstance(extension, dict)
    assert extension["opset_imports"] == [{"domain": "", "version": _OPSET}]
    # The downstream schema gate that queries "" now sees the version.
    assert _unsupported_operators(document) == ()


# --------------------------------------------------------------------------
# Finding 4: the capped smoke worker still compares normally end to end
# --------------------------------------------------------------------------


def test_numerical_comparison_succeeds_under_the_platform_cap(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=4)
    edited = tmp_path / "edited.onnx"
    edited.write_bytes(source.read_bytes())

    result = compare_numerically(
        source,
        edited,
        {"input": np.ones(4, dtype=np.float32)},
        approved=True,
        timeout_seconds=120.0,
    )
    assert result.passed
    assert result.runtime.startswith("onnx.reference")
