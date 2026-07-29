"""Golden Phase 4 export, reporting, and numerical smoke workflows."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from pytest import MonkeyPatch

from nneditor.adapters.onnx import (
    ExportError,
    index_model,
    read_tensor_bytes,
    report_from_json,
)
from nneditor.application.session import ApplicationService
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.editing.commands import (
    EditCommand,
    InsertUnaryNode,
    command_from_json,
    command_to_json,
)
from nneditor.editing.validation import (
    InsertUnaryRequest,
    ReconnectInputRequest,
    RemoveUnaryRequest,
    RenameNodeRequest,
    ReplaceOperatorRequest,
    SetAttributeRequest,
)
from nneditor.ir.capabilities import ExportFidelity
from nneditor.ir.core import AttrKind
from nneditor.storage.reader import hash_file
from tests.fixtures.onnx_models import (
    build_custom_domain_model,
    build_embedded_model,
    build_external_model,
    build_symbolic_shape_model,
)


def _softmax_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], name="prob", axis=1)],
        "attribute-edit",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )


def _rewire_model(path: Path) -> None:
    shape = [4]
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["left"], ["middle"], name="copy"),
            helper.make_node("Add", ["middle", "right"], ["output"], name="sum"),
        ],
        "rewire",
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, shape)],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )


def _assert_command_round_trip(command: EditCommand) -> None:
    assert command_from_json(command_to_json(command)) == command


def test_noop_export_is_byte_identical_and_reported_losslessly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=16)
    source_bytes = source.read_bytes()
    destination = tmp_path / "noop.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        report = session.export(destination)

    assert destination.read_bytes() == source_bytes
    assert source.read_bytes() == source_bytes
    assert report.fidelity is ExportFidelity.LOSSLESS
    assert report.target_hash == report.source_hash
    assert report.edit_manifest == ()
    assert report.tool_version
    assert dict(report.dependencies)["onnx"] == onnx.__version__
    assert report.source_components[0]["content_hash"] == report.source_hash
    assert report.revision_ids == ()
    assert report.report_path.exists()
    assert report_from_json(report.report_path) == report
    onnx.checker.check_model(destination, full_check=True)


def test_structural_commands_export_reopen_and_preserve_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=16)
    source_hash = hash_file(source)
    destination = tmp_path / "edited.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        scale, shift = session.document.main_graph.nodes
        rename = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, scale.id, "renamed")
        )
        _assert_command_round_trip(rename.commands[0])
        session.commit_edit(rename)
        replace = session.prepare_edit(
            ReplaceOperatorRequest(session.document.entry_graph, scale.id, "Add")
        )
        _assert_command_round_trip(replace.commands[0])
        session.commit_edit(replace)
        insert = session.prepare_edit(
            InsertUnaryRequest(
                session.document.entry_graph,
                shift.id,
                0,
                "Relu",
            )
        )
        _assert_command_round_trip(insert.commands[0])
        session.commit_edit(insert)
        report = session.export(destination, external_data_threshold=1 << 30)

    model = onnx.load_model(destination, load_external_data=False)
    assert [node.op_type for node in model.graph.node] == ["Add", "Relu", "Add"]
    assert model.graph.node[0].name == "renamed"
    assert len(report.edit_manifest) == 3
    assert len(report.revision_ids) == 3
    assert report.fidelity is ExportFidelity.SEMANTICALLY_EQUIVALENT
    assert "onnx.checker full_check passed" in report.structural_checks
    assert hash_file(source) == source_hash
    reopened = index_model(destination)
    assert reopened.node_count == 3


def test_attribute_edit_exports_and_reopens(tmp_path: Path) -> None:
    source = tmp_path / "softmax.onnx"
    _softmax_model(source)
    destination = tmp_path / "softmax-edited.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        node = session.document.main_graph.nodes[0]
        attribute = session.prepare_edit(
            SetAttributeRequest(
                session.document.entry_graph,
                node.id,
                "axis",
                AttrKind.INT,
                0,
            )
        )
        _assert_command_round_trip(attribute.commands[0])
        session.commit_edit(attribute)
        report = session.export(destination)

    model = onnx.load_model(destination)
    assert helper.get_attribute_value(model.graph.node[0].attribute[0]) == 0
    assert report.edit_manifest[0]["kind"] == "set-attribute"
    onnx.checker.check_model(destination, full_check=True)


def test_remove_and_reconnect_commands_export_in_order(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    _rewire_model(source)
    destination = tmp_path / "rewired.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        graph = session.document.main_graph
        target = graph.nodes[1]
        insertion = session.prepare_edit(
            InsertUnaryRequest(graph.id, target.id, 0, "Relu")
        )
        _assert_command_round_trip(insertion.commands[0])
        session.commit_edit(insertion)
        inserted = insertion.commands[0]
        assert isinstance(inserted, InsertUnaryNode)
        removal = session.prepare_edit(RemoveUnaryRequest(graph.id, inserted.node.id))
        _assert_command_round_trip(removal.commands[0])
        session.commit_edit(removal)
        input_id = session.document.main_graph.inputs[1]
        reconnect = session.prepare_edit(
            ReconnectInputRequest(graph.id, target.id, 0, input_id)
        )
        _assert_command_round_trip(reconnect.commands[0])
        session.commit_edit(reconnect)
        report = session.export(destination)

    model = onnx.load_model(destination)
    assert [node.op_type for node in model.graph.node] == ["Identity", "Add"]
    assert model.graph.node[1].input[0] == "right"
    assert [entry["kind"] for entry in report.edit_manifest] == [
        "insert-unary-node",
        "remove-unary-node",
        "reconnect-input",
    ]
    onnx.checker.check_model(destination, full_check=True)


def test_external_weight_edit_streams_to_a_relocated_complete_export(
    tmp_path: Path,
) -> None:
    source, values = build_external_model(tmp_path, elements=16)
    weights = tmp_path / "weights.bin"
    source_hashes = (hash_file(source), hash_file(weights))
    destination = tmp_path / "external-edited.onnx"
    replacement = struct.pack("<f", 123.25)

    with ApplicationService() as service:
        session = service.open_model(source)
        tensor_id = session.document.main_graph.initializers[0]
        session.editing.replace_bytes(tensor_id, 0, replacement)
        report = session.export(destination)

    assert (hash_file(source), hash_file(weights)) == source_hashes
    exported = index_model(destination)
    tensor = exported.main_graph.initializers[0]
    raw = read_tensor_bytes(exported, tensor)
    assert raw[:4] == replacement
    assert raw[4:] == values.tobytes()[4:]
    assert exported.external_files
    assert all(path != weights for path in exported.external_files)
    assert report.warnings
    assert report.report_path.exists()


def test_external_weight_edit_never_grades_lossless(tmp_path: Path) -> None:
    """An external-file weight splice changes the model like an embedded one.

    Audit finding: exporting an edited external tensor to a *different*
    directory previously reported LOSSLESS because only embedded edits and
    relocations were counted.
    """
    source, _values = build_external_model(tmp_path, elements=16)
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    destination = other_dir / "edited.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        tensor_id = session.document.main_graph.initializers[0]
        session.editing.replace_bytes(tensor_id, 0, struct.pack("<f", 9.75))
        report = session.export(destination)

    assert report.fidelity.value == "semantically equivalent"
    assert "IR invariants passed" not in report.structural_checks, (
        "the report only lists checks the exporter performed"
    )
    assert "onnx.checker full_check passed" in report.structural_checks


def test_typed_tensor_edit_exports_as_valid_raw_data(tmp_path: Path) -> None:
    source = tmp_path / "typed.onnx"
    build_embedded_model(source, elements=8)
    destination = tmp_path / "typed-edited.onnx"
    replacement = struct.pack("<f", -7.5)

    with ApplicationService() as service:
        session = service.open_model(source)
        bias_id = session.document.main_graph.initializers[1]
        session.editing.replace_bytes(bias_id, 0, replacement)
        session.export(destination)

    reopened = index_model(destination)
    bias = reopened.main_graph.initializers[1]
    assert read_tensor_bytes(reopened, bias) == replacement


def test_structural_export_externalizes_large_embedded_tensors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "embedded.onnx"
    build_embedded_model(source, elements=32)
    destination = tmp_path / "externalized.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        node = session.document.main_graph.nodes[0]
        session.commit_edit(
            session.prepare_edit(
                RenameNodeRequest(session.document.entry_graph, node.id, "renamed")
            )
        )
        report = session.export(destination, external_data_threshold=1)

    reopened = index_model(destination)
    assert reopened.external_files
    assert any(
        path.name == f"{destination.name}.data" for path in reopened.external_files
    )
    assert any(path.name == f"{destination.name}.data" for path in report.written_files)


def test_custom_domains_and_symbolic_shapes_are_disclosed(tmp_path: Path) -> None:
    custom = tmp_path / "custom.onnx"
    build_custom_domain_model(custom)
    symbolic = tmp_path / "symbolic.onnx"
    build_symbolic_shape_model(symbolic)

    with ApplicationService() as service:
        custom_report = service.open_model(custom).export(tmp_path / "custom-copy.onnx")
        symbolic_report = service.open_model(symbolic).export(
            tmp_path / "symbolic-copy.onnx"
        )

    assert "com.example.ops::FusedGelu" in custom_report.unsupported_operators
    assert custom_report.custom_runtime_needs == ("com.example.ops",)
    assert symbolic_report.unresolved_shapes


def test_existing_destination_and_changed_source_publish_nothing(
    tmp_path: Path,
) -> None:
    source, _values = build_external_model(tmp_path, elements=8)
    destination = tmp_path / "blocked.onnx"
    destination.write_bytes(b"keep")
    with ApplicationService() as service:
        session = service.open_model(source)
        with pytest.raises(ExportError, match="already exists"):
            session.export(destination)
    assert destination.read_bytes() == b"keep"
    assert not (tmp_path / "blocked.onnx.report.json").exists()

    destination.unlink()
    with ApplicationService() as service:
        session = service.open_model(source)
        (tmp_path / "weights.bin").write_bytes(b"\xff" * 32)
        with pytest.raises(ExportError, match="changed after open"):
            session.export(destination)
    assert not destination.exists()
    assert not (tmp_path / "blocked.onnx.report.json").exists()


def test_publish_failure_rolls_back_every_output(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source, _values = build_external_model(tmp_path, elements=8)
    destination = tmp_path / "rollback.onnx"
    report_path = tmp_path / "rollback.onnx.report.json"
    original_replace = os.replace

    def fail_report_publish(source_path: Path | str, target_path: Path | str) -> None:
        if Path(target_path) == report_path:
            raise OSError("simulated publish failure")
        original_replace(source_path, target_path)

    monkeypatch.setattr(os, "replace", fail_report_publish)
    with ApplicationService() as service:
        session = service.open_model(source)
        with pytest.raises(ExportError, match="simulated publish failure"):
            session.export(destination)

    assert not destination.exists()
    assert not report_path.exists()
    assert not (tmp_path / "rollback.data").exists()


def test_cancelled_export_publishes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=8)
    destination = tmp_path / "cancelled.onnx"
    token = CancellationToken()
    token.cancel()

    with ApplicationService() as service:
        session = service.open_model(source)
        with pytest.raises(OperationCancelled):
            session.export(destination, token=token)

    assert not destination.exists()
    assert not (tmp_path / "cancelled.onnx.report.json").exists()


def test_opt_in_numerical_smoke_is_isolated_and_reported(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=4)
    destination = tmp_path / "numerical.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        node = session.document.main_graph.nodes[0]
        session.commit_edit(
            session.prepare_edit(
                RenameNodeRequest(session.document.entry_graph, node.id, "renamed")
            )
        )
        report = session.export(
            destination,
            smoke_inputs={"input": np.ones(4, dtype=np.float32)},
            approve_numerical_execution=True,
        )

    assert report.numerical_comparison is not None
    assert report.numerical_comparison["passed"] is True
    assert "supplied inputs" in str(report.numerical_comparison["claim"])


def test_numerical_smoke_requires_explicit_approval(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    build_embedded_model(source, elements=4)
    destination = tmp_path / "not-approved.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        with pytest.raises(ExportError, match="explicit caller approval"):
            session.export(
                destination,
                smoke_inputs={"input": np.ones(4, dtype=np.float32)},
            )

    assert not destination.exists()
