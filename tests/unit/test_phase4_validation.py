"""Phase 4 command preparation, validation, persistence, and shell workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Never, cast

import flet as ft
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService
from nneditor.editing.commands import InsertUnaryNode
from nneditor.editing.cow import EditError
from nneditor.editing.validation import (
    FindingLevel,
    InsertUnaryRequest,
    ReconnectInputRequest,
    RemoveUnaryRequest,
    RenameNodeRequest,
    ReplaceOperatorRequest,
    SetAttributeRequest,
    ValidationFinding,
    ValidationPipeline,
)
from nneditor.ir.capabilities import Availability, Capability, CapabilityStatus
from nneditor.ir.core import AttrKind, Document, Graph
from nneditor.ui.app import Shell
from tests.fixtures.onnx_models import build_embedded_model
from tests.unit.test_shell import StubPage


def _softmax_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], name="prob", axis=1)],
        "attribute-edit",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
    )
    onnx.save_model(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
        ),
        path,
    )


def _session(tmp_path: Path) -> tuple[ApplicationService, Any]:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    service = ApplicationService()
    return service, service.open_model(path)


def test_rename_is_prepared_then_committed_atomically(tmp_path: Path) -> None:
    service, session = _session(tmp_path)
    with service:
        node = session.document.main_graph.nodes[0]
        transaction = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "scaled")
        )
        assert transaction.ok
        assert session.document.main_graph.node(node.id).source_name == "scale"

        session.commit_edit(transaction)
        assert session.document.main_graph.node(node.id).source_name == "scaled"
        assert session.editing.can_undo
        assert session.editing.preview().graph_changes
        session.undo_edit()
        assert session.document.main_graph.node(node.id).source_name == "scale"
        session.redo_edit()
        assert session.document.main_graph.node(node.id).source_name == "scaled"


def test_edit_validation_is_independent_from_export_availability(
    tmp_path: Path,
) -> None:
    service, session = _session(tmp_path)
    with service:
        source = session.document
        capabilities = [
            (
                CapabilityStatus(
                    Capability.EXPORT,
                    Availability.UNAVAILABLE,
                    "No writer is installed.",
                )
                if status.capability is Capability.EXPORT
                else status
            )
            for status in source.capabilities.values()
        ]
        document = Document(
            source=source.source,
            artifact_kind=source.artifact_kind,
            capabilities=capabilities,
            graphs=source.graphs.values(),
            tensors=source.tensors.values(),
            entry_graph=source.entry_graph,
            provenance=source.provenance,
            diagnostics=source.diagnostics,
            capability_notes=source.capability_notes,
            extensions=source.extensions,
        )
        node = document.main_graph.nodes[0]
        transaction = ValidationPipeline().prepare(
            document,
            RenameNodeRequest(document.entry_graph, node.id, "independent"),
            base_revision_id=None,
        )
        assert transaction.ok
        assert all(
            item.code != "edit.export-unavailable" for item in transaction.findings
        )


def test_validation_request_handlers_are_replaceable(tmp_path: Path) -> None:
    service, session = _session(tmp_path)
    calls: list[str] = []

    def reject_rename(
        _document: Document,
        _graph: Graph,
        request: object,
        _base_revision_id: str | None,
        _findings: list[ValidationFinding],
    ) -> Never:
        calls.append(type(request).__name__)
        raise EditError("custom rename policy")

    with service:
        node = session.document.main_graph.nodes[0]
        pipeline = ValidationPipeline({RenameNodeRequest: reject_rename})
        transaction = pipeline.prepare(
            session.document,
            RenameNodeRequest(session.document.entry_graph, node.id, "blocked"),
            base_revision_id=None,
        )

    assert calls == ["RenameNodeRequest"]
    assert not transaction.ok
    assert any("custom rename policy" in item.message for item in transaction.findings)


def test_reject_and_failed_validation_create_no_revision(tmp_path: Path) -> None:
    service, session = _session(tmp_path)
    with service:
        node = session.document.main_graph.nodes[0]
        invalid = session.prepare_edit(
            ReplaceOperatorRequest(
                session.document.entry_graph,
                node.id,
                "Relu",
            )
        )
        assert not invalid.ok
        assert any(
            item.code == "edit.precondition" and item.level is FindingLevel.ERROR
            for item in invalid.findings
        )
        session.reject_edit(invalid)
        assert not session.editing.is_dirty


def test_attribute_edits_are_schema_typed(tmp_path: Path) -> None:
    path = tmp_path / "softmax.onnx"
    _softmax_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        node = session.document.main_graph.nodes[0]
        valid = session.prepare_edit(
            SetAttributeRequest(
                session.document.entry_graph,
                node.id,
                "axis",
                AttrKind.INT,
                0,
            )
        )
        assert valid.ok
        session.commit_edit(valid)
        assert session.document.main_graph.node(node.id).attribute("axis").value == 0

        invalid = session.prepare_edit(
            SetAttributeRequest(
                session.document.entry_graph,
                node.id,
                "axis",
                AttrKind.STRING,
                "zero",
            )
        )
        assert not invalid.ok
        assert any(item.code == "edit.attribute-type" for item in invalid.findings)


def test_operator_insert_remove_and_reconnect_preconditions(tmp_path: Path) -> None:
    service, session = _session(tmp_path)
    with service:
        graph = session.document.main_graph
        scale, shift = graph.nodes

        replacement = session.prepare_edit(
            ReplaceOperatorRequest(graph.id, scale.id, "Add")
        )
        assert replacement.ok
        session.commit_edit(replacement)
        assert session.document.main_graph.node(scale.id).op_type == "Add"

        insertion = session.prepare_edit(
            InsertUnaryRequest(graph.id, shift.id, 0, "Relu")
        )
        assert insertion.ok
        session.commit_edit(insertion)
        inserted = insertion.commands[0]
        assert isinstance(inserted, InsertUnaryNode)
        inserted_id = inserted.node.id
        assert session.document.main_graph.node(inserted_id).op_type == "Relu"

        removal = session.prepare_edit(RemoveUnaryRequest(graph.id, inserted_id))
        assert removal.ok
        assert any(finding.code == "edit.valid" for finding in removal.findings), (
            "the rewired consumer passed its schema pass"
        )
        session.commit_edit(removal)
        assert all(node.id != inserted_id for node in session.document.main_graph.nodes)

        incompatible_value = session.document.main_graph.node(scale.id).inputs[0]
        incompatible = session.prepare_edit(
            ReconnectInputRequest(graph.id, shift.id, 1, incompatible_value)
        )
        assert not incompatible.ok
        assert "does not match" in " ".join(
            item.message for item in incompatible.findings
        )

        unresolved = session.prepare_edit(
            ReconnectInputRequest(
                graph.id,
                shift.id,
                0,
                session.document.main_graph.inputs[0],
            )
        )
        assert not unresolved.ok
        assert "unresolved" in " ".join(item.message for item in unresolved.findings)

        shape_changing = session.prepare_edit(
            InsertUnaryRequest(graph.id, shift.id, 0, "Flatten")
        )
        assert not shape_changing.ok
        assert "preserve shape and dtype" in " ".join(
            item.message for item in shape_changing.findings
        )

        unmodelled_replacement = session.prepare_edit(
            ReplaceOperatorRequest(graph.id, scale.id, "MatMul")
        )
        assert not unmodelled_replacement.ok
        assert "shape/dtype-preserving family" in " ".join(
            item.message for item in unmodelled_replacement.findings
        )

        cyclic = session.prepare_edit(
            ReconnectInputRequest(
                graph.id,
                scale.id,
                0,
                session.document.main_graph.outputs[0],
            )
        )
        assert not cyclic.ok
        assert "graph cycle" in " ".join(item.message for item in cyclic.findings)


def test_a_stale_transaction_cannot_commit(tmp_path: Path) -> None:
    service, session = _session(tmp_path)
    with service:
        node = session.document.main_graph.nodes[0]
        first = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "first")
        )
        stale = session.prepare_edit(
            RenameNodeRequest(session.document.entry_graph, node.id, "stale")
        )
        session.commit_edit(first)
        with pytest.raises(EditError, match="changed after validation"):
            session.commit_edit(stale)
        assert session.document.main_graph.node(node.id).source_name == "first"


def test_graph_revisions_recover_from_the_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    state = tmp_path / "state"
    with ApplicationService(state_store=SessionStateStore(state)) as service:
        session = service.open_model(path)
        node = session.document.main_graph.nodes[0]
        session.commit_edit(
            session.prepare_edit(
                RenameNodeRequest(session.document.entry_graph, node.id, "persisted")
            )
        )

    with ApplicationService(state_store=SessionStateStore(state)) as service:
        session = service.open_model(path)
        assert session.document.main_graph.nodes[0].source_name == "persisted"
        assert session.editing.recovered_revisions == 1


def test_shell_exposes_validate_commit_reject_and_undo(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=16)
    with ApplicationService() as service:
        page = StubPage()
        shell = Shell(cast(ft.Page, page), service)
        shell.build()
        session = service.open_model(path)
        shell.show_session(session)
        node = session.document.main_graph.nodes[0]
        shell.renderer.set_selection(frozenset({node.id}))
        shell._on_selected(frozenset({node.id}))
        shell.edit_kind.value = "rename"
        shell.edit_primary.value = "ui-renamed"

        shell._on_validate_edit(cast(Any, None))
        assert shell.pending_edit is not None and shell.pending_edit.ok
        assert not shell.commit_edit_button.disabled
        prepared_text = " ".join(
            control.value or ""
            for control in shell.edit_findings.controls
            if isinstance(control, ft.Text)
        )
        assert "Preview" in prepared_text
        assert "No artifact capability availability changes" in prepared_text

        shell._on_commit_edit(cast(Any, None))
        assert session.document.main_graph.node(node.id).source_name == "ui-renamed"
        assert not shell.undo_edit_button.disabled
        committed_text = " ".join(
            control.value or ""
            for control in shell.edit_findings.controls
            if isinstance(control, ft.Text)
        )
        assert "Committed diff" in committed_text
        assert "rename" in committed_text
        shell._on_undo_edit(cast(Any, None))
        assert session.document.main_graph.node(node.id).source_name == "scale"


def test_removal_schema_checks_the_rewired_consumer(tmp_path: Path) -> None:
    """Removing a node must schema-validate the consumer it rewires.

    Audit finding: removals skipped `_schema_findings` entirely. The
    precondition layer only compares the *declared* dtypes across the rewire,
    so when those agree (float in, float out) but the consumer's schema
    demands another type entirely (Not requires bool), nothing objected. The
    consumer must now fail its dtype constraint after the rewire.
    """
    graph = helper.make_graph(
        [
            helper.make_node("Abs", ["input"], ["abs_out"], name="absolute"),
            helper.make_node("Not", ["abs_out"], ["output"], name="negate"),
        ],
        "removal_dtype",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [4])],
        value_info=[helper.make_tensor_value_info("abs_out", TensorProto.FLOAT, [4])],
    )
    path = tmp_path / "removal.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
    )

    with ApplicationService() as service:
        session = service.open_model(path)
        graph_ir = session.document.main_graph
        absolute = next(n for n in graph_ir.nodes if n.op_type == "Abs")
        removal = session.prepare_edit(
            RemoveUnaryRequest(session.document.entry_graph, absolute.id)
        )
        assert not removal.ok
        assert any(
            finding.code == "edit.dtype-constraint" for finding in removal.findings
        ), [str(f.message) for f in removal.findings]
        assert not session.editing.is_dirty
