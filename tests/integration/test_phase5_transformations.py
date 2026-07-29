"""Phase 5 quantization/pruning previews, revisions, and ONNX exports."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx.reference import ReferenceEvaluator

from nneditor.adapters.onnx import index_model, report_from_json
from nneditor.application.persistence import SessionStateStore
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.editing.commands import (
    QuantizeGraph,
    ReplaceTensorBytes,
    ResizeTensor,
    command_from_json,
    command_to_json,
)
from nneditor.editing.cow import EditError
from nneditor.ir.capabilities import ExportFidelity
from nneditor.ir.core import Storage
from nneditor.transformations.engine import (
    GraphQuantizationRequest,
    LogicalPruningRequest,
    StructuredPruningRequest,
    WeightQuantizationRequest,
)
from nneditor.transformations.schema import (
    Granularity,
    PruningMode,
    StorageEffect,
    TransformationKind,
)
from tests.fixtures.onnx_models import build_matmul_model


def _weight_id(session: ModelSession) -> str:
    return session.document.main_graph.initializers[0]


@pytest.mark.parametrize(
    ("granularity", "axis", "parameter_count"),
    [
        (Granularity.PER_TENSOR, None, 1),
        (Granularity.PER_CHANNEL, 1, 4),
    ],
)
def test_weight_quantization_is_previewable_reproducible_and_reversible(
    tmp_path: Path,
    granularity: Granularity,
    axis: int | None,
    parameter_count: int,
) -> None:
    path = tmp_path / f"weight-{granularity.value}.onnx"
    destination = tmp_path / f"weight-{granularity.value}-converted.onnx"
    original = build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        before = session.editing.read(tensor_id)
        proposal = session.prepare_transformation(
            WeightQuantizationRequest(
                tensor_id,
                granularity=granularity,
                axis=axis,
            )
        )

        assert proposal.ok
        assert proposal.manifest is not None
        assert proposal.preview is not None
        assert proposal.manifest.kind is TransformationKind.WEIGHT_QUANTIZATION
        assert proposal.manifest.storage_effect is StorageEffect.UNCHANGED
        assert len(proposal.manifest.scale) == parameter_count
        assert proposal.preview.mathematical_conversion
        assert proposal.preview.executable_graph
        assert not proposal.preview.storage_reduction
        assert not proposal.preview.expected_acceleration
        command = proposal.commands[0]
        assert isinstance(command, ReplaceTensorBytes)
        assert command_from_json(command_to_json(command)) == command

        session.commit_transformation(proposal)
        converted = np.frombuffer(
            session.editing.read(tensor_id),
            dtype="<f4",
        ).reshape(original.shape)
        assert not np.array_equal(converted, original)
        assert session.editing.preview().transformations[0].preview == proposal.preview
        session.undo_edit()
        assert session.editing.read(tensor_id) == before
        session.redo_edit()
        assert np.array_equal(
            np.frombuffer(session.editing.read(tensor_id), dtype="<f4").reshape(
                original.shape
            ),
            converted,
        )
        report = session.export(destination)

    exported = onnx.load_model(destination).graph.initializer[0]
    assert np.array_equal(
        np.frombuffer(exported.raw_data, dtype="<f4").reshape(original.shape),
        converted,
    )
    manifest = cast(dict[str, object], report.transformations[0]["manifest"])
    assert manifest["kind"] == "weight-quantization"


def test_qdq_quantization_exports_real_onnx_and_matches_preview_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    build_matmul_model(source, opset=19)
    destination = tmp_path / "qdq.onnx"
    input_values = np.asarray(
        [[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]],
        dtype=np.float32,
    )

    with ApplicationService() as service:
        session = service.open_model(source)
        tensor_id = _weight_id(session)
        proposal = session.prepare_transformation(
            GraphQuantizationRequest(
                session.document.entry_graph,
                tensor_id,
                granularity=Granularity.PER_CHANNEL,
                axis=1,
                calibration_provenance=(("provider", "test"), ("samples", "2")),
                backend_plugin="test.backend",
            )
        )
        assert proposal.ok and proposal.preview is not None
        assert proposal.manifest is not None
        # Weight-derived Q/DQ needs no activation calibration; the attached
        # provenance is carried without changing the requirement.
        assert not proposal.manifest.calibration_required
        assert proposal.manifest.calibration_provenance == (
            ("provider", "test"),
            ("samples", "2"),
        )
        assert proposal.manifest.backend_plugin == "test.backend"
        command = proposal.commands[0]
        assert isinstance(command, QuantizeGraph)
        assert command_from_json(command_to_json(command)) == command
        session.commit_transformation(proposal)
        assert (
            session.document.tensors[command.scale_tensor.id].storage
            is Storage.GENERATED
        )
        assert session.editing.read(command.scale_tensor.id) == command.scale_bytes
        session.undo_edit()
        assert command.scale_tensor.id not in session.document.tensors
        session.redo_edit()
        assert session.editing.read(command.scale_tensor.id) == command.scale_bytes
        assert not session.prepare_transformation(
            GraphQuantizationRequest(
                session.document.entry_graph,
                tensor_id,
            )
        ).ok
        report = session.export(destination)

    model = onnx.load_model(destination)
    assert [node.op_type for node in model.graph.node] == [
        "QuantizeLinear",
        "DequantizeLinear",
        "MatMul",
    ]
    assert {item.name for item in model.graph.initializer} >= {
        command.scale_value.name,
        command.zero_value.name,
    }
    assert model.graph.node[-1].input[1] == command.dequantized_value.name
    onnx.checker.check_model(model, full_check=True)
    expected = cast(
        list[NDArray[np.float32]],
        ReferenceEvaluator(str(source)).run(None, {"input": input_values}),
    )[0]
    actual = cast(
        list[NDArray[np.float32]],
        ReferenceEvaluator(str(destination)).run(
            None,
            {"input": input_values},
        ),
    )[0]
    assert np.allclose(actual, expected, atol=proposal.preview.max_abs_error * 4)
    assert report.fidelity is ExportFidelity.LOSSY
    exported_manifest = cast(
        dict[str, object],
        report.transformations[0]["manifest"],
    )
    assert exported_manifest["operator_representation"] == "onnx-qdq"
    assert report_from_json(report.report_path) == report
    assert index_model(destination).node_count == 3


def test_unsigned_asymmetric_weight_quantization_records_its_scheme(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsigned.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        proposal = session.prepare_transformation(
            WeightQuantizationRequest(
                _weight_id(session),
                signed=False,
                symmetric=False,
            )
        )
    assert proposal.ok and proposal.manifest is not None
    assert proposal.manifest.signed is False
    assert proposal.manifest.symmetric is False
    assert proposal.manifest.zero_point == (0,)


def test_qdq_rejects_old_opset_and_unsupported_scheme(tmp_path: Path) -> None:
    path = tmp_path / "old.onnx"
    build_matmul_model(path, opset=18)
    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        old = session.prepare_transformation(
            GraphQuantizationRequest(session.document.entry_graph, tensor_id)
        )
        unsupported = session.prepare_transformation(
            WeightQuantizationRequest(tensor_id, bit_width=4)
        )
        unsigned_symmetric = session.prepare_transformation(
            WeightQuantizationRequest(tensor_id, signed=False, symmetric=True)
        )

    assert not old.ok and "opset 19" in old.findings[0].message
    assert not unsupported.ok and "only 8-bit" in unsupported.findings[0].message
    assert (
        not unsigned_symmetric.ok
        and "requires signed" in unsigned_symmetric.findings[0].message
    )


@pytest.mark.parametrize(
    "prune_request",
    [
        LogicalPruningRequest(
            "placeholder",
            PruningMode.MASK,
            mask=(
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
            ),
        ),
        LogicalPruningRequest(
            "placeholder",
            PruningMode.THRESHOLD,
            threshold=5.0,
        ),
        LogicalPruningRequest(
            "placeholder",
            PruningMode.NM,
            n=2,
            m=4,
        ),
    ],
    ids=["mask", "threshold", "2:4"],
)
def test_logical_pruning_zeroes_weights_without_storage_or_speed_claims(
    tmp_path: Path,
    prune_request: LogicalPruningRequest,
) -> None:
    path = tmp_path / f"{prune_request.mode.value.replace(':', '-')}.onnx"
    build_matmul_model(path)
    destination = tmp_path / f"{path.stem}-pruned.onnx"
    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        proposal = session.prepare_transformation(
            LogicalPruningRequest(
                tensor_id,
                prune_request.mode,
                threshold=prune_request.threshold,
                mask=prune_request.mask,
                n=prune_request.n,
                m=prune_request.m,
            )
        )
        assert proposal.ok and proposal.preview is not None
        assert proposal.preview.after_sparsity > proposal.preview.before_sparsity
        assert not proposal.preview.storage_reduction
        assert not proposal.preview.expected_acceleration
        session.commit_transformation(proposal)
        report = session.export(destination)

    initializer = onnx.load_model(destination).graph.initializer[0]
    values = np.frombuffer(initializer.raw_data, dtype="<f4")
    assert np.count_nonzero(values == 0) > 1
    assert report.fidelity is ExportFidelity.LOSSY
    manifest = cast(dict[str, object], report.transformations[0]["manifest"])
    assert manifest["storage_effect"] == "unchanged"


def test_invalid_logical_pruning_requests_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid-prune.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        proposals = (
            session.prepare_transformation(
                LogicalPruningRequest(tensor_id, PruningMode.MASK, mask=(True,))
            ),
            session.prepare_transformation(
                LogicalPruningRequest(
                    tensor_id,
                    PruningMode.THRESHOLD,
                    threshold=-1,
                )
            ),
            session.prepare_transformation(
                LogicalPruningRequest(tensor_id, PruningMode.NM, n=4, m=4)
            ),
        )
    assert all(not proposal.ok for proposal in proposals)


def test_rejected_and_stale_transformation_previews_never_mutate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "atomic.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        stale = session.prepare_transformation(WeightQuantizationRequest(tensor_id))
        rejected = session.prepare_transformation(
            WeightQuantizationRequest(tensor_id, bit_width=4)
        )
        before = session.editing.read(tensor_id)
        with pytest.raises(EditError, match="validation errors"):
            session.commit_transformation(rejected)
        assert not session.editing.applied_revisions
        assert session.editing.read(tensor_id) == before

        current = session.prepare_transformation(
            LogicalPruningRequest(
                tensor_id,
                PruningMode.THRESHOLD,
                threshold=5,
            )
        )
        session.commit_transformation(current)
        with pytest.raises(EditError, match="preview again"):
            session.commit_transformation(stale)
        assert len(session.editing.applied_revisions) == 1


def test_terminal_matmul_structured_pruning_propagates_shapes_and_reopens(
    tmp_path: Path,
) -> None:
    source = tmp_path / "matmul.onnx"
    original = build_matmul_model(source)
    destination = tmp_path / "matmul-small.onnx"
    with ApplicationService() as service:
        session = service.open_model(source)
        graph = session.document.main_graph
        tensor_id = _weight_id(session)
        proposal = session.prepare_transformation(
            StructuredPruningRequest(graph.id, graph.nodes[0].id, (0, 2))
        )
        assert proposal.ok and proposal.preview is not None
        assert proposal.preview.storage_reduction
        command = proposal.commands[0]
        assert isinstance(command, ResizeTensor)
        assert command_from_json(command_to_json(command)) == command
        session.commit_transformation(proposal)
        assert session.document.tensors[tensor_id].dims == (3, 2)
        assert session.document.main_graph.value(graph.outputs[0]).shape == (2, 2)
        assert session.editing.byte_length(tensor_id) == 3 * 2 * 4
        session.undo_edit()
        assert session.document.tensors[tensor_id].dims == (3, 4)
        session.redo_edit()
        assert session.document.tensors[tensor_id].dims == (3, 2)
        report = session.export(destination)

    model = onnx.load_model(destination)
    assert tuple(model.graph.initializer[0].dims) == (3, 2)
    assert tuple(
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ) == (2, 2)
    exported = np.frombuffer(model.graph.initializer[0].raw_data, dtype="<f4")
    assert np.array_equal(exported.reshape(3, 2), original[:, (0, 2)])
    onnx.checker.check_model(model, full_check=True)
    assert any("tensor added" not in item for item in report.dtype_layout_changes)
    exported_preview = cast(
        dict[str, object],
        report.transformations[0]["preview"],
    )
    assert exported_preview["storage_reduction"] is True


def test_structured_pruning_rejects_unproven_patterns(tmp_path: Path) -> None:
    path = tmp_path / "nonterminal.onnx"
    build_matmul_model(path, terminal=False)
    with ApplicationService() as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        proposal = session.prepare_transformation(
            StructuredPruningRequest(graph.id, graph.nodes[0].id, (0, 2))
        )
        unsorted = session.prepare_transformation(
            StructuredPruningRequest(graph.id, graph.nodes[0].id, (2, 0))
        )
    assert not proposal.ok and "terminal MatMul" in proposal.findings[0].message
    assert not unsorted.ok


def test_structured_pruning_rejects_a_shared_weight_initializer(
    tmp_path: Path,
) -> None:
    """A weight feeding two MatMuls must never be resized (audit finding)."""
    from tests.fixtures.onnx_models import build_shared_weight_matmul_model

    path = tmp_path / "shared.onnx"
    build_shared_weight_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        for node in graph.nodes:
            proposal = session.prepare_transformation(
                StructuredPruningRequest(graph.id, node.id, (0, 2))
            )
            assert not proposal.ok
            assert "cannot be proven consistent" in proposal.findings[0].message
        assert not session.editing.is_dirty, "rejection mutates nothing"


def test_structured_pruning_replaces_an_external_weight_without_copying_orphan_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "external-matmul.onnx"
    build_matmul_model(source)
    model = onnx.load_model(source)
    onnx.save_model(
        model,
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    external = tmp_path / "weights.bin"
    source_bytes = source.read_bytes()
    external_bytes = external.read_bytes()
    destination = tmp_path / "structured-embedded.onnx"

    with ApplicationService() as service:
        session = service.open_model(source)
        graph = session.document.main_graph
        proposal = session.prepare_transformation(
            StructuredPruningRequest(graph.id, graph.nodes[0].id, (1, 3))
        )
        assert proposal.ok
        session.commit_transformation(proposal)
        report = session.export(destination, external_data_threshold=1 << 30)

    exported = onnx.load_model(destination, load_external_data=False)
    weight = exported.graph.initializer[0]
    assert weight.data_location == onnx.TensorProto.DEFAULT
    assert tuple(weight.dims) == (3, 2)
    assert weight.raw_data
    assert source.read_bytes() == source_bytes
    assert external.read_bytes() == external_bytes
    assert not any("weights.bin" in str(path) for path in report.written_files)


def test_transformation_revision_recovers_from_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "recover.onnx"
    build_matmul_model(path)
    state = tmp_path / "state"
    with ApplicationService(state_store=SessionStateStore(state)) as service:
        session = service.open_model(path)
        tensor_id = _weight_id(session)
        proposal = session.prepare_transformation(
            LogicalPruningRequest(
                tensor_id,
                PruningMode.THRESHOLD,
                threshold=5,
            )
        )
        session.commit_transformation(proposal)
        committed = session.editing.read(tensor_id)

    with ApplicationService(state_store=SessionStateStore(state)) as service:
        recovered = service.open_model(path)
        assert recovered.editing.recovered_revisions == 1
        assert recovered.editing.read(_weight_id(recovered)) == committed


@pytest.mark.parametrize("kind", ["qdq", "structured"])
def test_generated_transformation_state_recovers_from_sidecar(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / f"recover-{kind}.onnx"
    build_matmul_model(path, opset=19)
    state = tmp_path / f"state-{kind}"
    with ApplicationService(state_store=SessionStateStore(state)) as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        proposal = (
            session.prepare_transformation(
                GraphQuantizationRequest(graph.id, graph.initializers[0])
            )
            if kind == "qdq"
            else session.prepare_transformation(
                StructuredPruningRequest(graph.id, graph.nodes[0].id, (0, 2))
            )
        )
        assert proposal.ok
        session.commit_transformation(proposal)
        generated = {
            tensor_id: session.editing.read(tensor_id)
            for tensor_id, tensor in session.document.tensors.items()
            if tensor.storage is Storage.GENERATED
        }

    with ApplicationService(state_store=SessionStateStore(state)) as service:
        recovered = service.open_model(path)
        assert recovered.editing.recovered_revisions == 1
        assert {
            tensor_id: recovered.editing.read(tensor_id)
            for tensor_id, tensor in recovered.document.tensors.items()
            if tensor.storage is Storage.GENERATED
        } == generated
