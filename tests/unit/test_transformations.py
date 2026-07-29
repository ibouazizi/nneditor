"""Phase 5 schemas and resource-bounded calibration plugin boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Never, cast

import flet as ft
import numpy as np
import pytest
from numpy.typing import NDArray

import nneditor.transformations as transformations
from nneditor.application.session import ApplicationService
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.editing.cow import EditError
from nneditor.ir.core import Document
from nneditor.transformations.calibration import (
    CalibrationLimits,
    run_calibration,
)
from nneditor.transformations.engine import (
    GraphQuantizationRequest,
    LogicalPruningRequest,
    StructuredPruningRequest,
    TransformationEngine,
    WeightQuantizationRequest,
)
from nneditor.transformations.schema import (
    Granularity,
    OperatorRepresentation,
    PruningMode,
    StorageEffect,
    TargetRuntime,
    TransformationKind,
    TransformationManifest,
    TransformationPreview,
)
from nneditor.ui.app import Shell
from tests.fixtures.onnx_models import build_matmul_model
from tests.unit.test_shell import StubPage


def _manifest() -> TransformationManifest:
    return TransformationManifest(
        kind=TransformationKind.GRAPH_QUANTIZATION,
        target_runtime=TargetRuntime.ONNX_REFERENCE,
        operator_representation=OperatorRepresentation.ONNX_QDQ,
        storage_effect=StorageEffect.INCREASED,
        bit_width=8,
        signed=True,
        symmetric=True,
        granularity=Granularity.PER_CHANNEL,
        scale=(0.25, 0.5),
        zero_point=(0, 0),
        axis=1,
        calibration_required=True,
        parameters=(("z", "last"), ("a", "first")),
        calibration_provenance=(("samples", "4"), ("provider", "unit")),
        backend_plugin="unit.backend",
    )


def _preview() -> TransformationPreview:
    return TransformationPreview(
        tensor_id="tensor",
        element_count=4,
        changed_elements=3,
        source_bytes=16,
        result_bytes=18,
        theoretical_packed_bytes=4,
        max_abs_error=0.1,
        mean_abs_error=0.025,
        rmse=0.05,
        error_basis="Elementwise error over the candidate.",
        before_sparsity=0.0,
        after_sparsity=0.25,
        mathematical_conversion=True,
        executable_graph=True,
        storage_reduction=False,
        expected_acceleration=False,
        acceleration_reason="Backend acceleration was not benchmarked.",
    )


def test_manifest_and_preview_round_trip_with_canonical_provenance() -> None:
    manifest = _manifest()
    preview = _preview()

    assert manifest.parameters == (("a", "first"), ("z", "last"))
    assert manifest.calibration_provenance == (
        ("provider", "unit"),
        ("samples", "4"),
    )
    assert TransformationManifest.from_json(manifest.to_json()) == manifest
    assert TransformationPreview.from_json(preview.to_json()) == preview
    assert transformations.TransformationEngine is TransformationEngine
    missing_export = "DoesNotExist"
    with pytest.raises(AttributeError):
        getattr(transformations, missing_export)


def test_transformation_handlers_are_replaceable(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_matmul_model(path)
    calls: list[str] = []

    def reject_weight_transform(
        _document: Document,
        request: object,
        _base_revision_id: str | None,
        _token: CancellationToken | None,
    ) -> Never:
        calls.append(type(request).__name__)
        raise EditError("custom transformation policy")

    with ApplicationService() as service:
        session = service.open_model(path)
        tensor_id = session.document.main_graph.initializers[0]
        engine = TransformationEngine(
            session.editing.read,
            handlers={WeightQuantizationRequest: reject_weight_transform},
        )
        proposal = engine.prepare(
            session.document,
            WeightQuantizationRequest(tensor_id),
            base_revision_id=None,
        )

    assert calls == ["WeightQuantizationRequest"]
    assert not proposal.ok
    assert any(
        "custom transformation policy" in finding.message
        for finding in proposal.findings
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: TransformationManifest(
            TransformationKind.WEIGHT_QUANTIZATION,
            TargetRuntime.PORTABLE_ONNX,
            OperatorRepresentation.DEQUANTIZED_FLOAT,
            StorageEffect.UNCHANGED,
            bit_width=0,
        ),
        lambda: TransformationManifest(
            TransformationKind.WEIGHT_QUANTIZATION,
            TargetRuntime.PORTABLE_ONNX,
            OperatorRepresentation.DEQUANTIZED_FLOAT,
            StorageEffect.UNCHANGED,
            bit_width=8,
            signed=True,
            symmetric=True,
            granularity=Granularity.PER_CHANNEL,
        ),
        lambda: TransformationManifest(
            TransformationKind.LOGICAL_PRUNING,
            TargetRuntime.PORTABLE_ONNX,
            OperatorRepresentation.UNCHANGED_GRAPH,
            StorageEffect.UNCHANGED,
        ),
        lambda: TransformationManifest(
            TransformationKind.LOGICAL_PRUNING,
            TargetRuntime.PORTABLE_ONNX,
            OperatorRepresentation.UNCHANGED_GRAPH,
            StorageEffect.UNCHANGED,
            pruning_mode=PruningMode.MASK,
            backend_plugin=" ",
        ),
        lambda: TransformationManifest(
            TransformationKind.LOGICAL_PRUNING,
            TargetRuntime.PORTABLE_ONNX,
            OperatorRepresentation.UNCHANGED_GRAPH,
            StorageEffect.UNCHANGED,
            pruning_mode=PruningMode.MASK,
            parameters=(("same", "1"), ("same", "2")),
        ),
    ],
)
def test_invalid_manifest_combinations_are_rejected(
    build: Callable[[], object],
) -> None:
    with pytest.raises(EditError):
        build()


@pytest.mark.parametrize(
    "change",
    [
        {"changed_elements": 5},
        {"source_bytes": -1},
        {"before_sparsity": -0.1},
        {"after_sparsity": 1.1},
        {"acceleration_reason": " "},
    ],
)
def test_invalid_preview_counts_and_claims_are_rejected(
    change: dict[str, object],
) -> None:
    payload = _preview().to_json()
    payload.update(change)
    with pytest.raises(EditError):
        TransformationPreview.from_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"kind": "missing"},
        {**_manifest().to_json(), "signed": "true"},
        {**_manifest().to_json(), "scale": "bad"},
        {**_manifest().to_json(), "parameters": []},
    ],
)
def test_malformed_manifest_payloads_are_rejected(payload: object) -> None:
    with pytest.raises(EditError):
        TransformationManifest.from_json(payload)


@pytest.mark.parametrize("payload", [[], {}, {**_preview().to_json(), "rmse": "bad"}])
def test_malformed_preview_payloads_are_rejected(payload: object) -> None:
    with pytest.raises(EditError):
        TransformationPreview.from_json(payload)


class RecordingProvider:
    provider_id = "unit.calibrator"
    version = "1.2"

    def __init__(self, *, result: float = 3.5) -> None:
        self.observed: list[tuple[str, ...]] = []
        self.result = result

    def observe(
        self,
        sample: Mapping[str, NDArray[np.generic]],
        token: CancellationToken,
    ) -> None:
        token.raise_if_cancelled()
        self.observed.append(tuple(sorted(sample)))

    def finish(self, token: CancellationToken) -> Mapping[str, float]:
        token.raise_if_cancelled()
        return {"maximum": self.result}


def _samples() -> list[dict[str, NDArray[np.generic]]]:
    return [
        {"input": np.ones((2, 3), dtype=np.float32)},
        {"input": np.zeros((1, 3), dtype=np.float32)},
    ]


def test_calibration_reports_bounded_provenance() -> None:
    provider = RecordingProvider()
    result = run_calibration(
        provider,
        _samples(),
        token=CancellationToken(),
        limits=CalibrationLimits(max_samples=2, max_total_bytes=36),
    )

    assert provider.observed == [("input",), ("input",)]
    assert result.sample_count == 2
    assert result.total_bytes == 36
    assert result.parameters == (("maximum", 3.5),)
    assert dict(result.provenance) == {
        "provider": "unit.calibrator",
        "provider_version": "1.2",
        "samples": "2",
        "bytes": "36",
    }


def test_calibration_is_cancellable_and_enforces_resource_limits() -> None:
    cancelled = CancellationToken()
    cancelled.cancel()
    with pytest.raises(OperationCancelled):
        run_calibration(RecordingProvider(), _samples(), token=cancelled)
    with pytest.raises(EditError, match="sample limit"):
        run_calibration(
            RecordingProvider(),
            _samples(),
            token=CancellationToken(),
            limits=CalibrationLimits(max_samples=1),
        )
    with pytest.raises(EditError, match="byte limit"):
        run_calibration(
            RecordingProvider(),
            _samples(),
            token=CancellationToken(),
            limits=CalibrationLimits(max_total_bytes=10),
        )


def test_calibration_rejects_invalid_provider_output_and_limits() -> None:
    with pytest.raises(EditError, match="positive"):
        CalibrationLimits(max_samples=0)
    provider = RecordingProvider(result=float("nan"))
    with pytest.raises(EditError, match="finite"):
        run_calibration(provider, [], token=CancellationToken())
    provider.provider_id = " "
    with pytest.raises(EditError, match="id and version"):
        run_calibration(provider, [], token=CancellationToken())


def test_calibration_runs_through_the_session_job_boundary(tmp_path: Path) -> None:
    path = tmp_path / "calibration.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        job = session.calibrate_async(RecordingProvider(), _samples())
        result = job.result(timeout=5)
    assert result.sample_count == 2
    assert result.parameters == (("maximum", 3.5),)


def test_shell_previews_capability_claims_before_applying(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ui-transform.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        page = StubPage()
        shell = Shell(cast(ft.Page, page), service)
        shell.build()
        session = service.open_model(path)
        shell.show_session(session)
        node = session.document.main_graph.nodes[0]
        shell.renderer.set_selection(frozenset({node.id}))
        shell._on_selected(frozenset({node.id}))

        shell.transformation_kind.value = "weight-quantization"
        shell._on_preview_transformation(cast(Any, None))
        assert shell.pending_transformation is not None
        assert shell.pending_transformation.ok
        assert not shell.commit_transformation_button.disabled
        text = " ".join(
            control.value or ""
            for control in shell.transformation_findings.controls
            if isinstance(control, ft.Text)
        )
        assert "Mathematical conversion: yes" in text
        assert "Executable support: claimed (verified at export" in text
        assert "Executable support: verified\n" not in text, (
            "preview must not claim verification before export runs it"
        )
        assert "Storage reduction: no" in text
        assert "Expected acceleration: no" in text

        shell._on_commit_transformation(cast(Any, None))
        assert shell.pending_transformation is None
        committed = " ".join(
            control.value or ""
            for control in shell.edit_findings.controls
            if isinstance(control, ft.Text)
        )
        assert "weight-quantization" in committed
        assert "expected acceleration: no" in committed


def test_engine_rejects_unsupported_targets_axes_and_pruning_shapes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rejected.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        tensor_id = graph.initializers[0]
        requests = (
            WeightQuantizationRequest(
                tensor_id,
                target_runtime=TargetRuntime.ONNX_REFERENCE,
            ),
            WeightQuantizationRequest(
                tensor_id,
                granularity=Granularity.PER_CHANNEL,
            ),
            WeightQuantizationRequest(
                tensor_id,
                granularity=Granularity.PER_CHANNEL,
                axis=9,
            ),
            GraphQuantizationRequest(
                graph.id,
                tensor_id,
                target_runtime=TargetRuntime.PORTABLE_ONNX,
            ),
            LogicalPruningRequest(
                tensor_id,
                PruningMode.THRESHOLD,
                target_runtime=TargetRuntime.ONNX_REFERENCE,
            ),
            LogicalPruningRequest(tensor_id, PruningMode.OUTPUT_CHANNELS),
            LogicalPruningRequest(
                tensor_id,
                PruningMode.NM,
                n=2,
                m=5,
            ),
            StructuredPruningRequest(
                graph.id,
                graph.nodes[0].id,
                (),
            ),
            StructuredPruningRequest(
                graph.id,
                graph.nodes[0].id,
                (0, 0),
            ),
            StructuredPruningRequest(
                graph.id,
                graph.nodes[0].id,
                (0, 4),
            ),
            StructuredPruningRequest(
                graph.id,
                graph.nodes[0].id,
                (0, 1, 2, 3),
            ),
            StructuredPruningRequest(
                graph.id,
                graph.nodes[0].id,
                (0,),
                target_runtime=TargetRuntime.ONNX_REFERENCE,
            ),
        )
        proposals = tuple(
            session.prepare_transformation(request) for request in requests
        )
        missing = session.prepare_transformation(WeightQuantizationRequest("missing"))

    assert all(not proposal.ok for proposal in proposals)
    assert not missing.ok and "no tensor" in missing.findings[0].message


def test_shell_builds_each_supported_transformation_request_and_rejects_preview(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ui-requests.onnx"
    build_matmul_model(path)
    with ApplicationService() as service:
        page = StubPage()
        shell = Shell(cast(ft.Page, page), service)
        shell.build()
        session = service.open_model(path)
        node = session.document.main_graph.nodes[0]
        shell.show_session(session)
        shell.renderer.set_selection(frozenset({node.id}))

        shell.transformation_kind.value = "graph-quantization"
        shell.transformation_granularity.value = Granularity.PER_CHANNEL.value
        shell.transformation_axis.value = "-1"
        assert isinstance(shell._transformation_request(), GraphQuantizationRequest)

        shell.transformation_kind.value = "threshold-pruning"
        shell.transformation_parameter.value = "2.5"
        assert isinstance(shell._transformation_request(), LogicalPruningRequest)

        shell.transformation_kind.value = "nm-pruning"
        assert isinstance(shell._transformation_request(), LogicalPruningRequest)

        shell.transformation_kind.value = "mask-pruning"
        shell.transformation_parameter.value = ",".join(["keep"] * 12)
        assert isinstance(shell._transformation_request(), LogicalPruningRequest)

        shell.transformation_kind.value = "structured-pruning"
        shell.transformation_parameter.value = "0,2"
        assert isinstance(shell._transformation_request(), StructuredPruningRequest)

        shell.transformation_parameter.value = "bad"
        shell._on_preview_transformation(cast(Any, None))
        assert shell.pending_transformation is None
        finding = shell.transformation_findings.controls[0]
        assert isinstance(finding, ft.Text)
        assert "comma-separated integers" in (finding.value or "")

        shell.transformation_kind.value = "mask-pruning"
        shell.transformation_parameter.value = "maybe"
        shell._on_preview_transformation(cast(Any, None))
        assert shell.pending_transformation is None
        shell.transformation_parameter.value = "keep"
        shell._on_preview_transformation(cast(Any, None))
        assert shell.pending_transformation is not None
        assert not shell.pending_transformation.ok
        shell._on_reject_transformation(cast(Any, None))
        assert shell.pending_transformation is None
