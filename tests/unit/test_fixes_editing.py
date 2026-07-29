"""Regression tests for verified editing/transformation defects.

Each test class pins one reviewed finding:

* asymmetric quantization must extend its range to include zero,
* N:M pruning must group along the innermost axis,
* input reconnection must respect the stored node order,
* byte-edit bounds must follow the current revision view,
* attribute edits must reject exact no-ops and normalize names,
* batched commits must verify preconditions against the evolving state,
* diff previews must count overlapping spans once and name their base.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.application.session import ApplicationService
from nneditor.editing.commands import (
    QuantizeGraph,
    ReplaceTensorBytes,
    ResizeTensor,
)
from nneditor.editing.cow import ByteSpanEdit, EditError
from nneditor.editing.diff import preview_diff
from nneditor.editing.revisions import RevisionChain, ValidationState
from nneditor.editing.validation import ReconnectInputRequest, SetAttributeRequest
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
)
from nneditor.ir.core import (
    ArtifactRef,
    AttrKind,
    Document,
    Graph,
    Storage,
    TensorRef,
)
from nneditor.storage.store import TensorStore
from nneditor.transformations.engine import (
    GraphQuantizationRequest,
    LogicalPruningRequest,
    TransformationEngine,
    TransformationProposal,
    _quantize,
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
from tests.fixtures.onnx_models import build_matmul_model, build_tensor_only_model


def _handcrafted_document(tensors: list[TensorRef]) -> Document:
    return Document(
        source=ArtifactRef(path="memory", content_hash="sha256:00", byte_size=0),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=[
            CapabilityStatus(capability, Availability.AVAILABLE, "test")
            for capability in Capability
        ],
        graphs=[Graph(id="g:main", name="main")],
        tensors=tensors,
    )


class TestAsymmetricQuantizationRange:
    """The [min, max] range must be extended to include zero (finding 1)."""

    @pytest.mark.parametrize(
        ("values", "signed"),
        [
            (np.linspace(2.0, 10.0, 32, dtype=np.float32), True),
            (np.linspace(2.0, 10.0, 32, dtype=np.float32), False),
            (np.linspace(-10.0, -2.0, 32, dtype=np.float32), True),
            (np.full(16, 1000.0, dtype=np.float32), True),
            (np.full(16, 1000.0, dtype=np.float32), False),
            (np.full(16, -750.0, dtype=np.float32), True),
        ],
        ids=[
            "positive-only-signed",
            "positive-only-unsigned",
            "negative-only-signed",
            "constant-signed",
            "constant-unsigned",
            "constant-negative-signed",
        ],
    )
    def test_reconstruction_stays_within_one_step(
        self, values: np.ndarray, signed: bool
    ) -> None:
        result = _quantize(
            values,
            bit_width=8,
            signed=signed,
            symmetric=False,
            granularity=Granularity.PER_TENSOR,
            axis=None,
        )
        (scale,) = result.scale
        (zero,) = result.zero_point
        qmin, qmax = (-128, 127) if signed else (0, 255)
        assert qmin <= zero <= qmax
        error = np.abs(result.dequantized.astype(np.float64) - values)
        assert float(error.max()) <= scale + 1e-6

    def test_per_channel_with_a_positive_only_channel(self) -> None:
        values = np.stack(
            [
                np.linspace(2.0, 10.0, 8, dtype=np.float32),
                np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            ],
            axis=0,
        )
        result = _quantize(
            values,
            bit_width=8,
            signed=True,
            symmetric=False,
            granularity=Granularity.PER_CHANNEL,
            axis=0,
        )
        zeros = np.asarray(result.zero_point)
        assert np.all((zeros >= -128) & (zeros <= 127))
        scales = np.asarray(result.scale, dtype=np.float64).reshape(2, 1)
        error = np.abs(result.dequantized.astype(np.float64) - values)
        assert np.all(error <= scales + 1e-6)


def _pruning_proposal(
    tmp_path: Path, dims: tuple[int, ...], values: np.ndarray
) -> TransformationProposal:
    path = tmp_path / "prune.onnx"
    build_tensor_only_model(
        path,
        dims=list(dims),
        raw_bytes=values.astype("<f4").tobytes(),
        data_type=TensorProto.FLOAT,
    )
    document = index_to_document(index_model(path))
    tensor_id = document.main_graph.initializers[0]
    with TensorStore(document) as store:
        engine = TransformationEngine(lambda tid: store.read(tid))
        return engine.prepare(
            document,
            LogicalPruningRequest(tensor_id, PruningMode.NM, n=2, m=4),
            base_revision_id=None,
        )


class TestNmPruningInnermostAxis:
    """N:M blocks must never straddle innermost-axis rows (finding 2)."""

    @pytest.mark.parametrize(
        "dims",
        [(2, 6), (2, 2)],
        ids=["size-divisible-rows-not", "single-straddling-block"],
    )
    def test_indivisible_innermost_axis_is_rejected(
        self, tmp_path: Path, dims: tuple[int, ...]
    ) -> None:
        count = int(np.prod(dims))
        values = np.arange(1, count + 1, dtype=np.float32).reshape(dims)
        proposal = _pruning_proposal(tmp_path, dims, values)
        assert not proposal.ok
        messages = " ".join(finding.message for finding in proposal.findings)
        assert "innermost" in messages

    def test_blocks_are_formed_per_row(self, tmp_path: Path) -> None:
        values = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        proposal = _pruning_proposal(tmp_path, (2, 8), values)
        assert proposal.ok
        (command,) = proposal.commands
        assert isinstance(command, ReplaceTensorBytes)
        after = np.frombuffer(command.edit.after, dtype="<f4").reshape(2, 8)
        expected = np.array(
            [
                [0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0],
                [8.0, 7.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(after, expected)
        assert proposal.manifest is not None
        assert ("axis", "innermost") in proposal.manifest.parameters


def _parallel_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["a_out"], name="alpha"),
            helper.make_node("Relu", ["input"], ["b_out"], name="beta"),
        ],
        "parallel",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [
            helper.make_tensor_value_info("a_out", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [4]),
        ],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )


class TestReconnectStoredOrder:
    """Reconnection must not create use-before-definition (finding 3)."""

    def test_later_producer_is_rejected_and_earlier_allowed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "parallel.onnx"
        _parallel_model(path)
        with ApplicationService() as service:
            session = service.open_model(path)
            graph = session.document.main_graph
            first, second = graph.nodes

            backwards = session.prepare_edit(
                ReconnectInputRequest(graph.id, first.id, 0, second.outputs[0])
            )
            assert not backwards.ok
            messages = " ".join(item.message for item in backwards.findings)
            assert "stored node order" in messages

            forwards = session.prepare_edit(
                ReconnectInputRequest(graph.id, second.id, 0, first.outputs[0])
            )
            assert forwards.ok


def _resize_manifest() -> TransformationManifest:
    return TransformationManifest(
        kind=TransformationKind.STRUCTURED_PRUNING,
        target_runtime=TargetRuntime.PORTABLE_ONNX,
        operator_representation=OperatorRepresentation.SHAPE_REWRITE,
        storage_effect=StorageEffect.REDUCED,
        pruning_mode=PruningMode.OUTPUT_CHANNELS,
    )


def _resize_preview(
    tensor_id: str, source_bytes: int, result_bytes: int
) -> TransformationPreview:
    return TransformationPreview(
        tensor_id=tensor_id,
        element_count=source_bytes // 4,
        changed_elements=0,
        source_bytes=source_bytes,
        result_bytes=result_bytes,
        theoretical_packed_bytes=result_bytes,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        rmse=0.0,
        error_basis="Test fixture; not elementwise-comparable.",
        before_sparsity=0.0,
        after_sparsity=0.0,
        mathematical_conversion=True,
        executable_graph=True,
        storage_reduction=False,
        expected_acceleration=False,
        acceleration_reason="Test fixture makes no acceleration claim.",
    )


class TestEditBoundsFollowCurrentView:
    """apply_replace bounds must come from the revision view (finding 4)."""

    def test_resized_tensor_accepts_edits_beyond_the_base_length(self) -> None:
        base = bytes(range(16))
        before_tensor = TensorRef("t:w", "float32", (4,), Storage.GENERATED)
        after_tensor = TensorRef("t:w", "float32", (8,), Storage.GENERATED)
        document = _handcrafted_document([before_tensor])
        command = ResizeTensor(
            "g:main",
            before_tensor,
            after_tensor,
            base,
            bytes(range(32)),
            (),
            _resize_manifest(),
            _resize_preview("t:w", 16, 32),
        )
        chain = RevisionChain(
            {"t:w": base}.__getitem__,
            tensor_length=lambda _tid: 16,  # deliberately stale base length
            base_document=document,
        )
        chain.apply_commands((command,), ValidationState(ok=True))
        assert chain.byte_length("t:w") == 32

        chain.apply_replace("t:w", 20, b"\xaa\xbb\xcc\xdd")
        assert chain.read("t:w", offset=20, length=4) == b"\xaa\xbb\xcc\xdd"
        with pytest.raises(EditError, match="outside the 32-byte tensor"):
            chain.apply_replace("t:w", 30, b"\x00\x01\x02")

    def test_quantization_generated_parameters_are_editable(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "qdq.onnx"
        build_matmul_model(path, opset=19)
        with ApplicationService() as service:
            session = service.open_model(path)
            graph = session.document.main_graph
            proposal = session.prepare_transformation(
                GraphQuantizationRequest(
                    session.document.entry_graph,
                    graph.initializers[0],
                )
            )
            assert proposal.ok
            session.commit_transformation(proposal)
            (command,) = proposal.commands
            assert isinstance(command, QuantizeGraph)
            scale_id = command.scale_tensor.id
            assert session.editing.read(scale_id) == command.scale_bytes

            replacement = struct.pack("<f", 123.5)
            session.editing.replace_bytes(scale_id, 0, replacement)
            assert session.editing.read(scale_id) == replacement


def _softmax_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], name="prob", axis=1)],
        "attribute-fixes",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )


class TestSetAttributeNormalization:
    """No-op attribute edits and unnormalized names (finding 5)."""

    def test_exact_noop_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "softmax.onnx"
        _softmax_model(path)
        with ApplicationService() as service:
            session = service.open_model(path)
            graph = session.document.main_graph
            node = graph.nodes[0]
            noop = session.prepare_edit(
                SetAttributeRequest(graph.id, node.id, "axis", AttrKind.INT, 1)
            )
            assert not noop.ok
            messages = " ".join(item.message for item in noop.findings)
            assert "already has that value" in messages

    def test_names_are_normalized_at_the_boundary(self, tmp_path: Path) -> None:
        path = tmp_path / "softmax.onnx"
        _softmax_model(path)
        with ApplicationService() as service:
            session = service.open_model(path)
            graph = session.document.main_graph
            node = graph.nodes[0]

            blank = session.prepare_edit(
                SetAttributeRequest(graph.id, node.id, "   ", AttrKind.INT, 0)
            )
            assert not blank.ok
            messages = " ".join(item.message for item in blank.findings)
            assert "name cannot be empty" in messages

            padded = session.prepare_edit(
                SetAttributeRequest(graph.id, node.id, " axis ", AttrKind.INT, 0)
            )
            assert padded.ok
            session.commit_edit(padded)
            updated = session.document.main_graph.node(node.id)
            assert updated.attribute("axis").value == 0
            assert len(updated.attributes) == 1


def _replace(
    tensor_id: str, offset: int, before: bytes, after: bytes
) -> ReplaceTensorBytes:
    return ReplaceTensorBytes(ByteSpanEdit(tensor_id, offset, before, after))


class TestBatchedPreconditionVerification:
    """apply_commands must verify against the evolving state (finding 6)."""

    def test_stacked_edits_on_one_span_commit(self) -> None:
        base = bytes(range(16))
        document = _handcrafted_document(
            [TensorRef("t:w", "float32", (4,), Storage.GENERATED)]
        )
        chain = RevisionChain({"t:w": base}.__getitem__, base_document=document)
        first = _replace("t:w", 0, bytes([0, 1]), b"\xaa\xbb")
        # The second edit's expected bytes are the *evolved* batch state.
        second = _replace("t:w", 1, b"\xbb\x02", b"\xcc\xdd")
        chain.apply_commands((first, second), ValidationState(ok=True))
        assert chain.read("t:w", offset=0, length=3) == b"\xaa\xcc\xdd"

    def test_documentless_chains_still_verify_before_bytes(self) -> None:
        base = bytes(range(16))
        chain = RevisionChain({"t:w": base}.__getitem__)
        stale = _replace("t:w", 0, b"\x09\x09", b"\xaa\xbb")
        with pytest.raises(EditError, match="precondition"):
            chain.apply_commands((stale,), ValidationState(ok=True))
        assert chain.revisions == ()

        good = _replace("t:w", 0, bytes([0, 1]), b"\xaa\xbb")
        chain.apply_commands((good,), ValidationState(ok=True))
        assert chain.read("t:w")[:2] == b"\xaa\xbb"


class TestDiffPreviewAccounting:
    """Overlap-safe byte counts and a populated base id (finding 7)."""

    def test_overlapping_spans_count_once_and_base_is_named(self) -> None:
        chain = RevisionChain({"t:w": bytes(range(16))}.__getitem__)
        first = chain.apply_replace("t:w", 0, b"\xaa\xbb")
        second = chain.apply_replace("t:w", 1, b"\xcc\xdd")
        document = _handcrafted_document(
            [TensorRef("t:w", "float32", (4,), Storage.GENERATED)]
        )

        preview = preview_diff(document, chain.applied)
        (tensor_diff,) = preview.tensors
        assert tensor_diff.span_count == 2
        assert tensor_diff.bytes_changed == 3, "union of [0, 2) and [1, 3)"
        assert tensor_diff.elements_changed == 1
        assert preview.base_revision_id is None, "the full chain starts at base"
        assert preview.target_revision_id == second.id

        suffix = preview_diff(document, chain.applied[1:])
        assert suffix.base_revision_id == first.id
        assert suffix.target_revision_id == second.id
