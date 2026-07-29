"""StableHLO ingestion and weights-only JAX sessions (P7.1-P7.5).

Fixtures come from the real ``jax.export`` lowering, so the native textual
reader is validated against exporter output rather than hand-written MLIR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.adapters.jax import open_orbax_checkpoint, open_stablehlo
from nneditor.adapters.jax.checkpoint import OrbaxError
from nneditor.adapters.jax.mlir_text import parse_module, parse_type
from nneditor.adapters.jax.stablehlo import StableHloError
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.serialize import document_from_bytes, document_to_bytes
from tests.fixtures.jax_models import (
    build_control_flow_module,
    build_custom_call_module,
    build_mlp_module,
    build_polymorphic_module,
)


class TestTypes:
    def test_static_ranked_tensors(self) -> None:
        parsed = parse_type("tensor<2x3xf32>")
        assert parsed.element_type == "float32"
        assert parsed.shape == (2, 3)

    def test_dynamic_and_scalar_tensors(self) -> None:
        assert parse_type("tensor<?x4xf32>").shape == (None, 4)
        assert parse_type("tensor<f32>").shape == ()
        assert parse_type("tensor<i1>").element_type == "bool"

    def test_non_tensor_types_keep_their_text(self) -> None:
        parsed = parse_type("!stablehlo.token")
        assert parsed.shape is None and parsed.text == "!stablehlo.token"


class TestStableHlo:
    def test_a_static_module_maps_completely(self, tmp_path: Path) -> None:
        path = build_mlp_module(tmp_path / "mlp.mlir")
        document = open_stablehlo(path)

        assert document.artifact_kind is ArtifactKind.JAX_STABLEHLO
        assert not document.has_errors
        graph = document.main_graph
        assert [node.op_type for node in graph.nodes] == ["dot_general", "tanh"]
        assert all(node.domain == "stablehlo" for node in graph.nodes)
        assert len(graph.inputs) == 2 and len(graph.outputs) == 1
        assert graph.value(graph.inputs[0]).shape == (2, 3)
        assert graph.value(graph.inputs[0]).element_type == "float32"

    def test_dataflow_links_derive_from_ssa_names(self, tmp_path: Path) -> None:
        path = build_mlp_module(tmp_path / "mlp.mlir")
        graph = open_stablehlo(path).main_graph
        dot = next(n for n in graph.nodes if n.op_type == "dot_general")
        tanh = next(n for n in graph.nodes if n.op_type == "tanh")
        assert graph.consumers(dot.outputs[0]) == ((tanh.id, 0),)
        assert graph.outputs == (tanh.outputs[0],)

    def test_polymorphic_shapes_import_as_dynamic(self, tmp_path: Path) -> None:
        path = build_polymorphic_module(tmp_path / "poly.mlir")
        document = open_stablehlo(path)
        graph = document.main_graph
        shapes = [graph.value(item).shape for item in graph.inputs]
        assert any(shape is not None and shape[0] is None for shape in shapes), (
            "the polymorphic batch extent is dynamic"
        )
        assert len(document.graphs) >= 2, "the wrapped callee is its own graph"

    def test_control_flow_regions_are_preserved_verbatim(self, tmp_path: Path) -> None:
        path = build_control_flow_module(tmp_path / "cf.mlir")
        document = open_stablehlo(path)
        assert any(
            item.code == "jax.region-not-modelled" for item in document.diagnostics
        )
        case = next(
            node for node in document.main_graph.nodes if node.op_type == "case"
        )
        regions = [a for a in case.attributes if a.name.startswith("region_")]
        assert regions and "stablehlo" in str(regions[0].value)

    def test_custom_calls_are_flagged_and_uneditable(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.mlir"
        path.write_text(
            """
module @custom {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.custom_call @ext(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
""",
            encoding="utf-8",
        )
        document = open_stablehlo(path)
        assert any(item.code == "jax.custom-call" for item in document.diagnostics)
        node = document.main_graph.nodes[0]
        notes = document.notes_for(node.id)
        assert notes and notes[0].availability is Availability.UNAVAILABLE

    def test_composites_survive(self, tmp_path: Path) -> None:
        path = build_custom_call_module(tmp_path / "cc.mlir")
        graph = open_stablehlo(path).main_graph
        assert [node.op_type for node in graph.nodes] == ["composite"]

    def test_documents_round_trip(self, tmp_path: Path) -> None:
        path = build_mlp_module(tmp_path / "mlp.mlir")
        document = open_stablehlo(path)
        payload = document_to_bytes(document)
        assert document_to_bytes(document_from_bytes(payload)) == payload

    def test_bytecode_and_garbage_are_refused_clearly(self, tmp_path: Path) -> None:
        bytecode = tmp_path / "module.mlirbc"
        bytecode.write_bytes(b"ML\xefR\x00\x00binary")
        with pytest.raises(StableHloError, match="bytecode"):
            open_stablehlo(bytecode)

        garbage = tmp_path / "notes.txt"
        garbage.write_text("this is not MLIR at all", encoding="utf-8")
        with pytest.raises(StableHloError, match="declares no module"):
            open_stablehlo(garbage)

    def test_a_module_without_functions_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.mlir"
        path.write_text("module @empty {\n}\n", encoding="utf-8")
        with pytest.raises(StableHloError, match="no functions"):
            open_stablehlo(path)

    def test_unparseable_operations_import_opaquely(self) -> None:
        module = parse_module(
            """
module @odd {
  func.func public @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    !!! not an operation !!!
    return %arg0 : tensor<2xf32>
  }
}
"""
        )
        assert module.functions[0].operations[0].opaque


class TestSessions:
    def test_stablehlo_opens_through_the_service(self, tmp_path: Path) -> None:
        path = build_mlp_module(tmp_path / "mlp.mlir")
        with ApplicationService() as service:
            session = service.open_model(path)
            assert session.document.artifact_kind is ArtifactKind.JAX_STABLEHLO
            layout = session.scene()
            assert layout.scene.node_count == 2, "the graph renders"
            assert session.search("tanh")

    def test_checkpoint_directory_opens_weights_only(self, tmp_path: Path) -> None:
        import json

        from nneditor.adapters.pytorch.safetensors import write_safetensors

        checkpoint = tmp_path / "orbax"
        checkpoint.mkdir()
        write_safetensors(
            checkpoint / "params.safetensors",
            [("dense.kernel", "float32", (2, 2), b"\x00" * 16)],
        )
        (checkpoint / "tree.json").write_text(
            json.dumps({"params": {"dense": {"kernel": "leaf"}}}), encoding="utf-8"
        )
        document = open_orbax_checkpoint(checkpoint)
        assert document.artifact_kind is ArtifactKind.ORBAX_CHECKPOINT
        assert (
            document.capability(Capability.TOPOLOGY).availability
            is Availability.UNAVAILABLE
        )
        assert len(document.main_graph.nodes) == 0
        assert dict(document.extensions)["x-orbax.tree"]

    def test_a_checkpoint_without_payload_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-checkpoint"
        empty.mkdir()
        with pytest.raises(OrbaxError, match="no safetensors payload"):
            open_orbax_checkpoint(empty)
