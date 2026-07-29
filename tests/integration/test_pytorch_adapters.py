"""Native PyTorch artifact ingestion, differentially validated (P6.1-P6.4).

Every reader is checked against the framework's own view of the same
artifact: tensor bytes must equal what torch holds, and safetensors files
written natively must be readable by the reference ``safetensors`` package.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pytest
import torch

from nneditor.adapters.pytorch import (
    open_checkpoint,
    open_fx_graph_module,
    open_pt2,
    open_safetensors,
    write_safetensors,
)
from nneditor.adapters.pytorch.checkpoint import CheckpointError
from nneditor.adapters.pytorch.pickle_scan import PickleScanError, scan_state_dict
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import Storage
from nneditor.ir.serialize import document_from_bytes, document_to_bytes
from nneditor.storage.store import TensorStore
from tests.fixtures.pytorch_models import (
    build_checkpoint,
    build_fx_module,
    build_pt2,
    build_safetensors_file,
    tiny_state_dict,
)


def tensor_by_name(document, name: str):  # type: ignore[no-untyped-def]
    for tensor_id, tensor in document.tensors.items():
        if tensor_id.endswith(f"#{name}"):
            return tensor
    raise AssertionError(f"no tensor named {name!r} in {list(document.tensors)}")


class TestCheckpoint:
    def test_tensors_match_torch_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt"
        state = build_checkpoint(path)
        document = open_checkpoint(path)

        assert document.artifact_kind is ArtifactKind.PYTORCH_STATE_DICT
        assert not document.has_errors
        assert len(document.tensors) == len(state)
        with TensorStore(document) as store:
            for name, expected in state.items():
                tensor = tensor_by_name(document, name)
                assert tensor.storage is Storage.EMBEDDED_RAW
                assert tensor.dims == tuple(expected.shape)
                assert tensor.element_type == "float32"
                assert store.read(tensor.id) == expected.numpy().tobytes()

    def test_topology_is_declared_unavailable(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt"
        build_checkpoint(path)
        document = open_checkpoint(path)
        assert len(document.main_graph.nodes) == 0, "no invented nodes"
        status = document.capability(Capability.TOPOLOGY)
        assert status.availability is Availability.UNAVAILABLE

    def test_documents_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt"
        build_checkpoint(path)
        document = open_checkpoint(path)
        payload = document_to_bytes(document)
        assert document_to_bytes(document_from_bytes(payload)) == payload

    def test_hostile_pickles_are_refused_without_execution(
        self, tmp_path: Path
    ) -> None:
        class Exploit:
            def __reduce__(self):  # type: ignore[no-untyped-def]
                return (print, ("pwned",))

        hostile = tmp_path / "hostile.pkl"
        hostile.write_bytes(pickle.dumps({"weight": Exploit()}))
        with pytest.raises(PickleScanError, match="refusing global"):
            scan_state_dict(hostile.read_bytes())

        # And through the checkpoint reader (zip container with the payload).
        import zipfile

        archive = tmp_path / "hostile.pt"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("hostile/data.pkl", pickle.dumps({"w": Exploit()}))
        with pytest.raises(CheckpointError, match=r"safe artifact mode"):
            open_checkpoint(archive)

    def test_non_checkpoint_zip_is_rejected(self, tmp_path: Path) -> None:
        import zipfile

        archive = tmp_path / "not-a-checkpoint.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("readme.txt", "hello")
        with pytest.raises(CheckpointError, match=re.escape("data.pkl")):
            open_checkpoint(archive)


class TestPt2:
    def test_graph_and_weights_match_torch(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt2"
        state = build_pt2(path)
        document = open_pt2(path)

        assert document.artifact_kind is ArtifactKind.PYTORCH_EXPORTED_PROGRAM
        assert not document.has_errors
        graph = document.main_graph
        op_types = [node.op_type for node in graph.nodes]
        assert "linear.default" in op_types
        assert "relu.default" in op_types
        assert all(node.domain == "pytorch.aten" for node in graph.nodes)
        assert len(graph.inputs) == 1, "parameters are initializers, not inputs"
        assert len(graph.outputs) == 1

        with TensorStore(document) as store:
            for name, expected in state.items():
                tensor = tensor_by_name(document, name)
                assert tensor.dims == tuple(expected.shape)
                assert store.read(tensor.id) == expected.detach().numpy().tobytes()

    def test_parameter_values_carry_their_module_names(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt2"
        build_pt2(path)
        document = open_pt2(path)
        names = {value.name for value in document.main_graph.values}
        assert "linear.weight" in names and "linear.bias" in names

    def test_dynamic_batch_survives_as_symbolic_dimension(self, tmp_path: Path) -> None:
        path = tmp_path / "dynamic.pt2"
        build_pt2(path, dynamic=True)
        document = open_pt2(path)
        graph = document.main_graph
        assert graph.symbolic_dimensions, "the batch dimension is symbolic"
        (input_id,) = graph.inputs
        shape = graph.value(input_id).shape
        assert shape is not None and isinstance(shape[0], str)

    def test_dataflow_links_are_derived(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt2"
        build_pt2(path)
        document = open_pt2(path)
        graph = document.main_graph
        linear = next(n for n in graph.nodes if n.op_type == "linear.default")
        relu = next(n for n in graph.nodes if n.op_type == "relu.default")
        assert graph.consumers(linear.outputs[0]) == ((relu.id, 0),)
        assert graph.producer(relu.outputs[0]) == (relu.id, 0)

    def test_documents_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.pt2"
        build_pt2(path)
        document = open_pt2(path)
        payload = document_to_bytes(document)
        assert document_to_bytes(document_from_bytes(payload)) == payload


class TestSafetensors:
    def test_reference_files_read_natively(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.safetensors"
        state = build_safetensors_file(path)
        document = open_safetensors(path)

        assert document.artifact_kind is ArtifactKind.SAFETENSORS
        assert not document.has_errors
        assert dict(document.extensions)["x-safetensors.metadata"] == {
            "producer": "nneditor-tests"
        }
        with TensorStore(document) as store:
            for name, expected in state.items():
                tensor = tensor_by_name(document, name)
                assert tensor.dims == tuple(expected.shape)
                assert store.read(tensor.id) == expected.numpy().tobytes()

    def test_native_writer_is_read_by_the_reference_package(
        self, tmp_path: Path
    ) -> None:
        from safetensors import safe_open

        state = tiny_state_dict()
        target = tmp_path / "written.safetensors"
        write_safetensors(
            target,
            [
                (name, "float32", tuple(value.shape), value.numpy().tobytes())
                for name, value in state.items()
            ],
            metadata={"written_by": "nneditor"},
        )
        with safe_open(str(target), framework="pt") as handle:
            assert handle.metadata() == {"written_by": "nneditor"}
            for name, expected in state.items():
                assert torch.equal(handle.get_tensor(name), expected)

    def test_malformed_entries_are_diagnosed_not_fatal(self, tmp_path: Path) -> None:
        import json
        import struct

        header = json.dumps(
            {
                "good": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8],
                },
                "liar": {
                    "dtype": "F32",
                    "shape": [1000],
                    "data_offsets": [8, 16],
                },
            }
        ).encode()
        path = tmp_path / "mixed.safetensors"
        path.write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, 2, 3, 4)
        )
        document = open_safetensors(path)
        assert tensor_by_name(document, "good").storage is Storage.EMBEDDED_RAW
        assert tensor_by_name(document, "liar").storage is Storage.ABSENT
        assert any(
            item.code == "pytorch.tensor-length-mismatch"
            for item in document.diagnostics
        )


class TestFx:
    def test_generated_source_topology_is_recovered(self, tmp_path: Path) -> None:
        path = tmp_path / "traced.pt"
        build_fx_module(path)
        document = open_fx_graph_module(path)

        assert document.artifact_kind is ArtifactKind.PYTORCH_FX_GRAPH_MODULE
        graph = document.main_graph
        labels = [node.op_type for node in graph.nodes]
        assert any("linear" in label for label in labels)
        assert len(graph.inputs) == 1 and len(graph.outputs) == 1
        imports = dict(document.extensions)["x-torch.fx"]
        assert isinstance(imports, dict)
        harvested = imports["pickle_imports"]
        assert isinstance(harvested, list)
        assert any("torch" in str(item) for item in harvested)

    def test_editing_and_export_stay_unavailable(self, tmp_path: Path) -> None:
        path = tmp_path / "traced.pt"
        build_fx_module(path)
        document = open_fx_graph_module(path)
        assert (
            document.capability(Capability.EDITING).availability
            is Availability.UNAVAILABLE
        )
        assert (
            document.capability(Capability.EXPORT).availability
            is Availability.UNAVAILABLE
        )

    def test_an_artifact_without_source_reports_and_survives(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "opaque.pkl"
        path.write_bytes(pickle.dumps({"just": "data"}))
        document = open_fx_graph_module(path)
        assert document.has_errors
        assert any(item.code == "pytorch.fx-no-source" for item in document.diagnostics)
