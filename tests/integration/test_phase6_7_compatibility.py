"""Cross-framework compatibility and failure modes (P6.6 / P7.6).

Covers the matrix each phase's exit gate names: artifact detection by
content, capability labelling per contract, weights-only export paths, and
the malformed/hostile inputs every reader must survive without executing
anything or inventing data.
"""

from __future__ import annotations

import json
import pickle
import struct
import zipfile
from pathlib import Path

import pytest
import torch

from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.pytorch import (
    SafetensorSource,
    open_safetensors,
    write_safetensors,
    write_safetensors_stream,
)
from nneditor.adapters.pytorch import safetensors as safetensors_module
from nneditor.adapters.pytorch.checkpoint import CheckpointError, open_checkpoint
from nneditor.adapters.pytorch.pickle_scan import PickleScanError, scan_state_dict
from nneditor.adapters.pytorch.pt2 import Pt2Error, open_pt2
from nneditor.adapters.pytorch.safetensors import SafetensorsError
from nneditor.adapters.pytorch.zip_store import ZipStoreError, zip_members
from nneditor.application.editing import EditingController
from nneditor.application.session import ApplicationService, SessionError
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    ExportFidelity,
)
from nneditor.storage.store import TensorStore
from tests.fixtures.jax_models import build_mlp_module
from tests.fixtures.onnx_models import build_embedded_model
from tests.fixtures.pytorch_models import (
    build_checkpoint,
    build_pt2,
    build_safetensors_file,
)


class TestDetection:
    def test_every_supported_family_is_detected_by_content(
        self, tmp_path: Path
    ) -> None:
        onnx_path = tmp_path / "model.onnx"
        build_embedded_model(onnx_path, elements=8)
        pt2_path = tmp_path / "program.pt2"
        build_pt2(pt2_path)
        checkpoint = tmp_path / "weights.pt"
        build_checkpoint(checkpoint)
        safetensors = tmp_path / "weights.safetensors"
        build_safetensors_file(safetensors)
        stablehlo = build_mlp_module(tmp_path / "module.mlir")

        assert detect_artifact_kind(onnx_path) is ArtifactKind.ONNX_MODEL
        assert detect_artifact_kind(pt2_path) is ArtifactKind.PYTORCH_EXPORTED_PROGRAM
        assert detect_artifact_kind(checkpoint) is ArtifactKind.PYTORCH_STATE_DICT
        assert detect_artifact_kind(safetensors) is ArtifactKind.SAFETENSORS
        assert detect_artifact_kind(stablehlo) is ArtifactKind.JAX_STABLEHLO

    def test_extensions_do_not_override_content(self, tmp_path: Path) -> None:
        """A checkpoint named .onnx is still a checkpoint (rule 2)."""
        misnamed = tmp_path / "actually-a-checkpoint.onnx"
        build_checkpoint(misnamed)
        assert detect_artifact_kind(misnamed) is ArtifactKind.PYTORCH_STATE_DICT

    def test_unknown_and_empty_files_are_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        with pytest.raises(DetectionError, match="empty"):
            detect_artifact_kind(empty)

        junk = tmp_path / "notes.rst"
        junk.write_bytes(b"\xfe\xed just prose, no model here")
        with pytest.raises(DetectionError, match="no supported artifact"):
            detect_artifact_kind(junk)

    def test_a_zip_without_a_model_is_refused(self, tmp_path: Path) -> None:
        archive = tmp_path / "photos.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("cat.txt", "meow")
        with pytest.raises(DetectionError, match="no recognized model"):
            detect_artifact_kind(archive)


class TestSessionsAcrossFrameworks:
    @pytest.mark.parametrize(
        ("builder", "kind", "topology"),
        [
            ("pt2", ArtifactKind.PYTORCH_EXPORTED_PROGRAM, Availability.AVAILABLE),
            (
                "checkpoint",
                ArtifactKind.PYTORCH_STATE_DICT,
                Availability.UNAVAILABLE,
            ),
            ("safetensors", ArtifactKind.SAFETENSORS, Availability.UNAVAILABLE),
            ("stablehlo", ArtifactKind.JAX_STABLEHLO, Availability.AVAILABLE),
        ],
    )
    def test_open_labels_capabilities_per_contract(
        self,
        tmp_path: Path,
        builder: str,
        kind: ArtifactKind,
        topology: Availability,
    ) -> None:
        path: Path
        if builder == "pt2":
            path = tmp_path / "m.pt2"
            build_pt2(path)
        elif builder == "checkpoint":
            path = tmp_path / "m.pt"
            build_checkpoint(path)
        elif builder == "safetensors":
            path = tmp_path / "m.safetensors"
            build_safetensors_file(path)
        else:
            path = build_mlp_module(tmp_path / "m.mlir")
        with ApplicationService() as service:
            session = service.open_model(path)
            assert session.document.artifact_kind is kind
            assert session.capability(Capability.TOPOLOGY).availability is topology
            # Every opened artifact renders and inspects without extra work.
            session.scene()

    def test_an_fx_module_falls_back_from_the_checkpoint_reader(
        self, tmp_path: Path
    ) -> None:
        from tests.fixtures.pytorch_models import build_fx_module

        path = tmp_path / "traced.pt"
        build_fx_module(path)
        with ApplicationService() as service:
            session = service.open_model(path)
            assert (
                session.document.artifact_kind is ArtifactKind.PYTORCH_FX_GRAPH_MODULE
            )
            assert (
                session.capability(Capability.EDITING).availability
                is Availability.UNAVAILABLE
            )

    def test_unreadable_artifacts_surface_as_session_errors(
        self, tmp_path: Path
    ) -> None:
        junk = tmp_path / "junk.dat"
        junk.write_bytes(b"\x01\x02\x03 not a model")
        with ApplicationService() as service:
            with pytest.raises(SessionError):
                service.open_model(junk)


class TestWeightsOnlyExport:
    def test_format_routed_export_returns_a_neutral_outcome(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "weights.safetensors"
        build_safetensors_file(source)
        destination = tmp_path / "routed.safetensors"
        with ApplicationService() as service:
            session = service.open_model(source)
            plan = session.export_plan()
            assert plan.available
            assert plan.allowed_extensions == ("safetensors",)
            outcome = session.export_artifact(destination)

        assert outcome.written_files == (destination,)
        assert outcome.fidelity is ExportFidelity.WEIGHTS_ONLY
        assert outcome.report_path is None
        assert open_safetensors(destination).tensors

    def test_stablehlo_export_is_honestly_unavailable(self, tmp_path: Path) -> None:
        source = build_mlp_module(tmp_path / "module.mlir")
        with ApplicationService() as service:
            session = service.open_model(source)
            plan = session.export_plan()
            assert not plan.available
            assert plan.fidelity is ExportFidelity.UNAVAILABLE
            assert (
                session.capability(Capability.EDITING).availability
                is Availability.UNAVAILABLE
            )
            with pytest.raises(SessionError, match="no StableHLO writer"):
                session.export_artifact(tmp_path / "out.mlir")

    def test_edited_checkpoint_exports_as_safetensors(self, tmp_path: Path) -> None:
        source = tmp_path / "weights.pt"
        state = build_checkpoint(source)
        destination = tmp_path / "edited.safetensors"

        with ApplicationService() as service:
            session = service.open_model(source)
            tensor_id = next(
                item
                for item in session.document.tensors
                if item.endswith("#linear.bias")
            )
            replacement = struct.pack("<f", -3.5)
            session.editing.replace_bytes(tensor_id, 0, replacement)
            session.export_weights_only(destination)

        # The export reopens and carries the edit; the source is untouched.
        exported = open_safetensors(destination)
        with TensorStore(exported) as store:
            bias_id = next(
                item for item in exported.tensors if item.endswith("#linear.bias")
            )
            assert store.read(bias_id)[:4] == struct.pack("<f", -3.5)
        metadata = dict(exported.extensions)["x-safetensors.metadata"]
        assert isinstance(metadata, dict)
        assert metadata["fidelity"] == "weights only"
        reloaded = torch.load(source, weights_only=True)
        assert torch.equal(reloaded["linear.bias"], state["linear.bias"])

    def test_session_export_reads_tensor_payloads_in_bounded_ranges(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "weights.safetensors"
        build_safetensors_file(source)
        destination = tmp_path / "streamed.safetensors"
        calls: list[tuple[int, int | None]] = []
        original = EditingController.read

        def recording_read(
            controller: EditingController,
            tensor_id: str,
            *,
            offset: int = 0,
            length: int | None = None,
        ) -> bytes:
            calls.append((offset, length))
            return original(controller, tensor_id, offset=offset, length=length)

        monkeypatch.setattr(safetensors_module, "_WRITE_CHUNK_BYTES", 4)
        monkeypatch.setattr(EditingController, "read", recording_read)
        with ApplicationService() as service:
            session = service.open_model(source)
            session.export_weights_only(destination)

        assert calls
        assert all(length is not None and length <= 4 for _offset, length in calls)
        assert open_safetensors(destination).tensors

    def test_streaming_writer_removes_partial_output_when_cancelled(
        self, tmp_path: Path
    ) -> None:
        destination = tmp_path / "cancelled.safetensors"
        payload = bytes(range(24))
        token = CancellationToken()

        def cancel_after_first_read(offset: int, length: int) -> bytes:
            token.cancel()
            return payload[offset : offset + length]

        source = SafetensorSource(
            name="weight",
            element_type="float32",
            dims=(6,),
            byte_length=len(payload),
            read=cancel_after_first_read,
        )
        with pytest.raises(OperationCancelled):
            write_safetensors_stream(
                destination,
                [source],
                token=token,
                chunk_bytes=4,
            )

        assert not destination.exists()
        assert not destination.with_name(destination.name + ".partial").exists()

    def test_export_never_overwrites(self, tmp_path: Path) -> None:
        source = tmp_path / "weights.safetensors"
        build_safetensors_file(source)
        with ApplicationService() as service:
            session = service.open_model(source)
            with pytest.raises(SessionError, match="already exists"):
                session.export_weights_only(source)

    def test_an_artifact_without_tensors_refuses_to_export(
        self, tmp_path: Path
    ) -> None:
        stablehlo = build_mlp_module(tmp_path / "module.mlir")
        with ApplicationService() as service:
            session = service.open_model(stablehlo)
            with pytest.raises(SessionError, match="no readable tensors"):
                session.export_weights_only(tmp_path / "out.safetensors")


class TestMalformedInputs:
    def test_truncated_and_bad_archives(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.pt2"
        broken.write_bytes(b"PK\x03\x04 truncated garbage")
        with pytest.raises(ZipStoreError, match="not a readable zip"):
            zip_members(broken)

        no_format = tmp_path / "no-format.pt2"
        with zipfile.ZipFile(no_format, "w") as bundle:
            bundle.writestr("thing/models/model.json", "{}")
        with pytest.raises(Pt2Error, match="archive_format"):
            open_pt2(no_format)

    def test_a_pt2_with_an_unsupported_schema_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "future.pt2"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("m/archive_format", "pt2")
            bundle.writestr("m/byteorder", "little")
            bundle.writestr(
                "m/models/model.json",
                json.dumps({"schema_version": {"major": 999, "minor": 0}}),
            )
        with pytest.raises(Pt2Error, match="schema major"):
            open_pt2(path)

    def test_big_endian_archives_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "big.pt2"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("m/archive_format", "pt2")
            bundle.writestr("m/byteorder", "big")
            bundle.writestr("m/models/model.json", "{}")
        with pytest.raises(Pt2Error, match="big-endian"):
            open_pt2(path)

    def test_safetensors_header_limits_and_truncation(self, tmp_path: Path) -> None:
        short = tmp_path / "short.safetensors"
        short.write_bytes(b"\x01\x02")
        with pytest.raises(SafetensorsError, match="too short"):
            open_safetensors(short)

        huge = tmp_path / "huge.safetensors"
        huge.write_bytes(struct.pack("<Q", 2**40))
        with pytest.raises(SafetensorsError, match="limit"):
            open_safetensors(huge)

        truncated = tmp_path / "truncated.safetensors"
        truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
        with pytest.raises(SafetensorsError, match="truncated"):
            open_safetensors(truncated)

        not_json = tmp_path / "notjson.safetensors"
        payload = b"not json at all!"
        not_json.write_bytes(struct.pack("<Q", len(payload)) + payload)
        with pytest.raises(SafetensorsError, match="not valid JSON"):
            open_safetensors(not_json)

    def test_safetensors_writer_rejects_bad_input(self, tmp_path: Path) -> None:
        with pytest.raises(SafetensorsError, match="cannot represent"):
            write_safetensors(
                tmp_path / "a.safetensors",
                [("t", "complex128", (1,), b"\x00" * 16)],
            )
        with pytest.raises(SafetensorsError, match="imply"):
            write_safetensors(
                tmp_path / "b.safetensors", [("t", "float32", (4,), b"\x00" * 8)]
            )
        with pytest.raises(SafetensorsError, match="duplicate"):
            write_safetensors(
                tmp_path / "c.safetensors",
                [
                    ("t", "float32", (1,), b"\x00" * 4),
                    ("t", "float32", (1,), b"\x00" * 4),
                ],
            )

    def test_hostile_pickle_opcodes_are_refused(self) -> None:
        # REDUCE of a disallowed callable.
        with pytest.raises(PickleScanError, match="refusing global"):
            scan_state_dict(pickle.dumps({"a": __import__("collections").Counter()}))
        # A protocol-5 out-of-band payload uses opcodes outside the vocabulary.
        with pytest.raises(PickleScanError):
            scan_state_dict(b"\x80\x05\x95\x00\x00\x00\x00\x00\x00\x00\x00X.")
        with pytest.raises(PickleScanError, match="truncated"):
            scan_state_dict(b"\x80\x02X\x10\x00\x00\x00short")

    def test_a_checkpoint_with_a_missing_storage_is_diagnosed(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "weights.pt"
        build_checkpoint(source)
        stripped = tmp_path / "stripped.pt"
        with (
            zipfile.ZipFile(source) as original,
            zipfile.ZipFile(stripped, "w") as target,
        ):
            for info in original.infolist():
                if info.filename.endswith("data/0"):
                    continue
                target.writestr(info, original.read(info.filename))
        document = open_checkpoint(stripped)
        assert document.has_errors
        assert any(
            item.code == "pytorch.storage-missing" for item in document.diagnostics
        )

    def test_a_compressed_checkpoint_storage_is_disclosed(self, tmp_path: Path) -> None:
        source = tmp_path / "weights.pt"
        build_checkpoint(source)
        deflated = tmp_path / "deflated.pt"
        with (
            zipfile.ZipFile(source) as original,
            zipfile.ZipFile(deflated, "w", zipfile.ZIP_DEFLATED) as target,
        ):
            for info in original.infolist():
                target.writestr(info.filename, original.read(info.filename))
        document = open_checkpoint(deflated)
        assert any(
            item.code == "pytorch.compressed-storage" for item in document.diagnostics
        )

    def test_a_checkpoint_whose_root_is_not_a_mapping(self, tmp_path: Path) -> None:
        archive = tmp_path / "listroot.pt"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("x/data.pkl", pickle.dumps([1, 2, 3]))
        with pytest.raises(CheckpointError, match="not a mapping"):
            open_checkpoint(archive)
