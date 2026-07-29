"""Tests for revisions, sidecar persistence, and diff previews (P3.5-P3.7)."""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import pytest

from nneditor.application.editing import EditingController
from nneditor.application.edits_store import RevisionSidecarStore
from nneditor.editing.cow import ByteSpanEdit, EditError
from nneditor.editing.diff import preview_diff
from nneditor.editing.revisions import (
    ReplaceTensorBytes,
    Revision,
    RevisionChain,
    ValidationState,
    command_from_json,
    command_to_json,
    revision_id_for,
)
from nneditor.storage.store import TensorStore
from tests.unit.test_tensor_store import embedded_document

BASE = {"weight": bytes(range(16)), "bias": b"\xff" * 4}


def make_chain() -> RevisionChain:
    return RevisionChain(BASE.__getitem__)


class TestRevisionIdentity:
    def test_ids_are_deterministic_and_content_bound(self) -> None:
        chain_a = make_chain()
        chain_b = make_chain()
        first_a = chain_a.apply_replace("weight", 0, b"\xaa")
        first_b = chain_b.apply_replace("weight", 0, b"\xaa")
        assert first_a.id == first_b.id, "same base, same command, same id"
        second = chain_a.apply_replace("weight", 1, b"\xbb")
        assert second.parent_id == first_a.id
        assert second.id != first_a.id

    def test_a_forged_id_is_rejected(self) -> None:
        parsed = command_from_json(
            {
                "kind": "replace-tensor-bytes",
                "tensor_id": "t",
                "offset": 0,
                "before": "00",
                "after": "01",
            }
        )
        assert isinstance(parsed, ReplaceTensorBytes)
        command = ReplaceTensorBytes(parsed.edit)
        with pytest.raises(EditError, match="does not match"):
            Revision(
                id="rev:0000000000000000",
                parent_id=None,
                commands=(command,),
                validation=ValidationState(ok=True),
            )

    def test_an_empty_revision_is_rejected(self) -> None:
        with pytest.raises(EditError, match="at least one command"):
            Revision(
                id=revision_id_for(None, ()),
                parent_id=None,
                commands=(),
                validation=ValidationState(ok=True),
            )

    def test_failed_validation_needs_findings(self) -> None:
        with pytest.raises(EditError, match="findings"):
            ValidationState(ok=False)


class TestChainMechanics:
    def test_range_edits_never_read_the_whole_tensor(self) -> None:
        source = bytes(range(64))
        whole_reads: list[str] = []
        range_reads: list[tuple[str, int, int]] = []

        def read_whole(tensor_id: str) -> bytes:
            whole_reads.append(tensor_id)
            return source

        def read_range(tensor_id: str, offset: int, length: int) -> bytes:
            range_reads.append((tensor_id, offset, length))
            return source[offset : offset + length]

        chain = RevisionChain(
            read_whole,
            read_range=read_range,
            tensor_length=lambda _tensor_id: len(source),
        )
        chain.apply_replace("weight", 24, b"\xaa\xbb")

        assert whole_reads == []
        assert range_reads == [("weight", 24, 2)]
        assert chain.read("weight", offset=23, length=4) == b"\x17\xaa\xbb\x1a"
        assert whole_reads == []

    def test_undo_and_redo_move_between_revisions(self) -> None:
        chain = make_chain()
        first = chain.apply_replace("weight", 0, b"\xaa")
        second = chain.apply_replace("weight", 1, b"\xbb")
        assert chain.read("weight")[:2] == b"\xaa\xbb"
        assert chain.current_revision_id == second.id

        undone = chain.undo()
        assert undone is second
        assert chain.read("weight")[:2] == b"\xaa\x01"
        assert chain.current_revision_id == first.id
        assert chain.can_redo

        chain.undo()
        assert chain.read("weight") == BASE["weight"]
        assert chain.current_revision_id is None
        assert not chain.is_dirty

        assert chain.redo() is first
        assert chain.redo() is second
        assert chain.read("weight")[:2] == b"\xaa\xbb"

    def test_applying_after_undo_discards_the_redo_tail(self) -> None:
        chain = make_chain()
        chain.apply_replace("weight", 0, b"\xaa")
        second = chain.apply_replace("weight", 1, b"\xbb")
        chain.undo()
        replacement = chain.apply_replace("weight", 2, b"\xcc")
        assert not chain.can_redo
        assert second not in chain.revisions
        assert replacement in chain.applied

    def test_validation_failures_leave_no_revision(self) -> None:
        chain = make_chain()
        with pytest.raises(EditError, match="outside"):
            chain.apply_replace("weight", 15, b"\xaa\xbb")
        with pytest.raises(EditError, match="already holds"):
            chain.apply_replace("weight", 2, bytes([2, 3]))
        with pytest.raises(EditError, match="at least one byte"):
            chain.apply_replace("weight", 0, b"")
        with pytest.raises(EditError, match="not readable"):
            chain.apply_replace("ghost", 0, b"\x00")
        assert chain.revisions == ()
        assert not chain.is_dirty

    def test_deltas_are_compact_and_grouped(self) -> None:
        chain = make_chain()
        chain.apply_replace("weight", 0, b"\xaa")
        revision = chain.apply_replace("bias", 0, b"\x00\x00")
        assert chain.delta_byte_cost == 2 + 4
        assert chain.edited_tensor_ids() == ("weight", "bias")
        assert set(revision.tensor_deltas) == {"bias"}

    def test_restore_verifies_parent_links(self) -> None:
        donor = make_chain()
        donor.apply_replace("weight", 0, b"\xaa")
        donor.apply_replace("weight", 1, b"\xbb")
        good = list(donor.revisions)

        fresh = make_chain()
        fresh.restore(good, 1)
        assert fresh.can_undo and fresh.can_redo

        with pytest.raises(EditError, match="corrupt"):
            make_chain().restore(list(reversed(good)), 1)
        with pytest.raises(EditError, match="cursor"):
            make_chain().restore(good, 3)


class TestCommandManifest:
    def test_round_trip(self) -> None:
        chain = make_chain()
        revision = chain.apply_replace("weight", 3, b"\xde\xad")
        (command,) = revision.commands
        assert isinstance(command, ReplaceTensorBytes)
        assert command_from_json(command_to_json(command)) == command
        assert "3" in command.summary and "weight" in command.summary

    def test_malformed_manifests_are_rejected(self) -> None:
        with pytest.raises(EditError, match="unknown command kind"):
            command_from_json({"kind": "explode"})
        with pytest.raises(EditError, match="malformed"):
            command_from_json({"kind": "replace-tensor-bytes"})


class TestSidecarPersistence:
    def test_crash_recovery_restores_chain_and_cursor(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        sidecar = RevisionSidecarStore(tmp_path / "state")

        with TensorStore(document) as store:
            controller = EditingController(document, store, sidecar=sidecar)
            controller.replace_bytes(weight_id, 0, b"\x01\x02\x03\x04")
            controller.replace_bytes(weight_id, 8, b"\x05\x06")
            controller.undo()
            edited_view = controller.read(weight_id)

        # "Crash": a new store and controller recover from the sidecar alone.
        with TensorStore(document) as store:
            recovered = EditingController(document, store, sidecar=sidecar)
            assert recovered.recovered_revisions == 2
            assert recovered.can_undo and recovered.can_redo
            assert recovered.read(weight_id) == edited_view
            recovered.redo()
            assert recovered.read(weight_id)[8:10] == b"\x05\x06"
        assert Path(document.source.path).read_bytes() is not None
        assert not recovered.sidecar_was_quarantined

    def test_corrupt_sidecars_are_quarantined_not_fatal(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        sidecar = RevisionSidecarStore(tmp_path / "state")
        target = sidecar._path_for(document.source.content_hash)
        target.parent.mkdir(parents=True)
        target.write_text("{broken", encoding="utf-8")

        with TensorStore(document) as store:
            controller = EditingController(document, store, sidecar=sidecar)
            assert not controller.is_dirty
        assert not target.exists()
        assert target.with_name(target.name + ".corrupt").exists()

    def test_tampered_chains_are_rejected_on_load(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        sidecar = RevisionSidecarStore(tmp_path / "state")
        with TensorStore(document) as store:
            controller = EditingController(document, store, sidecar=sidecar)
            controller.replace_bytes(weight_id, 0, b"\x01")

        target = sidecar._path_for(document.source.content_hash)
        data = json.loads(target.read_text(encoding="utf-8"))
        data["revisions"][0]["commands"][0]["after"] = "ff"  # forge the delta
        target.write_text(json.dumps(data), encoding="utf-8")

        assert sidecar.load(document.source.content_hash) is None, (
            "the forged revision no longer matches its id"
        )
        assert target.with_name(target.name + ".corrupt").exists()

    def test_unreplayable_chain_is_quarantined_without_blocking_open(
        self, tmp_path: Path
    ) -> None:
        document, _values = embedded_document(tmp_path)
        sidecar = RevisionSidecarStore(tmp_path / "state")
        command = ReplaceTensorBytes(ByteSpanEdit("t:missing", 0, b"\x00", b"\x01"))
        revision = Revision(
            id=revision_id_for(None, (command,)),
            parent_id=None,
            commands=(command,),
            validation=ValidationState(ok=True),
        )
        sidecar.save(document.source.content_hash, [revision], 1)

        with TensorStore(document) as store:
            controller = EditingController(document, store, sidecar=sidecar)

        target = sidecar._path_for(document.source.content_hash)
        assert controller.sidecar_was_quarantined
        assert not controller.is_dirty
        assert not target.exists()
        assert target.with_name(target.name + ".corrupt").exists()

    def test_discard_all_returns_to_base(self, tmp_path: Path) -> None:
        document, values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        sidecar = RevisionSidecarStore(tmp_path / "state")
        with TensorStore(document) as store:
            controller = EditingController(document, store, sidecar=sidecar)
            controller.replace_bytes(weight_id, 0, b"\x01")
            controller.discard_all()
            assert not controller.is_dirty
            assert controller.read(weight_id) == values.tobytes()
        assert sidecar.load(document.source.content_hash) is None

    def test_stable_view_blocks_revision_mutation_until_a_job_finishes(
        self,
        tmp_path: Path,
    ) -> None:
        document, _values = embedded_document(tmp_path)
        tensor_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            controller = EditingController(document, store)
            started = threading.Event()
            finished = threading.Event()

            def mutate() -> None:
                started.set()
                controller.replace_bytes(tensor_id, 0, b"\x01")
                finished.set()

            with controller.stable_view():
                worker = threading.Thread(target=mutate, daemon=True)
                worker.start()
                assert started.wait(1)
                assert not finished.wait(0.05)
                revision_id = controller.current_revision_id
                first = controller.read(tensor_id, offset=0, length=1)
                assert revision_id is None
                assert first != b"\x01"

            worker.join(timeout=1)
            assert finished.is_set()
            assert controller.current_revision_id is not None


class TestDiffPreview:
    def test_preview_comes_from_deltas_alone(self, tmp_path: Path) -> None:
        document, values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            controller = EditingController(document, store)
            new_value = struct.pack("<f", 42.5)
            controller.replace_bytes(weight_id, 4, new_value)
            preview = controller.preview()

        assert not preview.is_empty
        assert preview.target_revision_id == controller.current_revision_id
        (tensor_diff,) = preview.tensors
        assert tensor_diff.tensor_id == weight_id
        assert tensor_diff.element_type == "float32"
        assert tensor_diff.span_count == 1
        assert tensor_diff.bytes_changed == 4
        assert tensor_diff.elements_changed == 1
        (span,) = tensor_diff.spans
        assert span.offset == 4
        assert span.before_values is not None
        assert list(span.before_values) == pytest.approx([float(values[1])])
        assert span.after_values == (42.5,)
        assert not span.values_truncated

    def test_unaligned_spans_stay_byte_level(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            controller = EditingController(document, store)
            controller.replace_bytes(weight_id, 1, b"\x7f")
            preview = controller.preview()
        (tensor_diff,) = preview.tensors
        (span,) = tensor_diff.spans
        assert span.before_values is None and span.after_values is None
        assert tensor_diff.elements_changed == 1, "the containing element"

    def test_empty_chain_previews_empty(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        preview = preview_diff(document, ())
        assert preview.is_empty
        assert preview.commands == ()
        assert preview.target_revision_id is None
