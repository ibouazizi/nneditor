"""End-to-end copy-on-write edit and splice export (P0.6).

The acceptance cycle the task defines: change one weight without modifying the
source artifact, hold only a compact delta, undo and redo it, then export a
new ONNX artifact and validate it structurally — differentially, with the
reference ``onnx`` parser, which stays a test-only dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnx
import pytest

from nneditor.adapters.onnx import (
    ModelIndex,
    SpliceExportError,
    export_with_edits,
    index_model,
    read_tensor_bytes,
)
from nneditor.editing.cow import ByteSpanEdit, WorkingRevision
from nneditor.storage.reader import hash_file
from tests.fixtures.onnx_models import build_embedded_model, build_external_model

ELEMENTS = 1024
NEW_VALUE = 2.5
EDITED_ELEMENT = 3


def revision_over(index: ModelIndex) -> WorkingRevision:
    return WorkingRevision(lambda tensor_id: read_tensor_bytes(index, tensor_id))


def edit_one_float(revision: WorkingRevision, tensor_id: str) -> None:
    revision.replace_bytes(tensor_id, EDITED_ELEMENT * 4, struct.pack("<f", NEW_VALUE))


def test_the_full_p06_cycle_on_an_embedded_model(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    values = build_embedded_model(source, elements=ELEMENTS)
    source_hash = hash_file(source)
    index = index_model(source)
    weight = index.main_graph.initializers[0]

    revision = revision_over(index)
    edit_one_float(revision, weight.id)

    # The edit is visible through the revision, invisible in the source.
    edited = np.frombuffer(revision.read(weight.id), dtype=np.float32)
    assert edited[EDITED_ELEMENT] == np.float32(NEW_VALUE)
    np.testing.assert_array_equal(
        np.delete(edited, EDITED_ELEMENT), np.delete(values, EDITED_ELEMENT)
    )
    assert hash_file(source) == source_hash

    # The delta is compact: 8 bytes held against a 4 KiB tensor.
    assert revision.delta_byte_cost == 8
    assert weight.payload is not None and weight.payload.length == 4 * ELEMENTS

    # Undo restores the base view; redo restores the edit.
    revision.undo()
    assert revision.read(weight.id) == values.tobytes()
    revision.redo()

    # Export, then validate structurally with the reference implementation.
    destination = tmp_path / "out" / "edited.onnx"
    report = export_with_edits(index, revision.edits, destination)
    assert report.model_path == destination
    assert report.edited_tensor_ids == (weight.id,)
    assert report.spliced_bytes == 4
    assert not report.is_noop_copy
    assert hash_file(source) == source_hash, "export must not touch the source"

    reloaded = onnx.load(str(destination))
    onnx.checker.check_model(reloaded)
    exported_weight = next(
        init for init in reloaded.graph.initializer if init.name == "weight"
    )
    exported = np.frombuffer(exported_weight.raw_data, dtype=np.float32)
    assert exported[EDITED_ELEMENT] == np.float32(NEW_VALUE)
    np.testing.assert_array_equal(
        np.delete(exported, EDITED_ELEMENT), np.delete(values, EDITED_ELEMENT)
    )

    # The exported artifact reopens through the lazy indexer as well.
    reopened = index_model(destination)
    assert not reopened.diagnostics.has_errors
    assert reopened.content_hash != source_hash


def test_the_cycle_on_an_external_data_model(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source, values = build_external_model(source_dir, elements=ELEMENTS)
    weights_file = source_dir / "weights.bin"
    weights_hash = hash_file(weights_file)
    index = index_model(source)
    weight = index.main_graph.initializers[0]

    revision = revision_over(index)
    edit_one_float(revision, weight.id)

    destination = tmp_path / "exported" / "model.onnx"
    report = export_with_edits(index, revision.edits, destination)
    copied_weights = destination.parent / "weights.bin"
    assert set(report.written_files) == {destination, copied_weights}
    assert hash_file(weights_file) == weights_hash, "source weights untouched"

    exported = np.frombuffer(copied_weights.read_bytes(), dtype=np.float32)
    assert exported[EDITED_ELEMENT] == np.float32(NEW_VALUE)
    np.testing.assert_array_equal(
        np.delete(exported, EDITED_ELEMENT), np.delete(values, EDITED_ELEMENT)
    )

    reloaded = onnx.load(str(destination), load_external_data=True)
    onnx.checker.check_model(reloaded)
    reopened = index_model(destination)
    assert read_tensor_bytes(
        reopened, reopened.main_graph.initializers[0]
    ) == revision.read(weight.id)


def test_a_noop_export_is_a_faithful_copy(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    destination = tmp_path / "copy.onnx"
    report = export_with_edits(index, (), destination)
    assert report.is_noop_copy
    assert hash_file(destination) == hash_file(source), (
        "a no-op export round-trips byte-identically"
    )


def test_exports_never_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    with pytest.raises(SpliceExportError, match="already exists"):
        export_with_edits(index, (), source)
    other = tmp_path / "other.onnx"
    other.write_bytes(b"occupied")
    with pytest.raises(SpliceExportError, match="already exists"):
        export_with_edits(index, (), other)
    assert other.read_bytes() == b"occupied"


def test_an_external_export_refuses_to_clobber_source_weights(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source, _ = build_external_model(source_dir, elements=ELEMENTS)
    index = index_model(source)
    with pytest.raises(SpliceExportError, match="immutable"):
        export_with_edits(index, (), source_dir / "copy.onnx")


def test_stale_before_bytes_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    weight_id = index.main_graph.initializers[0].id
    stale = ByteSpanEdit(
        tensor_id=weight_id,
        offset=0,
        before=b"\xde\xad\xbe\xef",
        after=b"\x00\x00\x00\x00",
    )
    with pytest.raises(SpliceExportError, match="different artifact"):
        export_with_edits(index, (stale,), tmp_path / "out.onnx")


def test_edits_to_unknown_tensors_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    ghost = ByteSpanEdit(tensor_id="ghost", offset=0, before=b"a", after=b"b")
    with pytest.raises(SpliceExportError, match="unknown tensor"):
        export_with_edits(index, (ghost,), tmp_path / "out.onnx")


def test_typed_storage_is_reported_as_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    bias = index.main_graph.initializers[1]
    assert bias.name == "bias", "the fixture stores bias as typed float_data"
    edit = ByteSpanEdit(bias.id, 0, b"\x00\x00\x00\x3f", b"\x00\x00\x00\x40")
    with pytest.raises(SpliceExportError, match="embedded typed"):
        export_with_edits(index, (edit,), tmp_path / "out.onnx")


def test_spans_past_the_payload_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    weight = index.main_graph.initializers[0]
    oversized = ByteSpanEdit(weight.id, 4 * ELEMENTS - 2, b"\x00" * 4, b"\x01" * 4)
    with pytest.raises(SpliceExportError, match="exceeds"):
        export_with_edits(index, (oversized,), tmp_path / "out.onnx")


def test_a_failed_export_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    build_embedded_model(source, elements=ELEMENTS)
    index = index_model(source)
    revision = revision_over(index)
    weight = index.main_graph.initializers[0]
    edit_one_float(revision, weight.id)
    stale = ByteSpanEdit(weight.id, 0, b"\xde\xad\xbe\xef", b"\x00" * 4)
    destination = tmp_path / "out" / "edited.onnx"
    with pytest.raises(SpliceExportError):
        export_with_edits(index, (*revision.edits, stale), destination)
    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())
