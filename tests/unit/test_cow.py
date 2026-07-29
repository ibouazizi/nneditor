"""Tests for copy-on-write revisions and byte-span edits (P0.6)."""

from __future__ import annotations

import pytest

from nneditor.editing.cow import ByteSpanEdit, EditError, WorkingRevision

BASE = {
    "weight": bytes(range(16)),
    "bias": b"\xff" * 4,
}


def make_revision() -> WorkingRevision:
    return WorkingRevision(BASE.__getitem__)


class TestByteSpanEdit:
    def test_lengths_must_match(self) -> None:
        with pytest.raises(EditError, match="same-length"):
            ByteSpanEdit("t", 0, b"ab", b"abc")

    def test_empty_replacement_is_rejected(self) -> None:
        with pytest.raises(EditError, match="at least one byte"):
            ByteSpanEdit("t", 0, b"", b"")

    def test_negative_offset_is_rejected(self) -> None:
        with pytest.raises(EditError, match="non-negative"):
            ByteSpanEdit("t", -1, b"a", b"b")

    def test_a_tensor_id_is_required(self) -> None:
        with pytest.raises(EditError, match="name a tensor"):
            ByteSpanEdit("", 0, b"a", b"b")

    def test_byte_cost_counts_both_directions(self) -> None:
        edit = ByteSpanEdit("t", 3, b"ab", b"cd")
        assert edit.byte_cost == 4
        assert edit.length == 2
        assert edit.end == 5

    def test_inverted_swaps_before_and_after(self) -> None:
        edit = ByteSpanEdit("t", 3, b"ab", b"cd")
        assert edit.inverted() == ByteSpanEdit("t", 3, b"cd", b"ab")


class TestWorkingRevision:
    def test_reads_pass_through_when_clean(self) -> None:
        revision = make_revision()
        assert revision.read("weight") == BASE["weight"]
        assert not revision.is_dirty

    def test_an_edit_changes_only_its_span(self) -> None:
        revision = make_revision()
        edit = revision.replace_bytes("weight", 4, b"\xaa\xbb")
        assert edit.before == bytes([4, 5])
        seen = revision.read("weight")
        assert seen[4:6] == b"\xaa\xbb"
        assert seen[:4] == BASE["weight"][:4]
        assert seen[6:] == BASE["weight"][6:]
        assert revision.read("bias") == BASE["bias"]

    def test_the_delta_is_compact(self) -> None:
        revision = make_revision()
        revision.replace_bytes("weight", 0, b"\x99")
        assert revision.delta_byte_cost == 2, "one byte each way, not the tensor"

    def test_out_of_bounds_spans_are_rejected(self) -> None:
        revision = make_revision()
        with pytest.raises(EditError, match="outside"):
            revision.replace_bytes("weight", 15, b"ab")
        with pytest.raises(EditError, match="outside"):
            revision.replace_bytes("weight", -1, b"a")

    def test_a_noop_replacement_is_rejected(self) -> None:
        revision = make_revision()
        with pytest.raises(EditError, match="already holds"):
            revision.replace_bytes("weight", 2, bytes([2, 3]))

    def test_undo_redo_round_trip(self) -> None:
        revision = make_revision()
        revision.replace_bytes("weight", 0, b"\xaa")
        revision.replace_bytes("weight", 1, b"\xbb")
        assert revision.read("weight")[:2] == b"\xaa\xbb"

        undone = revision.undo()
        assert undone.after == b"\xbb"
        assert revision.read("weight")[:2] == b"\xaa\x01"
        assert revision.can_redo

        revision.undo()
        assert revision.read("weight") == BASE["weight"]
        assert not revision.is_dirty

        revision.redo()
        revision.redo()
        assert revision.read("weight")[:2] == b"\xaa\xbb"
        assert not revision.can_redo

    def test_undo_and_redo_on_empty_stacks_fail(self) -> None:
        revision = make_revision()
        with pytest.raises(EditError, match="nothing to undo"):
            revision.undo()
        with pytest.raises(EditError, match="nothing to redo"):
            revision.redo()

    def test_a_new_edit_discards_redo_history(self) -> None:
        revision = make_revision()
        revision.replace_bytes("weight", 0, b"\xaa")
        revision.undo()
        revision.replace_bytes("weight", 0, b"\xbb")
        assert not revision.can_redo

    def test_stacked_edits_to_one_span_undo_in_order(self) -> None:
        revision = make_revision()
        revision.replace_bytes("weight", 0, b"\xaa")
        revision.replace_bytes("weight", 0, b"\xbb")
        assert revision.read("weight")[0] == 0xBB
        revision.undo()
        assert revision.read("weight")[0] == 0xAA, (
            "the second edit's before-bytes came from the first edit's view"
        )

    def test_edited_tensor_ids_preserve_first_edit_order(self) -> None:
        revision = make_revision()
        revision.replace_bytes("bias", 0, b"\x00")
        revision.replace_bytes("weight", 0, b"\xaa")
        revision.replace_bytes("bias", 1, b"\x00")
        assert revision.edited_tensor_ids() == ("bias", "weight")
        assert len(revision.edits) == 3
