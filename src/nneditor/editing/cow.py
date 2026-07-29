"""Copy-on-write byte edits over immutable base tensors (task P0.6).

A :class:`WorkingRevision` overlays edits on tensors it can only read. The
source artifact is never written: an edit is a :class:`ByteSpanEdit` recording
which bytes it replaced and what it replaced them with, so the revision's
storage cost is proportional to what changed, not to the tensors involved —
that is the "compact delta" P0.6 exists to prove.

Recording the ``before`` bytes buys two properties:

* undo needs no access to the base artifact, and
* an exporter can verify an edit still matches the artifact it is being
  applied to, instead of silently splicing stale bytes into the wrong file.

The revision is deliberately format-agnostic: it sees tensors only as ids
resolved to bytes by a caller-supplied reader. Only same-length replacements
are supported — shape- or dtype-changing surgery is a Phase 4 concern with its
own validation pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ByteSpanEdit", "EditError", "WorkingRevision"]

BaseTensorReader = Callable[[str], bytes]
"""Resolves a tensor id to that tensor's unedited bytes."""


class EditError(ValueError):
    """Raised when an edit cannot be recorded, undone, or redone."""


@dataclass(frozen=True, slots=True)
class ByteSpanEdit:
    """One reversible replacement of ``len(after)`` bytes inside a tensor."""

    tensor_id: str
    offset: int
    before: bytes
    after: bytes

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise EditError("an edit must name a tensor")
        if self.offset < 0:
            raise EditError(f"edit offset must be non-negative, got {self.offset}")
        if len(self.after) == 0:
            raise EditError("an edit must replace at least one byte")
        if len(self.before) != len(self.after):
            raise EditError(
                "only same-length replacement is supported: before is "
                f"{len(self.before)} bytes, after is {len(self.after)}"
            )

    @property
    def length(self) -> int:
        """Bytes replaced."""
        return len(self.after)

    @property
    def end(self) -> int:
        """First tensor-relative offset past the edited span."""
        return self.offset + self.length

    @property
    def byte_cost(self) -> int:
        """Delta storage this edit needs (both directions of the change)."""
        return len(self.before) + len(self.after)

    def inverted(self) -> ByteSpanEdit:
        """The edit that undoes this one."""
        return ByteSpanEdit(
            tensor_id=self.tensor_id,
            offset=self.offset,
            before=self.after,
            after=self.before,
        )


class WorkingRevision:
    """An ordered stack of byte edits over read-only base tensors.

    ``read`` fetches the base bytes on every call rather than caching them:
    cache policy belongs to the Phase 3 storage layer, and this prototype must
    demonstrate correctness, not throughput.
    """

    __slots__ = ("_applied", "_read_base", "_redo")

    def __init__(self, read_base: BaseTensorReader) -> None:
        self._read_base = read_base
        self._applied: list[ByteSpanEdit] = []
        self._redo: list[ByteSpanEdit] = []

    @property
    def edits(self) -> tuple[ByteSpanEdit, ...]:
        """The applied edits in order — the revision's command manifest."""
        return tuple(self._applied)

    @property
    def is_dirty(self) -> bool:
        return bool(self._applied)

    @property
    def can_undo(self) -> bool:
        return bool(self._applied)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def delta_byte_cost(self) -> int:
        """Total delta storage held by this revision."""
        return sum(edit.byte_cost for edit in self._applied)

    def edited_tensor_ids(self) -> tuple[str, ...]:
        """Tensors touched by applied edits, in first-edit order."""
        seen: dict[str, None] = {}
        for edit in self._applied:
            seen.setdefault(edit.tensor_id, None)
        return tuple(seen)

    def read(self, tensor_id: str) -> bytes:
        """The tensor's bytes as this revision sees them."""
        base = self._read_base(tensor_id)
        edits = [edit for edit in self._applied if edit.tensor_id == tensor_id]
        if not edits:
            return base
        buffer = bytearray(base)
        for edit in edits:
            buffer[edit.offset : edit.end] = edit.after
        return bytes(buffer)

    def replace_bytes(
        self, tensor_id: str, offset: int, replacement: bytes
    ) -> ByteSpanEdit:
        """Record a replacement of ``replacement`` at ``offset``.

        The overwritten bytes are captured from the revision's *current* view,
        so stacked edits to the same span undo in reverse order. Applying a new
        edit discards the redo history, as any linear undo model must.
        """
        current = self.read(tensor_id)
        if offset < 0 or offset + len(replacement) > len(current):
            raise EditError(
                f"span [{offset}, {offset + len(replacement)}) is outside "
                f"tensor {tensor_id!r} of {len(current)} bytes"
            )
        edit = ByteSpanEdit(
            tensor_id=tensor_id,
            offset=offset,
            before=current[offset : offset + len(replacement)],
            after=replacement,
        )
        if edit.before == edit.after:
            raise EditError(
                f"the span at offset {offset} of tensor {tensor_id!r} already "
                "holds the replacement bytes"
            )
        self._applied.append(edit)
        self._redo.clear()
        return edit

    def undo(self) -> ByteSpanEdit:
        """Withdraw the most recent edit and return it."""
        if not self._applied:
            raise EditError("nothing to undo")
        edit = self._applied.pop()
        self._redo.append(edit)
        return edit

    def redo(self) -> ByteSpanEdit:
        """Re-apply the most recently undone edit and return it."""
        if not self._redo:
            raise EditError("nothing to redo")
        edit = self._redo.pop()
        self._applied.append(edit)
        return edit
