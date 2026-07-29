"""Revisions and commands (task P3.5), the durable form of editing.

The IR spec (§4) fixes the shape this module implements: a *revision* is an
immutable snapshot identified by a hash of its parent and its command
manifest; undo and redo move a cursor **between** revisions and never mutate
one; deltas store only what changed; and because the base revision is the
immutable source artifact, discarding every revision always recovers the
original file.

Identifiers are deterministic — the same base artifact and the same commands
produce the same revision ids on any machine — which is what lets sidecar
persistence (P3.6), diff previews (P3.7), and future export manifests refer
to revisions stably.

A command that fails validation raises with every finding and creates
nothing: failed edits must not leave a revision behind (the Phase 4 exit gate
depends on exactly this property).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Final, cast

from nneditor.editing.commands import (
    EditCommand,
    QuantizeGraph,
    ReplaceTensorBytes,
    ResizeTensor,
    apply_command,
    command_from_json,
    command_to_json,
)
from nneditor.editing.cow import ByteSpanEdit, EditError
from nneditor.ir.core import Document

__all__ = [
    "BASE_PARENT",
    "ReplaceTensorBytes",
    "Revision",
    "RevisionChain",
    "ValidationState",
    "command_from_json",
    "command_to_json",
    "revision_id_for",
]

BASE_PARENT: Final = None
"""A revision with no parent sits directly on the source artifact."""

_ID_DIGEST_LENGTH: Final = 16

BaseTensorReader = Callable[[str], bytes]
BaseTensorRangeReader = Callable[[str, int, int], bytes]
TensorLengthReader = Callable[[str], int | None]


def _chain_locked[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Serialize one ``RevisionChain`` operation, preserving its signature."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        chain = cast("RevisionChain", args[0])
        with chain._lock:
            return method(*args, **kwargs)

    return guarded


@dataclass(frozen=True, slots=True)
class ValidationState:
    """What the last validation pass concluded about a revision."""

    ok: bool
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ok and not self.findings:
            raise EditError("a failed validation must state its findings")


def revision_id_for(parent_id: str | None, commands: tuple[EditCommand, ...]) -> str:
    """The deterministic identity of a revision: parent plus manifest."""
    manifest = json.dumps(
        {
            "parent": parent_id,
            "commands": [command_to_json(command) for command in commands],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(manifest).hexdigest()[:_ID_DIGEST_LENGTH]
    return f"rev:{digest}"


@dataclass(frozen=True, slots=True)
class Revision:
    """One immutable snapshot in the chain."""

    id: str
    parent_id: str | None
    commands: tuple[EditCommand, ...]
    validation: ValidationState

    def __post_init__(self) -> None:
        if not self.commands:
            raise EditError("a revision must apply at least one command")
        expected = revision_id_for(self.parent_id, self.commands)
        if self.id != expected:
            raise EditError(
                f"revision id {self.id!r} does not match its content "
                f"(expected {expected!r})"
            )

    @property
    def tensor_deltas(self) -> Mapping[str, tuple[ByteSpanEdit, ...]]:
        """Copy-on-write overlays, keyed by tensor id, in command order."""
        deltas: dict[str, list[ByteSpanEdit]] = {}
        for command in self.commands:
            if isinstance(command, ReplaceTensorBytes):
                deltas.setdefault(command.edit.tensor_id, []).append(command.edit)
        return {tensor_id: tuple(edits) for tensor_id, edits in deltas.items()}

    @property
    def delta_byte_cost(self) -> int:
        total = 0
        for command in self.commands:
            if isinstance(command, ReplaceTensorBytes):
                total += command.edit.byte_cost
            elif isinstance(command, ResizeTensor):
                total += len(command.before_bytes) + len(command.after_bytes)
            elif isinstance(command, QuantizeGraph):
                total += len(command.scale_bytes) + len(command.zero_bytes)
        return total


class RevisionChain:
    """A linear history over one base artifact, with a movable cursor.

    ``revisions[:cursor]`` are in effect; the tail beyond the cursor is redo
    history, discarded the moment a new command is applied — the linear model
    every undo stack promises.
    """

    __slots__ = (
        "_base_document",
        "_cursor",
        "_lock",
        "_read_base",
        "_read_range",
        "_revisions",
        "_tensor_length",
    )

    def __init__(
        self,
        read_base: BaseTensorReader,
        *,
        read_range: BaseTensorRangeReader | None = None,
        tensor_length: TensorLengthReader | None = None,
        base_document: Document | None = None,
    ) -> None:
        self._read_base = read_base
        self._read_range = read_range
        self._tensor_length = tensor_length
        self._base_document = base_document
        self._revisions: list[Revision] = []
        self._cursor = 0
        self._lock = threading.RLock()

    # -- state -------------------------------------------------------------

    @property
    @_chain_locked
    def revisions(self) -> tuple[Revision, ...]:
        """Every retained revision, including undone redo candidates."""
        return tuple(self._revisions)

    @property
    @_chain_locked
    def applied(self) -> tuple[Revision, ...]:
        """The revisions currently in effect."""
        return tuple(self._revisions[: self._cursor])

    @property
    @_chain_locked
    def current_revision_id(self) -> str | None:
        """The id of the newest applied revision, or ``None`` at base."""
        if self._cursor == 0:
            return BASE_PARENT
        return self._revisions[self._cursor - 1].id

    @property
    @_chain_locked
    def is_dirty(self) -> bool:
        return self._cursor > 0

    @property
    @_chain_locked
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    @_chain_locked
    def can_redo(self) -> bool:
        return self._cursor < len(self._revisions)

    @property
    @_chain_locked
    def delta_byte_cost(self) -> int:
        """Delta storage held by the applied revisions."""
        return sum(revision.delta_byte_cost for revision in self.applied)

    @_chain_locked
    def edited_tensor_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for revision in self.applied:
            for command in revision.commands:
                if isinstance(command, ReplaceTensorBytes):
                    seen.setdefault(command.edit.tensor_id, None)
                elif isinstance(command, ResizeTensor):
                    seen.setdefault(command.tensor_id, None)
        return tuple(seen)

    @property
    @_chain_locked
    def document(self) -> Document | None:
        current = self._base_document
        if current is None:
            return None
        for revision in self.applied:
            for command in revision.commands:
                current = apply_command(current, command)
        return current

    # -- reads -------------------------------------------------------------

    @_chain_locked
    def read(
        self, tensor_id: str, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        """The tensor's bytes as the applied revisions see them."""
        if offset < 0 or (length is not None and length < 0):
            raise EditError("tensor read ranges must be non-negative")
        generated = self._generated_payload(tensor_id)
        if generated is not None:
            if length is None:
                length = max(0, len(generated) - offset)
            if offset + length > len(generated):
                raise EditError(
                    f"range [{offset}, {offset + length}) is outside generated "
                    f"tensor {tensor_id!r}"
                )
            return generated[offset : offset + length]
        if length is None:
            base = self._read_base(tensor_id)
            length = len(base) - offset
            if length < 0:
                raise EditError(f"offset {offset} is outside tensor {tensor_id!r}")
            base = base[offset : offset + length]
        elif self._read_range is not None:
            base = self._read_range(tensor_id, offset, length)
        else:
            whole = self._read_base(tensor_id)
            if offset + length > len(whole):
                raise EditError(
                    f"range [{offset}, {offset + length}) is outside tensor "
                    f"{tensor_id!r}"
                )
            base = whole[offset : offset + length]
        edits = [
            command.edit
            for revision in self.applied
            for command in revision.commands
            if isinstance(command, ReplaceTensorBytes)
            if command.edit.tensor_id == tensor_id
        ]
        if not edits:
            return base
        buffer = bytearray(base)
        for edit in edits:
            overlap_start = max(offset, edit.offset)
            overlap_end = min(offset + length, edit.end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - edit.offset
            target_start = overlap_start - offset
            size = overlap_end - overlap_start
            buffer[target_start : target_start + size] = edit.after[
                source_start : source_start + size
            ]
        return bytes(buffer)

    @_chain_locked
    def _generated_payload(self, tensor_id: str) -> bytes | None:
        current: bytes | None = None
        for revision in self.applied:
            for command in revision.commands:
                if isinstance(command, ResizeTensor) and command.tensor_id == tensor_id:
                    current = command.after_bytes
                elif isinstance(command, QuantizeGraph):
                    if command.scale_tensor.id == tensor_id:
                        current = command.scale_bytes
                    elif command.zero_tensor.id == tensor_id:
                        current = command.zero_bytes
                elif (
                    current is not None
                    and isinstance(command, ReplaceTensorBytes)
                    and command.edit.tensor_id == tensor_id
                ):
                    edit = command.edit
                    if edit.end > len(current):
                        raise EditError("generated tensor edit is outside its payload")
                    buffer = bytearray(current)
                    buffer[edit.offset : edit.end] = edit.after
                    current = bytes(buffer)
        return current

    @_chain_locked
    def byte_length(self, tensor_id: str) -> int | None:
        generated = self._generated_payload(tensor_id)
        if generated is not None:
            return len(generated)
        if self._tensor_length is not None:
            return self._tensor_length(tensor_id)
        return len(self._read_base(tensor_id))

    # -- mutation ----------------------------------------------------------

    @_chain_locked
    def apply_replace(
        self, tensor_id: str, offset: int, replacement: bytes
    ) -> Revision:
        """Validate and commit one byte-span replacement as a new revision.

        Validation failures raise with every finding collected; nothing is
        created or truncated on failure.
        """
        findings: list[str] = []
        if not replacement:
            findings.append("the replacement must be at least one byte")
        # Bounds come from the *current* revision view: a resized or
        # command-generated tensor has a length the base artifact never had.
        try:
            total = self.byte_length(tensor_id)
        except Exception as error:
            raise EditError(
                f"tensor {tensor_id!r} is not readable, so it cannot be edited: {error}"
            ) from error
        if total is None:
            try:
                total = len(self._read_base(tensor_id))
            except Exception as error:
                raise EditError(
                    f"tensor {tensor_id!r} is not readable, so it cannot be edited: "
                    f"{error}"
                ) from error
        if offset < 0 or offset + len(replacement) > total:
            findings.append(
                f"span [{offset}, {offset + len(replacement)}) is outside the "
                f"{total}-byte tensor"
            )
        current = b""
        if not findings:
            try:
                current = self.read(tensor_id, offset=offset, length=len(replacement))
            except Exception as error:
                raise EditError(
                    f"tensor {tensor_id!r} is not readable, so it cannot be edited: "
                    f"{error}"
                ) from error
        if not findings and current == replacement:
            findings.append("the span already holds the replacement bytes")
        if findings:
            raise EditError(f"cannot edit tensor {tensor_id!r}: " + "; ".join(findings))

        edit = ByteSpanEdit(
            tensor_id=tensor_id,
            offset=offset,
            before=current,
            after=replacement,
        )
        command = ReplaceTensorBytes(edit)
        parent = self.current_revision_id
        revision = Revision(
            id=revision_id_for(parent, (command,)),
            parent_id=parent,
            commands=(command,),
            validation=ValidationState(ok=True),
        )
        del self._revisions[self._cursor :]
        self._revisions.append(revision)
        self._cursor += 1
        return revision

    @_chain_locked
    def apply_commands(
        self,
        commands: tuple[EditCommand, ...],
        validation: ValidationState,
    ) -> Revision:
        """Commit already-validated commands atomically as one revision."""
        if not validation.ok:
            raise EditError(
                "cannot commit a failed validation: " + "; ".join(validation.findings)
            )
        if not commands:
            raise EditError("a revision must apply at least one command")
        current = self.document
        if current is None and any(
            not isinstance(command, ReplaceTensorBytes) for command in commands
        ):
            raise EditError("graph commands need a base document")

        # Byte preconditions are verified against the *evolving* batch state
        # — an earlier command in this revision changes what a later one must
        # see — and they are verified whether or not a base document exists.
        overlays: dict[str, bytes] = {}
        batch_edits: dict[str, list[ByteSpanEdit]] = {}

        def batch_view(tensor_id: str, offset: int, length: int | None) -> bytes:
            overlay = overlays.get(tensor_id)
            if overlay is not None:
                end = len(overlay) if length is None else offset + length
                if end > len(overlay):
                    raise EditError(
                        f"span [{offset}, {end}) is outside the batch view of "
                        f"tensor {tensor_id!r}"
                    )
                return overlay[offset:end]
            try:
                data = self.read(tensor_id, offset=offset, length=length)
            except EditError:
                raise
            except Exception as error:
                raise EditError(
                    f"tensor {tensor_id!r} is not readable, so its edit "
                    f"precondition cannot be verified: {error}"
                ) from error
            edits = batch_edits.get(tensor_id)
            if not edits:
                return data
            end = offset + len(data)
            buffer = bytearray(data)
            for edit in edits:
                overlap_start = max(offset, edit.offset)
                overlap_end = min(end, edit.end)
                if overlap_start >= overlap_end:
                    continue
                buffer[overlap_start - offset : overlap_end - offset] = edit.after[
                    overlap_start - edit.offset : overlap_end - edit.offset
                ]
            return bytes(buffer)

        for command in commands:
            if isinstance(command, ReplaceTensorBytes):
                edit = command.edit
                actual = batch_view(edit.tensor_id, edit.offset, edit.length)
                if actual != edit.before:
                    raise EditError(
                        "tensor transformation precondition no longer holds"
                    )
                overlay = overlays.get(edit.tensor_id)
                if overlay is not None:
                    buffer = bytearray(overlay)
                    buffer[edit.offset : edit.end] = edit.after
                    overlays[edit.tensor_id] = bytes(buffer)
                else:
                    batch_edits.setdefault(edit.tensor_id, []).append(edit)
            elif isinstance(command, ResizeTensor):
                actual = batch_view(command.tensor_id, 0, None)
                if actual != command.before_bytes:
                    raise EditError("tensor resize precondition no longer holds")
                overlays[command.tensor_id] = command.after_bytes
                batch_edits.pop(command.tensor_id, None)
            elif isinstance(command, QuantizeGraph):
                overlays[command.scale_tensor.id] = command.scale_bytes
                overlays[command.zero_tensor.id] = command.zero_bytes
            if current is not None:
                current = apply_command(current, command)
        revision = Revision(
            id=revision_id_for(self.current_revision_id, commands),
            parent_id=self.current_revision_id,
            commands=commands,
            validation=validation,
        )
        del self._revisions[self._cursor :]
        self._revisions.append(revision)
        self._cursor += 1
        return revision

    @_chain_locked
    def undo(self) -> Revision:
        """Step back one revision and return the one withdrawn."""
        if not self.can_undo:
            raise EditError("nothing to undo")
        self._cursor -= 1
        return self._revisions[self._cursor]

    @_chain_locked
    def redo(self) -> Revision:
        """Re-apply the next undone revision and return it."""
        if not self.can_redo:
            raise EditError("nothing to redo")
        revision = self._revisions[self._cursor]
        self._cursor += 1
        return revision

    # -- restoration (used by sidecar recovery, P3.6) ----------------------

    @_chain_locked
    def restore(self, revisions: list[Revision], cursor: int) -> None:
        """Adopt a persisted chain after its links have been verified.

        Raises when the chain's parent links or cursor are inconsistent;
        callers treat that as a corrupt sidecar.
        """
        parent = BASE_PARENT
        for revision in revisions:
            if revision.parent_id != parent:
                raise EditError(
                    f"revision {revision.id!r} does not chain from "
                    f"{parent!r}; the history is corrupt"
                )
            parent = revision.id
        if not 0 <= cursor <= len(revisions):
            raise EditError(f"cursor {cursor} is outside the restored chain")
        old_revisions = self._revisions
        old_cursor = self._cursor
        self._revisions = []
        self._cursor = 0
        try:
            for revision in revisions:
                current_document = self.document
                for command in revision.commands:
                    if isinstance(command, ReplaceTensorBytes):
                        current = self.read(
                            command.edit.tensor_id,
                            offset=command.edit.offset,
                            length=command.edit.length,
                        )
                        if current != command.edit.before:
                            raise EditError(
                                f"revision {revision.id!r} does not match the "
                                "current artifact bytes"
                            )
                    elif isinstance(command, ResizeTensor):
                        current = self.read(command.tensor_id)
                        if current != command.before_bytes:
                            raise EditError(
                                f"revision {revision.id!r} does not match the "
                                "current tensor before resize"
                            )
                        if current_document is not None:
                            current_document = apply_command(
                                current_document,
                                command,
                            )
                    elif current_document is not None:
                        current_document = apply_command(current_document, command)
                self._revisions.append(revision)
                self._cursor += 1
            self._cursor = cursor
        except BaseException:
            self._revisions = old_revisions
            self._cursor = old_cursor
            raise
