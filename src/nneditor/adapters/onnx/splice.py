"""Exports an edited ONNX artifact by byte-splicing (task P0.6).

The prototype export path exploits a protobuf property: replacing a tensor's
payload bytes with the *same number* of bytes leaves every offset, length
field, and message boundary in the file untouched. Materializing a revision is
therefore a byte-for-byte copy of the source artifact with the edited spans
overwritten in the copy — no serializer, nothing reinterpreted, and untouched
metadata survives verbatim, which is the strongest possible fidelity for the
supported edit shape. Full re-serialization (needed once edits can change
sizes) is task P4.5.

Safety rules enforced here:

* The source model and its external files are opened read-only; destinations
  must not collide with any source file (rule 1: sources are immutable).
* Every edit's recorded ``before`` bytes must match what the source actually
  holds, so a revision cannot be spliced into an artifact it was not made on.
* Output files are written completely, fsynced, and atomically renamed into
  place; a failed export leaves no partial artifact behind.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nneditor.adapters.onnx.index import ModelIndex, TensorIndex, TensorStorage
from nneditor.editing.cow import ByteSpanEdit

__all__ = ["SpliceExportError", "SpliceReport", "export_with_edits"]


class SpliceExportError(ValueError):
    """Raised when a revision cannot be spliced into a new artifact."""


@dataclass(frozen=True, slots=True)
class SpliceReport:
    """What an export produced, for the caller and the progress log."""

    model_path: Path
    written_files: tuple[Path, ...]
    edited_tensor_ids: tuple[str, ...]
    spliced_bytes: int

    @property
    def is_noop_copy(self) -> bool:
        """Whether the export was a pure copy with no edits applied."""
        return not self.edited_tensor_ids


def _file_offset(tensor: TensorIndex, edit: ByteSpanEdit) -> tuple[Path | None, int]:
    """Where ``edit`` lands: the file holding the span (``None`` for the model
    file itself) and the absolute offset inside it."""
    if tensor.storage is TensorStorage.EMBEDDED_RAW:
        assert tensor.payload is not None
        if edit.end > tensor.payload.length:
            raise SpliceExportError(
                f"edit span [{edit.offset}, {edit.end}) exceeds the "
                f"{tensor.payload.length}-byte payload of tensor {tensor.id!r}"
            )
        return None, tensor.payload.offset + edit.offset
    if tensor.storage is TensorStorage.EXTERNAL:
        reference = tensor.external
        if reference is None or not reference.is_usable:
            raise SpliceExportError(
                f"tensor {tensor.id!r} has an unusable external reference"
            )
        assert reference.resolved_path is not None
        if reference.length is not None and edit.end > reference.length:
            raise SpliceExportError(
                f"edit span [{edit.offset}, {edit.end}) exceeds the "
                f"{reference.length}-byte external payload of tensor {tensor.id!r}"
            )
        return reference.resolved_path, reference.offset + edit.offset
    raise SpliceExportError(
        f"tensor {tensor.id!r} uses {tensor.storage.value} storage, which "
        "byte splicing cannot edit; this export path supports embedded raw "
        "and external tensors"
    )


def _atomic_write(destination: Path, content: bytes) -> None:
    partial = destination.with_name(destination.name + ".partial")
    # Cleanup is scoped to after the exclusive create succeeds: if the open
    # itself fails, the .partial belongs to someone else and must survive.
    handle = open(partial, "xb")
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def export_with_edits(
    index: ModelIndex,
    edits: Iterable[ByteSpanEdit],
    destination: Path | str,
) -> SpliceReport:
    """Write a new artifact equal to the source plus ``edits``.

    External payload files referenced by edited *or* untouched tensors are
    copied next to ``destination`` under their original relative locations, so
    the exported model is complete and self-contained in its directory.
    """
    model_destination = Path(destination)
    if model_destination.exists():
        raise SpliceExportError(
            f"destination {model_destination} already exists; exports never overwrite"
        )
    source_files = {index.path.resolve(), *(p.resolve() for p in index.external_files)}
    # External locations were resolved during indexing, so the comparison base
    # must be resolved too; an index opened through a relative model path would
    # otherwise fail relative_to spuriously.
    source_root = index.path.resolve().parent

    # Plan every splice before writing anything: (source file, planned copy).
    edit_list = list(edits)
    plans: dict[Path, bytearray] = {}
    edited: dict[str, None] = {}
    spliced_bytes = 0
    output_for: dict[Path, Path] = {index.path.resolve(): model_destination}
    for external in index.external_files:
        try:
            relative = external.resolve().relative_to(source_root)
        except ValueError:
            raise SpliceExportError(
                f"external file {external} lies outside the model directory; "
                "its relative location cannot be preserved next to the export"
            ) from None
        output_for[external.resolve()] = model_destination.parent / relative
    claimed: dict[Path, Path] = {}
    for source, target in output_for.items():
        resolved_target = target.resolve()
        if resolved_target in source_files:
            raise SpliceExportError(
                f"export would overwrite source file {target}; sources are immutable"
            )
        previous = claimed.get(resolved_target)
        if previous is not None:
            raise SpliceExportError(
                f"export would write {target} twice: {previous.name} and "
                f"{source.name} both map to it; rename the destination so it "
                "does not collide with an external data file"
            )
        claimed[resolved_target] = source
    for edit in edit_list:
        try:
            tensor = index.tensor(edit.tensor_id)
        except KeyError:
            raise SpliceExportError(
                f"edit references unknown tensor {edit.tensor_id!r}"
            ) from None
        holder, offset = _file_offset(tensor, edit)
        source = index.path if holder is None else holder
        resolved = source.resolve()
        if resolved not in plans:
            plans[resolved] = bytearray(source.read_bytes())
        buffer = plans[resolved]
        if buffer[offset : offset + edit.length] != edit.before:
            raise SpliceExportError(
                f"edit to tensor {edit.tensor_id!r} does not match the source "
                "artifact: the recorded before-bytes differ, so this revision "
                "belongs to a different artifact or overlaps a prior edit "
                "incorrectly"
            )
        buffer[offset : offset + edit.length] = edit.after
        edited.setdefault(edit.tensor_id, None)
        spliced_bytes += edit.length

    # Untouched files are copied verbatim.
    for resolved in output_for:
        if resolved not in plans:
            plans[resolved] = bytearray(resolved.read_bytes())

    written: list[Path] = []
    try:
        for resolved, target in output_for.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, bytes(plans[resolved]))
            written.append(target)
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
        raise
    return SpliceReport(
        model_path=model_destination,
        written_files=tuple(written),
        edited_tensor_ids=tuple(edited),
        spliced_bytes=spliced_bytes,
    )
