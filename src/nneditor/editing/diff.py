"""Read-only diff preview between revisions (task P3.7).

The preview is computed **entirely from the deltas**: a changed tensor is
described by its edited spans, whose before/after bytes the commands already
carry — no tensor is re-read and nothing is copied, so previewing the diff of
a multi-gigabyte tensor costs bytes proportional to the edit.

Where the element type is a supported packed dtype, the changed spans are
decoded so the preview can show value-level change: how many elements were
touched and the before/after numbers themselves (capped per span — a preview
that prints a million floats is not a preview).
"""

from __future__ import annotations

from dataclasses import dataclass

from nneditor.analysis.statistics import decode_packed, element_width
from nneditor.editing.commands import (
    QuantizeGraph,
    ReplaceTensorBytes,
    ResizeTensor,
    command_summary,
    command_target,
)
from nneditor.editing.cow import ByteSpanEdit
from nneditor.editing.revisions import Revision
from nneditor.ir.core import Document
from nneditor.transformations.schema import (
    TransformationManifest,
    TransformationPreview,
)

__all__ = [
    "DiffPreview",
    "GraphChange",
    "SpanDiff",
    "TensorDiff",
    "TransformationDiff",
    "preview_diff",
]

_PREVIEW_ELEMENTS = 8
"""Decoded values shown per span; the byte-level truth is always complete."""


@dataclass(frozen=True, slots=True)
class SpanDiff:
    """One edited span, byte-exact plus a bounded decoded preview."""

    offset: int
    length: int
    before_values: tuple[float, ...] | None
    after_values: tuple[float, ...] | None
    values_truncated: bool


@dataclass(frozen=True, slots=True)
class TensorDiff:
    """Everything that changed in one tensor."""

    tensor_id: str
    element_type: str
    span_count: int
    bytes_changed: int
    elements_changed: int | None
    spans: tuple[SpanDiff, ...]


@dataclass(frozen=True, slots=True)
class GraphChange:
    """One metadata or topology command in the revision chain."""

    entity_id: str
    summary: str


@dataclass(frozen=True, slots=True)
class TransformationDiff:
    target_id: str
    manifest: TransformationManifest
    preview: TransformationPreview


@dataclass(frozen=True, slots=True)
class DiffPreview:
    """The change between the base revision and the applied chain."""

    base_revision_id: str | None
    target_revision_id: str | None
    commands: tuple[str, ...]
    tensors: tuple[TensorDiff, ...]
    graph_changes: tuple[GraphChange, ...] = ()
    transformations: tuple[TransformationDiff, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.tensors and not self.graph_changes and not self.transformations


def _span_diff(edit: ByteSpanEdit, element_type: str) -> SpanDiff:
    width = element_width(element_type)
    before_values: tuple[float, ...] | None = None
    after_values: tuple[float, ...] | None = None
    truncated = False
    if width is not None and edit.offset % width == 0 and edit.length % width == 0:
        decoded_before = decode_packed(element_type, edit.before)
        decoded_after = decode_packed(element_type, edit.after)
        if decoded_before is not None and decoded_after is not None:
            truncated = len(decoded_before) > _PREVIEW_ELEMENTS
            before_values = tuple(
                float(item) for item in decoded_before[:_PREVIEW_ELEMENTS]
            )
            after_values = tuple(
                float(item) for item in decoded_after[:_PREVIEW_ELEMENTS]
            )
    return SpanDiff(
        offset=edit.offset,
        length=edit.length,
        before_values=before_values,
        after_values=after_values,
        values_truncated=truncated,
    )


def _elements_changed(edits: list[ByteSpanEdit], width: int | None) -> int | None:
    if width is None:
        return None
    touched: set[int] = set()
    for edit in edits:
        first = edit.offset // width
        last = (edit.end - 1) // width
        touched.update(range(first, last + 1))
    return len(touched)


def _bytes_changed(edits: list[ByteSpanEdit]) -> int:
    """Bytes in the union of the edited spans; overlaps count once."""
    total = 0
    span_start: int | None = None
    span_end = 0
    for start, end in sorted((edit.offset, edit.end) for edit in edits):
        if span_start is None or start > span_end:
            if span_start is not None:
                total += span_end - span_start
            span_start, span_end = start, end
        else:
            span_end = max(span_end, end)
    if span_start is not None:
        total += span_end - span_start
    return total


def preview_diff(document: Document, applied: tuple[Revision, ...]) -> DiffPreview:
    """Summarize what the applied revisions change, from deltas alone."""
    per_tensor: dict[str, list[ByteSpanEdit]] = {}
    commands: list[str] = []
    graph_changes: list[GraphChange] = []
    transformations: list[TransformationDiff] = []
    for revision in applied:
        for command in revision.commands:
            commands.append(command_summary(command))
            if isinstance(command, ReplaceTensorBytes):
                per_tensor.setdefault(command.edit.tensor_id, []).append(command.edit)
                if command.transformation is not None and command.preview is not None:
                    transformations.append(
                        TransformationDiff(
                            command.edit.tensor_id,
                            command.transformation,
                            command.preview,
                        )
                    )
            elif isinstance(command, ResizeTensor):
                transformations.append(
                    TransformationDiff(
                        command.tensor_id,
                        command.transformation,
                        command.preview,
                    )
                )
            elif isinstance(command, QuantizeGraph):
                transformations.append(
                    TransformationDiff(
                        command.tensor_id,
                        command.transformation,
                        command.preview,
                    )
                )
                graph_changes.append(
                    GraphChange(command.tensor_id, command_summary(command))
                )
            else:
                graph_changes.append(
                    GraphChange(command_target(command), command_summary(command))
                )

    tensors: list[TensorDiff] = []
    for tensor_id, edits in per_tensor.items():
        reference = document.tensors.get(tensor_id)
        element_type = reference.element_type if reference is not None else "unknown"
        width = element_width(element_type)
        tensors.append(
            TensorDiff(
                tensor_id=tensor_id,
                element_type=element_type,
                span_count=len(edits),
                bytes_changed=_bytes_changed(edits),
                elements_changed=_elements_changed(edits, width),
                spans=tuple(_span_diff(edit, element_type) for edit in edits),
            )
        )
    return DiffPreview(
        base_revision_id=applied[0].parent_id if applied else None,
        target_revision_id=applied[-1].id if applied else None,
        commands=tuple(commands),
        tensors=tuple(tensors),
        graph_changes=tuple(graph_changes),
        transformations=tuple(transformations),
    )
