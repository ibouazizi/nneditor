"""Trace comparison metrics and renderer-neutral graph overlays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np

from nneditor.cancellation import CancellationToken
from nneditor.rendering.scene import Scene, ScenePatch
from nneditor.tracing.contracts import CaptureStatus, TraceResult
from nneditor.tracing.store import ActivationStore

__all__ = [
    "ActivationError",
    "NodeError",
    "TraceComparison",
    "compare_traces",
    "comparison_scene_patch",
    "trace_scene_patch",
]


class _GraphSlice(Protocol):
    @property
    def scene(self) -> Scene: ...

    @property
    def members_by_glyph(self) -> Mapping[str, frozenset[str]]: ...


@dataclass(frozen=True, slots=True)
class ActivationError:
    value_id: str
    node_id: str | None
    max_absolute: float
    max_relative: float
    cosine_similarity: float | None
    compared_elements: int
    partial: bool


@dataclass(frozen=True, slots=True)
class NodeError:
    node_id: str
    max_absolute: float
    max_relative: float
    minimum_cosine_similarity: float | None
    partial: bool


@dataclass(frozen=True, slots=True)
class TraceComparison:
    left_trace_id: str
    right_trace_id: str
    values: tuple[ActivationError, ...]
    nodes: tuple[NodeError, ...]
    diagnostics: tuple[str, ...] = ()

    def node(self, node_id: str) -> NodeError:
        for metric in self.nodes:
            if metric.node_id == node_id:
                return metric
        raise KeyError(node_id)


# Bounds the working set per value to roughly 64Ki elements per side (about
# 1 MiB of float64 each) instead of materializing whole payloads, so comparing
# two traces of a large model stays within a predictable memory ceiling.
_CHUNK_ELEMENTS = 1 << 16


@dataclass(slots=True)
class _Accumulator:
    """Running comparison state over the common captured element prefix."""

    max_absolute: float = 0.0
    max_relative: float = 0.0
    dot: float = 0.0
    left_square: float = 0.0
    right_square: float = 0.0
    nonfinite_pairs: bool = False
    nonfinite_mismatch: bool = False


def _accumulate(
    accumulator: _Accumulator,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> None:
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    nonfinite_equal = (~finite) & (
        (np.isnan(left_values) & np.isnan(right_values)) | (left_values == right_values)
    )
    if bool(np.any(~finite)):
        accumulator.nonfinite_pairs = True
    if bool(np.any((~finite) & ~nonfinite_equal)):
        accumulator.nonfinite_mismatch = True
    finite_left = left_values[finite]
    finite_right = right_values[finite]
    if not finite_left.size:
        return
    difference = np.abs(finite_left - finite_right)
    denominator = np.maximum(np.abs(finite_left), np.finfo(np.float64).eps)
    accumulator.max_absolute = max(accumulator.max_absolute, float(np.max(difference)))
    accumulator.max_relative = max(
        accumulator.max_relative, float(np.max(difference / denominator))
    )
    accumulator.dot += float(np.dot(finite_left, finite_right))
    accumulator.left_square += float(np.dot(finite_left, finite_left))
    accumulator.right_square += float(np.dot(finite_right, finite_right))


def _stream_comparison(
    store: ActivationStore,
    left: TraceResult,
    right: TraceResult,
    value_id: str,
    count: int,
    left_width: int,
    right_width: int,
    token: CancellationToken | None,
) -> _Accumulator:
    accumulator = _Accumulator()
    left_dtype = np.dtype(left.record(value_id).numpy_dtype)
    right_dtype = np.dtype(right.record(value_id).numpy_dtype)
    for start in range(0, count, _CHUNK_ELEMENTS):
        if token is not None:
            token.raise_if_cancelled()
        elements = min(_CHUNK_ELEMENTS, count - start)
        left_values = np.frombuffer(
            store.read(
                left.id,
                value_id,
                offset=start * left_width,
                length=elements * left_width,
                token=token,
            ),
            dtype=left_dtype,
        ).astype(np.float64)
        right_values = np.frombuffer(
            store.read(
                right.id,
                value_id,
                offset=start * right_width,
                length=elements * right_width,
                token=token,
            ),
            dtype=right_dtype,
        ).astype(np.float64)
        _accumulate(accumulator, left_values, right_values)
    return accumulator


def compare_traces(
    store: ActivationStore,
    left: TraceResult,
    right: TraceResult,
    *,
    token: CancellationToken | None = None,
) -> TraceComparison:
    """Compare readable common captures from the exact same input spec.

    Captures stream in bounded chunks and the token is honoured between every
    chunk, so comparing large traces neither materializes whole payloads nor
    becomes uninterruptible.

    `max_relative` divides by ``max(|left|, float64 eps)`` — a pure
    element-wise relative error. The export smoke comparison in
    `adapters.onnx.numerical` floors the same ratio by its numerical tolerance
    instead, so the two metrics deliberately differ near zero.
    """
    if left.key.artifact_hash != right.key.artifact_hash:
        raise ValueError("trace comparison requires the same source artifact")
    if left.key.input_specification_hash != right.key.input_specification_hash:
        raise ValueError("trace comparison requires the same input specification")
    left_records = {record.value_id: record for record in left.records}
    right_records = {record.value_id: record for record in right.records}
    common = sorted(left_records.keys() & right_records.keys())
    diagnostics: list[str] = []
    metrics: list[ActivationError] = []
    for value_id in common:
        if token is not None:
            token.raise_if_cancelled()
        left_record = left_records[value_id]
        right_record = right_records[value_id]
        if not left_record.readable or not right_record.readable:
            diagnostics.append(f"value {value_id!r} is not readable in both traces")
            continue
        left_width = np.dtype(left_record.numpy_dtype).itemsize
        right_width = np.dtype(right_record.numpy_dtype).itemsize
        left_size = left_record.stored_byte_length // left_width
        right_size = right_record.stored_byte_length // right_width
        count = min(left_size, right_size)
        if count == 0:
            diagnostics.append(f"value {value_id!r} has no captured elements")
            continue
        partial = (
            left.partial
            or right.partial
            or left_size != right_size
            or left_record.shape != right_record.shape
        )
        if left_size != right_size:
            diagnostics.append(
                f"value {value_id!r} compares only the common captured prefix"
            )
        accumulator = _stream_comparison(
            store,
            left,
            right,
            value_id,
            count,
            left_width,
            right_width,
            token,
        )
        maximum_absolute = accumulator.max_absolute
        maximum_relative = accumulator.max_relative
        if accumulator.nonfinite_mismatch:
            maximum_absolute = math.inf
            maximum_relative = math.inf
        if accumulator.nonfinite_pairs:
            partial = True
            diagnostics.append(
                f"value {value_id!r} contains non-finite pairs; finite pairs "
                "were used for cosine and mismatched non-finite values count "
                "as infinite error"
            )
        left_norm = math.sqrt(accumulator.left_square)
        right_norm = math.sqrt(accumulator.right_square)
        cosine = (
            None
            if left_norm == 0.0 or right_norm == 0.0
            else max(
                -1.0,
                min(1.0, accumulator.dot / (left_norm * right_norm)),
            )
        )
        metrics.append(
            ActivationError(
                value_id=value_id,
                node_id=left_record.node_id or right_record.node_id,
                max_absolute=maximum_absolute,
                max_relative=maximum_relative,
                cosine_similarity=cosine,
                compared_elements=count,
                partial=partial,
            )
        )

    by_node: dict[str, list[ActivationError]] = {}
    for metric in metrics:
        if metric.node_id is not None:
            by_node.setdefault(metric.node_id, []).append(metric)
    nodes = tuple(
        NodeError(
            node_id=node_id,
            max_absolute=max(item.max_absolute for item in items),
            max_relative=max(item.max_relative for item in items),
            minimum_cosine_similarity=(
                min(cosines)
                if (
                    cosines := [
                        item.cosine_similarity
                        for item in items
                        if item.cosine_similarity is not None
                    ]
                )
                else None
            ),
            partial=any(item.partial for item in items),
        )
        for node_id, items in sorted(by_node.items())
    )
    missing = sorted(left_records.keys() ^ right_records.keys())
    if missing:
        diagnostics.append(
            f"{len(missing)} value(s) exist in only one trace and were not compared"
        )
    return TraceComparison(
        left.id,
        right.id,
        tuple(metrics),
        nodes,
        tuple(diagnostics),
    )


# The short status marker a traced glyph carries after its own identity.
# "partial" is reserved for genuinely missing data; a preview trace whose
# every value kept a readable prefix is deliberate breadth, not a failure.
_CAPTURE_SUFFIXES: Final[Mapping[CaptureStatus, str]] = {
    CaptureStatus.PARTIAL: "activation partial",
    CaptureStatus.PREVIEW: "activation preview",
    CaptureStatus.COMPLETE: "activation captured",
}


def trace_scene_patch(graph_slice: _GraphSlice, result: TraceResult) -> ScenePatch:
    """Mark every visible glyph containing a node with a captured value.

    The capture status is additive metadata: the glyph's own identity — its
    semantic type, or its label when it has no type — always leads, so a
    renderer that truncates long labels drops the status marker, never the
    name, and no glyph ever reads as a bare status string.
    """
    captured_nodes = frozenset(
        record.node_id
        for record in result.records
        if record.node_id and record.readable
    )
    changed = []
    suffix = _CAPTURE_SUFFIXES[result.capture_status]
    for glyph in graph_slice.scene.nodes:
        if graph_slice.members_by_glyph.get(glyph.id, frozenset()) & captured_nodes:
            identity = glyph.display_type.strip() or glyph.label
            changed.append(replace(glyph, type_label=f"{identity} • {suffix}"))
    return ScenePatch(upsert_nodes=tuple(changed))


def comparison_scene_patch(
    graph_slice: _GraphSlice,
    comparison: TraceComparison,
) -> ScenePatch:
    """Add concise error summaries to every visible semantic representative."""
    by_node = {metric.node_id: metric for metric in comparison.nodes}
    changed = []
    for glyph in graph_slice.scene.nodes:
        metrics = [
            by_node[node_id]
            for node_id in graph_slice.members_by_glyph.get(glyph.id, frozenset())
            if node_id in by_node
        ]
        if not metrics:
            continue
        maximum = max(metric.max_absolute for metric in metrics)
        marker = "partial " if any(metric.partial for metric in metrics) else ""
        display = "∞" if math.isinf(maximum) else f"{maximum:.3g}"
        changed.append(
            replace(
                glyph,
                type_label=f"{glyph.display_type} • {marker}Δmax {display}",
            )
        )
    return ScenePatch(upsert_nodes=tuple(changed))
