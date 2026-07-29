"""Trace comparison metrics and renderer-neutral graph overlays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from nneditor.rendering.scene import Scene, ScenePatch
from nneditor.tracing.contracts import TraceResult
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


def _array(store: ActivationStore, result: TraceResult, value_id: str) -> np.ndarray:
    record = result.record(value_id)
    return np.frombuffer(
        store.read(result.id, value_id),
        dtype=np.dtype(record.numpy_dtype),
    ).astype(np.float64)


def compare_traces(
    store: ActivationStore,
    left: TraceResult,
    right: TraceResult,
) -> TraceComparison:
    """Compare readable common captures from the exact same input spec."""
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
        left_record = left_records[value_id]
        right_record = right_records[value_id]
        if not left_record.readable or not right_record.readable:
            diagnostics.append(f"value {value_id!r} is not readable in both traces")
            continue
        left_values = _array(store, left, value_id)
        right_values = _array(store, right, value_id)
        count = min(left_values.size, right_values.size)
        if count == 0:
            diagnostics.append(f"value {value_id!r} has no captured elements")
            continue
        partial = (
            left.partial
            or right.partial
            or left_values.size != right_values.size
            or left_record.shape != right_record.shape
        )
        if left_values.size != right_values.size:
            diagnostics.append(
                f"value {value_id!r} compares only the common captured prefix"
            )
        left_values = left_values[:count]
        right_values = right_values[:count]
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        nonfinite_equal = (~finite) & (
            (np.isnan(left_values) & np.isnan(right_values))
            | (left_values == right_values)
        )
        nonfinite_mismatch = (~finite) & ~nonfinite_equal
        finite_left = left_values[finite]
        finite_right = right_values[finite]
        difference = np.abs(finite_left - finite_right)
        denominator = np.maximum(np.abs(finite_left), np.finfo(np.float64).eps)
        maximum_absolute = float(np.max(difference)) if difference.size else 0.0
        maximum_relative = (
            float(np.max(difference / denominator)) if difference.size else 0.0
        )
        if np.any(nonfinite_mismatch):
            maximum_absolute = math.inf
            maximum_relative = math.inf
        if np.any(~finite):
            partial = True
            diagnostics.append(
                f"value {value_id!r} contains non-finite pairs; finite pairs "
                "were used for cosine and mismatched non-finite values count "
                "as infinite error"
            )
        left_norm = float(np.linalg.norm(finite_left))
        right_norm = float(np.linalg.norm(finite_right))
        cosine = (
            None
            if left_norm == 0.0 or right_norm == 0.0
            else max(
                -1.0,
                min(
                    1.0,
                    float(np.dot(finite_left, finite_right) / (left_norm * right_norm)),
                ),
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


def trace_scene_patch(graph_slice: _GraphSlice, result: TraceResult) -> ScenePatch:
    """Mark every visible glyph containing a node with a captured value."""
    captured_nodes = frozenset(
        record.node_id
        for record in result.records
        if record.node_id and record.readable
    )
    changed = []
    suffix = "activation partial" if result.partial else "activation captured"
    for glyph in graph_slice.scene.nodes:
        if graph_slice.members_by_glyph.get(glyph.id, frozenset()) & captured_nodes:
            changed.append(
                replace(glyph, type_label=f"{glyph.display_type} • {suffix}")
            )
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
