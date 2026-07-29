"""Pure presentation logic for the shell (task P1.8).

Everything the shell displays is computed here from IR values, as plain data
the UI merely renders. Keeping this layer free of Flet imports means the
shell's information content is unit-testable headlessly, and a renderer or
toolkit swap cannot change what the user is told.
"""

from __future__ import annotations

from dataclasses import dataclass

from nneditor.analysis.statistics import TensorStatistics, decode_packed
from nneditor.diagnostics import describe
from nneditor.ir.core import Attribute, Document, Storage, Value
from nneditor.transformations.engine import (
    GraphQuantizationRequest,
    LogicalPruningRequest,
    StructuredPruningRequest,
    TransformationRequest,
    WeightQuantizationRequest,
)
from nneditor.transformations.schema import Granularity, PruningMode

__all__ = [
    "GraphEntry",
    "capability_lines",
    "compact_bytes",
    "diagnostic_lines",
    "graph_entries",
    "histogram_rows",
    "humanize_identifier",
    "model_summary",
    "node_details",
    "node_tensor_ids",
    "parse_shape_overrides",
    "preview_values_text",
    "statistics_lines",
    "tensor_lines",
    "transformation_request",
]


@dataclass(frozen=True, slots=True)
class GraphEntry:
    """One row of the graph panel."""

    graph_id: str
    label: str
    depth: int
    node_count: int


def humanize_identifier(value: str) -> str:
    """Turn schema identifiers into compact product-facing labels."""
    acronyms = {
        "fx": "FX",
        "jax": "JAX",
        "onnx": "ONNX",
        "pytorch": "PyTorch",
        "stablehlo": "StableHLO",
    }
    return " ".join(acronyms.get(part, part.capitalize()) for part in value.split("_"))


def compact_bytes(byte_size: int) -> str:
    """Render a byte count for overview metrics."""
    size = float(byte_size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size):,} {unit}"
            rendered = f"{size:.1f}".removesuffix(".0")
            return f"{rendered} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def parse_shape_overrides(text: str) -> dict[str, tuple[int, ...]]:
    """Parse ``name=2x3; other=1,4`` without guessing missing extents."""
    if not text.strip():
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for item in text.split(";"):
        if "=" not in item:
            raise ValueError(
                "symbolic input shapes use name=extentxextent, separated by ';'"
            )
        name, raw_shape = (part.strip() for part in item.split("=", 1))
        if not name or name in result:
            raise ValueError("symbolic input shape names must be non-empty and unique")
        parts = [
            part.strip() for part in raw_shape.lower().replace(",", "x").split("x")
        ]
        try:
            shape = tuple(int(part) for part in parts if part)
        except ValueError:
            raise ValueError(
                f"shape for input {name!r} contains a non-integer"
            ) from None
        if (
            not shape
            or len(shape) != len(parts)
            or any(extent <= 0 for extent in shape)
        ):
            raise ValueError(
                f"shape for input {name!r} needs positive explicit extents"
            )
        result[name] = shape
    return result


def transformation_request(
    *,
    kind: str,
    graph_id: str,
    node_id: str,
    tensor_id: str | None,
    granularity_value: str,
    axis_value: str,
    parameter: str,
) -> TransformationRequest:
    """Parse transformation form values without depending on Flet controls."""
    if kind == "structured-pruning":
        try:
            keep = tuple(
                int(item.strip()) for item in parameter.split(",") if item.strip()
            )
        except ValueError as error:
            raise ValueError(
                "kept channels must be comma-separated integers"
            ) from error
        return StructuredPruningRequest(graph_id, node_id, keep)
    if tensor_id is None:
        raise ValueError("the selected operator has no initializer input")

    granularity = Granularity(granularity_value)
    axis = None if granularity is Granularity.PER_TENSOR else int(axis_value)
    if kind == "weight-quantization":
        return WeightQuantizationRequest(
            tensor_id,
            granularity=granularity,
            axis=axis,
        )
    if kind == "graph-quantization":
        return GraphQuantizationRequest(
            graph_id,
            tensor_id,
            granularity=granularity,
            axis=axis,
        )
    if kind == "threshold-pruning":
        return LogicalPruningRequest(
            tensor_id,
            PruningMode.THRESHOLD,
            threshold=float(parameter),
        )
    if kind == "mask-pruning":
        entries = tuple(item.strip().lower() for item in parameter.split(","))
        allowed = {"0", "1", "false", "true", "drop", "keep"}
        if not entries or any(item not in allowed for item in entries):
            raise ValueError(
                "mask entries must be comma-separated 1/0, true/false, "
                "or keep/drop values"
            )
        return LogicalPruningRequest(
            tensor_id,
            PruningMode.MASK,
            mask=tuple(item in {"1", "true", "keep"} for item in entries),
        )
    if kind != "nm-pruning":
        raise ValueError(f"unknown transformation kind {kind!r}")
    return LogicalPruningRequest(tensor_id, PruningMode.NM, n=2, m=4)


def graph_entries(document: Document) -> tuple[GraphEntry, ...]:
    """Graphs as an indented tree: entry graph first, children beneath their
    owners, function bodies last."""
    entries: list[GraphEntry] = []
    children: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for graph in document.graphs.values():
        if graph.parent_node is not None:
            for owner in document.graphs.values():
                if any(node.id == graph.parent_node for node in owner.nodes):
                    children.setdefault(owner.id, []).append(graph.id)
                    owners[graph.id] = owner.id
                    break

    def walk(graph_id: str, depth: int) -> None:
        graph = document.graphs[graph_id]
        label = graph.name or graph_id
        entries.append(GraphEntry(graph_id, label, depth, len(graph.nodes)))
        for child in children.get(graph_id, ()):
            walk(child, depth + 1)

    walk(document.entry_graph, 0)
    for graph in document.graphs.values():
        if graph.id not in {entry.graph_id for entry in entries}:
            walk(graph.id, 0)
    return tuple(entries)


def model_summary(document: Document) -> tuple[tuple[str, str], ...]:
    """Key/value lines describing the opened artifact."""
    total_nodes = sum(len(graph.nodes) for graph in document.graphs.values())
    lines = [
        ("Artifact", document.source.path),
        ("Kind", document.artifact_kind.value),
        ("Content hash", document.source.content_hash),
        ("Size", f"{document.source.byte_size:,} bytes"),
        ("Graphs", str(len(document.graphs))),
        ("Nodes", str(total_nodes)),
        ("Tensors", str(len(document.tensors))),
    ]
    model_extension = dict(document.extensions).get("x-onnx.model")
    if isinstance(model_extension, dict):
        producer = f"{model_extension.get('producer_name', '')} " + str(
            model_extension.get("producer_version", "")
        )
        if producer.strip():
            lines.append(("Producer", producer.strip()))
    return tuple(lines)


def capability_lines(document: Document) -> tuple[tuple[str, str, str], ...]:
    """``(capability, availability, reason)`` rows, in registry order."""
    return tuple(
        (
            status.capability.value,
            status.availability.value,
            status.reason,
        )
        for status in document.capabilities.values()
    )


def diagnostic_lines(document: Document) -> tuple[tuple[str, str, str], ...]:
    """``(severity, title, message)`` rows for the diagnostics panel."""
    return tuple(
        (
            item.severity.value,
            describe(item.code).title,
            item.message + (f" [{item.target}]" if item.target else ""),
        )
        for item in document.diagnostics
    )


def _shape_text(value: Value) -> str:
    if value.shape is None:
        return "?"
    rendered = ", ".join("?" if dim is None else str(dim) for dim in value.shape)
    return f"[{rendered}]"


def _attribute_text(attribute: Attribute) -> str:
    value = attribute.value
    if isinstance(value, tuple):
        inner = ", ".join(str(item) for item in value)
        return f"({inner})"
    return str(value)


def node_details(
    document: Document, graph_id: str, node_id: str
) -> tuple[tuple[str, str], ...]:
    """Key/value lines for one selected node."""
    graph = document.graphs[graph_id]
    node = graph.node(node_id)
    lines: list[tuple[str, str]] = [
        ("Operator", node.qualified_op_type),
        ("Node id", node.id),
    ]
    if node.source_name:
        lines.append(("Name", node.source_name))
    for position, value_id in enumerate(node.inputs):
        value = graph.value(value_id)
        label = value.name or "(omitted)"
        dtype = value.element_type or "?"
        lines.append((f"Input {position}", f"{label}: {dtype} {_shape_text(value)}"))
    for position, value_id in enumerate(node.outputs):
        value = graph.value(value_id)
        dtype = value.element_type or "?"
        lines.append(
            (f"Output {position}", f"{value.name}: {dtype} {_shape_text(value)}")
        )
    for attribute in node.attributes:
        lines.append((f"Attr {attribute.name}", _attribute_text(attribute)))
    for subgraph_id in node.subgraphs:
        lines.append(("Subgraph", subgraph_id))
    for note in document.notes_for(node_id):
        lines.append(
            (f"{note.capability.value} {note.availability.value}", note.reason)
        )
    # Initializer-backed inputs get their storage summarized so the user sees
    # weight locations without a single tensor byte being read.
    for value_id in node.inputs:
        if not graph.has_value(value_id):
            continue
        value = graph.value(value_id)
        for tensor_id in graph.initializers:
            tensor = document.tensors[tensor_id]
            if tensor.id.endswith(f"#{value.name}") and value.name:
                where = {
                    Storage.EMBEDDED_RAW: "embedded",
                    Storage.EMBEDDED_TYPED: "embedded (typed)",
                    Storage.EXTERNAL: "external",
                    Storage.GENERATED: "generated revision data",
                    Storage.ABSENT: "absent",
                }[tensor.storage]
                lines.append(
                    (
                        f"Weight {value.name}",
                        f"{tensor.element_type} {list(tensor.dims)} ({where})",
                    )
                )
    return tuple(lines)


def node_tensor_ids(document: Document, graph_id: str, node_id: str) -> tuple[str, ...]:
    """Initializer tensors feeding one node, in input-port order."""
    graph = document.graphs[graph_id]
    node = graph.node(node_id)
    found: list[str] = []
    for value_id in node.inputs:
        if not graph.has_value(value_id):
            continue
        value = graph.value(value_id)
        if not value.name:
            continue
        for tensor_id in graph.initializers:
            if tensor_id.endswith(f"#{value.name}") and tensor_id not in found:
                found.append(tensor_id)
    return tuple(found)


def tensor_lines(
    document: Document,
    tensor_id: str,
    *,
    materialization: str,
    byte_length: int | None,
) -> tuple[tuple[str, str], ...]:
    """The inspector's tensor summary; never reads a byte itself.

    ``materialization`` is the store's disclosure (range / full-parse /
    unavailable); the inspector must show the cost of inspection *before* the
    user pays it.
    """
    tensor = document.tensors[tensor_id]
    storage_names = {
        Storage.EMBEDDED_RAW: "embedded raw bytes",
        Storage.EMBEDDED_TYPED: "embedded typed fields",
        Storage.EXTERNAL: "external file",
        Storage.GENERATED: "generated revision data",
        Storage.ABSENT: "absent",
    }
    lines: list[tuple[str, str]] = [
        ("Tensor", tensor_id),
        ("Dtype", tensor.element_type),
        ("Shape", str(list(tensor.dims))),
        ("Elements", f"{tensor.element_count:,}"),
        ("Storage", storage_names[tensor.storage]),
        ("Access", materialization),
    ]
    if byte_length is not None:
        lines.append(("Bytes", f"{byte_length:,}"))
        lines.append(("Est. memory", f"{byte_length / (1024 * 1024):.2f} MiB"))
    if tensor.external is not None:
        lines.append(("Location", tensor.external.location))
    # Quantization and layout metadata are not modelled until Phase 5; saying
    # so beats a silently missing row.
    lines.append(("Quantization", "none recorded"))
    for entry in document.provenance:
        if entry.operation == "import":
            lines.append(("Provenance", f"import of {entry.source_artifact}"))
            break
    return tuple(lines)


def preview_values_text(element_type: str, raw: bytes, *, limit: int = 8) -> str:
    """A short human-readable slice preview from already-read bytes."""
    decoded = decode_packed(element_type, raw)
    if decoded is None:
        return f"{len(raw)} raw byte(s); values not decodable for this dtype"
    shown = [f"{float(value):.6g}" for value in list(decoded)[:limit]]
    suffix = ", …" if len(decoded) > limit else ""
    return "[" + ", ".join(shown) + suffix + "]"


def statistics_lines(stats: TensorStatistics) -> tuple[tuple[str, str], ...]:
    """The computed-statistics block of the inspector."""

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6g}"

    return (
        ("Min", number(stats.minimum)),
        ("Max", number(stats.maximum)),
        ("Mean", number(stats.mean)),
        ("Std", number(stats.std)),
        ("Sparsity", f"{stats.sparsity:.2%} ({stats.zero_count:,} zero)"),
        ("Non-finite", f"{stats.nan_count:,} NaN, {stats.inf_count:,} Inf"),
    )


def histogram_rows(
    stats: TensorStatistics, *, max_rows: int = 16
) -> tuple[tuple[str, int, float], ...]:
    """``(range label, count, fraction-of-peak)`` rows for a bar display.

    Adjacent bins are merged when there are more bins than rows, so the
    on-screen histogram never exceeds ``max_rows`` regardless of how finely
    the statistics were computed.
    """
    counts = list(stats.histogram_counts)
    edges = list(stats.histogram_edges)
    if not counts:
        return ()
    while len(counts) > max_rows:
        merged_counts: list[int] = []
        merged_edges: list[float] = [edges[0]]
        for index in range(0, len(counts) - 1, 2):
            merged_counts.append(counts[index] + counts[index + 1])
            merged_edges.append(edges[index + 2])
        if len(counts) % 2:
            merged_counts.append(counts[-1])
            merged_edges.append(edges[-1])
        counts = merged_counts
        edges = merged_edges
    peak = max(counts)
    rows = []
    for index, count in enumerate(counts):
        label = f"{edges[index]:.4g} … {edges[index + 1]:.4g}"
        rows.append((label, count, count / peak if peak else 0.0))
    return tuple(rows)
