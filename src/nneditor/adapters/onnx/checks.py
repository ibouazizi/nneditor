"""Structural consistency checks run while an ONNX model is imported.

These checks answer a question the format itself cannot: is the artifact
*self-consistent*? An exporter can emit a model that loads, and even runs,
while carrying annotations that contradict the data they describe. ONNX's own
type inference then refuses the model, which blocks export and tracing with an
error that names neither the value nor the file — the failure surfaces long
after import, where it is hardest to interpret.

Every check here works from the :class:`ModelIndex` alone. That keeps them
free: the index already holds declared types, shapes, and tensor locations, so
no check opens a payload, materializes a tensor, or needs the official ONNX
bindings. They therefore run on every import, including of multi-gigabyte
models, without weakening the lazy-loading guarantee.

Findings are reported, never repaired. A contradiction is evidence about the
producing exporter, and silently rewriting it would hide a real defect and
change the artifact's meaning.
"""

from __future__ import annotations

from nneditor.adapters.onnx.index import GraphIndex, ModelIndex, ValueIndex
from nneditor.diagnostics import Diagnostic, Severity

__all__ = ["check_model_index"]


def _declared_values(graph: GraphIndex) -> dict[str, ValueIndex]:
    """Every name the graph gives an explicit type or shape annotation."""
    declared: dict[str, ValueIndex] = {}
    for entry in (*graph.inputs, *graph.outputs, *graph.value_info):
        declared.setdefault(entry.name, entry)
    return declared


def _annotation_conflicts(graph: GraphIndex) -> list[Diagnostic]:
    """Annotations that contradict the initializer they describe.

    An initializer carries its own element type and dimensions, so a
    ``value_info`` entry for the same name is redundant. When the two disagree
    the model is not merely untidy: ONNX infers the initializer's type, finds
    the conflicting annotation, and raises a type-inference error.
    """
    findings: list[Diagnostic] = []
    declared = _declared_values(graph)
    for tensor in graph.initializers:
        entry = declared.get(tensor.name)
        if entry is None:
            continue
        if (
            entry.element_type is not None
            and entry.element_type.code != tensor.element_type.code
        ):
            findings.append(
                Diagnostic(
                    "onnx.type-annotation-conflict",
                    Severity.WARNING,
                    f"Value {tensor.name!r} is annotated as "
                    f"{entry.element_type.name} but its initializer holds "
                    f"{tensor.element_type.name}. ONNX type inference rejects "
                    "this model, so export and tracing cannot run until the "
                    "annotation is removed or corrected.",
                    tensor.id,
                )
            )
        if entry.shape is not None and len(entry.shape) != len(tensor.dims):
            findings.append(
                Diagnostic(
                    "onnx.shape-annotation-conflict",
                    Severity.WARNING,
                    f"Value {tensor.name!r} is annotated with rank "
                    f"{len(entry.shape)} but its initializer has rank "
                    f"{len(tensor.dims)}. Shape inference rejects this model.",
                    tensor.id,
                )
            )
            continue
        if entry.shape is None:
            continue
        for position, (annotated, actual) in enumerate(
            zip(entry.shape, tensor.dims, strict=False)
        ):
            if isinstance(annotated, int) and annotated != actual:
                findings.append(
                    Diagnostic(
                        "onnx.shape-annotation-conflict",
                        Severity.WARNING,
                        f"Value {tensor.name!r} is annotated with extent "
                        f"{annotated} at dimension {position} but its "
                        f"initializer has {actual}. Shape inference rejects "
                        "this model.",
                        tensor.id,
                    )
                )
                break
    return findings


def _duplicate_initializers(graph: GraphIndex) -> list[Diagnostic]:
    """Two initializers competing for one name in the same graph."""
    findings: list[Diagnostic] = []
    seen: set[str] = set()
    for tensor in graph.initializers:
        if tensor.name in seen:
            findings.append(
                Diagnostic(
                    "onnx.duplicate-initializer",
                    Severity.WARNING,
                    f"Initializer {tensor.name!r} is declared more than once "
                    "in this graph; which tensor a consumer receives depends "
                    "on the runtime's resolution order.",
                    tensor.id,
                )
            )
        seen.add(tensor.name)
    return findings


def _visible_names(index: ModelIndex, graph: GraphIndex) -> set[str]:
    """Names a graph can reference, including its enclosing scopes.

    ONNX subgraphs read values from the graphs that contain them, so a name
    with no local producer is only dangling when no ancestor supplies it
    either. Resolving that correctly is what keeps control-flow bodies from
    being reported as broken.
    """
    owner_of_node = {
        node.id: candidate.id
        for candidate in index.graphs.values()
        for node in candidate.nodes
    }
    names: set[str] = set()
    current: GraphIndex | None = graph
    seen_graphs: set[str] = set()
    while current is not None and current.id not in seen_graphs:
        seen_graphs.add(current.id)
        names.update(entry.name for entry in current.inputs)
        names.update(tensor.name for tensor in current.initializers)
        for node in current.nodes:
            names.update(name for name in node.outputs if name)
        if current.parent_node_id is None:
            break
        parent_graph_id = owner_of_node.get(current.parent_node_id)
        current = (
            index.graphs.get(parent_graph_id) if parent_graph_id is not None else None
        )
    return names


def _unresolved_inputs(index: ModelIndex, graph: GraphIndex) -> list[Diagnostic]:
    """Node inputs that nothing in scope produces."""
    findings: list[Diagnostic] = []
    visible = _visible_names(index, graph)
    for node in graph.nodes:
        for position, name in enumerate(node.inputs):
            # An empty name is ONNX's way of skipping an optional input.
            if not name or name in visible:
                continue
            findings.append(
                Diagnostic(
                    "onnx.unresolved-input",
                    Severity.ERROR,
                    f"Input {position} of node {node.id!r} reads {name!r}, "
                    "which no initializer, graph input, or node output in "
                    "scope produces. The graph is incomplete.",
                    node.id,
                )
            )
    return findings


def check_model_index(index: ModelIndex) -> list[Diagnostic]:
    """Run every structural consistency check over an indexed model."""
    findings: list[Diagnostic] = []
    for graph in index.graphs.values():
        findings.extend(_annotation_conflicts(graph))
        findings.extend(_duplicate_initializers(graph))
        findings.extend(_unresolved_inputs(index, graph))
    return findings
