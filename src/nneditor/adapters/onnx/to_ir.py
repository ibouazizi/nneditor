"""Converts a lazy ONNX index into a canonical IR document (task P1.2).

The :class:`~nneditor.adapters.onnx.index.ModelIndex` already carries the
model's topology and tensor locations; this module reshapes it into the
validated IR of `docs/ir-schema.md`. Two responsibilities live here and
nowhere else:

* **Value synthesis.** ONNX names edges implicitly at node ports; the IR
  declares every value. Ports referencing undeclared names get synthesized
  values, and an omitted optional port (an empty name in ONNX) keeps its
  position through a placeholder value, so operator schema indices stay
  meaningful.
* **Loss awareness.** Model metadata, opset imports, and function signatures
  that the core IR does not interpret ride in ``x-onnx.*`` extensions;
  capability statuses are narrowed from the artifact contract by what the
  import actually found, with per-entity notes carrying the reasons.
"""

from __future__ import annotations

import importlib.metadata
from typing import Final

from nneditor.adapters.onnx.dtypes import ElementType
from nneditor.adapters.onnx.index import (
    GraphIndex,
    ModelIndex,
    NodeIndex,
    TensorIndex,
    TensorStorage,
    ValueIndex,
)
from nneditor.diagnostics import Diagnostic, Severity
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
    contract_for,
)
from nneditor.ir.core import (
    ArtifactRef,
    CapabilityNote,
    Document,
    ExternalRef,
    Graph,
    JsonValue,
    Node,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import missing_value_id, value_id

__all__ = ["index_to_document"]

_STANDARD_DOMAINS: Final = ("", "ai.onnx")

_ELEMENT_NAME_OVERRIDES: Final = {"FLOAT": "float32", "DOUBLE": "float64"}


def _element_name(element_type: ElementType | None) -> str | None:
    if element_type is None:
        return None
    return _ELEMENT_NAME_OVERRIDES.get(element_type.name, element_type.name.lower())


_STORAGE_MAP: Final = {
    TensorStorage.EMBEDDED_RAW: Storage.EMBEDDED_RAW,
    TensorStorage.EMBEDDED_TYPED: Storage.EMBEDDED_TYPED,
    TensorStorage.EXTERNAL: Storage.EXTERNAL,
    TensorStorage.ABSENT: Storage.ABSENT,
}


def _tensor_ref(tensor: TensorIndex) -> TensorRef:
    """Map an indexed tensor to a location-only reference.

    A tensor whose location failed validation is represented as *absent*
    rather than dropped: its identity, dtype, and shape remain addressable,
    and the import diagnostics already explain what is wrong with the bytes.
    """
    storage = _STORAGE_MAP[tensor.storage]
    payload = None
    external = None
    if storage is Storage.EMBEDDED_RAW:
        if tensor.payload is None:
            storage = Storage.ABSENT
        else:
            payload = PayloadRange(tensor.payload.offset, tensor.payload.length)
    elif storage is Storage.EXTERNAL:
        reference = tensor.external
        if reference is None or not reference.location:
            storage = Storage.ABSENT
        else:
            external = ExternalRef(
                location=reference.location,
                offset=reference.offset,
                length=reference.length,
                checksum=reference.checksum,
            )
    typed_span = None
    if storage is Storage.EMBEDDED_TYPED and tensor.message_span is not None:
        typed_span = PayloadRange(
            tensor.message_span.offset, tensor.message_span.length
        )
    return TensorRef(
        id=tensor.id,
        element_type=_element_name(tensor.element_type) or "undefined",
        dims=tensor.dims,
        storage=storage,
        payload=payload,
        external=external,
        typed_span=typed_span,
    )


class _GraphBuilder:
    """Collects declared and synthesized values for one graph."""

    def __init__(self, graph: GraphIndex) -> None:
        self.graph = graph
        self.values: dict[str, Value] = {}
        self.name_to_id: dict[str, str] = {}

    def declare(self, entry: ValueIndex) -> str:
        existing = self.name_to_id.get(entry.name)
        if existing is not None:
            return existing
        value = Value(
            id=entry.id,
            name=entry.name,
            element_type=_element_name(entry.element_type),
            shape=entry.shape,
        )
        self.values[value.id] = value
        self.name_to_id[entry.name] = value.id
        return value.id

    def declare_initializer(self, tensor: TensorIndex) -> str:
        existing = self.name_to_id.get(tensor.name)
        if existing is not None:
            return existing
        new_id = value_id(self.graph.id, tensor.name)
        self.values[new_id] = Value(
            id=new_id,
            name=tensor.name,
            element_type=_element_name(tensor.element_type),
            shape=tensor.dims,
        )
        self.name_to_id[tensor.name] = new_id
        return new_id

    def port(self, node_id: str, name: str, position: int) -> str:
        """Resolve one node port to a value id, synthesizing as needed."""
        if not name:
            placeholder = missing_value_id(node_id, position)
            if placeholder not in self.values:
                self.values[placeholder] = Value(id=placeholder, name="")
            return placeholder
        existing = self.name_to_id.get(name)
        if existing is not None:
            return existing
        new_id = value_id(self.graph.id, name)
        self.values[new_id] = Value(id=new_id, name=name)
        self.name_to_id[name] = new_id
        return new_id


def _convert_node(builder: _GraphBuilder, node: NodeIndex) -> Node:
    inputs = tuple(
        builder.port(node.id, name, position)
        for position, name in enumerate(node.inputs)
    )
    # Placeholder ids for omitted *outputs* are offset by the input count so
    # a node omitting input 1 and output 1 cannot mint the same value id.
    outputs = tuple(
        builder.port(node.id, name, len(node.inputs) + position)
        for position, name in enumerate(node.outputs)
    )
    return Node(
        id=node.id,
        id_stability=node.id_stability,
        op_type=node.op_type,
        domain="" if node.domain in _STANDARD_DOMAINS else node.domain,
        overload=node.overload,
        inputs=inputs,
        outputs=outputs,
        attributes=node.attributes,
        subgraphs=node.subgraph_ids,
        source_name=node.name or None,
    )


def _convert_graph(
    graph: GraphIndex,
    *,
    io_names: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
) -> Graph:
    """Convert one indexed graph.

    ``io_names`` supplies input/output *names* for function bodies, whose
    signatures live on the function record rather than in the graph itself.
    """
    builder = _GraphBuilder(graph)
    inputs = [builder.declare(entry) for entry in graph.inputs]
    outputs = [builder.declare(entry) for entry in graph.outputs]
    for entry in graph.value_info:
        builder.declare(entry)
    initializer_ids = []
    for tensor in graph.initializers:
        builder.declare_initializer(tensor)
        initializer_ids.append(tensor.id)
    nodes = [_convert_node(builder, node) for node in graph.nodes]
    if io_names is not None:
        input_names, output_names = io_names
        inputs = [builder.port("", name, 0) for name in input_names if name]
        outputs = [builder.port("", name, 0) for name in output_names if name]
    return Graph(
        id=graph.id,
        name=graph.name,
        nodes=nodes,
        values=builder.values.values(),
        inputs=inputs,
        outputs=outputs,
        initializers=initializer_ids,
        parent_node=graph.parent_node_id,
    )


def _narrow_capabilities(
    index: ModelIndex,
) -> tuple[tuple[CapabilityStatus, ...], tuple[CapabilityNote, ...]]:
    contract = contract_for(ArtifactKind.ONNX_MODEL)
    statuses: dict[Capability, CapabilityStatus] = {
        status.capability: status for status in contract.statuses
    }
    notes: list[CapabilityNote] = []

    unreadable = [
        tensor
        for tensor in index.iter_tensors()
        if tensor.storage is not TensorStorage.EMBEDDED_TYPED
        and not tensor.is_readable
        and tensor.storage is not TensorStorage.ABSENT
    ]
    for tensor in unreadable:
        notes.append(
            CapabilityNote(
                entity_id=tensor.id,
                capability=Capability.WEIGHTS,
                availability=Availability.UNAVAILABLE,
                reason=tensor.error
                or (
                    tensor.external.error
                    if tensor.external is not None and tensor.external.error
                    else "the tensor's bytes could not be located"
                ),
            )
        )
    if unreadable:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.PARTIAL,
            f"{len(unreadable)} of {index.tensor_count} tensors have no "
            "readable bytes; see the capability notes on each tensor",
        )

    for graph in index.graphs.values():
        for node in graph.nodes:
            if node.domain and node.domain not in _STANDARD_DOMAINS:
                notes.append(
                    CapabilityNote(
                        entity_id=node.id,
                        capability=Capability.EDITING,
                        availability=Availability.UNAVAILABLE,
                        reason=f"operator {node.qualified_op_type} comes from "
                        f"custom domain {node.domain!r}, whose schema is "
                        "unknown, so edits cannot be validated",
                    )
                )
    return tuple(statuses.values()), tuple(notes)


def _model_extension(index: ModelIndex) -> dict[str, JsonValue]:
    return {
        "model_hash": index.model_hash,
        "external_hashes": [
            {"location": location, "content_hash": content_hash}
            for location, content_hash in index.external_hashes
        ],
        "ir_version": index.ir_version,
        "producer_name": index.producer_name,
        "producer_version": index.producer_version,
        "domain": index.domain,
        "model_version": index.model_version,
        "opset_imports": [
            # "ai.onnx" is a legal alias for the default domain; normalizing
            # it at import lets downstream opset gates and schema lookups,
            # which query "", see the declared version.
            {
                "domain": "" if item.domain == "ai.onnx" else item.domain,
                "version": item.version,
            }
            for item in index.opset_imports
        ],
        "metadata_props": [[key, value] for key, value in index.metadata_props],
    }


def _conversion_diagnostics(index: ModelIndex) -> list[Diagnostic]:
    """Findings that only make sense at the document level (task P1.3)."""
    found: list[Diagnostic] = []
    for domain in index.custom_domains:
        found.append(
            Diagnostic(
                "onnx.custom-domain",
                Severity.INFO,
                f"The model uses operators from custom domain {domain!r}; "
                "their schemas are unknown, so validation and editing are "
                "disabled for those nodes.",
                domain,
            )
        )
    main = index.main_graph
    for entry in (*main.inputs, *main.outputs):
        if not entry.is_fully_specified:
            found.append(
                Diagnostic(
                    "onnx.incomplete-io-shape",
                    Severity.INFO,
                    f"Graph value {entry.name!r} does not declare a complete "
                    "static shape; shape-dependent validation will be "
                    "conservative for paths through it.",
                    entry.id,
                )
            )
    return found


def _tool_version() -> str:
    try:
        return f"nneditor {importlib.metadata.version('nneditor')}"
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "nneditor unknown"


def index_to_document(index: ModelIndex) -> Document:
    """Build the canonical IR document for an indexed ONNX model."""
    function_io = {
        function.graph_id: (function.inputs, function.outputs)
        for function in index.functions
    }
    graphs = [
        _convert_graph(graph, io_names=function_io.get(graph.id))
        for graph in index.graphs.values()
    ]
    tensors = [_tensor_ref(tensor) for tensor in index.iter_tensors()]
    statuses, notes = _narrow_capabilities(index)

    extensions: list[tuple[str, JsonValue]] = [
        ("x-onnx.model", _model_extension(index))
    ]
    if index.functions:
        extensions.append(
            (
                "x-onnx.functions",
                [
                    {
                        "graph_id": function.graph_id,
                        "name": function.name,
                        "domain": function.domain,
                        "overload": function.overload,
                        "inputs": list(function.inputs),
                        "outputs": list(function.outputs),
                        "opset_imports": [
                            {"domain": item.domain, "version": item.version}
                            for item in function.opset_imports
                        ],
                    }
                    for function in index.functions
                ],
            )
        )

    return Document(
        source=ArtifactRef(
            path=str(index.path),
            content_hash=index.content_hash,
            byte_size=index.byte_size,
        ),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=statuses,
        graphs=graphs,
        tensors=tensors,
        entry_graph=index.main_graph_id,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=_tool_version(),
                target=index.main_graph_id,
                parameters=(("loading_mode", "safe artifact"),),
                source_artifact=index.content_hash,
            )
        ],
        diagnostics=(*index.diagnostics, *_conversion_diagnostics(index)),
        capability_notes=notes,
        extensions=extensions,
    )
