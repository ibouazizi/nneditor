"""Typed, immutable IR core (task P1.1).

These are the value types `docs/ir-schema.md` specifies: documents, graphs,
nodes, values, tensor references, provenance, and capability notes. Everything
is validated at construction — a Document that exists is internally consistent
— and everything serializes deterministically: the same document always
produces the same bytes, because golden tests, revision hashing, and cached
artifacts all depend on that.

Three spec rules shape the implementation:

* Tensor *values* never appear here; a :class:`TensorRef` is a location.
* Ports are positional and never renumbered; an omitted optional input keeps
  its position through a placeholder value.
* Anything the core cannot model rides in an extension namespace rather than
  being dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from nneditor.diagnostics import Diagnostic, Severity
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, NodeIdStability
from nneditor.ir.schema import (
    IR_SCHEMA_VERSION,
    SchemaVersion,
    validate_extension_namespace,
)

__all__ = [
    "ArtifactRef",
    "AttrKind",
    "Attribute",
    "CapabilityNote",
    "Dim",
    "Document",
    "ExternalRef",
    "Graph",
    "IrError",
    "Node",
    "PayloadRange",
    "ProvenanceEntry",
    "Storage",
    "TensorRef",
    "Value",
]

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)

Dim = int | str | None
"""A static extent, a graph-scoped symbolic name, or an unknown extent."""


class IrError(ValueError):
    """Raised when IR values do not form a consistent document."""


class Storage(StrEnum):
    """Where a referenced tensor's bytes live."""

    EMBEDDED_RAW = "embedded_raw"
    EMBEDDED_TYPED = "embedded_typed"
    EXTERNAL = "external"
    GENERATED = "generated"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class PayloadRange:
    """A half-open byte span inside the source artifact."""

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 0:
            raise IrError(
                f"payload range must be non-negative, got offset={self.offset} "
                f"length={self.length}"
            )


@dataclass(frozen=True, slots=True)
class ExternalRef:
    """Location of tensor bytes outside the source artifact."""

    location: str
    offset: int
    length: int | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.location:
            raise IrError("external reference needs a location")
        if self.offset < 0:
            raise IrError(f"external offset must be non-negative, got {self.offset}")
        if self.length is not None and self.length < 0:
            raise IrError(f"external length must be non-negative, got {self.length}")


@dataclass(frozen=True, slots=True)
class TensorRef:
    """A tensor described by location, never by content.

    ``typed_span`` (schema 1.1) locates the whole serialized ``TensorProto``
    message of an embedded-typed tensor, whose values live in typed protobuf
    fields rather than raw bytes. It is what lets the store materialize such a
    tensor on demand — a full parse of that one message, disclosed as such —
    without the import ever doing so.
    """

    id: str
    element_type: str
    dims: tuple[int, ...]
    storage: Storage
    payload: PayloadRange | None = None
    external: ExternalRef | None = None
    typed_span: PayloadRange | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise IrError("tensor reference needs an id")
        if not self.element_type:
            raise IrError(f"tensor {self.id!r} needs an element type")
        if any(dim < 0 for dim in self.dims):
            raise IrError(f"tensor {self.id!r} has a negative dimension")
        if self.storage is Storage.EMBEDDED_RAW and self.payload is None:
            raise IrError(f"embedded-raw tensor {self.id!r} needs a payload range")
        if self.storage is Storage.EXTERNAL and self.external is None:
            raise IrError(f"external tensor {self.id!r} needs an external reference")
        if self.storage in (
            Storage.EMBEDDED_TYPED,
            Storage.GENERATED,
            Storage.ABSENT,
        ) and (self.payload is not None or self.external is not None):
            raise IrError(
                f"tensor {self.id!r} with {self.storage.value} storage cannot "
                "carry a payload or external reference"
            )
        if self.typed_span is not None and self.storage is not Storage.EMBEDDED_TYPED:
            raise IrError(
                f"tensor {self.id!r} carries a typed span but has "
                f"{self.storage.value} storage"
            )

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.dims:
            count *= dim
        return count


class AttrKind(StrEnum):
    """The typed shapes a node attribute can take."""

    INT = "int"
    FLOAT = "float"
    STRING = "string"
    TENSOR = "tensor"
    GRAPH = "graph"
    INTS = "ints"
    FLOATS = "floats"
    STRINGS = "strings"
    TENSORS = "tensors"
    GRAPHS = "graphs"


_SCALAR_TYPES: Final[dict[AttrKind, type]] = {
    AttrKind.INT: int,
    AttrKind.FLOAT: float,
    AttrKind.STRING: str,
    AttrKind.TENSOR: str,
    AttrKind.GRAPH: str,
}
_LIST_TYPES: Final[dict[AttrKind, type]] = {
    AttrKind.INTS: int,
    AttrKind.FLOATS: float,
    AttrKind.STRINGS: str,
    AttrKind.TENSORS: str,
    AttrKind.GRAPHS: str,
}


@dataclass(frozen=True, slots=True)
class Attribute:
    """One typed node attribute.

    Tensor- and graph-valued attributes hold *references* (tensor ids and
    graph ids); the referents live in the document, where their consistency is
    validated.
    """

    name: str
    kind: AttrKind
    value: int | float | str | tuple[int, ...] | tuple[float, ...] | tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise IrError("attribute needs a name")
        scalar = _SCALAR_TYPES.get(self.kind)
        if scalar is not None:
            if not isinstance(self.value, scalar) or isinstance(self.value, bool):
                raise IrError(
                    f"attribute {self.name!r} of kind {self.kind.value} needs a "
                    f"{scalar.__name__} value, got {type(self.value).__name__}"
                )
            return
        element = _LIST_TYPES[self.kind]
        if not isinstance(self.value, tuple) or any(
            not isinstance(item, element) or isinstance(item, bool)
            for item in self.value
        ):
            raise IrError(
                f"attribute {self.name!r} of kind {self.kind.value} needs a "
                f"tuple of {element.__name__}"
            )

    def referenced_tensor_ids(self) -> tuple[str, ...]:
        if self.kind is AttrKind.TENSOR:
            assert isinstance(self.value, str)
            return (self.value,)
        if self.kind is AttrKind.TENSORS:
            assert isinstance(self.value, tuple)
            return tuple(str(item) for item in self.value)
        return ()

    def referenced_graph_ids(self) -> tuple[str, ...]:
        if self.kind is AttrKind.GRAPH:
            assert isinstance(self.value, str)
            return (self.value,)
        if self.kind is AttrKind.GRAPHS:
            assert isinstance(self.value, tuple)
            return tuple(str(item) for item in self.value)
        return ()


@dataclass(frozen=True, slots=True)
class Value:
    """An edge in the dataflow graph, not a buffer."""

    id: str
    name: str
    element_type: str | None = None
    shape: tuple[Dim, ...] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise IrError("value needs an id")
        for dim in self.shape or ():
            if isinstance(dim, int) and dim < 0:
                raise IrError(f"value {self.id!r} has a negative dimension")
            if isinstance(dim, str) and not dim:
                raise IrError(f"value {self.id!r} has an empty symbolic dimension")

    @property
    def has_symbolic_dimensions(self) -> bool:
        return any(isinstance(dim, str) for dim in self.shape or ())

    @property
    def is_fully_specified(self) -> bool:
        if self.element_type is None or self.shape is None:
            return False
        return all(isinstance(dim, int) for dim in self.shape)


@dataclass(frozen=True, slots=True)
class Node:
    """One operator invocation with positional ports."""

    id: str
    id_stability: NodeIdStability
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    domain: str = ""
    overload: str = ""
    attributes: tuple[Attribute, ...] = ()
    subgraphs: tuple[str, ...] = ()
    source_name: str | None = None
    source_location: str | None = None
    has_side_effects: bool | None = None
    mutates_inputs: bool | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise IrError("node needs an id")
        if not self.op_type:
            raise IrError(f"node {self.id!r} needs an op_type")
        names = [attribute.name for attribute in self.attributes]
        if len(set(names)) != len(names):
            raise IrError(f"node {self.id!r} has duplicate attribute names")
        declared = set(self.subgraphs)
        referenced = {
            graph_id
            for attribute in self.attributes
            for graph_id in attribute.referenced_graph_ids()
        }
        if referenced - declared:
            missing = ", ".join(sorted(referenced - declared))
            raise IrError(
                f"node {self.id!r} references subgraphs not declared in "
                f"`subgraphs`: {missing}"
            )

    @property
    def qualified_op_type(self) -> str:
        base = f"{self.domain}::{self.op_type}" if self.domain else self.op_type
        return f"{base}:{self.overload}" if self.overload else base

    def attribute(self, name: str) -> Attribute:
        for item in self.attributes:
            if item.name == name:
                return item
        raise KeyError(name)


class Graph:
    """A validated dataflow graph.

    Producer and consumer links are *derived* here rather than stored on each
    value: storing both directions would create a second source of truth that
    validation would then have to police. Serialization writes only the
    declarative fields; the links are rebuilt on read.
    """

    __slots__ = (
        "_consumers",
        "_node_map",
        "_producers",
        "_value_map",
        "id",
        "initializers",
        "inputs",
        "name",
        "nodes",
        "outputs",
        "parent_node",
        "values",
    )

    def __init__(
        self,
        *,
        id: str,
        name: str,
        nodes: Iterable[Node] = (),
        values: Iterable[Value] = (),
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        initializers: Iterable[str] = (),
        parent_node: str | None = None,
    ) -> None:
        if not id:
            raise IrError("graph needs an id")
        self.id = id
        self.name = name
        self.nodes = tuple(nodes)
        self.values = tuple(values)
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.initializers = tuple(initializers)
        self.parent_node = parent_node

        value_map: dict[str, Value] = {}
        for value in self.values:
            if value.id in value_map:
                raise IrError(f"graph {id!r} declares value {value.id!r} twice")
            value_map[value.id] = value
        node_map: dict[str, Node] = {}
        producers: dict[str, tuple[str, int]] = {}
        consumers: dict[str, list[tuple[str, int]]] = {}
        for node in self.nodes:
            if node.id in node_map:
                raise IrError(f"graph {id!r} declares node {node.id!r} twice")
            node_map[node.id] = node
            for port, value_id in enumerate(node.inputs):
                if value_id not in value_map:
                    raise IrError(
                        f"node {node.id!r} input {port} references undeclared "
                        f"value {value_id!r}"
                    )
                consumers.setdefault(value_id, []).append((node.id, port))
            for port, value_id in enumerate(node.outputs):
                if value_id not in value_map:
                    raise IrError(
                        f"node {node.id!r} output {port} references undeclared "
                        f"value {value_id!r}"
                    )
                if value_id in producers:
                    raise IrError(
                        f"value {value_id!r} is produced twice: by "
                        f"{producers[value_id][0]!r} and {node.id!r}"
                    )
                producers[value_id] = (node.id, port)
        for value_id in self.inputs:
            if value_id not in value_map:
                raise IrError(f"graph input {value_id!r} is not a declared value")
            if value_id in producers:
                raise IrError(
                    f"graph input {value_id!r} is also produced by node "
                    f"{producers[value_id][0]!r}"
                )
        for value_id in self.outputs:
            if value_id not in value_map:
                raise IrError(f"graph output {value_id!r} is not a declared value")
        self._value_map = value_map
        self._node_map = node_map
        self._producers = producers
        self._consumers = {
            value_id: tuple(ports) for value_id, ports in consumers.items()
        }

    def node(self, node_id: str) -> Node:
        return self._node_map[node_id]

    def value(self, value_id: str) -> Value:
        return self._value_map[value_id]

    def has_value(self, value_id: str) -> bool:
        return value_id in self._value_map

    def producer(self, value_id: str) -> tuple[str, int] | None:
        """The ``(node id, output port)`` producing a value, if any."""
        if value_id not in self._value_map:
            raise KeyError(value_id)
        return self._producers.get(value_id)

    def consumers(self, value_id: str) -> tuple[tuple[str, int], ...]:
        """Every ``(node id, input port)`` consuming a value."""
        if value_id not in self._value_map:
            raise KeyError(value_id)
        return self._consumers.get(value_id, ())

    @property
    def symbolic_dimensions(self) -> tuple[str, ...]:
        """Symbolic dimension names used in this graph scope, sorted."""
        names = {
            dim
            for value in self.values
            for dim in value.shape or ()
            if isinstance(dim, str)
        }
        return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """The immutable source artifact a document was imported from."""

    path: str
    content_hash: str
    byte_size: int

    def __post_init__(self) -> None:
        if not self.path:
            raise IrError("artifact reference needs a path")
        if ":" not in self.content_hash:
            raise IrError(
                "content hash must be algorithm-prefixed, e.g. 'sha256:...', "
                f"got {self.content_hash!r}"
            )
        if self.byte_size < 0:
            raise IrError("artifact byte size must be non-negative")


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    """One recorded operation in a document's history.

    No timestamp by design: reopening a model and re-running the same commands
    must produce identical document bytes (spec §6).
    """

    operation: str
    tool_version: str
    target: str | None = None
    parameters: tuple[tuple[str, str], ...] = ()
    dependency_versions: tuple[tuple[str, str], ...] = ()
    source_artifact: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.operation:
            raise IrError("provenance entry needs an operation")
        if not self.tool_version:
            raise IrError("provenance entry needs a tool version")


@dataclass(frozen=True, slots=True)
class CapabilityNote:
    """A per-entity capability narrowing with its user-facing reason."""

    entity_id: str
    capability: Capability
    availability: Availability
    reason: str

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise IrError("capability note needs an entity id")
        if not self.reason.strip():
            raise IrError("capability note needs a reason")


class Document:
    """A validated, immutable IR document.

    Cross-graph consistency is enforced here because only the document sees
    every graph: subgraph attachment, tensor-reference resolution, and
    extension namespace validity.
    """

    __slots__ = (
        "artifact_kind",
        "capabilities",
        "capability_notes",
        "diagnostics",
        "entry_graph",
        "extensions",
        "graphs",
        "provenance",
        "schema_version",
        "source",
        "tensors",
    )

    def __init__(
        self,
        *,
        source: ArtifactRef,
        artifact_kind: ArtifactKind,
        capabilities: Iterable[CapabilityStatus],
        graphs: Iterable[Graph],
        tensors: Iterable[TensorRef] = (),
        entry_graph: str = ROOT_GRAPH_ID,
        provenance: Iterable[ProvenanceEntry] = (),
        diagnostics: Iterable[Diagnostic] = (),
        capability_notes: Iterable[CapabilityNote] = (),
        extensions: Iterable[tuple[str, JsonValue]] = (),
        schema_version: SchemaVersion = IR_SCHEMA_VERSION,
    ) -> None:
        self.schema_version = schema_version
        self.source = source
        self.artifact_kind = artifact_kind
        self.entry_graph = entry_graph
        self.provenance = tuple(provenance)
        self.diagnostics = tuple(diagnostics)
        self.capability_notes = tuple(capability_notes)
        self.extensions = tuple(extensions)

        statuses = tuple(capabilities)
        declared = [status.capability for status in statuses]
        if len(set(declared)) != len(declared):
            raise IrError("duplicate document capability status")
        missing = set(Capability) - set(declared)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise IrError(f"document is missing capability statuses: {names}")
        self.capabilities: Mapping[Capability, CapabilityStatus] = MappingProxyType(
            {status.capability: status for status in statuses}
        )

        graph_map: dict[str, Graph] = {}
        for graph in graphs:
            if graph.id in graph_map:
                raise IrError(f"document declares graph {graph.id!r} twice")
            graph_map[graph.id] = graph
        self.graphs: Mapping[str, Graph] = MappingProxyType(graph_map)
        if entry_graph not in graph_map:
            raise IrError(f"entry graph {entry_graph!r} is not in the document")

        tensor_map: dict[str, TensorRef] = {}
        for tensor in tensors:
            if tensor.id in tensor_map:
                raise IrError(f"document declares tensor {tensor.id!r} twice")
            tensor_map[tensor.id] = tensor
        self.tensors: Mapping[str, TensorRef] = MappingProxyType(tensor_map)

        for graph in graph_map.values():
            for tensor_id in graph.initializers:
                if tensor_id not in tensor_map:
                    raise IrError(
                        f"graph {graph.id!r} initializer {tensor_id!r} is not a "
                        "declared tensor"
                    )
            for node in graph.nodes:
                for subgraph_id in node.subgraphs:
                    subgraph = graph_map.get(subgraph_id)
                    if subgraph is None:
                        raise IrError(
                            f"node {node.id!r} references missing subgraph "
                            f"{subgraph_id!r}"
                        )
                    if subgraph.parent_node != node.id:
                        raise IrError(
                            f"subgraph {subgraph_id!r} does not link back to its "
                            f"owning node {node.id!r}"
                        )
                for attribute in node.attributes:
                    for tensor_id in attribute.referenced_tensor_ids():
                        if tensor_id not in tensor_map:
                            raise IrError(
                                f"node {node.id!r} attribute {attribute.name!r} "
                                f"references missing tensor {tensor_id!r}"
                            )
        for namespace, _ in self.extensions:
            validate_extension_namespace(namespace)
        seen_namespaces = [namespace for namespace, _ in self.extensions]
        duplicates = sorted(
            {name for name in seen_namespaces if seen_namespaces.count(name) > 1}
        )
        if duplicates:
            raise IrError("duplicate extension namespace: " + ", ".join(duplicates))

    @property
    def main_graph(self) -> Graph:
        return self.graphs[self.entry_graph]

    def capability(self, capability: Capability) -> CapabilityStatus:
        return self.capabilities[capability]

    def notes_for(self, entity_id: str) -> tuple[CapabilityNote, ...]:
        """Capability narrowings attached to one entity."""
        return tuple(
            note for note in self.capability_notes if note.entity_id == entity_id
        )

    @property
    def has_errors(self) -> bool:
        return any(item.severity is Severity.ERROR for item in self.diagnostics)
