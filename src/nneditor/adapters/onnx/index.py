"""The compact index produced by the ONNX lazy indexer.

Nothing here holds tensor values. A tensor is described by *where* its bytes
live — a byte range inside the model file, or an external file with an offset
and a length — so a large model costs roughly its topology, not its weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from nneditor.adapters.onnx.dtypes import ElementType, packed_byte_length
from nneditor.diagnostics import DiagnosticLog
from nneditor.ir.core import Attribute
from nneditor.ir.identity import NodeIdStability
from nneditor.storage.reader import ByteRange, ReadStats

__all__ = [
    "Dimension",
    "ExternalDataReference",
    "FunctionIndex",
    "GraphIndex",
    "ModelIndex",
    "NodeIndex",
    "OpsetImport",
    "TensorIndex",
    "TensorStorage",
    "ValueIndex",
]

Dimension = int | str | None
"""A static extent, a symbolic dimension name, or an unknown extent."""


class TensorStorage(StrEnum):
    """Where a tensor's bytes are kept."""

    EMBEDDED_RAW = "embedded raw"
    """Contiguous ``raw_data`` inside the model file; range readable."""

    EMBEDDED_TYPED = "embedded typed"
    """A typed field such as ``float_data``; readable but not a byte range."""

    EXTERNAL = "external"
    """A separate file addressed by location, offset, and length."""

    ABSENT = "absent"
    """The tensor declares no data at all."""


@dataclass(frozen=True, slots=True)
class ExternalDataReference:
    """A validated ``external_data`` entry."""

    location: str
    offset: int
    length: int | None
    checksum: str | None = None
    resolved_path: Path | None = None
    error: str | None = None

    @property
    def is_usable(self) -> bool:
        """Whether the reference passed validation and can be read."""
        return self.error is None and self.resolved_path is not None


@dataclass(frozen=True, slots=True)
class TensorIndex:
    """One initializer or attribute tensor, described by location only."""

    id: str
    name: str
    graph_id: str
    element_type: ElementType
    dims: tuple[int, ...]
    storage: TensorStorage
    payload: ByteRange | None = None
    external: ExternalDataReference | None = None
    error: str | None = None
    message_span: ByteRange | None = None
    """The whole serialized ``TensorProto`` message, recorded for
    embedded-typed tensors so their values can be materialized on demand."""

    @property
    def element_count(self) -> int:
        """Number of elements implied by the declared shape."""
        count = 1
        for dim in self.dims:
            count *= dim
        return count

    @property
    def expected_byte_length(self) -> int | None:
        """Bytes the declared shape and element type imply, when derivable."""
        return packed_byte_length(self.element_type, self.element_count)

    @property
    def is_readable(self) -> bool:
        """Whether the bytes can be fetched as a range without a full parse."""
        if self.error is not None:
            return False
        if self.storage is TensorStorage.EMBEDDED_RAW:
            return self.payload is not None
        if self.storage is TensorStorage.EXTERNAL:
            return self.external is not None and self.external.is_usable
        return False


@dataclass(frozen=True, slots=True)
class NodeIndex:
    """One operator invocation."""

    id: str
    id_stability: NodeIdStability
    name: str
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    overload: str = ""
    attributes: tuple[Attribute, ...] = ()
    unsupported_attributes: tuple[str, ...] = ()
    subgraph_ids: tuple[str, ...] = ()

    @property
    def attribute_names(self) -> tuple[str, ...]:
        """Names of the attributes whose values were captured."""
        return tuple(attribute.name for attribute in self.attributes)

    @property
    def qualified_op_type(self) -> str:
        """``domain::op`` for custom domains, or the bare operator name."""
        base = f"{self.domain}::{self.op_type}" if self.domain else self.op_type
        return f"{base}:{self.overload}" if self.overload else base


@dataclass(frozen=True, slots=True)
class ValueIndex:
    """A graph input, output, or annotated intermediate value."""

    id: str
    name: str
    element_type: ElementType | None
    shape: tuple[Dimension, ...] | None

    @property
    def has_symbolic_dimensions(self) -> bool:
        """Whether any extent is a named symbolic dimension."""
        return any(isinstance(dim, str) for dim in self.shape or ())

    @property
    def is_fully_specified(self) -> bool:
        """Whether the type and every extent are known and static."""
        if self.element_type is None or self.shape is None:
            return False
        return all(isinstance(dim, int) for dim in self.shape)


@dataclass(frozen=True, slots=True)
class GraphIndex:
    """One graph: the main graph, a control-flow body, or a function body."""

    id: str
    name: str
    nodes: tuple[NodeIndex, ...] = ()
    inputs: tuple[ValueIndex, ...] = ()
    outputs: tuple[ValueIndex, ...] = ()
    value_info: tuple[ValueIndex, ...] = ()
    initializers: tuple[TensorIndex, ...] = ()
    attribute_tensors: tuple[TensorIndex, ...] = ()
    sparse_initializer_count: int = 0
    parent_node_id: str | None = None

    @property
    def tensors(self) -> tuple[TensorIndex, ...]:
        """Initializers followed by tensors carried inside node attributes."""
        return self.initializers + self.attribute_tensors


@dataclass(frozen=True, slots=True)
class OpsetImport:
    """One operator-set import."""

    domain: str
    version: int

    @property
    def is_default_domain(self) -> bool:
        """Whether this is the standard ONNX operator set."""
        return self.domain in ("", "ai.onnx")


@dataclass(frozen=True, slots=True)
class FunctionIndex:
    """A model-local function definition."""

    graph_id: str
    name: str
    domain: str
    overload: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    opset_imports: tuple[OpsetImport, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelIndex:
    """Everything the viewer knows about a model before touching a tensor."""

    path: Path
    content_hash: str
    model_hash: str
    byte_size: int
    ir_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    opset_imports: tuple[OpsetImport, ...]
    metadata_props: tuple[tuple[str, str], ...]
    main_graph_id: str
    graphs: Mapping[str, GraphIndex]
    functions: tuple[FunctionIndex, ...]
    diagnostics: DiagnosticLog
    stats: ReadStats
    external_files: tuple[Path, ...] = ()
    external_hashes: tuple[tuple[str, str], ...] = ()
    _tensor_lookup: Mapping[str, TensorIndex] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    @property
    def main_graph(self) -> GraphIndex:
        """The model's entry graph."""
        return self.graphs[self.main_graph_id]

    @property
    def node_count(self) -> int:
        """Nodes across every graph, including subgraphs and functions."""
        return sum(len(graph.nodes) for graph in self.graphs.values())

    @property
    def tensor_count(self) -> int:
        """Indexed tensors across every graph."""
        return sum(len(graph.tensors) for graph in self.graphs.values())

    def iter_tensors(self) -> Iterator[TensorIndex]:
        """Iterate every indexed tensor in graph order."""
        for graph in self.graphs.values():
            yield from graph.tensors

    def tensor(self, tensor_id: str) -> TensorIndex:
        """Look up a tensor by identifier."""
        return self._tensor_lookup[tensor_id]

    @property
    def declared_tensor_bytes(self) -> int:
        """Sum of the declared payload sizes, which is *not* what was read."""
        total = 0
        for tensor in self.iter_tensors():
            if tensor.payload is not None:
                total += tensor.payload.length
            elif tensor.external is not None and tensor.external.length is not None:
                total += tensor.external.length
        return total

    @property
    def custom_domains(self) -> tuple[str, ...]:
        """Operator domains outside the standard ONNX set, in first-seen order."""
        seen: dict[str, None] = {}
        for graph in self.graphs.values():
            for node in graph.nodes:
                if node.domain and node.domain != "ai.onnx":
                    seen.setdefault(node.domain, None)
        return tuple(seen)
