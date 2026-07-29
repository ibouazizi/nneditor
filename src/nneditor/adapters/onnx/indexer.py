"""The ONNX lazy indexer (task P0.4).

``index_model`` walks the serialized ``ModelProto`` and returns a
:class:`~nneditor.adapters.onnx.index.ModelIndex`. It first streams each source
component through SHA-256 to establish the complete artifact identity. The
structural parser then reads the bytes that describe structure and skips the
bytes that hold values: an embedded tensor becomes a recorded byte range and
an external tensor becomes a validated file, offset, and length. No tensor is
decoded or retained. ``index.stats`` lists every range the parser requested;
identity hashing is deliberately accounted separately.

Validation happens before anything is trusted:

* external locations are confined to an approved root
  (:func:`nneditor.storage.paths.resolve_within`);
* offsets and lengths are checked against the real size of the target file;
* declared payload sizes are compared with the size implied by the shape and
  element type;
* producer-declared external checksums are verified only on request; the
  independent SHA-256 identity digest is always computed.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from nneditor.adapters.onnx.dtypes import ElementType, element_type_for
from nneditor.adapters.onnx.index import (
    Dimension,
    ExternalDataReference,
    FunctionIndex,
    GraphIndex,
    ModelIndex,
    NodeIndex,
    OpsetImport,
    TensorIndex,
    TensorStorage,
    ValueIndex,
)
from nneditor.adapters.onnx.wire import (
    MalformedProtobufError,
    MessageCursor,
    ScanLimits,
    WireField,
    WireType,
)
from nneditor.cancellation import CancellationToken
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.core import Attribute, AttrKind, IrError
from nneditor.ir.identity import (
    ROOT_GRAPH_ID,
    NodeIdStability,
    attribute_subgraph_id,
    encode_segment,
    initializer_id,
    value_id,
)
from nneditor.ir.identity import node_id as derive_node_id
from nneditor.storage.paths import UnsafePathError, resolve_within
from nneditor.storage.reader import (
    DEFAULT_BLOCK_SIZE,
    ArtifactReader,
    ByteRange,
    hash_file,
)

__all__ = ["OnnxIndexError", "ScanLimits", "index_model", "read_tensor_bytes"]


class OnnxIndexError(ValueError):
    """Raised when the artifact cannot be indexed at all."""


# ---------------------------------------------------------------------------
# Field numbers from onnx.proto. Kept as named constants so the scanner reads
# like the schema it implements.
# ---------------------------------------------------------------------------

_MODEL_IR_VERSION: Final = 1
_MODEL_PRODUCER_NAME: Final = 2
_MODEL_PRODUCER_VERSION: Final = 3
_MODEL_DOMAIN: Final = 4
_MODEL_VERSION: Final = 5
_MODEL_GRAPH: Final = 7
_MODEL_OPSET_IMPORT: Final = 8
_MODEL_METADATA_PROPS: Final = 14
_MODEL_FUNCTIONS: Final = 25

_GRAPH_NODE: Final = 1
_GRAPH_NAME: Final = 2
_GRAPH_INITIALIZER: Final = 5
_GRAPH_INPUT: Final = 11
_GRAPH_OUTPUT: Final = 12
_GRAPH_VALUE_INFO: Final = 13
_GRAPH_SPARSE_INITIALIZER: Final = 15

_NODE_INPUT: Final = 1
_NODE_OUTPUT: Final = 2
_NODE_NAME: Final = 3
_NODE_OP_TYPE: Final = 4
_NODE_ATTRIBUTE: Final = 5
_NODE_DOMAIN: Final = 7
_NODE_OVERLOAD: Final = 8

_ATTR_NAME: Final = 1
_ATTR_F: Final = 2
_ATTR_I: Final = 3
_ATTR_S: Final = 4
_ATTR_TENSOR: Final = 5
_ATTR_GRAPH: Final = 6
_ATTR_FLOATS: Final = 7
_ATTR_INTS: Final = 8
_ATTR_STRINGS: Final = 9
_ATTR_TENSORS: Final = 10
_ATTR_GRAPHS: Final = 11
_ATTR_TP: Final = 14
_ATTR_TYPE_PROTOS: Final = 15
_ATTR_TYPE: Final = 20
_ATTR_REF_NAME: Final = 21
_ATTR_SPARSE_TENSOR: Final = 22
_ATTR_SPARSE_TENSORS: Final = 23

_ATTR_TYPE_TO_KIND: Final[dict[int, AttrKind]] = {
    1: AttrKind.FLOAT,
    2: AttrKind.INT,
    3: AttrKind.STRING,
    4: AttrKind.TENSOR,
    5: AttrKind.GRAPH,
    6: AttrKind.FLOATS,
    7: AttrKind.INTS,
    8: AttrKind.STRINGS,
    9: AttrKind.TENSORS,
    10: AttrKind.GRAPHS,
}

_TENSOR_DIMS: Final = 1
_TENSOR_DATA_TYPE: Final = 2
_TENSOR_FLOAT_DATA: Final = 4
_TENSOR_INT32_DATA: Final = 5
_TENSOR_STRING_DATA: Final = 6
_TENSOR_INT64_DATA: Final = 7
_TENSOR_NAME: Final = 8
_TENSOR_RAW_DATA: Final = 9
_TENSOR_DOUBLE_DATA: Final = 10
_TENSOR_UINT64_DATA: Final = 11
_TENSOR_EXTERNAL_DATA: Final = 13
_TENSOR_DATA_LOCATION: Final = 14

_TYPED_DATA_FIELDS: Final = frozenset(
    {
        _TENSOR_FLOAT_DATA,
        _TENSOR_INT32_DATA,
        _TENSOR_STRING_DATA,
        _TENSOR_INT64_DATA,
        _TENSOR_DOUBLE_DATA,
        _TENSOR_UINT64_DATA,
    }
)
_DATA_LOCATION_EXTERNAL: Final = 1

_VALUE_INFO_NAME: Final = 1
_VALUE_INFO_TYPE: Final = 2

_TYPE_TENSOR_TYPE: Final = 1
_TENSOR_TYPE_ELEM_TYPE: Final = 1
_TENSOR_TYPE_SHAPE: Final = 2
_SHAPE_DIM: Final = 1
_DIM_VALUE: Final = 1
_DIM_PARAM: Final = 2

_ENTRY_KEY: Final = 1
_ENTRY_VALUE: Final = 2

_OPSET_DOMAIN: Final = 1
_OPSET_VERSION: Final = 2

_FUNCTION_NAME: Final = 1
_FUNCTION_INPUT: Final = 4
_FUNCTION_OUTPUT: Final = 5
_FUNCTION_NODE: Final = 7
_FUNCTION_OPSET_IMPORT: Final = 9
_FUNCTION_DOMAIN: Final = 10
_FUNCTION_OVERLOAD: Final = 13

_MAX_ELEMENT_COUNT: Final = 1 << 60

_TENSOR_READ_CHUNK: Final = 64 * 1024 * 1024
"""Bytes per read while materializing one tensor; stays under MAX_SINGLE_READ."""


@dataclass(slots=True)
class _NodeParts:
    """Scalar node fields gathered before attributes are interpreted."""

    name: str = ""
    op_type: str = ""
    domain: str = ""
    overload: str = ""
    inputs: list[str] | None = None
    outputs: list[str] | None = None


class _Indexer:
    """One indexing pass over one artifact."""

    def __init__(
        self,
        reader: ArtifactReader,
        *,
        limits: ScanLimits,
        external_root: Path,
        verify_checksums: bool,
        token: CancellationToken | None,
    ) -> None:
        self._reader = reader
        self._limits = limits
        self._external_root = external_root
        self._verify_checksums = verify_checksums
        self._token = token
        self.log = DiagnosticLog()
        self._graphs: dict[str, GraphIndex] = {}
        self._tensors: dict[str, TensorIndex] = {}
        self._external_files: dict[Path, None] = {}
        self._external_sizes: dict[Path, int] = {}

    # -- model ------------------------------------------------------------

    def run(self, content_hash: str) -> ModelIndex:
        if self._token is not None:
            self._token.raise_if_cancelled()
        cursor = MessageCursor(
            self._reader,
            0,
            self._reader.size,
            self._limits,
            token=self._token,
        )
        ir_version = 0
        producer_name = ""
        producer_version = ""
        domain = ""
        model_version = 0
        opset_imports: list[OpsetImport] = []
        metadata: list[tuple[str, str]] = []
        graph_field: WireField | None = None
        function_fields: list[WireField] = []

        for item in cursor:
            if self._token is not None:
                self._token.raise_if_cancelled()
            if item.number == _MODEL_IR_VERSION and item.wire_type is WireType.VARINT:
                ir_version = item.as_int64
            elif item.number == _MODEL_VERSION and item.wire_type is WireType.VARINT:
                model_version = item.as_int64
            elif item.wire_type is not WireType.LEN:
                continue
            elif item.number == _MODEL_PRODUCER_NAME:
                producer_name = cursor.text(item)
            elif item.number == _MODEL_PRODUCER_VERSION:
                producer_version = cursor.text(item)
            elif item.number == _MODEL_DOMAIN:
                domain = cursor.text(item)
            elif item.number == _MODEL_OPSET_IMPORT:
                opset_imports.append(self._opset_import(cursor, item))
            elif item.number == _MODEL_METADATA_PROPS:
                metadata.append(self._string_entry(cursor, item))
            elif item.number == _MODEL_GRAPH:
                graph_field = item
            elif item.number == _MODEL_FUNCTIONS:
                function_fields.append(item)

        if graph_field is None:
            raise OnnxIndexError(
                f"{self._reader.path.name} has no graph; it is not an ONNX model"
            )

        self._graph(cursor.submessage(graph_field), ROOT_GRAPH_ID)
        functions = tuple(
            self._function(cursor.submessage(item)) for item in function_fields
        )

        return ModelIndex(
            path=self._reader.path,
            content_hash=content_hash,
            model_hash=content_hash,
            byte_size=self._reader.size,
            ir_version=ir_version,
            producer_name=producer_name,
            producer_version=producer_version,
            domain=domain,
            model_version=model_version,
            opset_imports=tuple(opset_imports),
            metadata_props=tuple(metadata),
            main_graph_id=ROOT_GRAPH_ID,
            graphs=MappingProxyType(dict(self._graphs)),
            functions=functions,
            diagnostics=self.log,
            stats=self._reader.stats,
            external_files=tuple(self._external_files),
            _tensor_lookup=MappingProxyType(dict(self._tensors)),
        )

    def _opset_import(self, cursor: MessageCursor, item: WireField) -> OpsetImport:
        inner = cursor.submessage(item)
        opset_domain = ""
        version = 0
        for entry in inner:
            if entry.number == _OPSET_DOMAIN and entry.wire_type is WireType.LEN:
                opset_domain = inner.text(entry)
            elif entry.number == _OPSET_VERSION and entry.wire_type is WireType.VARINT:
                version = entry.as_int64
        return OpsetImport(opset_domain, version)

    def _string_entry(self, cursor: MessageCursor, item: WireField) -> tuple[str, str]:
        inner = cursor.submessage(item)
        key = ""
        value = ""
        for entry in inner:
            if entry.wire_type is not WireType.LEN:
                continue
            if entry.number == _ENTRY_KEY:
                key = inner.text(entry)
            elif entry.number == _ENTRY_VALUE:
                value = inner.text(entry)
        return key, value

    # -- graphs -----------------------------------------------------------

    def _graph(
        self,
        cursor: MessageCursor,
        graph_id: str,
        parent_node_id: str | None = None,
    ) -> GraphIndex:
        name = ""
        node_fields: list[WireField] = []
        initializer_fields: list[WireField] = []
        input_fields: list[WireField] = []
        output_fields: list[WireField] = []
        value_info_fields: list[WireField] = []
        sparse_count = 0

        for item in cursor:
            if item.wire_type is not WireType.LEN:
                continue
            if item.number == _GRAPH_NAME:
                name = cursor.text(item)
            elif item.number == _GRAPH_NODE:
                node_fields.append(item)
            elif item.number == _GRAPH_INITIALIZER:
                initializer_fields.append(item)
            elif item.number == _GRAPH_INPUT:
                input_fields.append(item)
            elif item.number == _GRAPH_OUTPUT:
                output_fields.append(item)
            elif item.number == _GRAPH_VALUE_INFO:
                value_info_fields.append(item)
            elif item.number == _GRAPH_SPARSE_INITIALIZER:
                sparse_count += 1

        if sparse_count:
            self.log.add(
                "onnx.sparse-initializer-unsupported",
                Severity.WARNING,
                f"{sparse_count} sparse initializer(s) are indexed as absent; "
                "sparse tensor storage is not modelled yet.",
                graph_id,
            )

        initializers = tuple(
            self._tensor(
                cursor.submessage(item),
                graph_id,
                ordinal,
                span=ByteRange(item.offset, item.length),
            )
            for ordinal, item in enumerate(initializer_fields)
        )
        nodes: list[NodeIndex] = []
        attribute_tensors: list[TensorIndex] = []
        seen_ids: set[str] = set()
        for ordinal, item in enumerate(node_fields):
            node, tensors = self._node(
                cursor.submessage(item), graph_id, ordinal, seen_ids
            )
            nodes.append(node)
            attribute_tensors.extend(tensors)

        graph = GraphIndex(
            id=graph_id,
            name=name,
            nodes=tuple(nodes),
            inputs=tuple(
                self._value_info(cursor.submessage(item), graph_id)
                for item in input_fields
            ),
            outputs=tuple(
                self._value_info(cursor.submessage(item), graph_id)
                for item in output_fields
            ),
            value_info=tuple(
                self._value_info(cursor.submessage(item), graph_id)
                for item in value_info_fields
            ),
            initializers=initializers,
            attribute_tensors=tuple(attribute_tensors),
            sparse_initializer_count=sparse_count,
            parent_node_id=parent_node_id,
        )
        self._graphs[graph_id] = graph
        return graph

    def _function(self, cursor: MessageCursor) -> FunctionIndex:
        name = ""
        domain = ""
        overload = ""
        inputs: list[str] = []
        outputs: list[str] = []
        node_fields: list[WireField] = []
        opset_imports: list[OpsetImport] = []

        for item in cursor:
            if item.wire_type is not WireType.LEN:
                continue
            if item.number == _FUNCTION_NAME:
                name = cursor.text(item)
            elif item.number == _FUNCTION_DOMAIN:
                domain = cursor.text(item)
            elif item.number == _FUNCTION_OVERLOAD:
                overload = cursor.text(item)
            elif item.number == _FUNCTION_INPUT:
                inputs.append(cursor.text(item))
            elif item.number == _FUNCTION_OUTPUT:
                outputs.append(cursor.text(item))
            elif item.number == _FUNCTION_NODE:
                node_fields.append(item)
            elif item.number == _FUNCTION_OPSET_IMPORT:
                opset_imports.append(self._opset_import(cursor, item))

        suffix = f"/{encode_segment(overload)}" if overload else ""
        graph_id = f"g:fn:{encode_segment(domain)}/{encode_segment(name)}{suffix}"
        nodes: list[NodeIndex] = []
        attribute_tensors: list[TensorIndex] = []
        seen_ids: set[str] = set()
        for ordinal, item in enumerate(node_fields):
            node, tensors = self._node(
                cursor.submessage(item), graph_id, ordinal, seen_ids
            )
            nodes.append(node)
            attribute_tensors.extend(tensors)
        self._graphs[graph_id] = GraphIndex(
            id=graph_id,
            name=name,
            nodes=tuple(nodes),
            attribute_tensors=tuple(attribute_tensors),
        )
        return FunctionIndex(
            graph_id=graph_id,
            name=name,
            domain=domain,
            overload=overload,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            opset_imports=tuple(opset_imports),
        )

    # -- nodes ------------------------------------------------------------

    def _node(
        self,
        cursor: MessageCursor,
        graph_id: str,
        ordinal: int,
        seen_ids: set[str],
    ) -> tuple[NodeIndex, list[TensorIndex]]:
        parts = _NodeParts(inputs=[], outputs=[])
        attribute_fields: list[WireField] = []
        for item in cursor:
            if item.wire_type is not WireType.LEN:
                continue
            if item.number == _NODE_INPUT:
                assert parts.inputs is not None
                parts.inputs.append(cursor.text(item))
            elif item.number == _NODE_OUTPUT:
                assert parts.outputs is not None
                parts.outputs.append(cursor.text(item))
            elif item.number == _NODE_NAME:
                parts.name = cursor.text(item)
            elif item.number == _NODE_OP_TYPE:
                parts.op_type = cursor.text(item)
            elif item.number == _NODE_DOMAIN:
                parts.domain = cursor.text(item)
            elif item.number == _NODE_OVERLOAD:
                parts.overload = cursor.text(item)
            elif item.number == _NODE_ATTRIBUTE:
                attribute_fields.append(item)

        inputs = tuple(parts.inputs or ())
        outputs = tuple(parts.outputs or ())
        node_id, stability = _derive_node_id(graph_id, parts, outputs, ordinal)
        if node_id in seen_ids:
            self.log.add(
                "onnx.duplicate-node-id",
                Severity.WARNING,
                "Two nodes derive the same identifier; the later one falls back "
                "to its position, which is not stable across edits.",
                node_id,
            )
            node_id = f"n:{graph_id}#idx:{ordinal}"
            stability = NodeIdStability.POSITIONAL
        seen_ids.add(node_id)
        if stability is NodeIdStability.POSITIONAL:
            self.log.add(
                "onnx.positional-node-id",
                Severity.INFO,
                "Node has neither a name nor an output value name, so its "
                "identifier depends on its position in the graph.",
                node_id,
            )

        attributes: list[Attribute] = []
        unsupported: list[str] = []
        subgraph_ids: list[str] = []
        tensors: list[TensorIndex] = []
        for item in attribute_fields:
            attribute, attr_name, attr_subgraphs, attr_tensors = self._attribute(
                cursor.submessage(item), graph_id, node_id
            )
            if attribute is not None:
                attributes.append(attribute)
            elif attr_name:
                unsupported.append(attr_name)
            subgraph_ids.extend(attr_subgraphs)
            tensors.extend(attr_tensors)

        node = NodeIndex(
            id=node_id,
            id_stability=stability,
            name=parts.name,
            op_type=parts.op_type,
            domain=parts.domain,
            overload=parts.overload,
            inputs=inputs,
            outputs=outputs,
            attributes=tuple(attributes),
            unsupported_attributes=tuple(unsupported),
            subgraph_ids=tuple(subgraph_ids),
        )
        return node, tensors

    def _attribute(
        self, cursor: MessageCursor, graph_id: str, node_id: str
    ) -> tuple[Attribute | None, str, list[str], list[TensorIndex]]:
        """Parse one attribute into a typed IR attribute plus its referents.

        Returns ``(attribute, name, subgraph ids, attribute tensors)``; the
        attribute is ``None`` when the value cannot be represented, in which
        case a diagnostic explains why and the name reports it as unsupported.
        """
        name = ""
        declared_type = 0
        scalar_float: float | None = None
        scalar_int: int | None = None
        scalar_bytes: bytes | None = None
        floats: list[float] = []
        ints: list[int] = []
        strings: list[str] = []
        unsupported_field = False
        oversized_field = False
        graph_fields: list[tuple[WireField, bool]] = []
        tensor_fields: list[tuple[WireField, bool]] = []
        for item in cursor:
            if item.number == _ATTR_NAME and item.wire_type is WireType.LEN:
                name = cursor.text(item)
            elif item.number == _ATTR_TYPE and item.wire_type is WireType.VARINT:
                declared_type = item.as_int32
            elif item.number == _ATTR_F and item.wire_type is WireType.I32:
                scalar_float = self._fixed32_float(item)
            elif item.number == _ATTR_I and item.wire_type is WireType.VARINT:
                scalar_int = item.as_int64
            elif item.number == _ATTR_S and item.wire_type is WireType.LEN:
                # Over-limit value payloads are never buffered: the attribute
                # is reported as unavailable instead of failing the parse.
                if self._over_limit(item):
                    oversized_field = True
                else:
                    scalar_bytes = cursor.payload(item)
            elif item.number == _ATTR_FLOATS:
                if item.wire_type is WireType.I32:
                    floats.append(self._fixed32_float(item))
                elif item.wire_type is WireType.LEN:
                    if self._over_limit(item):
                        oversized_field = True
                    else:
                        floats.extend(self._packed_floats(cursor, item))
            elif item.number == _ATTR_INTS:
                if item.wire_type is WireType.VARINT:
                    ints.append(item.as_int64)
                elif item.wire_type is WireType.LEN:
                    if self._over_limit(item):
                        oversized_field = True
                    else:
                        ints.extend(_unpack_varints(cursor.payload(item)))
            elif item.number == _ATTR_STRINGS and item.wire_type is WireType.LEN:
                if self._over_limit(item):
                    oversized_field = True
                else:
                    strings.append(_decode_attr_text(cursor.payload(item)))
            elif item.number == _ATTR_GRAPH and item.wire_type is WireType.LEN:
                graph_fields.append((item, False))
            elif item.number == _ATTR_GRAPHS and item.wire_type is WireType.LEN:
                graph_fields.append((item, True))
            elif item.number == _ATTR_TENSOR and item.wire_type is WireType.LEN:
                tensor_fields.append((item, False))
            elif item.number == _ATTR_TENSORS and item.wire_type is WireType.LEN:
                tensor_fields.append((item, True))
            elif item.number in (
                _ATTR_TP,
                _ATTR_TYPE_PROTOS,
                _ATTR_REF_NAME,
                _ATTR_SPARSE_TENSOR,
                _ATTR_SPARSE_TENSORS,
            ):
                unsupported_field = True

        subgraph_ids: list[str] = []
        repeated_index = 0
        for item, repeated in graph_fields:
            index = repeated_index if repeated else None
            subgraph_id = attribute_subgraph_id(graph_id, node_id, name, index)
            self._graph(cursor.submessage(item), subgraph_id, parent_node_id=node_id)
            subgraph_ids.append(subgraph_id)
            repeated_index += 1 if repeated else 0

        tensors: list[TensorIndex] = []
        for ordinal, (item, _repeated) in enumerate(tensor_fields):
            tensors.append(
                self._tensor(
                    cursor.submessage(item),
                    graph_id,
                    ordinal,
                    owner=f"{node_id}:{name}",
                    span=ByteRange(item.offset, item.length),
                )
            )

        if not name:
            self.log.add(
                "onnx.nameless-attribute",
                Severity.WARNING,
                "An attribute without a name cannot be represented.",
                node_id,
            )
            return None, name, subgraph_ids, tensors
        if oversized_field:
            self.log.add(
                "onnx.attribute-too-large",
                Severity.WARNING,
                f"Attribute {name!r} carries a value larger than the "
                f"{self._limits.max_string_bytes}-byte import limit; the "
                "attribute is listed by name only and its value survives "
                "unchanged in the source artifact.",
                node_id,
            )
            return None, name, subgraph_ids, tensors
        if unsupported_field:
            self.log.add(
                "onnx.unsupported-attribute",
                Severity.WARNING,
                f"Attribute {name!r} uses a sparse-tensor, type-proto, or "
                "function-reference value, which the IR does not model yet; "
                "its value is not imported.",
                node_id,
            )
            return None, name, subgraph_ids, tensors

        kind = _ATTR_TYPE_TO_KIND.get(declared_type)
        if kind is None:
            # Older exporters may omit the type field; infer it from which
            # value field is present. Singular wins over repeated because a
            # single observed element cannot distinguish INTS from INT.
            if tensor_fields:
                kind = AttrKind.TENSORS if tensor_fields[0][1] else AttrKind.TENSOR
            elif graph_fields:
                kind = AttrKind.GRAPHS if graph_fields[0][1] else AttrKind.GRAPH
            elif scalar_float is not None:
                kind = AttrKind.FLOAT
            elif scalar_int is not None:
                kind = AttrKind.INT
            elif scalar_bytes is not None:
                kind = AttrKind.STRING
            elif floats:
                kind = AttrKind.FLOATS
            elif ints:
                kind = AttrKind.INTS
            elif strings:
                kind = AttrKind.STRINGS
        if kind is None:
            self.log.add(
                "onnx.unknown-attribute-type",
                Severity.WARNING,
                f"Attribute {name!r} declares type {declared_type}, which is "
                "not a representable attribute type.",
                node_id,
            )
            return None, name, subgraph_ids, tensors

        value: int | float | str | tuple[int, ...] | tuple[float, ...] | tuple[str, ...]
        if kind is AttrKind.INT:
            value = scalar_int if scalar_int is not None else 0
        elif kind is AttrKind.FLOAT:
            value = scalar_float if scalar_float is not None else 0.0
        elif kind is AttrKind.STRING:
            value = _decode_attr_text(scalar_bytes or b"")
        elif kind is AttrKind.TENSOR or kind is AttrKind.TENSORS:
            ids = tuple(tensor.id for tensor in tensors)
            if kind is AttrKind.TENSOR:
                if not ids:
                    self.log.add(
                        "onnx.malformed-attribute",
                        Severity.WARNING,
                        f"Tensor attribute {name!r} carries no tensor.",
                        node_id,
                    )
                    return None, name, subgraph_ids, tensors
                value = ids[0]
            else:
                value = ids
        elif kind is AttrKind.GRAPH or kind is AttrKind.GRAPHS:
            if kind is AttrKind.GRAPH:
                if not subgraph_ids:
                    self.log.add(
                        "onnx.malformed-attribute",
                        Severity.WARNING,
                        f"Graph attribute {name!r} carries no graph.",
                        node_id,
                    )
                    return None, name, subgraph_ids, tensors
                value = subgraph_ids[0]
            else:
                value = tuple(subgraph_ids)
        elif kind is AttrKind.INTS:
            value = tuple(ints)
        elif kind is AttrKind.FLOATS:
            value = tuple(floats)
        else:
            value = tuple(strings)
        try:
            attribute = Attribute(name=name, kind=kind, value=value)
        except IrError as error:  # pragma: no cover - defensive: kinds match
            self.log.add(
                "onnx.malformed-attribute", Severity.WARNING, str(error), node_id
            )
            return None, name, subgraph_ids, tensors
        return attribute, name, subgraph_ids, tensors

    def _over_limit(self, item: WireField) -> bool:
        """Whether a value payload exceeds the per-field buffering limit."""
        return item.length > self._limits.max_string_bytes

    def _fixed32_float(self, item: WireField) -> float:
        """Read a fixed32 field's payload as an IEEE-754 float."""
        raw = self._reader.read(item.offset, 4)
        return float(struct.unpack("<f", raw)[0])

    def _packed_floats(self, cursor: MessageCursor, item: WireField) -> list[float]:
        payload = cursor.payload(item)
        if len(payload) % 4:
            raise MalformedProtobufError(
                f"packed float payload of {len(payload)} bytes is not a "
                "multiple of four"
            )
        return [float(value[0]) for value in struct.iter_unpack("<f", payload)]

    # -- tensors ----------------------------------------------------------

    def _tensor(
        self,
        cursor: MessageCursor,
        graph_id: str,
        ordinal: int,
        owner: str | None = None,
        span: ByteRange | None = None,
    ) -> TensorIndex:
        name = ""
        data_type = 0
        dims: list[int] = []
        raw_field: WireField | None = None
        typed_field_count = 0
        external_entries: list[tuple[str, str]] = []
        data_location = 0

        for item in cursor:
            if item.number == _TENSOR_DIMS:
                if item.wire_type is WireType.VARINT:
                    dims.append(item.as_int64)
                elif item.wire_type is WireType.LEN:
                    dims.extend(cursor.packed_varints(item))
            elif item.number == _TENSOR_DATA_TYPE and item.wire_type is WireType.VARINT:
                data_type = item.as_int32
            elif (
                item.number == _TENSOR_DATA_LOCATION
                and item.wire_type is WireType.VARINT
            ):
                data_location = item.as_int32
            elif item.wire_type is not WireType.LEN:
                continue
            elif item.number == _TENSOR_NAME:
                name = cursor.text(item)
            elif item.number == _TENSOR_RAW_DATA:
                raw_field = item
            elif item.number == _TENSOR_EXTERNAL_DATA:
                external_entries.append(self._string_entry(cursor, item))
            elif item.number in _TYPED_DATA_FIELDS:
                typed_field_count += 1

        element_type = element_type_for(data_type)
        if data_type not in (0,) and element_type.name.startswith("UNKNOWN"):
            self.log.add(
                "onnx.unknown-element-type",
                Severity.WARNING,
                f"Element type code {data_type} is not known to this build; the "
                "tensor is indexed but its size cannot be derived.",
                name or f"{graph_id}#{ordinal}",
            )

        label = name or (f"{owner}" if owner else f"initializer:{ordinal}")
        tensor_id = (
            f"t:{graph_id}#attr:{encode_segment(label)}"
            if owner
            else initializer_id(graph_id, label)
        )
        error: str | None = None
        dims_tuple = tuple(dims)
        if any(dim < 0 for dim in dims_tuple):
            error = f"tensor declares a negative dimension: {dims_tuple}"
            self.log.add("onnx.negative-dimension", Severity.ERROR, error, tensor_id)
        elif _element_count(dims_tuple) > _MAX_ELEMENT_COUNT:
            error = f"tensor declares an implausible element count for {dims_tuple}"
            self.log.add("onnx.implausible-shape", Severity.ERROR, error, tensor_id)

        storage = TensorStorage.ABSENT
        payload: ByteRange | None = None
        external: ExternalDataReference | None = None
        if data_location == _DATA_LOCATION_EXTERNAL or external_entries:
            storage = TensorStorage.EXTERNAL
            external = self._external_reference(
                external_entries, tensor_id, element_type, dims_tuple
            )
            if external.error is not None and error is None:
                error = external.error
        elif raw_field is not None:
            storage = TensorStorage.EMBEDDED_RAW
            payload = ByteRange(raw_field.offset, raw_field.length)
            expected = _expected_bytes(element_type, dims_tuple)
            if expected is not None and expected != raw_field.length:
                error = (
                    f"embedded payload is {raw_field.length} bytes but the declared "
                    f"shape and type imply {expected}"
                )
                self.log.add(
                    "onnx.payload-length-mismatch", Severity.ERROR, error, tensor_id
                )
        elif typed_field_count:
            storage = TensorStorage.EMBEDDED_TYPED
            self.log.add(
                "onnx.typed-tensor-field",
                Severity.INFO,
                "Tensor uses a typed data field rather than raw bytes, so reading "
                "it requires decoding the field instead of a range read.",
                tensor_id,
            )

        tensor = TensorIndex(
            id=tensor_id,
            name=name,
            graph_id=graph_id,
            element_type=element_type,
            dims=dims_tuple,
            storage=storage,
            payload=payload,
            external=external,
            error=error,
            message_span=span if storage is TensorStorage.EMBEDDED_TYPED else None,
        )
        if tensor_id in self._tensors:
            self.log.add(
                "onnx.duplicate-tensor-id",
                Severity.WARNING,
                "Two tensors derive the same identifier; lookups and edits "
                "resolve to the later definition, while both stay listed in "
                "their graph.",
                tensor_id,
            )
        self._tensors[tensor_id] = tensor
        return tensor

    def _external_reference(
        self,
        entries: Sequence[tuple[str, str]],
        tensor_id: str,
        element_type: ElementType,
        dims: tuple[int, ...],
    ) -> ExternalDataReference:
        fields = dict(entries)
        location = fields.get("location", "")
        checksum = fields.get("checksum") or None

        offset, offset_error = _parse_int(fields.get("offset", "0"), "offset")
        length, length_error = (
            _parse_int(fields["length"], "length")
            if "length" in fields
            else (None, None)
        )
        metadata_error = offset_error or length_error
        if metadata_error is not None:
            self.log.add(
                "onnx.external-metadata-invalid",
                Severity.ERROR,
                metadata_error,
                tensor_id,
            )
            return ExternalDataReference(
                location, offset or 0, length, checksum, None, metadata_error
            )

        try:
            resolved = resolve_within(self._external_root, location)
        except UnsafePathError as unsafe:
            self.log.add(
                "onnx.external-path-unsafe", Severity.ERROR, str(unsafe), tensor_id
            )
            return ExternalDataReference(
                location, offset or 0, length, checksum, None, str(unsafe)
            )

        size = self._external_size(resolved)
        if size is None:
            message = f"external data file is missing: {location!r}"
            self.log.add(
                "onnx.external-file-missing", Severity.ERROR, message, tensor_id
            )
            return ExternalDataReference(
                location, offset or 0, length, checksum, None, message
            )

        self._external_files.setdefault(resolved, None)
        resolved_length = size - (offset or 0) if length is None else length
        assert offset is not None
        if offset < 0 or resolved_length < 0 or offset + resolved_length > size:
            message = (
                f"external range [{offset}, {offset + resolved_length}) is outside "
                f"{location!r}, which is {size} bytes"
            )
            self.log.add(
                "onnx.external-range-out-of-bounds",
                Severity.ERROR,
                message,
                tensor_id,
            )
            return ExternalDataReference(
                location, offset, length, checksum, resolved, message
            )

        expected = _expected_bytes(element_type, dims)
        if expected is not None and expected != resolved_length:
            message = (
                f"external payload is {resolved_length} bytes but the declared "
                f"shape and type imply {expected}"
            )
            self.log.add(
                "onnx.external-length-mismatch", Severity.ERROR, message, tensor_id
            )
            return ExternalDataReference(
                location, offset, resolved_length, checksum, resolved, message
            )

        reference = ExternalDataReference(
            location, offset, resolved_length, checksum, resolved, None
        )
        if self._verify_checksums and checksum:
            failure = _verify_checksum(reference, checksum)
            if failure is not None:
                self.log.add(
                    "onnx.checksum-mismatch", Severity.ERROR, failure, tensor_id
                )
                return ExternalDataReference(
                    location, offset, resolved_length, checksum, resolved, failure
                )
        return reference

    def _external_size(self, path: Path) -> int | None:
        cached = self._external_sizes.get(path)
        if cached is not None:
            return cached
        try:
            size = os.stat(path).st_size
        except OSError:
            return None
        self._external_sizes[path] = size
        return size

    # -- values -----------------------------------------------------------

    def _value_info(self, cursor: MessageCursor, graph_id: str) -> ValueIndex:
        name = ""
        element_type: ElementType | None = None
        shape: tuple[Dimension, ...] | None = None
        for item in cursor:
            if item.wire_type is not WireType.LEN:
                continue
            if item.number == _VALUE_INFO_NAME:
                name = cursor.text(item)
            elif item.number == _VALUE_INFO_TYPE:
                element_type, shape = self._type_proto(cursor.submessage(item))
        return ValueIndex(
            id=value_id(graph_id, name),
            name=name,
            element_type=element_type,
            shape=shape,
        )

    def _type_proto(
        self, cursor: MessageCursor
    ) -> tuple[ElementType | None, tuple[Dimension, ...] | None]:
        for item in cursor:
            if item.number == _TYPE_TENSOR_TYPE and item.wire_type is WireType.LEN:
                return self._tensor_type(cursor.submessage(item))
        return None, None

    def _tensor_type(
        self, cursor: MessageCursor
    ) -> tuple[ElementType | None, tuple[Dimension, ...] | None]:
        element_type: ElementType | None = None
        shape: tuple[Dimension, ...] | None = None
        for item in cursor:
            if (
                item.number == _TENSOR_TYPE_ELEM_TYPE
                and item.wire_type is WireType.VARINT
            ):
                element_type = element_type_for(item.as_int32)
            elif item.number == _TENSOR_TYPE_SHAPE and item.wire_type is WireType.LEN:
                shape = self._shape(cursor.submessage(item))
        return element_type, shape

    def _shape(self, cursor: MessageCursor) -> tuple[Dimension, ...]:
        dims: list[Dimension] = []
        for item in cursor:
            if item.number != _SHAPE_DIM or item.wire_type is not WireType.LEN:
                continue
            inner = cursor.submessage(item)
            dim: Dimension = None
            for entry in inner:
                if entry.number == _DIM_VALUE and entry.wire_type is WireType.VARINT:
                    dim = entry.as_int64
                elif entry.number == _DIM_PARAM and entry.wire_type is WireType.LEN:
                    dim = inner.text(entry)
            dims.append(dim)
        return tuple(dims)


def _unpack_varints(payload: bytes) -> list[int]:
    """Decode a packed repeated-int64 payload into signed integers."""
    values: list[int] = []
    position = 0
    while position < len(payload):
        result = 0
        shift = 0
        while True:
            if position >= len(payload) or shift > 63:
                raise MalformedProtobufError("truncated packed varint payload")
            byte = payload[position]
            position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        result &= (1 << 64) - 1
        values.append(result - (1 << 64) if result & (1 << 63) else result)
    return values


def _decode_attr_text(payload: bytes) -> str:
    """Decode attribute bytes as text.

    ONNX stores string attributes as ``bytes`` with no declared encoding;
    non-UTF-8 payloads are carried with replacement characters rather than
    failing the whole import.
    """
    return payload.decode("utf-8", errors="replace")


def _derive_node_id(
    graph_id: str,
    parts: _NodeParts,
    outputs: tuple[str, ...],
    ordinal: int,
) -> tuple[str, NodeIdStability]:
    first_output = next((item for item in outputs if item), "")
    return derive_node_id(
        graph_id, name=parts.name, first_output=first_output, ordinal=ordinal
    )


def _element_count(dims: tuple[int, ...]) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


def _expected_bytes(element_type: ElementType, dims: tuple[int, ...]) -> int | None:
    if not element_type.is_fixed_width:
        return None
    count = _element_count(dims)
    if count < 0 or count > _MAX_ELEMENT_COUNT:
        return None
    return (count * element_type.bits + 7) // 8


def _parse_int(text: str, label: str) -> tuple[int | None, str | None]:
    stripped = text.strip()
    try:
        value = int(stripped, 10)
    except ValueError:
        return None, f"external data {label} is not an integer: {text!r}"
    if value < 0:
        return None, f"external data {label} is negative: {value}"
    return value, None


def _verify_checksum(reference: ExternalDataReference, checksum: str) -> str | None:
    algorithm, _, expected = checksum.rpartition(":")
    algorithm = algorithm or "sha256"
    try:
        digest = hashlib.new(algorithm)
    except ValueError:
        return f"unsupported checksum algorithm: {algorithm!r}"
    assert reference.resolved_path is not None
    assert reference.length is not None
    with open(reference.resolved_path, "rb") as handle:
        handle.seek(reference.offset)
        remaining = reference.length
        while remaining > 0:
            chunk = handle.read(min(remaining, 1 << 20))
            if not chunk:  # pragma: no cover - the file shrank after validation
                break
            digest.update(chunk)
            remaining -= len(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        return f"checksum mismatch: expected {expected}, computed {actual}"
    return None


def index_model(
    path: Path | str,
    *,
    limits: ScanLimits | None = None,
    external_root: Path | str | None = None,
    verify_checksums: bool = False,
    block_size: int | None = None,
    token: CancellationToken | None = None,
) -> ModelIndex:
    """Index an ONNX model without materializing any tensor.

    ``external_root`` defaults to the model's own directory and confines every
    ``external_data`` location. Model and external files are streamed through
    SHA-256 for artifact identity. ``verify_checksums`` separately controls
    comparison with producer-declared checksum metadata.
    """
    model_path = Path(path)
    root = Path(external_root) if external_root is not None else model_path.parent
    model_hash = hash_file(model_path, token=token)
    with ArtifactReader(
        model_path, block_size=block_size or DEFAULT_BLOCK_SIZE
    ) as reader:
        indexer = _Indexer(
            reader,
            limits=limits or ScanLimits(),
            external_root=root,
            verify_checksums=verify_checksums,
            token=token,
        )
        try:
            index = indexer.run(model_hash)
        except MalformedProtobufError as error:
            raise OnnxIndexError(
                f"{model_path.name} is not a readable ONNX model: {error}"
            ) from error
    if not index.external_files:
        return index
    resolved_root = root.resolve()
    external_hashes = tuple(
        (
            item.relative_to(resolved_root).as_posix(),
            hash_file(item, token=token),
        )
        for item in sorted(index.external_files)
    )
    manifest = json.dumps(
        {"model": model_hash, "external": external_hashes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return replace(
        index,
        content_hash=f"sha256:{hashlib.sha256(manifest).hexdigest()}",
        external_hashes=external_hashes,
    )


def read_tensor_bytes(
    index: ModelIndex,
    tensor: TensorIndex | str,
    *,
    external_root: Path | None = None,
) -> bytes:
    """Materialize one tensor's bytes. This is the opt-in, non-lazy path.

    Raises :class:`OnnxIndexError` when the tensor was not indexed as a
    readable byte range, which keeps the failure explicit rather than returning
    a silently wrong buffer.
    """
    entry = index.tensor(tensor) if isinstance(tensor, str) else tensor
    if not entry.is_readable:
        raise OnnxIndexError(
            f"tensor {entry.id!r} is not readable as a byte range "
            f"({entry.storage.value}{': ' + entry.error if entry.error else ''})"
        )
    if entry.storage is TensorStorage.EMBEDDED_RAW:
        assert entry.payload is not None
        with ArtifactReader(index.path) as reader:
            return _read_range(reader, entry.payload.offset, entry.payload.length)
    assert entry.external is not None
    reference = entry.external
    assert reference.resolved_path is not None
    assert reference.length is not None
    target = reference.resolved_path
    if external_root is not None:
        target = resolve_within(external_root, reference.location)
    with ArtifactReader(target) as reader:
        return _read_range(reader, reference.offset, reference.length)


def _read_range(reader: ArtifactReader, offset: int, length: int) -> bytes:
    """Read one payload in bounded chunks.

    A single :meth:`ArtifactReader.read` refuses anything above
    ``MAX_SINGLE_READ``; chunking keeps legitimate multi-gigabyte initializers
    servable, matching how :class:`~nneditor.storage.store.TensorStore` reads.
    """
    if length <= _TENSOR_READ_CHUNK:
        return reader.read(offset, length)
    chunks: list[bytes] = []
    position = 0
    while position < length:
        step = min(_TENSOR_READ_CHUNK, length - position)
        chunks.append(reader.read(offset + position, step))
        position += step
    return b"".join(chunks)
