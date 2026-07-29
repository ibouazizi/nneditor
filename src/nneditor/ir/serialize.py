"""Deterministic IR document serialization (task P1.1).

One document, one byte sequence: field order is fixed, map keys are sorted,
floats round-trip exactly through JSON's shortest-repr rules, and nothing
optional is written when it holds its default. Determinism is a contract —
revision identifiers hash these bytes and golden tests diff them.

Reading follows the spec's read plan: a same-major/newer-minor document is
read in degraded mode with unrecognized top-level fields preserved under the
``x-nneditor.unknown`` extension; an unsupported version is refused with the
plan's reason rather than a parse error.
"""

from __future__ import annotations

import json
from typing import cast

from nneditor.diagnostics import Diagnostic, Severity
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
)
from nneditor.ir.core import (
    ArtifactRef,
    Attribute,
    AttrKind,
    CapabilityNote,
    Dim,
    Document,
    ExternalRef,
    Graph,
    IrError,
    JsonValue,
    Node,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import NodeIdStability
from nneditor.ir.schema import (
    UNKNOWN_FIELD_NAMESPACE,
    ReadStrategy,
    SchemaVersion,
    plan_read,
)

__all__ = [
    "document_from_bytes",
    "document_from_json",
    "document_to_bytes",
    "document_to_json",
]

_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "artifact_kind",
        "capabilities",
        "graphs",
        "entry_graph",
        "tensors",
        "provenance",
        "diagnostics",
        "capability_notes",
        "extensions",
    }
)

_UNKNOWN_DIM = "?"
"""Serialized marker for a known-rank, unknown-extent dimension.

A literal ``null`` cannot be used because JSON has no distinct marker left for
"no shape at all", and a symbolic dimension named ``?`` is impossible: symbolic
names come from source formats where ``?`` is not an identifier.
"""


def _dim_to_json(dim: Dim) -> int | str:
    if dim is None:
        return _UNKNOWN_DIM
    return dim


def _dim_from_json(raw: JsonValue) -> Dim:
    if raw == _UNKNOWN_DIM:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw
    raise IrError(f"malformed dimension: {raw!r}")


def _value_to_json(value: Value) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {"id": value.id, "name": value.name}
    if value.element_type is not None:
        data["element_type"] = value.element_type
    if value.shape is not None:
        data["shape"] = [_dim_to_json(dim) for dim in value.shape]
    return data


def _value_from_json(data: dict[str, JsonValue]) -> Value:
    shape = data.get("shape")
    return Value(
        id=_expect_str(data, "id"),
        name=_expect_str(data, "name"),
        element_type=cast(str | None, data.get("element_type")),
        shape=tuple(_dim_from_json(dim) for dim in cast(list[JsonValue], shape))
        if shape is not None
        else None,
    )


def _attribute_to_json(attribute: Attribute) -> dict[str, JsonValue]:
    value: JsonValue
    if isinstance(attribute.value, tuple):
        value = list(attribute.value)
    else:
        value = attribute.value
    return {"name": attribute.name, "kind": attribute.kind.value, "value": value}


def _attribute_from_json(data: dict[str, JsonValue]) -> Attribute:
    kind = AttrKind(_expect_str(data, "kind"))
    raw = data.get("value")
    value = tuple(raw) if isinstance(raw, list) else raw
    return Attribute(
        name=_expect_str(data, "name"),
        kind=kind,
        value=cast("int | float | str | tuple[int, ...]", value),
    )


def _node_to_json(node: Node) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "id": node.id,
        "id_stability": node.id_stability.value,
        "op_type": node.op_type,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
    }
    if node.domain:
        data["domain"] = node.domain
    if node.overload:
        data["overload"] = node.overload
    if node.attributes:
        data["attributes"] = [_attribute_to_json(item) for item in node.attributes]
    if node.subgraphs:
        data["subgraphs"] = list(node.subgraphs)
    if node.source_name is not None:
        data["source_name"] = node.source_name
    if node.source_location is not None:
        data["source_location"] = node.source_location
    if node.has_side_effects is not None:
        data["has_side_effects"] = node.has_side_effects
    if node.mutates_inputs is not None:
        data["mutates_inputs"] = node.mutates_inputs
    return data


def _node_from_json(data: dict[str, JsonValue]) -> Node:
    return Node(
        id=_expect_str(data, "id"),
        id_stability=NodeIdStability(_expect_str(data, "id_stability")),
        op_type=_expect_str(data, "op_type"),
        inputs=tuple(cast(list[str], data.get("inputs", []))),
        outputs=tuple(cast(list[str], data.get("outputs", []))),
        domain=cast(str, data.get("domain", "")),
        overload=cast(str, data.get("overload", "")),
        attributes=tuple(
            _attribute_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("attributes", []))
        ),
        subgraphs=tuple(cast(list[str], data.get("subgraphs", []))),
        source_name=cast(str | None, data.get("source_name")),
        source_location=cast(str | None, data.get("source_location")),
        has_side_effects=cast(bool | None, data.get("has_side_effects")),
        mutates_inputs=cast(bool | None, data.get("mutates_inputs")),
    )


def _graph_to_json(graph: Graph) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "id": graph.id,
        "name": graph.name,
        "values": [_value_to_json(item) for item in graph.values],
        "nodes": [_node_to_json(item) for item in graph.nodes],
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
    }
    if graph.initializers:
        data["initializers"] = list(graph.initializers)
    if graph.parent_node is not None:
        data["parent_node"] = graph.parent_node
    return data


def _graph_from_json(data: dict[str, JsonValue]) -> Graph:
    return Graph(
        id=_expect_str(data, "id"),
        name=_expect_str(data, "name"),
        values=(
            _value_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("values", []))
        ),
        nodes=(
            _node_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("nodes", []))
        ),
        inputs=cast(list[str], data.get("inputs", [])),
        outputs=cast(list[str], data.get("outputs", [])),
        initializers=cast(list[str], data.get("initializers", [])),
        parent_node=cast(str | None, data.get("parent_node")),
    )


def _tensor_to_json(tensor: TensorRef) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "id": tensor.id,
        "element_type": tensor.element_type,
        "dims": list(tensor.dims),
        "storage": tensor.storage.value,
    }
    if tensor.payload is not None:
        data["payload"] = {
            "offset": tensor.payload.offset,
            "length": tensor.payload.length,
        }
    if tensor.external is not None:
        external: dict[str, JsonValue] = {
            "location": tensor.external.location,
            "offset": tensor.external.offset,
        }
        if tensor.external.length is not None:
            external["length"] = tensor.external.length
        if tensor.external.checksum is not None:
            external["checksum"] = tensor.external.checksum
        data["external"] = external
    if tensor.typed_span is not None:
        data["typed_span"] = {
            "offset": tensor.typed_span.offset,
            "length": tensor.typed_span.length,
        }
    return data


def _payload_range_from_json(raw: JsonValue, context: str) -> PayloadRange:
    data = _expect_object(raw, context)
    return PayloadRange(
        offset=_expect_int(data, "offset"),
        length=_expect_int(data, "length"),
    )


def _tensor_from_json(data: dict[str, JsonValue]) -> TensorRef:
    raw_payload = data.get("payload")
    raw_external = data.get("external")
    raw_typed = data.get("typed_span")
    external = None
    if raw_external is not None:
        external_data = _expect_object(raw_external, "tensor 'external' reference")
        external = ExternalRef(
            location=_expect_str(external_data, "location"),
            offset=_expect_int(external_data, "offset"),
            length=_expect_optional_int(external_data, "length"),
            checksum=_expect_optional_str(external_data, "checksum"),
        )
    return TensorRef(
        id=_expect_str(data, "id"),
        element_type=_expect_str(data, "element_type"),
        dims=tuple(cast(list[int], data.get("dims", []))),
        storage=Storage(_expect_str(data, "storage")),
        payload=_payload_range_from_json(raw_payload, "tensor 'payload' range")
        if raw_payload is not None
        else None,
        external=external,
        typed_span=_payload_range_from_json(raw_typed, "tensor 'typed_span' range")
        if raw_typed is not None
        else None,
    )


def _provenance_to_json(entry: ProvenanceEntry) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "operation": entry.operation,
        "tool_version": entry.tool_version,
    }
    if entry.target is not None:
        data["target"] = entry.target
    if entry.parameters:
        data["parameters"] = {key: value for key, value in entry.parameters}
    if entry.dependency_versions:
        data["dependency_versions"] = dict(entry.dependency_versions)
    if entry.source_artifact is not None:
        data["source_artifact"] = entry.source_artifact
    if entry.revision is not None:
        data["revision"] = entry.revision
    return data


def _provenance_from_json(data: dict[str, JsonValue]) -> ProvenanceEntry:
    parameters = cast(dict[str, str], data.get("parameters", {}))
    dependencies = cast(dict[str, str], data.get("dependency_versions", {}))
    return ProvenanceEntry(
        operation=_expect_str(data, "operation"),
        tool_version=_expect_str(data, "tool_version"),
        target=cast(str | None, data.get("target")),
        parameters=tuple(sorted(parameters.items())),
        dependency_versions=tuple(sorted(dependencies.items())),
        source_artifact=cast(str | None, data.get("source_artifact")),
        revision=cast(str | None, data.get("revision")),
    )


def _expect_str(data: dict[str, JsonValue] | None, key: str) -> str:
    if data is None or not isinstance(data.get(key), str):
        raise IrError(f"missing or malformed field {key!r}")
    return cast(str, data[key])


def _expect_int(data: dict[str, JsonValue], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise IrError(f"missing or malformed field {key!r}")
    return value


def _expect_optional_int(data: dict[str, JsonValue], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise IrError(f"missing or malformed field {key!r}")
    return value


def _expect_optional_str(data: dict[str, JsonValue], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise IrError(f"missing or malformed field {key!r}")
    return value


def _expect_object(value: JsonValue, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise IrError(f"{context} must be a JSON object")
    return value


def document_to_json(document: Document) -> dict[str, JsonValue]:
    """Serialize a document to a JSON-compatible dict with fixed field order."""
    return {
        "schema_version": str(document.schema_version),
        "source": {
            "path": document.source.path,
            "content_hash": document.source.content_hash,
            "byte_size": document.source.byte_size,
        },
        "artifact_kind": document.artifact_kind.value,
        "capabilities": [
            {
                "capability": status.capability.value,
                "availability": status.availability.value,
                "reason": status.reason,
            }
            # Sorted so serialization does not depend on declaration order.
            for status in sorted(
                document.capabilities.values(), key=lambda s: s.capability.value
            )
        ],
        "entry_graph": document.entry_graph,
        "graphs": [_graph_to_json(graph) for graph in document.graphs.values()],
        "tensors": [_tensor_to_json(item) for item in document.tensors.values()],
        "provenance": [_provenance_to_json(item) for item in document.provenance],
        "diagnostics": [
            {
                "code": item.code,
                "severity": item.severity.value,
                "message": item.message,
                **({"target": item.target} if item.target is not None else {}),
            }
            for item in document.diagnostics
        ],
        "capability_notes": [
            {
                "entity_id": note.entity_id,
                "capability": note.capability.value,
                "availability": note.availability.value,
                "reason": note.reason,
            }
            for note in document.capability_notes
        ],
        "extensions": {namespace: value for namespace, value in document.extensions},
    }


def document_to_bytes(document: Document) -> bytes:
    """Canonical bytes: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        document_to_json(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def document_from_json(data: dict[str, JsonValue]) -> Document:
    """Rebuild a document, following the schema read plan."""
    version = SchemaVersion.parse(_expect_str(data, "schema_version"))
    plan = plan_read(version)
    if plan.strategy is ReadStrategy.UNSUPPORTED:
        raise IrError(f"cannot read document version {version}: {plan.reason}")
    if plan.strategy is ReadStrategy.MIGRATE:  # pragma: no cover - no majors yet
        raise IrError(
            f"document version {version} needs migrations that are not "
            "implemented in this build"
        )

    extensions_data = _expect_object(
        data.get("extensions", {}), "document field 'extensions'"
    )
    extensions = list(extensions_data.items())
    if plan.strategy is ReadStrategy.DEGRADED:
        unknown = {key: data[key] for key in data if key not in _DOCUMENT_FIELDS}
        if unknown:
            existing = extensions_data.get(UNKNOWN_FIELD_NAMESPACE)
            if isinstance(existing, dict):
                # The document already parks fields from an earlier degraded
                # read; merge rather than appending a duplicate namespace the
                # document would then refuse. Freshly-read top-level values
                # win a key collision — they are what this file carries now.
                unknown = {**existing, **unknown}
            parked = cast(JsonValue, unknown)
            if UNKNOWN_FIELD_NAMESPACE in extensions_data:
                extensions = [
                    (
                        namespace,
                        parked if namespace == UNKNOWN_FIELD_NAMESPACE else value,
                    )
                    for namespace, value in extensions
                ]
            else:
                extensions.append((UNKNOWN_FIELD_NAMESPACE, parked))

    raw_source = data.get("source")
    if raw_source is None:
        raise IrError("document is missing its source artifact reference")
    source_data = _expect_object(raw_source, "document field 'source'")
    byte_size = _expect_optional_int(source_data, "byte_size")
    return Document(
        schema_version=version,
        source=ArtifactRef(
            path=_expect_str(source_data, "path"),
            content_hash=_expect_str(source_data, "content_hash"),
            byte_size=byte_size if byte_size is not None else 0,
        ),
        artifact_kind=ArtifactKind(_expect_str(data, "artifact_kind")),
        capabilities=(
            CapabilityStatus(
                capability=Capability(_expect_str(item, "capability")),
                availability=Availability(_expect_str(item, "availability")),
                reason=_expect_str(item, "reason"),
            )
            for item in cast("list[dict[str, JsonValue]]", data.get("capabilities", []))
        ),
        entry_graph=_expect_str(data, "entry_graph"),
        graphs=(
            _graph_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("graphs", []))
        ),
        tensors=(
            _tensor_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("tensors", []))
        ),
        provenance=(
            _provenance_from_json(cast(dict[str, JsonValue], item))
            for item in cast(list[JsonValue], data.get("provenance", []))
        ),
        diagnostics=(
            Diagnostic(
                code=_expect_str(item, "code"),
                severity=Severity(_expect_str(item, "severity")),
                message=_expect_str(item, "message"),
                target=cast(str | None, item.get("target")),
            )
            for item in cast("list[dict[str, JsonValue]]", data.get("diagnostics", []))
        ),
        capability_notes=(
            CapabilityNote(
                entity_id=_expect_str(item, "entity_id"),
                capability=Capability(_expect_str(item, "capability")),
                availability=Availability(_expect_str(item, "availability")),
                reason=_expect_str(item, "reason"),
            )
            for item in cast(
                "list[dict[str, JsonValue]]", data.get("capability_notes", [])
            )
        ),
        extensions=extensions,
    )


def document_from_bytes(payload: bytes) -> Document:
    """Parse canonical (or any JSON) document bytes."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise IrError(f"document bytes are not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise IrError("document JSON must be an object")
    return document_from_json(cast(dict[str, JsonValue], data))
