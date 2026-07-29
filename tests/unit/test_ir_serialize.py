"""Tests for deterministic IR serialization (P1.1)."""

from __future__ import annotations

import json

import pytest

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
    Document,
    ExternalRef,
    Graph,
    IrError,
    Node,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import NodeIdStability
from nneditor.ir.schema import UNKNOWN_FIELD_NAMESPACE
from nneditor.ir.serialize import (
    document_from_bytes,
    document_from_json,
    document_to_bytes,
    document_to_json,
)


def full_capabilities(order: bool = False) -> tuple[CapabilityStatus, ...]:
    statuses = tuple(
        CapabilityStatus(capability, Availability.PARTIAL, "fixture reason")
        for capability in Capability
    )
    return tuple(reversed(statuses)) if order else statuses


def rich_document() -> Document:
    """A document exercising every serialized field at least once."""
    child = Graph(
        id="g:child",
        name="then_body",
        values=[Value(id="v:c", name="c", element_type="float32", shape=(1,))],
        nodes=[
            Node(
                id="n:const",
                id_stability=NodeIdStability.OUTPUT_DERIVED,
                op_type="Constant",
                inputs=(),
                outputs=("v:c",),
                attributes=(Attribute("value", AttrKind.TENSOR, "t:const"),),
            )
        ],
        outputs=["v:c"],
        parent_node="n:if",
    )
    main = Graph(
        id="g:main",
        name="main",
        values=[
            Value(
                id="v:in", name="in", element_type="float32", shape=("batch", None, 3)
            ),
            Value(id="v:w", name="w", element_type="float32", shape=(3,)),
            Value(id="v:out", name="out"),
            Value(id="v:cond", name="cond", element_type="bool", shape=()),
        ],
        nodes=[
            Node(
                id="n:if",
                id_stability=NodeIdStability.NAMED,
                op_type="If",
                inputs=("v:cond",),
                outputs=("v:out",),
                attributes=(Attribute("then_branch", AttrKind.GRAPH, "g:child"),),
                subgraphs=("g:child",),
                source_name="branch",
                source_location="model.py:10",
                has_side_effects=False,
                mutates_inputs=False,
            )
        ],
        inputs=["v:in", "v:cond"],
        outputs=["v:out"],
        initializers=["t:w"],
    )
    return Document(
        source=ArtifactRef(path="model.onnx", content_hash="sha256:ab", byte_size=42),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=full_capabilities(),
        graphs=[main, child],
        tensors=[
            TensorRef(
                id="t:w",
                element_type="float32",
                dims=(3,),
                storage=Storage.EMBEDDED_RAW,
                payload=PayloadRange(offset=100, length=12),
            ),
            TensorRef(
                id="t:const",
                element_type="float32",
                dims=(1,),
                storage=Storage.EXTERNAL,
                external=ExternalRef(
                    location="weights.bin", offset=0, length=4, checksum="sha256:cd"
                ),
            ),
        ],
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version="nneditor 1.0.0",
                target="g:main",
                parameters=(("mode", "safe"),),
                dependency_versions=(("flet", "0.86.4"),),
                source_artifact="sha256:ab",
                revision="rev:base",
            )
        ],
        diagnostics=[
            Diagnostic("onnx.custom-domain", Severity.WARNING, "custom op", "n:if")
        ],
        capability_notes=[
            CapabilityNote(
                entity_id="n:if",
                capability=Capability.EDITING,
                availability=Availability.UNAVAILABLE,
                reason="control flow is read-only in phase 1",
            )
        ],
        extensions=[("x-onnx.metadata_props", {"author": "test"})],
    )


def test_round_trip_preserves_everything() -> None:
    original = rich_document()
    restored = document_from_bytes(document_to_bytes(original))
    assert document_to_bytes(restored) == document_to_bytes(original)
    assert restored.entry_graph == "g:main"
    assert restored.main_graph.producer("v:out") == ("n:if", 0)
    assert restored.graphs["g:child"].parent_node == "n:if"
    assert restored.tensors["t:const"].external is not None
    assert restored.tensors["t:const"].external.checksum == "sha256:cd"
    assert restored.capability_notes == rich_document().capability_notes
    assert restored.diagnostics[0].code == "onnx.custom-domain"
    assert dict(restored.extensions)["x-onnx.metadata_props"] == {"author": "test"}


def test_serialization_is_deterministic_across_input_order() -> None:
    base = rich_document()
    reordered = Document(
        source=base.source,
        artifact_kind=base.artifact_kind,
        capabilities=full_capabilities(order=True),
        graphs=base.graphs.values(),
        tensors=base.tensors.values(),
        provenance=base.provenance,
        diagnostics=base.diagnostics,
        capability_notes=base.capability_notes,
        extensions=base.extensions,
    )
    assert document_to_bytes(reordered) == document_to_bytes(base)


def test_symbolic_and_unknown_dimensions_round_trip() -> None:
    restored = document_from_bytes(document_to_bytes(rich_document()))
    shape = restored.main_graph.value("v:in").shape
    assert shape == ("batch", None, 3)


def test_newer_minor_reads_degraded_and_parks_unknown_fields() -> None:
    data = document_to_json(rich_document())
    data["schema_version"] = "1.99"
    data["novel_field"] = {"future": True}
    restored = document_from_json(data)
    parked = dict(restored.extensions)[UNKNOWN_FIELD_NAMESPACE]
    assert parked == {"novel_field": {"future": True}}


def test_newer_major_is_refused() -> None:
    data = document_to_json(rich_document())
    data["schema_version"] = "2.0"
    with pytest.raises(IrError, match="cannot read"):
        document_from_json(data)


def test_malformed_bytes_are_refused() -> None:
    with pytest.raises(IrError, match="not valid JSON"):
        document_from_bytes(b"nope{")
    with pytest.raises(IrError, match="must be an object"):
        document_from_bytes(b"[1]")


def test_malformed_dimension_is_refused() -> None:
    data = document_to_json(rich_document())
    graphs = data["graphs"]
    assert isinstance(graphs, list)
    graph = graphs[0]
    assert isinstance(graph, dict)
    values = graph["values"]
    assert isinstance(values, list)
    entry = values[0]
    assert isinstance(entry, dict)
    entry["shape"] = [True]
    with pytest.raises(IrError, match="malformed dimension"):
        document_from_json(data)


def test_missing_required_field_is_refused() -> None:
    data = document_to_json(rich_document())
    del data["entry_graph"]
    with pytest.raises(IrError, match="entry_graph"):
        document_from_json(data)


def test_canonical_bytes_are_stable_json() -> None:
    payload = document_to_bytes(rich_document())
    parsed = json.loads(payload)
    assert parsed["schema_version"] == "1.1"
    assert payload == json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
