"""Tests for the typed IR core (P1.1)."""

from __future__ import annotations

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


def full_capabilities() -> tuple[CapabilityStatus, ...]:
    return tuple(
        CapabilityStatus(capability, Availability.AVAILABLE, "test fixture")
        for capability in Capability
    )


def artifact() -> ArtifactRef:
    return ArtifactRef(path="model.onnx", content_hash="sha256:00", byte_size=10)


def value(value_id: str, name: str | None = None) -> Value:
    return Value(id=value_id, name=name if name is not None else value_id)


def node(
    node_id: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    **kwargs: object,
) -> Node:
    return Node(
        id=node_id,
        id_stability=NodeIdStability.NAMED,
        op_type="Mul",
        inputs=inputs,
        outputs=outputs,
        **kwargs,  # type: ignore[arg-type]
    )


class TestTensorRef:
    def test_embedded_raw_requires_a_payload(self) -> None:
        with pytest.raises(IrError, match="payload range"):
            TensorRef(
                id="t", element_type="float32", dims=(1,), storage=Storage.EMBEDDED_RAW
            )

    def test_external_requires_a_reference(self) -> None:
        with pytest.raises(IrError, match="external reference"):
            TensorRef(
                id="t", element_type="float32", dims=(1,), storage=Storage.EXTERNAL
            )

    def test_typed_storage_cannot_carry_locations(self) -> None:
        with pytest.raises(IrError, match="cannot carry"):
            TensorRef(
                id="t",
                element_type="float32",
                dims=(1,),
                storage=Storage.EMBEDDED_TYPED,
                payload=PayloadRange(0, 4),
            )

    def test_negative_dimensions_are_rejected(self) -> None:
        with pytest.raises(IrError, match="negative"):
            TensorRef(
                id="t",
                element_type="float32",
                dims=(2, -1),
                storage=Storage.ABSENT,
            )

    def test_element_count(self) -> None:
        tensor = TensorRef(
            id="t", element_type="int8", dims=(2, 3, 4), storage=Storage.ABSENT
        )
        assert tensor.element_count == 24

    def test_external_ref_validation(self) -> None:
        with pytest.raises(IrError, match="location"):
            ExternalRef(location="", offset=0)
        with pytest.raises(IrError, match="offset"):
            ExternalRef(location="w.bin", offset=-1)


class TestAttribute:
    def test_scalar_kinds_enforce_types(self) -> None:
        assert Attribute("axis", AttrKind.INT, 2).value == 2
        with pytest.raises(IrError, match="needs a int"):
            Attribute("axis", AttrKind.INT, "2")
        with pytest.raises(IrError, match="needs a float"):
            Attribute("alpha", AttrKind.FLOAT, 1)

    def test_booleans_are_not_ints(self) -> None:
        with pytest.raises(IrError, match="needs a int"):
            Attribute("axis", AttrKind.INT, True)

    def test_list_kinds_enforce_element_types(self) -> None:
        assert Attribute("pads", AttrKind.INTS, (0, 1)).value == (0, 1)
        with pytest.raises(IrError, match="tuple of int"):
            Attribute("pads", AttrKind.INTS, (0, "1"))  # type: ignore[arg-type]
        with pytest.raises(IrError, match="tuple of int"):
            Attribute("pads", AttrKind.INTS, [0, 1])  # type: ignore[arg-type]

    def test_reference_extraction(self) -> None:
        assert Attribute("value", AttrKind.TENSOR, "t:a").referenced_tensor_ids() == (
            "t:a",
        )
        assert Attribute(
            "bodies", AttrKind.GRAPHS, ("g:a", "g:b")
        ).referenced_graph_ids() == ("g:a", "g:b")
        assert Attribute("axis", AttrKind.INT, 1).referenced_tensor_ids() == ()


class TestNode:
    def test_duplicate_attribute_names_are_rejected(self) -> None:
        with pytest.raises(IrError, match="duplicate attribute"):
            node(
                "n",
                (),
                ("v",),
                attributes=(
                    Attribute("axis", AttrKind.INT, 0),
                    Attribute("axis", AttrKind.INT, 1),
                ),
            )

    def test_graph_attributes_must_be_declared_subgraphs(self) -> None:
        with pytest.raises(IrError, match="not declared"):
            node(
                "n",
                (),
                ("v",),
                attributes=(Attribute("then_branch", AttrKind.GRAPH, "g:then"),),
            )

    def test_qualified_op_type(self) -> None:
        plain = node("n", (), ("v",))
        assert plain.qualified_op_type == "Mul"
        custom = Node(
            id="n2",
            id_stability=NodeIdStability.NAMED,
            op_type="FusedGelu",
            domain="com.example",
            overload="fast",
            inputs=(),
            outputs=("v",),
        )
        assert custom.qualified_op_type == "com.example::FusedGelu:fast"

    def test_attribute_lookup(self) -> None:
        item = node("n", (), ("v",), attributes=(Attribute("axis", AttrKind.INT, 1),))
        assert item.attribute("axis").value == 1
        with pytest.raises(KeyError):
            item.attribute("missing")


class TestValue:
    def test_shape_properties(self) -> None:
        dynamic = Value(
            id="v", name="v", element_type="float32", shape=("batch", 3, None)
        )
        assert dynamic.has_symbolic_dimensions
        assert not dynamic.is_fully_specified
        static = Value(id="v", name="v", element_type="float32", shape=(1, 3))
        assert static.is_fully_specified

    def test_empty_symbolic_names_are_rejected(self) -> None:
        with pytest.raises(IrError, match="empty symbolic"):
            Value(id="v", name="v", shape=("",))


class TestGraph:
    def test_ports_must_reference_declared_values(self) -> None:
        with pytest.raises(IrError, match="undeclared"):
            Graph(id="g", name="g", nodes=[node("n", ("ghost",), ())], values=[])

    def test_values_have_one_producer(self) -> None:
        with pytest.raises(IrError, match="produced twice"):
            Graph(
                id="g",
                name="g",
                values=[value("v")],
                nodes=[node("a", (), ("v",)), node("b", (), ("v",))],
            )

    def test_graph_inputs_cannot_be_produced(self) -> None:
        with pytest.raises(IrError, match="also produced"):
            Graph(
                id="g",
                name="g",
                values=[value("v")],
                nodes=[node("a", (), ("v",))],
                inputs=["v"],
            )

    def test_outputs_must_be_declared(self) -> None:
        with pytest.raises(IrError, match="not a declared value"):
            Graph(id="g", name="g", outputs=["ghost"])

    def test_producer_and_consumer_derivation(self) -> None:
        graph = Graph(
            id="g",
            name="g",
            values=[value("x"), value("y"), value("z")],
            nodes=[node("a", ("x",), ("y",)), node("b", ("y", "x"), ("z",))],
            inputs=["x"],
            outputs=["z"],
        )
        assert graph.producer("y") == ("a", 0)
        assert graph.producer("x") is None
        assert graph.consumers("x") == (("a", 0), ("b", 1))
        assert graph.consumers("z") == ()
        with pytest.raises(KeyError):
            graph.producer("ghost")
        with pytest.raises(KeyError):
            graph.consumers("ghost")

    def test_symbolic_dimensions_are_collected_sorted(self) -> None:
        graph = Graph(
            id="g",
            name="g",
            values=[
                Value(id="a", name="a", shape=("width", 3)),
                Value(id="b", name="b", shape=("batch", "width")),
            ],
        )
        assert graph.symbolic_dimensions == ("batch", "width")


class TestDocument:
    def make(self, **overrides: object) -> Document:
        graph = Graph(
            id="g:main",
            name="main",
            values=[value("v")],
            nodes=[node("n", (), ("v",))],
            outputs=["v"],
        )
        defaults: dict[str, object] = {
            "source": artifact(),
            "artifact_kind": ArtifactKind.ONNX_MODEL,
            "capabilities": full_capabilities(),
            "graphs": [graph],
        }
        defaults.update(overrides)
        return Document(**defaults)  # type: ignore[arg-type]

    def test_all_capabilities_must_be_answered(self) -> None:
        with pytest.raises(IrError, match="missing capability"):
            self.make(capabilities=full_capabilities()[:3])

    def test_entry_graph_must_exist(self) -> None:
        with pytest.raises(IrError, match="entry graph"):
            self.make(entry_graph="g:ghost")

    def test_initializers_must_resolve(self) -> None:
        graph = Graph(
            id="g:main", name="main", values=[value("v")], initializers=["t:ghost"]
        )
        with pytest.raises(IrError, match="not a declared tensor"):
            self.make(graphs=[graph])

    def test_subgraphs_must_link_back_to_their_node(self) -> None:
        child = Graph(id="g:child", name="child", parent_node="n:other")
        parent = Graph(
            id="g:main",
            name="main",
            values=[value("v")],
            nodes=[node("n", (), ("v",), subgraphs=("g:child",))],
        )
        with pytest.raises(IrError, match="link back"):
            self.make(graphs=[parent, child])

    def test_attribute_tensor_references_must_resolve(self) -> None:
        graph = Graph(
            id="g:main",
            name="main",
            values=[value("v")],
            nodes=[
                node(
                    "n",
                    (),
                    ("v",),
                    attributes=(Attribute("value", AttrKind.TENSOR, "t:ghost"),),
                )
            ],
        )
        with pytest.raises(IrError, match="missing tensor"):
            self.make(graphs=[graph])

    def test_extension_namespaces_are_validated(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            self.make(extensions=[("not-an-extension", 1)])
        with pytest.raises(IrError, match="duplicate extension"):
            self.make(extensions=[("x-onnx.meta", 1), ("x-onnx.meta", 2)])

    def test_capability_and_note_lookup(self) -> None:
        note = CapabilityNote(
            entity_id="n",
            capability=Capability.EDITING,
            availability=Availability.UNAVAILABLE,
            reason="custom domain",
        )
        document = self.make(capability_notes=[note])
        assert document.capability(Capability.TOPOLOGY).is_usable
        assert document.notes_for("n") == (note,)
        assert document.notes_for("other") == ()

    def test_has_errors_reflects_diagnostics(self) -> None:
        assert not self.make().has_errors
        noisy = self.make(diagnostics=[Diagnostic("x", Severity.ERROR, "boom", None)])
        assert noisy.has_errors

    def test_main_graph_shortcut(self) -> None:
        assert self.make().main_graph.id == "g:main"


class TestSupportTypes:
    def test_artifact_ref_requires_prefixed_hash(self) -> None:
        with pytest.raises(IrError, match="algorithm-prefixed"):
            ArtifactRef(path="m.onnx", content_hash="badhash", byte_size=1)

    def test_provenance_requires_operation_and_tool(self) -> None:
        with pytest.raises(IrError, match="operation"):
            ProvenanceEntry(operation="", tool_version="nneditor 0.1")
        with pytest.raises(IrError, match="tool version"):
            ProvenanceEntry(operation="import", tool_version="")

    def test_capability_note_requires_reason(self) -> None:
        with pytest.raises(IrError, match="reason"):
            CapabilityNote("n", Capability.EDITING, Availability.UNAVAILABLE, "  ")
