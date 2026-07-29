"""Reversible, serializable edit commands for tensor and graph revisions.

Every command carries both the state it expects and the state it produces.
That makes a command suitable for deterministic revision identities, recovery
validation, diff previews, and export manifests without consulting mutable UI
state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, cast

from nneditor.editing.cow import ByteSpanEdit, EditError
from nneditor.ir.core import (
    Attribute,
    AttrKind,
    Document,
    Graph,
    Node,
    TensorRef,
    Value,
)
from nneditor.ir.identity import NodeIdStability
from nneditor.transformations.schema import (
    TransformationManifest,
    TransformationPreview,
)

__all__ = [
    "COMMAND_REGISTRY",
    "CommandDescriptor",
    "CommandRegistry",
    "EditCommand",
    "GraphEditCommand",
    "InsertUnaryNode",
    "QuantizeGraph",
    "ReconnectInput",
    "RemoveUnaryNode",
    "RenameNode",
    "ReplaceOperator",
    "ReplaceTensorBytes",
    "ResizeTensor",
    "SetAttribute",
    "ValueMetadataChange",
    "apply_command",
    "command_from_json",
    "command_summary",
    "command_target",
    "command_to_json",
]


@dataclass(frozen=True, slots=True)
class ReplaceTensorBytes:
    """Replace a same-length byte span in one tensor."""

    edit: ByteSpanEdit
    transformation: TransformationManifest | None = None
    preview: TransformationPreview | None = None

    KIND: ClassVar[str] = "replace-tensor-bytes"

    @property
    def target(self) -> str:
        return self.edit.tensor_id

    @property
    def summary(self) -> str:
        if self.transformation is not None:
            return (
                f"{self.transformation.kind.value} on {self.edit.tensor_id} "
                f"({self.preview.changed_elements if self.preview else '?'} "
                "element(s) changed)"
            )
        return (
            f"replace {self.edit.length} byte(s) at offset {self.edit.offset} "
            f"of {self.edit.tensor_id}"
        )


@dataclass(frozen=True, slots=True)
class RenameNode:
    graph_id: str
    node_id: str
    before_name: str | None
    after_name: str

    KIND: ClassVar[str] = "rename-node"


@dataclass(frozen=True, slots=True)
class SetAttribute:
    graph_id: str
    node_id: str
    before: Attribute | None
    after: Attribute | None

    KIND: ClassVar[str] = "set-attribute"

    def __post_init__(self) -> None:
        if self.before is None and self.after is None:
            raise EditError("an attribute command must add, change, or remove a value")
        before_name = self.before.name if self.before is not None else None
        after_name = self.after.name if self.after is not None else None
        if (
            before_name is not None
            and after_name is not None
            and before_name != after_name
        ):
            raise EditError("an attribute command cannot change the attribute name")


@dataclass(frozen=True, slots=True)
class ReplaceOperator:
    graph_id: str
    node_id: str
    before_domain: str
    before_op_type: str
    after_domain: str
    after_op_type: str

    KIND: ClassVar[str] = "replace-operator"


@dataclass(frozen=True, slots=True)
class ReconnectInput:
    graph_id: str
    node_id: str
    input_index: int
    before_value_id: str
    after_value_id: str

    KIND: ClassVar[str] = "reconnect-input"


@dataclass(frozen=True, slots=True)
class InsertUnaryNode:
    graph_id: str
    node: Node
    output_value: Value
    target_node_id: str
    target_input_index: int
    source_value_id: str

    KIND: ClassVar[str] = "insert-unary-node"


@dataclass(frozen=True, slots=True)
class RemoveUnaryNode:
    graph_id: str
    node: Node
    output_value: Value
    target_node_id: str
    target_input_index: int
    source_value_id: str

    KIND: ClassVar[str] = "remove-unary-node"


@dataclass(frozen=True, slots=True)
class ValueMetadataChange:
    graph_id: str
    before: Value
    after: Value

    def __post_init__(self) -> None:
        if self.before.id != self.after.id:
            raise EditError("value metadata changes cannot change value identity")


@dataclass(frozen=True, slots=True)
class ResizeTensor:
    """Replace a full tensor while changing its dtype or dimensions."""

    graph_id: str
    before: TensorRef
    after: TensorRef
    before_bytes: bytes
    after_bytes: bytes
    value_changes: tuple[ValueMetadataChange, ...]
    transformation: TransformationManifest
    preview: TransformationPreview

    KIND: ClassVar[str] = "resize-tensor"

    def __post_init__(self) -> None:
        if self.before.id != self.after.id:
            raise EditError("tensor resize cannot change tensor identity")
        if not self.before_bytes or not self.after_bytes:
            raise EditError("tensor resize needs before and after bytes")

    @property
    def tensor_id(self) -> str:
        return self.before.id


@dataclass(frozen=True, slots=True)
class QuantizeGraph:
    """Insert a portable QuantizeLinear/DequantizeLinear weight boundary."""

    graph_id: str
    tensor_id: str
    source_value_id: str
    consumer_inputs: tuple[tuple[str, int], ...]
    scale_tensor: TensorRef
    zero_tensor: TensorRef
    scale_value: Value
    zero_value: Value
    quantized_value: Value
    dequantized_value: Value
    quantize_node: Node
    dequantize_node: Node
    scale_bytes: bytes
    zero_bytes: bytes
    transformation: TransformationManifest
    preview: TransformationPreview

    KIND: ClassVar[str] = "quantize-graph"

    def __post_init__(self) -> None:
        if not self.consumer_inputs:
            raise EditError("graph quantization needs at least one consumer")
        if self.scale_tensor.id != self.scale_value.id:
            raise EditError("scale tensor/value identities differ")
        if self.zero_tensor.id != self.zero_value.id:
            raise EditError("zero-point tensor/value identities differ")


type GraphEditCommand = (
    RenameNode
    | SetAttribute
    | ReplaceOperator
    | ReconnectInput
    | InsertUnaryNode
    | RemoveUnaryNode
    | QuantizeGraph
)
type EditCommand = ReplaceTensorBytes | ResizeTensor | GraphEditCommand


CommandText = Callable[[object], str]
CommandEncoder = Callable[[object], dict[str, object]]
CommandDecoder = Callable[[dict[str, object]], EditCommand]
CommandApplier = Callable[[Document, object], Document]


@dataclass(frozen=True, slots=True)
class CommandDescriptor:
    """Dispatch metadata for one reversible command class."""

    command_type: type[object]
    kind: str
    target: CommandText
    summary: CommandText
    to_json: CommandEncoder
    from_json: CommandDecoder
    apply: CommandApplier


class CommandRegistry:
    """Type/kind registry shared by previews and future plugin codecs."""

    __slots__ = ("_by_kind", "_by_type")

    def __init__(self, descriptors: Iterable[CommandDescriptor] = ()) -> None:
        self._by_type: dict[type[object], CommandDescriptor] = {}
        self._by_kind: dict[str, CommandDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CommandDescriptor) -> None:
        existing_type = self._by_type.get(descriptor.command_type)
        existing_kind = self._by_kind.get(descriptor.kind)
        if existing_type is not None and existing_type != descriptor:
            raise ValueError(
                f"command type {descriptor.command_type.__name__} is registered twice"
            )
        if existing_kind is not None and existing_kind != descriptor:
            raise ValueError(f"command kind {descriptor.kind!r} is registered twice")
        self._by_type[descriptor.command_type] = descriptor
        self._by_kind[descriptor.kind] = descriptor

    @property
    def by_kind(self) -> MappingProxyType[str, CommandDescriptor]:
        return MappingProxyType(self._by_kind)

    def descriptor_for(self, command: object) -> CommandDescriptor:
        try:
            return self._by_type[type(command)]
        except KeyError:
            raise EditError(
                f"no command handler is registered for {type(command).__name__}"
            ) from None

    def descriptor_for_kind(self, kind: str) -> CommandDescriptor:
        try:
            return self._by_kind[kind]
        except KeyError:
            raise EditError(f"unknown command kind {kind!r}") from None


def _replace_target(command: object) -> str:
    return cast(ReplaceTensorBytes, command).target


def _resize_target(command: object) -> str:
    return cast(ResizeTensor, command).tensor_id


def _quantize_target(command: object) -> str:
    return cast(QuantizeGraph, command).tensor_id


def _node_target(command: object) -> str:
    return cast(
        RenameNode | SetAttribute | ReplaceOperator | ReconnectInput,
        command,
    ).node_id


def _unary_target(command: object) -> str:
    return cast(InsertUnaryNode | RemoveUnaryNode, command).node.id


def _replace_summary(command: object) -> str:
    return cast(ReplaceTensorBytes, command).summary


def _rename_summary(command: object) -> str:
    item = cast(RenameNode, command)
    return f"rename {item.node_id} to {item.after_name!r}"


def _attribute_summary(command: object) -> str:
    item = cast(SetAttribute, command)
    attribute = item.after or item.before
    assert attribute is not None
    action = "remove" if item.after is None else "add" if item.before is None else "set"
    return f"{action} attribute {attribute.name!r} on {item.node_id}"


def _operator_summary(command: object) -> str:
    item = cast(ReplaceOperator, command)
    return (
        f"replace {item.before_domain}::{item.before_op_type} with "
        f"{item.after_domain}::{item.after_op_type} on {item.node_id}"
    )


def _reconnect_summary(command: object) -> str:
    item = cast(ReconnectInput, command)
    return (
        f"reconnect input {item.input_index} of {item.node_id} to {item.after_value_id}"
    )


def _insert_summary(command: object) -> str:
    item = cast(InsertUnaryNode, command)
    return f"insert {item.node.qualified_op_type} before {item.target_node_id}"


def _remove_summary(command: object) -> str:
    return f"remove unary node {cast(RemoveUnaryNode, command).node.id}"


def _resize_summary(command: object) -> str:
    item = cast(ResizeTensor, command)
    return (
        f"{item.transformation.kind.value} {item.tensor_id}: "
        f"{item.before.dims} -> {item.after.dims}"
    )


def _quantize_summary(command: object) -> str:
    item = cast(QuantizeGraph, command)
    granularity = item.transformation.granularity
    assert granularity is not None
    return f"insert ONNX Q/DQ for {item.tensor_id} ({granularity.value})"


COMMAND_REGISTRY: CommandRegistry


def command_target(command: EditCommand) -> str:
    return COMMAND_REGISTRY.descriptor_for(command).target(command)


def command_summary(command: EditCommand) -> str:
    return COMMAND_REGISTRY.descriptor_for(command).summary(command)


def _attribute_to_json(attribute: Attribute | None) -> object:
    if attribute is None:
        return None
    value: object = (
        list(attribute.value) if isinstance(attribute.value, tuple) else attribute.value
    )
    return {"name": attribute.name, "kind": attribute.kind.value, "value": value}


def _attribute_from_json(raw: object) -> Attribute | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EditError("malformed attribute command payload")
    try:
        kind = AttrKind(str(raw["kind"]))
        value = raw["value"]
        if kind in {
            AttrKind.INTS,
            AttrKind.FLOATS,
            AttrKind.STRINGS,
            AttrKind.TENSORS,
            AttrKind.GRAPHS,
        }:
            if not isinstance(value, list):
                raise EditError("list-valued attribute is not a list")
            value = tuple(value)
        return Attribute(name=str(raw["name"]), kind=kind, value=value)
    except (KeyError, TypeError, ValueError) as error:
        raise EditError(f"malformed attribute command payload: {error}") from error


def _value_to_json(value: Value) -> dict[str, object]:
    return {
        "id": value.id,
        "name": value.name,
        "element_type": value.element_type,
        "shape": None if value.shape is None else list(value.shape),
    }


def _value_from_json(raw: object) -> Value:
    if not isinstance(raw, dict):
        raise EditError("malformed value command payload")
    try:
        shape_raw = raw.get("shape")
        if shape_raw is not None and not isinstance(shape_raw, list):
            raise EditError("value shape must be a list or null")
        return Value(
            id=str(raw["id"]),
            name=str(raw["name"]),
            element_type=(
                None if raw.get("element_type") is None else str(raw["element_type"])
            ),
            shape=None if shape_raw is None else tuple(shape_raw),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EditError(f"malformed value command payload: {error}") from error


def _tensor_to_json(tensor: TensorRef) -> dict[str, object]:
    return {
        "id": tensor.id,
        "element_type": tensor.element_type,
        "dims": list(tensor.dims),
        "storage": tensor.storage.value,
        "payload": (
            None
            if tensor.payload is None
            else {
                "offset": tensor.payload.offset,
                "length": tensor.payload.length,
            }
        ),
        "external": (
            None
            if tensor.external is None
            else {
                "location": tensor.external.location,
                "offset": tensor.external.offset,
                "length": tensor.external.length,
                "checksum": tensor.external.checksum,
            }
        ),
        "typed_span": (
            None
            if tensor.typed_span is None
            else {
                "offset": tensor.typed_span.offset,
                "length": tensor.typed_span.length,
            }
        ),
    }


def _tensor_from_json(raw: object) -> TensorRef:
    if not isinstance(raw, dict):
        raise EditError("malformed tensor command payload")
    try:
        from nneditor.ir.core import ExternalRef, PayloadRange, Storage

        payload = raw.get("payload")
        external = raw.get("external")
        typed_span = raw.get("typed_span")
        if payload is not None and not isinstance(payload, dict):
            raise EditError("tensor payload range is malformed")
        if external is not None and not isinstance(external, dict):
            raise EditError("tensor external reference is malformed")
        if typed_span is not None and not isinstance(typed_span, dict):
            raise EditError("tensor typed span is malformed")
        return TensorRef(
            id=str(raw["id"]),
            element_type=str(raw["element_type"]),
            dims=tuple(int(item) for item in raw["dims"]),
            storage=Storage(str(raw["storage"])),
            payload=(
                None
                if payload is None
                else PayloadRange(int(payload["offset"]), int(payload["length"]))
            ),
            external=(
                None
                if external is None
                else ExternalRef(
                    location=str(external["location"]),
                    offset=int(external["offset"]),
                    length=(
                        None
                        if external.get("length") is None
                        else int(external["length"])
                    ),
                    checksum=(
                        None
                        if external.get("checksum") is None
                        else str(external["checksum"])
                    ),
                )
            ),
            typed_span=(
                None
                if typed_span is None
                else PayloadRange(
                    int(typed_span["offset"]),
                    int(typed_span["length"]),
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EditError(f"malformed tensor command payload: {error}") from error


def _node_to_json(node: Node) -> dict[str, object]:
    return {
        "id": node.id,
        "id_stability": node.id_stability.value,
        "op_type": node.op_type,
        "domain": node.domain,
        "overload": node.overload,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
        "attributes": [_attribute_to_json(item) for item in node.attributes],
        "subgraphs": list(node.subgraphs),
        "source_name": node.source_name,
        "source_location": node.source_location,
        "has_side_effects": node.has_side_effects,
        "mutates_inputs": node.mutates_inputs,
    }


def _node_from_json(raw: object) -> Node:
    if not isinstance(raw, dict):
        raise EditError("malformed node command payload")
    try:
        attributes_raw = raw.get("attributes", [])
        if not isinstance(attributes_raw, list):
            raise EditError("node attributes must be a list")
        attributes = tuple(
            attribute
            for item in attributes_raw
            if (attribute := _attribute_from_json(item)) is not None
        )
        return Node(
            id=str(raw["id"]),
            id_stability=NodeIdStability(str(raw["id_stability"])),
            op_type=str(raw["op_type"]),
            domain=str(raw.get("domain", "")),
            overload=str(raw.get("overload", "")),
            inputs=tuple(str(item) for item in raw["inputs"]),
            outputs=tuple(str(item) for item in raw["outputs"]),
            attributes=attributes,
            subgraphs=tuple(str(item) for item in raw.get("subgraphs", [])),
            source_name=(
                None if raw.get("source_name") is None else str(raw["source_name"])
            ),
            source_location=(
                None
                if raw.get("source_location") is None
                else str(raw["source_location"])
            ),
            has_side_effects=(
                raw.get("has_side_effects")
                if isinstance(raw.get("has_side_effects"), bool)
                else None
            ),
            mutates_inputs=(
                raw.get("mutates_inputs")
                if isinstance(raw.get("mutates_inputs"), bool)
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EditError(f"malformed node command payload: {error}") from error


def _encode_replace(command: object) -> dict[str, object]:
    item = cast(ReplaceTensorBytes, command)
    payload: dict[str, object] = {
        "kind": item.KIND,
        "tensor_id": item.edit.tensor_id,
        "offset": item.edit.offset,
        "before": item.edit.before.hex(),
        "after": item.edit.after.hex(),
    }
    if item.transformation is not None:
        payload["transformation"] = item.transformation.to_json()
    if item.preview is not None:
        payload["preview"] = item.preview.to_json()
    return payload


def _decode_replace(data: dict[str, object]) -> EditCommand:
    return ReplaceTensorBytes(
        ByteSpanEdit(
            tensor_id=str(data["tensor_id"]),
            offset=int(data["offset"]),  # type: ignore[call-overload]
            before=bytes.fromhex(str(data["before"])),
            after=bytes.fromhex(str(data["after"])),
        ),
        transformation=(
            None
            if data.get("transformation") is None
            else TransformationManifest.from_json(data["transformation"])
        ),
        preview=(
            None
            if data.get("preview") is None
            else TransformationPreview.from_json(data["preview"])
        ),
    )


def _encode_resize(command: object) -> dict[str, object]:
    item = cast(ResizeTensor, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "before_tensor": _tensor_to_json(item.before),
        "after_tensor": _tensor_to_json(item.after),
        "before_bytes": item.before_bytes.hex(),
        "after_bytes": item.after_bytes.hex(),
        "value_changes": [
            {
                "graph_id": change.graph_id,
                "before": _value_to_json(change.before),
                "after": _value_to_json(change.after),
            }
            for change in item.value_changes
        ],
        "transformation": item.transformation.to_json(),
        "preview": item.preview.to_json(),
    }


def _decode_resize(data: dict[str, object]) -> EditCommand:
    changes = data.get("value_changes")
    if not isinstance(changes, list):
        raise EditError("resize value changes are malformed")
    return ResizeTensor(
        graph_id=str(data["graph_id"]),
        before=_tensor_from_json(data["before_tensor"]),
        after=_tensor_from_json(data["after_tensor"]),
        before_bytes=bytes.fromhex(str(data["before_bytes"])),
        after_bytes=bytes.fromhex(str(data["after_bytes"])),
        value_changes=tuple(
            ValueMetadataChange(
                graph_id=str(item["graph_id"]),
                before=_value_from_json(item["before"]),
                after=_value_from_json(item["after"]),
            )
            for item in changes
            if isinstance(item, dict)
        ),
        transformation=TransformationManifest.from_json(data["transformation"]),
        preview=TransformationPreview.from_json(data["preview"]),
    )


def _encode_quantize(command: object) -> dict[str, object]:
    item = cast(QuantizeGraph, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "tensor_id": item.tensor_id,
        "source_value_id": item.source_value_id,
        "consumer_inputs": [list(port) for port in item.consumer_inputs],
        "scale_tensor": _tensor_to_json(item.scale_tensor),
        "zero_tensor": _tensor_to_json(item.zero_tensor),
        "scale_value": _value_to_json(item.scale_value),
        "zero_value": _value_to_json(item.zero_value),
        "quantized_value": _value_to_json(item.quantized_value),
        "dequantized_value": _value_to_json(item.dequantized_value),
        "quantize_node": _node_to_json(item.quantize_node),
        "dequantize_node": _node_to_json(item.dequantize_node),
        "scale_bytes": item.scale_bytes.hex(),
        "zero_bytes": item.zero_bytes.hex(),
        "transformation": item.transformation.to_json(),
        "preview": item.preview.to_json(),
    }


def _decode_quantize(data: dict[str, object]) -> EditCommand:
    consumers = data.get("consumer_inputs")
    if not isinstance(consumers, list):
        raise EditError("quantization consumers are malformed")
    return QuantizeGraph(
        graph_id=str(data["graph_id"]),
        tensor_id=str(data["tensor_id"]),
        source_value_id=str(data["source_value_id"]),
        consumer_inputs=tuple(
            (str(item[0]), int(item[1]))
            for item in consumers
            if isinstance(item, list) and len(item) == 2
        ),
        scale_tensor=_tensor_from_json(data["scale_tensor"]),
        zero_tensor=_tensor_from_json(data["zero_tensor"]),
        scale_value=_value_from_json(data["scale_value"]),
        zero_value=_value_from_json(data["zero_value"]),
        quantized_value=_value_from_json(data["quantized_value"]),
        dequantized_value=_value_from_json(data["dequantized_value"]),
        quantize_node=_node_from_json(data["quantize_node"]),
        dequantize_node=_node_from_json(data["dequantize_node"]),
        scale_bytes=bytes.fromhex(str(data["scale_bytes"])),
        zero_bytes=bytes.fromhex(str(data["zero_bytes"])),
        transformation=TransformationManifest.from_json(data["transformation"]),
        preview=TransformationPreview.from_json(data["preview"]),
    )


def _encode_rename(command: object) -> dict[str, object]:
    item = cast(RenameNode, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "node_id": item.node_id,
        "before_name": item.before_name,
        "after_name": item.after_name,
    }


def _decode_rename(data: dict[str, object]) -> EditCommand:
    return RenameNode(
        graph_id=str(data["graph_id"]),
        node_id=str(data["node_id"]),
        before_name=(
            None if data.get("before_name") is None else str(data["before_name"])
        ),
        after_name=str(data["after_name"]),
    )


def _encode_attribute(command: object) -> dict[str, object]:
    item = cast(SetAttribute, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "node_id": item.node_id,
        "before": _attribute_to_json(item.before),
        "after": _attribute_to_json(item.after),
    }


def _decode_attribute(data: dict[str, object]) -> EditCommand:
    return SetAttribute(
        graph_id=str(data["graph_id"]),
        node_id=str(data["node_id"]),
        before=_attribute_from_json(data.get("before")),
        after=_attribute_from_json(data.get("after")),
    )


def _encode_operator(command: object) -> dict[str, object]:
    item = cast(ReplaceOperator, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "node_id": item.node_id,
        "before_domain": item.before_domain,
        "before_op_type": item.before_op_type,
        "after_domain": item.after_domain,
        "after_op_type": item.after_op_type,
    }


def _decode_operator(data: dict[str, object]) -> EditCommand:
    return ReplaceOperator(
        graph_id=str(data["graph_id"]),
        node_id=str(data["node_id"]),
        before_domain=str(data["before_domain"]),
        before_op_type=str(data["before_op_type"]),
        after_domain=str(data["after_domain"]),
        after_op_type=str(data["after_op_type"]),
    )


def _encode_reconnect(command: object) -> dict[str, object]:
    item = cast(ReconnectInput, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "node_id": item.node_id,
        "input_index": item.input_index,
        "before_value_id": item.before_value_id,
        "after_value_id": item.after_value_id,
    }


def _decode_reconnect(data: dict[str, object]) -> EditCommand:
    return ReconnectInput(
        graph_id=str(data["graph_id"]),
        node_id=str(data["node_id"]),
        input_index=int(data["input_index"]),  # type: ignore[call-overload]
        before_value_id=str(data["before_value_id"]),
        after_value_id=str(data["after_value_id"]),
    )


def _encode_unary(command: object) -> dict[str, object]:
    item = cast(InsertUnaryNode | RemoveUnaryNode, command)
    return {
        "kind": item.KIND,
        "graph_id": item.graph_id,
        "node": _node_to_json(item.node),
        "output_value": _value_to_json(item.output_value),
        "target_node_id": item.target_node_id,
        "target_input_index": item.target_input_index,
        "source_value_id": item.source_value_id,
    }


def _decode_unary(
    data: dict[str, object],
    command_type: type[InsertUnaryNode] | type[RemoveUnaryNode],
) -> EditCommand:
    return command_type(
        graph_id=str(data["graph_id"]),
        node=_node_from_json(data["node"]),
        output_value=_value_from_json(data["output_value"]),
        target_node_id=str(data["target_node_id"]),
        target_input_index=int(data["target_input_index"]),  # type: ignore[call-overload]
        source_value_id=str(data["source_value_id"]),
    )


def _decode_insert(data: dict[str, object]) -> EditCommand:
    return _decode_unary(data, InsertUnaryNode)


def _decode_remove(data: dict[str, object]) -> EditCommand:
    return _decode_unary(data, RemoveUnaryNode)


def command_to_json(command: EditCommand) -> dict[str, object]:
    """Serialize through the registered codec for the concrete command type."""
    return COMMAND_REGISTRY.descriptor_for(command).to_json(command)


def command_from_json(data: dict[str, object]) -> EditCommand:
    """Deserialize through the codec registered for the manifest kind."""
    try:
        kind = str(data["kind"])
        return COMMAND_REGISTRY.descriptor_for_kind(kind).from_json(data)
    except EditError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise EditError(f"malformed command manifest: {error}") from error


def _replace_document(
    document: Document,
    *,
    updated_graphs: tuple[Graph, ...] = (),
    tensors: tuple[TensorRef, ...] | None = None,
) -> Document:
    replacements = {graph.id: graph for graph in updated_graphs}
    graphs = [replacements.get(graph.id, graph) for graph in document.graphs.values()]
    return Document(
        source=document.source,
        artifact_kind=document.artifact_kind,
        capabilities=document.capabilities.values(),
        graphs=graphs,
        tensors=document.tensors.values() if tensors is None else tensors,
        entry_graph=document.entry_graph,
        provenance=document.provenance,
        diagnostics=document.diagnostics,
        capability_notes=document.capability_notes,
        extensions=document.extensions,
        schema_version=document.schema_version,
    )


def _replace_graph(document: Document, updated: Graph) -> Document:
    return _replace_document(document, updated_graphs=(updated,))


def _clone_graph(
    graph: Graph,
    *,
    nodes: tuple[Node, ...] | None = None,
    values: tuple[Value, ...] | None = None,
    initializers: tuple[str, ...] | None = None,
) -> Graph:
    return Graph(
        id=graph.id,
        name=graph.name,
        nodes=graph.nodes if nodes is None else nodes,
        values=graph.values if values is None else values,
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializers=graph.initializers if initializers is None else initializers,
        parent_node=graph.parent_node,
    )


def _replace_node(graph: Graph, updated: Node) -> Graph:
    if all(node.id != updated.id for node in graph.nodes):
        raise EditError(f"graph {graph.id!r} has no node {updated.id!r}")
    return _clone_graph(
        graph,
        nodes=tuple(updated if node.id == updated.id else node for node in graph.nodes),
    )


def _command_graph(document: Document, graph_id: str) -> Graph:
    try:
        return document.graphs[graph_id]
    except KeyError:
        raise EditError(f"document has no graph {graph_id!r}") from None


def _apply_replace(document: Document, _command: object) -> Document:
    return document


def _apply_resize(document: Document, command: object) -> Document:
    item = cast(ResizeTensor, command)
    current = document.tensors.get(item.tensor_id)
    if current != item.before:
        raise EditError("tensor resize precondition no longer holds")
    graphs = dict(document.graphs)
    for change in item.value_changes:
        graph = graphs.get(change.graph_id)
        if graph is None:
            raise EditError(
                f"document has no graph {change.graph_id!r} for value change"
            )
        try:
            existing = graph.value(change.before.id)
        except KeyError:
            raise EditError(
                f"graph {graph.id!r} has no value {change.before.id!r}"
            ) from None
        if existing != change.before:
            raise EditError("value metadata precondition no longer holds")
        values = tuple(
            change.after if value.id == change.before.id else value
            for value in graph.values
        )
        graphs[graph.id] = _clone_graph(graph, values=values)
    tensors = tuple(
        item.after if tensor.id == item.tensor_id else tensor
        for tensor in document.tensors.values()
    )
    return _replace_document(
        document,
        updated_graphs=tuple(graphs.values()),
        tensors=tensors,
    )


def _apply_quantize(document: Document, command: object) -> Document:
    item = cast(QuantizeGraph, command)
    graph = _command_graph(document, item.graph_id)
    if item.tensor_id not in graph.initializers:
        raise EditError("quantized value is not an initializer")
    new_ids = {
        item.scale_tensor.id,
        item.zero_tensor.id,
        item.quantized_value.id,
        item.dequantized_value.id,
        item.quantize_node.id,
        item.dequantize_node.id,
    }
    if any(
        graph.has_value(identity)
        or any(node.id == identity for node in graph.nodes)
        or identity in document.tensors
        for identity in new_ids
    ):
        raise EditError("quantization-generated identity already exists")
    consumer_map = dict(item.consumer_inputs)
    for node_id, input_index in item.consumer_inputs:
        node = graph.node(node_id)
        if (
            not 0 <= input_index < len(node.inputs)
            or node.inputs[input_index] != item.source_value_id
        ):
            raise EditError("quantization consumer precondition no longer holds")
    first_consumer = min(
        index for index, node in enumerate(graph.nodes) if node.id in consumer_map
    )
    updated_nodes: list[Node] = []
    for index, node in enumerate(graph.nodes):
        if index == first_consumer:
            updated_nodes.extend((item.quantize_node, item.dequantize_node))
        matching_ports = [
            port for consumer_id, port in item.consumer_inputs if consumer_id == node.id
        ]
        if matching_ports:
            inputs = list(node.inputs)
            for port in matching_ports:
                inputs[port] = item.dequantized_value.id
            node = replace(node, inputs=tuple(inputs))
        updated_nodes.append(node)
    updated_graph = _clone_graph(
        graph,
        nodes=tuple(updated_nodes),
        values=(
            *graph.values,
            item.scale_value,
            item.zero_value,
            item.quantized_value,
            item.dequantized_value,
        ),
        initializers=(
            *graph.initializers,
            item.scale_tensor.id,
            item.zero_tensor.id,
        ),
    )
    return _replace_document(
        document,
        updated_graphs=(updated_graph,),
        tensors=(
            *document.tensors.values(),
            item.scale_tensor,
            item.zero_tensor,
        ),
    )


def _apply_rename(document: Document, command: object) -> Document:
    item = cast(RenameNode, command)
    graph = _command_graph(document, item.graph_id)
    node = graph.node(item.node_id)
    if node.source_name != item.before_name:
        raise EditError("rename precondition no longer holds")
    updated = replace(node, source_name=item.after_name)
    return _replace_graph(document, _replace_node(graph, updated))


def _apply_attribute(document: Document, command: object) -> Document:
    item = cast(SetAttribute, command)
    graph = _command_graph(document, item.graph_id)
    node = graph.node(item.node_id)
    attributes = list(node.attributes)
    attribute = item.before if item.before is not None else item.after
    assert attribute is not None
    existing_attribute = next(
        (value for value in attributes if value.name == attribute.name),
        None,
    )
    if existing_attribute != item.before:
        raise EditError("attribute precondition no longer holds")
    if existing_attribute is not None:
        attributes.remove(existing_attribute)
    if item.after is not None:
        attributes.append(item.after)
    updated = replace(node, attributes=tuple(attributes))
    return _replace_graph(document, _replace_node(graph, updated))


def _apply_operator(document: Document, command: object) -> Document:
    item = cast(ReplaceOperator, command)
    graph = _command_graph(document, item.graph_id)
    node = graph.node(item.node_id)
    if (node.domain, node.op_type) != (
        item.before_domain,
        item.before_op_type,
    ):
        raise EditError("operator replacement precondition no longer holds")
    updated = replace(
        node,
        domain=item.after_domain,
        op_type=item.after_op_type,
    )
    return _replace_graph(document, _replace_node(graph, updated))


def _apply_reconnect(document: Document, command: object) -> Document:
    item = cast(ReconnectInput, command)
    graph = _command_graph(document, item.graph_id)
    node = graph.node(item.node_id)
    if not 0 <= item.input_index < len(node.inputs):
        raise EditError("reconnect input index is outside the node")
    if node.inputs[item.input_index] != item.before_value_id:
        raise EditError("reconnect precondition no longer holds")
    inputs = list(node.inputs)
    inputs[item.input_index] = item.after_value_id
    updated = replace(node, inputs=tuple(inputs))
    return _replace_graph(document, _replace_node(graph, updated))


def _unary_target_state(
    document: Document,
    command: InsertUnaryNode | RemoveUnaryNode,
) -> tuple[Graph, Node, list[str]]:
    graph = _command_graph(document, command.graph_id)
    target = graph.node(command.target_node_id)
    if not 0 <= command.target_input_index < len(target.inputs):
        raise EditError("unary command target input is outside the node")
    return graph, target, list(target.inputs)


def _apply_insert(document: Document, command: object) -> Document:
    item = cast(InsertUnaryNode, command)
    graph, target, target_inputs = _unary_target_state(document, item)
    if target_inputs[item.target_input_index] != item.source_value_id:
        raise EditError("unary insertion precondition no longer holds")
    target_inputs[item.target_input_index] = item.output_value.id
    target = replace(target, inputs=tuple(target_inputs))
    nodes: list[Node] = []
    for node in graph.nodes:
        if node.id == target.id:
            nodes.extend((item.node, target))
        else:
            nodes.append(node)
    return _replace_graph(
        document,
        _clone_graph(
            graph,
            nodes=tuple(nodes),
            values=(*graph.values, item.output_value),
        ),
    )


def _apply_remove(document: Document, command: object) -> Document:
    item = cast(RemoveUnaryNode, command)
    graph, target, target_inputs = _unary_target_state(document, item)
    if target_inputs[item.target_input_index] != item.output_value.id:
        raise EditError("unary removal precondition no longer holds")
    target_inputs[item.target_input_index] = item.source_value_id
    target = replace(target, inputs=tuple(target_inputs))
    remaining_nodes = tuple(
        target if node.id == target.id else node
        for node in graph.nodes
        if node.id != item.node.id
    )
    values = tuple(value for value in graph.values if value.id != item.output_value.id)
    return _replace_graph(
        document,
        _clone_graph(graph, nodes=remaining_nodes, values=values),
    )


COMMAND_REGISTRY = CommandRegistry(
    (
        CommandDescriptor(
            ReplaceTensorBytes,
            ReplaceTensorBytes.KIND,
            _replace_target,
            _replace_summary,
            _encode_replace,
            _decode_replace,
            _apply_replace,
        ),
        CommandDescriptor(
            ResizeTensor,
            ResizeTensor.KIND,
            _resize_target,
            _resize_summary,
            _encode_resize,
            _decode_resize,
            _apply_resize,
        ),
        CommandDescriptor(
            QuantizeGraph,
            QuantizeGraph.KIND,
            _quantize_target,
            _quantize_summary,
            _encode_quantize,
            _decode_quantize,
            _apply_quantize,
        ),
        CommandDescriptor(
            RenameNode,
            RenameNode.KIND,
            _node_target,
            _rename_summary,
            _encode_rename,
            _decode_rename,
            _apply_rename,
        ),
        CommandDescriptor(
            SetAttribute,
            SetAttribute.KIND,
            _node_target,
            _attribute_summary,
            _encode_attribute,
            _decode_attribute,
            _apply_attribute,
        ),
        CommandDescriptor(
            ReplaceOperator,
            ReplaceOperator.KIND,
            _node_target,
            _operator_summary,
            _encode_operator,
            _decode_operator,
            _apply_operator,
        ),
        CommandDescriptor(
            ReconnectInput,
            ReconnectInput.KIND,
            _node_target,
            _reconnect_summary,
            _encode_reconnect,
            _decode_reconnect,
            _apply_reconnect,
        ),
        CommandDescriptor(
            InsertUnaryNode,
            InsertUnaryNode.KIND,
            _unary_target,
            _insert_summary,
            _encode_unary,
            _decode_insert,
            _apply_insert,
        ),
        CommandDescriptor(
            RemoveUnaryNode,
            RemoveUnaryNode.KIND,
            _unary_target,
            _remove_summary,
            _encode_unary,
            _decode_remove,
            _apply_remove,
        ),
    )
)


def apply_command(document: Document, command: EditCommand) -> Document:
    """Apply through the registered handler for the concrete command type."""
    return COMMAND_REGISTRY.descriptor_for(command).apply(document, command)
