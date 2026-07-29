"""Phase 4 transactional edit preparation and validation.

The validator turns user requests into reversible commands.  Nothing mutates
until the caller commits the returned transaction, and every rejection carries
stable finding codes suitable for the UI and export report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from onnx import defs

from nneditor.editing.commands import (
    EditCommand,
    InsertUnaryNode,
    ReconnectInput,
    RemoveUnaryNode,
    RenameNode,
    ReplaceOperator,
    SetAttribute,
    apply_command,
    command_summary,
    command_target,
)
from nneditor.editing.cow import EditError
from nneditor.editing.revisions import ValidationState
from nneditor.ir.capabilities import Availability, Capability
from nneditor.ir.core import Attribute, AttrKind, Document, Graph, IrError, Node, Value
from nneditor.ir.identity import NodeIdStability, node_id, value_id

__all__ = [
    "EditRequest",
    "EditTransaction",
    "FindingLevel",
    "InsertUnaryRequest",
    "ReconnectInputRequest",
    "RemoveUnaryRequest",
    "RenameNodeRequest",
    "ReplaceOperatorRequest",
    "SetAttributeRequest",
    "ValidationFinding",
    "ValidationPipeline",
    "parse_attribute_value",
]


class FindingLevel(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    level: FindingLevel
    message: str
    entity_id: str | None = None

    def __str__(self) -> str:
        target = f" [{self.entity_id}]" if self.entity_id else ""
        return f"{self.code}{target}: {self.message}"


@dataclass(frozen=True, slots=True)
class RenameNodeRequest:
    graph_id: str
    node_id: str
    new_name: str


@dataclass(frozen=True, slots=True)
class SetAttributeRequest:
    graph_id: str
    node_id: str
    name: str
    kind: AttrKind | None = None
    value: object | None = None
    remove: bool = False


@dataclass(frozen=True, slots=True)
class ReplaceOperatorRequest:
    graph_id: str
    node_id: str
    op_type: str
    domain: str = ""


@dataclass(frozen=True, slots=True)
class InsertUnaryRequest:
    graph_id: str
    target_node_id: str
    target_input_index: int
    op_type: str
    domain: str = ""
    name: str = ""


@dataclass(frozen=True, slots=True)
class RemoveUnaryRequest:
    graph_id: str
    node_id: str


@dataclass(frozen=True, slots=True)
class ReconnectInputRequest:
    graph_id: str
    node_id: str
    input_index: int
    new_value_id: str


type EditRequest = (
    RenameNodeRequest
    | SetAttributeRequest
    | ReplaceOperatorRequest
    | InsertUnaryRequest
    | RemoveUnaryRequest
    | ReconnectInputRequest
)


def _capability_changes(before: Document, after: Document) -> tuple[str, ...]:
    changes: list[str] = []
    for capability in Capability:
        old = before.capability(capability)
        new = after.capability(capability)
        if old != new:
            changes.append(
                f"{capability.value}: {old.availability.value} -> "
                f"{new.availability.value} ({new.reason})"
            )
    return tuple(changes) or ("No artifact capability availability changes.",)


@dataclass(frozen=True, slots=True)
class EditTransaction:
    base_revision_id: str | None
    commands: tuple[EditCommand, ...]
    candidate: Document
    findings: tuple[ValidationFinding, ...]
    capability_changes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.commands) and not any(
            item.level is FindingLevel.ERROR for item in self.findings
        )

    @property
    def summaries(self) -> tuple[str, ...]:
        return tuple(command_summary(command) for command in self.commands)

    @property
    def validation_state(self) -> ValidationState:
        return ValidationState(
            ok=self.ok,
            findings=tuple(str(item) for item in self.findings),
        )


_ATTRIBUTE_TYPES = {
    AttrKind.INT: defs.OpSchema.AttrType.INT,
    AttrKind.FLOAT: defs.OpSchema.AttrType.FLOAT,
    AttrKind.STRING: defs.OpSchema.AttrType.STRING,
    AttrKind.INTS: defs.OpSchema.AttrType.INTS,
    AttrKind.FLOATS: defs.OpSchema.AttrType.FLOATS,
    AttrKind.STRINGS: defs.OpSchema.AttrType.STRINGS,
    AttrKind.TENSOR: defs.OpSchema.AttrType.TENSOR,
    AttrKind.TENSORS: defs.OpSchema.AttrType.TENSORS,
    AttrKind.GRAPH: defs.OpSchema.AttrType.GRAPH,
    AttrKind.GRAPHS: defs.OpSchema.AttrType.GRAPHS,
}

_SHAPE_PRESERVING_UNARY = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Ceil",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "HardSigmoid",
        "Identity",
        "LeakyRelu",
        "Log",
        "Neg",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    }
)
"""Default-domain unary operators whose output shape and dtype equal input."""

_COMPATIBLE_OPERATOR_FAMILIES = (
    frozenset({"Add", "Sub", "Mul", "Div", "Max", "Min", "Mean"}),
    _SHAPE_PRESERVING_UNARY,
)


def parse_attribute_value(kind: AttrKind, text: str) -> object:
    """Parse the compact textual representation used by the edit panel."""
    if kind is AttrKind.INT:
        return int(text)
    if kind is AttrKind.FLOAT:
        return float(text)
    if kind is AttrKind.STRING:
        return text
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    if kind is AttrKind.INTS:
        return tuple(int(item) for item in parts)
    if kind is AttrKind.FLOATS:
        return tuple(float(item) for item in parts)
    if kind is AttrKind.STRINGS:
        return parts
    raise ValueError("tensor- and graph-valued attributes are not editable in Phase 4")


def _opsets(document: Document) -> dict[str, int]:
    for namespace, value in document.extensions:
        if namespace != "x-onnx.model" or not isinstance(value, dict):
            continue
        entries = value.get("opset_imports")
        if not isinstance(entries, list):
            return {}
        found: dict[str, int] = {}
        for entry in entries:
            domain = entry.get("domain") if isinstance(entry, dict) else None
            version = entry.get("version") if isinstance(entry, dict) else None
            if (
                isinstance(entry, dict)
                and isinstance(domain, str)
                and isinstance(version, int)
                and not isinstance(version, bool)
            ):
                found[domain] = version
        return found
    return {}


def _schema(document: Document, domain: str, op_type: str) -> defs.OpSchema:
    normalized = "" if domain == "ai.onnx" else domain
    version = _opsets(document).get(normalized)
    if version is None:
        raise EditError(f"the model imports no opset for domain {normalized!r}")
    try:
        return defs.get_schema(op_type, version, normalized)
    except defs.SchemaError as error:
        raise EditError(
            f"{normalized or 'ai.onnx'}::{op_type} has no schema at opset {version}"
        ) from error


def _value_type(value: Value) -> str | None:
    if value.element_type is None:
        return None
    aliases = {
        "float32": "float",
        "float64": "double",
        "float16": "float16",
        "int8": "int8",
        "uint8": "uint8",
        "int16": "int16",
        "uint16": "uint16",
        "int32": "int32",
        "uint32": "uint32",
        "int64": "int64",
        "uint64": "uint64",
        "bool": "bool",
        "string": "string",
        "bfloat16": "bfloat16",
    }
    return f"tensor({aliases.get(value.element_type, value.element_type)})"


def _compatible_values(
    before: Value,
    after: Value,
    *,
    allow_unresolved: bool = False,
) -> tuple[bool, str | None]:
    if (
        before.element_type is not None
        and after.element_type is not None
        and before.element_type != after.element_type
    ):
        return False, (
            f"dtype {after.element_type!r} does not match {before.element_type!r}"
        )
    if before.element_type is None or after.element_type is None:
        return (
            (True, "one side has no declared dtype")
            if allow_unresolved
            else (False, "dtype compatibility is unresolved")
        )
    if before.shape is None or after.shape is None:
        return (
            (True, "one side has no declared shape")
            if allow_unresolved
            else (False, "shape compatibility is unresolved")
        )
    if len(before.shape) != len(after.shape):
        return False, f"rank {len(after.shape)} does not match rank {len(before.shape)}"
    unresolved = False
    for expected, actual in zip(before.shape, after.shape, strict=True):
        if isinstance(expected, int) and isinstance(actual, int) and expected != actual:
            return False, f"dimension {actual} does not match {expected}"
        if not isinstance(expected, int) or not isinstance(actual, int):
            if expected != actual and not allow_unresolved:
                return False, "symbolic shape constraints do not match"
            unresolved = unresolved or expected != actual or expected is None
    return (
        True,
        "symbolic dimensions remain unresolved and will be disclosed at export"
        if unresolved
        else None,
    )


def _would_create_cycle(graph: Graph, target_node_id: str, value_id_: str) -> bool:
    producer = graph.producer(value_id_)
    if producer is None:
        return False
    sought = producer[0]
    pending = [target_node_id]
    visited: set[str] = set()
    while pending:
        node_id_ = pending.pop()
        if node_id_ == sought:
            return True
        if node_id_ in visited:
            continue
        visited.add(node_id_)
        node = graph.node(node_id_)
        for output_id in node.outputs:
            pending.extend(
                consumer_id
                for consumer_id, _port in graph.consumers(output_id)
                if consumer_id not in visited
            )
    return False


def _schema_findings(
    document: Document, graph: Graph, node: Node
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    try:
        schema = _schema(document, node.domain, node.op_type)
    except EditError as error:
        return [
            ValidationFinding(
                "edit.schema-unavailable",
                FindingLevel.ERROR,
                str(error),
                node.id,
            )
        ]

    if not schema.min_input <= len(node.inputs) <= schema.max_input:
        findings.append(
            ValidationFinding(
                "edit.input-arity",
                FindingLevel.ERROR,
                f"{node.qualified_op_type} accepts {schema.min_input}.."
                f"{schema.max_input} inputs, got {len(node.inputs)}",
                node.id,
            )
        )
    if not schema.min_output <= len(node.outputs) <= schema.max_output:
        findings.append(
            ValidationFinding(
                "edit.output-arity",
                FindingLevel.ERROR,
                f"{node.qualified_op_type} accepts {schema.min_output}.."
                f"{schema.max_output} outputs, got {len(node.outputs)}",
                node.id,
            )
        )

    for attribute in node.attributes:
        declaration = schema.attributes.get(attribute.name)
        if declaration is None:
            findings.append(
                ValidationFinding(
                    "edit.attribute-unknown",
                    FindingLevel.ERROR,
                    f"{node.qualified_op_type} has no attribute {attribute.name!r}",
                    node.id,
                )
            )
        elif _ATTRIBUTE_TYPES[attribute.kind] != declaration.type:
            findings.append(
                ValidationFinding(
                    "edit.attribute-type",
                    FindingLevel.ERROR,
                    f"attribute {attribute.name!r} requires {declaration.type}, "
                    f"not {attribute.kind.value}",
                    node.id,
                )
            )
    present = {attribute.name for attribute in node.attributes}
    for name, declaration in schema.attributes.items():
        if declaration.required and name not in present:
            findings.append(
                ValidationFinding(
                    "edit.attribute-required",
                    FindingLevel.ERROR,
                    f"required attribute {name!r} is missing",
                    node.id,
                )
            )

    constraints = {
        item.type_param_str: set(item.allowed_type_strs)
        for item in schema.type_constraints
    }
    for index, value_id_ in enumerate(node.inputs):
        if not schema.inputs or not graph.has_value(value_id_):
            continue
        formal = schema.inputs[min(index, len(schema.inputs) - 1)]
        allowed = constraints.get(formal.type_str)
        actual = _value_type(graph.value(value_id_))
        if allowed is not None and actual is not None and actual not in allowed:
            findings.append(
                ValidationFinding(
                    "edit.dtype-constraint",
                    FindingLevel.ERROR,
                    f"input {index} has {actual}, outside {formal.type_str}",
                    node.id,
                )
            )
    for index, value_id_ in enumerate(node.outputs):
        if not schema.outputs or not graph.has_value(value_id_):
            continue
        formal = schema.outputs[min(index, len(schema.outputs) - 1)]
        allowed = constraints.get(formal.type_str)
        actual = _value_type(graph.value(value_id_))
        if allowed is not None and actual is not None and actual not in allowed:
            findings.append(
                ValidationFinding(
                    "edit.dtype-constraint",
                    FindingLevel.ERROR,
                    f"output {index} has {actual}, outside {formal.type_str}",
                    node.id,
                )
            )
    return findings


type EditRequestHandler = Callable[
    [Document, Graph, object, str | None, list[ValidationFinding]],
    EditCommand,
]


def _rename_handler(
    _document: Document,
    graph: Graph,
    raw_request: object,
    _base_revision_id: str | None,
    _findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(RenameNodeRequest, raw_request)
    node = graph.node(request.node_id)
    name = request.new_name.strip()
    if not name:
        raise EditError("node name cannot be empty")
    if node.source_name == name:
        raise EditError("node already has that name")
    if any(item.source_name == name for item in graph.nodes if item.id != node.id):
        raise EditError(f"another node is already named {name!r}")
    return RenameNode(graph.id, node.id, node.source_name, name)


def _attribute_handler(
    _document: Document,
    graph: Graph,
    raw_request: object,
    _base_revision_id: str | None,
    _findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(SetAttributeRequest, raw_request)
    node = graph.node(request.node_id)
    name = request.name.strip()
    if not name:
        raise EditError("attribute name cannot be empty")
    before = next((item for item in node.attributes if item.name == name), None)
    after = None
    if not request.remove:
        if request.kind is None:
            raise EditError("attribute kind is required")
        if request.kind in {
            AttrKind.TENSOR,
            AttrKind.TENSORS,
            AttrKind.GRAPH,
            AttrKind.GRAPHS,
        }:
            raise EditError(
                "tensor- and graph-valued attributes are read-only in Phase 4"
            )
        after = Attribute(name, request.kind, request.value)  # type: ignore[arg-type]
        if after == before:
            raise EditError(f"attribute {name!r} already has that value")
    return SetAttribute(graph.id, node.id, before, after)


def _replace_operator_handler(
    _document: Document,
    graph: Graph,
    raw_request: object,
    _base_revision_id: str | None,
    _findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(ReplaceOperatorRequest, raw_request)
    node = graph.node(request.node_id)
    op_type = request.op_type.strip()
    if not op_type:
        raise EditError("operator type cannot be empty")
    if (node.domain, node.op_type) == (request.domain, op_type):
        raise EditError("node already uses that operator")
    normalized_before = "" if node.domain == "ai.onnx" else node.domain
    normalized_after = "" if request.domain == "ai.onnx" else request.domain
    if (
        normalized_before
        or normalized_after
        or not any(
            node.op_type in family and op_type in family
            for family in _COMPATIBLE_OPERATOR_FAMILIES
        )
    ):
        raise EditError(
            "operator replacement is limited to the same modelled "
            "shape/dtype-preserving family"
        )
    return ReplaceOperator(
        graph.id,
        node.id,
        node.domain,
        node.op_type,
        request.domain,
        op_type,
    )


def _reconnect_handler(
    _document: Document,
    graph: Graph,
    raw_request: object,
    _base_revision_id: str | None,
    findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(ReconnectInputRequest, raw_request)
    node = graph.node(request.node_id)
    if not 0 <= request.input_index < len(node.inputs):
        raise EditError("input index is outside the node")
    if not graph.has_value(request.new_value_id):
        raise EditError(f"graph has no value {request.new_value_id!r}")
    before_id = node.inputs[request.input_index]
    if before_id == request.new_value_id:
        raise EditError("input is already connected to that value")
    if _would_create_cycle(graph, node.id, request.new_value_id):
        raise EditError("reconnection would create a graph cycle")
    producer = graph.producer(request.new_value_id)
    if producer is not None:
        order = {item.id: index for index, item in enumerate(graph.nodes)}
        if order[producer[0]] >= order[node.id]:
            raise EditError(
                f"value {request.new_value_id!r} is produced by "
                f"{producer[0]!r}, which does not precede {node.id!r} in the "
                "stored node order; the export would fail its "
                "use-before-definition check"
            )
    compatible, reason = _compatible_values(
        graph.value(before_id),
        graph.value(request.new_value_id),
    )
    if not compatible:
        raise EditError(f"reconnection is incompatible: {reason}")
    if reason is not None:
        findings.append(
            ValidationFinding(
                "edit.symbolic-shape",
                FindingLevel.WARNING,
                reason,
                node.id,
            )
        )
    return ReconnectInput(
        graph.id,
        node.id,
        request.input_index,
        before_id,
        request.new_value_id,
    )


def _insert_unary_handler(
    document: Document,
    graph: Graph,
    raw_request: object,
    base_revision_id: str | None,
    _findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(InsertUnaryRequest, raw_request)
    target = graph.node(request.target_node_id)
    if not 0 <= request.target_input_index < len(target.inputs):
        raise EditError("input index is outside the target node")
    source_id = target.inputs[request.target_input_index]
    source = graph.value(source_id)
    digest = hashlib.blake2b(
        (
            f"{base_revision_id}|{target.id}|{request.target_input_index}|"
            f"{request.domain}|{request.op_type}"
        ).encode(),
        digest_size=8,
    ).hexdigest()
    name = request.name.strip() or f"nneditor_{request.op_type}_{digest}"
    new_node_id, _ = node_id(graph.id, name=name, ordinal=len(graph.nodes))
    output_name = f"_nneditor_{digest}"
    output = Value(
        id=value_id(graph.id, output_name),
        name=output_name,
        element_type=source.element_type,
        shape=source.shape,
    )
    node = Node(
        id=new_node_id,
        id_stability=NodeIdStability.NAMED,
        op_type=request.op_type.strip(),
        domain=request.domain,
        inputs=(source_id,),
        outputs=(output.id,),
        source_name=name,
    )
    schema = _schema(document, node.domain, node.op_type)
    if not (
        schema.min_input <= 1 <= schema.max_input
        and schema.min_output <= 1 <= schema.max_output
    ):
        raise EditError(f"{node.qualified_op_type} is not a unary operator")
    normalized = "" if node.domain == "ai.onnx" else node.domain
    if normalized or node.op_type not in _SHAPE_PRESERVING_UNARY:
        raise EditError(
            "unary insertion is limited to modelled default-domain "
            "operators that preserve shape and dtype"
        )
    return InsertUnaryNode(
        graph.id,
        node,
        output,
        target.id,
        request.target_input_index,
        source_id,
    )


def _remove_unary_handler(
    _document: Document,
    graph: Graph,
    raw_request: object,
    _base_revision_id: str | None,
    findings: list[ValidationFinding],
) -> EditCommand:
    request = cast(RemoveUnaryRequest, raw_request)
    node = graph.node(request.node_id)
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise EditError("only one-input/one-output nodes can be removed")
    if node.subgraphs or node.has_side_effects is True or node.mutates_inputs is True:
        raise EditError("stateful or nested-graph nodes cannot be removed")
    normalized = "" if node.domain == "ai.onnx" else node.domain
    if normalized or node.op_type not in _SHAPE_PRESERVING_UNARY:
        raise EditError(
            "unary removal is limited to modelled default-domain operators "
            "that preserve shape and dtype"
        )
    output_id = node.outputs[0]
    if output_id in graph.outputs:
        raise EditError("a node producing a graph output cannot be removed")
    consumers = graph.consumers(output_id)
    if len(consumers) != 1:
        raise EditError("the unary output must have exactly one consumer")
    target_id, input_index = consumers[0]
    source_id = node.inputs[0]
    compatible, reason = _compatible_values(
        graph.value(output_id),
        graph.value(source_id),
        allow_unresolved=True,
    )
    if not compatible:
        raise EditError(f"removal is incompatible: {reason}")
    if reason is not None:
        findings.append(
            ValidationFinding(
                "edit.symbolic-shape",
                FindingLevel.WARNING,
                reason,
                node.id,
            )
        )
    return RemoveUnaryNode(
        graph.id,
        node,
        graph.value(output_id),
        target_id,
        input_index,
        source_id,
    )


_DEFAULT_EDIT_REQUEST_HANDLERS: Mapping[type[object], EditRequestHandler] = {
    RenameNodeRequest: _rename_handler,
    SetAttributeRequest: _attribute_handler,
    ReplaceOperatorRequest: _replace_operator_handler,
    ReconnectInputRequest: _reconnect_handler,
    InsertUnaryRequest: _insert_unary_handler,
    RemoveUnaryRequest: _remove_unary_handler,
}


class ValidationPipeline:
    """Prepare and validate the deliberately constrained Phase 4 commands."""

    __slots__ = ("_handlers",)

    def __init__(
        self,
        handlers: Mapping[type[object], EditRequestHandler] | None = None,
    ) -> None:
        self._handlers = dict(_DEFAULT_EDIT_REQUEST_HANDLERS)
        if handlers is not None:
            self._handlers.update(handlers)

    def register_handler(
        self,
        request_type: type[object],
        handler: EditRequestHandler,
    ) -> None:
        """Register or replace one request-to-command validation rule."""
        self._handlers[request_type] = handler

    def prepare(
        self,
        document: Document,
        request: EditRequest,
        *,
        base_revision_id: str | None,
    ) -> EditTransaction:
        findings: list[ValidationFinding] = []
        graph_id = request.graph_id
        if graph_id != document.entry_graph:
            findings.append(
                ValidationFinding(
                    "edit.entry-graph-only",
                    FindingLevel.ERROR,
                    "Phase 4 graph edits are confined to the entry graph",
                    graph_id,
                )
            )
            return EditTransaction(base_revision_id, (), document, tuple(findings))
        editing = document.capability(Capability.EDITING)
        if editing.availability not in {Availability.AVAILABLE, Availability.PARTIAL}:
            findings.append(
                ValidationFinding(
                    "edit.capability-unavailable",
                    FindingLevel.ERROR,
                    editing.reason,
                )
            )
        try:
            graph = document.graphs[graph_id]
            command = self._command(
                document, graph, request, base_revision_id, findings
            )
            candidate = apply_command(document, command)
        except (EditError, IrError, KeyError, ValueError) as error:
            findings.append(
                ValidationFinding(
                    "edit.precondition",
                    FindingLevel.ERROR,
                    str(error),
                )
            )
            return EditTransaction(base_revision_id, (), document, tuple(findings))

        touched = command_target(command)
        candidate_graph = candidate.graphs[graph_id]
        if isinstance(command, RemoveUnaryNode):
            # The removed node no longer exists in the candidate; the node
            # whose contract changed is the rewired consumer, so *it* gets
            # the schema/arity/dtype pass (audit finding: this was skipped).
            findings.extend(
                _schema_findings(
                    candidate,
                    candidate_graph,
                    candidate_graph.node(command.target_node_id),
                )
            )
        else:
            findings.extend(
                _schema_findings(
                    candidate, candidate_graph, candidate_graph.node(touched)
                )
            )
        if any(item.level is FindingLevel.ERROR for item in findings):
            return EditTransaction(
                base_revision_id,
                (command,),
                candidate,
                tuple(findings),
                _capability_changes(document, candidate),
            )
        findings.append(
            ValidationFinding(
                "edit.valid",
                FindingLevel.INFO,
                "IR invariants, ONNX schema/opset, arity, dtype, shape, and "
                "export capability checks passed",
                touched,
            )
        )
        return EditTransaction(
            base_revision_id,
            (command,),
            candidate,
            tuple(findings),
            _capability_changes(document, candidate),
        )

    def _command(
        self,
        document: Document,
        graph: Graph,
        request: EditRequest,
        base_revision_id: str | None,
        findings: list[ValidationFinding],
    ) -> EditCommand:
        handler = self._handlers.get(type(request))
        if handler is None:
            raise EditError(
                f"no validation rule is registered for {type(request).__name__}"
            )
        return handler(document, graph, request, base_revision_id, findings)
