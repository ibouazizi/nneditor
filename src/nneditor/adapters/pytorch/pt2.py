"""Native torch.export (PT2 archive) ingestion (P6.2).

A ``.pt2`` archive is a zip of JSON plus raw weight files — no pickle is
required anywhere on this path. The frozen contract (verified against torch
2.13, serde schema major 8, archive_version 0) is:

* ``<root>/archive_format`` = ``pt2``, ``<root>/byteorder`` = ``little``;
* ``<root>/models/<name>.json`` — the serialized exported program: graph
  nodes with ``torch.ops.*`` targets, named arguments as a tagged union,
  tensor metadata, and a signature classifying placeholders as parameters,
  buffers, or user inputs;
* ``<root>/data/weights/<path>`` — one raw little-endian file per parameter,
  uncompressed, linked by ``model_weights_config.json``.

Mapping decisions: parameter/buffer placeholders take their *parameter
names* ("linear.weight") as value names so weights read naturally in the
inspector; ATen targets become nodes in the ``pytorch.aten`` domain;
non-tensor arguments become typed attributes prefixed with the formal
argument name; symbolic sizes survive as named dimensions. Anything the core
IR cannot model rides in ``x-torch.*`` extensions with a diagnostic, never
silently dropped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.scalar_types import (
    SERDE_SCALAR_TYPES,
    element_width_bytes,
)
from nneditor.adapters.pytorch.zip_store import (
    ZipMember,
    ZipStoreError,
    read_member,
    zip_members,
)
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    contract_for,
)
from nneditor.ir.core import (
    ArtifactRef,
    Attribute,
    AttrKind,
    CapabilityNote,
    Document,
    Graph,
    JsonValue,
    Node,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import (
    ROOT_GRAPH_ID,
    initializer_id,
    node_id,
    value_id,
)
from nneditor.storage.reader import hash_file

__all__ = ["Pt2Error", "open_pt2"]

_SUPPORTED_SCHEMA_MAJOR: Final = 8
_MAX_JSON_BYTES: Final = 512 * 1024 * 1024


class Pt2Error(ValueError):
    """Raised when a PT2 archive cannot be opened in safe artifact mode."""


def _as_list(value: JsonValue) -> list[JsonValue]:
    """Narrow a JSON value to a list; anything else reads as empty."""
    return value if isinstance(value, list) else []


def _as_dict(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a JSON value to an object; anything else reads as empty."""
    return value if isinstance(value, dict) else {}


def _root_prefix(members: dict[str, ZipMember]) -> str:
    for name in members:
        if name.endswith("archive_format"):
            return name.rsplit("archive_format", 1)[0]
    raise Pt2Error("the archive declares no archive_format; not a PT2 archive")


def _dim(raw: JsonValue, log: DiagnosticLog, owner: str) -> int | str | None:
    if isinstance(raw, dict):
        if "as_int" in raw and isinstance(raw["as_int"], int):
            return raw["as_int"]
        expr = raw.get("as_expr")
        if isinstance(expr, dict):
            text = expr.get("expr_str")
            if isinstance(text, str) and text:
                return text
        if "as_sym_int" in raw:
            return _dim(raw["as_sym_int"], log, owner)
        name = raw.get("as_name")
        if isinstance(name, str) and name:
            return name
    log.add(
        "pytorch.unsupported-dimension",
        Severity.WARNING,
        f"a dimension of {owner} could not be interpreted and is unknown",
        owner,
    )
    return None


def _tensor_meta_shape(
    meta: dict[str, JsonValue], log: DiagnosticLog, owner: str
) -> tuple[int | str | None, ...]:
    sizes = meta.get("sizes")
    if not isinstance(sizes, list):
        return ()
    return tuple(_dim(item, log, owner) for item in sizes)


def _element_type(meta: dict[str, JsonValue]) -> str | None:
    code = meta.get("dtype")
    if isinstance(code, int):
        return SERDE_SCALAR_TYPES.get(code)
    return None


_SCALAR_ARGS: Final[dict[str, AttrKind]] = {
    "as_int": AttrKind.INT,
    "as_float": AttrKind.FLOAT,
    "as_string": AttrKind.STRING,
    "as_bool": AttrKind.INT,
    "as_scalar_type": AttrKind.INT,
    "as_memory_format": AttrKind.INT,
    "as_layout": AttrKind.INT,
}
_LIST_ARGS: Final[dict[str, AttrKind]] = {
    "as_ints": AttrKind.INTS,
    "as_floats": AttrKind.FLOATS,
    "as_strings": AttrKind.STRINGS,
    "as_bools": AttrKind.INTS,
}


def _attribute_int(payload: JsonValue) -> int | None:
    """A JSON scalar as an attribute int; hostile payloads read as None.

    The tag is attacker-typed: ``as_int`` may carry a string, an infinity, or
    a NaN, any of which would make a bare ``int()`` raise.
    """
    if isinstance(payload, (bool, int)):
        return int(payload)
    if isinstance(payload, float) and math.isfinite(payload):
        return int(payload)
    return None


def _attribute_float(payload: JsonValue) -> float | None:
    """A JSON scalar as an attribute float; unconvertible payloads are None."""
    if isinstance(payload, (bool, float)):
        return float(payload)
    if isinstance(payload, int):
        try:
            return float(payload)
        except OverflowError:
            return None
    return None


def _argument_attribute(name: str, tag: str, payload: JsonValue) -> Attribute | None:
    kind = _SCALAR_ARGS.get(tag)
    if kind is not None and isinstance(payload, (int, float, str, bool)):
        if kind is AttrKind.INT:
            scalar = _attribute_int(payload)
            return None if scalar is None else Attribute(name, kind, scalar)
        if kind is AttrKind.FLOAT:
            value = _attribute_float(payload)
            return None if value is None else Attribute(name, kind, value)
        return Attribute(name, kind, str(payload))
    list_kind = _LIST_ARGS.get(tag)
    if list_kind is not None and isinstance(payload, list):
        if list_kind is AttrKind.INTS:
            ints = [_attribute_int(item) for item in payload]
            return Attribute(
                name, list_kind, tuple(item for item in ints if item is not None)
            )
        if list_kind is AttrKind.FLOATS:
            floats = [_attribute_float(item) for item in payload]
            return Attribute(
                name, list_kind, tuple(item for item in floats if item is not None)
            )
        return Attribute(
            name,
            list_kind,
            tuple(
                str(item)
                for item in payload
                if isinstance(item, (str, int, float, bool)) or item is None
            ),
        )
    if tag == "as_sym_int" and isinstance(payload, dict):
        inner = payload.get("as_name") or payload.get("as_int")
        if isinstance(inner, (int, str)):
            return Attribute(name, AttrKind.STRING, str(inner))
    if tag == "as_device" and isinstance(payload, dict):
        device = payload.get("type", "unknown")
        if isinstance(device, (str, int, float, bool)) or device is None:
            return Attribute(name, AttrKind.STRING, str(device))
    return None


def open_pt2(path: Path | str) -> Document:
    """Open a PT2 exported-program archive as an IR document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`Pt2Error`; the layered error contract in the session depends
    on adapters never leaking raw exceptions.
    """
    source = Path(path)
    try:
        return _open_pt2(source)
    except Pt2Error:
        raise
    except Exception as error:
        raise Pt2Error(
            f"{source.name} could not be opened as a PT2 archive: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_pt2(source: Path) -> Document:
    try:
        members = zip_members(source)
    except ZipStoreError as error:
        raise Pt2Error(str(error)) from error
    prefix = _root_prefix(members)
    log = DiagnosticLog()

    def member(name: str) -> ZipMember | None:
        return members.get(prefix + name)

    def member_bytes(name: str, entry: ZipMember, limit: int) -> bytes:
        try:
            return read_member(source, entry, limit=limit)
        except ZipStoreError as error:
            raise Pt2Error(f"{name} could not be read: {error}") from error

    def member_json(name: str) -> JsonValue:
        entry = member(name)
        if entry is None:
            raise Pt2Error(f"the archive is missing {name}")
        try:
            # ValueError covers JSONDecodeError, oversized integer literals,
            # and malformed UTF-8; RecursionError covers pathological nesting
            # depth — all attacker-reachable through the archive's JSON.
            parsed: JsonValue = json.loads(member_bytes(name, entry, _MAX_JSON_BYTES))
            return parsed
        except Pt2Error:
            raise
        except (ValueError, RecursionError) as error:
            raise Pt2Error(f"{name} could not be parsed: {error}") from error

    archive_format = member("archive_format")
    if (
        archive_format is None
        or member_bytes("archive_format", archive_format, 64) != b"pt2"
    ):
        raise Pt2Error(f"{source.name} does not declare the pt2 archive format")
    byteorder_member = member("byteorder")
    if (
        byteorder_member is not None
        and member_bytes("byteorder", byteorder_member, 64) != b"little"
    ):
        raise Pt2Error("big-endian PT2 archives are not supported")

    model_name = next(
        (
            name[len(prefix) :]
            for name in members
            if name.startswith(prefix + "models/") and name.endswith(".json")
        ),
        None,
    )
    if model_name is None:
        raise Pt2Error("the archive holds no serialized model")
    model = member_json(model_name)
    if not isinstance(model, dict):
        raise Pt2Error("the serialized model is not an object")

    schema_version = model.get("schema_version")
    major = schema_version.get("major") if isinstance(schema_version, dict) else None
    if major != _SUPPORTED_SCHEMA_MAJOR:
        raise Pt2Error(
            f"exported-program schema major {major!r} is not the supported "
            f"major {_SUPPORTED_SCHEMA_MAJOR}; re-export with a supported "
            "torch version"
        )

    graph_module = model.get("graph_module")
    if not isinstance(graph_module, dict):
        raise Pt2Error("the serialized model holds no graph module")
    graph_raw = graph_module.get("graph")
    if not isinstance(graph_raw, dict):
        raise Pt2Error("the graph module holds no graph")

    # -- signature: classify placeholders ---------------------------------
    placeholder_names: dict[str, str] = {}
    user_inputs: list[str] = []
    signature = graph_module.get("signature")
    input_specs = (
        signature.get("input_specs", []) if isinstance(signature, dict) else []
    )
    for spec in _as_list(input_specs):
        if not isinstance(spec, dict):
            continue
        for role in ("parameter", "buffer", "tensor_constant"):
            body = spec.get(role)
            if isinstance(body, dict):
                arg = body.get("arg")
                placeholder = arg.get("name") if isinstance(arg, dict) else None
                target = body.get(f"{role}_name") or body.get("target") or placeholder
                if isinstance(placeholder, str) and isinstance(target, str):
                    placeholder_names[placeholder] = target
        user = spec.get("user_input")
        if isinstance(user, dict):
            arg = user.get("arg")
            tensor = arg.get("as_tensor") if isinstance(arg, dict) else None
            name = tensor.get("name") if isinstance(tensor, dict) else None
            if isinstance(name, str):
                user_inputs.append(name)

    # -- values ------------------------------------------------------------
    values: dict[str, Value] = {}

    def declare(placeholder: str) -> str:
        vid = value_id(ROOT_GRAPH_ID, placeholder)
        if vid not in values:
            values[vid] = Value(
                id=vid, name=placeholder_names.get(placeholder, placeholder)
            )
        return vid

    tensor_values = _as_dict(graph_raw.get("tensor_values"))
    for placeholder, meta in tensor_values.items():
        if not isinstance(meta, dict):
            continue
        vid = value_id(ROOT_GRAPH_ID, placeholder)
        values[vid] = Value(
            id=vid,
            name=placeholder_names.get(placeholder, placeholder),
            element_type=_element_type(meta),
            shape=_tensor_meta_shape(meta, log, placeholder),
        )

    # -- nodes -------------------------------------------------------------
    nodes: list[Node] = []
    custom_targets: set[str] = set()
    for ordinal, node_raw in enumerate(_as_list(graph_raw.get("nodes"))):
        if not isinstance(node_raw, dict):
            continue
        target = str(node_raw.get("target", ""))
        if target.startswith("torch.ops.aten."):
            op_type = target[len("torch.ops.aten.") :]
            domain = "pytorch.aten"
        elif target.startswith("torch.ops."):
            op_type = target[len("torch.ops.") :]
            domain = "pytorch.custom"
            custom_targets.add(target)
        else:
            op_type = target or "unknown"
            domain = "pytorch"
            custom_targets.add(target)
        inputs: list[str] = []
        attributes: list[Attribute] = []
        for argument in _as_list(node_raw.get("inputs")):
            if not isinstance(argument, dict):
                continue
            arg_name = str(argument.get("name", f"arg{len(inputs)}"))
            body = argument.get("arg")
            if not isinstance(body, dict) or not body:
                continue
            tag, payload = next(iter(body.items()))
            if tag == "as_tensor" and isinstance(payload, dict):
                referenced = payload.get("name")
                if isinstance(referenced, str):
                    inputs.append(declare(referenced))
                continue
            if tag == "as_tensors" and isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        inputs.append(declare(str(item["name"])))
                continue
            if tag == "as_none":
                continue
            attribute = _argument_attribute(arg_name, tag, payload)
            if attribute is not None:
                attributes.append(attribute)
            else:
                log.add(
                    "pytorch.unsupported-argument",
                    Severity.WARNING,
                    f"argument {arg_name!r} of {target} has kind {tag!r}, "
                    "which is carried as unsupported",
                    target,
                )
        outputs: list[str] = []
        for out_raw in _as_list(node_raw.get("outputs")):
            if not isinstance(out_raw, dict) or not out_raw:
                continue
            tag, payload = next(iter(out_raw.items()))
            if tag == "as_tensor" and isinstance(payload, dict):
                referenced = payload.get("name")
                if isinstance(referenced, str):
                    outputs.append(declare(referenced))
            elif tag == "as_tensors" and isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        outputs.append(declare(str(item["name"])))
            elif tag != "as_none":
                log.add(
                    "pytorch.unsupported-argument",
                    Severity.WARNING,
                    f"an output of {target} has kind {tag!r}",
                    target,
                )
        metadata = node_raw.get("metadata")
        stack_trace = None
        module_stack = None
        if isinstance(metadata, dict):
            raw_trace = metadata.get("stack_trace")
            stack_trace = (
                raw_trace.strip().splitlines()[-1].strip()
                if isinstance(raw_trace, str) and raw_trace.strip()
                else None
            )
            raw_stack = metadata.get("nn_module_stack")
            module_stack = raw_stack if isinstance(raw_stack, str) else None
        first_output = next((values[vid].name for vid in outputs if vid in values), "")
        derived_id, stability = node_id(
            ROOT_GRAPH_ID,
            name="",
            first_output=first_output,
            ordinal=ordinal,
        )
        nodes.append(
            Node(
                id=derived_id,
                id_stability=stability,
                op_type=op_type,
                domain=domain,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                attributes=tuple(attributes),
                source_name=(
                    module_stack.rsplit(",", 1)[-1] or None if module_stack else None
                ),
                source_location=stack_trace,
            )
        )

    # -- graph outputs -----------------------------------------------------
    graph_outputs: list[str] = []
    for out_raw in _as_list(graph_raw.get("outputs")):
        if isinstance(out_raw, dict):
            tensor = out_raw.get("as_tensor")
            if isinstance(tensor, dict) and isinstance(tensor.get("name"), str):
                graph_outputs.append(declare(str(tensor["name"])))

    # -- weights -----------------------------------------------------------
    tensors: list[TensorRef] = []
    initializer_ids: list[str] = []
    weights_config_raw = member("data/weights/model_weights_config.json")
    weights_config: dict[str, JsonValue] = {}
    if weights_config_raw is not None:
        parsed = member_json("data/weights/model_weights_config.json")
        weights_config = _as_dict(_as_dict(parsed).get("config"))
    for parameter_name, entry in weights_config.items():
        if not isinstance(entry, dict):
            continue
        tensor_id = initializer_id(ROOT_GRAPH_ID, str(parameter_name))
        meta = entry.get("tensor_meta")
        element_type = (
            _element_type(meta) if isinstance(meta, dict) else None
        ) or "undefined"
        dims_raw = (
            _tensor_meta_shape(meta, log, str(parameter_name))
            if isinstance(meta, dict)
            else ()
        )
        dims = tuple(dim if isinstance(dim, int) else 0 for dim in dims_raw)
        weight_member = member(f"data/weights/{entry.get('path_name')}")
        span = weight_member.span if weight_member is not None else None
        if entry.get("use_pickle") or weight_member is None or span is None:
            log.add(
                "pytorch.weight-not-range-readable",
                Severity.WARNING,
                f"parameter {parameter_name!r} is pickled or compressed and "
                "is not range-readable in safe mode",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            initializer_ids.append(tensor_id)
            continue
        offset, length = span
        width = element_width_bytes(element_type)
        expected = None
        if width is not None:
            count = 1
            for dim in dims:
                count *= dim
            expected = count * width
        if expected is not None and expected != length:
            log.add(
                "pytorch.tensor-length-mismatch",
                Severity.ERROR,
                f"parameter {parameter_name!r} holds {length} bytes but its "
                f"metadata implies {expected}",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            initializer_ids.append(tensor_id)
            continue
        tensors.append(
            TensorRef(
                tensor_id,
                element_type,
                dims,
                Storage.EMBEDDED_RAW,
                payload=PayloadRange(offset, length),
            )
        )
        initializer_ids.append(tensor_id)

    # Graph inputs: user inputs only; parameters read as initializers. The
    # signature is artifact-controlled, so a user input naming no declared
    # value is diagnosed rather than allowed to fail graph validation.
    graph_inputs: list[str] = []
    for name in user_inputs:
        vid = value_id(ROOT_GRAPH_ID, name)
        if vid in values:
            graph_inputs.append(vid)
        else:
            log.add(
                "pytorch.unknown-user-input",
                Severity.WARNING,
                f"the signature declares user input {name!r}, which the graph "
                "does not define; it is dropped from the input list",
                ROOT_GRAPH_ID,
            )
    graph = Graph(
        id=ROOT_GRAPH_ID,
        name=model_name.rsplit("/", 1)[-1].removesuffix(".json"),
        nodes=nodes,
        values=values.values(),
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializers=initializer_ids,
    )

    notes = [
        CapabilityNote(
            entity_id=node.id,
            capability=Capability.EDITING,
            availability=Availability.UNAVAILABLE,
            reason=f"target {node.op_type} is outside the ATen operator set, "
            "so edits cannot be validated",
        )
        for node in nodes
        if node.domain != "pytorch.aten"
    ]
    if custom_targets:
        log.add(
            "pytorch.custom-operator",
            Severity.INFO,
            "the program calls operators outside torch.ops.aten: "
            + ", ".join(sorted(custom_targets)),
            ROOT_GRAPH_ID,
        )

    extensions: list[tuple[str, JsonValue]] = [
        (
            "x-torch.model",
            {
                "torch_version": model.get("torch_version"),
                "opset_version": model.get("opset_version"),
                "schema_version": schema_version,
                "range_constraints": model.get("range_constraints"),
                "guards_code": model.get("guards_code"),
            },
        )
    ]
    if placeholder_names:
        extensions.append(
            ("x-torch.parameters", dict(sorted(placeholder_names.items())))
        )
    module_call_graph = graph_module.get("module_call_graph")
    if module_call_graph:
        extensions.append(("x-torch.module_call_graph", module_call_graph))

    contract = contract_for(ArtifactKind.PYTORCH_EXPORTED_PROGRAM)
    return Document(
        source=ArtifactRef(
            path=str(source),
            content_hash=hash_file(source),
            byte_size=source.stat().st_size,
        ),
        artifact_kind=ArtifactKind.PYTORCH_EXPORTED_PROGRAM,
        capabilities=contract.statuses,
        graphs=[graph],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("loading_mode", "safe artifact"),
                    ("reader", "native pt2 archive parser"),
                ),
            )
        ],
        diagnostics=tuple(log),
        capability_notes=notes,
        extensions=extensions,
    )
