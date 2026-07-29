"""Transactional ONNX export for committed Phase 4/5 revisions.

Imports stay on NNEditor's lazy reader.  Export is the explicit point where
the official ONNX bindings may materialize the structural protobuf so schemas,
shape inference, and ``onnx.checker`` can validate the artifact that will be
published.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, helper

from nneditor import __version__
from nneditor.adapters.onnx.index import ModelIndex, TensorStorage
from nneditor.adapters.onnx.indexer import index_model
from nneditor.adapters.onnx.numerical import compare_numerically
from nneditor.adapters.onnx.to_ir import index_to_document
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.editing.commands import (
    EditCommand,
    InsertUnaryNode,
    QuantizeGraph,
    ReconnectInput,
    RemoveUnaryNode,
    RenameNode,
    ReplaceOperator,
    ReplaceTensorBytes,
    ResizeTensor,
    SetAttribute,
    command_summary,
    command_target,
    command_to_json,
)
from nneditor.editing.revisions import Revision
from nneditor.ir.capabilities import ExportFidelity
from nneditor.ir.core import Attribute, AttrKind, Document, Node
from nneditor.storage.paths import UnsafePathError, resolve_within

__all__ = [
    "ExportError",
    "ExportReport",
    "export_revision",
    "report_from_json",
]

DEFAULT_EXTERNAL_THRESHOLD: Final = 64 * 1024 * 1024


class ExportError(ValueError):
    """Raised when no complete, validated export can be published."""


@dataclass(frozen=True, slots=True)
class ExportReport:
    destination: Path
    report_path: Path
    source_hash: str
    target_hash: str
    source_components: tuple[dict[str, str], ...]
    tool_version: str
    dependencies: tuple[tuple[str, str], ...]
    revision_ids: tuple[str, ...]
    fidelity: ExportFidelity
    written_files: tuple[Path, ...]
    structural_checks: tuple[str, ...]
    unsupported_operators: tuple[str, ...]
    metadata_changes: tuple[str, ...]
    dtype_layout_changes: tuple[str, ...]
    unresolved_shapes: tuple[str, ...]
    custom_runtime_needs: tuple[str, ...]
    edit_manifest: tuple[dict[str, object], ...]
    transformations: tuple[dict[str, object], ...] = ()
    warnings: tuple[str, ...] = ()
    numerical_comparison: dict[str, object] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "destination": str(self.destination),
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "source_components": list(self.source_components),
            "tool_version": self.tool_version,
            "dependencies": dict(self.dependencies),
            "revision_ids": list(self.revision_ids),
            "fidelity": self.fidelity.value,
            "written_files": [str(item) for item in self.written_files],
            "structural_checks": list(self.structural_checks),
            "unsupported_operators": list(self.unsupported_operators),
            "metadata_changes": list(self.metadata_changes),
            "dtype_layout_changes": list(self.dtype_layout_changes),
            "unresolved_shapes": list(self.unresolved_shapes),
            "custom_runtime_needs": list(self.custom_runtime_needs),
            "edit_manifest": list(self.edit_manifest),
            "transformations": list(self.transformations),
            "warnings": list(self.warnings),
            "numerical_comparison": self.numerical_comparison,
        }


def report_from_json(path: Path | str) -> ExportReport:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"export report could not be read: {error}") from error
    if not isinstance(raw, dict):
        raise ExportError("export report is not an object")
    try:
        return _report_from_dict(raw, target)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ExportError(f"export report is malformed: {error}") from error


def _report_from_dict(raw: dict[str, Any], target: Path) -> ExportReport:
    return ExportReport(
        destination=Path(str(raw["destination"])),
        report_path=target,
        source_hash=str(raw["source_hash"]),
        target_hash=str(raw["target_hash"]),
        source_components=tuple(
            {str(key): str(value) for key, value in item.items()}
            for item in raw.get("source_components", [])
            if isinstance(item, dict)
        ),
        tool_version=str(raw.get("tool_version", "unknown")),
        dependencies=tuple(
            sorted(
                (str(key), str(value))
                for key, value in (
                    raw.get("dependencies", {}).items()
                    if isinstance(raw.get("dependencies"), dict)
                    else ()
                )
            )
        ),
        revision_ids=tuple(str(item) for item in raw.get("revision_ids", [])),
        fidelity=ExportFidelity(str(raw["fidelity"])),
        written_files=tuple(Path(str(item)) for item in raw["written_files"]),
        structural_checks=tuple(str(item) for item in raw["structural_checks"]),
        unsupported_operators=tuple(str(item) for item in raw["unsupported_operators"]),
        metadata_changes=tuple(str(item) for item in raw["metadata_changes"]),
        dtype_layout_changes=tuple(str(item) for item in raw["dtype_layout_changes"]),
        unresolved_shapes=tuple(str(item) for item in raw["unresolved_shapes"]),
        custom_runtime_needs=tuple(str(item) for item in raw["custom_runtime_needs"]),
        edit_manifest=tuple(
            dict(item) for item in raw["edit_manifest"] if isinstance(item, dict)
        ),
        transformations=tuple(
            dict(item)
            for item in raw.get("transformations", [])
            if isinstance(item, dict)
        ),
        warnings=tuple(str(item) for item in raw.get("warnings", [])),
        numerical_comparison=(
            dict(raw["numerical_comparison"])
            if isinstance(raw.get("numerical_comparison"), dict)
            else None
        ),
    )


def _attribute_proto(attribute: Attribute) -> AttributeProto:
    if attribute.kind in {
        AttrKind.TENSOR,
        AttrKind.TENSORS,
        AttrKind.GRAPH,
        AttrKind.GRAPHS,
    }:
        raise ExportError(
            f"attribute {attribute.name!r} has a reference-valued type that "
            "Phase 4 does not rewrite"
        )
    value = (
        list(attribute.value) if isinstance(attribute.value, tuple) else attribute.value
    )
    return helper.make_attribute(attribute.name, value)


def _copy_node(node: NodeProto) -> NodeProto:
    copied = NodeProto()
    copied.CopyFrom(node)
    return copied


def _node_order(index: ModelIndex, model: ModelProto) -> list[tuple[str, NodeProto]]:
    indexed = index.main_graph.nodes
    if len(indexed) != len(model.graph.node):
        raise ExportError("source graph changed between lazy indexing and export")
    return [
        (entry.id, _copy_node(node))
        for entry, node in zip(indexed, model.graph.node, strict=True)
    ]


def _node_map(order: list[tuple[str, NodeProto]]) -> dict[str, NodeProto]:
    return {node_id: node for node_id, node in order}


def _value_names(document: Document) -> dict[str, str]:
    return {value.id: value.name for value in document.main_graph.values if value.name}


_TENSOR_DATA_TYPES: Final[dict[str, int]] = {
    "float32": TensorProto.FLOAT,
    "float64": TensorProto.DOUBLE,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "int16": TensorProto.INT16,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "uint32": TensorProto.UINT32,
    "int64": TensorProto.INT64,
    "uint64": TensorProto.UINT64,
    "bool": TensorProto.BOOL,
}


def _tensor_data_type(element_type: str) -> int:
    try:
        return _TENSOR_DATA_TYPES[element_type]
    except KeyError:
        raise ExportError(
            f"cannot encode generated tensor element type {element_type!r}"
        ) from None


def _generated_initializer(
    tensor_id: str,
    values: Mapping[str, str],
    element_type: str,
    dims: tuple[int, ...],
    raw: bytes,
) -> TensorProto:
    try:
        name = values[tensor_id]
    except KeyError:
        raise ExportError(f"generated tensor {tensor_id!r} has no ONNX name") from None
    return helper.make_tensor(
        name,
        _tensor_data_type(element_type),
        dims,
        raw,
        raw=True,
    )


def _generated_node(node: Node, values: Mapping[str, str]) -> NodeProto:
    try:
        inputs = [values[value_id] for value_id in node.inputs]
        outputs = [values[value_id] for value_id in node.outputs]
    except KeyError as error:
        raise ExportError(f"generated node has an unnamed value: {error}") from error
    generated = helper.make_node(
        node.op_type,
        inputs,
        outputs,
        name=node.source_name or "",
        domain=node.domain,
    )
    generated.attribute.extend(_attribute_proto(item) for item in node.attributes)
    return generated


def _update_value_info(
    model: ModelProto,
    name: str,
    element_type: str,
    shape: object,
) -> bool:
    entries = (*model.graph.input, *model.graph.output, *model.graph.value_info)
    value = next((item for item in entries if item.name == name), None)
    if value is None:
        return False
    value.type.tensor_type.elem_type = _tensor_data_type(element_type)
    value.type.tensor_type.ClearField("shape")
    if not isinstance(shape, tuple):
        raise ExportError(f"generated value {name!r} has no exportable shape")
    for extent in shape:
        dimension = value.type.tensor_type.shape.dim.add()
        if isinstance(extent, int):
            dimension.dim_value = extent
        elif isinstance(extent, str):
            dimension.dim_param = extent
    return True


def _apply_graph_commands(
    model: ModelProto,
    index: ModelIndex,
    base: Document,
    commands: Iterable[EditCommand],
    token: CancellationToken | None = None,
) -> bool:
    order = _node_order(index, model)
    nodes = _node_map(order)
    values = _value_names(base)
    changed = False

    for command in commands:
        if token is not None:
            token.raise_if_cancelled()
        if isinstance(command, ReplaceTensorBytes):
            continue
        if command.graph_id != base.entry_graph:
            raise ExportError("ONNX graph export only supports entry-graph commands")
        changed = True
        if isinstance(command, ResizeTensor):
            for change in command.value_changes:
                updated = _update_value_info(
                    model,
                    change.after.name,
                    change.after.element_type or command.after.element_type,
                    change.after.shape,
                )
                if not updated and not command.tensor_id.endswith(
                    f"#{change.after.name}"
                ):
                    raise ExportError(
                        f"resized value {change.after.name!r} has no ONNX metadata"
                    )
        elif isinstance(command, RenameNode):
            nodes[command.node_id].name = command.after_name
        elif isinstance(command, SetAttribute):
            node = nodes[command.node_id]
            attribute = command.after or command.before
            assert attribute is not None
            retained = [item for item in node.attribute if item.name != attribute.name]
            del node.attribute[:]
            node.attribute.extend(retained)
            if command.after is not None:
                node.attribute.append(_attribute_proto(command.after))
        elif isinstance(command, ReplaceOperator):
            node = nodes[command.node_id]
            node.domain = command.after_domain
            node.op_type = command.after_op_type
        elif isinstance(command, ReconnectInput):
            try:
                nodes[command.node_id].input[command.input_index] = values[
                    command.after_value_id
                ]
            except (IndexError, KeyError) as error:
                raise ExportError(
                    f"cannot materialize reconnection: {error}"
                ) from error
        elif isinstance(command, InsertUnaryNode):
            source_name = values.get(command.source_value_id)
            if source_name is None:
                raise ExportError(
                    f"insert source {command.source_value_id!r} has no ONNX name"
                )
            output_name = command.output_value.name
            inserted = helper.make_node(
                command.node.op_type,
                [source_name],
                [output_name],
                name=command.node.source_name or "",
                domain=command.node.domain,
            )
            target = nodes[command.target_node_id]
            target.input[command.target_input_index] = output_name
            target_position = next(
                position
                for position, (node_id, _node) in enumerate(order)
                if node_id == command.target_node_id
            )
            order.insert(target_position, (command.node.id, inserted))
            nodes[command.node.id] = inserted
            values[command.output_value.id] = output_name
        elif isinstance(command, RemoveUnaryNode):
            target = nodes[command.target_node_id]
            source_name = values.get(command.source_value_id)
            if source_name is None:
                raise ExportError(
                    f"remove source {command.source_value_id!r} has no ONNX name"
                )
            target.input[command.target_input_index] = source_name
            order = [
                (node_id, node) for node_id, node in order if node_id != command.node.id
            ]
            nodes.pop(command.node.id, None)
            values.pop(command.output_value.id, None)
        elif isinstance(command, QuantizeGraph):
            for value in (
                command.scale_value,
                command.zero_value,
                command.quantized_value,
                command.dequantized_value,
            ):
                if not value.name:
                    raise ExportError(f"generated value {value.id!r} has no ONNX name")
                values[value.id] = value.name
            scale = _generated_initializer(
                command.scale_tensor.id,
                values,
                command.scale_tensor.element_type,
                command.scale_tensor.dims,
                command.scale_bytes,
            )
            zero = _generated_initializer(
                command.zero_tensor.id,
                values,
                command.zero_tensor.element_type,
                command.zero_tensor.dims,
                command.zero_bytes,
            )
            model.graph.initializer.extend((scale, zero))
            quantize = _generated_node(command.quantize_node, values)
            dequantize = _generated_node(command.dequantize_node, values)
            try:
                positions = [
                    next(
                        position
                        for position, (node_id, _node) in enumerate(order)
                        if node_id == consumer_id
                    )
                    for consumer_id, _input_index in command.consumer_inputs
                ]
                for consumer_id, input_index in command.consumer_inputs:
                    nodes[consumer_id].input[input_index] = (
                        command.dequantized_value.name
                    )
            except (IndexError, KeyError, StopIteration) as error:
                raise ExportError(
                    f"cannot materialize Q/DQ consumer boundary: {error}"
                ) from error
            order[min(positions) : min(positions)] = [
                (command.quantize_node.id, quantize),
                (command.dequantize_node.id, dequantize),
            ]
            nodes[command.quantize_node.id] = quantize
            nodes[command.dequantize_node.id] = dequantize

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(node for _node_id, node in order)
    return changed


def _tensor_proto(model: ModelProto, name: str) -> TensorProto:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return cast(TensorProto, tensor)
    raise ExportError(f"entry graph has no initializer {name!r}")


def _clear_tensor_payload(tensor: TensorProto) -> None:
    for field in (
        "float_data",
        "int32_data",
        "string_data",
        "int64_data",
        "double_data",
        "uint64_data",
        "external_data",
    ):
        tensor.ClearField(field)
    tensor.data_location = TensorProto.DEFAULT


def _embedded_tensor_edits(
    model: ModelProto,
    index: ModelIndex,
    commands: tuple[EditCommand, ...],
    tensor_reader: Callable[[str], bytes],
    token: CancellationToken | None = None,
) -> bool:
    edited_ids = {
        command.edit.tensor_id
        for command in commands
        if isinstance(command, ReplaceTensorBytes)
        and index.tensor(command.edit.tensor_id).storage
        in {TensorStorage.EMBEDDED_RAW, TensorStorage.EMBEDDED_TYPED}
    }
    for tensor_id in edited_ids:
        if token is not None:
            token.raise_if_cancelled()
        indexed = index.tensor(tensor_id)
        tensor = _tensor_proto(model, indexed.name)
        raw = tensor_reader(tensor_id)
        expected = indexed.expected_byte_length
        if expected is not None and len(raw) != expected:
            raise ExportError(
                f"edited tensor {tensor_id!r} has {len(raw)} bytes, expected {expected}"
            )
        _clear_tensor_payload(tensor)
        tensor.raw_data = raw
    resized = {
        command.tensor_id: command
        for command in commands
        if isinstance(command, ResizeTensor)
    }
    for tensor_id, command in resized.items():
        if token is not None:
            token.raise_if_cancelled()
        indexed = index.tensor(tensor_id)
        tensor = _tensor_proto(model, indexed.name)
        raw = tensor_reader(tensor_id)
        if len(raw) != len(command.after_bytes):
            raise ExportError(
                f"resized tensor {tensor_id!r} has {len(raw)} bytes, "
                f"expected {len(command.after_bytes)}"
            )
        _clear_tensor_payload(tensor)
        del tensor.dims[:]
        tensor.dims.extend(command.after.dims)
        tensor.data_type = _tensor_data_type(command.after.element_type)
        tensor.raw_data = raw
    return bool(edited_ids or resized)


def _external_targets(
    index: ModelIndex,
    destination: Path,
    commands: tuple[EditCommand, ...],
) -> tuple[dict[Path, Path], bool]:
    mapping: dict[Path, Path] = {}
    relocated = False
    resized = {
        command.tensor_id for command in commands if isinstance(command, ResizeTensor)
    }
    needed_sources = {
        tensor.external.resolved_path.resolve()
        for tensor in index.iter_tensors()
        if tensor.id not in resized
        and tensor.external is not None
        and tensor.external.resolved_path is not None
    }
    for source in index.external_files:
        if source.resolve() not in needed_sources:
            continue
        try:
            relative = source.relative_to(index.path.parent)
        except ValueError as error:
            raise ExportError(
                f"external file {source} is outside the model directory"
            ) from error
        try:
            target = resolve_within(destination.parent, relative.as_posix())
        except UnsafePathError as error:
            raise ExportError(f"unsafe external export path: {error}") from error
        if target.resolve() == source.resolve():
            relative = Path(f"{destination.stem}.data") / relative
            try:
                target = resolve_within(destination.parent, relative.as_posix())
            except UnsafePathError as error:
                raise ExportError(f"unsafe relocated export path: {error}") from error
            relocated = True
        mapping[source.resolve()] = target
    return mapping, relocated


def _iter_graph_protos(graph: GraphProto) -> Iterator[GraphProto]:
    """Yield ``graph`` and every subgraph nested in node attributes."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _iter_graph_protos(attribute.g)
            for nested in attribute.graphs:
                yield from _iter_graph_protos(nested)


def _relocate_external_references(
    model: ModelProto,
    index: ModelIndex,
    mapping: dict[Path, Path],
    destination: Path,
) -> None:
    # Every graph is walked: a subgraph initializer left pointing at the
    # source directory would make the export silently read the source's files
    # (audit finding: only main-graph initializers were rewritten).
    source_root = index.path.resolve().parent
    for graph in _iter_graph_protos(model.graph):
        for tensor in graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            for entry in tensor.external_data:
                if entry.key != "location":
                    continue
                source = (source_root / entry.value).resolve()
                target = mapping.get(source)
                if target is not None:
                    entry.value = target.relative_to(destination.parent).as_posix()


def _generated_external_location(
    destination: Path,
    mapping: Mapping[Path, Path],
) -> str:
    used = {
        target.relative_to(destination.parent).as_posix() for target in mapping.values()
    }
    candidate = f"{destination.name}.data"
    suffix = 1
    while candidate in used:
        candidate = f"{destination.name}.generated-{suffix}.data"
        suffix += 1
    return candidate


def _stage_external_files(
    stage: Path,
    destination: Path,
    mapping: dict[Path, Path],
    expected_hashes: Mapping[Path, str],
    token: CancellationToken | None = None,
) -> dict[Path, Path]:
    staged: dict[Path, Path] = {}
    for source, final in mapping.items():
        relative = final.relative_to(destination.parent)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(
            source,
            target,
            token,
            expected_hash=expected_hashes.get(source),
        )
        staged[source] = target
    return staged


def _copy_file(
    source: Path,
    target: Path,
    token: CancellationToken | None,
    *,
    expected_hash: str | None = None,
) -> None:
    """Copy with bounded memory and cooperative cancellation."""
    digest = hashlib.sha256()
    with open(source, "rb") as reader, open(target, "xb") as writer:
        while chunk := reader.read(4 * 1024 * 1024):
            if token is not None:
                token.raise_if_cancelled()
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    actual_hash = f"sha256:{digest.hexdigest()}"
    if expected_hash is not None and actual_hash != expected_hash:
        raise ExportError(
            f"{source.name} changed while it was being staged: expected "
            f"{expected_hash}, copied {actual_hash}"
        )


def _splice_external_edits(
    index: ModelIndex,
    commands: tuple[EditCommand, ...],
    staged: dict[Path, Path],
    token: CancellationToken | None = None,
) -> None:
    resized = {
        command.tensor_id for command in commands if isinstance(command, ResizeTensor)
    }
    for command in commands:
        if token is not None:
            token.raise_if_cancelled()
        if not isinstance(command, ReplaceTensorBytes):
            continue
        if command.edit.tensor_id in resized:
            continue
        tensor = index.tensor(command.edit.tensor_id)
        if tensor.storage is not TensorStorage.EXTERNAL:
            continue
        reference = tensor.external
        if reference is None or reference.resolved_path is None:
            raise ExportError(f"external tensor {tensor.id!r} is unusable")
        target = staged[reference.resolved_path.resolve()]
        absolute = reference.offset + command.edit.offset
        with open(target, "r+b") as handle:
            handle.seek(absolute)
            before = handle.read(command.edit.length)
            if before != command.edit.before:
                raise ExportError(
                    f"external edit for {tensor.id!r} does not match source bytes"
                )
            handle.seek(absolute)
            handle.write(command.edit.after)
            handle.flush()
            os.fsync(handle.fileno())


def _unsupported_operators(document: Document) -> tuple[str, ...]:
    opsets: dict[str, int] = {}
    for namespace, value in document.extensions:
        if namespace == "x-onnx.model" and isinstance(value, dict):
            entries = value.get("opset_imports")
            if isinstance(entries, list):
                for entry in entries:
                    domain = entry.get("domain") if isinstance(entry, dict) else None
                    version = entry.get("version") if isinstance(entry, dict) else None
                    if (
                        isinstance(entry, dict)
                        and isinstance(domain, str)
                        and isinstance(version, int)
                        and not isinstance(version, bool)
                    ):
                        opsets[domain] = version
    unsupported: list[str] = []
    for graph in document.graphs.values():
        for node in graph.nodes:
            domain = "" if node.domain == "ai.onnx" else node.domain
            version = opsets.get(domain)
            if version is None:
                unsupported.append(node.qualified_op_type)
                continue
            try:
                onnx.defs.get_schema(node.op_type, version, domain)
            except onnx.defs.SchemaError:
                unsupported.append(node.qualified_op_type)
    return tuple(dict.fromkeys(unsupported))


def _unresolved_shapes(document: Document) -> tuple[str, ...]:
    return tuple(
        value.id
        for graph in document.graphs.values()
        for value in graph.values
        if value.shape is None
        or any(not isinstance(dimension, int) for dimension in value.shape)
    )


def _custom_runtime_needs(document: Document) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            node.domain
            for graph in document.graphs.values()
            for node in graph.nodes
            if node.domain not in {"", "ai.onnx"}
        )
    )


def _dtype_layout_changes(before: Document, after: Document) -> tuple[str, ...]:
    changes: list[str] = []
    for tensor_id, current in after.tensors.items():
        original = before.tensors.get(tensor_id)
        if original is None:
            changes.append(f"{tensor_id}: tensor added")
            continue
        if (
            original.element_type,
            original.dims,
            original.storage,
        ) != (
            current.element_type,
            current.dims,
            current.storage,
        ):
            changes.append(
                f"{tensor_id}: {original.element_type}{original.dims}/"
                f"{original.storage.value} -> {current.element_type}{current.dims}/"
                f"{current.storage.value}"
            )
    changes.extend(
        f"{tensor_id}: tensor removed"
        for tensor_id in before.tensors
        if tensor_id not in after.tensors
    )
    return tuple(changes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_components(index: ModelIndex) -> tuple[dict[str, str], ...]:
    return (
        {"location": index.path.name, "content_hash": index.model_hash},
        *(
            {"location": location, "content_hash": content_hash}
            for location, content_hash in index.external_hashes
        ),
    )


def _dependency_versions() -> tuple[tuple[str, str], ...]:
    return (
        ("numpy", np.__version__),
        ("onnx", onnx.__version__),
        ("python", sys.version.split()[0]),
    )


def _publish(
    stage: Path,
    destination: Path,
    report_name: str,
) -> tuple[Path, ...]:
    relative_files = sorted(
        (
            item.relative_to(stage)
            for item in stage.rglob("*")
            if item.is_file() and item.name != report_name
        ),
        key=lambda item: (item == Path(destination.name), str(item)),
    )
    outputs = tuple(destination.parent / relative for relative in relative_files)
    report_target = destination.parent / report_name
    for target in (*outputs, report_target):
        if target.exists():
            raise ExportError(f"export target already exists: {target}")
    published: list[Path] = []
    created_directories: list[Path] = []
    try:
        # External payloads first, model last.  A reader can never observe a
        # model whose referenced files have not yet arrived.
        for relative, target in zip(relative_files, outputs, strict=True):
            parent = target.parent
            while parent != destination.parent and not parent.exists():
                created_directories.append(parent)
                parent = parent.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, target)
            published.append(target)
        os.replace(stage / report_name, report_target)
        published.append(report_target)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        for directory in sorted(
            set(created_directories),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    return tuple(published)


def export_revision(
    document: Document,
    revisions: Iterable[Revision],
    destination: Path | str,
    *,
    tensor_reader: Callable[[str], bytes],
    external_data_threshold: int = DEFAULT_EXTERNAL_THRESHOLD,
    smoke_inputs: Mapping[str, NDArray[np.generic]] | None = None,
    approve_numerical_execution: bool = False,
    numerical_atol: float = 1e-5,
    numerical_rtol: float = 1e-5,
    numerical_timeout_seconds: float = 30.0,
    token: CancellationToken | None = None,
) -> ExportReport:
    """Materialize and atomically publish one committed working revision."""
    target = Path(destination).resolve()
    source = Path(document.source.path).resolve()
    if target == source:
        raise ExportError("exports never overwrite the source artifact")
    if target.suffix.lower() != ".onnx":
        raise ExportError("ONNX exports must use an .onnx destination")
    if external_data_threshold < 0:
        raise ExportError("external data threshold must be non-negative")
    if token is not None:
        token.raise_if_cancelled()
    target.parent.mkdir(parents=True, exist_ok=True)

    # The whole pre-staging phase honors the declared contract: any failure —
    # a protobuf DecodeError from the official bindings, a missing name, an
    # exhausted iterator — surfaces as ExportError, never as a raw internal
    # exception.
    try:
        applied = tuple(revisions)
        commands = tuple(
            command for revision in applied for command in revision.commands
        )
        index = index_model(source, token=token)
        if index.content_hash != document.source.content_hash:
            raise ExportError(
                "the source artifact or one of its external files changed after open"
            )
        base = index_to_document(index)
        if token is not None:
            token.raise_if_cancelled()
        model = onnx.load_model(source, load_external_data=False)
        if token is not None:
            token.raise_if_cancelled()
        graph_changed = _apply_graph_commands(model, index, base, commands, token)
        embedded_changed = _embedded_tensor_edits(
            model,
            index,
            commands,
            tensor_reader,
            token,
        )
        external_mapping, relocated = _external_targets(index, target, commands)
        expected_external_hashes = {
            (index.path.parent / location).resolve(): content_hash
            for location, content_hash in index.external_hashes
        }
        generated_external_location = _generated_external_location(
            target,
            external_mapping,
        )
        if relocated:
            _relocate_external_references(model, index, external_mapping, target)
    except (ExportError, OperationCancelled):
        raise
    except Exception as error:
        raise ExportError(f"export preparation failed: {error}") from error

    stage_owner = tempfile.TemporaryDirectory(
        prefix=f".{target.stem}-",
        dir=target.parent,
    )
    stage = Path(stage_owner.name)
    staged_model = stage / target.name
    report_name = f"{target.name}.report.json"
    try:
        staged_external = _stage_external_files(
            stage,
            target,
            external_mapping,
            expected_external_hashes,
            token,
        )
        _splice_external_edits(index, commands, staged_external, token)

        model_needs_save = graph_changed or embedded_changed or relocated
        if model_needs_save:
            if token is not None:
                token.raise_if_cancelled()
            onnx.save_model(
                model,
                staged_model,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=generated_external_location,
                size_threshold=external_data_threshold,
                convert_attribute=False,
            )
        else:
            _copy_file(
                source,
                staged_model,
                token,
                expected_hash=index.model_hash,
            )

        # Only checks the export path itself performed are reported; commit-time
        # validation is already recorded per revision (audit finding: this list
        # previously asserted checks the exporter never ran).
        structural_checks = []
        if token is not None:
            token.raise_if_cancelled()
        try:
            onnx.checker.check_model(staged_model, full_check=True)
        except Exception as error:
            raise ExportError(
                f"onnx.checker rejected the staged artifact: {error}"
            ) from error
        structural_checks.append("onnx.checker full_check passed")
        if token is not None:
            token.raise_if_cancelled()
        reopened = index_model(staged_model, token=token)
        structural_checks.append("lazy reopen passed")
        current_source = index_model(source, token=token)
        if current_source.content_hash != index.content_hash:
            raise ExportError(
                "the source artifact changed while the export was being staged"
            )
        structural_checks.append("source snapshot remained stable")
        numerical_comparison = None
        if smoke_inputs is not None:
            numerical_comparison = compare_numerically(
                source,
                staged_model,
                smoke_inputs,
                approved=approve_numerical_execution,
                atol=numerical_atol,
                rtol=numerical_rtol,
                timeout_seconds=numerical_timeout_seconds,
            ).to_json()

        unsupported = _unsupported_operators(document)
        metadata_changes = tuple(
            command_summary(command)
            for command in commands
            if isinstance(command, (RenameNode, SetAttribute))
        )
        dtype_changes = _dtype_layout_changes(base, document)
        manifest = tuple(command_to_json(command) for command in commands)
        transformations: tuple[dict[str, object], ...] = tuple(
            {
                "target_id": command_target(command),
                "manifest": command.transformation.to_json(),
                "preview": command.preview.to_json(),
            }
            for command in commands
            if isinstance(command, (ReplaceTensorBytes, ResizeTensor, QuantizeGraph))
            and command.transformation is not None
            and command.preview is not None
        )
        # Weight edits grade identically no matter where the bytes live: an
        # external-file splice changes the model exactly as much as an
        # embedded one (audit finding: external edits previously reported
        # LOSSLESS).
        external_weights_changed = any(
            isinstance(command, (ReplaceTensorBytes, ResizeTensor))
            and index.tensor(command_target(command)).storage is TensorStorage.EXTERNAL
            for command in commands
        )
        fidelity = (
            ExportFidelity.LOSSY
            if transformations
            else (
                ExportFidelity.LOSSLESS
                if not graph_changed
                and not embedded_changed
                and not relocated
                and not external_weights_changed
                else ExportFidelity.SEMANTICALLY_EQUIVALENT
            )
        )
        written_files = tuple(
            target.parent / item.relative_to(stage)
            for item in sorted(stage.rglob("*"))
            if item.is_file()
        )
        report = ExportReport(
            destination=target,
            report_path=target.parent / report_name,
            source_hash=document.source.content_hash,
            target_hash=reopened.content_hash,
            source_components=_source_components(index),
            tool_version=__version__,
            dependencies=_dependency_versions(),
            revision_ids=tuple(revision.id for revision in applied),
            fidelity=fidelity,
            written_files=written_files,
            structural_checks=tuple(structural_checks),
            unsupported_operators=unsupported,
            metadata_changes=metadata_changes,
            dtype_layout_changes=dtype_changes,
            unresolved_shapes=_unresolved_shapes(document),
            custom_runtime_needs=_custom_runtime_needs(document),
            edit_manifest=manifest,
            transformations=transformations,
            warnings=(
                ("External files were relocated to protect the source.",)
                if relocated
                else ()
            ),
            numerical_comparison=numerical_comparison,
        )
        _write_json(stage / report_name, report.to_json())
        if token is not None:
            token.raise_if_cancelled()
        _publish(stage, target, report_name)
        return report
    except (ExportError, OperationCancelled):
        raise
    except Exception as error:
        raise ExportError(str(error)) from error
    finally:
        stage_owner.cleanup()
