"""Axis-capable Scan execution and semantics-preserving axis normalization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, helper
from onnx.reference.op_run import OpRun

__all__ = [
    "AxisAwareScan",
    "ScanNormalizationError",
    "ScanNormalizationReport",
    "normalize_scan_axes",
]


class ScanNormalizationError(ValueError):
    """A nonzero Scan axis could not be rewritten without guessing its rank."""


def _values(
    raw: Sequence[int] | np.ndarray[Any, Any] | None,
    count: int,
    default: int,
    *,
    label: str,
) -> list[int]:
    values = [] if raw is None else [int(item) for item in raw]
    if len(values) > count:
        raise RuntimeError(f"{label} has {len(values)} entries for {count} values")
    values.extend(default for _ in range(count - len(values)))
    return values


def _axis(value: int, rank: int, *, label: str) -> int:
    resolved = value + rank if value < 0 else value
    if resolved < 0 or resolved >= rank:
        raise RuntimeError(f"{label} axis {value} is outside rank {rank}")
    return resolved


class AxisAwareScan(OpRun):
    """Complete tensor-axis and direction support for the standard ONNX Scan."""

    op_domain = ""
    body: Any
    input_names: list[str]
    num_scan_inputs: int
    output_names: list[str]
    scan_input_axes: Sequence[int] | None
    scan_input_directions: Sequence[int] | None
    scan_output_axes: Sequence[int] | None
    scan_output_directions: Sequence[int] | None
    _run_body: Callable[[dict[str, Any]], list[Any]]

    def __init__(
        self,
        onnx_node: NodeProto,
        run_params: dict[str, Any],
        schema: Any = None,
    ) -> None:
        super().__init__(onnx_node, run_params, schema)
        if not hasattr(self.body, "run"):
            raise RuntimeError("Scan body must be an executable graph")
        if self.num_scan_inputs <= 0:
            raise RuntimeError(
                f"Scan requires num_scan_inputs > 0, got {self.num_scan_inputs}"
            )
        self.input_names = self.body.input_names
        self.output_names = self.body.output_names

    def need_context(self) -> bool:
        return True

    def _run(
        self,
        *args: np.ndarray[Any, Any],
        context: dict[str, Any] | None = None,
        body: Any = None,
        num_scan_inputs: int | None = None,
        scan_input_axes: Sequence[int] | None = None,
        scan_input_directions: Sequence[int] | None = None,
        scan_output_axes: Sequence[int] | None = None,
        scan_output_directions: Sequence[int] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], ...]:
        state_count = len(args) - int(self.num_scan_inputs)
        output_count = len(self.output_names) - state_count
        if state_count < 0 or output_count < 0:
            raise RuntimeError("Scan state/input/output arity is inconsistent")

        scan_values = [np.asarray(value) for value in args[state_count:]]
        input_axes = _values(
            self.scan_input_axes,
            len(scan_values),
            0,
            label="scan_input_axes",
        )
        input_directions = _values(
            self.scan_input_directions,
            len(scan_values),
            0,
            label="scan_input_directions",
        )
        resolved_input_axes = [
            _axis(axis, value.ndim, label="Scan input")
            for axis, value in zip(input_axes, scan_values, strict=True)
        ]
        if any(direction not in {0, 1} for direction in input_directions):
            raise RuntimeError("Scan input directions must be 0 or 1")
        lengths = [
            int(value.shape[axis])
            for value, axis in zip(scan_values, resolved_input_axes, strict=True)
        ]
        if len(set(lengths)) > 1:
            raise RuntimeError(
                f"Scan inputs have different sequence lengths: {lengths}"
            )
        iterations = lengths[0] if lengths else 0

        output_axes = _values(
            self.scan_output_axes,
            output_count,
            0,
            label="scan_output_axes",
        )
        output_directions = _values(
            self.scan_output_directions,
            output_count,
            0,
            label="scan_output_directions",
        )
        if any(direction not in {0, 1} for direction in output_directions):
            raise RuntimeError("Scan output directions must be 0 or 1")

        state_names_in = self.input_names[:state_count]
        state_names_out = self.output_names[:state_count]
        scan_names_in = self.input_names[state_count:]
        scan_names_out = self.output_names[state_count:]
        states = [np.asarray(value) for value in args[:state_count]]
        accumulated: list[list[np.ndarray[Any, Any]]] = [
            [] for _ in range(output_count)
        ]

        for iteration in range(iterations):
            inputs = dict(context or {})
            inputs.update(dict(zip(state_names_in, states, strict=True)))
            for name, value, axis, direction in zip(
                scan_names_in,
                scan_values,
                resolved_input_axes,
                input_directions,
                strict=True,
            ):
                position = iteration if direction == 0 else iterations - iteration - 1
                inputs[name] = np.take(value, position, axis=axis)
            outputs_list = self._run_body(inputs)
            outputs = dict(zip(self.output_names, outputs_list, strict=True))
            states = [np.asarray(outputs[name]) for name in state_names_out]
            for index, name in enumerate(scan_names_out):
                accumulated[index].append(np.asarray(outputs[name]))

        if iterations == 0 and accumulated:
            raise RuntimeError(
                "Scan with zero scan-input length and scan outputs is not supported"
            )
        for values, raw_axis, direction in zip(
            accumulated,
            output_axes,
            output_directions,
            strict=True,
        ):
            if direction == 1:
                values.reverse()
            rank = values[0].ndim + 1
            axis = _axis(raw_axis, rank, label="Scan output")
            states.append(np.stack(values, axis=axis))
        return tuple(states)


# ReferenceEvaluator identifies custom operators by the class name.
AxisAwareScan.__name__ = "Scan"


@dataclass(frozen=True, slots=True)
class ScanNormalizationReport:
    rewritten_scans: int
    inserted_transposes: int
    node_names: tuple[str, ...]


def _rank(value_info: onnx.ValueInfoProto) -> int | None:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return None
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return len(tensor_type.shape.dim)


def _ranks(graph: GraphProto) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        rank = _rank(value)
        if rank is not None:
            ranks[value.name] = rank
    for initializer in graph.initializer:
        ranks[initializer.name] = len(initializer.dims)
    return ranks


def _attribute(node: NodeProto, name: str) -> AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def _ints(node: NodeProto, name: str, count: int, default: int = 0) -> list[int]:
    attribute = _attribute(node, name)
    values = [] if attribute is None else [int(item) for item in attribute.ints]
    if len(values) > count:
        raise ScanNormalizationError(
            f"{node.name or node.output[0]!r} has too many {name} entries"
        )
    values.extend(default for _ in range(count - len(values)))
    return values


def _set_ints(node: NodeProto, name: str, values: Iterable[int]) -> None:
    attribute = _attribute(node, name)
    if attribute is None:
        node.attribute.append(helper.make_attribute(name, list(values)))
        return
    del attribute.ints[:]
    attribute.ints.extend(values)


def _unique(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _move_to_front(rank: int, axis: int) -> list[int]:
    return [axis, *(item for item in range(rank) if item != axis)]


def _move_front_to(rank: int, axis: int) -> list[int]:
    return [*range(1, axis + 1), 0, *range(axis + 1, rank)]


def _normalize_graph(
    graph: GraphProto,
    rewritten: list[str],
    transpose_count: list[int],
) -> None:
    ranks = _ranks(graph)
    used_values = {
        value.name for value in (*graph.input, *graph.value_info, *graph.output)
    }
    used_values.update(initializer.name for initializer in graph.initializer)
    used_values.update(
        output for node in graph.node for output in node.output if output
    )
    used_nodes = {node.name for node in graph.node if node.name}
    rebuilt: list[NodeProto] = []

    for original in list(graph.node):
        node = NodeProto()
        node.CopyFrom(original)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                _normalize_graph(attribute.g, rewritten, transpose_count)
            elif attribute.type == AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    _normalize_graph(child, rewritten, transpose_count)
        if node.op_type != "Scan" or node.domain:
            rebuilt.append(node)
            continue

        count_attribute = _attribute(node, "num_scan_inputs")
        if count_attribute is None:
            raise ScanNormalizationError("Scan is missing num_scan_inputs")
        scan_input_count = int(count_attribute.i)
        state_count = len(node.input) - scan_input_count
        body_attribute = _attribute(node, "body")
        if body_attribute is None or body_attribute.type != AttributeProto.GRAPH:
            raise ScanNormalizationError("Scan is missing its body graph")
        scan_output_count = len(body_attribute.g.output) - state_count
        if state_count < 0 or scan_output_count < 0:
            raise ScanNormalizationError("Scan input/output arity is inconsistent")

        input_axes = _ints(node, "scan_input_axes", scan_input_count)
        output_axes = _ints(node, "scan_output_axes", scan_output_count)
        needs_rewrite = any(axis != 0 for axis in (*input_axes, *output_axes))
        if not needs_rewrite:
            rebuilt.append(node)
            continue

        label = node.name or (node.output[0] if node.output else "Scan")
        before: list[NodeProto] = []
        after: list[NodeProto] = []
        for index, raw_axis in enumerate(input_axes):
            input_index = state_count + index
            source = node.input[input_index]
            rank = ranks.get(source)
            if rank is None:
                raise ScanNormalizationError(
                    f"cannot normalize Scan {label!r} input {source!r}: rank is unknown"
                )
            axis = _axis(raw_axis, rank, label=f"Scan {label!r} input")
            if axis == 0:
                continue
            normalized = _unique(f"{source}__nneditor_scan_axis0", used_values)
            name = _unique(f"{label}__input_{index}_axis0", used_nodes)
            before.append(
                helper.make_node(
                    "Transpose",
                    [source],
                    [normalized],
                    name=name,
                    perm=_move_to_front(rank, axis),
                )
            )
            node.input[input_index] = normalized
            transpose_count[0] += 1

        for index, raw_axis in enumerate(output_axes):
            output_index = state_count + index
            destination = node.output[output_index]
            rank = ranks.get(destination)
            if rank is None:
                raise ScanNormalizationError(
                    f"cannot normalize Scan {label!r} output {destination!r}: "
                    "rank is unknown"
                )
            axis = _axis(raw_axis, rank, label=f"Scan {label!r} output")
            if axis == 0:
                continue
            normalized = _unique(
                f"{destination}__nneditor_scan_axis0",
                used_values,
            )
            name = _unique(f"{label}__output_{index}_restore_axis", used_nodes)
            node.output[output_index] = normalized
            after.append(
                helper.make_node(
                    "Transpose",
                    [normalized],
                    [destination],
                    name=name,
                    perm=_move_front_to(rank, axis),
                )
            )
            transpose_count[0] += 1

        _set_ints(node, "scan_input_axes", [0] * scan_input_count)
        _set_ints(node, "scan_output_axes", [0] * scan_output_count)
        rebuilt.extend((*before, node, *after))
        rewritten.append(label)

    graph.ClearField("node")
    graph.node.extend(rebuilt)


def normalize_scan_axes(
    model: ModelProto,
    *,
    in_place: bool = False,
    check: bool = True,
) -> tuple[ModelProto, ScanNormalizationReport]:
    """Move every nonzero Scan sequence axis to zero with explicit Transposes."""

    normalized = model
    if not in_place:
        normalized = ModelProto()
        normalized.CopyFrom(model)
    rewritten: list[str] = []
    transposes = [0]
    _normalize_graph(normalized.graph, rewritten, transposes)
    if check:
        onnx.checker.check_model(normalized)
    return normalized, ScanNormalizationReport(
        rewritten_scans=len(rewritten),
        inserted_transposes=transposes[0],
        node_names=tuple(rewritten),
    )
