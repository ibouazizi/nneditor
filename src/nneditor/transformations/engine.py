"""Preview and prepare the constrained Phase 5 transformations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nneditor.cancellation import CancellationToken
from nneditor.editing.commands import (
    EditCommand,
    QuantizeGraph,
    ReplaceTensorBytes,
    ResizeTensor,
    ValueMetadataChange,
    apply_command,
)
from nneditor.editing.cow import ByteSpanEdit, EditError
from nneditor.editing.revisions import ValidationState
from nneditor.editing.validation import FindingLevel, ValidationFinding
from nneditor.ir.core import (
    Attribute,
    AttrKind,
    Document,
    Graph,
    Node,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import NodeIdStability, node_id, value_id
from nneditor.transformations.schema import (
    Granularity,
    OperatorRepresentation,
    PruningMode,
    StorageEffect,
    TargetRuntime,
    TransformationKind,
    TransformationManifest,
    TransformationPreview,
)

__all__ = [
    "GraphQuantizationRequest",
    "LogicalPruningRequest",
    "StructuredPruningRequest",
    "TransformationEngine",
    "TransformationProposal",
    "TransformationRequest",
    "WeightQuantizationRequest",
]

TensorReader = Callable[[str], bytes]
_REFERENCE_QDQ_MIN_OPSET = 19


@dataclass(frozen=True, slots=True)
class WeightQuantizationRequest:
    tensor_id: str
    bit_width: int = 8
    signed: bool = True
    symmetric: bool = True
    granularity: Granularity = Granularity.PER_TENSOR
    axis: int | None = None
    target_runtime: TargetRuntime = TargetRuntime.PORTABLE_ONNX


@dataclass(frozen=True, slots=True)
class GraphQuantizationRequest:
    graph_id: str
    tensor_id: str
    bit_width: int = 8
    signed: bool = True
    symmetric: bool = True
    granularity: Granularity = Granularity.PER_TENSOR
    axis: int | None = None
    target_runtime: TargetRuntime = TargetRuntime.ONNX_REFERENCE
    calibration_provenance: tuple[tuple[str, str], ...] = ()
    backend_plugin: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalPruningRequest:
    tensor_id: str
    mode: PruningMode
    threshold: float = 0.0
    mask: tuple[bool, ...] = ()
    n: int = 2
    m: int = 4
    target_runtime: TargetRuntime = TargetRuntime.PORTABLE_ONNX


@dataclass(frozen=True, slots=True)
class StructuredPruningRequest:
    graph_id: str
    node_id: str
    keep_output_channels: tuple[int, ...]
    target_runtime: TargetRuntime = TargetRuntime.PORTABLE_ONNX


type TransformationRequest = (
    WeightQuantizationRequest
    | GraphQuantizationRequest
    | LogicalPruningRequest
    | StructuredPruningRequest
)
type TransformationCommand = ReplaceTensorBytes | ResizeTensor | QuantizeGraph
type TransformationHandler = Callable[
    [
        Document,
        object,
        str | None,
        CancellationToken | None,
    ],
    TransformationCommand,
]


@dataclass(frozen=True, slots=True)
class TransformationProposal:
    base_revision_id: str | None
    commands: tuple[EditCommand, ...]
    candidate: Document
    manifest: TransformationManifest | None
    preview: TransformationPreview | None
    findings: tuple[ValidationFinding, ...]

    @property
    def ok(self) -> bool:
        return bool(self.commands) and not any(
            finding.level is FindingLevel.ERROR for finding in self.findings
        )

    @property
    def validation_state(self) -> ValidationState:
        return ValidationState(
            ok=self.ok,
            findings=tuple(str(item) for item in self.findings),
        )


@dataclass(frozen=True, slots=True)
class _Quantized:
    quantized: NDArray[np.int8] | NDArray[np.uint8]
    dequantized: NDArray[np.float32]
    scale: tuple[float, ...]
    zero_point: tuple[int, ...]
    axis: int | None


def _float_tensor(
    document: Document,
    tensor_id: str,
    read: TensorReader,
    token: CancellationToken | None,
) -> tuple[TensorRef, bytes, NDArray[np.float32]]:
    try:
        tensor = document.tensors[tensor_id]
    except KeyError:
        raise EditError(f"document has no tensor {tensor_id!r}") from None
    if tensor.element_type != "float32":
        raise EditError("Phase 5 core transformations currently require float32")
    if token is not None:
        token.raise_if_cancelled()
    raw = read(tensor_id)
    expected = tensor.element_count * 4
    if len(raw) != expected:
        raise EditError(
            f"tensor has {len(raw)} readable bytes, expected {expected} from shape"
        )
    values = np.frombuffer(raw, dtype="<f4").reshape(tensor.dims).copy()
    if not np.all(np.isfinite(values)):
        raise EditError("transformations require finite tensor values")
    return tensor, raw, values


def _normalize_axis(axis: int | None, rank: int) -> int:
    if axis is None:
        raise EditError("per-channel quantization requires an axis")
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise EditError(f"axis {axis} is outside a rank-{rank} tensor")
    return normalized


def _quantize(
    values: NDArray[np.float32],
    *,
    bit_width: int,
    signed: bool,
    symmetric: bool,
    granularity: Granularity,
    axis: int | None,
) -> _Quantized:
    if bit_width != 8:
        raise EditError("the Phase 5 core supports only 8-bit quantization")
    if symmetric and not signed:
        raise EditError("symmetric core quantization requires signed integers")
    if values.size == 0:
        raise EditError("cannot quantize an empty tensor")
    qmin, qmax = (-127, 127) if symmetric else ((-128, 127) if signed else (0, 255))
    normalized_axis: int | None = None
    if granularity is Granularity.PER_TENSOR:
        minimum = np.asarray(values.min(), dtype=np.float32)
        maximum = np.asarray(values.max(), dtype=np.float32)
        parameter_shape: tuple[int, ...] = ()
    else:
        normalized_axis = _normalize_axis(axis, values.ndim)
        reduce_axes = tuple(
            index for index in range(values.ndim) if index != normalized_axis
        )
        minimum = values.min(axis=reduce_axes)
        maximum = values.max(axis=reduce_axes)
        parameter_shape = tuple(
            values.shape[index] if index == normalized_axis else 1
            for index in range(values.ndim)
        )
    if symmetric:
        magnitude = np.maximum(np.abs(minimum), np.abs(maximum))
        scale = np.where(magnitude == 0, 1.0, magnitude / qmax).astype(np.float32)
        zero = np.zeros_like(scale, dtype=np.int64)
    else:
        # Standard ONNX/TFLite practice: extend the representable range to
        # include zero so the zero-point is exact and lies inside
        # [qmin, qmax]. Clamping an out-of-range zero-point instead would
        # shift every dequantized value by the clamped amount.
        minimum = np.minimum(minimum, 0.0).astype(np.float32)
        maximum = np.maximum(maximum, 0.0).astype(np.float32)
        span = maximum - minimum
        scale = np.where(span == 0, 1.0, span / (qmax - qmin)).astype(np.float32)
        zero = np.clip(np.rint(qmin - minimum / scale), qmin, qmax).astype(np.int64)
    broadcast_scale = scale.reshape(parameter_shape) if parameter_shape else scale
    broadcast_zero = zero.reshape(parameter_shape) if parameter_shape else zero
    quantized_i64 = np.clip(
        np.rint(values / broadcast_scale) + broadcast_zero,
        qmin,
        qmax,
    ).astype(np.int64)
    quantized = (
        quantized_i64.astype(np.int8) if signed else quantized_i64.astype(np.uint8)
    )
    dequantized = ((quantized_i64 - broadcast_zero) * broadcast_scale).astype(
        np.float32
    )
    return _Quantized(
        quantized=quantized,
        dequantized=dequantized,
        scale=tuple(float(item) for item in np.asarray(scale).reshape(-1)),
        zero_point=tuple(int(item) for item in np.asarray(zero).reshape(-1)),
        axis=normalized_axis,
    )


def _errors(
    before: NDArray[np.float32],
    after: NDArray[np.float32],
) -> tuple[int, float, float, float]:
    difference = np.abs(after.astype(np.float64) - before.astype(np.float64))
    changed = int(np.count_nonzero(after != before))
    return (
        changed,
        float(difference.max(initial=0.0)),
        float(difference.mean()) if difference.size else 0.0,
        float(math.sqrt(float(np.mean(np.square(difference)))))
        if difference.size
        else 0.0,
    )


def _sparsity(values: NDArray[np.float32]) -> float:
    return float(np.count_nonzero(values == 0.0) / values.size) if values.size else 0.0


def _preview(
    tensor_id: str,
    before: NDArray[np.float32],
    after: NDArray[np.float32],
    *,
    source_bytes: int,
    result_bytes: int,
    theoretical_bytes: int,
    executable: bool,
    storage_reduction: bool,
    reason: str,
    changed_override: int | None = None,
) -> TransformationPreview:
    changed, maximum, mean, rmse = _errors(before, after)
    return TransformationPreview(
        tensor_id=tensor_id,
        element_count=int(before.size),
        changed_elements=changed if changed_override is None else changed_override,
        source_bytes=source_bytes,
        result_bytes=result_bytes,
        theoretical_packed_bytes=theoretical_bytes,
        max_abs_error=maximum,
        mean_abs_error=mean,
        rmse=rmse,
        error_basis="Elementwise error over the source and result tensor.",
        before_sparsity=_sparsity(before),
        after_sparsity=_sparsity(after),
        mathematical_conversion=True,
        executable_graph=executable,
        storage_reduction=storage_reduction,
        expected_acceleration=False,
        acceleration_reason=reason,
    )


def _opsets(document: Document) -> dict[str, int]:
    for namespace, value in document.extensions:
        if namespace != "x-onnx.model" or not isinstance(value, dict):
            continue
        imports = value.get("opset_imports")
        if not isinstance(imports, list):
            return {}
        found: dict[str, int] = {}
        for item in imports:
            if not isinstance(item, dict):
                continue
            domain = item.get("domain")
            version = item.get("version")
            if (
                isinstance(domain, str)
                and isinstance(version, int)
                and not isinstance(version, bool)
            ):
                found[domain] = version
        return found
    return {}


def _initializer_value(
    document: Document,
    graph: Graph,
    tensor_id: str,
) -> Value:
    if tensor_id not in graph.initializers or tensor_id not in document.tensors:
        raise EditError("transformation target is not an initializer")
    matches = tuple(
        value
        for value in graph.values
        if value.name and tensor_id.endswith(f"#{value.name}")
    )
    if len(matches) != 1:
        raise EditError("initializer-to-value binding is not uniquely provable")
    return matches[0]


class TransformationEngine:
    """Prepare transformations without mutating the working revision."""

    def __init__(
        self,
        read: TensorReader,
        *,
        handlers: Mapping[type[object], TransformationHandler] | None = None,
    ) -> None:
        self._read = read
        self._handlers: dict[type[object], TransformationHandler] = {
            WeightQuantizationRequest: self._dispatch_weight_quantization,
            GraphQuantizationRequest: self._dispatch_graph_quantization,
            LogicalPruningRequest: self._dispatch_logical_pruning,
            StructuredPruningRequest: self._dispatch_structured_pruning,
        }
        if handlers is not None:
            self._handlers.update(handlers)

    def register_handler(
        self,
        request_type: type[object],
        handler: TransformationHandler,
    ) -> None:
        """Register a transformation provider at the core dispatch seam."""
        self._handlers[request_type] = handler

    def prepare(
        self,
        document: Document,
        request: TransformationRequest,
        *,
        base_revision_id: str | None,
        token: CancellationToken | None = None,
    ) -> TransformationProposal:
        command: TransformationCommand
        try:
            handler = self._handlers.get(type(request))
            if handler is None:
                raise EditError(
                    f"no transformation handler is registered for "
                    f"{type(request).__name__}"
                )
            command = handler(document, request, base_revision_id, token)
            candidate = apply_command(document, command)
        except (EditError, KeyError, ValueError) as error:
            finding = ValidationFinding(
                "transform.rejected",
                FindingLevel.ERROR,
                str(error),
            )
            return TransformationProposal(
                base_revision_id,
                (),
                document,
                None,
                None,
                (finding,),
            )
        manifest = command.transformation
        preview = command.preview
        finding = ValidationFinding(
            "transform.valid",
            FindingLevel.INFO,
            "Transformation preconditions and tensor shape/dtype/target-"
            "runtime checks passed; ONNX structural validation "
            "(checker full_check and lazy reopen) runs at export.",
            command.tensor_id
            if not isinstance(command, ReplaceTensorBytes)
            else command.target,
        )
        return TransformationProposal(
            base_revision_id,
            (command,),
            candidate,
            manifest,
            preview,
            (finding,),
        )

    def _dispatch_weight_quantization(
        self,
        document: Document,
        request: object,
        _base_revision_id: str | None,
        token: CancellationToken | None,
    ) -> TransformationCommand:
        return self._weight_quantization(
            document,
            cast(WeightQuantizationRequest, request),
            token,
        )

    def _dispatch_graph_quantization(
        self,
        document: Document,
        request: object,
        base_revision_id: str | None,
        token: CancellationToken | None,
    ) -> TransformationCommand:
        return self._graph_quantization(
            document,
            cast(GraphQuantizationRequest, request),
            base_revision_id,
            token,
        )

    def _dispatch_logical_pruning(
        self,
        document: Document,
        request: object,
        _base_revision_id: str | None,
        token: CancellationToken | None,
    ) -> TransformationCommand:
        return self._logical_pruning(
            document,
            cast(LogicalPruningRequest, request),
            token,
        )

    def _dispatch_structured_pruning(
        self,
        document: Document,
        request: object,
        _base_revision_id: str | None,
        token: CancellationToken | None,
    ) -> TransformationCommand:
        return self._structured_pruning(
            document,
            cast(StructuredPruningRequest, request),
            token,
        )

    def _weight_quantization(
        self,
        document: Document,
        request: WeightQuantizationRequest,
        token: CancellationToken | None,
    ) -> ReplaceTensorBytes:
        if request.target_runtime is not TargetRuntime.PORTABLE_ONNX:
            raise EditError("dequantized weight conversion supports portable-onnx only")
        tensor, raw, values = _float_tensor(
            document, request.tensor_id, self._read, token
        )
        quantized = _quantize(
            values,
            bit_width=request.bit_width,
            signed=request.signed,
            symmetric=request.symmetric,
            granularity=request.granularity,
            axis=request.axis,
        )
        after = quantized.dequantized.astype("<f4").tobytes()
        if after == raw:
            raise EditError("quantization does not change any tensor value")
        manifest = TransformationManifest(
            kind=TransformationKind.WEIGHT_QUANTIZATION,
            target_runtime=request.target_runtime,
            operator_representation=OperatorRepresentation.DEQUANTIZED_FLOAT,
            storage_effect=StorageEffect.UNCHANGED,
            bit_width=request.bit_width,
            signed=request.signed,
            symmetric=request.symmetric,
            granularity=request.granularity,
            scale=quantized.scale,
            zero_point=quantized.zero_point,
            axis=quantized.axis,
            calibration_required=False,
            parameters=(("dtype", tensor.element_type),),
        )
        preview = _preview(
            request.tensor_id,
            values,
            quantized.dequantized,
            source_bytes=len(raw),
            result_bytes=len(after),
            theoretical_bytes=math.ceil(values.size * request.bit_width / 8),
            executable=True,
            storage_reduction=False,
            reason=(
                "The graph remains executable, but dequantized-float storage "
                "does not imply a smaller file or faster runtime."
            ),
        )
        return ReplaceTensorBytes(
            ByteSpanEdit(request.tensor_id, 0, raw, after),
            manifest,
            preview,
        )

    def _graph_quantization(
        self,
        document: Document,
        request: GraphQuantizationRequest,
        base_revision_id: str | None,
        token: CancellationToken | None,
    ) -> QuantizeGraph:
        if request.graph_id != document.entry_graph:
            raise EditError("graph quantization is confined to the entry graph")
        if request.target_runtime is not TargetRuntime.ONNX_REFERENCE:
            raise EditError("the core Q/DQ path supports onnx-reference only")
        if _opsets(document).get("", 0) < _REFERENCE_QDQ_MIN_OPSET:
            raise EditError(
                "ONNX reference Q/DQ quantization requires default opset "
                f"{_REFERENCE_QDQ_MIN_OPSET} or newer"
            )
        tensor, raw, values = _float_tensor(
            document, request.tensor_id, self._read, token
        )
        graph = document.graphs[request.graph_id]
        source_value = _initializer_value(document, graph, request.tensor_id)
        consumers = graph.consumers(source_value.id)
        if not consumers:
            raise EditError("initializer has no operator consumers")
        if any(
            graph.node(node_id).op_type == "QuantizeLinear"
            for node_id, _input_index in consumers
        ):
            raise EditError("initializer already has a QuantizeLinear boundary")
        quantized = _quantize(
            values,
            bit_width=request.bit_width,
            signed=request.signed,
            symmetric=request.symmetric,
            granularity=request.granularity,
            axis=request.axis,
        )
        digest = hashlib.sha256(
            (
                f"{base_revision_id}|{request.tensor_id}|{request.bit_width}|"
                f"{request.signed}|{request.symmetric}|{request.granularity.value}|"
                f"{quantized.axis}"
            ).encode()
        ).hexdigest()[:16]
        prefix = f"_nneditor_q_{digest}"
        scale_name = f"{prefix}_scale"
        zero_name = f"{prefix}_zero"
        quantized_name = f"{prefix}_quantized"
        dequantized_name = f"{prefix}_dequantized"
        scale_id = value_id(graph.id, scale_name)
        zero_id = value_id(graph.id, zero_name)
        quantized_id = value_id(graph.id, quantized_name)
        dequantized_id = value_id(graph.id, dequantized_name)
        parameter_dims = (
            ()
            if request.granularity is Granularity.PER_TENSOR
            else (tensor.dims[quantized.axis or 0],)
        )
        quantized_type = "int8" if request.signed else "uint8"
        scale_tensor = TensorRef(
            scale_id,
            "float32",
            parameter_dims,
            Storage.GENERATED,
        )
        zero_tensor = TensorRef(
            zero_id,
            quantized_type,
            parameter_dims,
            Storage.GENERATED,
        )
        scale_value = Value(scale_id, scale_name, "float32", parameter_dims)
        zero_value = Value(zero_id, zero_name, quantized_type, parameter_dims)
        quantized_value = Value(
            quantized_id,
            quantized_name,
            quantized_type,
            tensor.dims,
        )
        dequantized_value = Value(
            dequantized_id,
            dequantized_name,
            "float32",
            tensor.dims,
        )
        quantize_id, _ = node_id(graph.id, name=f"{prefix}_QuantizeLinear", ordinal=0)
        dequantize_id, _ = node_id(
            graph.id,
            name=f"{prefix}_DequantizeLinear",
            ordinal=1,
        )
        attributes = (
            ()
            if request.granularity is Granularity.PER_TENSOR
            else (Attribute("axis", AttrKind.INT, quantized.axis or 0),)
        )
        quantize_node = Node(
            quantize_id,
            NodeIdStability.NAMED,
            "QuantizeLinear",
            (source_value.id, scale_id, zero_id),
            (quantized_id,),
            attributes=attributes,
            source_name=f"{prefix}_QuantizeLinear",
        )
        dequantize_node = Node(
            dequantize_id,
            NodeIdStability.NAMED,
            "DequantizeLinear",
            (quantized_id, scale_id, zero_id),
            (dequantized_id,),
            attributes=attributes,
            source_name=f"{prefix}_DequantizeLinear",
        )
        added_bytes = len(quantized.scale) * 4 + len(quantized.zero_point)
        manifest = TransformationManifest(
            kind=TransformationKind.GRAPH_QUANTIZATION,
            target_runtime=request.target_runtime,
            operator_representation=OperatorRepresentation.ONNX_QDQ,
            storage_effect=StorageEffect.INCREASED,
            bit_width=request.bit_width,
            signed=request.signed,
            symmetric=request.symmetric,
            granularity=request.granularity,
            scale=quantized.scale,
            zero_point=quantized.zero_point,
            axis=quantized.axis,
            # Weight-derived Q/DQ never *requires* activation calibration;
            # attached provenance is recorded in the parameters instead
            # (audit finding: this field previously meant "provenance was
            # attached", inverting its schema definition).
            calibration_required=False,
            calibration_provenance=request.calibration_provenance,
            backend_plugin=request.backend_plugin,
            parameters=(("opset_minimum", str(_REFERENCE_QDQ_MIN_OPSET)),),
        )
        preview = _preview(
            request.tensor_id,
            values,
            quantized.dequantized,
            source_bytes=len(raw),
            result_bytes=len(raw) + added_bytes,
            theoretical_bytes=math.ceil(values.size * request.bit_width / 8),
            executable=True,
            storage_reduction=False,
            reason=(
                "Portable Q/DQ is executable in the declared reference runtime; "
                "acceleration depends on a backend optimizer and is not claimed."
            ),
        )
        scale_bytes = (
            np.asarray(quantized.scale, dtype="<f4").reshape(parameter_dims).tobytes()
        )
        zero_bytes = (
            np.asarray(
                quantized.zero_point,
                dtype=np.int8 if request.signed else np.uint8,
            )
            .reshape(parameter_dims)
            .tobytes()
        )
        return QuantizeGraph(
            graph.id,
            request.tensor_id,
            source_value.id,
            consumers,
            scale_tensor,
            zero_tensor,
            scale_value,
            zero_value,
            quantized_value,
            dequantized_value,
            quantize_node,
            dequantize_node,
            scale_bytes,
            zero_bytes,
            manifest,
            preview,
        )

    def _logical_pruning(
        self,
        document: Document,
        request: LogicalPruningRequest,
        token: CancellationToken | None,
    ) -> ReplaceTensorBytes:
        if request.target_runtime is not TargetRuntime.PORTABLE_ONNX:
            raise EditError("logical pruning supports portable-onnx only")
        _tensor, raw, values = _float_tensor(
            document, request.tensor_id, self._read, token
        )
        flattened = values.reshape(-1)
        keep = np.ones(flattened.size, dtype=np.bool_)
        parameters: tuple[tuple[str, str], ...]
        if request.mode is PruningMode.MASK:
            if len(request.mask) != flattened.size:
                raise EditError("pruning mask length does not match tensor size")
            keep = np.asarray(request.mask, dtype=np.bool_)
            packed_mask = np.packbits(keep, bitorder="little").tobytes()
            parameters = (
                ("mask_bits_little_endian", packed_mask.hex()),
                ("mask_length", str(keep.size)),
                ("mask_sha256", hashlib.sha256(keep.tobytes()).hexdigest()),
            )
        elif request.mode is PruningMode.THRESHOLD:
            if request.threshold < 0:
                raise EditError("pruning threshold must be non-negative")
            keep = np.abs(flattened) > request.threshold
            parameters = (("threshold", repr(request.threshold)),)
        elif request.mode is PruningMode.NM:
            if not 0 < request.n < request.m:
                raise EditError("N:M pruning requires 0 < N < M")
            if values.ndim == 0:
                raise EditError("N:M pruning requires a tensor with at least one axis")
            innermost = values.shape[-1]
            if innermost % request.m:
                raise EditError(
                    "N:M pruning groups along the innermost axis, so its "
                    f"extent {innermost} must be divisible by M={request.m}"
                )
            keep[:] = False
            # With the innermost extent divisible by M, every M-block of the
            # C-order flattening falls inside one innermost-axis run — no
            # block straddles a row boundary.
            blocks = flattened.reshape(-1, request.m)
            for block_index, block in enumerate(blocks):
                selected = np.argsort(np.abs(block), kind="stable")[-request.n :]
                keep[block_index * request.m + selected] = True
            parameters = (
                ("axis", "innermost"),
                ("n", str(request.n)),
                ("m", str(request.m)),
            )
        else:
            raise EditError("output-channel pruning needs the structured request")
        pruned_flat = values.copy().reshape(-1)
        pruned_flat[~keep] = 0.0
        after_values = pruned_flat.reshape(values.shape)
        after = after_values.astype("<f4").tobytes()
        if after == raw:
            raise EditError("pruning does not change any tensor value")
        manifest = TransformationManifest(
            kind=TransformationKind.LOGICAL_PRUNING,
            target_runtime=request.target_runtime,
            operator_representation=OperatorRepresentation.UNCHANGED_GRAPH,
            storage_effect=StorageEffect.UNCHANGED,
            calibration_required=False,
            pruning_mode=request.mode,
            parameters=parameters,
        )
        preview = _preview(
            request.tensor_id,
            values,
            after_values,
            source_bytes=len(raw),
            result_bytes=len(after),
            theoretical_bytes=len(after),
            executable=True,
            storage_reduction=False,
            reason=(
                "Logical zeros preserve the graph and file representation; "
                "sparse-kernel acceleration requires a separately verified backend."
            ),
        )
        return ReplaceTensorBytes(
            ByteSpanEdit(request.tensor_id, 0, raw, after),
            manifest,
            preview,
        )

    def _structured_pruning(
        self,
        document: Document,
        request: StructuredPruningRequest,
        token: CancellationToken | None,
    ) -> ResizeTensor:
        if request.target_runtime is not TargetRuntime.PORTABLE_ONNX:
            raise EditError("structured pruning supports portable-onnx only")
        if request.graph_id != document.entry_graph:
            raise EditError("structured pruning is confined to the entry graph")
        graph = document.graphs[request.graph_id]
        node = graph.node(request.node_id)
        if node.domain not in {"", "ai.onnx"} or node.op_type != "MatMul":
            raise EditError(
                "only the modelled MatMul output-channel pattern is supported"
            )
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise EditError("the MatMul pattern requires two inputs and one output")
        weight_value = graph.value(node.inputs[1])
        # Resizing the weight is only provably consistent when this MatMul is
        # its *sole* consumer: a second consumer would silently receive a
        # tensor whose shape no longer matches its own contract.
        if graph.consumers(weight_value.id) != ((node.id, 1),):
            raise EditError(
                "the weight initializer feeds more than this MatMul input; "
                "resizing it cannot be proven consistent for the other "
                "consumers, so the rewrite is rejected"
            )
        if weight_value.id in graph.outputs:
            raise EditError(
                "the weight initializer is also a graph output; resizing it "
                "would change the model's declared interface"
            )
        matches = tuple(
            tensor_id
            for tensor_id in graph.initializers
            if weight_value.name and tensor_id.endswith(f"#{weight_value.name}")
        )
        if len(matches) != 1:
            raise EditError("MatMul weight input is not an initializer")
        (tensor_id,) = matches
        tensor, raw, values = _float_tensor(document, tensor_id, self._read, token)
        if len(tensor.dims) != 2:
            raise EditError("MatMul structured pruning requires a rank-2 weight")
        rows, columns = tensor.dims
        input_value = graph.value(node.inputs[0])
        output_value = graph.value(node.outputs[0])
        if (
            input_value.shape is None
            or output_value.shape is None
            or not input_value.shape
            or not output_value.shape
            or input_value.shape[-1] != rows
            or output_value.shape[-1] != columns
            or weight_value.shape != tensor.dims
            or any(not isinstance(dim, int) for dim in output_value.shape)
        ):
            raise EditError("MatMul input/output shapes are not fully provable")
        if graph.consumers(output_value.id):
            raise EditError("structured pruning currently requires a terminal MatMul")
        if output_value.id not in graph.outputs:
            raise EditError("structured pruning currently requires a graph output")
        keep = tuple(sorted(set(request.keep_output_channels)))
        if not keep or keep != request.keep_output_channels:
            raise EditError("kept channels must be non-empty, unique, and sorted")
        if keep[0] < 0 or keep[-1] >= columns:
            raise EditError("kept output channel is outside the weight")
        if len(keep) == columns:
            raise EditError("structured pruning must remove at least one channel")
        after_values = values[:, keep].copy()
        after_bytes = after_values.astype("<f4").tobytes()
        after_tensor = TensorRef(
            tensor.id,
            tensor.element_type,
            (rows, len(keep)),
            Storage.GENERATED,
        )
        weight_after = replace(weight_value, shape=(rows, len(keep)))
        output_shape = (*output_value.shape[:-1], len(keep))
        output_after = replace(output_value, shape=output_shape)
        manifest = TransformationManifest(
            kind=TransformationKind.STRUCTURED_PRUNING,
            target_runtime=request.target_runtime,
            operator_representation=OperatorRepresentation.SHAPE_REWRITE,
            storage_effect=StorageEffect.REDUCED,
            calibration_required=False,
            pruning_mode=PruningMode.OUTPUT_CHANNELS,
            parameters=(
                ("pattern", "terminal-matmul-output-channels"),
                ("kept_channels", ",".join(str(item) for item in keep)),
            ),
        )
        preview = TransformationPreview(
            tensor_id=tensor.id,
            element_count=int(values.size),
            changed_elements=rows * (columns - len(keep)),
            source_bytes=len(raw),
            result_bytes=len(after_bytes),
            theoretical_packed_bytes=len(after_bytes),
            max_abs_error=0.0,
            mean_abs_error=0.0,
            rmse=0.0,
            error_basis=(
                "Not elementwise-comparable: the result removes output channels."
            ),
            before_sparsity=_sparsity(values),
            after_sparsity=_sparsity(after_values),
            mathematical_conversion=True,
            executable_graph=True,
            storage_reduction=True,
            expected_acceleration=False,
            acceleration_reason=(
                "The MatMul is smaller, but acceleration remains backend- and "
                "shape-dependent until benchmarked."
            ),
        )
        return ResizeTensor(
            graph.id,
            tensor,
            after_tensor,
            raw,
            after_bytes,
            (
                ValueMetadataChange(graph.id, weight_value, weight_after),
                ValueMetadataChange(graph.id, output_value, output_after),
            ),
            manifest,
            preview,
        )
