"""Reproducible schemas shared by transformation previews and commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nneditor.editing.cow import EditError

__all__ = [
    "Granularity",
    "OperatorRepresentation",
    "PruningMode",
    "StorageEffect",
    "TargetRuntime",
    "TransformationKind",
    "TransformationManifest",
    "TransformationPreview",
]


class TransformationKind(StrEnum):
    WEIGHT_QUANTIZATION = "weight-quantization"
    GRAPH_QUANTIZATION = "graph-quantization"
    LOGICAL_PRUNING = "logical-pruning"
    STRUCTURED_PRUNING = "structured-pruning"


class TargetRuntime(StrEnum):
    PORTABLE_ONNX = "portable-onnx"
    ONNX_REFERENCE = "onnx-reference"


class Granularity(StrEnum):
    PER_TENSOR = "per-tensor"
    PER_CHANNEL = "per-channel"


class OperatorRepresentation(StrEnum):
    DEQUANTIZED_FLOAT = "dequantized-float"
    ONNX_QDQ = "onnx-qdq"
    UNCHANGED_GRAPH = "unchanged-graph"
    SHAPE_REWRITE = "shape-rewrite"


class PruningMode(StrEnum):
    MASK = "mask"
    THRESHOLD = "threshold"
    NM = "n:m"
    OUTPUT_CHANNELS = "output-channels"


class StorageEffect(StrEnum):
    UNCHANGED = "unchanged"
    REDUCED = "reduced"
    INCREASED = "increased"


def _optional_bool(raw: dict[object, object], key: str) -> bool | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise EditError(f"transformation field {key!r} must be boolean or null")
    return value


def _bool(raw: dict[object, object], key: str, *, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise EditError(f"transformation field {key!r} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class TransformationManifest:
    """Everything needed to explain and reproduce a transformation."""

    kind: TransformationKind
    target_runtime: TargetRuntime
    operator_representation: OperatorRepresentation
    storage_effect: StorageEffect
    bit_width: int | None = None
    signed: bool | None = None
    symmetric: bool | None = None
    granularity: Granularity | None = None
    scale: tuple[float, ...] = ()
    zero_point: tuple[int, ...] = ()
    axis: int | None = None
    calibration_required: bool = False
    pruning_mode: PruningMode | None = None
    parameters: tuple[tuple[str, str], ...] = ()
    calibration_provenance: tuple[tuple[str, str], ...] = ()
    backend_plugin: str | None = None

    def __post_init__(self) -> None:
        if len({key for key, _value in self.parameters}) != len(self.parameters):
            raise EditError("transformation parameters contain duplicate keys")
        if len({key for key, _value in self.calibration_provenance}) != len(
            self.calibration_provenance
        ):
            raise EditError("calibration provenance contains duplicate keys")
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))
        object.__setattr__(
            self,
            "calibration_provenance",
            tuple(sorted(self.calibration_provenance)),
        )
        if self.bit_width is not None and self.bit_width <= 0:
            raise EditError("quantization bit width must be positive")
        if self.granularity is Granularity.PER_CHANNEL and self.axis is None:
            raise EditError("per-channel quantization requires an axis")
        if self.kind in {
            TransformationKind.WEIGHT_QUANTIZATION,
            TransformationKind.GRAPH_QUANTIZATION,
        }:
            if (
                self.bit_width is None
                or self.signed is None
                or self.symmetric is None
                or self.granularity is None
                or not self.scale
                or not self.zero_point
            ):
                raise EditError("quantization manifests need a complete scheme")
        if (
            self.kind
            in {
                TransformationKind.LOGICAL_PRUNING,
                TransformationKind.STRUCTURED_PRUNING,
            }
            and self.pruning_mode is None
        ):
            raise EditError("pruning manifests need a pruning mode")
        if self.backend_plugin is not None and not self.backend_plugin.strip():
            raise EditError("backend plugin id cannot be empty")

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "target_runtime": self.target_runtime.value,
            "operator_representation": self.operator_representation.value,
            "storage_effect": self.storage_effect.value,
            "bit_width": self.bit_width,
            "signed": self.signed,
            "symmetric": self.symmetric,
            "granularity": (
                None if self.granularity is None else self.granularity.value
            ),
            "scale": list(self.scale),
            "zero_point": list(self.zero_point),
            "axis": self.axis,
            "calibration_required": self.calibration_required,
            "pruning_mode": (
                None if self.pruning_mode is None else self.pruning_mode.value
            ),
            "parameters": dict(self.parameters),
            "calibration_provenance": dict(self.calibration_provenance),
            "backend_plugin": self.backend_plugin,
        }

    @classmethod
    def from_json(cls, raw: object) -> TransformationManifest:
        if not isinstance(raw, dict):
            raise EditError("transformation manifest is not an object")
        try:
            granularity_raw = raw.get("granularity")
            pruning_raw = raw.get("pruning_mode")
            parameters = raw.get("parameters", {})
            provenance = raw.get("calibration_provenance", {})
            if not isinstance(parameters, dict) or not isinstance(provenance, dict):
                raise EditError("transformation provenance is malformed")
            scale = raw.get("scale", [])
            zero_point = raw.get("zero_point", [])
            if not isinstance(scale, list) or not isinstance(zero_point, list):
                raise EditError("quantization parameters must be lists")
            return cls(
                kind=TransformationKind(str(raw["kind"])),
                target_runtime=TargetRuntime(str(raw["target_runtime"])),
                operator_representation=OperatorRepresentation(
                    str(raw["operator_representation"])
                ),
                storage_effect=StorageEffect(str(raw["storage_effect"])),
                bit_width=(
                    None if raw.get("bit_width") is None else int(raw["bit_width"])
                ),
                signed=_optional_bool(raw, "signed"),
                symmetric=_optional_bool(raw, "symmetric"),
                granularity=(
                    None
                    if granularity_raw is None
                    else Granularity(str(granularity_raw))
                ),
                scale=tuple(float(item) for item in scale),
                zero_point=tuple(int(item) for item in zero_point),
                axis=None if raw.get("axis") is None else int(raw["axis"]),
                calibration_required=_bool(raw, "calibration_required"),
                pruning_mode=(
                    None if pruning_raw is None else PruningMode(str(pruning_raw))
                ),
                parameters=tuple(
                    sorted((str(key), str(value)) for key, value in parameters.items())
                ),
                calibration_provenance=tuple(
                    sorted((str(key), str(value)) for key, value in provenance.items())
                ),
                backend_plugin=(
                    None
                    if raw.get("backend_plugin") is None
                    else str(raw["backend_plugin"])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EditError(f"malformed transformation manifest: {error}") from error


@dataclass(frozen=True, slots=True)
class TransformationPreview:
    """Bounded numerical and product-impact summary shown before commit."""

    tensor_id: str
    element_count: int
    changed_elements: int
    source_bytes: int
    result_bytes: int
    theoretical_packed_bytes: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    error_basis: str
    before_sparsity: float
    after_sparsity: float
    mathematical_conversion: bool
    executable_graph: bool
    storage_reduction: bool
    expected_acceleration: bool
    acceleration_reason: str

    def __post_init__(self) -> None:
        if (
            min(
                self.element_count,
                self.changed_elements,
                self.source_bytes,
                self.result_bytes,
                self.theoretical_packed_bytes,
            )
            < 0
        ):
            raise EditError("transformation preview counts must be non-negative")
        if self.changed_elements > self.element_count:
            raise EditError("changed element count exceeds tensor size")
        if not 0.0 <= self.before_sparsity <= 1.0:
            raise EditError("before sparsity must be between zero and one")
        if not 0.0 <= self.after_sparsity <= 1.0:
            raise EditError("after sparsity must be between zero and one")
        if not self.acceleration_reason.strip():
            raise EditError("acceleration expectation needs a reason")
        if not self.error_basis.strip():
            raise EditError("transformation error summary needs a basis")

    def to_json(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "element_count": self.element_count,
            "changed_elements": self.changed_elements,
            "source_bytes": self.source_bytes,
            "result_bytes": self.result_bytes,
            "theoretical_packed_bytes": self.theoretical_packed_bytes,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "error_basis": self.error_basis,
            "before_sparsity": self.before_sparsity,
            "after_sparsity": self.after_sparsity,
            "mathematical_conversion": self.mathematical_conversion,
            "executable_graph": self.executable_graph,
            "storage_reduction": self.storage_reduction,
            "expected_acceleration": self.expected_acceleration,
            "acceleration_reason": self.acceleration_reason,
        }

    @classmethod
    def from_json(cls, raw: object) -> TransformationPreview:
        if not isinstance(raw, dict):
            raise EditError("transformation preview is not an object")
        try:
            return cls(
                tensor_id=str(raw["tensor_id"]),
                element_count=int(raw["element_count"]),
                changed_elements=int(raw["changed_elements"]),
                source_bytes=int(raw["source_bytes"]),
                result_bytes=int(raw["result_bytes"]),
                theoretical_packed_bytes=int(raw["theoretical_packed_bytes"]),
                max_abs_error=float(raw["max_abs_error"]),
                mean_abs_error=float(raw["mean_abs_error"]),
                rmse=float(raw["rmse"]),
                error_basis=str(raw["error_basis"]),
                before_sparsity=float(raw["before_sparsity"]),
                after_sparsity=float(raw["after_sparsity"]),
                mathematical_conversion=_bool(raw, "mathematical_conversion"),
                executable_graph=_bool(raw, "executable_graph"),
                storage_reduction=_bool(raw, "storage_reduction"),
                expected_acceleration=_bool(raw, "expected_acceleration"),
                acceleration_reason=str(raw["acceleration_reason"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EditError(f"malformed transformation preview: {error}") from error
