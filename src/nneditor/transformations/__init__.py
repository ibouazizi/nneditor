"""Validated, previewable weight transformations."""

from importlib import import_module

from nneditor.transformations.calibration import (
    CalibrationLimits,
    CalibrationProvider,
    CalibrationResult,
    run_calibration,
)
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
    "CalibrationLimits",
    "CalibrationProvider",
    "CalibrationResult",
    "Granularity",
    "GraphQuantizationRequest",
    "LogicalPruningRequest",
    "OperatorRepresentation",
    "PruningMode",
    "StorageEffect",
    "StructuredPruningRequest",
    "TargetRuntime",
    "TransformationEngine",
    "TransformationKind",
    "TransformationManifest",
    "TransformationPreview",
    "TransformationProposal",
    "WeightQuantizationRequest",
    "run_calibration",
]

_ENGINE_EXPORTS = frozenset(
    {
        "GraphQuantizationRequest",
        "LogicalPruningRequest",
        "StructuredPruningRequest",
        "TransformationEngine",
        "TransformationProposal",
        "WeightQuantizationRequest",
    }
)


def __getattr__(name: str) -> object:
    """Load command-dependent engine exports without creating an import cycle."""
    if name in _ENGINE_EXPORTS:
        return getattr(import_module("nneditor.transformations.engine"), name)
    raise AttributeError(name)
