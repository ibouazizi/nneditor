"""Consent-gated inference tracing and activation inspection."""

from nneditor.tracing.comparison import (
    ActivationError as ActivationError,
)
from nneditor.tracing.comparison import (
    NodeError as NodeError,
)
from nneditor.tracing.comparison import (
    TraceComparison as TraceComparison,
)
from nneditor.tracing.comparison import (
    compare_traces as compare_traces,
)
from nneditor.tracing.comparison import (
    comparison_scene_patch as comparison_scene_patch,
)
from nneditor.tracing.comparison import (
    trace_scene_patch as trace_scene_patch,
)
from nneditor.tracing.contracts import (
    ActivationRecord as ActivationRecord,
)
from nneditor.tracing.contracts import (
    CaptureState as CaptureState,
)
from nneditor.tracing.contracts import (
    CaptureStatus as CaptureStatus,
)
from nneditor.tracing.contracts import (
    InputBinding as InputBinding,
)
from nneditor.tracing.contracts import (
    InputSource as InputSource,
)
from nneditor.tracing.contracts import (
    InputSpecification as InputSpecification,
)
from nneditor.tracing.contracts import (
    TraceApproval as TraceApproval,
)
from nneditor.tracing.contracts import (
    TraceBackend as TraceBackend,
)
from nneditor.tracing.contracts import (
    TraceDevice as TraceDevice,
)
from nneditor.tracing.contracts import (
    TraceKey as TraceKey,
)
from nneditor.tracing.contracts import (
    TraceLimits as TraceLimits,
)
from nneditor.tracing.contracts import (
    TraceRequest as TraceRequest,
)
from nneditor.tracing.contracts import (
    TraceResult as TraceResult,
)
from nneditor.tracing.inputs import (
    InputSpecificationStore as InputSpecificationStore,
)
from nneditor.tracing.inputs import (
    bind_inputs as bind_inputs,
)
from nneditor.tracing.inputs import (
    default_input_specification as default_input_specification,
)
from nneditor.tracing.inputs import (
    tensor_file_binding as tensor_file_binding,
)
from nneditor.tracing.inputs import (
    tensor_file_binding_for_input as tensor_file_binding_for_input,
)
from nneditor.tracing.runner import (
    TraceError as TraceError,
)
from nneditor.tracing.runner import (
    recommended_trace_limits as recommended_trace_limits,
)
from nneditor.tracing.runner import (
    run_onnx_trace as run_onnx_trace,
)
from nneditor.tracing.scan import (
    AxisAwareScan as AxisAwareScan,
)
from nneditor.tracing.scan import (
    ScanNormalizationError as ScanNormalizationError,
)
from nneditor.tracing.scan import (
    ScanNormalizationReport as ScanNormalizationReport,
)
from nneditor.tracing.scan import (
    normalize_scan_axes as normalize_scan_axes,
)
from nneditor.tracing.store import (
    ActivationStore as ActivationStore,
)
from nneditor.tracing.store import (
    ActivationView as ActivationView,
)
from nneditor.tracing.visualization import (
    ActivationVisualization as ActivationVisualization,
)
from nneditor.tracing.visualization import (
    PlotKind as PlotKind,
)
from nneditor.tracing.visualization import (
    build_activation_visualizations as build_activation_visualizations,
)

__all__ = [
    "ActivationError",
    "ActivationRecord",
    "ActivationStore",
    "ActivationView",
    "ActivationVisualization",
    "AxisAwareScan",
    "CaptureState",
    "CaptureStatus",
    "InputBinding",
    "InputSource",
    "InputSpecification",
    "InputSpecificationStore",
    "NodeError",
    "PlotKind",
    "ScanNormalizationError",
    "ScanNormalizationReport",
    "TraceApproval",
    "TraceBackend",
    "TraceComparison",
    "TraceDevice",
    "TraceError",
    "TraceKey",
    "TraceLimits",
    "TraceRequest",
    "TraceResult",
    "bind_inputs",
    "build_activation_visualizations",
    "compare_traces",
    "comparison_scene_patch",
    "default_input_specification",
    "normalize_scan_axes",
    "recommended_trace_limits",
    "run_onnx_trace",
    "tensor_file_binding",
    "tensor_file_binding_for_input",
    "trace_scene_patch",
]
