"""ONNX import, IR conversion, tensor access helpers, and verified export.

Normal import stays on the lazy wire index. The official ONNX bindings are
used for schemas and only materialize a full model at the explicit Phase 4
export or numerical-comparison boundary.
"""

from nneditor.adapters.onnx.exporter import (
    ExportError,
    ExportReport,
    export_revision,
    report_from_json,
)
from nneditor.adapters.onnx.index import (
    ExternalDataReference,
    GraphIndex,
    ModelIndex,
    NodeIndex,
    TensorIndex,
    TensorStorage,
    ValueIndex,
)
from nneditor.adapters.onnx.indexer import (
    OnnxIndexError,
    ScanLimits,
    index_model,
    read_tensor_bytes,
)
from nneditor.adapters.onnx.splice import (
    SpliceExportError,
    SpliceReport,
    export_with_edits,
)
from nneditor.adapters.onnx.to_ir import index_to_document

__all__ = [
    "ExportError",
    "ExportReport",
    "ExternalDataReference",
    "GraphIndex",
    "ModelIndex",
    "NodeIndex",
    "OnnxIndexError",
    "ScanLimits",
    "SpliceExportError",
    "SpliceReport",
    "TensorIndex",
    "TensorStorage",
    "ValueIndex",
    "export_revision",
    "export_with_edits",
    "index_model",
    "index_to_document",
    "read_tensor_bytes",
    "report_from_json",
]
