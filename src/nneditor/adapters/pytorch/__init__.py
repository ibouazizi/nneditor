"""PyTorch artifact adapters (Phase 6).

Everything here honors the safe-artifact boundary: exported programs, state
dicts, safetensors, and FX artifacts are parsed natively — JSON, zip
structure, a restricted weights-only pickle scan, and Python's ``ast`` — and
no framework code from an artifact is ever executed. The ``torch`` package is
a *dev-only* dependency used by tests to generate fixtures and differentially
validate this adapter, exactly as ``onnx`` once validated the lazy indexer.
"""

from nneditor.adapters.pytorch.checkpoint import open_checkpoint
from nneditor.adapters.pytorch.fx import open_fx_graph_module
from nneditor.adapters.pytorch.pt2 import open_pt2
from nneditor.adapters.pytorch.safetensors import (
    SafetensorsError,
    open_safetensors,
    write_safetensors,
)

__all__ = [
    "SafetensorsError",
    "open_checkpoint",
    "open_fx_graph_module",
    "open_pt2",
    "open_safetensors",
    "write_safetensors",
]
