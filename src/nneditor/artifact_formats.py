"""User-facing file-extension hints for supported artifact adapters.

Artifact detection remains content based.  These extensions exist for native
file pickers and desktop shell registration, where the operating system needs
an explicit candidate list before NNEditor has an opportunity to inspect the
file bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

MODEL_FILE_EXTENSION_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ONNX": ("onnx", "pb"),
        "PyTorch": ("pt2", "pt", "pth", "ckpt", "bin"),
        "Safetensors": ("safetensors",),
        "StableHLO": ("mlir", "stablehlo"),
    }
)
"""Extension hints grouped by the artifact family shown to the user."""

MODEL_FILE_EXTENSIONS: Final[tuple[str, ...]] = tuple(
    extension
    for extensions in MODEL_FILE_EXTENSION_GROUPS.values()
    for extension in extensions
)
"""Extension hints without a leading period, suitable for Flet pickers."""

MODEL_FILE_SUFFIXES: Final[tuple[str, ...]] = tuple(
    f".{extension}" for extension in MODEL_FILE_EXTENSIONS
)
"""Extension hints with a leading period, suitable for desktop shells."""
