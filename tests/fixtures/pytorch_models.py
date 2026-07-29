"""Deterministic PyTorch fixtures (Phase 6).

``torch`` and ``safetensors`` are dev-only dependencies: they generate these
artifacts and differentially validate the native readers, mirroring how the
``onnx`` package validates the lazy ONNX indexer without being part of the
product's import path.
"""

from pathlib import Path

import torch
from torch import nn

# NOTE: deliberately no `from __future__ import annotations` here. Torch 2.13
# serializes an FX GraphModule by rebuilding its import block from the traced
# function's globals; with PEP 563 in effect the annotations arrive as strings
# and `torch.package`'s importer raises on them. Fixtures must exercise the
# real serialization path, so the module keeps eager annotations.

__all__ = [
    "TinyModel",
    "build_checkpoint",
    "build_fx_module",
    "build_pt2",
    "build_safetensors_file",
    "tiny_state_dict",
]


class TinyModel(nn.Module):
    """Linear + ReLU: one parameterized op, one activation, two weights."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _seeded_model() -> TinyModel:
    torch.manual_seed(7)
    return TinyModel()


def tiny_state_dict() -> "dict[str, torch.Tensor]":
    return _seeded_model().state_dict()


def build_pt2(path: Path, *, dynamic: bool = False) -> "dict[str, torch.Tensor]":
    """A saved exported program; ``dynamic`` marks the batch dim symbolic."""
    model = _seeded_model()
    example = (torch.randn(2, 3),)
    dynamic_shapes = None
    if dynamic:
        batch = torch.export.Dim("batch", min=1, max=64)
        dynamic_shapes = {"x": {0: batch}}
    program = torch.export.export(model, example, dynamic_shapes=dynamic_shapes)
    torch.export.save(program, str(path))
    return dict(program.state_dict)


def build_checkpoint(path: Path) -> "dict[str, torch.Tensor]":
    state = tiny_state_dict()
    torch.save(state, str(path))
    return state


def build_fx_module(path: Path) -> None:
    traced = torch.fx.symbolic_trace(_seeded_model())
    torch.save(traced, str(path))


def build_safetensors_file(path: Path) -> "dict[str, torch.Tensor]":
    from safetensors.torch import save_file

    state = tiny_state_dict()
    save_file(state, str(path), metadata={"producer": "nneditor-tests"})
    return state
