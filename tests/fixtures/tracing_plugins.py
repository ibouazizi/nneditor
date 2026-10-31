"""A tiny real decoder and a standalone extension implementing its forward pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tests.fixtures.gguf_models import GgufFixtureTensor, write_gguf

LINEAR_PLUGIN = """from nneditor.tracing.sdk import TraceOutcome
from .helper import SCALE

def trace(context):
    context.capture_inputs()
    hidden = context.load_weight("token_embd.weight")[context.inputs["tokens"]]
    context.capture("embed", hidden)
    logits = hidden @ context.load_weight("output.weight").T * SCALE
    context.capture("lm-head", logits)
    return TraceOutcome(provider="test-numpy", notes=("Test linear decoder",))
"""


def write_trace_plugin(root: Path, **changes: Any) -> Path:
    code = changes.pop("code", LINEAR_PLUGIN)
    assert isinstance(code, str)
    root.mkdir(parents=True)
    manifest = {
        "api_version": 1,
        "id": "test.linear",
        "name": "Test linear decoder",
        "version": "0.1.0",
        "entry_point": "tracer.py:trace",
        "artifact_kinds": ["gguf_model"],
        "architectures": ["plugin_linear"],
        "element_types": ["float32"],
        "operators": ["Embedding", "Linear"],
        "description": "A small external test decoder.",
        **changes,
    }
    (root / "nneditor-tracer.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "tracer.py").write_text(code, encoding="utf-8")
    (root / "helper.py").write_text("SCALE = 1\n", encoding="utf-8")
    return root


def write_linear_gguf(path: Path) -> Path:
    embedding = np.array([[1, 2], [3, 4], [5, 6]], dtype="<f4")
    projection = np.array([[1, 0], [0, 1], [2, 3]], dtype="<f4")
    return write_gguf(
        path,
        version=3,
        kv=(
            ("general.architecture", 8, "plugin_linear"),
            ("plugin_linear.embedding_length", 4, 2),
            ("plugin_linear.vocab_size", 4, 3),
        ),
        tensors=(
            GgufFixtureTensor(
                "token_embd.weight", embedding.shape, 0, embedding.tobytes()
            ),
            GgufFixtureTensor(
                "output.weight", projection.shape, 0, projection.tobytes()
            ),
        ),
    )
