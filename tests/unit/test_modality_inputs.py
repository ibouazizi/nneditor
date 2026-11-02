"""Media defaults and early validation must preserve cross-input semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.gguf.media import full_whisper
from nneditor.adapters.huggingface.to_ir import open_huggingface
from nneditor.ir.core import Document
from nneditor.modality_inputs import modality_input_defaults
from nneditor.tracing.modality_validation import validate_modality_inputs
from nneditor.ui.input_workspace import InputTarget, preset_for_target
from tests.unit.test_modality_graphs import _snapshot


def _qwen(tmp_path: Path) -> Document:
    return open_huggingface(
        _snapshot(
            tmp_path / "qwen",
            {
                "model_type": "qwen2_vl",
                "image_token_id": 1,
                "vision_start_token_id": 3,
                "vision_end_token_id": 4,
                "vision_config": {
                    "depth": 2,
                    "in_channels": 3,
                    "temporal_patch_size": 2,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                },
                "text_config": {"num_hidden_layers": 1, "vocab_size": 32},
            },
        )
    )


@pytest.mark.parametrize("name", ["image-grid", "token-types", "image_grid_thw"])
def test_structural_metadata_never_uses_prompt_hashes(name: str) -> None:
    preset = preset_for_target(InputTarget(name, "int64", ("batch", 3)))
    assert preset.kind != "text-tokens"
    assert not preset.ready


def test_fixed_token_extent_survives_symbolic_batch() -> None:
    preset = preset_for_target(InputTarget("input_ids", "int64", ("batch", 3)))
    assert preset.sequence_length == 3


def test_decoder_alias_is_not_silently_treated_as_encoder_only() -> None:
    assert full_whisper({"encoder.blk.0.attn_q.weight", "decoder.blk.0.attn_q.weight"})


@pytest.mark.parametrize("error", ["grid", "types", "vocabulary", "placeholders"])
def test_qwen_rejects_inconsistent_media_inputs(tmp_path: Path, error: str) -> None:
    document = _qwen(tmp_path)
    feeds: dict[str, np.ndarray[Any, Any]] = {
        "pixel-values": np.zeros((4, 1176), dtype=np.float32),
        "image-grid": np.array([[1, 2, 2]], dtype=np.int64),
        "tokens": np.array([[3, 1, 4]], dtype=np.int64),
        "token-types": np.array([[0, 1, 0]], dtype=np.int64),
    }
    validate_modality_inputs(document, feeds)
    if error == "grid":
        feeds["image-grid"][0, 1] = 4
    elif error == "types":
        feeds["token-types"][0, 0] = 700
    elif error == "vocabulary":
        feeds["tokens"][0, 0] = 32
    else:
        feeds["tokens"][0, 1] = 0
        feeds["token-types"][0, 1] = 0
    with pytest.raises(ValueError):
        validate_modality_inputs(document, feeds)


def test_missing_special_token_metadata_does_not_guess_a_prefix(tmp_path: Path) -> None:
    document = open_huggingface(
        _snapshot(
            tmp_path / "whisper",
            {
                "model_type": "whisper",
                "encoder_layers": 1,
                "decoder_layers": 1,
                "num_mel_bins": 8,
                "max_source_positions": 32,
                "vocab_size": 32,
            },
        )
    )
    defaults = modality_input_defaults(document)
    assert "decoder_start_token_id" in (
        defaults["decoder-tokens"].unavailable_reason or ""
    )
    assert document.main_graph.value(document.main_graph.inputs[0]).shape == (
        "batch",
        8,
        64,
    )
