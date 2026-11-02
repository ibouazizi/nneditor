"""Bounded GGUF matrix reads and Prism activation-basis transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nneditor.adapters.gguf.constants import GGML_IR_DTYPES, GGML_QUANT_SIZES
from nneditor.adapters.gguf.dequant import dequantize
from nneditor.adapters.gguf.preview import preview_kv_array
from nneditor.adapters.gguf.prism import PRISM_LAYOUTS
from nneditor.ir.core import Document, JsonValue
from nneditor.storage.store import TensorStore
from nneditor.tracing.llm_runtime import LlmTraceError, _extension_map, _tensor_name

FloatArray = NDArray[np.float32]
_LAYOUTS = {
    name: GGML_QUANT_SIZES[code]
    for code, name in GGML_IR_DTYPES.items()
    if code in GGML_QUANT_SIZES
} | PRISM_LAYOUTS
_DECODE_BYTES = 8 * 1024 * 1024


def hadamard(x: FloatArray, block: int) -> FloatArray:
    """Normalized Sylvester transform along contiguous feature blocks."""
    if block <= 0 or block & (block - 1) or x.shape[-1] % block:
        raise LlmTraceError("Invalid Hadamard block or feature width")
    y = np.array(x, dtype=np.float32, copy=True).reshape(-1, block)
    stride = 1
    while stride < block:
        pairs = y.reshape(-1, block // (2 * stride), 2, stride)
        a, b = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0] = a + b
        pairs[:, :, 1] = a - b
        stride *= 2
    y *= np.float32(1 / np.sqrt(block))
    return y.reshape(x.shape)


def _array_count(value: JsonValue) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and value.get("type") == "array":
        count = value.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    raise LlmTraceError("Prism Hadamard metadata requires an array")


def _tied_output(meta: dict[str, JsonValue]) -> bool:
    """Version 2 is version 1 plus a tied latent output head."""
    tied = meta.get("prism.hadamard.tied_output", False)
    if not isinstance(tied, bool):
        raise LlmTraceError("Unsupported Prism Hadamard tied output")
    if tied != (meta.get("prism.hadamard.version") == 2):
        raise LlmTraceError(
            "Prism Hadamard version 2 requires tied_output=true; version 1 forbids it"
        )
    return tied


def validate_prism(document: Document) -> None:
    """Validate declared transform semantics without reading any payload."""
    meta = _extension_map(document, "x-gguf.metadata")
    if not any(key.startswith("prism.hadamard.") for key in meta):
        return
    if meta.get("prism.hadamard.version") not in (1, 2):
        raise LlmTraceError("Unsupported or missing Prism Hadamard version")
    _tied_output(meta)
    expected: dict[str, JsonValue] = {
        "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension",
    }
    for suffix, value in expected.items():
        if meta.get("prism.hadamard." + suffix) != value:
            raise LlmTraceError(f"Unsupported or missing Prism Hadamard {suffix}")
    block = meta.get("prism.hadamard.block_size")
    if not isinstance(block, int) or block <= 0 or block > 65536 or block & (block - 1):
        raise LlmTraceError("Invalid Prism Hadamard block size")
    mode = meta.get("prism.hadamard.sign_mode")
    if mode not in ("identity", "explicit"):
        raise LlmTraceError("Unsupported or missing Prism Hadamard sign mode")
    names = meta.get("prism.hadamard.weight_names")
    if not 0 < _array_count(names) <= len(document.tensors):
        raise LlmTraceError("Invalid Prism Hadamard weight list")
    if mode == "explicit":
        widths = meta.get("prism.hadamard.sign_widths")
        if (
            not isinstance(widths, list)
            or not widths
            or any(not isinstance(w, int) or w <= 0 or w % block for w in widths)
        ):
            raise LlmTraceError("Invalid Prism Hadamard sign widths")
        total = sum(int(str(w)) for w in widths)
        if (
            total > 4 * 1024 * 1024
            or _array_count(meta.get("prism.hadamard.sign_values")) != total
        ):
            raise LlmTraceError("Invalid Prism Hadamard sign vector length")


class GgufWeights:
    """Stream projections by output rows, never materializing a whole model."""

    def __init__(self, document: Document, store: TensorStore, endian: str) -> None:
        self.document, self.store, self.endian = document, store, endian
        self.ids = {_tensor_name(t.id): t.id for t in document.tensors.values()}
        self.meta = _extension_map(document, "x-gguf.metadata")
        self.block = 0
        self.folded: set[str] = set()
        self.inverses: set[str] = set()
        self.signs: dict[int, FloatArray] = {}
        self.tied_output = False
        validate_prism(document)
        if "prism.hadamard.version" not in self.meta:
            return
        self.tied_output = _tied_output(self.meta)
        self.block = int(str(self.meta["prism.hadamard.block_size"]))
        self.folded = self._names("weight_names")
        self.inverses = self._names("inverse_weight_names", optional=True)
        if self.tied_output and self.inverses != {"token_embd.weight"}:
            raise LlmTraceError(
                "Prism Hadamard tied output requires a latent token embedding"
            )
        if any(
            name == "token_embd.weight" or ".ssm_conv1d." in name
            for name in self.folded
        ):
            raise LlmTraceError("Prism folded weights must be matrix projections")
        if self.folded & self.inverses or self.inverses - {"token_embd.weight"}:
            raise LlmTraceError("Unsupported Prism inverse weight list")
        if self.meta["prism.hadamard.sign_mode"] == "explicit":
            widths = self._array("sign_widths")
            values = np.asarray(self._array("sign_values"), dtype=np.float32)
            if not np.all((values == 1) | (values == -1)):
                raise LlmTraceError("Prism Hadamard signs must be +1 or -1")
            offset = 0
            for item in widths:
                width = int(str(item))
                if width in self.signs:
                    raise LlmTraceError("Duplicate Prism sign width")
                self.signs[width] = values[offset : offset + width]
                offset += width
        for name in self.folded | self.inverses:
            tensor = store.metadata(self.ids[name])
            if len(tensor.dims) != 2 or tensor.dims[-1] % self.block:
                raise LlmTraceError(f"Invalid Hadamard weight dimensions: {name}")
            if self.signs and tensor.dims[-1] not in self.signs:
                raise LlmTraceError(f"Missing Hadamard signs for {name}")

    def _array(self, suffix: str) -> list[JsonValue]:
        raw = self.meta.get("prism.hadamard." + suffix)
        if isinstance(raw, list):
            return raw
        count = _array_count(raw)
        if not isinstance(raw, dict) or count > 4 * 1024 * 1024:
            raise LlmTraceError("Invalid Prism metadata array")
        model = _extension_map(self.document, "x-gguf.model")
        items, actual = preview_kv_array(
            Path(self.document.source.path),
            offset=int(str(raw["offset"])),
            length=int(str(raw["length"])),
            version=int(str(model["version"])),
            endian=self.endian,
            limit=max(1, count),
        )
        if actual != count or len(items) != count:
            raise LlmTraceError("Prism metadata array changed since indexing")
        return list(items)

    def _names(self, suffix: str, *, optional: bool = False) -> set[str]:
        if optional and "prism.hadamard." + suffix not in self.meta:
            return set()
        values = self._array(suffix)
        if any(not isinstance(v, str) or v not in self.ids for v in values):
            raise LlmTraceError("Prism transform references an unknown weight")
        return {str(v) for v in values}

    def load(self, name: str) -> FloatArray:
        tensor = self.store.metadata(self.ids[name])
        return dequantize(
            tensor.element_type,
            tensor.dims,
            self.store.read(tensor.id),
            endian=self.endian,
        )

    def rows(self, name: str, start: int, stop: int) -> FloatArray:
        tensor = self.store.metadata(self.ids[name])
        width = tensor.dims[-1]
        block, size = _LAYOUTS[tensor.element_type]
        if width % block:
            raise LlmTraceError(f"{name}: input width is not block aligned")
        row_bytes = width // block * size
        payload = self.store.read(
            tensor.id, offset=start * row_bytes, length=(stop - start) * row_bytes
        )
        return dequantize(
            tensor.element_type, (stop - start, width), payload, endian=self.endian
        )

    def embedding(self, tokens: NDArray[np.integer]) -> FloatArray:
        unique, indices = np.unique(tokens, return_inverse=True)
        rows = np.concatenate(
            [self.rows("token_embd.weight", int(t), int(t) + 1) for t in unique]
        )
        x = rows[indices].reshape(*tokens.shape, rows.shape[-1])
        if "token_embd.weight" in self.inverses:
            x = hadamard(x, self.block)
            if self.signs:
                x *= self.signs[x.shape[-1]]
        return x

    def _forward_folded(self, name: str) -> bool:
        # The metadata list keeps the embedding inverse-only. Version 2 also
        # registers that table on the output matmul, which folds the activation.
        return name in self.folded or (self.tied_output and name == "token_embd.weight")

    def linear(self, name: str, x: FloatArray) -> FloatArray:
        tensor = self.store.metadata(self.ids[name])
        if self._forward_folded(name):
            if name.endswith(".ssm_out.weight") and self.meta.get(
                "prism.hadamard.gdn_v_grouped"
            ):
                nk = int(str(self.meta["qwen35.ssm.group_count"]))
                nv = int(str(self.meta["qwen35.ssm.time_step_rank"]))
                # Reorder native tiled heads for the grouped ssm_out basis.
                x = x.reshape(*x.shape[:-1], nv // nk, nk, x.shape[-1] // nv)
                x = x.swapaxes(-3, -2).reshape(*x.shape[:-3], tensor.dims[-1])
            if self.signs:
                x = x * self.signs[x.shape[-1]]
            x = hadamard(x, self.block)
        count, width = tensor.dims
        chunk = max(1, _DECODE_BYTES // (width * 4))
        result = np.empty((*x.shape[:-1], count), dtype=np.float32)
        for start in range(0, count, chunk):
            stop = min(count, start + chunk)
            result[..., start:stop] = x @ self.rows(name, start, stop).T
        return result
