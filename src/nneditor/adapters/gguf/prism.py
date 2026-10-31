"""Prism's verified g128 ternary layouts (fork IDs 142 and 143).

These names are assigned only to files declaring prism.hadamard.version.
The public ggml type table deliberately keeps unrelated fork IDs opaque.
Layout reference: PrismML-Eng/llama.cpp, ggml-common.h and ggml-quants.c.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PRISM_TYPES = {142: "prism.pq2_0", 143: "prism.ptq1_0"}
PRISM_LAYOUTS = {"prism.pq2_0": (128, 34), "prism.ptq1_0": (128, 28)}


def decode_ternary(
    payload: bytes, n_elems: int, endian: str, *, dense: bool
) -> NDArray[np.float32]:
    """Vectorized decoding; temporary storage scales with the supplied slice."""
    size = 28 if dense else 34
    if n_elems % 128 or len(payload) < n_elems // 128 * size:
        raise ValueError("Prism ternary payload requires complete g128 blocks")
    raw = np.frombuffer(payload, np.uint8, count=n_elems // 128 * size)
    blocks = raw.reshape(-1, size)
    prefix = "<" if endian == "little" else ">"
    scale_bytes = blocks[:, 26:28] if dense else blocks[:, :2]
    scales = np.frombuffer(scale_bytes.copy().tobytes(), prefix + "f2")
    if not dense:
        codes = (blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
        trits = codes.reshape(-1, 128).astype(np.float32) - np.float32(1)
    else:
        trits = np.empty((len(blocks), 128), dtype=np.float32)
        out = 0
        # PTQ1_0 emits five base-3 digits, lane-major, in 16 then 8 lanes.
        for start, width, digits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
            packed = blocks[:, start : start + width].astype(np.uint16)
            for digit in range(digits):
                q = (packed * (3**digit)) & 255
                trits[:, out : out + width] = ((q * 3) >> 8).astype(np.float32) - 1
                out += width
    trits *= scales.astype(np.float32)[:, None]
    return trits.reshape(-1)
