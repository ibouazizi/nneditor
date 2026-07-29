"""Generate the example tracing inputs in ``examples/trace-inputs``.

Pillow is intentionally not a runtime dependency. Run this utility with:

    uv run --with pillow python tools/generate_example_inputs.py \
        --image path/to/source.png \
        --output-dir examples/trace-inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image  # type: ignore[import-not-found]

_TIME_STEPS = 256
_IMAGE_SIZE = 224
_SOURCE_PREVIEW_SIZE = 512
_MASK_SIZE = 64


def _save_array(path: Path, array: NDArray[np.generic]) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _time_series() -> NDArray[np.float32]:
    """Return one deterministic, standardized four-channel sensor window."""
    rng = np.random.default_rng(20260729)
    seconds = np.arange(_TIME_STEPS, dtype=np.float32) / np.float32(10.0)

    temperature = (
        22.0
        + 1.8 * np.sin(2.0 * np.pi * seconds / 24.0)
        + 0.12 * rng.standard_normal(_TIME_STEPS)
    )
    humidity = (
        52.0
        - 4.5 * np.sin(2.0 * np.pi * seconds / 24.0)
        + 0.35 * rng.standard_normal(_TIME_STEPS)
    )
    vibration = (
        0.55 * np.sin(2.0 * np.pi * 1.4 * seconds)
        + 0.18 * np.sin(2.0 * np.pi * 3.6 * seconds)
        + 0.05 * rng.standard_normal(_TIME_STEPS)
    )
    event = 1.4 * np.exp(-0.5 * ((seconds - 15.0) / 0.75) ** 2)

    channels = np.stack((temperature, humidity, vibration, event), axis=-1)
    means = channels.mean(axis=0, keepdims=True)
    standard_deviations = channels.std(axis=0, keepdims=True)
    standardized = (channels - means) / standard_deviations
    return standardized.astype(np.float32, copy=False)[np.newaxis, ...]


def _center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    return image.crop((left, top, left + edge, top + edge))


def _image_tensor(
    source: Path,
    preview_destination: Path,
) -> NDArray[np.float32]:
    with Image.open(source) as opened:
        square = _center_square(opened.convert("RGB"))
        preview = square.resize(
            (_SOURCE_PREVIEW_SIZE, _SOURCE_PREVIEW_SIZE),
            Image.Resampling.LANCZOS,
        )
        preview.save(preview_destination, format="PNG", optimize=True)
        model_image = preview.resize(
            (_IMAGE_SIZE, _IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(model_image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(
        np.transpose(pixels, (2, 0, 1))[np.newaxis, ...],
        dtype=np.float32,
    )


def generate(source: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    time_series_path = output_directory / "sensor_time_series_btf32.npy"
    image_path = output_directory / "fox_image_nchw_f32.npy"
    mask_path = output_directory / "pixel_mask_bhw_i64.npy"
    preview_path = output_directory / "fox_image_source.png"

    time_series = _time_series()
    image = _image_tensor(source, preview_path)
    mask = np.ones((1, _MASK_SIZE, _MASK_SIZE), dtype=np.int64)
    _save_array(time_series_path, time_series)
    _save_array(image_path, image)
    _save_array(mask_path, mask)

    manifest = {
        "format": "NNEditor example tracing inputs",
        "arrays": {
            time_series_path.name: {
                "description": (
                    "Standardized temperature, humidity, vibration, and event channels."
                ),
                "dtype": str(time_series.dtype),
                "layout": "batch, time, features",
                "shape": list(time_series.shape),
                "minimum": float(time_series.min()),
                "maximum": float(time_series.max()),
                "sha256": _sha256(time_series_path),
            },
            image_path.name: {
                "description": "RGB fox image scaled to the [0, 1] range.",
                "dtype": str(image.dtype),
                "layout": "batch, channels, height, width",
                "shape": list(image.shape),
                "minimum": float(image.min()),
                "maximum": float(image.max()),
                "source_preview": preview_path.name,
                "sha256": _sha256(image_path),
            },
            mask_path.name: {
                "description": (
                    "All-valid pixel mask for a required int64 pixel_mask input."
                ),
                "dtype": str(mask.dtype),
                "layout": "batch, height, width",
                "shape": list(mask.shape),
                "minimum": int(mask.min()),
                "maximum": int(mask.max()),
                "sha256": _sha256(mask_path),
            },
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.image.resolve(), arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
