"""Bounded generation of safe NumPy trace-input artifacts."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from nneditor.tokenization import Codebook

MAX_GENERATED_TENSOR_BYTES: Final = 512 * 1024 * 1024
MAX_TABULAR_SOURCE_BYTES: Final = 64 * 1024 * 1024
MAX_IMAGE_EXTENT: Final = 16_384

NumericArray = NDArray[np.generic]
ImageLayout = Literal[
    "NCHW",
    "NHWC",
    "CHW",
    "HWC",
    "NHW",
    "HW",
    "QWEN3VL_PATCHES",
]
ImageColorMode = Literal["rgb", "grayscale"]
ImageNormalization = Literal["none", "zero-one", "minus-one-one", "imagenet"]
MaskFill = Literal["ones", "zeros", "checkerboard", "random-binary"]
TimeSeriesWaveform = Literal["sine", "cosine", "sawtooth", "random-walk"]
TimeSeriesLayout = Literal["NTC", "NCT", "TC", "CT"]
TokenLayout = Literal["NT", "T"]
SyntheticDistribution = Literal[
    "zeros",
    "ones",
    "uniform",
    "normal",
    "ramp",
]

_NUMERIC_DTYPES: Final = frozenset(
    {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    }
)
_FLOAT_DTYPES: Final = frozenset({"float16", "float32", "float64"})
_TOKEN_DTYPES: Final = frozenset({"int32", "int64", "uint32", "uint64"})


class InputGenerationError(ValueError):
    """A requested test tensor is invalid or exceeds the generation bounds."""


@dataclass(frozen=True, slots=True)
class GeneratedTensor:
    """Metadata for one completely published ``.npy`` artifact."""

    path: Path
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    source: str


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse ``1,3,224,224`` and ``1x3x224x224`` style shapes."""
    normalized = (
        value.strip()
        .removeprefix("[")
        .removesuffix("]")
        .removeprefix("(")
        .removesuffix(")")
        .lower()
        .replace("\N{MULTIPLICATION SIGN}", "x")
        .replace("x", ",")
        .replace(" ", ",")
    )
    parts = tuple(part for part in normalized.split(",") if part)
    if not parts:
        raise InputGenerationError("enter at least one concrete tensor extent")
    try:
        shape = tuple(int(part, 10) for part in parts)
    except ValueError as error:
        raise InputGenerationError(
            "tensor shapes must contain only concrete integer extents"
        ) from error
    if any(extent <= 0 for extent in shape):
        raise InputGenerationError("every tensor extent must be greater than zero")
    if len(shape) > 16:
        raise InputGenerationError("tensor rank is limited to 16 dimensions")
    return shape


def parse_columns(value: str) -> tuple[int, ...] | None:
    """Parse an optional comma-separated CSV column selection."""
    normalized = value.strip()
    if not normalized:
        return None
    try:
        columns = tuple(int(part.strip(), 10) for part in normalized.split(","))
    except ValueError as error:
        raise InputGenerationError(
            "CSV columns must be zero-based integer indexes"
        ) from error
    if not columns or any(column < 0 for column in columns):
        raise InputGenerationError("CSV column indexes cannot be negative")
    if len(set(columns)) != len(columns):
        raise InputGenerationError("CSV column indexes cannot be repeated")
    return columns


def generate_image_tensor(
    image_path: Path | str,
    destination: Path | str,
    *,
    width: int,
    height: int,
    layout: ImageLayout = "NCHW",
    color_mode: ImageColorMode = "rgb",
    normalization: ImageNormalization = "zero-one",
    dtype: str = "float32",
) -> GeneratedTensor:
    """Resize an image and publish it as a bounded NumPy tensor."""
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise InputGenerationError(f"image file does not exist: {source}")
    if width <= 0 or height <= 0:
        raise InputGenerationError("image width and height must be greater than zero")
    if width > MAX_IMAGE_EXTENT or height > MAX_IMAGE_EXTENT:
        raise InputGenerationError(
            f"image width and height are limited to {MAX_IMAGE_EXTENT:,}"
        )
    target_dtype = _dtype(dtype)
    if normalization != "none" and dtype not in _FLOAT_DTYPES:
        raise InputGenerationError(
            "image normalization requires float32 or float64 output"
        )
    if normalization == "imagenet" and color_mode != "rgb":
        raise InputGenerationError("ImageNet normalization requires an RGB image")
    if layout == "QWEN3VL_PATCHES":
        if color_mode != "rgb":
            raise InputGenerationError("Qwen3-VL patch inputs require an RGB image")
        if normalization != "minus-one-one":
            raise InputGenerationError(
                "Qwen3-VL patch inputs require -1 to 1 normalization"
            )
        if width % 32 or height % 32:
            raise InputGenerationError(
                "Qwen3-VL image width and height must be divisible by 32"
            )
    channels = 3 if color_mode == "rgb" else 1
    _bounded_allocation((height, width, channels), target_dtype)

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB" if color_mode == "rgb" else "L")
            resampling = (
                Image.Resampling.BICUBIC
                if layout == "QWEN3VL_PATCHES"
                else Image.Resampling.BILINEAR
            )
            image = image.resize((width, height), resampling)
            array = np.asarray(image).copy()
    except (OSError, ValueError) as error:
        raise InputGenerationError(f"could not decode image: {error}") from error

    if normalization == "none":
        array = array.astype(target_dtype, copy=False)
    else:
        floating = array.astype(target_dtype) / 255.0
        if normalization == "zero-one":
            array = floating
        elif normalization == "minus-one-one":
            array = floating * 2.0 - 1.0
        else:
            mean = np.asarray((0.485, 0.456, 0.406), dtype=target_dtype)
            standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=target_dtype)
            array = (floating - mean) / standard_deviation

    if layout == "QWEN3VL_PATCHES":
        array = _qwen3vl_patches(array, target_dtype)
        source_description = (
            f"Qwen3-VL image {source.name} resized to H={height} x W={width}"
        )
    else:
        array = _apply_image_layout(array, layout, color_mode)
        source_description = f"image {source.name} resized to H={height} x W={width}"
    return _publish_array(
        array,
        destination,
        source=source_description,
    )


def _qwen3vl_patches(
    array: NumericArray,
    dtype: np.dtype[np.generic],
) -> NumericArray:
    """Match the official Qwen3-VL image processor's flattened patch order."""
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    height, width, channels = array.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    output_shape = (
        grid_height * grid_width,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    _bounded_allocation(output_shape, dtype)

    channel_first = np.transpose(array, (2, 0, 1))[np.newaxis, ...]
    patches = channel_first.reshape(
        1,
        channels,
        grid_height // merge_size,
        merge_size,
        patch_size,
        grid_width // merge_size,
        merge_size,
        patch_size,
    )
    patches = np.transpose(patches, (0, 2, 5, 3, 6, 1, 4, 7))
    patches = np.repeat(patches[:, :, :, :, :, :, np.newaxis, :, :], 2, axis=6)
    flattened: NumericArray = patches.reshape((1, *output_shape))[0]
    return flattened


def generate_mask_tensor(
    destination: Path | str,
    *,
    shape: tuple[int, ...],
    fill: MaskFill = "ones",
    dtype: str = "int64",
    seed: int = 0,
) -> GeneratedTensor:
    """Create an all-valid, empty, checkerboard, or random binary mask."""
    target_dtype = _dtype(dtype)
    _bounded_allocation(shape, target_dtype)
    if fill == "ones":
        array = np.ones(shape, dtype=target_dtype)
    elif fill == "zeros":
        array = np.zeros(shape, dtype=target_dtype)
    elif fill == "checkerboard":
        if len(shape) < 2:
            raise InputGenerationError(
                "checkerboard masks require at least two dimensions"
            )
        rows, columns = np.indices(shape[-2:])
        plane = ((rows + columns) % 2).astype(target_dtype)
        array = np.broadcast_to(plane, shape).copy()
    elif fill == "random-binary":
        generator = np.random.default_rng(seed)
        array = generator.integers(0, 2, size=shape, dtype=np.uint8).astype(
            target_dtype
        )
    else:
        raise InputGenerationError(f"unsupported mask fill: {fill}")
    return _publish_array(array, destination, source=f"{fill} mask")


def generate_token_ids_tensor(
    destination: Path | str,
    *,
    text: str,
    codebook: Codebook,
    sequence_length: int | None = None,
    layout: TokenLayout = "NT",
    dtype: str = "int64",
    pad_id: int = 0,
    bos_id: int | None = None,
    eos_id: int | None = None,
    truncate: bool = True,
) -> GeneratedTensor:
    """Encode ``text`` into a language-model token-id tensor.

    ``sequence_length`` pads with ``pad_id`` or truncates to a fixed length;
    omit it to keep the natural encoded length. Optional ``bos_id``/``eos_id``
    wrap the sequence and are counted against the length, so a truncated
    sequence still ends with its end-of-sequence id.

    Models that also take an attention mask need one generated alongside this
    tensor; a fixed length here means the mask is all-valid up to the token
    count reported in :attr:`GeneratedTensor.source`.
    """
    target_dtype = _dtype(dtype, allowed=_TOKEN_DTYPES)
    encoded = list(codebook.encode(text))
    if bos_id is not None:
        encoded.insert(0, bos_id)
    if eos_id is not None:
        encoded.append(eos_id)
    if not encoded:
        raise InputGenerationError(
            "the text produced no tokens; supply text or a bos/eos id"
        )
    limit = codebook.vocab_size
    for identifier in (pad_id, bos_id, eos_id):
        if identifier is not None and not 0 <= identifier < limit:
            raise InputGenerationError(
                f"token id {identifier} is outside the codebook's vocabulary of {limit}"
            )
    out_of_range = [item for item in encoded if not 0 <= item < limit]
    if out_of_range:
        raise InputGenerationError(
            f"{len(out_of_range)} encoded id(s) fall outside the codebook's "
            f"vocabulary of {limit}"
        )

    encoded_count = len(encoded)
    padded = 0
    if sequence_length is not None:
        if sequence_length < 1:
            raise InputGenerationError("sequence_length must be positive")
        if len(encoded) > sequence_length:
            if not truncate:
                raise InputGenerationError(
                    f"{len(encoded)} tokens exceed the requested "
                    f"sequence_length of {sequence_length}"
                )
            encoded = encoded[:sequence_length]
            if eos_id is not None:
                encoded[-1] = eos_id
        else:
            padded = sequence_length - len(encoded)
            encoded.extend([pad_id] * padded)

    length = len(encoded)
    if layout == "NT":
        shape: tuple[int, ...] = (1, length)
    elif layout == "T":
        shape = (length,)
    else:
        raise InputGenerationError(f"unsupported token layout: {layout}")
    _bounded_allocation(shape, target_dtype)
    array = np.asarray(encoded, dtype=target_dtype).reshape(shape)

    detail = f"{min(encoded_count, length)} of {encoded_count} token(s)"
    if padded:
        detail += f", {padded} pad id(s)"
    elif encoded_count > length:
        detail += ", truncated"
    return _publish_array(
        array,
        destination,
        source=f"{codebook.name} tokens ({detail}; {codebook.fidelity})",
    )


def generate_csv_tensor(
    csv_path: Path | str,
    destination: Path | str,
    *,
    delimiter: str = ",",
    skip_rows: int = 0,
    columns: tuple[int, ...] | None = None,
    transpose: bool = False,
    add_batch: bool = False,
    dtype: str = "float32",
) -> GeneratedTensor:
    """Load bounded numeric CSV/TSV/time-series data into a NumPy tensor."""
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise InputGenerationError(f"data file does not exist: {source}")
    source_size = source.stat().st_size
    if source_size > MAX_TABULAR_SOURCE_BYTES:
        raise InputGenerationError(
            f"tabular source files are limited to "
            f"{MAX_TABULAR_SOURCE_BYTES / (1024 * 1024):,.0f} MiB"
        )
    if skip_rows < 0:
        raise InputGenerationError("rows to skip cannot be negative")
    target_dtype = _dtype(dtype)
    resolved_delimiter = None if delimiter in {"", "whitespace"} else delimiter
    if resolved_delimiter == r"\t":
        resolved_delimiter = "\t"
    if resolved_delimiter is not None and len(resolved_delimiter) != 1:
        raise InputGenerationError("the CSV delimiter must be one character")
    try:
        loaded = np.genfromtxt(
            source,
            delimiter=resolved_delimiter,
            skip_header=skip_rows,
            dtype=np.float64,
            ndmin=2,
        )
    except (OSError, TypeError, ValueError) as error:
        raise InputGenerationError(f"could not parse numeric data: {error}") from error
    if loaded.size == 0:
        raise InputGenerationError("the selected data file contains no numeric rows")
    if not np.all(np.isfinite(loaded)):
        raise InputGenerationError(
            "the selected data contains missing or non-finite numeric values"
        )
    if columns is not None:
        if any(column >= loaded.shape[1] for column in columns):
            raise InputGenerationError(
                f"CSV column selection exceeds the {loaded.shape[1]} available columns"
            )
        loaded = loaded[:, columns]
    if transpose:
        loaded = loaded.T
    if add_batch:
        loaded = loaded[np.newaxis, ...]
    array = loaded.astype(target_dtype)
    return _publish_array(array, destination, source=f"numeric data {source.name}")


def generate_time_series_tensor(
    destination: Path | str,
    *,
    samples: int,
    channels: int = 1,
    waveform: TimeSeriesWaveform = "sine",
    sample_rate: float = 100.0,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    layout: TimeSeriesLayout = "NTC",
    dtype: str = "float32",
    seed: int = 0,
) -> GeneratedTensor:
    """Create a deterministic multi-channel synthetic time series."""
    if samples <= 0 or channels <= 0:
        raise InputGenerationError("samples and channels must be greater than zero")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise InputGenerationError("sample rate must be a finite positive number")
    if not math.isfinite(frequency) or frequency < 0:
        raise InputGenerationError("frequency must be finite and non-negative")
    if not math.isfinite(amplitude):
        raise InputGenerationError("amplitude must be finite")
    target_dtype = _dtype(dtype, allowed=_FLOAT_DTYPES)
    # Wave synthesis uses float64 intermediates before the final cast.
    _bounded_allocation((samples, channels), np.dtype("float64"))

    time = np.arange(samples, dtype=np.float64) / sample_rate
    phases = np.arange(channels, dtype=np.float64) * (math.pi / 8.0)
    angle = 2.0 * math.pi * frequency * time[:, np.newaxis] + phases
    if waveform == "sine":
        values = np.sin(angle)
    elif waveform == "cosine":
        values = np.cos(angle)
    elif waveform == "sawtooth":
        cycles = frequency * time[:, np.newaxis] + phases / (2.0 * math.pi)
        values = 2.0 * (cycles - np.floor(cycles + 0.5))
    elif waveform == "random-walk":
        generator = np.random.default_rng(seed)
        values = np.cumsum(generator.normal(0.0, 1.0, size=(samples, channels)), axis=0)
        peak = float(np.max(np.abs(values)))
        if peak:
            values /= peak
    else:
        raise InputGenerationError(f"unsupported time-series waveform: {waveform}")
    values *= amplitude
    if layout == "TC":
        array = values
    elif layout == "CT":
        array = values.T
    elif layout == "NTC":
        array = values[np.newaxis, ...]
    elif layout == "NCT":
        array = values.T[np.newaxis, ...]
    else:
        raise InputGenerationError(f"unsupported time-series layout: {layout}")
    return _publish_array(
        array.astype(target_dtype),
        destination,
        source=f"{waveform} time series",
    )


def generate_synthetic_tensor(
    destination: Path | str,
    *,
    shape: tuple[int, ...],
    distribution: SyntheticDistribution = "zeros",
    dtype: str = "float32",
    seed: int = 0,
) -> GeneratedTensor:
    """Create a generic deterministic tensor for non-image model inputs."""
    target_dtype = _dtype(dtype)
    _bounded_allocation(shape, target_dtype)
    if distribution == "zeros":
        array = np.zeros(shape, dtype=target_dtype)
    elif distribution == "ones":
        array = np.ones(shape, dtype=target_dtype)
    elif distribution == "uniform":
        generator = np.random.default_rng(seed)
        array = generator.uniform(0.0, 1.0, size=shape).astype(target_dtype)
    elif distribution == "normal":
        generator = np.random.default_rng(seed)
        array = generator.normal(0.0, 1.0, size=shape).astype(target_dtype)
    elif distribution == "ramp":
        array = np.arange(math.prod(shape), dtype=target_dtype).reshape(shape)
    else:
        raise InputGenerationError(
            f"unsupported synthetic distribution: {distribution}"
        )
    return _publish_array(array, destination, source=f"{distribution} tensor")


def _dtype(
    name: str,
    *,
    allowed: frozenset[str] = _NUMERIC_DTYPES,
) -> np.dtype[np.generic]:
    if name not in allowed:
        raise InputGenerationError(
            f"unsupported dtype {name!r}; choose one of {', '.join(sorted(allowed))}"
        )
    return np.dtype(name)


def _bounded_allocation(
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
) -> int:
    if not shape or any(extent <= 0 for extent in shape):
        raise InputGenerationError("every tensor extent must be greater than zero")
    elements = math.prod(shape)
    byte_size = elements * dtype.itemsize
    if byte_size > MAX_GENERATED_TENSOR_BYTES:
        raise InputGenerationError(
            f"the requested tensor needs {byte_size / (1024 * 1024):,.1f} MiB; "
            f"the generator limit is "
            f"{MAX_GENERATED_TENSOR_BYTES / (1024 * 1024):,.0f} MiB"
        )
    return byte_size


def _apply_image_layout(
    array: NumericArray,
    layout: ImageLayout,
    color_mode: ImageColorMode,
) -> NumericArray:
    if color_mode == "rgb":
        if layout == "HWC":
            return array
        if layout == "CHW":
            return np.transpose(array, (2, 0, 1))
        if layout == "NHWC":
            return array[np.newaxis, ...]
        if layout == "NCHW":
            return np.transpose(array, (2, 0, 1))[np.newaxis, ...]
        raise InputGenerationError(f"layout {layout} is unavailable for RGB images")
    if layout == "HW":
        return array
    if layout == "NHW":
        return array[np.newaxis, ...]
    with_channel = array[..., np.newaxis]
    if layout == "HWC":
        return with_channel
    if layout == "CHW":
        return np.transpose(with_channel, (2, 0, 1))
    if layout == "NHWC":
        return with_channel[np.newaxis, ...]
    if layout == "NCHW":
        return np.transpose(with_channel, (2, 0, 1))[np.newaxis, ...]
    raise InputGenerationError(f"unsupported grayscale layout: {layout}")


def _publish_array(
    array: NumericArray,
    destination: Path | str,
    *,
    source: str,
) -> GeneratedTensor:
    contiguous = np.ascontiguousarray(array)
    _bounded_allocation(
        tuple(int(extent) for extent in contiguous.shape), contiguous.dtype
    )
    target = Path(destination).expanduser()
    if target.suffix.lower() != ".npy":
        target = target.with_suffix(".npy")
    target = target.resolve()
    if not target.parent.is_dir():
        raise InputGenerationError(
            f"destination directory does not exist: {target.parent}"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.save(temporary, contiguous, allow_pickle=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except OSError as error:
        raise InputGenerationError(f"could not save {target.name}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return GeneratedTensor(
        path=target,
        shape=tuple(int(extent) for extent in contiguous.shape),
        dtype=str(contiguous.dtype),
        byte_size=int(contiguous.nbytes),
        source=source,
    )
