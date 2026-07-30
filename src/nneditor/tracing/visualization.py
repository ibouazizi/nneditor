"""Renderer-agnostic activation visualization view models."""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image

from nneditor.tracing.contracts import ActivationRecord, CaptureState

__all__ = [
    "ActivationVisualization",
    "PlotKind",
    "build_activation_visualizations",
]


class PlotKind(StrEnum):
    LINE = "line"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    FEATURE_MAP_GRID = "feature-map-grid"
    ATTENTION_MAP = "attention-map"
    RGB_IMAGE = "rgb-image"
    IMAGE_CHANNEL_STACK = "image-channel-stack"
    TENSOR_LAYER_STACK = "tensor-layer-stack"


@dataclass(frozen=True, slots=True)
class ActivationVisualization:
    """Numeric presentation data without a dependency on any UI toolkit."""

    kind: PlotKind
    title: str
    values: tuple[float, ...]
    shape: tuple[int, ...]
    source_shape: tuple[int, ...]
    colormap: str
    normalization: str
    downsampling: str
    partial: bool
    bin_edges: tuple[float, ...] = ()
    colors: tuple[str, ...] = ()
    raster_png: bytes = b""
    raster_size: tuple[int, int] | None = None
    layer_pngs: tuple[bytes, ...] = ()
    layer_labels: tuple[str, ...] = ()
    layer_indices: tuple[int, ...] = ()
    layer_shape: tuple[int, int] | None = None


_VIRIDIS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)


def _viridis_lut() -> np.ndarray:
    anchors = np.linspace(0, 255, len(_VIRIDIS))
    positions = np.arange(256)
    palette = np.asarray(_VIRIDIS, dtype=np.float64)
    channels = [
        np.interp(positions, anchors, palette[:, channel]) for channel in range(3)
    ]
    return np.asarray(np.rint(np.stack(channels, axis=1)), dtype=np.uint8)


_VIRIDIS_LUT = _viridis_lut()


def _viridis_rgb(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    numeric = values.astype(np.float32, copy=False)
    finite = np.isfinite(numeric)
    if not finite.any():
        return np.full((*numeric.shape, 3), (152, 162, 179), dtype=np.uint8)
    span = maximum - minimum
    normalized = (
        np.full(numeric.shape, 0.5, dtype=np.float32)
        if span == 0.0
        else np.clip((numeric - minimum) / span, 0.0, 1.0)
    )
    normalized[~finite] = 0.5
    indices = np.asarray(
        np.rint(normalized * (len(_VIRIDIS_LUT) - 1)),
        dtype=np.uint8,
    )
    rgb = _VIRIDIS_LUT[indices]
    rgb[~finite] = (152, 162, 179)
    return np.asarray(rgb, dtype=np.uint8)


def _feature_map_raster(maps: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    count, rows, columns = maps.shape
    grid_columns = max(1, int(np.ceil(np.sqrt(count))))
    grid_rows = max(1, int(np.ceil(count / grid_columns)))
    gap = 1
    width = grid_columns * columns + (grid_columns - 1) * gap
    height = grid_rows * rows + (grid_rows - 1) * gap
    raster = np.full((height, width, 3), 255, dtype=np.uint8)
    minimum = float("inf")
    maximum = float("-inf")
    for feature in maps:
        numeric = feature.astype(np.float32, copy=False)
        finite = np.isfinite(numeric)
        if finite.any():
            minimum = min(minimum, float(np.min(numeric[finite])))
            maximum = max(maximum, float(np.max(numeric[finite])))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        minimum = maximum = 0.0
    for index in range(count):
        column = index % grid_columns
        row = index // grid_columns
        left = column * (columns + gap)
        top = row * (rows + gap)
        raster[top : top + rows, left : left + columns] = _viridis_rgb(
            maps[index],
            minimum=minimum,
            maximum=maximum,
        )
    encoded = io.BytesIO()
    Image.fromarray(raster).save(encoded, format="PNG")
    return encoded.getvalue(), (width, height)


def _encode_png(array: np.ndarray) -> bytes:
    encoded = io.BytesIO()
    Image.fromarray(array).save(encoded, format="PNG")
    return encoded.getvalue()


def _qwen_patch_grid(patch_count: int) -> tuple[int, int] | None:
    candidates = [
        (height, patch_count // height)
        for height in range(2, int(np.sqrt(patch_count)) + 1, 2)
        if patch_count % height == 0 and (patch_count // height) % 2 == 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1] / item[0])


def _qwen_image_planes(array: np.ndarray) -> np.ndarray | None:
    if array.ndim != 2 or array.shape[1] != 3 * 2 * 16 * 16:
        return None
    grid = _qwen_patch_grid(int(array.shape[0]))
    if grid is None:
        return None
    grid_height, grid_width = grid
    packed = array.reshape(
        1,
        grid_height // 2,
        grid_width // 2,
        2,
        2,
        3,
        2,
        16,
        16,
    )
    one_temporal_frame = packed[:, :, :, :, :, :, 0, :, :]
    channel_first = np.transpose(
        one_temporal_frame,
        (0, 5, 1, 3, 6, 2, 4, 7),
    )
    return channel_first.reshape((3, grid_height * 16, grid_width * 16))


def _image_planes(
    record: ActivationRecord,
    array: np.ndarray,
) -> tuple[np.ndarray, str] | None:
    normalized_name = record.value_name.lower().replace("-", "_")
    if "pixel" in normalized_name:
        qwen = _qwen_image_planes(array)
        if qwen is not None:
            return qwen, "Qwen3-VL flattened patches"
    if not any(marker in normalized_name for marker in ("pixel", "image", "rgb")):
        return None
    candidate = array
    if candidate.ndim == 4 and candidate.shape[0] == 1:
        candidate = candidate[0]
    if candidate.ndim != 3:
        return None
    if candidate.shape[0] == 3:
        return candidate, "channel-first RGB tensor"
    if candidate.shape[-1] == 3:
        return np.transpose(candidate, (2, 0, 1)), "channel-last RGB tensor"
    return None


def _rgb_pixels(planes: np.ndarray) -> tuple[np.ndarray, str]:
    numeric = planes.astype(np.float32, copy=False)
    finite = numeric[np.isfinite(numeric)]
    if not finite.size:
        return (
            np.full((*numeric.shape[1:], 3), 152, dtype=np.uint8),
            "non-finite values masked",
        )
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum >= -1.001 and maximum <= 1.001 and minimum < 0.0:
        normalized = (numeric + 1.0) / 2.0
        disclosure = "-1 to 1 values restored to RGB"
    elif minimum >= 0.0 and maximum <= 1.001:
        normalized = numeric
        disclosure = "0 to 1 values restored to RGB"
    elif minimum >= 0.0 and maximum <= 255.0:
        normalized = numeric / 255.0
        disclosure = "0 to 255 values restored to RGB"
    else:
        span = maximum - minimum
        normalized = (
            np.full(numeric.shape, 0.5, dtype=np.float32)
            if span == 0.0
            else (numeric - minimum) / span
        )
        disclosure = "finite values min-max normalized for RGB display"
    normalized = np.where(np.isfinite(normalized), normalized, 0.5)
    pixels = np.asarray(
        np.rint(np.clip(normalized, 0.0, 1.0) * 255.0),
        dtype=np.uint8,
    )
    return np.transpose(pixels, (1, 2, 0)), disclosure


def _layer_pngs(
    maps: np.ndarray,
    *,
    channel_tints: tuple[int, ...] | None = None,
) -> tuple[bytes, ...]:
    numeric = maps.astype(np.float32, copy=False)
    finite = numeric[np.isfinite(numeric)]
    minimum = float(np.min(finite)) if finite.size else 0.0
    maximum = float(np.max(finite)) if finite.size else 0.0
    encoded: list[bytes] = []
    for index, layer in enumerate(numeric):
        if channel_tints is None:
            pixels = _viridis_rgb(
                layer,
                minimum=minimum,
                maximum=maximum,
            )
        else:
            span = maximum - minimum
            normalized = (
                np.full(layer.shape, 0.5, dtype=np.float32)
                if span == 0.0
                else np.clip((layer - minimum) / span, 0.0, 1.0)
            )
            normalized = np.where(np.isfinite(normalized), normalized, 0.5)
            values = np.asarray(np.rint(normalized * 255), dtype=np.uint8)
            pixels = np.zeros((*layer.shape, 3), dtype=np.uint8)
            pixels[..., channel_tints[index]] = values
        encoded.append(_encode_png(pixels))
    return tuple(encoded)


def _tensor_layer_view(
    record: ActivationRecord,
    array: np.ndarray,
    *,
    maximum_layers: int,
    maximum_extent: int,
) -> ActivationVisualization | None:
    if array.ndim < 2 or array.ndim > 4:
        return None
    leading_shape = array.shape[:-2]
    maps = np.asarray(array).reshape((-1, *array.shape[-2:]))
    indices = _sample_axis(maps.shape[0], maximum_layers)
    selected = maps[indices]
    selected, sampled_from = _bound_spatial(selected, maximum_extent)
    labels: list[str] = []
    for index in indices:
        coordinates = (
            np.unravel_index(int(index), leading_shape) if leading_shape else ()
        )
        prefix = ", ".join(str(int(item)) for item in coordinates)
        labels.append(f"[{prefix + ', ' if prefix else ''}:, :]")
    rows, columns = selected.shape[-2:]
    reduced = len(indices) != maps.shape[0]
    return ActivationVisualization(
        kind=PlotKind.TENSOR_LAYER_STACK,
        title="Activation layers",
        values=(),
        shape=tuple(int(item) for item in array.shape),
        source_shape=record.shape,
        colormap="viridis (sequential), shared across layers",
        normalization="global finite-value min-max across displayed layers",
        downsampling=(
            (
                f"deterministic evenly-spaced selection of {len(indices)} from "
                f"{maps.shape[0]} layers; "
                if reduced
                else f"all {maps.shape[0]} layers; "
            )
            + (
                f"exact {rows}x{columns} spatial axes"
                if sampled_from is None
                else f"spatial axes deterministically sampled to {rows}x{columns} "
                f"from {sampled_from[0]}x{sampled_from[1]}"
            )
        ),
        partial=False,
        layer_pngs=_layer_pngs(selected),
        layer_labels=tuple(labels),
        layer_indices=tuple(int(index) for index in indices),
        layer_shape=(rows, columns),
    )


def _image_views(
    record: ActivationRecord,
    array: np.ndarray,
    *,
    partial: bool,
) -> tuple[ActivationVisualization, ...]:
    resolved = _image_planes(record, array)
    if resolved is None:
        return ()
    planes, layout = resolved
    rgb, normalization = _rgb_pixels(planes)
    rows, columns = planes.shape[-2:]
    image_png = _encode_png(rgb)
    exact = f"{layout}; exact {rows}x{columns} spatial axes"
    return (
        ActivationVisualization(
            kind=PlotKind.RGB_IMAGE,
            title="Reconstructed RGB image",
            values=(),
            shape=(rows, columns, 3),
            source_shape=record.shape,
            colormap="RGB",
            normalization=normalization,
            downsampling=exact,
            partial=partial,
            raster_png=image_png,
            raster_size=(columns, rows),
        ),
        ActivationVisualization(
            kind=PlotKind.IMAGE_CHANNEL_STACK,
            title="RGB channel planes",
            values=(),
            shape=(3, rows, columns),
            source_shape=record.shape,
            colormap="red, green, and blue channel tints",
            normalization=normalization,
            downsampling=exact + "; three planes offset for depth",
            partial=partial,
            layer_pngs=_layer_pngs(
                planes,
                channel_tints=(0, 1, 2),
            ),
            layer_labels=("Red · [0, :, :]", "Green · [1, :, :]", "Blue · [2, :, :]"),
            layer_indices=(0, 1, 2),
            layer_shape=(rows, columns),
        ),
    )


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values.astype(np.float64, copy=False))]


def _sample_axis(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _bound_spatial(
    maps: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Deterministically sample the trailing two axes down to `maximum`.

    Rasterizing a large matrix at full resolution costs one uint8 RGB pixel
    per element, so an unbounded attention map would build a raster (and PNG)
    hundreds of times larger than the panel can show, inside a job thread.
    Returns the array unchanged with `None` when both axes already fit, and
    otherwise the sampled array plus the original extent so callers disclose
    that the view is a sample rather than the exact tensor.
    """
    rows, columns = (int(item) for item in maps.shape[-2:])
    if rows <= maximum and columns <= maximum:
        return maps, None
    row_indices = _sample_axis(rows, maximum)
    column_indices = _sample_axis(columns, maximum)
    sampled = maps[..., row_indices[:, None], column_indices[None, :]]
    return sampled, (rows, columns)


def _matrix_view(
    matrix: np.ndarray,
    *,
    kind: PlotKind,
    source_shape: tuple[int, ...],
    partial: bool,
    maximum_extent: int,
    leading_selection: str = "",
) -> ActivationVisualization:
    matrix, sampled_from = _bound_spatial(matrix, maximum_extent)
    raster_png, raster_size = _feature_map_raster(matrix.reshape((1, *matrix.shape)))
    rows, columns = matrix.shape[-2:]
    return ActivationVisualization(
        kind=kind,
        title=(
            "Attention map" if kind is PlotKind.ATTENTION_MAP else "Activation heatmap"
        ),
        values=(),
        shape=tuple(int(item) for item in matrix.shape),
        source_shape=source_shape,
        colormap="viridis (sequential)",
        normalization="global finite-value min-max; non-finite values are masked",
        downsampling=(
            leading_selection
            + (
                f"spatial axes preserved at exact {rows}x{columns}"
                if sampled_from is None
                else "spatial axes deterministically sampled to "
                f"{rows}x{columns} from {sampled_from[0]}x{sampled_from[1]}"
            )
        ),
        partial=partial,
        raster_png=raster_png,
        raster_size=raster_size,
    )


def build_activation_visualizations(
    record: ActivationRecord,
    raw: bytes,
    *,
    attention: bool = False,
    max_points: int = 256,
    # Caps the trailing two axes of every rasterized summary view. 512 keeps
    # ordinary activations exact while bounding a rasterized view to roughly
    # 512x512 pixels, so a large attention matrix cannot build a raster (and
    # PNG) orders of magnitude bigger than the panel can display. The
    # reconstructed RGB image view is deliberately exempt: it is a faithful
    # image rather than a summary, and sampling it would defeat its purpose.
    max_map_extent: int = 512,
    max_feature_maps: int = 16,
) -> tuple[ActivationVisualization, ...]:
    """Build deterministic line, histogram, heatmap, and feature-map views."""
    if max_points <= 0 or max_map_extent <= 0 or max_feature_maps <= 0:
        raise ValueError("visualization limits must be positive")
    try:
        dtype = np.dtype(record.numpy_dtype)
    except TypeError as error:
        raise ValueError(
            f"activation dtype {record.numpy_dtype!r} cannot be visualized"
        ) from error
    usable = len(raw) - len(raw) % max(1, dtype.itemsize)
    flat = np.frombuffer(raw[:usable], dtype=dtype)
    partial = record.state is not CaptureState.COMPLETE
    expected = int(np.prod(record.shape, dtype=np.int64)) if record.shape else 1
    complete_shape = (
        not partial
        and all(dimension >= 0 for dimension in record.shape)
        and flat.size == expected
    )
    array = flat.reshape(record.shape) if complete_shape else flat
    source_shape = record.shape
    views: list[ActivationVisualization] = []

    finite = _finite(flat)
    if finite.size:
        bins = min(32, max(1, int(np.sqrt(finite.size))))
        counts, edges = np.histogram(finite.astype(np.float64), bins=bins)
        views.append(
            ActivationVisualization(
                kind=PlotKind.HISTOGRAM,
                title="Value distribution",
                values=tuple(float(value) for value in counts),
                shape=(len(counts),),
                source_shape=source_shape,
                colormap="single-color accent",
                normalization="raw finite-value counts; NaN/Inf excluded",
                downsampling="none; all captured finite values contribute",
                partial=partial,
                bin_edges=tuple(float(value) for value in edges),
                colors=tuple("#5B5CE2" for _ in counts),
            )
        )

    image_views = _image_views(
        record,
        array,
        partial=partial,
    )
    if image_views:
        return (*image_views, *views)

    layer_view = _tensor_layer_view(
        record,
        array,
        maximum_layers=max_feature_maps,
        maximum_extent=max_map_extent,
    )
    if layer_view is not None:
        views.insert(0, layer_view)

    if array.ndim <= 1:
        indices = _sample_axis(flat.size, max_points)
        views.append(
            ActivationVisualization(
                kind=PlotKind.LINE,
                title="Activation values",
                values=tuple(float(flat[index]) for index in indices),
                shape=(len(indices),),
                source_shape=source_shape,
                colormap="single-color accent",
                normalization="raw values",
                downsampling=(
                    f"deterministic evenly-spaced sample of {len(indices)} "
                    f"from {flat.size} captured values"
                    if flat.size > max_points
                    else "none"
                ),
                partial=partial,
                colors=tuple("#5B5CE2" for _ in indices),
            )
        )
        return tuple(views)

    if attention and array.ndim >= 2:
        attention_matrix = np.asarray(array).reshape((-1, *array.shape[-2:]))[0]
        views.append(
            _matrix_view(
                attention_matrix,
                kind=PlotKind.ATTENTION_MAP,
                source_shape=source_shape,
                partial=partial,
                maximum_extent=max_map_extent,
                leading_selection=(
                    "first leading-index matrix; " if array.ndim > 2 else ""
                ),
            )
        )
    elif array.ndim == 2:
        views.append(
            _matrix_view(
                array,
                kind=PlotKind.HEATMAP,
                source_shape=source_shape,
                partial=partial,
                maximum_extent=max_map_extent,
            )
        )
    else:
        maps = np.asarray(array)
        leading_selection = ""
        if maps.ndim >= 4:
            leading_selection = "first batch; "
            maps = maps[0]
        maps = maps.reshape((-1, *maps.shape[-2:]))
        map_indices = _sample_axis(maps.shape[0], max_feature_maps)
        sampled = maps[map_indices]
        reduced = len(map_indices) != maps.shape[0]
        sampled, sampled_from = _bound_spatial(sampled, max_map_extent)
        raster_png, raster_size = _feature_map_raster(sampled)
        rows, columns = sampled.shape[-2:]
        spatial = (
            f"spatial axes preserved at exact {rows}x{columns}"
            if sampled_from is None
            else f"spatial axes deterministically sampled to {rows}x{columns} "
            f"from {sampled_from[0]}x{sampled_from[1]}"
        )
        views.append(
            ActivationVisualization(
                kind=PlotKind.FEATURE_MAP_GRID,
                title="Feature maps",
                values=(),
                shape=tuple(int(item) for item in sampled.shape),
                source_shape=source_shape,
                colormap="viridis (sequential), shared across maps",
                normalization="global finite-value min-max across displayed maps",
                downsampling=(
                    leading_selection
                    + f"deterministic evenly-spaced maps to {max_feature_maps}; "
                    + spatial
                    if reduced
                    else leading_selection + spatial
                ),
                partial=partial,
                raster_png=raster_png,
                raster_size=raster_size,
            )
        )
    return tuple(views)
