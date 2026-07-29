"""Renderer-agnostic activation visualization view models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

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


_VIRIDIS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)


def _viridis_colors(values: np.ndarray) -> tuple[str, ...]:
    numeric = values.astype(np.float64, copy=False).ravel()
    finite = numeric[np.isfinite(numeric)]
    if not finite.size:
        return tuple("#98A2B3" for _ in numeric)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = maximum - minimum
    colors: list[str] = []
    for value in numeric:
        if not np.isfinite(value):
            colors.append("#98A2B3")
            continue
        normalized = 0.5 if span == 0.0 else (float(value) - minimum) / span
        position = normalized * (len(_VIRIDIS) - 1)
        lower = min(int(position), len(_VIRIDIS) - 2)
        fraction = position - lower
        rgb = tuple(
            round(start + (end - start) * fraction)
            for start, end in zip(_VIRIDIS[lower], _VIRIDIS[lower + 1], strict=True)
        )
        colors.append("#" + "".join(f"{channel:02X}" for channel in rgb))
    return tuple(colors)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values.astype(np.float64, copy=False))]


def _sample_axis(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _matrix_view(
    matrix: np.ndarray,
    *,
    kind: PlotKind,
    source_shape: tuple[int, ...],
    partial: bool,
    maximum: int,
    leading_selection: str = "",
) -> ActivationVisualization:
    rows = _sample_axis(matrix.shape[-2], maximum)
    columns = _sample_axis(matrix.shape[-1], maximum)
    sampled = matrix[np.ix_(rows, columns)]
    reduced = sampled.shape != matrix.shape
    return ActivationVisualization(
        kind=kind,
        title=(
            "Attention map" if kind is PlotKind.ATTENTION_MAP else "Activation heatmap"
        ),
        values=tuple(float(value) for value in sampled.ravel()),
        shape=tuple(int(item) for item in sampled.shape),
        source_shape=source_shape,
        colormap="viridis (sequential)",
        normalization="global finite-value min-max; non-finite values are masked",
        downsampling=(
            leading_selection
            + f"deterministic evenly-spaced axes to at most {maximum}x{maximum}"
            if reduced
            else leading_selection or "none"
        ),
        partial=partial,
        colors=_viridis_colors(sampled),
    )


def build_activation_visualizations(
    record: ActivationRecord,
    raw: bytes,
    *,
    attention: bool = False,
    max_points: int = 256,
    max_map_extent: int = 32,
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
                maximum=max_map_extent,
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
                maximum=max_map_extent,
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
        row_indices = _sample_axis(maps.shape[-2], max_map_extent)
        column_indices = _sample_axis(maps.shape[-1], max_map_extent)
        sampled = maps[map_indices][:, row_indices][:, :, column_indices]
        reduced = sampled.shape != maps.shape
        views.append(
            ActivationVisualization(
                kind=PlotKind.FEATURE_MAP_GRID,
                title="Feature maps",
                values=tuple(float(value) for value in sampled.ravel()),
                shape=tuple(int(item) for item in sampled.shape),
                source_shape=source_shape,
                colormap="viridis (sequential), shared across maps",
                normalization="global finite-value min-max across displayed maps",
                downsampling=(
                    leading_selection
                    + "deterministic evenly-spaced maps and spatial axes to "
                    f"{max_feature_maps} maps of at most "
                    f"{max_map_extent}x{max_map_extent}"
                    if reduced
                    else leading_selection or "none"
                ),
                partial=partial,
                colors=_viridis_colors(sampled),
            )
        )
    return tuple(views)
