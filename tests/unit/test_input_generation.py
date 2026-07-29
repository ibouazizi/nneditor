"""Safe test-input artifact generation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from nneditor import input_generation
from nneditor.input_generation import (
    InputGenerationError,
    generate_csv_tensor,
    generate_image_tensor,
    generate_mask_tensor,
    generate_synthetic_tensor,
    generate_time_series_tensor,
    parse_columns,
    parse_shape,
)


def test_shape_and_column_parsers_accept_compact_user_input() -> None:
    assert parse_shape("1 x 3 x 224 x 224") == (1, 3, 224, 224)
    assert parse_shape("[2, 8]") == (2, 8)
    assert parse_columns("0, 2,4") == (0, 2, 4)
    assert parse_columns("") is None
    assert parse_shape("1\N{MULTIPLICATION SIGN}2") == (1, 2)


@pytest.mark.parametrize("shape", ["", "1,0,3", "batch,3", "1,-2"])
def test_shape_parser_rejects_missing_or_non_concrete_extents(shape: str) -> None:
    with pytest.raises(InputGenerationError):
        parse_shape(shape)


def test_shape_and_column_parsers_reject_rank_and_column_errors() -> None:
    with pytest.raises(InputGenerationError, match="rank"):
        parse_shape(",".join("1" for _ in range(17)))
    for value in ("name", "-1", "1,1"):
        with pytest.raises(InputGenerationError):
            parse_columns(value)


def test_image_generation_resizes_normalizes_and_lays_out_rgb(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    Image.fromarray(pixels, mode="RGB").save(source)

    generated = generate_image_tensor(
        source,
        tmp_path / "image.npy",
        width=3,
        height=2,
        layout="NCHW",
        normalization="zero-one",
        dtype="float32",
    )

    array = np.load(generated.path, allow_pickle=False)
    assert array.shape == (1, 3, 2, 3)
    assert array.dtype == np.dtype("float32")
    assert np.all(array[:, 0] == 1.0)
    assert np.all(array[:, 1:] == 0.0)
    assert generated.byte_size == array.nbytes


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("HW", (2, 3)),
        ("NHW", (1, 2, 3)),
        ("HWC", (2, 3, 1)),
        ("CHW", (1, 2, 3)),
        ("NHWC", (1, 2, 3, 1)),
        ("NCHW", (1, 1, 2, 3)),
    ],
)
def test_grayscale_image_layouts(
    tmp_path: Path,
    layout: str,
    shape: tuple[int, ...],
) -> None:
    source = tmp_path / "gray.png"
    Image.fromarray(np.arange(6, dtype=np.uint8).reshape(2, 3), mode="L").save(source)

    generated = generate_image_tensor(
        source,
        tmp_path / f"{layout}.npy",
        width=3,
        height=2,
        layout=cast(Any, layout),
        color_mode="grayscale",
        normalization="none",
        dtype="uint8",
    )

    assert np.load(generated.path, allow_pickle=False).shape == shape


@pytest.mark.parametrize("normalization", ["minus-one-one", "imagenet"])
def test_image_float_normalization_modes(
    tmp_path: Path,
    normalization: str,
) -> None:
    source = tmp_path / "white.png"
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8), mode="RGB").save(source)

    generated = generate_image_tensor(
        source,
        tmp_path / f"{normalization}.npy",
        width=2,
        height=2,
        layout="HWC",
        normalization=cast(Any, normalization),
    )

    assert np.all(np.isfinite(np.load(generated.path, allow_pickle=False)))


def test_image_generation_rejects_invalid_sources_and_configurations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(source)
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")

    calls: list[Callable[[], object]] = [
        lambda: generate_image_tensor(
            tmp_path / "missing.png", tmp_path / "out.npy", width=2, height=2
        ),
        lambda: generate_image_tensor(source, tmp_path / "out.npy", width=0, height=2),
        lambda: generate_image_tensor(
            source,
            tmp_path / "out.npy",
            width=input_generation.MAX_IMAGE_EXTENT + 1,
            height=2,
        ),
        lambda: generate_image_tensor(
            source,
            tmp_path / "out.npy",
            width=2,
            height=2,
            normalization="zero-one",
            dtype="uint8",
        ),
        lambda: generate_image_tensor(
            source,
            tmp_path / "out.npy",
            width=2,
            height=2,
            color_mode="grayscale",
            normalization="imagenet",
        ),
        lambda: generate_image_tensor(invalid, tmp_path / "out.npy", width=2, height=2),
        lambda: generate_image_tensor(
            source,
            tmp_path / "out.npy",
            width=2,
            height=2,
            layout=cast(Any, "HW"),
        ),
    ]
    for call in calls:
        with pytest.raises(InputGenerationError):
            call()


def test_masks_include_all_valid_and_checkerboard_modes(tmp_path: Path) -> None:
    all_valid = generate_mask_tensor(
        tmp_path / "valid.npy",
        shape=(1, 2, 3),
        fill="ones",
        dtype="int64",
    )
    checkerboard = generate_mask_tensor(
        tmp_path / "checker.npy",
        shape=(1, 2, 3),
        fill="checkerboard",
        dtype="uint8",
    )

    assert np.array_equal(
        np.load(all_valid.path, allow_pickle=False),
        np.ones((1, 2, 3), dtype=np.int64),
    )
    assert np.array_equal(
        np.load(checkerboard.path, allow_pickle=False),
        np.asarray([[[0, 1, 0], [1, 0, 1]]], dtype=np.uint8),
    )


def test_zero_and_random_masks_are_supported_and_seeded(tmp_path: Path) -> None:
    zero = generate_mask_tensor(
        tmp_path / "zero.npy",
        shape=(2, 2),
        fill="zeros",
        dtype="bool",
    )
    first = generate_mask_tensor(
        tmp_path / "first.npy",
        shape=(3, 3),
        fill="random-binary",
        seed=7,
    )
    second = generate_mask_tensor(
        tmp_path / "second.npy",
        shape=(3, 3),
        fill="random-binary",
        seed=7,
    )
    assert not np.load(zero.path, allow_pickle=False).any()
    np.testing.assert_array_equal(
        np.load(first.path, allow_pickle=False),
        np.load(second.path, allow_pickle=False),
    )
    with pytest.raises(InputGenerationError, match="two dimensions"):
        generate_mask_tensor(
            tmp_path / "bad.npy",
            shape=(2,),
            fill="checkerboard",
        )
    with pytest.raises(InputGenerationError, match="unsupported mask"):
        generate_mask_tensor(
            tmp_path / "bad.npy",
            shape=(2, 2),
            fill=cast(Any, "diagonal"),
        )


def test_csv_generation_selects_columns_transposes_and_batches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "series.csv"
    source.write_text("time,a,b\n0,1,10\n1,2,20\n", encoding="utf-8")

    generated = generate_csv_tensor(
        source,
        tmp_path / "series.npy",
        skip_rows=1,
        columns=(1, 2),
        transpose=True,
        add_batch=True,
    )

    assert np.array_equal(
        np.load(generated.path, allow_pickle=False),
        np.asarray([[[1, 2], [10, 20]]], dtype=np.float32),
    )


def test_csv_whitespace_tab_and_validation_paths(tmp_path: Path) -> None:
    whitespace = tmp_path / "space.txt"
    whitespace.write_text("1 2\n3 4\n", encoding="utf-8")
    tabular = tmp_path / "tab.tsv"
    tabular.write_text("1\t2\n", encoding="utf-8")
    nonfinite = tmp_path / "bad.csv"
    nonfinite.write_text("1,nan\n", encoding="utf-8")

    assert generate_csv_tensor(
        whitespace,
        tmp_path / "space.npy",
        delimiter="whitespace",
    ).shape == (2, 2)
    assert generate_csv_tensor(
        tabular,
        tmp_path / "tab.npy",
        delimiter=r"\t",
    ).shape == (1, 2)

    calls: list[Callable[[], object]] = [
        lambda: generate_csv_tensor(tmp_path / "missing.csv", tmp_path / "out.npy"),
        lambda: generate_csv_tensor(whitespace, tmp_path / "out.npy", skip_rows=-1),
        lambda: generate_csv_tensor(whitespace, tmp_path / "out.npy", delimiter="::"),
        lambda: generate_csv_tensor(whitespace, tmp_path / "out.npy", columns=(4,)),
        lambda: generate_csv_tensor(nonfinite, tmp_path / "out.npy"),
    ]
    for call in calls:
        with pytest.raises(InputGenerationError):
            call()


def test_time_series_is_deterministic_and_respects_layout(tmp_path: Path) -> None:
    generated = generate_time_series_tensor(
        tmp_path / "wave.npy",
        samples=4,
        channels=2,
        waveform="sine",
        sample_rate=4,
        frequency=1,
        layout="NCT",
    )

    array = np.load(generated.path, allow_pickle=False)
    assert array.shape == (1, 2, 4)
    assert array.dtype == np.dtype("float32")
    assert array[0, 0] == pytest.approx([0, 1, 0, -1], abs=1e-6)


@pytest.mark.parametrize(
    ("waveform", "layout", "shape"),
    [
        ("cosine", "TC", (5, 2)),
        ("sawtooth", "CT", (2, 5)),
        ("random-walk", "NTC", (1, 5, 2)),
    ],
)
def test_additional_time_series_modes(
    tmp_path: Path,
    waveform: str,
    layout: str,
    shape: tuple[int, ...],
) -> None:
    generated = generate_time_series_tensor(
        tmp_path / f"{waveform}.npy",
        samples=5,
        channels=2,
        waveform=cast(Any, waveform),
        layout=cast(Any, layout),
        seed=9,
    )
    assert generated.shape == shape


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": 0},
        {"samples": 2, "sample_rate": 0},
        {"samples": 2, "frequency": -1},
        {"samples": 2, "amplitude": float("nan")},
        {"samples": 2, "dtype": "int64"},
        {"samples": 2, "waveform": "unknown"},
        {"samples": 2, "layout": "unknown"},
    ],
)
def test_time_series_rejects_invalid_configurations(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(InputGenerationError):
        generate_time_series_tensor(
            tmp_path / "bad.npy",
            **cast(Any, overrides),
        )


@pytest.mark.parametrize("distribution", ["zeros", "ones", "uniform", "normal"])
def test_synthetic_distribution_modes(
    tmp_path: Path,
    distribution: str,
) -> None:
    generated = generate_synthetic_tensor(
        tmp_path / distribution,
        shape=(2, 3),
        distribution=cast(Any, distribution),
        seed=4,
    )
    assert generated.path.suffix == ".npy"
    assert generated.shape == (2, 3)

    if distribution == "uniform":
        array = np.load(generated.path, allow_pickle=False)
        assert np.all((array >= 0) & (array <= 1))


def test_generation_rejects_bad_dtype_distribution_and_destination(
    tmp_path: Path,
) -> None:
    with pytest.raises(InputGenerationError, match="unsupported dtype"):
        generate_synthetic_tensor(
            tmp_path / "bad.npy",
            shape=(2,),
            dtype="object",
        )
    with pytest.raises(InputGenerationError, match="unsupported synthetic"):
        generate_synthetic_tensor(
            tmp_path / "bad.npy",
            shape=(2,),
            distribution=cast(Any, "triangle"),
        )
    with pytest.raises(InputGenerationError, match="destination directory"):
        generate_synthetic_tensor(
            tmp_path / "missing" / "bad.npy",
            shape=(2,),
        )


def test_generator_refuses_unbounded_allocations(tmp_path: Path) -> None:
    with pytest.raises(InputGenerationError, match="generator limit"):
        generate_synthetic_tensor(
            tmp_path / "too-large.npy",
            shape=(1024, 1024, 1024),
            dtype="float64",
        )
    assert not (tmp_path / "too-large.npy").exists()
