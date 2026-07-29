"""Headless Phase 9 tracing contracts, stores, views, and comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from nneditor.application.slices import GraphSlice
from nneditor.input_generation import generate_image_tensor
from nneditor.rendering.scene import NodeGlyph, Scene
from nneditor.storage.store import TensorUnavailableError
from nneditor.tracing import (
    ActivationRecord,
    ActivationStore,
    CaptureState,
    InputBinding,
    InputSource,
    InputSpecification,
    InputSpecificationStore,
    PlotKind,
    TraceApproval,
    TraceKey,
    TraceLimits,
    TraceResult,
    bind_inputs,
    build_activation_visualizations,
    compare_traces,
    comparison_scene_patch,
    default_input_specification,
    tensor_file_binding,
    trace_scene_patch,
)
from nneditor.tracing.contracts import resolved_file
from tests.fixtures.onnx_models import (
    build_embedded_model,
    build_masked_image_model,
    build_symbolic_shape_model,
)


def _record(
    value_id: str,
    payload: bytes,
    *,
    node_id: str = "node",
    state: CaptureState = CaptureState.COMPLETE,
    shape: tuple[int, ...] = (2,),
) -> ActivationRecord:
    return ActivationRecord(
        value_id=value_id,
        value_name=value_id,
        node_id=node_id,
        role="node-output",
        element_type="float32",
        numpy_dtype="float32",
        shape=shape,
        state=state,
        full_byte_length=(
            len(payload) * 2 if state is CaptureState.TRUNCATED else len(payload)
        ),
        stored_byte_length=len(payload),
        file_name=f"captures/{value_id}.bin",
        reason=("capture was truncated" if state is CaptureState.TRUNCATED else None),
    )


def _commit(
    store: ActivationStore,
    key: TraceKey,
    values: dict[str, np.ndarray],
    *,
    state: CaptureState = CaptureState.COMPLETE,
) -> TraceResult:
    staging = store.begin_staging()
    (staging / "captures").mkdir()
    records = []
    for value_id, array in values.items():
        payload = np.asarray(array, dtype=np.float32).tobytes()
        (staging / "captures" / f"{value_id}.bin").write_bytes(payload)
        records.append(
            _record(
                value_id,
                payload,
                state=state,
                shape=tuple(int(item) for item in array.shape),
            )
        )
    return store.commit(key, tuple(records), "test-runtime", (), staging)


def test_seeded_inputs_are_deterministic_and_persist_by_hash(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=8)
    with ApplicationService() as service:
        session = service.open_model(model)
        first = default_input_specification(session.document, seed=17)
        second = default_input_specification(session.document, seed=17)
    assert first == second
    assert first.hash == second.hash

    store = InputSpecificationStore(tmp_path / "state")
    store.save(first)
    store.save(second)
    reloaded = InputSpecificationStore(tmp_path / "state")
    assert reloaded.specifications(first.artifact_hash) == (first,)
    assert not (tmp_path / "state" / "trace-inputs.json.partial").exists()


def test_symbolic_inputs_require_explicit_extents(tmp_path: Path) -> None:
    model = tmp_path / "symbolic.onnx"
    build_symbolic_shape_model(model)
    with ApplicationService() as service:
        session = service.open_model(model)
        with pytest.raises(ValueError, match="symbolic dimensions"):
            session.default_trace_inputs()
        specification = session.default_trace_inputs(shapes={"input": (2, 3, 16, 224)})
    assert specification.bindings[0].shape == (2, 3, 16, 224)


def test_tensor_file_binding_hashes_safe_npy_payload(tmp_path: Path) -> None:
    path = tmp_path / "input.npy"
    np.save(path, np.arange(4, dtype=np.float32))
    binding = tensor_file_binding("input", "float32", (4,), path)
    assert binding.source is InputSource.TENSOR_FILE
    assert binding.file_hash is not None
    assert binding.tensor_file == str(path.resolve())


def test_session_tensor_picker_binding_overrides_only_named_input(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    path = tmp_path / "input.npy"
    np.save(path, np.arange(4, dtype=np.float32))
    with ApplicationService() as service:
        session = service.open_model(model)
        binding = session.trace_tensor_input("input", path)
        specification = session.default_trace_inputs(
            seed=19,
            bindings={"input": binding},
        )
    assert specification.bindings == (binding,)
    assert binding.source is InputSource.TENSOR_FILE


def test_required_mask_defaults_to_all_valid_and_infers_shared_batch(
    tmp_path: Path,
) -> None:
    model = tmp_path / "masked-image.onnx"
    image = tmp_path / "image.npy"
    build_masked_image_model(model)
    np.save(image, np.zeros((1, 3, 224, 224), dtype=np.float32))

    with ApplicationService() as service:
        session = service.open_model(model)
        image_binding = session.trace_tensor_input("pixel_values", image)
        specification = session.default_trace_inputs(
            bindings={"pixel_values": image_binding}
        )
        feeds = bind_inputs(session.document, specification)

    mask = next(
        binding for binding in specification.bindings if binding.name == "pixel_mask"
    )
    assert mask.source is InputSource.CONSTANT
    assert mask.element_type == "int64"
    assert mask.shape == (1, 64, 64)
    assert mask.constant == (1,)
    assert feeds["pixel_mask"].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(feeds["pixel_mask"], np.ones((1, 64, 64)))


def test_session_tensor_picker_rejects_wrong_dtype_and_shape(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    wrong_dtype = tmp_path / "wrong-dtype.npy"
    wrong_shape = tmp_path / "wrong-shape.npy"
    np.save(wrong_dtype, np.arange(4, dtype=np.int64))
    np.save(wrong_shape, np.arange(3, dtype=np.float32))
    with ApplicationService() as service:
        session = service.open_model(model)
        with pytest.raises(ValueError, match="dtype"):
            session.trace_tensor_input("input", wrong_dtype)
        with pytest.raises(ValueError, match="extent"):
            session.trace_tensor_input("input", wrong_shape)


def test_session_tensor_picker_rejects_unknown_unsafe_and_missing_files(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    text = tmp_path / "input.bin"
    text.write_bytes(b"not a tensor")
    with ApplicationService() as service:
        session = service.open_model(model)
        with pytest.raises(ValueError, match="not a model input"):
            session.trace_tensor_input("unknown", text)
        with pytest.raises(ValueError, match=r"safe \.npy"):
            session.trace_tensor_input("input", text)
        with pytest.raises(ValueError, match="not a readable file"):
            session.trace_tensor_input("input", tmp_path / "missing.npy")


def test_input_binding_overrides_reject_wrong_and_unknown_names(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    with ApplicationService() as service:
        session = service.open_model(model)
        wrong = InputBinding(
            "wrong",
            "float32",
            (4,),
            InputSource.SEEDED_RANDOM,
            seed=1,
        )
        with pytest.raises(ValueError, match="cannot replace"):
            session.default_trace_inputs(bindings={"input": wrong})
        with pytest.raises(ValueError, match="unknown model input"):
            session.default_trace_inputs(bindings={"unknown": wrong})


def test_file_and_constant_bindings_validate_dtype_shape_and_content(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    array_path = tmp_path / "input.npy"
    np.save(array_path, np.arange(4, dtype=np.float32))
    with ApplicationService() as service:
        document = service.open_model(model).document
        file_binding = tensor_file_binding("input", "float32", (4,), array_path)
        file_spec = InputSpecification(document.source.content_hash, (file_binding,))
        assert np.array_equal(
            bind_inputs(document, file_spec)["input"],
            np.arange(4, dtype=np.float32),
        )

        constant_spec = InputSpecification(
            document.source.content_hash,
            (
                InputBinding(
                    "input",
                    "float32",
                    (4,),
                    InputSource.CONSTANT,
                    constant=(2.0,),
                ),
            ),
        )
        assert np.array_equal(
            bind_inputs(document, constant_spec)["input"],
            np.full((4,), 2.0, dtype=np.float32),
        )

        np.save(array_path, np.ones(4, dtype=np.float32))
        with pytest.raises(ValueError, match="changed"):
            bind_inputs(document, file_spec)


def test_input_binding_errors_never_guess_or_coerce(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a readable file"):
        tensor_file_binding("input", "float32", (1,), tmp_path / "missing.npy")

    model = tmp_path / "model.onnx"
    build_embedded_model(model, elements=4)
    with ApplicationService() as service:
        session = service.open_model(model)
        document = session.document
        with pytest.raises(ValueError, match="does not match declared"):
            session.default_trace_inputs(shapes={"input": (5,)})

        valid = session.default_trace_inputs()
        wrong_artifact = InputSpecification("another", valid.bindings)
        with pytest.raises(ValueError, match="different artifact"):
            bind_inputs(document, wrong_artifact)

        wrong_name = InputSpecification(
            document.source.content_hash,
            (
                InputBinding(
                    "other", "float32", (4,), InputSource.SEEDED_RANDOM, seed=0
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing="):
            bind_inputs(document, wrong_name)

        wrong_dtype = InputSpecification(
            document.source.content_hash,
            (
                InputBinding(
                    "input", "float64", (4,), InputSource.SEEDED_RANDOM, seed=0
                ),
            ),
        )
        with pytest.raises(ValueError, match="dtype"):
            bind_inputs(document, wrong_dtype)

        unsafe_path = tmp_path / "input.bin"
        unsafe_path.write_bytes(np.arange(4, dtype=np.float32).tobytes())
        unsafe = InputSpecification(
            document.source.content_hash,
            (tensor_file_binding("input", "float32", (4,), unsafe_path),),
        )
        with pytest.raises(ValueError, match=r"safe \.npy"):
            bind_inputs(document, unsafe)


def test_input_binding_rejects_nonpositive_shapes_and_ambiguous_sources() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        InputBinding("x", "float32", (0,), InputSource.SEEDED_RANDOM, seed=0)
    with pytest.raises(ValueError, match="cannot also carry"):
        InputBinding(
            "x",
            "float32",
            (1,),
            InputSource.SEEDED_RANDOM,
            seed=0,
            constant=(1.0,),
        )


def test_activation_store_range_statistics_and_budget_eviction(
    tmp_path: Path,
) -> None:
    store = ActivationStore(tmp_path / "captures", byte_budget=12)
    key1 = TraceKey("artifact", None, "inputs")
    first = _commit(store, key1, {"a": np.array([1.0, 3.0], dtype=np.float32)})
    view = store.open(first.id)
    assert view.read("a", offset=4, length=4) == np.float32(3).tobytes()
    assert view.statistics("a").mean == pytest.approx(2.0)

    key2 = TraceKey("artifact", "revision", "inputs")
    second = _commit(store, key2, {"b": np.array([2.0, 4.0], dtype=np.float32)})
    assert second.id != first.id
    evicted = store.result(first.id)
    assert evicted.record("a").state is CaptureState.EVICTED
    assert evicted.record("a").readable is False
    assert store.current_bytes == 8


def test_narrowed_rerun_preserves_other_values_under_the_value_key(
    tmp_path: Path,
) -> None:
    store = ActivationStore(tmp_path / "captures")
    key = TraceKey("artifact", None, "inputs")
    first = _commit(
        store,
        key,
        {
            "a": np.array([1.0], dtype=np.float32),
            "b": np.array([2.0], dtype=np.float32),
        },
    )
    second = _commit(
        store,
        key,
        {"a": np.array([3.0], dtype=np.float32)},
    )
    assert first.id == second.id
    assert {record.value_id for record in second.records} == {"a", "b"}
    assert store.open(second.id).read("b") == np.float32(2.0).tobytes()


def test_visualization_builders_preserve_exact_matrix_shape() -> None:
    matrix = np.arange(36, dtype=np.float32).reshape(6, 6)
    record = _record("matrix", matrix.tobytes(), shape=matrix.shape)
    first = build_activation_visualizations(record, matrix.tobytes())
    second = build_activation_visualizations(record, matrix.tobytes())
    assert first == second
    assert {view.kind for view in first} == {
        PlotKind.HISTOGRAM,
        PlotKind.HEATMAP,
        PlotKind.TENSOR_LAYER_STACK,
    }
    heatmap = next(view for view in first if view.kind is PlotKind.HEATMAP)
    assert heatmap.shape == (6, 6)
    assert "exact 6x6" in heatmap.downsampling
    assert heatmap.colormap.startswith("viridis")
    assert "min-max" in heatmap.normalization
    assert heatmap.raster_size == (6, 6)
    assert heatmap.raster_png.startswith(b"\x89PNG")


def test_feature_and_attention_view_selection() -> None:
    tensor = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    record = _record("feature", tensor.tobytes(), shape=tensor.shape)
    regular = build_activation_visualizations(record, tensor.tobytes())
    attention = build_activation_visualizations(
        record, tensor.tobytes(), attention=True
    )
    assert PlotKind.FEATURE_MAP_GRID in {view.kind for view in regular}
    assert PlotKind.ATTENTION_MAP in {view.kind for view in attention}
    feature = next(view for view in regular if view.kind is PlotKind.FEATURE_MAP_GRID)
    assert feature.shape == (3, 4, 4)
    assert feature.raster_png.startswith(b"\x89PNG")
    assert "exact 4x4" in feature.downsampling


def test_feature_maps_preserve_non_square_spatial_dimensions() -> None:
    tensor = np.arange(2 * 18 * 47, dtype=np.float32).reshape(1, 2, 18, 47)
    record = _record("feature", tensor.tobytes(), shape=tensor.shape)

    feature = next(
        view
        for view in build_activation_visualizations(record, tensor.tobytes())
        if view.kind is PlotKind.FEATURE_MAP_GRID
    )

    assert feature.shape == (2, 18, 47)
    assert feature.raster_size == (95, 18)
    assert "exact 18x47" in feature.downsampling


def test_qwen_pixel_patches_reconstruct_exact_rgb_image_and_channel_planes(
    tmp_path: Path,
) -> None:
    pixels = np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3)
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    generated = generate_image_tensor(
        source,
        tmp_path / "qwen.npy",
        width=64,
        height=32,
        layout="QWEN3VL_PATCHES",
        normalization="minus-one-one",
        dtype="float32",
    )
    patches = np.load(generated.path, allow_pickle=False)
    record = _record("pixel_values", patches.tobytes(), shape=patches.shape)

    views = build_activation_visualizations(record, patches.tobytes())
    image = next(view for view in views if view.kind is PlotKind.RGB_IMAGE)
    channels = next(view for view in views if view.kind is PlotKind.IMAGE_CHANNEL_STACK)
    reconstructed = np.asarray(Image.open(BytesIO(image.raster_png)).convert("RGB"))

    assert image.shape == (32, 64, 3)
    assert image.raster_size == (64, 32)
    np.testing.assert_array_equal(reconstructed, pixels)
    assert channels.shape == (3, 32, 64)
    assert channels.layer_shape == (32, 64)
    assert len(channels.layer_pngs) == 3
    assert channels.layer_labels[0].startswith("Red")
    assert "three planes offset for depth" in channels.downsampling


def test_ordinary_rgb_tensor_gets_image_and_layered_channel_views() -> None:
    tensor = np.arange(3 * 18 * 47, dtype=np.float32).reshape(1, 3, 18, 47)
    record = _record("pixel_values", tensor.tobytes(), shape=tensor.shape)

    views = build_activation_visualizations(record, tensor.tobytes())

    assert next(view for view in views if view.kind is PlotKind.RGB_IMAGE).shape == (
        18,
        47,
        3,
    )
    assert next(
        view for view in views if view.kind is PlotKind.IMAGE_CHANNEL_STACK
    ).shape == (3, 18, 47)


@pytest.mark.parametrize(
    ("shape", "layer_count", "last_label"),
    [
        ((5, 7), 1, "[:, :]"),
        ((3, 5, 7), 3, "[2, :, :]"),
        ((2, 3, 5, 7), 6, "[1, 2, :, :]"),
    ],
)
def test_tensor_layer_stack_maps_leading_axes_to_exact_2d_layers(
    shape: tuple[int, ...],
    layer_count: int,
    last_label: str,
) -> None:
    tensor = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    record = _record("activation", tensor.tobytes(), shape=tensor.shape)

    views = build_activation_visualizations(record, tensor.tobytes())
    stack = next(view for view in views if view.kind is PlotKind.TENSOR_LAYER_STACK)

    assert views[0] is stack
    assert stack.shape == shape
    assert stack.layer_shape == shape[-2:]
    assert len(stack.layer_pngs) == layer_count
    assert stack.layer_labels[-1] == last_label
    assert stack.layer_indices == tuple(range(layer_count))


def test_comparison_metrics_and_scene_overlays(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path / "captures")
    left = _commit(
        store,
        TraceKey("artifact", None, "same-input"),
        {"value": np.array([1.0, 2.0], dtype=np.float32)},
    )
    right = _commit(
        store,
        TraceKey("artifact", "edited", "same-input"),
        {"value": np.array([1.0, 4.0], dtype=np.float32)},
    )
    comparison = compare_traces(store, left, right)
    metric = comparison.values[0]
    assert metric.max_absolute == pytest.approx(2.0)
    assert metric.max_relative == pytest.approx(1.0)
    assert metric.cosine_similarity == pytest.approx(9 / np.sqrt(85))

    scene = Scene([NodeGlyph("glyph", 0, 0, 10, 10, "op", "Node")])
    graph_slice = GraphSlice(
        scene,
        (("glyph",),),
        (),
        DetailLevel.OPERATOR,
        "hierarchy",
        None,
        {"glyph": frozenset({"node"})},
    )
    traced_scene = scene.apply(trace_scene_patch(graph_slice, left))
    assert "activation captured" in traced_scene.node("glyph").type_label
    compared_scene = scene.apply(comparison_scene_patch(graph_slice, comparison))
    assert "Δmax 2" in compared_scene.node("glyph").type_label


def test_comparison_refuses_different_input_specifications(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path / "captures")
    left = _commit(
        store,
        TraceKey("artifact", None, "input-a"),
        {"value": np.array([1.0], dtype=np.float32)},
    )
    right = _commit(
        store,
        TraceKey("artifact", "edited", "input-b"),
        {"value": np.array([1.0], dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="same input specification"):
        compare_traces(store, left, right)


def test_comparison_discloses_nonfinite_mismatches(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path / "captures")
    left = _commit(
        store,
        TraceKey("artifact", None, "inputs"),
        {"value": np.array([1.0, np.nan, np.inf], dtype=np.float32)},
    )
    right = _commit(
        store,
        TraceKey("artifact", "edited", "inputs"),
        {"value": np.array([2.0, np.nan, -np.inf], dtype=np.float32)},
    )
    comparison = compare_traces(store, left, right)
    assert comparison.values[0].max_absolute == np.inf
    assert comparison.values[0].partial
    assert any("non-finite" in diagnostic for diagnostic in comparison.diagnostics)


def test_constant_specification_has_content_identity() -> None:
    first = InputSpecification(
        "artifact",
        (
            InputBinding(
                "x",
                "float32",
                (2,),
                InputSource.CONSTANT,
                constant=(1.0, 2.0),
            ),
        ),
    )
    second = InputSpecification(
        "artifact",
        (
            InputBinding(
                "x",
                "float32",
                (2,),
                InputSource.CONSTANT,
                constant=(1.0, 3.0),
            ),
        ),
    )
    assert first.hash != second.hash


def test_trace_contract_json_round_trips_and_consent_identity() -> None:
    limits = TraceLimits(4.5, 1024, 512, 64)
    assert TraceLimits.from_json(limits.to_json()) == limits
    with pytest.raises(ValueError, match="malformed"):
        TraceLimits.from_json([])
    for changes in (
        {"wall_seconds": 0.0},
        {"wall_seconds": float("nan")},
        {"memory_bytes": 0},
        {"capture_bytes": 0},
        {"chunk_bytes": 513},
    ):
        values = limits.to_json()
        values.update(changes)
        with pytest.raises(ValueError):
            TraceLimits.from_json(values)

    binding = InputBinding(
        "x", "float32", (2,), InputSource.CONSTANT, constant=(1.0, 2.0)
    )
    assert InputBinding.from_json(binding.to_json()) == binding
    specification = InputSpecification("artifact", (binding,))
    assert InputSpecification.from_json(specification.to_json()) == specification
    approval = TraceApproval.approve("model", "artifact", specification, limits)
    approval.validate(
        model_title="model",
        artifact_hash="artifact",
        specification=specification,
        limits=limits,
    )
    with pytest.raises(ValueError, match="does not name"):
        approval.validate(
            model_title="another",
            artifact_hash="artifact",
            specification=specification,
            limits=limits,
        )

    record = _record("value", np.zeros(2, dtype=np.float32).tobytes())
    assert ActivationRecord.from_json(record.to_json()) == record
    result = TraceResult(TraceKey("artifact", None, specification.hash), (record,), "x")
    assert TraceResult.from_json(result.to_json()) == result
    with pytest.raises(KeyError):
        result.record("missing")


def test_trace_contracts_reject_malformed_and_incomplete_values() -> None:
    with pytest.raises(ValueError, match="malformed"):
        InputBinding.from_json([])
    with pytest.raises(ValueError, match="shape/constant"):
        InputBinding.from_json({"shape": "bad", "constant": []})
    with pytest.raises(ValueError, match="numeric"):
        InputBinding.from_json(
            {
                "name": "x",
                "element_type": "float32",
                "shape": [1],
                "source": "constant",
                "constant": ["bad"],
            }
        )
    with pytest.raises(ValueError, match="unique"):
        InputSpecification("artifact", ())
    with pytest.raises(ValueError, match="malformed"):
        InputSpecification.from_json(None)
    with pytest.raises(ValueError, match="malformed"):
        ActivationRecord.from_json({})
    with pytest.raises(ValueError, match="malformed"):
        TraceResult.from_json({})
    binding = InputBinding("x", "float32", (1,), InputSource.SEEDED_RANDOM, seed=0)
    with pytest.raises(ValueError, match="no tensor file"):
        resolved_file(binding)

    with pytest.raises(ValueError, match="trace key"):
        TraceKey("", None, "inputs")
    valid_record = _record("value", np.zeros(2, dtype=np.float32).tobytes())
    invalid_factories: tuple[Callable[[], ActivationRecord], ...] = (
        lambda: replace(valid_record, value_name=""),
        lambda: replace(valid_record, shape=(-2,)),
        lambda: replace(valid_record, full_byte_length=-1),
        lambda: replace(valid_record, stored_byte_length=9),
        lambda: replace(valid_record, stored_byte_length=4),
        lambda: replace(valid_record, state=CaptureState.TRUNCATED),
        lambda: replace(valid_record, state=CaptureState.DROPPED),
    )
    for factory in invalid_factories:
        with pytest.raises(ValueError):
            factory()
    with pytest.raises(ValueError, match="duplicate"):
        TraceResult(
            TraceKey("artifact", None, "inputs"),
            (valid_record, valid_record),
            "runtime",
        )


def test_activation_store_rejects_invalid_paths_ranges_and_payloads(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ActivationStore(tmp_path / "bad", byte_budget=0)
    store = ActivationStore(tmp_path / "captures")
    with pytest.raises(ValueError, match="identity"):
        store._trace_directory("not-a-trace")
    with pytest.raises(ValueError, match="non-staging"):
        store.discard_staging(tmp_path)

    staging = store.begin_staging()
    missing = _record("missing", b"\x00" * 8)
    with pytest.raises(ValueError, match="missing"):
        store.commit(
            TraceKey("artifact", None, "inputs"),
            (missing,),
            "x",
            (),
            staging,
        )
    store.discard_staging(staging)

    result = _commit(
        store,
        TraceKey("artifact", None, "inputs"),
        {"value": np.array([1.0, 2.0], dtype=np.float32)},
    )
    view = store.open(result.id)
    assert view.read("value") == view.read("value")
    assert store.cache_snapshot().hits >= 1
    with pytest.raises(TensorUnavailableError, match="outside"):
        view.read("value", offset=-1, length=1)
    with pytest.raises(KeyError):
        store.result("trace:" + "0" * 64)

    reloaded = ActivationStore(store.root)
    assert reloaded.result(result.id) == result
    assert reloaded.results("another-artifact") == ()
