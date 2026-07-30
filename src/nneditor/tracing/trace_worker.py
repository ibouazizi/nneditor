"""Minimal subprocess entry point for ONNX intermediate-value capture."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from nneditor.tracing.contracts import TraceBackend, TraceDevice
from nneditor.tracing.scan import AxisAwareScan, normalize_scan_axes

EvaluatorFactory = Callable[[onnx.ModelProto], Any]


@dataclass(frozen=True, slots=True)
class OrtProviderCandidate:
    """One strict ONNX Runtime provider configuration to try."""

    device: TraceDevice
    label: str
    providers: tuple[str, ...]
    provider_options: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        if (
            not self.providers
            or len(self.providers) != len(self.provider_options)
            or self.device is TraceDevice.AUTO
        ):
            raise ValueError("invalid ONNX Runtime provider candidate")

    @property
    def provider(self) -> str:
        return ", ".join(self.providers)


def _provider_candidates(
    available: list[str] | tuple[str, ...],
    requested: TraceDevice,
) -> tuple[OrtProviderCandidate, ...]:
    """Return installed provider configurations in deterministic priority order."""
    installed = set(available)
    candidates: list[OrtProviderCandidate] = []

    def add(
        device: TraceDevice,
        label: str,
        providers: tuple[str, ...],
        options: tuple[dict[str, str], ...] | None = None,
    ) -> None:
        if all(provider in installed for provider in providers):
            candidates.append(
                OrtProviderCandidate(
                    device,
                    label,
                    providers,
                    options or tuple({} for _ in providers),
                )
            )

    if requested in {TraceDevice.AUTO, TraceDevice.GPU}:
        add(
            TraceDevice.GPU,
            "TensorRT / CUDA",
            ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        )
        add(TraceDevice.GPU, "CUDA", ("CUDAExecutionProvider",))
    if requested in {TraceDevice.AUTO, TraceDevice.NPU}:
        add(
            TraceDevice.NPU,
            "OpenVINO NPU",
            ("OpenVINOExecutionProvider",),
            ({"device_type": "NPU"},),
        )
        add(
            TraceDevice.NPU,
            "Qualcomm QNN HTP",
            ("QNNExecutionProvider",),
            ({"backend_type": "htp"},),
        )
        add(TraceDevice.NPU, "AMD Vitis AI", ("VitisAIExecutionProvider",))
    if requested in {TraceDevice.AUTO, TraceDevice.GPU}:
        add(TraceDevice.GPU, "DirectML", ("DmlExecutionProvider",))
        add(
            TraceDevice.GPU,
            "OpenVINO GPU",
            ("OpenVINOExecutionProvider",),
            ({"device_type": "GPU"},),
        )
        add(TraceDevice.GPU, "AMD MIGraphX", ("MIGraphXExecutionProvider",))
        add(TraceDevice.GPU, "AMD ROCm", ("ROCMExecutionProvider",))
    if requested in {TraceDevice.AUTO, TraceDevice.CPU}:
        add(TraceDevice.CPU, "CPU", ("CPUExecutionProvider",))
    return tuple(candidates)


# The published package deliberately has no hard ONNX Runtime dependency:
# every accelerator family ships its own distribution and all of them own the
# same `onnxruntime` import name, so pinning one would block the others.
_RUNTIME_INSTALL_HINTS: dict[TraceDevice, str] = {
    TraceDevice.AUTO: "nneditor[runtime]",
    TraceDevice.CPU: "nneditor[runtime]",
    TraceDevice.GPU: (
        "nneditor[runtime-gpu] for CUDA or nneditor[runtime-directml] "
        "for any Windows GPU"
    ),
    TraceDevice.NPU: (
        "nneditor[runtime-qnn] for Qualcomm, nneditor[runtime-openvino] "
        "for Intel, or a vendor ONNX Runtime build"
    ),
}


def _missing_runtime_message(device: TraceDevice) -> str:
    """Name the exact extra that enables the requested device."""
    return (
        "ONNX Runtime is not installed in this environment; install "
        f"{_RUNTIME_INSTALL_HINTS[device]} to enable it"
    )


def _axis_reference_evaluator(model: onnx.ModelProto) -> Any:
    return ReferenceEvaluator(model, new_ops=[AxisAwareScan])


def _standard_reference_evaluator(model: onnx.ModelProto) -> Any:
    return ReferenceEvaluator(model)


def _limit_process(raw: dict[str, object]) -> None:
    if os.name == "nt":
        return
    try:
        import resource as resource_module
    except ImportError as error:
        raise RuntimeError(
            "this platform cannot enforce trace worker resource limits"
        ) from error
    resource = cast(Any, resource_module)
    try:
        cpu = max(1, int(float(str(raw["wall_seconds"]))))
        memory = int(str(raw["memory_bytes"]))

        def cap(resource_id: int, requested: int) -> None:
            _soft, hard = resource.getrlimit(resource_id)
            resolved = (
                requested
                if hard == resource.RLIM_INFINITY
                else min(requested, int(hard))
            )
            resource.setrlimit(resource_id, (resolved, resolved))

        cap(resource.RLIMIT_CPU, cpu)
        cap(resource.RLIMIT_AS, memory)
        cap(resource.RLIMIT_NOFILE, 64)
    except (AttributeError, OSError, ValueError) as error:
        raise RuntimeError(
            "trace worker resource limits could not be enforced"
        ) from error


def _evaluate_prefix(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    name: str,
    *,
    producer_limit: int,
    evaluator_factory: EvaluatorFactory | None = None,
) -> np.ndarray[Any, Any]:
    """Evaluate one value without copying the model's weights.

    The fallback removes only nodes after the requested value while the
    evaluator is built and run. Protobuf repeated fields keep the detached
    node messages alive, so they can be restored without duplicating the
    model's potentially multi-gigabyte initializer payload.
    """
    trailing_nodes = list(model.graph.node[producer_limit + 1 :])
    del model.graph.node[producer_limit + 1 :]
    try:
        values = cast(
            list[np.ndarray[Any, Any]],
            (evaluator_factory or _axis_reference_evaluator)(model).run([name], feeds),
        )
        return np.asarray(values[0])
    finally:
        model.graph.node.extend(trailing_nodes)


def _prefix_builds(
    model: onnx.ModelProto,
    all_nodes: list[onnx.NodeProto],
    limit: int,
    evaluator_factory: EvaluatorFactory | None = None,
) -> bool:
    """Whether the runtime can build an evaluator over ``nodes[:limit + 1]``.

    The graph is restored before returning. Detaching nodes keeps the model's
    initializers untouched, so probing never copies the weight payload.
    """
    model.graph.ClearField("node")
    model.graph.node.extend(all_nodes[: limit + 1])
    try:
        (evaluator_factory or _axis_reference_evaluator)(model)
        return True
    except Exception:
        # Any build failure answers the question this probe asks.
        return False
    finally:
        model.graph.ClearField("node")
        model.graph.node.extend(all_nodes)


def _buildable_frontier(
    model: onnx.ModelProto,
    evaluator_factory: EvaluatorFactory | None = None,
) -> int:
    """Index of the first node the runtime cannot build an evaluator over.

    Buildability is prefix-closed: if the runtime refuses node *j*, it refuses
    every prefix containing *j*. That makes a binary search valid and costs
    about log2(n) evaluator builds instead of one per captured value — ten
    probes on a 587-node model instead of hundreds of futile attempts, each of
    which would otherwise rebuild the whole evaluator before failing.

    Returns ``len(nodes)`` when every prefix builds.
    """
    all_nodes = list(model.graph.node)
    low, high = 0, len(all_nodes)
    while low < high:
        middle = (low + high) // 2
        if _prefix_builds(model, all_nodes, middle, evaluator_factory):
            low = middle + 1
        else:
            high = middle
    return low


class CaptureSource:
    """Yields one captured value at a time so nothing accumulates.

    Holding every capture until the run finishes makes peak memory grow with
    the number of values requested, which is what exhausts the worker on a
    graph with hundreds of them. Callers take a value, write it, and drop it,
    so only one activation is live at a time beyond whatever the runtime
    itself retains.
    """

    __slots__ = (
        "_evaluator_factory",
        "_exhausted",
        "_feeds",
        "_frontier",
        "_frontier_reason",
        "_model",
        "_producer_by_output",
        "_whole",
        "diagnostics",
    )

    def __init__(
        self,
        model: onnx.ModelProto,
        feeds: dict[str, np.ndarray[Any, Any]],
        whole: dict[str, np.ndarray[Any, Any]] | None,
        diagnostics: list[str],
        frontier: int,
        frontier_reason: str | None,
        evaluator_factory: EvaluatorFactory | None = None,
    ) -> None:
        self._model = model
        self._feeds = feeds
        self._evaluator_factory = evaluator_factory
        self._whole = whole
        self._frontier = frontier
        self._frontier_reason = frontier_reason
        self._exhausted: str | None = None
        self.diagnostics = diagnostics
        self._producer_by_output = {
            output: index
            for index, node in enumerate(model.graph.node)
            for output in node.output
            if output
        }

    def take(
        self, target: dict[str, object]
    ) -> tuple[np.ndarray[Any, Any] | None, str | None]:
        """Return this target's value, or the reason it has none."""
        name = str(target["value_name"])
        if name in self._feeds:
            return self._feeds[name], None
        if self._whole is not None:
            # Pop rather than read: the caller writes it next, and dropping the
            # reference here is what keeps the whole-model path from holding
            # every activation at once.
            captured = self._whole.pop(str(target["value_id"]), None)
            if captured is None:
                return None, "the reference runtime produced no value"
            return captured, None
        if self._exhausted is not None:
            return None, self._exhausted
        producer = self._producer_by_output.get(name)
        if producer is None:
            self.diagnostics.append(f"value {name!r} has no executable producer")
            return None, "this value has no executable producer"
        if producer >= self._frontier:
            # Its prefix necessarily contains the node the runtime refuses, so
            # attempting it would rebuild the evaluator only to fail again.
            return None, self._frontier_reason
        try:
            return (
                (
                    _evaluate_prefix(
                        self._model,
                        self._feeds,
                        name,
                        producer_limit=producer,
                    )
                    if self._evaluator_factory is None
                    else _evaluate_prefix(
                        self._model,
                        self._feeds,
                        name,
                        producer_limit=producer,
                        evaluator_factory=self._evaluator_factory,
                    )
                ),
                None,
            )
        except MemoryError as error:
            self._exhausted = (
                "per-value capture exhausted the approved memory limit; "
                "raise the Memory limit, or capture fewer values"
            )
            self.diagnostics.append(
                f"per-value capture ran out of memory: {error}; the remaining "
                "values were not attempted. Raise the Memory limit or select "
                "fewer values to capture."
            )
            return None, self._exhausted
        except Exception as error:
            reason = f"not captured: {type(error).__name__}: {error}"
            self.diagnostics.append(f"value {name!r} was {reason}")
            return None, reason

    def size_hint(self, target: dict[str, object]) -> int | None:
        """Return an already-materialized capture size without producing it."""
        name = str(target["value_name"])
        if name in self._feeds:
            return int(self._feeds[name].nbytes)
        if self._whole is None:
            return None
        captured = self._whole.get(str(target["value_id"]))
        return None if captured is None else int(captured.nbytes)


def open_captures(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
    *,
    evaluator_factory: EvaluatorFactory | None = None,
) -> CaptureSource:
    """Prepare capture, choosing whole-model evaluation when it is possible."""
    diagnostics: list[str] = []
    input_names = set(feeds)
    runtime_targets = [
        target for target in targets if str(target["value_name"]) not in input_names
    ]
    if not runtime_targets:
        return CaptureSource(
            model,
            feeds,
            {},
            diagnostics,
            len(model.graph.node),
            None,
            evaluator_factory,
        )

    names = [str(target["value_name"]) for target in runtime_targets]
    try:
        evaluator = (evaluator_factory or _axis_reference_evaluator)(model)
    except MemoryError as error:
        diagnostics.append(
            "building the reference runtime exhausted the approved memory "
            f"limit: {error}; increase the Memory limit and approve the trace "
            "again"
        )
        return CaptureSource(
            model,
            feeds,
            {},
            diagnostics,
            0,
            "the runtime ran out of memory",
            evaluator_factory,
        )
    except Exception as error:
        diagnostics.append(
            "the reference runtime cannot execute the whole model: "
            f"{type(error).__name__}: {error}; capturing per value instead"
        )
        return _prefix_source(model, feeds, diagnostics, evaluator_factory)
    try:
        values = evaluator.run(names, feeds)
    except MemoryError as error:
        diagnostics.append(
            "full trace exhausted the approved memory limit during evaluation: "
            f"{error}; increase the Memory limit and approve the trace again"
        )
        return CaptureSource(
            model,
            feeds,
            {},
            diagnostics,
            0,
            "the runtime ran out of memory",
            evaluator_factory,
        )
    except Exception as error:
        diagnostics.append(
            "full trace degraded to per-value prefix evaluation: "
            f"{type(error).__name__}: {error}"
        )
        return _prefix_source(model, feeds, diagnostics, evaluator_factory)
    whole = {
        str(target["value_id"]): np.asarray(value)
        for target, value in zip(runtime_targets, values, strict=True)
    }
    return CaptureSource(
        model,
        feeds,
        whole,
        diagnostics,
        len(model.graph.node),
        None,
        evaluator_factory,
    )


def _prefix_source(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    diagnostics: list[str],
    evaluator_factory: EvaluatorFactory | None = None,
) -> CaptureSource:
    """Set up per-value capture, first locating what the runtime cannot build.

    Finding the frontier costs about log2(n) evaluator builds and spares every
    value beyond it a rebuild that is guaranteed to fail — the difference
    between ten probes and hundreds of futile evaluations on a large graph.
    """
    frontier = _buildable_frontier(model, evaluator_factory)
    reason: str | None = None
    if frontier < len(model.graph.node):
        blocking = model.graph.node[frontier]
        label = blocking.name or f"index {frontier}"
        reason = (
            f"unavailable: the reference runtime cannot build {blocking.op_type} "
            f"({label}), which this value depends on"
        )
        diagnostics.append(
            f"the reference runtime cannot build {blocking.op_type} at node "
            f"index {frontier} of {len(model.graph.node)}; values produced at "
            "or after it were reported unavailable without being attempted"
        )
    return CaptureSource(
        model,
        feeds,
        None,
        diagnostics,
        frontier,
        reason,
        evaluator_factory,
    )


def _stage_capture_model(model_path: Path, output_names: list[str]) -> Path:
    """Write an output-augmented copy of the staged model for ONNX Runtime.

    The runtime reads the model from this file, so the fully loaded in-memory
    proto is never re-serialized: ``SerializeToString`` would peak at twice
    the model's size and fails outright at the 2 GiB protobuf ceiling. The
    skeleton loaded here is a private copy, so moving its embedded tensors to
    external data cannot disturb the reference-evaluator fallback, and the
    file is written beside the model inside the worker's scratch directory,
    which keeps the model's existing external-data references valid.
    """
    skeleton = onnx.load_model(str(model_path), load_external_data=False)
    existing = {value.name for value in skeleton.graph.output}
    for name in output_names:
        if name in existing:
            continue
        value_info = onnx.ValueInfoProto()
        value_info.name = name
        skeleton.graph.output.append(value_info)
        existing.add(name)
    capture_path = model_path.with_name(model_path.name + ".capture.onnx")
    data_name = capture_path.name + ".data"
    # onnx appends to an existing external-data file, so a stale one from an
    # earlier attempt must not survive into this staging.
    for stale in (capture_path, capture_path.with_name(data_name)):
        with contextlib.suppress(OSError):
            stale.unlink()
    onnx.save_model(
        skeleton,
        str(capture_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        convert_attribute=False,
    )
    return capture_path


def _onnxruntime_captures(
    model: onnx.ModelProto,
    capture_model: Path,
    feeds: dict[str, np.ndarray[Any, Any]],
    runtime_targets: list[dict[str, object]],
    candidate: OrtProviderCandidate,
) -> CaptureSource:
    """Capture requested values through a staged ONNX Runtime session."""
    import onnxruntime as ort  # type: ignore[import-untyped]

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    if candidate.device is not TraceDevice.CPU:
        # An explicit accelerator candidate must either execute without
        # the generic CPU EP or fail so Auto can try the next candidate.
        # This keeps the reported device honest.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(capture_model),
        sess_options=options,
        providers=list(candidate.providers),
        provider_options=list(candidate.provider_options),
    )
    names = [str(target["value_name"]) for target in runtime_targets]
    values = session.run(names, feeds)
    whole = {
        str(target["value_id"]): np.asarray(value)
        for target, value in zip(runtime_targets, values, strict=True)
    }
    return CaptureSource(model, feeds, whole, [], len(model.graph.node), None)


def _failed_runtime_source(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    message: str,
) -> CaptureSource:
    return CaptureSource(model, feeds, None, [message], 0, message)


def _open_onnxruntime(
    model: onnx.ModelProto,
    model_path: Path,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
    device: TraceDevice,
) -> tuple[CaptureSource, str, TraceDevice, str]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(_missing_runtime_message(device)) from error

    available = list(ort.get_available_providers())
    candidates = _provider_candidates(available, device)
    if not candidates:
        raise RuntimeError(
            f"no {device.value.upper()} execution provider is installed; "
            f"available providers: {', '.join(available) or 'none'}"
        )
    input_names = set(feeds)
    runtime_targets = [
        target for target in targets if str(target["value_name"]) not in input_names
    ]
    if not runtime_targets:
        source = CaptureSource(model, feeds, {}, [], len(model.graph.node), None)
        first = candidates[0]
        runtime = f"onnxruntime {ort.__version__} · {first.label}"
        return source, runtime, first.device, first.provider
    try:
        capture_model = _stage_capture_model(
            model_path,
            [str(target["value_name"]) for target in runtime_targets],
        )
    except Exception as error:
        raise RuntimeError(
            "the capture model could not be staged for ONNX Runtime: "
            f"{type(error).__name__}: {error}"
        ) from error
    failures: list[str] = []
    try:
        for candidate in candidates:
            try:
                source = _onnxruntime_captures(
                    model,
                    capture_model,
                    feeds,
                    runtime_targets,
                    candidate,
                )
            except Exception as error:
                failures.append(f"{candidate.label}: {type(error).__name__}: {error}")
                continue
            runtime = f"onnxruntime {ort.__version__} · {candidate.label}"
            return source, runtime, candidate.device, candidate.provider
    finally:
        # Captures are materialized eagerly, so the staged file is spent by
        # now. A runtime that still maps it blocks deletion on Windows;
        # scratch teardown reclaims whatever this best-effort pass cannot.
        data_path = capture_model.with_name(capture_model.name + ".data")
        for stale in (capture_model, data_path):
            with contextlib.suppress(OSError):
                stale.unlink()
    detail = "; ".join(failures)
    raise RuntimeError(
        f"no usable {device.value.upper()} execution provider accepted the model"
        + (f": {detail}" if detail else "")
    )


def _open_backend(
    model: onnx.ModelProto,
    model_path: Path,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
    backend: TraceBackend,
    device: TraceDevice = TraceDevice.AUTO,
) -> tuple[CaptureSource, str, TraceDevice, str]:
    if backend is TraceBackend.ONNX_RUNTIME:
        try:
            return _open_onnxruntime(model, model_path, feeds, targets, device)
        except Exception as error:
            message = (
                "ONNX Runtime could not execute the trace: "
                f"{type(error).__name__}: {error}"
            )
            failed_device = TraceDevice.CPU if device is TraceDevice.AUTO else device
            return (
                _failed_runtime_source(model, feeds, message),
                "onnxruntime failed",
                failed_device,
                "Unavailable",
            )
    if backend is TraceBackend.REFERENCE:
        return (
            open_captures(model, feeds, targets),
            f"onnx.reference axis-aware {onnx.__version__}",
            TraceDevice.CPU,
            "ReferenceEvaluator",
        )
    if backend is TraceBackend.REFERENCE_NORMALIZED:
        try:
            normalized, report = normalize_scan_axes(model, in_place=True)
        except Exception as error:
            message = f"Scan axis normalization failed: {type(error).__name__}: {error}"
            return (
                _failed_runtime_source(model, feeds, message),
                "onnx.reference normalized failed",
                TraceDevice.CPU,
                "ReferenceEvaluator",
            )
        source = open_captures(
            normalized,
            feeds,
            targets,
            evaluator_factory=_standard_reference_evaluator,
        )
        detail = (
            f"; {report.rewritten_scans} Scan / {report.inserted_transposes} Transpose"
            if report.rewritten_scans
            else ""
        )
        return (
            source,
            f"onnx.reference normalized {onnx.__version__}{detail}",
            TraceDevice.CPU,
            "ReferenceEvaluator",
        )

    try:
        return _open_onnxruntime(model, model_path, feeds, targets, device)
    except Exception as error:
        if device in {TraceDevice.GPU, TraceDevice.NPU}:
            message = (
                f"automatic backend selection could not use the requested "
                f"{device.value.upper()} device: {type(error).__name__}: {error}"
            )
            return (
                _failed_runtime_source(model, feeds, message),
                "onnxruntime accelerator unavailable",
                device,
                "Unavailable",
            )
        source = open_captures(model, feeds, targets)
        source.diagnostics.insert(
            0,
            "automatic backend selection fell back from ONNX Runtime to the "
            f"axis-aware reference evaluator: {type(error).__name__}: {error}",
        )
        return (
            source,
            f"onnx.reference axis-aware {onnx.__version__}",
            TraceDevice.CPU,
            "ReferenceEvaluator",
        )


def _write_capture(
    output: Path,
    target: dict[str, object],
    array: np.ndarray[Any, Any] | None,
    *,
    remaining: int,
    chunk_bytes: int,
    reason: str | None,
) -> tuple[dict[str, object], int]:
    base = {
        "value_id": str(target["value_id"]),
        "value_name": str(target["value_name"]),
        "node_id": target.get("node_id"),
        "role": str(target["role"]),
    }
    if array is None:
        raw_shape = target["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError("trace target shape is malformed")
        return (
            {
                **base,
                "element_type": str(target["element_type"]),
                "numpy_dtype": str(target["element_type"]),
                "shape": list(raw_shape),
                "state": "dropped",
                "full_byte_length": 0,
                "stored_byte_length": 0,
                "file_name": None,
                "reason": reason or "the reference evaluator produced no value",
            },
            remaining,
        )
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        return (
            {
                **base,
                "element_type": str(contiguous.dtype),
                "numpy_dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "state": "dropped",
                "full_byte_length": 0,
                "stored_byte_length": 0,
                "file_name": None,
                "reason": (
                    "object-backed activations require typed serialization and "
                    "cannot be captured as raw bytes"
                ),
            },
            remaining,
        )
    full_length = int(contiguous.nbytes)
    itemsize = max(1, int(contiguous.dtype.itemsize))
    stored = min(full_length, remaining)
    stored -= stored % itemsize
    if stored <= 0:
        return (
            {
                **base,
                "element_type": str(contiguous.dtype),
                "numpy_dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "state": "dropped",
                "full_byte_length": full_length,
                "stored_byte_length": 0,
                "file_name": None,
                "reason": "capture byte ceiling was exhausted before this value",
            },
            remaining,
        )
    file_name = (
        "captures/"
        + hashlib.sha256(str(target["value_id"]).encode()).hexdigest()
        + ".bin"
    )
    path = output / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    view = memoryview(contiguous).cast("B")
    with open(path, "wb") as handle:
        for offset in range(0, stored, chunk_bytes):
            handle.write(view[offset : min(stored, offset + chunk_bytes)])
        handle.flush()
        os.fsync(handle.fileno())
    truncated = stored < full_length
    return (
        {
            **base,
            "element_type": str(contiguous.dtype),
            "numpy_dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "state": "truncated" if truncated else "complete",
            "full_byte_length": full_length,
            "stored_byte_length": stored,
            "file_name": file_name,
            "reason": (
                "capture was truncated at the configured byte ceiling"
                if truncated
                else None
            ),
        },
        remaining - stored,
    )


def run(request_path: Path) -> None:
    if os.environ.get("NNEDITOR_TRACE_WORKER") != "1":
        raise RuntimeError("trace worker must be launched by NNEditor")
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("targets"), list):
        raise ValueError("trace request is malformed")
    _limit_process(raw)
    with np.load(str(raw["inputs"]), allow_pickle=False) as archive:
        feeds = {name: np.asarray(archive[name]) for name in archive.files}
    try:
        model = onnx.load_model(str(raw["model"]), load_external_data=True)
    except MemoryError as error:
        model_mib = Path(str(raw["model"])).stat().st_size / (1024 * 1024)
        limit_mib = int(raw["memory_bytes"]) / (1024 * 1024)
        raise RuntimeError(
            f"the {model_mib:,.0f} MiB ONNX model could not be loaded within "
            f"the approved {limit_mib:,.0f} MiB trace memory limit; increase "
            "the Memory limit and approve the trace again"
        ) from error
    targets = [dict(item) for item in raw["targets"] if isinstance(item, dict)]
    try:
        backend = TraceBackend(str(raw.get("backend", TraceBackend.AUTO.value)))
    except ValueError as error:
        raise ValueError(f"unknown trace backend {raw.get('backend')!r}") from error
    try:
        device = TraceDevice(str(raw.get("device", TraceDevice.AUTO.value)))
    except ValueError as error:
        raise ValueError(f"unknown trace device {raw.get('device')!r}") from error
    source, runtime, execution_device, execution_provider = _open_backend(
        model,
        Path(str(raw["model"])),
        feeds,
        targets,
        backend,
        device,
    )
    diagnostics = source.diagnostics
    output = Path(str(raw["output"]))
    remaining = int(raw["capture_bytes"])
    chunk_bytes = int(raw["chunk_bytes"])
    records_by_index: dict[int, dict[str, object]] = {}
    prioritized = [
        index
        for index, target in enumerate(targets)
        if target["role"] == "graph-input" or bool(target.get("graph_output"))
    ]
    # Membership is tested once per target, so keep it O(1): a list scan here
    # is quadratic in the number of captured values.
    prioritized_indices = set(prioritized)
    ordinary = [
        index for index in range(len(targets)) if index not in prioritized_indices
    ]
    for phase in (prioritized, ordinary):
        sizes = [source.size_hint(targets[index]) for index in phase]
        exact_budgets = (
            all(size is not None for size in sizes)
            and sum(cast(int, size) for size in sizes) <= remaining
        )
        for offset, index in enumerate(phase):
            target = targets[index]
            # Produce this value only now, and drop it once written, so peak
            # memory tracks the largest single activation rather than the sum
            # of every value the trace was asked to capture.
            capture, failure = source.take(target)
            # Reserve an equal share for each remaining value in this phase.
            # Small tensors return their unused share to the pool, while large
            # tensors keep a useful prefix rather than starving later nodes.
            # When the even share floors to zero the pool is nearly spent, so
            # offer what is genuinely left rather than dropping a value while
            # bytes remain free.
            if exact_budgets:
                budget = cast(int, sizes[offset])
            else:
                share = remaining // (len(phase) - offset)
                budget = share if share > 0 else remaining
            record, unused_budget = _write_capture(
                output,
                target,
                capture,
                remaining=budget,
                chunk_bytes=chunk_bytes,
                reason=failure,
            )
            remaining -= budget - unused_budget
            records_by_index[index] = record
            # The bytes are on disk now; releasing here is what bounds peak
            # memory to one activation rather than the whole trace.
            capture = None
    records = [records_by_index[index] for index in range(len(targets))]
    response = {
        "runtime": runtime,
        "execution_device": execution_device.value,
        "execution_provider": execution_provider,
        "records": records,
        "diagnostics": diagnostics,
    }
    Path(str(raw["response"])).write_text(
        json.dumps(response, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nneditor.tracing.trace_worker REQUEST")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
