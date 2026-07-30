"""Minimal subprocess entry point for ONNX intermediate-value capture."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


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
            ReferenceEvaluator(model).run([name], feeds),
        )
        return np.asarray(values[0])
    finally:
        model.graph.node.extend(trailing_nodes)


def _prefix_builds(
    model: onnx.ModelProto,
    all_nodes: list[onnx.NodeProto],
    limit: int,
) -> bool:
    """Whether the runtime can build an evaluator over ``nodes[:limit + 1]``.

    The graph is restored before returning. Detaching nodes keeps the model's
    initializers untouched, so probing never copies the weight payload.
    """
    model.graph.ClearField("node")
    model.graph.node.extend(all_nodes[: limit + 1])
    try:
        ReferenceEvaluator(model)
        return True
    except Exception:
        # Any build failure answers the question this probe asks.
        return False
    finally:
        model.graph.ClearField("node")
        model.graph.node.extend(all_nodes)


def _buildable_frontier(model: onnx.ModelProto) -> int:
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
        if _prefix_builds(model, all_nodes, middle):
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
    ) -> None:
        self._model = model
        self._feeds = feeds
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
                _evaluate_prefix(
                    self._model, self._feeds, name, producer_limit=producer
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


def open_captures(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
) -> CaptureSource:
    """Prepare capture, choosing whole-model evaluation when it is possible."""
    diagnostics: list[str] = []
    input_names = set(feeds)
    runtime_targets = [
        target for target in targets if str(target["value_name"]) not in input_names
    ]
    if not runtime_targets:
        return CaptureSource(model, feeds, {}, diagnostics, len(model.graph.node), None)

    names = [str(target["value_name"]) for target in runtime_targets]
    try:
        evaluator = ReferenceEvaluator(model)
    except MemoryError as error:
        diagnostics.append(
            "building the reference runtime exhausted the approved memory "
            f"limit: {error}; increase the Memory limit and approve the trace "
            "again"
        )
        return CaptureSource(
            model, feeds, {}, diagnostics, 0, "the runtime ran out of memory"
        )
    except Exception as error:
        diagnostics.append(
            "the reference runtime cannot execute the whole model: "
            f"{type(error).__name__}: {error}; capturing per value instead"
        )
        return _prefix_source(model, feeds, diagnostics)
    try:
        values = evaluator.run(names, feeds)
    except MemoryError as error:
        diagnostics.append(
            "full trace exhausted the approved memory limit during evaluation: "
            f"{error}; increase the Memory limit and approve the trace again"
        )
        return CaptureSource(
            model, feeds, {}, diagnostics, 0, "the runtime ran out of memory"
        )
    except Exception as error:
        diagnostics.append(
            "full trace degraded to per-value prefix evaluation: "
            f"{type(error).__name__}: {error}"
        )
        return _prefix_source(model, feeds, diagnostics)
    whole = {
        str(target["value_id"]): np.asarray(value)
        for target, value in zip(runtime_targets, values, strict=True)
    }
    return CaptureSource(model, feeds, whole, diagnostics, len(model.graph.node), None)


def _prefix_source(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    diagnostics: list[str],
) -> CaptureSource:
    """Set up per-value capture, first locating what the runtime cannot build.

    Finding the frontier costs about log2(n) evaluator builds and spares every
    value beyond it a rebuild that is guaranteed to fail — the difference
    between ten probes and hundreds of futile evaluations on a large graph.
    """
    frontier = _buildable_frontier(model)
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
    return CaptureSource(model, feeds, None, diagnostics, frontier, reason)


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
    source = open_captures(model, feeds, targets)
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
        "runtime": f"onnx.reference {onnx.__version__}",
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
