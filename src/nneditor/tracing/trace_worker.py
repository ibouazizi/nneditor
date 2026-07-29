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


def _evaluate(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray[Any, Any]],
    targets: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray[Any, Any]], list[str]]:
    captures: dict[str, np.ndarray[Any, Any]] = {}
    diagnostics: list[str] = []
    input_names = set(feeds)
    runtime_targets = [
        target for target in targets if str(target["value_name"]) not in input_names
    ]
    for target in targets:
        name = str(target["value_name"])
        if name in feeds:
            captures[str(target["value_id"])] = feeds[name]
    if not runtime_targets:
        return captures, diagnostics
    names = [str(target["value_name"]) for target in runtime_targets]
    try:
        values = ReferenceEvaluator(model).run(names, feeds)
        for target, value in zip(runtime_targets, values, strict=True):
            captures[str(target["value_id"])] = np.asarray(value)
        return captures, diagnostics
    except MemoryError as error:
        diagnostics.append(
            "full trace exhausted the approved memory limit during evaluation: "
            f"{error}; increase the Memory limit and approve the trace again"
        )
        return captures, diagnostics
    except Exception as error:
        diagnostics.append(
            "full trace degraded to per-value prefix evaluation: "
            f"{type(error).__name__}: {error}"
        )

    producer_by_output = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    for target in runtime_targets:
        name = str(target["value_name"])
        producer = producer_by_output.get(name)
        if producer is None:
            diagnostics.append(f"value {name!r} has no executable producer")
            continue
        try:
            captures[str(target["value_id"])] = _evaluate_prefix(
                model,
                feeds,
                name,
                producer_limit=producer,
            )
        except Exception as error:
            diagnostics.append(
                f"value {name!r} was not captured: {type(error).__name__}: {error}"
            )
    return captures, diagnostics


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
    captures, diagnostics = _evaluate(model, feeds, targets)
    output = Path(str(raw["output"]))
    remaining = int(raw["capture_bytes"])
    chunk_bytes = int(raw["chunk_bytes"])
    records_by_index: dict[int, dict[str, object]] = {}
    prioritized = [
        index
        for index, target in enumerate(targets)
        if target["role"] == "graph-input" or bool(target.get("graph_output"))
    ]
    ordinary = [index for index in range(len(targets)) if index not in prioritized]
    for phase in (prioritized, ordinary):
        for offset, index in enumerate(phase):
            target = targets[index]
            value_id = str(target["value_id"])
            capture = captures.get(value_id)
            failure = next(
                (
                    diagnostic
                    for diagnostic in diagnostics
                    if repr(str(target["value_name"])) in diagnostic
                ),
                None,
            )
            # Reserve an equal share for each remaining value in this phase.
            # Small tensors return their unused share to the pool, while large
            # tensors keep a useful prefix rather than starving later nodes.
            budget = remaining // (len(phase) - offset)
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
