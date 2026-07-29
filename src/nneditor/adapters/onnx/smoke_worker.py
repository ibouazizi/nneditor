"""Subprocess entry point for the opt-in ONNX numerical smoke comparison."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def _limit_process() -> None:
    """Apply conservative POSIX limits when the platform exposes ``resource``.

    On Windows the parent caps this process instead: ``compare_numerically``
    assigns it to a Job Object with a memory ceiling and kill-on-job-close.
    """
    if os.name == "nt":
        return
    try:
        import resource

        resource_api = cast(Any, resource)
        resource_api.setrlimit(resource_api.RLIMIT_CPU, (30, 30))
        memory = 2 * 1024 * 1024 * 1024
        resource_api.setrlimit(
            resource_api.RLIMIT_AS,
            (memory, memory),
        )
        resource_api.setrlimit(
            resource_api.RLIMIT_NOFILE,
            (64, 64),
        )
    except (ImportError, OSError, ValueError):
        return


def _evaluate(
    path: str, feeds: dict[str, np.ndarray[Any, Any]]
) -> list[np.ndarray[Any, Any]]:
    model = onnx.load_model(path, load_external_data=True)
    evaluator = ReferenceEvaluator(model)
    return [np.asarray(item) for item in evaluator.run(None, feeds)]


def run(request_path: Path) -> None:
    if os.environ.get("NNEDITOR_SMOKE_WORKER") != "1":
        raise RuntimeError("worker must be launched by NNEditor")
    _limit_process()
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    inputs_file = Path(str(raw["inputs"]))
    with np.load(inputs_file, allow_pickle=False) as archive:
        feeds = {name: np.asarray(archive[name]) for name in archive.files}
    original = _evaluate(str(raw["original"]), feeds)
    edited = _evaluate(str(raw["edited"]), feeds)
    if len(original) != len(edited):
        raise RuntimeError(
            f"output count changed from {len(original)} to {len(edited)}"
        )
    atol = float(raw["atol"])
    rtol = float(raw["rtol"])
    reports: list[dict[str, object]] = []
    passed = True
    for index, (before, after) in enumerate(zip(original, edited, strict=True)):
        same_shape = before.shape == after.shape
        numeric = np.issubdtype(before.dtype, np.number) and np.issubdtype(
            after.dtype, np.number
        )
        if numeric and same_shape:
            close = np.isclose(before, after, atol=atol, rtol=rtol, equal_nan=True)
            output_passed = bool(np.all(close))
            difference = np.abs(before.astype(np.float64) - after.astype(np.float64))
            max_absolute = float(np.nanmax(difference)) if difference.size else 0.0
            denominator = np.maximum(np.abs(before.astype(np.float64)), atol)
            relative = difference / denominator
            max_relative = float(np.nanmax(relative)) if relative.size else 0.0
        else:
            output_passed = same_shape and bool(np.array_equal(before, after))
            max_absolute = None
            max_relative = None
        passed = passed and output_passed
        reports.append(
            {
                "index": index,
                "passed": output_passed,
                "before_shape": list(before.shape),
                "after_shape": list(after.shape),
                "before_dtype": str(before.dtype),
                "after_dtype": str(after.dtype),
                "max_absolute_error": max_absolute,
                "max_relative_error": max_relative,
            }
        )
    response = {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "outputs": reports,
        "runtime": f"onnx.reference {onnx.__version__}",
    }
    Path(str(raw["response"])).write_text(
        json.dumps(response, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nneditor.adapters.onnx.smoke_worker REQUEST")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
