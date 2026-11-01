"""User-authored tensor processing, isolated from NNEditor's UI process."""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from nneditor.tracing.trace_worker import _limit_process


class BoundedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.truncated = False

    def write(self, value: str) -> int:
        remaining = max(0, 65536 - self.tell())
        super().write(value[:remaining])
        self.truncated |= len(value) > remaining
        return len(value)

    def captured(self) -> str:
        return self.getvalue() + (
            "\n[Output truncated at 64 KiB]" if self.truncated else ""
        )


def run(request_path: Path) -> None:
    if os.environ.get("NNEDITOR_TRACE_WORKER") != "1":
        raise ValueError("Tensor scripts must be launched by NNEditor.")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # RLIMIT_CPU counts startup CPU too. Leave the parent's wall-clock timer
    # time to report expiration before the CPU fuse can kill the worker.
    limits = dict(request)
    limits["wall_seconds"] = (
        math.ceil(time.process_time() + float(request.get("wall_seconds", 120))) + 1
    )
    _limit_process(limits)
    scratch = request_path.parent
    output, errors = BoundedOutput(), BoundedOutput()
    response: dict[str, Any] = {"plots": [], "result": "", "error": None}
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            scipy = importlib.import_module("scipy")
            sklearn = importlib.import_module("sklearn")

            metadata = request["record"]
            dtype = np.dtype(metadata["numpy_dtype"])
            count = metadata["stored_byte_length"] // dtype.itemsize
            complete = metadata["stored_byte_length"] == metadata["full_byte_length"]
            shape = tuple(metadata["shape"]) if complete else (count,)
            # Private copy-on-write pages preserve captured data during in-place work.
            tensor = np.memmap(
                scratch / "tensor.bin", mode="c", dtype=dtype, shape=shape
            )
            namespace: dict[str, Any] = {
                "__name__": "__tensor_script__",
                "tensor": tensor,
                "metadata": metadata,
                "math": math,
                "np": np,
                "numpy": np,
                "matplotlib": matplotlib,
                "scipy": scipy,
                "sklearn": sklearn,
                "scikit": sklearn,
                "plt": plt,
            }
            tree = ast.parse(request["source"], filename="<tensor script>", mode="exec")
            expression = (
                tree.body.pop()
                if tree.body and isinstance(tree.body[-1], ast.Expr)
                else None
            )
            exec(compile(tree, "<tensor script>", "exec"), namespace, namespace)
            value = (
                eval(
                    compile(
                        ast.Expression(expression.value), "<tensor script>", "eval"
                    ),
                    namespace,
                    namespace,
                )
                if isinstance(expression, ast.Expr)
                else namespace.get("result")
            )
            if value is not None:
                response["result"] = repr(value)[:65536]
            for index, number in enumerate(plt.get_fignums()[:12]):
                figure = plt.figure(number)
                width, height = figure.get_size_inches() * figure.dpi
                dpi = float(
                    figure.dpi * min(1.0, math.sqrt(4_000_000 / max(1, width * height)))
                )
                name = f"plot-{index}.png"
                figure.savefig(scratch / name, dpi=dpi, bbox_inches="tight")
                if (scratch / name).stat().st_size <= 8 * 1024**2:
                    response["plots"].append(name)
                else:
                    print("A plot exceeded 8 MiB and was omitted.", file=sys.stderr)
            plt.close("all")
        except BaseException:
            response["error"] = traceback.format_exc()[-16384:]
    response.update(stdout=output.captured(), stderr=errors.captured())
    (scratch / "response.json").write_text(json.dumps(response), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
