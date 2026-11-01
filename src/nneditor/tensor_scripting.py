"""Cancellable tensor scripts in a separate scientific Python process."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from nneditor.adapters.onnx.numerical import _cap_worker_process, _close_job_object
from nneditor.cancellation import CancellationToken
from nneditor.tracing.contracts import ActivationRecord
from nneditor.tracing.runner import (
    _check_darwin_worker_memory,
    _spawn_worker,
)


class TensorScriptError(ValueError):
    """A script could not run within its selected limits."""


class _WorkerScratch(tempfile.TemporaryDirectory[str]):
    """Scratch directory that survives a just-killed Windows worker.

    ``process.wait()`` can return while the worker still holds
    ``worker-stderr.log``. The stock cleanup then raises ``PermissionError``
    and hides the timeout or cancellation error that just occurred.
    """

    def cleanup(self) -> None:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                super().cleanup()
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    self._ignore_cleanup_errors = True
                    super().cleanup()
                    return
                time.sleep(0.05)


@dataclass(frozen=True, slots=True)
class TensorScriptResult:
    stdout: str
    stderr: str
    result: str
    error: str | None
    plots: tuple[bytes, ...]
    seconds: float


def run_tensor_script(
    record: ActivationRecord,
    read: Callable[..., bytes],
    source: str,
    *,
    token: CancellationToken,
    wall_seconds: float = 120,
    memory_bytes: int = 4 * 1024**3,
) -> TensorScriptResult:
    if not record.readable:
        raise TensorScriptError("This tensor has no captured values to process.")
    if not source.strip() or len(source.encode("utf-8")) > 256 * 1024:
        raise TensorScriptError("Enter a Python script below 256 KiB.")
    if (
        not 1 <= wall_seconds <= 3600
        or not 128 * 1024**2 <= memory_bytes <= 64 * 1024**3
    ):
        raise TensorScriptError(
            "Choose a timeout of 1-3600 seconds and RAM of 128-65536 MiB."
        )
    with _WorkerScratch(prefix="nneditor-tensor-script-") as directory:
        scratch = Path(directory)
        payload = scratch / "tensor.bin"
        with payload.open("wb") as handle:
            for offset in range(0, record.stored_byte_length, 1024**2):
                token.raise_if_cancelled()
                handle.write(
                    read(
                        record.value_id,
                        offset=offset,
                        length=min(1024**2, record.stored_byte_length - offset),
                        token=token,
                    )
                )
        request = scratch / "request.json"
        request.write_text(
            json.dumps(
                {
                    "record": asdict(record),
                    "source": source,
                    "wall_seconds": wall_seconds,
                    "memory_bytes": memory_bytes,
                }
            ),
            encoding="utf-8",
        )
        process, stdout_path, stderr_path = _spawn_worker(
            scratch,
            request,
            module="nneditor.tensor_script_worker",
        )
        job = _cap_worker_process(process, memory_bytes)
        if os.name == "nt" and job is None:
            process.kill()
            process.wait()
            raise TensorScriptError("Could not apply the selected script memory limit.")
        started = time.monotonic()
        try:
            while process.poll() is None:
                token.raise_if_cancelled()
                _check_darwin_worker_memory(process, memory_bytes)
                if time.monotonic() - started > wall_seconds:
                    raise TensorScriptError(
                        f"Script exceeded its {wall_seconds:g}-second timeout."
                    )
                if any(
                    path.stat().st_size > 2 * 1024**2
                    for path in (stdout_path, stderr_path)
                ):
                    raise TensorScriptError("Script process output exceeded 2 MiB.")
                time.sleep(0.03)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            _close_job_object(job)
        token.raise_if_cancelled()
        response = scratch / "response.json"
        if process.returncode != 0 or not response.is_file():
            message = stderr_path.read_text(encoding="utf-8", errors="replace")[:8192]
            raise TensorScriptError(
                message
                or "The script process exited before returning results. "
                "Check the selected RAM limit."
            )
        if response.stat().st_size > 2 * 1024**2:
            raise TensorScriptError("Script response exceeded its output limit.")
        result = json.loads(response.read_text(encoding="utf-8"))
        plots: list[bytes] = []
        for name in result["plots"][:12]:
            path = scratch / name
            if (
                path.parent != scratch
                or path.suffix != ".png"
                or path.stat().st_size > 8 * 1024**2
            ):
                raise TensorScriptError("Invalid script plot output.")
            plots.append(path.read_bytes())
        return TensorScriptResult(
            result["stdout"],
            result["stderr"],
            result["result"],
            result["error"],
            tuple(plots),
            time.monotonic() - started,
        )
