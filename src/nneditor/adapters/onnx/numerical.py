"""Opt-in numerical smoke comparison in an isolated worker process.

The worker limits itself with POSIX rlimits where the platform has them; on
Windows the *parent* applies a Job Object to the subprocess instead — a
process-memory ceiling plus kill-on-job-close, so an orphaned worker cannot
outlive the editor. Job failures degrade gracefully to an uncapped worker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = ["NumericalComparison", "NumericalComparisonError", "compare_numerically"]

_WORKER_MEMORY_LIMIT: Final = 2 * 1024 * 1024 * 1024
"""Same ceiling the POSIX rlimit path sets inside the worker."""

_JOB_OBJECT_LIMIT_PROCESS_MEMORY: Final = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Final = 9


class NumericalComparisonError(RuntimeError):
    """The optional worker could not produce a trustworthy comparison."""


@dataclass(frozen=True, slots=True)
class NumericalComparison:
    passed: bool
    atol: float
    rtol: float
    outputs: tuple[dict[str, object], ...]
    runtime: str

    def to_json(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "outputs": list(self.outputs),
            "runtime": self.runtime,
            "claim": (
                "Numerical smoke comparison passed for the supplied inputs."
                if self.passed
                else "Numerical smoke comparison found a mismatch."
            ),
        }


def _create_job_object(memory_limit: int = _WORKER_MEMORY_LIMIT) -> int | None:
    """Create a Windows Job Object capping worker memory, or ``None``.

    The job is configured with ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``: a
    per-process memory ceiling and kill-on-job-close, so every process in the
    job dies when the last handle closes. Returns ``None`` off Windows or when
    the API refuses.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:  # pragma: no cover - kernel32 is always present on NT
        return None

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    limits.ProcessMemoryLimit = memory_limit
    configured = kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    )
    if not configured:
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _assign_process_to_job(job: int, process_handle: int) -> bool:
    """Attach one process to the job; ``False`` means the cap is not applied."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    return bool(kernel32.AssignProcessToJobObject(job, process_handle))


def _close_job_object(job: int | None) -> None:
    """Close the job handle, which kills any process still inside it."""
    if job is None or sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(job)


def _cap_worker_process(
    process: subprocess.Popen[str],
    memory_limit: int = _WORKER_MEMORY_LIMIT,
) -> int | None:
    """Cap the worker on Windows via a Job Object; ``None`` means no cap.

    POSIX workers apply ``resource`` rlimits to themselves; Windows has no
    in-child equivalent, so the parent owns the cap. Creation or assignment
    failure degrades to an uncapped worker rather than refusing the
    comparison.
    """
    if sys.platform != "win32":
        return None
    job = _create_job_object(memory_limit)
    if job is None:
        return None
    handle = getattr(process, "_handle", None)
    if handle is None or not _assign_process_to_job(job, int(handle)):
        _close_job_object(job)
        return None
    return job


def _worker_environment(
    marker: str = "NNEDITOR_SMOKE_WORKER",
) -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "TEMP",
            "TMP",
            "WINDIR",
            "LD_LIBRARY_PATH",
        }
    }
    allowed["PYTHONNOUSERSITE"] = "1"
    allowed[marker] = "1"
    return allowed


def compare_numerically(
    original: Path | str,
    edited: Path | str,
    inputs: Mapping[str, NDArray[np.generic]],
    *,
    approved: bool,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    timeout_seconds: float = 30.0,
) -> NumericalComparison:
    """Compare outputs only after the caller explicitly approves execution.

    The worker uses ONNX's reference evaluator.  It does not import model code,
    custom Python operators, or an arbitrary runtime.  This is evidence for the
    supplied inputs only and is never presented as a proof of equivalence.
    """
    if not approved:
        raise NumericalComparisonError(
            "numerical comparison requires explicit caller approval"
        )
    if atol < 0 or rtol < 0:
        raise NumericalComparisonError("comparison tolerances must be non-negative")
    if timeout_seconds <= 0:
        raise NumericalComparisonError("comparison timeout must be positive")
    source = Path(original).resolve()
    target = Path(edited).resolve()
    with tempfile.TemporaryDirectory(prefix="nneditor-smoke-") as temporary:
        root = Path(temporary)
        inputs_path = root / "inputs.npz"
        request_path = root / "request.json"
        response_path = root / "response.json"
        np.savez(inputs_path, **dict(inputs))  # type: ignore[arg-type]
        request_path.write_text(
            json.dumps(
                {
                    "original": str(source),
                    "edited": str(target),
                    "inputs": str(inputs_path),
                    "atol": atol,
                    "rtol": rtol,
                    "response": str(response_path),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "nneditor.adapters.onnx.smoke_worker",
                    str(request_path),
                ],
                cwd=root,
                env=_worker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            raise NumericalComparisonError(
                f"numerical worker could not start: {error}"
            ) from error
        job = _cap_worker_process(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise NumericalComparisonError(
                f"numerical worker exceeded {timeout_seconds:g} seconds"
            ) from error
        finally:
            _close_job_object(job)
        if process.returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise NumericalComparisonError(
                f"numerical worker failed: {message or process.returncode}"
            )
        try:
            raw = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("outputs"), list):
                raise ValueError("malformed response")
            return NumericalComparison(
                passed=bool(raw["passed"]),
                atol=float(raw["atol"]),
                rtol=float(raw["rtol"]),
                outputs=tuple(
                    dict(item) for item in raw["outputs"] if isinstance(item, dict)
                ),
                runtime=str(raw["runtime"]),
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise NumericalComparisonError(
                f"numerical worker returned an invalid response: {error}"
            ) from error
