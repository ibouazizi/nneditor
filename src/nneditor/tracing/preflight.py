"""Out-of-process ONNX Runtime availability probe.

The UI process must never import ``onnxruntime`` itself: every accelerator
family ships its own distribution under that one import name, and loading it
in-process would pin native libraries for the lifetime of the editor. A short
throwaway interpreter answers what a trace worker would find.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from nneditor.adapters.onnx.numerical import _worker_environment

__all__ = ["RuntimeStatus", "runtime_status"]

_MISSING_RUNTIME_ERROR = (
    "ONNX Runtime is not installed in this environment; install "
    "nneditor[runtime] (or an accelerator extra such as "
    "nneditor[runtime-gpu]) to enable it"
)

# The probe prints one JSON line. preload_dlls, when present, resolves NVIDIA
# libraries from site-packages first, so the reported providers reflect what
# a real trace session could actually load.
_PROBE_SCRIPT = """\
import json
import onnxruntime
preload = getattr(onnxruntime, "preload_dlls", None)
if preload is not None:
    try:
        preload()
    except Exception:
        pass
print(json.dumps({
    "version": str(onnxruntime.__version__),
    "providers": [str(item) for item in onnxruntime.get_available_providers()],
}))
"""


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """What a fresh worker process would find of ONNX Runtime."""

    available: tuple[str, ...]
    version: str | None
    error: str | None


_cached_status: RuntimeStatus | None = None


def _launch_probe(timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        env=_worker_environment("NNEDITOR_RUNTIME_PREFLIGHT"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _probe_runtime(timeout_seconds: float) -> RuntimeStatus:
    try:
        completed = _launch_probe(timeout_seconds)
    except subprocess.TimeoutExpired:
        return RuntimeStatus(
            (),
            None,
            f"the ONNX Runtime probe did not answer within {timeout_seconds:g} seconds",
        )
    except OSError as error:
        return RuntimeStatus(
            (), None, f"the ONNX Runtime probe could not start: {error}"
        )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        if "ModuleNotFoundError" in detail or "ImportError" in detail:
            return RuntimeStatus((), None, _MISSING_RUNTIME_ERROR)
        return RuntimeStatus(
            (),
            None,
            f"the ONNX Runtime probe failed: {detail or completed.returncode}",
        )
    # preload_dlls may chat on stdout before the answer; only the last
    # non-empty line is the probe's JSON.
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
        if not isinstance(payload, dict) or not isinstance(
            payload.get("providers"), list
        ):
            raise ValueError("malformed probe payload")
        providers = tuple(str(item) for item in payload["providers"])
        version = None if payload.get("version") is None else str(payload["version"])
    except ValueError:
        return RuntimeStatus(
            (), None, "the ONNX Runtime probe returned an unreadable answer"
        )
    return RuntimeStatus(providers, version, None)


def runtime_status(
    timeout_seconds: float = 10.0,
    *,
    refresh: bool = False,
) -> RuntimeStatus:
    """Report the providers a fresh trace worker could load, without loading
    ONNX Runtime in this process.

    The first call spawns a short probe interpreter and the answer is cached
    for the life of this process; pass ``refresh=True`` after installing or
    removing a runtime to probe again.
    """
    global _cached_status
    if _cached_status is None or refresh:
        _cached_status = _probe_runtime(timeout_seconds)
    return _cached_status
