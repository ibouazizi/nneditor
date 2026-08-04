"""Out-of-process ONNX Runtime preflight probing."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import pytest

from nneditor.tracing import preflight


def _completed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "_cached_status", None)


def test_probe_output_parses_and_the_answer_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake(timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        calls.append(timeout_seconds)
        return _completed(
            stdout=(
                "preload chatter\n"
                '{"version": "1.23.0", "providers": '
                '["CUDAExecutionProvider", "CPUExecutionProvider"]}\n'
            )
        )

    monkeypatch.setattr(preflight, "_launch_probe", fake)
    status = preflight.runtime_status()
    assert status.available == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert status.version == "1.23.0"
    assert status.error is None
    # The second call answers from the per-process cache.
    assert preflight.runtime_status() is status
    assert calls == [10.0]


def test_missing_runtime_reports_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_launch_probe",
        lambda timeout_seconds: _completed(
            returncode=1,
            stderr="ModuleNotFoundError: No module named 'onnxruntime'",
        ),
    )
    status = preflight.runtime_status()
    assert status.available == ()
    assert status.version is None
    assert status.error is not None
    assert "nneditor[runtime]" in status.error


def test_refresh_reprobes_and_replaces_the_cached_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: Iterator[subprocess.CompletedProcess[str]] = iter(
        [
            _completed(
                stdout='{"version": "1.23.0", "providers": ["CPUExecutionProvider"]}'
            ),
            _completed(
                returncode=1,
                stderr="ModuleNotFoundError: No module named 'onnxruntime'",
            ),
        ]
    )
    monkeypatch.setattr(
        preflight, "_launch_probe", lambda timeout_seconds: next(answers)
    )
    first = preflight.runtime_status()
    assert first.available == ("CPUExecutionProvider",)
    # Without refresh the cache answers; the second probe result is only
    # observed once refresh is explicit.
    assert preflight.runtime_status() is first
    second = preflight.runtime_status(refresh=True)
    assert second.available == ()
    assert second.error is not None
    assert preflight.runtime_status() is second


def test_probe_failures_surface_as_errors_not_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timing_out(timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="python", timeout=timeout_seconds)

    monkeypatch.setattr(preflight, "_launch_probe", timing_out)
    status = preflight.runtime_status(timeout_seconds=0.5)
    assert status.available == ()
    assert status.error is not None
    assert "0.5" in status.error

    monkeypatch.setattr(
        preflight,
        "_launch_probe",
        lambda timeout_seconds: _completed(stdout="not json at all"),
    )
    garbage = preflight.runtime_status(refresh=True)
    assert garbage.available == ()
    assert garbage.error is not None
    assert "unreadable" in garbage.error


def test_real_probe_answers_from_a_subprocess() -> None:
    """The genuine probe script runs; this process never imports onnxruntime."""
    status = preflight.runtime_status(timeout_seconds=120.0, refresh=True)
    if status.error is None:
        assert "CPUExecutionProvider" in status.available
        assert status.version
    else:
        # An environment without any runtime extra still gets a useful answer.
        assert status.available == ()
        assert "nneditor[runtime]" in status.error or "probe" in status.error
