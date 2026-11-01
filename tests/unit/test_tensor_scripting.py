"""Scientific tensor scripts run without mutating captures or blocking the UI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.tensor_scripting import TensorScriptError, run_tensor_script
from tests.unit.test_tensor_page import record_for


def test_scientific_globals_print_last_expression_plots_and_private_mutation() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = array.tobytes()
    result = run_tensor_script(
        record_for(array),
        lambda _id, **kw: raw[kw["offset"] : kw["offset"] + kw["length"]],
        """print(math.sqrt(9), scipy.__name__, sklearn.__name__, scikit is sklearn)
tensor += 5
plt.plot(tensor[0])
float(tensor.mean())
""",
        token=CancellationToken(),
        wall_seconds=90,
    )
    assert "3.0 scipy sklearn True" in result.stdout
    assert result.result == "10.5"
    assert result.error is None
    assert len(result.plots) == 1 and result.plots[0].startswith(b"\x89PNG")
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype=np.float32).reshape(3, 4), array
    )


def test_script_errors_keep_printed_output_and_bound_large_output() -> None:
    array = np.arange(8, dtype=np.float32)
    result = run_tensor_script(
        record_for(array),
        lambda *_a, **_kw: array.tobytes(),
        "print('x'*100_000)\nraise ValueError('user error')",
        token=CancellationToken(),
        wall_seconds=90,
    )
    assert "Output truncated" in result.stdout and len(result.stdout) < 66_000
    assert result.error and "ValueError: user error" in result.error


def test_partial_capture_exposes_flat_prefix_and_original_metadata() -> None:
    array = np.arange(12, dtype=np.int64).reshape(3, 4)
    record = record_for(array, stored=24)
    result = run_tensor_script(
        record,
        lambda *_a, **_kw: array.tobytes()[:24],
        "print(tensor.shape, metadata['shape'])\nint(tensor.sum())",
        token=CancellationToken(),
        wall_seconds=90,
    )
    assert result.result == "3"
    assert "(3,) [3, 4]" in result.stdout


def test_script_timeout_and_cancellation() -> None:
    array = np.ones(1, np.float32)
    with pytest.raises(TensorScriptError, match="timeout"):
        run_tensor_script(
            record_for(array),
            lambda *_a, **_kw: array.tobytes(),
            "while True: pass",
            token=CancellationToken(),
            wall_seconds=1,
        )
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        run_tensor_script(
            record_for(array),
            lambda *_a, **_kw: array.tobytes(),
            "tensor.sum()",
            token=token,
        )


def test_script_limits_reject_before_reading() -> None:
    array = np.ones(1, np.float32)
    for source, timeout, ram in (("", 10, 2**30), ("1", 0, 2**30), ("1", 10, 1)):
        with pytest.raises(TensorScriptError):
            run_tensor_script(
                record_for(array),
                lambda *_a, **_kw: pytest.fail("read"),
                source,
                token=CancellationToken(),
                wall_seconds=timeout,
                memory_bytes=ram,
            )


def test_worker_executes_in_place_without_writing_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nneditor import tensor_script_worker

    array = np.arange(4, dtype=np.float32)
    original = array.tobytes()
    (tmp_path / "tensor.bin").write_bytes(original)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "record": asdict(record_for(array)),
                "source": "tensor *= 2\nprint(math.pi)\nplt.plot(tensor)\n"
                "result = float(tensor.sum())",
            }
        )
    )
    monkeypatch.setenv("NNEDITOR_TRACE_WORKER", "1")
    monkeypatch.setattr(tensor_script_worker, "_limit_process", lambda _: None)
    tensor_script_worker.run(request)
    response = json.loads((tmp_path / "response.json").read_text())
    assert response["result"] == "12.0" and response["error"] is None
    assert "3.14159" in response["stdout"] and response["plots"] == ["plot-0.png"]
    assert (tmp_path / "tensor.bin").read_bytes() == original
    request.write_text(
        json.dumps(
            {
                "record": asdict(record_for(array)),
                "source": "print('kept')\nraise RuntimeError('script error')",
            }
        )
    )
    tensor_script_worker.run(request)
    response = json.loads((tmp_path / "response.json").read_text())
    assert "kept" in response["stdout"] and "script error" in response["error"]
