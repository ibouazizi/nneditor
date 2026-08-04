"""End-to-end consent-gated ONNX tracing through the isolated worker."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from nneditor.application.session import ApplicationService, ModelSession
from nneditor.cancellation import OperationCancelled
from nneditor.tracing import (
    CaptureState,
    TraceApproval,
    TraceDevice,
    TraceLimits,
    TraceRequest,
)
from nneditor.tracing.runner import TraceError, recommended_trace_limits
from tests.fixtures.onnx_models import (
    build_custom_domain_model,
    build_embedded_model,
)


def _request(
    session: ModelSession, *, capture_bytes: int = 1024 * 1024
) -> TraceRequest:
    specification = session.default_trace_inputs(seed=29)
    limits = TraceLimits(
        wall_seconds=10,
        memory_bytes=1024 * 1024 * 1024,
        capture_bytes=capture_bytes,
        chunk_bytes=min(64, capture_bytes),
    )
    approval = TraceApproval.approve(
        session.title,
        session.document.source.content_hash,
        specification,
        limits,
    )
    return TraceRequest(specification, limits, approval)


def test_approved_trace_captures_inputs_and_intermediates_out_of_process(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    source_bytes = path.read_bytes()
    os.environ.pop("NNEDITOR_TRACE_WORKER", None)
    with ApplicationService() as service:
        session = service.open_model(path)
        result = session.trace_async(_request(session)).result(timeout=20)
        assert result.partial is False
        # The notes channel survives the worker → runner → store round trip;
        # a clean CPU run has nothing to note.
        assert result.notes == ()
        assert {record.value_name for record in result.records} == {
            "input",
            "scaled",
            "output",
        }
        assert all(record.readable for record in result.records)
        assert result.runtime.startswith("onnxruntime")
        assert result.execution_device is TraceDevice.CPU
        assert result.execution_provider == "CPUExecutionProvider"
        assert os.environ.get("NNEDITOR_TRACE_WORKER") is None
        assert session.trace_input_specifications() == (
            _request(session).specification,
        )
        output = next(
            record for record in result.records if record.value_name == "output"
        )
        statistics = session.compute_activation_statistics_async(
            result.id, output.value_id
        ).result(timeout=10)
        assert statistics.element_count == 8
        assert path.read_bytes() == source_bytes


def test_seeded_rerun_is_byte_deterministic_and_selection_is_honored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        session = service.open_model(path)
        request = _request(session)
        output_id = session.document.main_graph.outputs[0]
        selected_request = TraceRequest(
            request.specification,
            request.limits,
            request.approval,
            frozenset({output_id}),
        )
        first = session.trace_async(selected_request).result(timeout=20)
        first_bytes = session.activations(first.id).read(output_id)
        second = session.trace_async(selected_request).result(timeout=20)
        second_bytes = session.activations(second.id).read(output_id)

        assert first.id == second.id
        assert first_bytes == second_bytes
        assert {record.value_id for record in second.records} == {output_id}


def test_trace_refuses_missing_or_mismatched_per_run_approval(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        session = service.open_model(path)
        request = _request(session)
        refused = TraceRequest(
            request.specification,
            request.limits,
            TraceApproval(
                session.title,
                session.document.source.content_hash,
                request.specification.hash,
                request.limits.hash,
                False,
            ),
        )
        with pytest.raises(ValueError, match="explicit per-run approval"):
            session.trace_async(refused).result(timeout=10)
        assert session.traces() == ()


def test_preview_trace_keeps_breadth_and_a_narrowed_rerun_upgrades_it(
    tmp_path: Path,
) -> None:
    """A preview keeps a sliver of every value; a whole-value re-trace of one
    value lands in the same trace id and upgrades that record via the merge."""
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService() as service:
        session = service.open_model(path)
        specification = session.default_trace_inputs(seed=29)
        # 320 bytes: the 128-byte input and output boundaries complete via
        # the exact-budget path, leaving 64 bytes for the ordinary 'scaled'
        # value — a truncated preview share instead of a greedy whole value.
        preview_limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=1024 * 1024 * 1024,
            capture_bytes=320,
            chunk_bytes=64,
        )
        preview = session.trace_async(
            TraceRequest(
                specification,
                preview_limits,
                TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    preview_limits,
                ),
                capture_policy="preview",
            )
        ).result(timeout=20)
        assert {record.value_name for record in preview.records} == {
            "input",
            "scaled",
            "output",
        }
        # Breadth: every requested value is readable, truncated or complete.
        assert all(record.readable for record in preview.records)
        scaled = next(
            record for record in preview.records if record.value_name == "scaled"
        )
        assert scaled.state is CaptureState.TRUNCATED
        assert scaled.stored_byte_length == 64

        # The narrowed whole-value re-trace of the same specification shares
        # the trace id — the policy never joins the key — so the store's
        # merge upgrades the preview record in place and keeps the rest.
        rerun_limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=1024 * 1024 * 1024,
            capture_bytes=1024 * 1024,
            chunk_bytes=64,
        )
        upgraded = session.trace_async(
            TraceRequest(
                specification,
                rerun_limits,
                TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    rerun_limits,
                ),
                value_ids=frozenset({scaled.value_id}),
            )
        ).result(timeout=20)

        assert upgraded.id == preview.id
        record = upgraded.record(scaled.value_id)
        assert record.state is CaptureState.COMPLETE
        assert record.stored_byte_length == 32 * 4
        # The untouched preview captures survive the merge alongside it.
        assert {item.value_name for item in upgraded.records} == {
            "input",
            "scaled",
            "output",
        }


def test_capture_ceiling_is_visible_as_partial_not_silent(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=32)
    with ApplicationService() as service:
        session = service.open_model(path)
        result = session.trace_async(_request(session, capture_bytes=64)).result(
            timeout=20
        )
        assert result.partial
        assert any(record.reason for record in result.records)
        assert any(
            record.state.value in {"truncated", "dropped"} for record in result.records
        )


def test_unsupported_custom_operator_degrades_to_diagnostic_partial_trace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.onnx"
    build_custom_domain_model(path)
    with ApplicationService() as service:
        session = service.open_model(path)
        result = session.trace_async(_request(session)).result(timeout=20)
        assert result.partial
        assert result.diagnostics
        assert result.record(session.document.main_graph.inputs[0]).readable
        assert any(not record.readable for record in result.records)


def test_wall_clock_limit_failure_commits_no_partial_trace(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=128)
    with ApplicationService() as service:
        session = service.open_model(path)
        specification = session.default_trace_inputs()
        limits = TraceLimits(
            wall_seconds=0.0001,
            memory_bytes=1024 * 1024 * 1024,
            capture_bytes=1024,
            chunk_bytes=64,
        )
        request = TraceRequest(
            specification,
            limits,
            TraceApproval.approve(
                session.title,
                session.document.source.content_hash,
                specification,
                limits,
            ),
        )
        with pytest.raises(TraceError, match="exceeded"):
            session.trace_async(request).result(timeout=20)
        assert session.traces() == ()


def test_inputs_cannot_exceed_approved_worker_memory(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        session = service.open_model(path)
        specification = session.default_trace_inputs()
        limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=8,
            capture_bytes=16,
            chunk_bytes=4,
        )
        request = TraceRequest(
            specification,
            limits,
            TraceApproval.approve(
                session.title,
                session.document.source.content_hash,
                specification,
                limits,
            ),
        )
        with pytest.raises(TraceError, match="inputs require"):
            session.trace_async(request).result(timeout=10)
        assert session.traces() == ()


def test_model_load_memory_is_rejected_before_worker_launch(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        session = service.open_model(path)
        specification = session.default_trace_inputs()
        limits = TraceLimits(
            wall_seconds=10,
            memory_bytes=128 * 1024 * 1024,
            capture_bytes=1024,
            chunk_bytes=64,
        )
        request = TraceRequest(
            specification,
            limits,
            TraceApproval.approve(
                session.title,
                session.document.source.content_hash,
                specification,
                limits,
            ),
        )

        with pytest.raises(TraceError, match="before evaluation"):
            session.trace_async(request).result(timeout=10)
        assert session.traces() == ()


def test_large_inline_onnx_model_gets_practical_trace_defaults() -> None:
    limits = recommended_trace_limits(1_326_283_019)

    # Twelve model-sized sets: a measured full-graph CUDA trace of this
    # 1.33 GB model needed more host commit than the previous 16 GiB cap.
    assert limits.memory_bytes == 32768 * 1024 * 1024
    assert limits.wall_seconds == 300


def test_worker_streams_are_files_never_undrained_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll loop never drains pipes, so pipes would deadlock the worker.

    A worker whose stderr outgrew the OS pipe buffer blocked mid-write and
    was killed as a wall-clock overrun on a real 1.33 GB full-graph trace.
    """
    import subprocess

    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    spawns: list[dict[str, object]] = []
    original_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> object:
        spawns.append(kwargs)
        return original_popen(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    with ApplicationService() as service:
        session = service.open_model(path)
        result = session.trace_async(_request(session)).result(timeout=20)

    assert result.partial is False
    worker_spawns = [
        kwargs
        for kwargs in spawns
        if "trace_worker" in str(kwargs.get("cwd", "")) or kwargs.get("stdout")
    ]
    assert worker_spawns
    for kwargs in worker_spawns:
        assert kwargs.get("stdout") is not subprocess.PIPE
        assert kwargs.get("stderr") is not subprocess.PIPE


def test_small_models_get_the_accelerator_warmup_wall_clock_floor() -> None:
    # A TensorRT session compiles an engine before the first inference, so
    # the recommendation must cover first-run warm-up even for tiny models.
    assert recommended_trace_limits(1024).wall_seconds == 300
    assert recommended_trace_limits(255 * 1024 * 1024).wall_seconds == 300
    # Larger models keep the size-scaled allowance beyond the floor.
    assert recommended_trace_limits(2 * 1024**3).wall_seconds == 480


def test_cancellation_discards_worker_staging(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=512 * 1024)
    with ApplicationService() as service:
        session = service.open_model(path)
        job = session.trace_async(_request(session, capture_bytes=16 * 1024 * 1024))
        deadline = time.monotonic() + 2
        while job.state.value == "pending" and time.monotonic() < deadline:
            time.sleep(0.005)
        job.cancel()
        with pytest.raises(OperationCancelled):
            job.result(timeout=20)
        assert session.traces() == ()


def test_base_and_edited_revision_compare_at_affected_nodes(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=8)
    with ApplicationService() as service:
        session = service.open_model(path)
        request = _request(session)
        base = session.trace_async(request).result(timeout=20)
        weight_id = session.document.main_graph.initializers[0]
        session.editing.replace_bytes(weight_id, 0, np.float32(20.0).tobytes())
        edited = session.trace_async(_request(session)).result(timeout=20)

        assert base.key.revision_id is None
        assert edited.key.revision_id is not None
        comparison = session.compare_traces_async(base.id, edited.id).result(timeout=20)
        assert comparison.nodes
        assert max(metric.max_absolute for metric in comparison.nodes) > 0
