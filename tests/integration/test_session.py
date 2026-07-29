"""Tests for sessions and the application service (P1.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nneditor.application.session import ApplicationService, SessionError
from nneditor.ir.capabilities import Availability, Capability
from nneditor.storage.store import TensorUnavailableError
from tests.fixtures.onnx_models import build_embedded_model, build_external_model
from tests.fixtures.pytorch_models import build_safetensors_file

ELEMENTS = 64


def test_open_inspect_close_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    values = build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        assert session.title == "model.onnx"
        assert session.capability(Capability.TOPOLOGY).availability in (
            Availability.AVAILABLE,
            Availability.PARTIAL,
        )

        layout = session.scene()
        assert layout.scene.node_count == 2

        weight_id = session.document.main_graph.initializers[0]
        assert session.store.read(weight_id) == values.tobytes()

        assert service.open_sessions == (session,)
        service.close_session(session.id)
        assert session.closed
        with pytest.raises(SessionError, match="closed"):
            session.scene()
        with pytest.raises(TensorUnavailableError, match="closed"):
            session.store.read(weight_id)
        with pytest.raises(SessionError, match="no session"):
            service.session(session.id)


def test_browser_uploads_are_isolated_and_cleaned_up() -> None:
    service = ApplicationService()
    first = service.stage_upload("nested/model.onnx", b"first")
    second = service.stage_upload("model.onnx", b"second")

    assert first.name == second.name == "model.onnx"
    assert first.parent != second.parent
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    upload_root = first.parent.parent
    service.close()
    assert not upload_root.exists()


def test_async_open_reports_through_the_job(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    events: list[str] = []
    with ApplicationService(
        job_listener=lambda job: events.append(job.state.value)
    ) as service:
        job = service.open_model_async(path)
        session = job.result(timeout=10)
        assert len(session.document.main_graph.nodes) == 2
        assert service.session(session.id) is session
    assert "succeeded" in events


def test_browser_upload_staging_and_architecture_preparation_are_async(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        job = service.open_upload_async("model.onnx", path.read_bytes())
        session = job.result(timeout=10)
        assert session.document.main_graph.nodes
        assert service.session(session.id) is session


def test_streamed_browser_upload_is_adopted_on_the_worker(tmp_path: Path) -> None:
    path = tmp_path / "browser-upload.bin"
    model = tmp_path / "source.onnx"
    build_embedded_model(model, elements=ELEMENTS)
    path.write_bytes(model.read_bytes())
    with ApplicationService() as service:
        job = service.open_uploaded_file_async("friendly-name.onnx", path)
        session = job.result(timeout=10)
        assert session.title == "friendly-name.onnx"
        assert path.exists(), "adopting an upload never mutates the source file"


def test_browser_upload_accepts_non_onnx_artifacts_by_content(tmp_path: Path) -> None:
    source = tmp_path / "weights.safetensors"
    build_safetensors_file(source)
    with ApplicationService() as service:
        staged = service.stage_upload("nested/weights.safetensors", source.read_bytes())
        assert staged.name == "weights.safetensors"
        session = service.open_model(staged)
        assert session.document.tensors


def test_streamed_upload_name_does_not_gate_content_detection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    build_safetensors_file(source)
    with ApplicationService() as service:
        job = service.open_uploaded_file_async("friendly.bin", source)
        session = job.result(timeout=10)
        assert session.title == "friendly.bin"
        assert session.document.tensors


def test_async_open_failure_is_contained(tmp_path: Path) -> None:
    bogus = tmp_path / "not-a-model.onnx"
    bogus.write_bytes(b"this is not a protobuf")
    with ApplicationService() as service:
        job = service.open_model_async(bogus)
        job.wait(timeout=10)
        assert job.state.value == "failed"
        assert service.open_sessions == ()


def test_search_and_neighborhood_queries(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        hits = session.search("scale")
        assert len(hits) == 1
        near = session.neighborhood(session.document.entry_graph, hits[0], hops=1)
        assert len(near) == 2, "scale and its consumer shift"


def test_two_sessions_share_layouts_by_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        first = service.open_model(path)
        second = service.open_model(path)
        assert first.scene() is second.scene(), (
            "same content hash and settings reuse the cached layout"
        )


def test_closing_one_session_preserves_shared_layouts(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=ELEMENTS)
    with ApplicationService() as service:
        first = service.open_model(path)
        second = service.open_model(path)
        cached = first.scene()
        service.close_session(first.id)
        assert second.scene() is cached


def test_external_model_session_reads_lazily(tmp_path: Path) -> None:
    path, values = build_external_model(tmp_path, elements=ELEMENTS)
    with ApplicationService() as service:
        session = service.open_model(path)
        weight_id = session.document.main_graph.initializers[0]
        assert session.store.open_file_count == 0, "opening reads no tensor"
        assert (
            session.store.read(weight_id, offset=0, length=8) == (values.tobytes()[:8])
        )
