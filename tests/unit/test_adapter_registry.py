"""Artifact adapter registry and application-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.adapters.registry import (
    ArtifactAdapterError,
    ArtifactAdapterRegistry,
)
from nneditor.application.session import ApplicationService, SessionError
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document
from tests.fixtures.onnx_models import build_embedded_model


@dataclass
class RecordingAdapter:
    document: Document
    kind: ArtifactKind = ArtifactKind.ONNX_MODEL
    opened: int = 0

    def open(
        self,
        path: Path,
        *,
        token: CancellationToken | None = None,
    ) -> Document:
        assert path.exists()
        if token is not None:
            token.raise_if_cancelled()
        self.opened += 1
        return self.document


def test_application_uses_an_injected_artifact_adapter_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    document = index_to_document(index_model(path))
    adapter = RecordingAdapter(document)
    registry = ArtifactAdapterRegistry((adapter,))

    with ApplicationService(adapter_registry=registry) as service:
        session = service.open_model(path)

    assert session.document is document
    assert adapter.opened == 1


def test_registry_rejects_duplicates_and_missing_readers(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    document = index_to_document(index_model(path))
    adapter = RecordingAdapter(document)
    with pytest.raises(ValueError, match="registered twice"):
        ArtifactAdapterRegistry((adapter, adapter))

    empty = ArtifactAdapterRegistry(())
    with pytest.raises(ArtifactAdapterError, match="no reader"):
        empty.open(path)


def test_registry_honors_cancellation_before_adapter_work(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    document = index_to_document(index_model(path))
    adapter = RecordingAdapter(document)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        ArtifactAdapterRegistry((adapter,)).open(path, token=token)
    assert adapter.opened == 0


def test_malformed_onnx_is_wrapped_as_a_session_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.onnx"
    path.write_bytes(b"\x08\x01\xff")
    with ApplicationService() as service:
        with pytest.raises(SessionError, match="could not be opened as onnx_model"):
            service.open_model(path)
