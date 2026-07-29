"""Tests for the tensor store (P1.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.adapters.pytorch.safetensors import open_safetensors
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
)
from nneditor.ir.core import (
    ArtifactRef,
    Document,
    ExternalRef,
    Graph,
    Storage,
    TensorRef,
)
from nneditor.storage.store import TensorStore, TensorUnavailableError
from tests.fixtures.onnx_models import build_embedded_model, build_external_model
from tests.fixtures.pytorch_models import build_safetensors_file

ELEMENTS = 256


def embedded_document(tmp_path: Path) -> tuple[Document, np.ndarray]:
    path = tmp_path / "model.onnx"
    values = build_embedded_model(path, elements=ELEMENTS)
    return index_to_document(index_model(path)), values


def handcrafted_document(path: Path, tensors: list[TensorRef]) -> Document:
    return Document(
        source=ArtifactRef(path=str(path), content_hash="sha256:00", byte_size=0),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=[
            CapabilityStatus(capability, Availability.AVAILABLE, "test")
            for capability in Capability
        ],
        graphs=[Graph(id="g:main", name="main")],
        tensors=tensors,
    )


class TestMetadataAccess:
    def test_metadata_never_opens_a_file(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        with TensorStore(document) as store:
            weight_id = document.main_graph.initializers[0]
            assert store.metadata(weight_id).element_type == "float32"
            assert store.readable(weight_id)
            assert store.byte_length(weight_id) == 4 * ELEMENTS
            assert store.open_file_count == 0

    def test_unknown_tensors_are_reported(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="not in the document"):
                store.metadata("t:ghost")

    def test_typed_storage_is_disclosed_and_materializable(
        self, tmp_path: Path
    ) -> None:
        import struct

        from nneditor.storage.store import Materialization

        document, _ = embedded_document(tmp_path)
        bias_id = document.main_graph.initializers[1]
        with TensorStore(document) as store:
            assert store.materialization(bias_id) is Materialization.FULL_PARSE
            assert store.readable(bias_id)
            assert store.byte_length(bias_id) == 4, "one float32"
            assert store.read(bias_id) == struct.pack("<f", 0.5)


class TestReads:
    def test_full_and_ranged_reads(self, tmp_path: Path) -> None:
        document, values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            assert store.read(weight_id) == values.tobytes()
            assert store.read(weight_id, offset=4, length=8) == values.tobytes()[4:12]
            assert store.read(weight_id, offset=4 * ELEMENTS) == b""
            assert store.open_file_count == 1

    def test_external_tensors_read_from_their_file(self, tmp_path: Path) -> None:
        path, values = build_external_model(tmp_path, elements=ELEMENTS)
        document = index_to_document(index_model(path))
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            assert store.read(weight_id) == values.tobytes()

    def test_changed_external_payload_is_rejected_before_read(
        self, tmp_path: Path
    ) -> None:
        path, _values = build_external_model(tmp_path, elements=ELEMENTS)
        document = index_to_document(index_model(path))
        weight_id = document.main_graph.initializers[0]
        (tmp_path / "weights.bin").write_bytes(b"\xff" * (4 * ELEMENTS))

        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="changed after"):
                store.read(weight_id, offset=0, length=4)
            assert store.open_file_count == 0

    def test_out_of_payload_ranges_are_rejected(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="exceeds"):
                store.read(weight_id, offset=4 * ELEMENTS - 2, length=4)
            with pytest.raises(TensorUnavailableError, match="negative"):
                store.read(weight_id, offset=-1)

    def test_traversal_locations_are_refused(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret.bin"
        secret.write_bytes(b"top secret")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        model_path.write_bytes(b"")
        document = handcrafted_document(
            model_path,
            [
                TensorRef(
                    id="t:evil",
                    element_type="uint8",
                    dims=(10,),
                    storage=Storage.EXTERNAL,
                    external=ExternalRef(location="../secret.bin", offset=0, length=10),
                )
            ],
        )
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="unsafe"):
                store.read("t:evil")

    def test_missing_external_file_is_reported(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"")
        document = handcrafted_document(
            model_path,
            [
                TensorRef(
                    id="t:gone",
                    element_type="uint8",
                    dims=(4,),
                    storage=Storage.EXTERNAL,
                    external=ExternalRef(location="gone.bin", offset=0, length=4),
                )
            ],
        )
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="cannot open"):
                store.read("t:gone")

    def test_declared_length_beyond_file_is_reported(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"")
        (tmp_path / "small.bin").write_bytes(b"1234")
        document = handcrafted_document(
            model_path,
            [
                TensorRef(
                    id="t:liar",
                    element_type="uint8",
                    dims=(100,),
                    storage=Storage.EXTERNAL,
                    external=ExternalRef(location="small.bin", offset=0, length=100),
                )
            ],
        )
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="past the end"):
                store.read("t:liar")


class TestCancellation:
    def test_a_cancelled_token_stops_before_any_read(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        token = CancellationToken()
        token.cancel()
        with TensorStore(document) as store:
            with pytest.raises(OperationCancelled):
                store.read(weight_id, token=token)
            assert store.open_file_count == 0

    def test_an_uncancelled_token_does_not_interfere(self, tmp_path: Path) -> None:
        document, values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            token = CancellationToken()
            assert store.read(weight_id, token=token) == values.tobytes()
            assert not token.cancelled


class TestCacheAndLifecycle:
    def test_non_onnx_source_swap_is_detected_before_read(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "weights.safetensors"
        build_safetensors_file(path)
        document = open_safetensors(path)
        tensor_id = next(iter(document.tensors))
        changed = bytearray(path.read_bytes())
        changed[-1] ^= 0xFF
        path.write_bytes(changed)

        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="changed after"):
                store.read(tensor_id)

    def test_repeat_reads_hit_the_cache(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            first = store.read(weight_id)
            second = store.read(weight_id)
            assert first == second
            assert store.stats.hits == 1
            assert store.stats.misses == 1

    def test_the_cache_budget_evicts_least_recent(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document, cache_budget=64) as store:
            store.read(weight_id, offset=0, length=48)
            store.read(weight_id, offset=48, length=48)
            assert store.stats.evictions == 1
            store.read(weight_id, offset=0, length=48)
            assert store.stats.misses == 3, "the first range was evicted"

    def test_oversized_results_bypass_the_cache(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document, cache_budget=8) as store:
            store.read(weight_id)
            store.read(weight_id)
            assert store.stats.hits == 0
            assert store.stats.evictions == 0

    def test_close_releases_files_and_refuses_reads(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        store = TensorStore(document)
        store.read(weight_id)
        assert store.open_file_count == 1
        store.close()
        assert store.closed
        assert store.open_file_count == 0
        with pytest.raises(TensorUnavailableError, match="closed"):
            store.read(weight_id)

    def test_negative_budget_is_rejected(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            TensorStore(document, cache_budget=-1)
