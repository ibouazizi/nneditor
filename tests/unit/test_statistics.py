"""Tests for streaming tensor statistics (P3.3).

Float results are checked differentially against numpy, which stays a
test-only dependency: the product streams in pure Python, the test recomputes
with the reference stack.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from nneditor.analysis.statistics import (
    STATISTICS_VERSION,
    compute_statistics,
    statistics_from_json,
    statistics_to_json,
)
from nneditor.application.session import ApplicationService, SessionError
from nneditor.application.statistics_store import StatisticsSidecarStore
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.storage.store import TensorStore, TensorUnavailableError
from tests.fixtures.onnx_models import (
    build_embedded_model,
    build_external_model,
    build_tensor_only_model,
)
from tests.unit.test_tensor_store import embedded_document, handcrafted_document


def store_for_raw(
    tmp_path: Path, raw: bytes, *, dims: list[int], data_type: int
) -> tuple[TensorStore, str]:
    path = tmp_path / "model.onnx"
    build_tensor_only_model(path, dims=dims, raw_bytes=raw, data_type=data_type)
    from nneditor.adapters.onnx import index_model, index_to_document

    document = index_to_document(index_model(path))
    tensor_id = document.main_graph.initializers[0]
    return TensorStore(document), tensor_id


class TestFloatStatistics:
    def test_matches_numpy_reference(self, tmp_path: Path) -> None:
        values = np.random.default_rng(3).standard_normal(4096).astype(np.float32)
        store, tensor_id = store_for_raw(
            tmp_path, values.tobytes(), dims=[4096], data_type=1
        )
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.element_count == 4096
        assert stats.minimum == pytest.approx(float(values.min()))
        assert stats.maximum == pytest.approx(float(values.max()))
        assert stats.mean == pytest.approx(float(values.mean()), rel=1e-5)
        assert stats.std == pytest.approx(float(values.std()), rel=1e-4)
        assert stats.zero_count == int((values == 0).sum())
        assert sum(stats.histogram_counts) == 4096
        assert len(stats.histogram_edges) == len(stats.histogram_counts) + 1
        numpy_counts, _ = np.histogram(
            values.astype(np.float64),
            bins=len(stats.histogram_counts),
            range=(float(values.min()), float(values.max())),
        )
        assert list(stats.histogram_counts) == numpy_counts.tolist()

    def test_non_finite_values_are_counted_not_propagated(self, tmp_path: Path) -> None:
        values = np.array(
            [1.0, float("nan"), float("inf"), -float("inf"), 0.0, 3.0],
            dtype=np.float32,
        )
        store, tensor_id = store_for_raw(
            tmp_path, values.tobytes(), dims=[6], data_type=1
        )
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.nan_count == 1
        assert stats.inf_count == 2
        assert stats.non_finite_count == 3
        assert stats.minimum == 0.0
        assert stats.maximum == 3.0
        assert stats.mean == pytest.approx(4.0 / 3.0)
        assert sum(stats.histogram_counts) == 3, "only finite values are binned"

    def test_all_non_finite_yields_no_extrema(self, tmp_path: Path) -> None:
        values = np.array([float("nan")] * 4, dtype=np.float32)
        store, tensor_id = store_for_raw(
            tmp_path, values.tobytes(), dims=[4], data_type=1
        )
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.minimum is None and stats.maximum is None
        assert stats.mean is None and stats.std is None
        assert stats.histogram_counts == ()

    def test_constant_tensor_gets_a_single_bin(self, tmp_path: Path) -> None:
        raw = struct.pack("<4f", 2.5, 2.5, 2.5, 2.5)
        store, tensor_id = store_for_raw(tmp_path, raw, dims=[4], data_type=1)
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.histogram_edges == (2.5, 2.5)
        assert stats.histogram_counts == (4,)


class TestIntegerStatistics:
    def test_int64_sparsity_and_extrema(self, tmp_path: Path) -> None:
        values = np.array([0, 0, -5, 10, 0, 3], dtype=np.int64)
        store, tensor_id = store_for_raw(
            tmp_path, values.tobytes(), dims=[6], data_type=7
        )
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.minimum == -5.0 and stats.maximum == 10.0
        assert stats.zero_count == 3
        assert stats.sparsity == pytest.approx(0.5)
        assert stats.nan_count == 0 and stats.inf_count == 0

    def test_uint8_values(self, tmp_path: Path) -> None:
        raw = bytes([0, 255, 128, 0])
        store, tensor_id = store_for_raw(tmp_path, raw, dims=[4], data_type=2)
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.minimum == 0.0 and stats.maximum == 255.0
        assert stats.zero_count == 2


class TestFailureModes:
    def test_unsupported_dtype_is_reported(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        object.__setattr__(document.tensors[weight_id], "element_type", "complex64")
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="not supported"):
                compute_statistics(store, weight_id)

    def test_absent_payloads_are_reported(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"")
        from nneditor.ir.core import Storage, TensorRef

        document = handcrafted_document(
            model_path,
            [
                TensorRef(
                    id="t:absent",
                    element_type="float32",
                    dims=(4,),
                    storage=Storage.ABSENT,
                )
            ],
        )
        with TensorStore(document) as store:
            with pytest.raises(TensorUnavailableError, match="no readable payload"):
                compute_statistics(store, "t:absent")

    def test_invalid_bins_are_rejected(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        with TensorStore(document) as store:
            with pytest.raises(ValueError, match="positive"):
                compute_statistics(store, document.main_graph.initializers[0], bins=0)

    def test_cancellation_stops_the_stream(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        token = CancellationToken()
        token.cancel()
        with TensorStore(document) as store:
            with pytest.raises(OperationCancelled):
                compute_statistics(
                    store, document.main_graph.initializers[0], token=token
                )


class TestPersistenceAndSession:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            stats = compute_statistics(store, weight_id)
        restored = statistics_from_json(statistics_to_json(stats))
        assert restored == stats

    def test_stale_versions_are_discarded(self) -> None:
        data = {"version": STATISTICS_VERSION + 1, "tensor_id": "t"}
        assert statistics_from_json(data) is None
        assert statistics_from_json({"garbage": True}) is None

    def test_sidecar_survives_reopen_and_ignores_corruption(
        self, tmp_path: Path
    ) -> None:
        document, _values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        sidecar = StatisticsSidecarStore(tmp_path / "state")
        with TensorStore(document) as store:
            stats = compute_statistics(store, weight_id)
        sidecar.save(document.source.content_hash, {weight_id: stats})
        assert sidecar.load(document.source.content_hash) == {weight_id: stats}

        target = sidecar._path_for(document.source.content_hash)
        payload = target.read_text(encoding="utf-8")
        target.write_text(
            payload.replace(document.source.content_hash, "sha256:wrong"),
            encoding="utf-8",
        )
        assert sidecar.load(document.source.content_hash) == {}
        target.write_text(payload, encoding="utf-8")
        target.write_text("{broken", encoding="utf-8")
        assert sidecar.load(document.source.content_hash) == {}

    def test_session_computes_persists_and_reuses(self, tmp_path: Path) -> None:
        from nneditor.application.persistence import SessionStateStore

        model = tmp_path / "model.onnx"
        build_embedded_model(model, elements=64)
        state_dir = tmp_path / "state"

        with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
            session = service.open_model(model)
            weight_id = session.document.main_graph.initializers[0]
            assert session.statistics(weight_id) is None, "no compute yet"
            job = session.compute_statistics_async(weight_id)
            stats = job.result(timeout=30)
            assert stats.element_count == 64
            assert session.statistics(weight_id) == stats

        # A fresh service and session reuse the persisted result untouched.
        with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
            session = service.open_model(model)
            weight_id = session.document.main_graph.initializers[0]
            assert session.statistics(weight_id) is not None

    def test_external_change_cannot_reuse_persisted_statistics(
        self, tmp_path: Path
    ) -> None:
        from nneditor.application.persistence import SessionStateStore

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model, _values = build_external_model(model_dir, elements=16)
        state_dir = tmp_path / "state"
        with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
            original = service.open_model(model)
            tensor_id = original.document.main_graph.initializers[0]
            original_hash = original.document.source.content_hash
            original.compute_statistics(tensor_id)

        (model_dir / "weights.bin").write_bytes(b"\xff" * 64)
        with ApplicationService(state_store=SessionStateStore(state_dir)) as service:
            changed = service.open_model(model)
            tensor_id = changed.document.main_graph.initializers[0]
            assert changed.document.source.content_hash != original_hash
            assert changed.statistics(tensor_id) is None

    def test_session_without_jobs_rejects_async(self, tmp_path: Path) -> None:
        from nneditor.application.hierarchy import HierarchyController
        from nneditor.application.session import ModelSession
        from nneditor.application.slices import GraphSlicer

        document, _values = embedded_document(tmp_path)
        session = ModelSession(
            1, document, GraphSlicer(), HierarchyController(document)
        )
        with pytest.raises(SessionError, match="job manager"):
            session.compute_statistics_async("t")
