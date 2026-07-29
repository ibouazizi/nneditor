"""Regression tests for verified storage and IR defect fixes.

Each test class pins one finding:

* ReadStats growth is bounded while byte totals stay exact.
* Component verification hashes through the reader's own descriptor, outside
  the store-wide lock, honoring cancellation.
* A concurrent truncation surfaces as an error, never as short data.
* Cache misses are re-checked under the lock, so racing threads never repeat
  a physical read or a full typed parse.
* An oversized typed materialization is still a one-time cost.
* Replacing a cache entry with an unretainable value counts as an eviction.
* Identifier segment encoding is injective across astral code points and its
  decoder refuses malformed escapes with a clear error.
* A degraded read merges into an existing ``x-nneditor.unknown`` namespace.
* Malformed documents are refused with ``IrError``, not raw exceptions.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nneditor.adapters.onnx import index_model, index_to_document
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
    Graph,
    IrError,
    PayloadRange,
    Storage,
    TensorRef,
)
from nneditor.ir.identity import decode_segment, encode_segment
from nneditor.ir.schema import UNKNOWN_FIELD_NAMESPACE
from nneditor.ir.serialize import (
    document_from_json,
    document_to_bytes,
)
from nneditor.storage.cache import BudgetedCache
from nneditor.storage.reader import (
    ArtifactReader,
    ByteRange,
    RangeReadError,
    ReadStats,
    hash_file,
)
from nneditor.storage.store import TensorStore
from tests.fixtures.onnx_models import build_embedded_model, build_external_model

ELEMENTS = 256


def embedded_document(tmp_path: Path) -> tuple[Document, np.ndarray]:
    path = tmp_path / "model.onnx"
    values = build_embedded_model(path, elements=ELEMENTS)
    return index_to_document(index_model(path)), values


class TestReadStatsBounded:
    """Finding 1: instrumentation must not leak for the life of a session."""

    def test_lists_are_capped_with_an_overflow_counter(self) -> None:
        stats = ReadStats(max_recorded_ranges=2)
        for index in range(5):
            stats.record_logical(ByteRange(index * 10, 10))
        assert len(stats.logical_ranges) == 2
        assert stats.dropped_ranges == 3
        assert stats.logical_bytes == 50, "byte totals stay exact past the cap"

    def test_physical_ranges_are_capped_too(self) -> None:
        stats = ReadStats(max_recorded_ranges=1)
        stats.record_physical(ByteRange(0, 64))
        stats.record_physical(ByteRange(64, 64))
        assert len(stats.physical_ranges) == 1
        assert stats.dropped_ranges == 1
        assert stats.physical_bytes == 128

    def test_reset_clears_the_overflow_state(self) -> None:
        stats = ReadStats(max_recorded_ranges=1)
        stats.record_logical(ByteRange(0, 4))
        stats.record_logical(ByteRange(4, 4))
        stats.reset()
        assert stats.dropped_ranges == 0
        assert stats.logical_bytes == 0
        stats.record_logical(ByteRange(0, 8))
        assert stats.logical_bytes == 8

    def test_reader_recording_respects_the_cap(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.bin"
        path.write_bytes(bytes(1000))
        with ArtifactReader(path, block_size=64) as reader:
            reader.stats.max_recorded_ranges = 3
            for index in range(10):
                reader.read(index * 64, 64)
            assert len(reader.stats.logical_ranges) == 3
            assert len(reader.stats.physical_ranges) == 3
            assert reader.stats.dropped_ranges == 14
            assert reader.stats.logical_bytes == 640
            assert reader.stats.physical_bytes == 640


class TestHashThroughTheReader:
    """Finding 2: verify through the open descriptor, cancellably."""

    def test_content_hash_matches_hash_file(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(range(256)) * 8)
        with ArtifactReader(path) as reader:
            assert reader.content_hash() == hash_file(path)
            assert reader.stats.logical_ranges == [], "hashing is not parsing"
            assert reader.stats.physical_ranges == []

    def test_content_hash_honors_cancellation(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(1024))
        token = CancellationToken()
        token.cancel()
        with ArtifactReader(path) as reader:
            with pytest.raises(OperationCancelled):
                reader.content_hash(token=token)

    def test_content_hash_refuses_a_closed_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(16))
        reader = ArtifactReader(path)
        reader.close()
        with pytest.raises(RangeReadError, match="closed"):
            reader.content_hash()

    def test_cancelling_during_verification_leaves_no_open_file(
        self, tmp_path: Path
    ) -> None:
        """A token cancelled mid-hash aborts cleanly and a retry recovers."""

        class TriggeredToken(CancellationToken):
            def __init__(self, after: int) -> None:
                super().__init__()
                self.checkpoints = 0
                self._after = after

            def raise_if_cancelled(self) -> None:
                self.checkpoints += 1
                if self.checkpoints > self._after:
                    self.cancel()
                super().raise_if_cancelled()

        path, values = build_external_model(tmp_path, elements=ELEMENTS)
        document = index_to_document(index_model(path))
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            token = TriggeredToken(after=1)
            with pytest.raises(OperationCancelled):
                store.read(weight_id, token=token)
            assert token.checkpoints > 1, "cancellation fired inside hashing"
            assert store.open_file_count == 0
            assert store.read(weight_id) == values.tobytes(), (
                "an aborted verification must not poison later reads"
            )


class TestShortReads:
    """Finding 3: truncation must raise, never return short data."""

    def test_truncated_file_raises_instead_of_returning_short_data(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "shrink.bin"
        path.write_bytes(bytes(8192))
        reader = ArtifactReader(path)
        try:
            with open(path, "r+b") as handle:
                handle.truncate(100)
            with pytest.raises(RangeReadError, match="short read"):
                reader.read(0, 4096)
        finally:
            reader.close()


class TestCacheRecheckUnderTheLock:
    """Finding 4: two racing misses must not both do the physical work."""

    def test_slice_read_rechecks_the_cache_under_the_lock(self, tmp_path: Path) -> None:
        document, _ = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        store = TensorStore(document)
        try:
            lock = store._path_lock(store._source_resolved)
            started = threading.Event()
            results: list[bytes] = []

            def worker() -> None:
                started.set()
                results.append(store.read(weight_id, offset=0, length=16))

            lock.acquire()
            try:
                thread = threading.Thread(target=worker)
                thread.start()
                assert started.wait(timeout=5)
                time.sleep(0.1)  # let the worker miss the cache and block
                store._slice_cache.put((weight_id, 0, 16), b"seeded-by-winner")
            finally:
                lock.release()
            thread.join(timeout=5)
            assert results == [b"seeded-by-winner"]
            assert store.open_file_count == 0, (
                "the re-check under the lock must skip the physical read"
            )
        finally:
            store.close()

    def test_concurrent_typed_reads_parse_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nneditor.adapters.onnx import typed_data

        document, _ = embedded_document(tmp_path)
        bias_id = document.main_graph.initializers[1]
        real = typed_data.materialize_typed_tensor
        calls: list[str] = []

        def counting(*args: Any, **kwargs: Any) -> bytes:
            calls.append("parse")
            time.sleep(0.05)
            return real(*args, **kwargs)

        monkeypatch.setattr(typed_data, "materialize_typed_tensor", counting)
        barrier = threading.Barrier(2)
        results: list[bytes] = []
        errors: list[BaseException] = []

        def worker(store: TensorStore) -> None:
            try:
                barrier.wait(timeout=5)
                results.append(store.read(bias_id))
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        with TensorStore(document) as store:
            threads = [threading.Thread(target=worker, args=(store,)) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        assert not errors
        assert results[0] == results[1] == struct.pack("<f", 0.5)
        assert len(calls) == 1, "the losing thread must reuse the winner's parse"


class TestPeek:
    """Support for finding 4: the re-check must not inflate the miss rate."""

    def test_peek_does_not_touch_the_counters(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 100, cost=len)
        assert cache.peek("a") is None
        cache.put("a", b"1234")
        assert cache.peek("a") == b"1234"
        snapshot = cache.snapshot()
        assert (snapshot.hits, snapshot.misses) == (0, 0)

    def test_peek_still_refreshes_recency(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 10, cost=len)
        cache.put("a", b"aaaa")
        cache.put("b", b"bbbb")
        assert cache.peek("a") is not None, "touch a so b is least recent"
        cache.put("c", b"cccc")
        assert cache.peek("b") is None, "b was evicted"
        assert cache.peek("a") is not None


class TestOversizedTypedMaterialization:
    """Finding 5: FULL_PARSE promises a one-time cost; keep it honest."""

    def test_oversized_materialization_is_still_parsed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nneditor.adapters.onnx import typed_data

        document, _ = embedded_document(tmp_path)
        bias_id = document.main_graph.initializers[1]  # 4 packed bytes
        real = typed_data.materialize_typed_tensor
        calls: list[str] = []

        def counting(*args: Any, **kwargs: Any) -> bytes:
            calls.append("parse")
            return real(*args, **kwargs)

        monkeypatch.setattr(typed_data, "materialize_typed_tensor", counting)
        with TensorStore(document, typed_budget=2) as store:
            first = store.read(bias_id)
            second = store.read(bias_id)
            snapshots = {item.name: item for item in store.cache_snapshots()}
        assert first == second == struct.pack("<f", 0.5)
        assert len(calls) == 1, "an oversized tensor must not re-parse per slice"
        typed = snapshots["typed-materializations"]
        assert typed.entry_count == 1
        assert typed.current_cost == 4
        assert typed.current_cost > typed.budget, "retained although oversized"
        assert typed.hits >= 1


class TestCacheEvictionAccounting:
    """Finding 6 (and the cache half of 5)."""

    def test_replacing_with_an_unretainable_value_counts_an_eviction(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 4, cost=len)
        cache.put("k", b"12")
        cache.put("k", b"123456")
        assert cache.get("k") is None
        assert len(cache) == 0
        assert cache.current_cost == 0
        assert cache.snapshot().evictions == 1, (
            "the vanished entry must be visible in the counters"
        )

    def test_oversized_put_on_a_fresh_key_still_counts_no_eviction(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache("t", 4, cost=len)
        cache.put("big", b"123456")
        assert len(cache) == 0
        assert cache.snapshot().evictions == 0

    def test_admit_oversized_retains_a_single_oversized_entry(self) -> None:
        cache: BudgetedCache[str, bytes] = BudgetedCache(
            "t", 4, cost=len, admit_oversized=True
        )
        cache.put("small", b"12")
        cache.put("big", b"123456")
        assert cache.get("big") == b"123456"
        assert cache.get("small") is None, "displaced by the oversized entry"
        assert cache.snapshot().evictions == 1
        assert cache.current_cost == 6
        cache.put("bigger", b"1234567")
        assert cache.get("big") is None
        assert cache.get("bigger") == b"1234567"
        assert len(cache) == 1


class TestIdentityEscapes:
    """Finding 7: astral code points and strict decoding."""

    def test_astral_escape_round_trips(self) -> None:
        hostile = "a\U000e0001b"  # U+E0001 LANGUAGE TAG, non-printable astral
        encoded = encode_segment(hostile)
        assert encoded == "a%U000E0001b"
        assert decode_segment(encoded) == hostile

    def test_astral_and_bmp_escapes_do_not_collide(self) -> None:
        # chr(0xE000) + "1" used to encode identically to chr(0xE0001).
        assert encode_segment(chr(0xE000) + "1") != encode_segment(chr(0xE0001))

    def test_bmp_escapes_keep_their_legacy_spelling(self) -> None:
        """Existing identifiers must stay byte-for-byte stable."""
        assert encode_segment("a\u2028b") == "a%u2028b"
        assert encode_segment("a\nb") == "a%0Ab"
        assert encode_segment("a%2Fb") == "a%252Fb"

    @pytest.mark.parametrize(
        "raw",
        [
            "plain",
            "with/slash",
            "a\U000e0001b",
            chr(0xE000) + "1",
            "emoji \U0001f642",
            "a\x00b",
        ],
    )
    def test_round_trip_including_astral_input(self, raw: str) -> None:
        assert decode_segment(encode_segment(raw)) == raw

    @pytest.mark.parametrize(
        "malformed",
        ["%", "a%", "%1", "%z1", "%u12", "%u123z", "%U000E001", "%UFFFFFFFF"],
    )
    def test_malformed_escapes_raise_a_clear_error(self, malformed: str) -> None:
        with pytest.raises(ValueError, match="malformed identifier segment"):
            decode_segment(malformed)


def _full_capabilities() -> list[CapabilityStatus]:
    return [
        CapabilityStatus(capability, Availability.PARTIAL, "fixture reason")
        for capability in Capability
    ]


def _minimal_document(extensions: list[tuple[str, Any]] | None = None) -> Document:
    return Document(
        source=ArtifactRef(path="m.onnx", content_hash="sha256:00", byte_size=1),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=_full_capabilities(),
        graphs=[Graph(id="g:main", name="main")],
        tensors=[
            TensorRef(
                id="t:w",
                element_type="float32",
                dims=(1,),
                storage=Storage.EMBEDDED_RAW,
                payload=PayloadRange(offset=0, length=4),
            )
        ],
        extensions=extensions or [],
    )


def _document_json(document: Document) -> Any:
    return json.loads(document_to_bytes(document))


class TestDegradedReadMergesUnknownNamespace:
    """Finding 8: a pre-existing parked namespace must be merged, not doubled."""

    def test_merge_into_the_existing_namespace(self) -> None:
        document = _minimal_document(
            extensions=[(UNKNOWN_FIELD_NAMESPACE, {"old_field": 1})]
        )
        data = _document_json(document)
        data["schema_version"] = "1.99"
        data["novel_field"] = {"future": True}
        restored = document_from_json(data)
        parked = dict(restored.extensions)[UNKNOWN_FIELD_NAMESPACE]
        assert parked == {"old_field": 1, "novel_field": {"future": True}}
        namespaces = [namespace for namespace, _ in restored.extensions]
        assert namespaces.count(UNKNOWN_FIELD_NAMESPACE) == 1

    def test_freshly_read_top_level_values_win_a_collision(self) -> None:
        document = _minimal_document(
            extensions=[(UNKNOWN_FIELD_NAMESPACE, {"clash": "old"})]
        )
        data = _document_json(document)
        data["schema_version"] = "1.99"
        data["clash"] = "new"
        restored = document_from_json(data)
        parked = dict(restored.extensions)[UNKNOWN_FIELD_NAMESPACE]
        assert parked == {"clash": "new"}

    def test_a_document_without_the_namespace_still_parks_fields(self) -> None:
        data = _document_json(_minimal_document())
        data["schema_version"] = "1.99"
        data["novel_field"] = 7
        restored = document_from_json(data)
        assert dict(restored.extensions)[UNKNOWN_FIELD_NAMESPACE] == {"novel_field": 7}


class TestMalformedDocumentsAreRefused:
    """Finding 9: bad documents raise IrError with a reason, never KeyError."""

    @staticmethod
    def _payload_without_offset(data: Any) -> None:
        data["tensors"][0]["payload"] = {"length": 4}

    @staticmethod
    def _payload_with_string_offset(data: Any) -> None:
        data["tensors"][0]["payload"] = {"offset": "0", "length": 4}

    @staticmethod
    def _payload_not_an_object(data: Any) -> None:
        data["tensors"][0]["payload"] = 7

    @staticmethod
    def _payload_with_boolean_length(data: Any) -> None:
        data["tensors"][0]["payload"] = {"offset": 0, "length": True}

    @staticmethod
    def _external_without_offset(data: Any) -> None:
        data["tensors"][0] = {
            "id": "t:w",
            "element_type": "float32",
            "dims": [1],
            "storage": "external",
            "external": {"location": "w.bin"},
        }

    @staticmethod
    def _external_with_string_length(data: Any) -> None:
        data["tensors"][0] = {
            "id": "t:w",
            "element_type": "float32",
            "dims": [1],
            "storage": "external",
            "external": {"location": "w.bin", "offset": 0, "length": "4"},
        }

    @staticmethod
    def _typed_span_with_boolean_offset(data: Any) -> None:
        data["tensors"][0] = {
            "id": "t:w",
            "element_type": "float32",
            "dims": [1],
            "storage": "embedded_typed",
            "typed_span": {"offset": True, "length": 4},
        }

    @staticmethod
    def _extensions_as_a_list(data: Any) -> None:
        data["extensions"] = []

    @staticmethod
    def _source_as_a_list(data: Any) -> None:
        data["source"] = []

    @staticmethod
    def _source_with_string_byte_size(data: Any) -> None:
        data["source"]["byte_size"] = "big"

    @pytest.mark.parametrize(
        "mutate",
        [
            _payload_without_offset,
            _payload_with_string_offset,
            _payload_not_an_object,
            _payload_with_boolean_length,
            _external_without_offset,
            _external_with_string_length,
            _typed_span_with_boolean_offset,
            _extensions_as_a_list,
            _source_as_a_list,
            _source_with_string_byte_size,
        ],
    )
    def test_each_malformation_raises_ir_error(self, mutate: Any) -> None:
        data = _document_json(_minimal_document())
        mutate.__func__(data)
        with pytest.raises(IrError):
            document_from_json(data)


class TestResolvedOnce:
    """Finding 10: one canonical key per file, resolved at setup."""

    def test_repeat_reads_share_one_reader(self, tmp_path: Path) -> None:
        document, values = embedded_document(tmp_path)
        weight_id = document.main_graph.initializers[0]
        with TensorStore(document) as store:
            assert store._source_resolved == Path(document.source.path).resolve()
            assert store.read(weight_id, offset=0, length=8) == values.tobytes()[:8]
            assert store.read(weight_id, offset=8, length=8) == values.tobytes()[8:16]
            assert store.open_file_count == 1
