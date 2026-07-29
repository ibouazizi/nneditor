"""Unit regression tests for the verified ONNX adapter defect fixes."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto

from nneditor.adapters.onnx import index_model, read_tensor_bytes
from nneditor.adapters.onnx.exporter import ExportError, report_from_json
from nneditor.adapters.onnx.numerical import (
    _assign_process_to_job,
    _close_job_object,
    _create_job_object,
)
from nneditor.adapters.onnx.splice import _atomic_write
from nneditor.adapters.onnx.typed_data import materialize_typed_tensor
from nneditor.adapters.onnx.wire import MalformedProtobufError, MessageCursor
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.core import PayloadRange, Storage, TensorRef
from nneditor.storage.reader import ArtifactReader
from nneditor.storage.store import TensorUnavailableError
from tests.fixtures.onnx_models import build_tensor_only_model
from tests.fixtures.protobuf import length_field, varint, varint_field

# TensorProto field numbers used to hand-encode typed tensors; see onnx.proto.
_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_FLOAT_DATA = 4
_TENSOR_INT32_DATA = 5


class _MemorySource:
    """An in-memory RangeSource for wire-level tests."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.size = len(data)

    def read(self, offset: int, length: int, *, record: bool = True) -> bytes:
        return self._data[offset : offset + length]

    def record_logical(self, offset: int, length: int) -> None:
        return None


def _cursor(data: bytes) -> MessageCursor:
    return MessageCursor(_MemorySource(data), 0, len(data))


# --------------------------------------------------------------------------
# Finding 5: packed_varints must reject values beyond 64 bits
# --------------------------------------------------------------------------


def test_packed_varints_reject_values_beyond_64_bits() -> None:
    # Ten bytes FF*9 7F decode to a 70-bit integer; strict decoding refuses it
    # exactly like _read_varint does for standalone fields.
    reader = _cursor(length_field(1, b"\xff" * 9 + b"\x7f"))
    with pytest.raises(MalformedProtobufError, match="exceeds 64 bits"):
        reader.packed_varints(next(iter(reader)))


def test_packed_varints_still_accept_the_full_unsigned_range() -> None:
    reader = _cursor(length_field(1, varint((1 << 64) - 1)))
    assert reader.packed_varints(next(iter(reader))) == (-1,)


# --------------------------------------------------------------------------
# Finding 1: typed-tensor materialization scales with the declared tensor
# --------------------------------------------------------------------------


def _typed_ref(element_type: str, dims: tuple[int, ...], size: int) -> TensorRef:
    return TensorRef(
        id="t:g:main#w",
        element_type=element_type,
        dims=dims,
        storage=Storage.EMBEDDED_TYPED,
        typed_span=PayloadRange(0, size),
    )


def test_a_typed_field_larger_than_the_shape_justifies_is_refused(
    tmp_path: Path,
) -> None:
    body = (
        varint_field(_TENSOR_DIMS, 2)
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.INT32)
        + length_field(_TENSOR_INT32_DATA, b"\x00" * ((1 << 20) + 1))
    )
    path = tmp_path / "hostile.bin"
    path.write_bytes(body)
    with ArtifactReader(path) as reader:
        with pytest.raises(TensorUnavailableError, match="justify at most"):
            materialize_typed_tensor(reader, _typed_ref("int32", (2,), len(body)))


def test_a_fixed_typed_field_must_be_a_multiple_of_the_element_width(
    tmp_path: Path,
) -> None:
    body = (
        varint_field(_TENSOR_DIMS, 1)
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.FLOAT)
        + length_field(_TENSOR_FLOAT_DATA, b"\x00" * 6)
    )
    path = tmp_path / "ragged.bin"
    path.write_bytes(body)
    with ArtifactReader(path) as reader:
        with pytest.raises(TensorUnavailableError, match="not a multiple"):
            materialize_typed_tensor(reader, _typed_ref("float32", (1,), len(body)))


def test_packed_typed_varints_decode_including_negative_values(
    tmp_path: Path,
) -> None:
    payload = varint(1) + varint(2) + varint((1 << 64) - 1)
    body = (
        varint_field(_TENSOR_DIMS, 3)
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.INT32)
        + length_field(_TENSOR_INT32_DATA, payload)
    )
    path = tmp_path / "packed.bin"
    path.write_bytes(body)
    with ArtifactReader(path) as reader:
        packed = materialize_typed_tensor(reader, _typed_ref("int32", (3,), len(body)))
    assert packed == struct.pack("<3i", 1, 2, -1)


def test_typed_materialization_checkpoints_a_cancelled_token(
    tmp_path: Path,
) -> None:
    body = (
        varint_field(_TENSOR_DIMS, 2)
        + varint_field(_TENSOR_DATA_TYPE, TensorProto.INT32)
        + length_field(_TENSOR_INT32_DATA, varint(1) + varint(2))
    )
    path = tmp_path / "cancel.bin"
    path.write_bytes(body)
    token = CancellationToken()
    token.cancel()
    with ArtifactReader(path) as reader:
        with pytest.raises(OperationCancelled):
            materialize_typed_tensor(
                reader, _typed_ref("int32", (2,), len(body)), token=token
            )


# --------------------------------------------------------------------------
# Finding 6: read_tensor_bytes chunks past the single-read cap
# --------------------------------------------------------------------------


def test_read_tensor_bytes_chunks_reads_beyond_the_single_read_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = np.arange(4096, dtype=np.float32).tobytes()
    path = tmp_path / "big.onnx"
    build_tensor_only_model(
        path, dims=[4096], raw_bytes=payload, data_type=TensorProto.FLOAT
    )
    index = index_model(path)
    # Shrink the cap so a 16 KiB payload behaves like a >1 GiB one: a single
    # unchunked read would raise RangeReadError.
    monkeypatch.setattr("nneditor.storage.reader.MAX_SINGLE_READ", 4096)
    monkeypatch.setattr("nneditor.adapters.onnx.indexer._TENSOR_READ_CHUNK", 1024)
    assert read_tensor_bytes(index, index.main_graph.initializers[0]) == payload


# --------------------------------------------------------------------------
# Finding 11: _atomic_write only cleans up a partial it created
# --------------------------------------------------------------------------


def test_atomic_write_preserves_a_partial_it_did_not_create(tmp_path: Path) -> None:
    destination = tmp_path / "out.bin"
    partial = tmp_path / "out.bin.partial"
    partial.write_bytes(b"someone else's export in flight")
    with pytest.raises(FileExistsError):
        _atomic_write(destination, b"content")
    assert partial.read_bytes() == b"someone else's export in flight"
    assert not destination.exists()


# --------------------------------------------------------------------------
# Finding 9: report parsing honors the ExportError contract
# --------------------------------------------------------------------------


def test_report_from_json_wraps_unreadable_files(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportError, match="could not be read"):
        report_from_json(path)
    with pytest.raises(ExportError, match="could not be read"):
        report_from_json(tmp_path / "missing.json")


def test_report_from_json_wraps_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"destination": "x"}), encoding="utf-8")
    with pytest.raises(ExportError, match="malformed"):
        report_from_json(path)


def test_report_from_json_still_rejects_non_objects(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ExportError, match="not an object"):
        report_from_json(path)


# --------------------------------------------------------------------------
# Finding 4: Windows Job Object caps for the smoke worker
# --------------------------------------------------------------------------


def _spawn_python(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", code])


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects only")
def test_job_object_configures_and_kills_on_close() -> None:
    process = _spawn_python("import time; time.sleep(30)")
    try:
        job = _create_job_object()
        assert job is not None
        handle = int(getattr(process, "_handle", 0))
        assert _assign_process_to_job(job, handle)
        _close_job_object(job)
        # Kill-on-job-close reaps the worker as soon as the handle closes.
        process.wait(timeout=15)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects only")
def test_job_object_memory_cap_stops_a_greedy_worker() -> None:
    process = _spawn_python(
        "import time; time.sleep(1.0); data = bytearray(512 * 1024 * 1024); "
        "time.sleep(30)"
    )
    job = None
    try:
        job = _create_job_object(128 * 1024 * 1024)
        assert job is not None
        assert _assign_process_to_job(job, int(getattr(process, "_handle", 0)))
        assert process.wait(timeout=30) != 0
    finally:
        _close_job_object(job)
        if process.poll() is None:
            process.kill()


def test_job_object_helpers_are_null_off_windows() -> None:
    if os.name == "nt":
        pytest.skip("POSIX degradation path")
    assert _create_job_object() is None
    assert _assign_process_to_job(1, 1) is False
    _close_job_object(None)
