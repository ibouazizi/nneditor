"""Tests for typed-tensor materialization and disclosure (P3.2)."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.adapters.onnx import index_model, index_to_document
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.core import Document, Storage
from nneditor.ir.serialize import document_from_bytes, document_to_bytes
from nneditor.storage.store import (
    Materialization,
    TensorStore,
    TensorUnavailableError,
)

_OPSET = 18


def typed_model(
    path: Path, *, data_type: int, dims: list[int], values: Sequence[Any]
) -> None:
    """A one-initializer model whose values live in a typed field."""
    weight = helper.make_tensor(
        name="weight", data_type=data_type, dims=dims, vals=values, raw=False
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["weight"], ["output"], name="passthrough")
        ],
        name="typed",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", data_type, dims)],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)


def open_document(path: Path) -> Document:
    return index_to_document(index_model(path))


@pytest.mark.parametrize(
    ("data_type", "dims", "values", "expected"),
    [
        (
            TensorProto.FLOAT,
            [4],
            [0.5, -1.5, 2.0, 3.25],
            struct.pack("<4f", 0.5, -1.5, 2.0, 3.25),
        ),
        (
            TensorProto.DOUBLE,
            [2],
            [1.25, -2.5],
            struct.pack("<2d", 1.25, -2.5),
        ),
        (
            TensorProto.INT64,
            [3],
            [-1, 2, -(2**40)],
            struct.pack("<3q", -1, 2, -(2**40)),
        ),
        (TensorProto.INT32, [3], [-1, 0, 7], struct.pack("<3i", -1, 0, 7)),
        (TensorProto.INT8, [3], [-8, 0, 127], struct.pack("<3b", -8, 0, 127)),
        (TensorProto.UINT8, [3], [0, 128, 255], struct.pack("<3B", 0, 128, 255)),
        (TensorProto.BOOL, [3], [1, 0, 1], struct.pack("<3B", 1, 0, 1)),
        (TensorProto.UINT64, [2], [1, 2**63], struct.pack("<2Q", 1, 2**63)),
        (TensorProto.UINT32, [2], [1, 2**31], struct.pack("<2I", 1, 2**31)),
        (TensorProto.INT16, [2], [-300, 300], struct.pack("<2h", -300, 300)),
    ],
)
def test_typed_tensors_materialize_to_packed_bytes(
    tmp_path: Path,
    data_type: int,
    dims: list[int],
    values: list[object],
    expected: bytes,
) -> None:
    path = tmp_path / "model.onnx"
    typed_model(path, data_type=data_type, dims=dims, values=values)
    document = open_document(path)
    tensor_id = document.main_graph.initializers[0]
    tensor = document.tensors[tensor_id]
    assert tensor.storage is Storage.EMBEDDED_TYPED
    assert tensor.typed_span is not None
    with TensorStore(document) as store:
        assert store.materialization(tensor_id) is Materialization.FULL_PARSE
        assert store.byte_length(tensor_id) == len(expected)
        assert store.read(tensor_id) == expected


def test_float16_bit_patterns_survive(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    values = np.array([0.5, -2.0, 65504.0], dtype=np.float16)
    typed_model(
        path,
        data_type=TensorProto.FLOAT16,
        dims=[3],
        values=[0.5, -2.0, 65504.0],
    )
    document = open_document(path)
    tensor_id = document.main_graph.initializers[0]
    with TensorStore(document) as store:
        assert store.read(tensor_id) == values.tobytes()


def test_slices_come_from_one_materialization(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    typed_model(
        path, data_type=TensorProto.FLOAT, dims=[4], values=[1.0, 2.0, 3.0, 4.0]
    )
    document = open_document(path)
    tensor_id = document.main_graph.initializers[0]
    with TensorStore(document) as store:
        assert store.read(tensor_id, offset=4, length=4) == struct.pack("<f", 2.0)
        assert store.read(tensor_id, offset=12, length=4) == struct.pack("<f", 4.0)
        snapshots = {item.name: item for item in store.cache_snapshots()}
        typed = snapshots["typed-materializations"]
        assert typed.entry_count == 1, "one parse serves every slice"
        assert typed.hits == 1
        with pytest.raises(TensorUnavailableError, match="exceeds"):
            store.read(tensor_id, offset=14, length=8)


def test_string_tensors_stay_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    typed_model(path, data_type=TensorProto.STRING, dims=[2], values=[b"ab", b"cd"])
    document = open_document(path)
    tensor_id = document.main_graph.initializers[0]
    with TensorStore(document) as store:
        assert store.byte_length(tensor_id) is None
        with pytest.raises(TensorUnavailableError, match="not supported"):
            store.read(tensor_id)


def test_typed_span_survives_serialization(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    typed_model(path, data_type=TensorProto.FLOAT, dims=[2], values=[1.0, 2.0])
    document = open_document(path)
    restored = document_from_bytes(document_to_bytes(document))
    tensor = restored.tensors[restored.main_graph.initializers[0]]
    assert tensor.typed_span is not None
    with TensorStore(restored) as store:
        assert store.read(tensor.id) == struct.pack("<2f", 1.0, 2.0)


def test_a_document_without_typed_span_discloses_unavailable(
    tmp_path: Path,
) -> None:
    """A 1.0-era document lacks typed spans; the store must say so cleanly."""
    path = tmp_path / "model.onnx"
    typed_model(path, data_type=TensorProto.FLOAT, dims=[2], values=[1.0, 2.0])
    document = open_document(path)
    import json

    data = json.loads(document_to_bytes(document))
    data["schema_version"] = "1.0"
    for entry in data["tensors"]:
        entry.pop("typed_span", None)
    old = document_from_bytes(json.dumps(data).encode())
    tensor_id = old.main_graph.initializers[0]
    with TensorStore(old) as store:
        assert store.materialization(tensor_id) is Materialization.UNAVAILABLE
        with pytest.raises(TensorUnavailableError, match="recorded message span"):
            store.read(tensor_id)


def test_materialization_can_be_cancelled(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    typed_model(
        path,
        data_type=TensorProto.FLOAT,
        dims=[256],
        values=[float(i) for i in range(256)],
    )
    document = open_document(path)
    tensor_id = document.main_graph.initializers[0]
    token = CancellationToken()
    token.cancel()
    with TensorStore(document) as store:
        with pytest.raises(OperationCancelled):
            store.read(tensor_id, token=token)
