"""PT2 mapping decisions exercised directly (P6.2).

Hand-built archives reach the branches a well-formed torch export never
takes: unusual argument kinds, symbolic dimensions, custom operators,
pickled weights, and length mismatches.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Any

from nneditor.adapters.pytorch.pt2 import open_pt2
from nneditor.ir.capabilities import Availability, Capability
from nneditor.ir.core import Storage


def write_archive(
    path: Path,
    model: dict[str, Any],
    *,
    weights: dict[str, bytes] | None = None,
    weights_config: dict[str, Any] | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("m/archive_format", "pt2")
        bundle.writestr("m/byteorder", "little")
        bundle.writestr("m/models/model.json", json.dumps(model))
        if weights_config is not None:
            bundle.writestr(
                "m/data/weights/model_weights_config.json",
                json.dumps({"config": weights_config}),
            )
        for name, payload in (weights or {}).items():
            bundle.writestr(f"m/data/weights/{name}", payload)
    return path


def model_json(nodes: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "inputs": [{"as_tensor": {"name": "x"}}],
        "outputs": [{"as_tensor": {"name": "out"}}],
        "nodes": nodes,
        "tensor_values": {
            "x": {"dtype": 7, "sizes": [{"as_int": 2}]},
            "out": {"dtype": 7, "sizes": [{"as_int": 2}]},
        },
    }
    graph.update(extra.pop("graph", {}))
    return {
        "schema_version": {"major": 8, "minor": 20},
        "torch_version": "2.13.0",
        "graph_module": {
            "graph": graph,
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}]
            },
            **extra.pop("graph_module", {}),
        },
        **extra,
    }


def test_typed_arguments_become_attributes(tmp_path: Path) -> None:
    node = {
        "target": "torch.ops.aten.clamp.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
            {"name": "min", "arg": {"as_float": 0.5}},
            {"name": "count", "arg": {"as_int": 3}},
            {"name": "mode", "arg": {"as_string": "floor"}},
            {"name": "flag", "arg": {"as_bool": True}},
            {"name": "pads", "arg": {"as_ints": [1, 2]}},
            {"name": "scales", "arg": {"as_floats": [0.5, 2.0]}},
            {"name": "names", "arg": {"as_strings": ["a", "b"]}},
            {"name": "device", "arg": {"as_device": {"type": "cpu"}}},
            {"name": "dim", "arg": {"as_sym_int": {"as_name": "s0"}}},
            {"name": "skipped", "arg": {"as_none": None}},
        ],
        "outputs": [{"as_tensor": {"name": "out"}}],
    }
    document = open_pt2(write_archive(tmp_path / "a.pt2", model_json([node])))
    (imported,) = document.main_graph.nodes
    attributes = {item.name: item.value for item in imported.attributes}
    assert attributes["min"] == 0.5
    assert attributes["count"] == 3
    assert attributes["mode"] == "floor"
    assert attributes["pads"] == (1, 2)
    assert attributes["scales"] == (0.5, 2.0)
    assert attributes["names"] == ("a", "b")
    assert attributes["device"] == "cpu"
    assert attributes["dim"] == "s0"
    assert "skipped" not in attributes
    assert imported.inputs == (document.main_graph.inputs[0],)


def test_unsupported_argument_kinds_are_diagnosed(tmp_path: Path) -> None:
    node = {
        "target": "torch.ops.aten.weird.default",
        "inputs": [
            {"name": "obj", "arg": {"as_custom_obj": {"name": "thing"}}},
        ],
        "outputs": [{"as_custom": {"name": "nope"}}],
    }
    document = open_pt2(write_archive(tmp_path / "b.pt2", model_json([node])))
    codes = {item.code for item in document.diagnostics}
    assert "pytorch.unsupported-argument" in codes


def test_custom_operators_are_flagged_and_uneditable(tmp_path: Path) -> None:
    node = {
        "target": "torch.ops.mylib.fused_thing.default",
        "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}}],
        "outputs": [{"as_tensor": {"name": "out"}}],
    }
    document = open_pt2(write_archive(tmp_path / "c.pt2", model_json([node])))
    (imported,) = document.main_graph.nodes
    assert imported.domain == "pytorch.custom"
    assert any(item.code == "pytorch.custom-operator" for item in document.diagnostics)
    notes = document.notes_for(imported.id)
    assert notes and notes[0].capability is Capability.EDITING
    assert notes[0].availability is Availability.UNAVAILABLE


def test_symbolic_and_unreadable_dimensions(tmp_path: Path) -> None:
    model = model_json(
        [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
            }
        ]
    )
    model["graph_module"]["graph"]["tensor_values"]["x"] = {
        "dtype": 7,
        "sizes": [{"as_expr": {"expr_str": "batch"}}, {"nonsense": 1}],
    }
    document = open_pt2(write_archive(tmp_path / "d.pt2", model))
    shape = document.main_graph.value(document.main_graph.inputs[0]).shape
    assert shape == ("batch", None)
    assert any(
        item.code == "pytorch.unsupported-dimension" for item in document.diagnostics
    )


def test_tensor_list_arguments_expand_into_ports(tmp_path: Path) -> None:
    model = model_json(
        [
            {
                "target": "torch.ops.aten.cat.default",
                "inputs": [
                    {
                        "name": "tensors",
                        "arg": {"as_tensors": [{"name": "x"}, {"name": "second"}]},
                    }
                ],
                "outputs": [{"as_tensors": [{"name": "out"}]}],
            }
        ]
    )
    document = open_pt2(write_archive(tmp_path / "e.pt2", model))
    (imported,) = document.main_graph.nodes
    assert len(imported.inputs) == 2
    assert len(imported.outputs) == 1


def test_weights_are_range_read_and_validated(tmp_path: Path) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    model = model_json(
        [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
            }
        ]
    )
    archive = write_archive(
        tmp_path / "f.pt2",
        model,
        weights={"w0": payload, "w1": b"\x00\x00"},
        weights_config={
            "layer.weight": {
                "path_name": "w0",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": {"dtype": 7, "sizes": [{"as_int": 4}]},
            },
            "layer.bias": {
                "path_name": "w1",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": {"dtype": 7, "sizes": [{"as_int": 4}]},
            },
            "layer.pickled": {
                "path_name": "w2",
                "is_param": True,
                "use_pickle": True,
                "tensor_meta": {"dtype": 7, "sizes": [{"as_int": 1}]},
            },
        },
    )
    document = open_pt2(archive)
    from nneditor.storage.store import TensorStore

    good = next(t for t in document.tensors.values() if t.id.endswith("#layer.weight"))
    bad = next(t for t in document.tensors.values() if t.id.endswith("#layer.bias"))
    pickled = next(
        t for t in document.tensors.values() if t.id.endswith("#layer.pickled")
    )
    assert good.storage is Storage.EMBEDDED_RAW
    assert bad.storage is Storage.ABSENT, "length mismatch is not trusted"
    assert pickled.storage is Storage.ABSENT
    codes = {item.code for item in document.diagnostics}
    assert "pytorch.tensor-length-mismatch" in codes
    assert "pytorch.weight-not-range-readable" in codes
    with TensorStore(document) as store:
        assert store.read(good.id) == payload


def test_parameter_placeholders_take_their_module_names(tmp_path: Path) -> None:
    model = model_json(
        [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "p_w"}}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
            }
        ]
    )
    model["graph_module"]["signature"]["input_specs"] = [
        {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "layer.weight"}},
        {"buffer": {"arg": {"name": "b_r"}, "buffer_name": "layer.running"}},
        {"user_input": {"arg": {"as_tensor": {"name": "x"}}}},
    ]
    document = open_pt2(write_archive(tmp_path / "g.pt2", model))
    names = {value.name for value in document.main_graph.values}
    assert "layer.weight" in names
    extensions = dict(document.extensions)
    assert extensions["x-torch.parameters"] == {
        "b_r": "layer.running",
        "p_w": "layer.weight",
    }


def test_archives_missing_their_model_are_refused(tmp_path: Path) -> None:
    import pytest

    from nneditor.adapters.pytorch.pt2 import Pt2Error

    path = tmp_path / "h.pt2"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("m/archive_format", "pt2")
    with pytest.raises(Pt2Error, match="no serialized model"):
        open_pt2(path)
