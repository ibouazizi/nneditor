"""Regression tests for the PyTorch-adapter security fixes.

Every test here feeds hostile or malformed input to a parser of untrusted
artifacts and pins the contract: bounded work plus a typed error or a
diagnostic — never a raw exception, a hang, or silently wrong bytes.
"""

from __future__ import annotations

import json
import pickle
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.pytorch import checkpoint as checkpoint_module
from nneditor.adapters.pytorch import fx as fx_module
from nneditor.adapters.pytorch import pickle_scan as pickle_scan_module
from nneditor.adapters.pytorch.checkpoint import (
    CheckpointError,
    _flatten,
    open_checkpoint,
)
from nneditor.adapters.pytorch.fx import FxError, open_fx_graph_module
from nneditor.adapters.pytorch.pickle_scan import PickleScanError, scan_state_dict
from nneditor.adapters.pytorch.pt2 import Pt2Error, open_pt2
from nneditor.adapters.pytorch.safetensors import (
    SafetensorsError,
    open_safetensors,
    write_safetensors,
)
from nneditor.adapters.pytorch.zip_store import ZipStoreError, zip_members
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Storage
from tests.unit.test_pt2_mapping import model_json, write_archive

# --- hand-assembled pickle helpers ------------------------------------------


def _su(text: str) -> bytes:
    """SHORT_BINUNICODE."""
    raw = text.encode()
    return b"\x8c" + bytes([len(raw)]) + raw


def _bi(value: int) -> bytes:
    """BININT."""
    return b"J" + struct.pack("<i", value)


def _tup(items: tuple[int, ...]) -> bytes:
    """MARK + BININTs + TUPLE."""
    return b"(" + b"".join(_bi(item) for item in items) + b"t"


def tensor_pickle(
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    key: str = "0",
    numel: int = 4,
) -> bytes:
    """A state dict {"w": tensor} with attacker-chosen geometry."""
    storage = (
        b"("
        + _su("storage")
        + b"ctorch\nFloatStorage\n"
        + _su(key)
        + _su("cpu")
        + _bi(numel)
        + b"tQ"
    )
    args = b"(" + storage + _bi(offset) + _tup(shape) + _tup(stride) + b"\x89)t"
    return (
        b"\x80\x02}("
        + _su("w")
        + b"ctorch._utils\n_rebuild_tensor_v2\n"
        + args
        + b"Ru."
    )


def write_checkpoint(path: Path, payload: bytes, storages: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        for key, blob in storages.items():
            archive.writestr(f"archive/data/{key}", blob)
    return path


# --- pickle scan: exception contract ----------------------------------------


class TestPickleScanContract:
    @pytest.mark.parametrize(
        "payload",
        [
            b"\x80\x02\x85.",  # TUPLE1 underflow
            b"\x80\x02\x86.",  # TUPLE2 underflow
            b"\x80\x02\x87.",  # TUPLE3 underflow
            b"\x80\x02q\x00.",  # BINPUT with nothing on the stack
            b"\x80\x02r\x00\x00\x00\x00.",  # LONG_BINPUT with an empty stack
            b"\x80\x02\x94.",  # MEMOIZE with an empty stack
            b"\x80\x02h\x00.",  # BINGET before any store
            b"\x80\x02j\x07\x00\x00\x00.",  # LONG_BINGET before any store
            b"\x80\x02\x93.",  # STACK_GLOBAL underflow
            b"\x80\x02Q.",  # BINPERSID underflow
            b"\x80\x02R.",  # REDUCE underflow
            b"\x80\x02s.",  # SETITEM underflow
            b"\x80\x02b.",  # BUILD underflow
            b"\x80\x022.",  # DUP with an empty stack
            b"\x80\x020.",  # POP with an empty stack
            b"\x80\x02a.",  # APPEND underflow
            b"\x80\x02}s.",  # SETITEM with a dict but no key/value
        ],
    )
    def test_stack_and_memo_abuse_is_a_scan_error(self, payload: bytes) -> None:
        with pytest.raises(PickleScanError):
            scan_state_dict(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            b"\x80\x02X\x02\x00\x00\x00\xff\xfe.",  # BINUNICODE, invalid UTF-8
            b"\x80\x02\x8c\x02\xff\xfe.",  # SHORT_BINUNICODE, invalid UTF-8
            b"\x80\x02c\xff\xfe\nsystem\n.",  # GLOBAL line, invalid UTF-8
        ],
    )
    def test_invalid_utf8_is_a_scan_error(self, payload: bytes) -> None:
        with pytest.raises(PickleScanError):
            scan_state_dict(payload)

    def test_memoize_respects_the_memo_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pickle_scan_module, "_MAX_MEMO", 4)
        payload = b"\x80\x02}" + b"\x94" * 6 + b"."
        with pytest.raises(PickleScanError, match="safety ceiling"):
            scan_state_dict(payload)

    def test_binput_respects_the_memo_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pickle_scan_module, "_MAX_MEMO", 2)
        with pytest.raises(PickleScanError, match="safety ceiling"):
            scan_state_dict(b"\x80\x02}q\xff.")
        with pytest.raises(PickleScanError, match="safety ceiling"):
            scan_state_dict(b"\x80\x02}q\x00q\x01q\x02.")

    def test_negative_geometry_is_refused_at_reduce_time(self) -> None:
        with pytest.raises(PickleScanError, match="negative"):
            scan_state_dict(tensor_pickle(offset=-4, shape=(2,), stride=(1,)))
        with pytest.raises(PickleScanError, match="negative"):
            scan_state_dict(tensor_pickle(offset=0, shape=(-2,), stride=(1,)))

    def test_state_applies_only_to_the_root_mapping(self) -> None:
        # BUILD on a *nested* OrderedDict: its state must not attach to the
        # root, and the pairing is held by reference rather than by id().
        stream = (
            b"\x80\x02}"
            + _su("n")
            + b"ccollections\nOrderedDict\n)R"
            + b"}"
            + _su("v")
            + _bi(1)
            + b"s"
            + b"b"
            + b"s."
        )
        result = scan_state_dict(stream)
        assert result.mapping == {"n": {}}
        assert result.root_state == {}


# --- checkpoint flattening: resource bounds ---------------------------------


class TestFlattenBounds:
    def test_shared_subtrees_flatten_below_the_budget(self) -> None:
        shared = {"leaf": 1}
        root = {"a": shared, "b": shared}
        scan = scan_state_dict(pickle.dumps(root, protocol=2))
        tensors, extras = _flatten(scan.mapping)
        assert tensors == {}
        assert extras == {"a.leaf": 1, "b.leaf": 1}

    def test_exponential_sharing_is_bounded(self) -> None:
        level: dict[str, object] = {"x": 0}
        for _ in range(40):
            level = {"a": level, "b": level}
        scan = scan_state_dict(pickle.dumps(level, protocol=2))
        start = time.perf_counter()
        with pytest.raises(CheckpointError, match="safety ceiling"):
            _flatten(scan.mapping)
        assert time.perf_counter() - start < 10.0

    def test_self_referential_mappings_are_refused(self) -> None:
        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        scan = scan_state_dict(pickle.dumps(cyclic, protocol=2))
        with pytest.raises(CheckpointError, match="references itself"):
            _flatten(scan.mapping)


# --- checkpoint geometry ----------------------------------------------------


class TestCheckpointGeometry:
    def test_valid_offsets_still_open(self, tmp_path: Path) -> None:
        payload = tensor_pickle(offset=1, shape=(2,), stride=(1,), numel=3)
        path = write_checkpoint(tmp_path / "ok.pt", payload, {"0": b"\x01" * 16})
        document = open_checkpoint(path)
        (tensor,) = document.tensors.values()
        assert tensor.storage is Storage.EMBEDDED_RAW

    def test_negative_storage_offset_is_refused(self, tmp_path: Path) -> None:
        payload = tensor_pickle(offset=-4, shape=(2,), stride=(1,))
        path = write_checkpoint(tmp_path / "neg.pt", payload, {"0": b"\x00" * 16})
        with pytest.raises(CheckpointError):
            open_checkpoint(path)

    def test_negative_dimensions_are_refused(self, tmp_path: Path) -> None:
        payload = tensor_pickle(offset=0, shape=(-2,), stride=(1,))
        path = write_checkpoint(tmp_path / "dim.pt", payload, {"0": b"\x00" * 16})
        with pytest.raises(CheckpointError):
            open_checkpoint(path)

    def test_unexpected_failures_become_checkpoint_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = tensor_pickle(offset=0, shape=(2,), stride=(1,))
        path = write_checkpoint(tmp_path / "boom.pt", payload, {"0": b"\x00" * 16})

        def explode(mapping: dict[str, object]) -> None:
            raise RecursionError("synthetic blow-up")

        monkeypatch.setattr(checkpoint_module, "_flatten", explode)
        with pytest.raises(CheckpointError, match="RecursionError"):
            open_checkpoint(path)


# --- zip store: central-directory lies --------------------------------------


def _patched_zip(path: Path, *, csize: int | None, usize: int | None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("a.bin", b"12345678")
        archive.writestr("b.bin", b"B" * 64)
    raw = bytearray(path.read_bytes())
    signature = raw.find(b"PK\x01\x02")
    assert signature >= 0
    assert raw[signature + 20 : signature + 24] == struct.pack("<I", 8)
    if csize is not None:
        raw[signature + 20 : signature + 24] = struct.pack("<I", csize)
    if usize is not None:
        raw[signature + 24 : signature + 28] = struct.pack("<I", usize)
    path.write_bytes(bytes(raw))
    return path


class TestZipStoreValidation:
    def test_honest_archives_pass(self, tmp_path: Path) -> None:
        path = _patched_zip(tmp_path / "ok.zip", csize=None, usize=None)
        members = zip_members(path)
        assert members["a.bin"].span == (members["a.bin"].data_offset, 8)

    def test_inflated_file_size_is_refused(self, tmp_path: Path) -> None:
        # The verified exploit: usize patched from 8 to a size covering the
        # next member and beyond; only the central directory is touched.
        path = _patched_zip(tmp_path / "usize.zip", csize=None, usize=4096)
        with pytest.raises(ZipStoreError):
            zip_members(path)

    def test_local_header_disagreement_is_refused(self, tmp_path: Path) -> None:
        # Both central sizes patched consistently, but the local header
        # still says 8 bytes — the span would swallow the next member.
        path = _patched_zip(tmp_path / "both.zip", csize=40, usize=40)
        with pytest.raises(ZipStoreError, match="disagree"):
            zip_members(path)


# --- pt2: attacker-typed JSON -----------------------------------------------


class TestPt2Hostility:
    def test_hostile_typed_arguments_never_crash(self, tmp_path: Path) -> None:
        node = {
            "target": "torch.ops.aten.clamp.default",
            "inputs": [
                {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
                {"name": "count", "arg": {"as_int": "abc"}},
                {"name": "huge", "arg": {"as_int": 1e400}},
                {"name": "nanish", "arg": {"as_float": "not-a-number"}},
                {"name": "mixed", "arg": {"as_ints": [1, "x", 1e400, 2]}},
                {"name": "floats", "arg": {"as_floats": [1.5, "bad"]}},
            ],
            "outputs": [{"as_tensor": {"name": "out"}}],
        }
        archive = write_archive(tmp_path / "hostile.pt2", model_json([node]))
        document = open_pt2(archive)
        (imported,) = document.main_graph.nodes
        attributes = {item.name: item.value for item in imported.attributes}
        assert "count" not in attributes
        assert "huge" not in attributes
        assert "nanish" not in attributes
        assert attributes["mixed"] == (1, 2)
        assert attributes["floats"] == (1.5,)
        assert any(
            item.code == "pytorch.unsupported-argument" for item in document.diagnostics
        )

    def test_deeply_nested_model_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "deep.pt2"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("m/archive_format", "pt2")
            bundle.writestr("m/byteorder", "little")
            bundle.writestr("m/models/model.json", b"[" * 200_000)
        with pytest.raises(Pt2Error):
            open_pt2(path)

    def test_huge_integer_literals_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bigint.pt2"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("m/archive_format", "pt2")
            bundle.writestr("m/byteorder", "little")
            bundle.writestr(
                "m/models/model.json", b'{"schema_version": ' + b"9" * 5000 + b"}"
            )
        with pytest.raises(Pt2Error):
            open_pt2(path)

    def test_unreadable_control_members_become_pt2_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "byteorder.pt2"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("m/archive_format", "pt2")
            bundle.writestr("m/byteorder", "x" * 128)
        with pytest.raises(Pt2Error, match="byteorder"):
            open_pt2(path)

    def test_unknown_user_inputs_are_diagnosed(self, tmp_path: Path) -> None:
        model = model_json(
            [
                {
                    "target": "torch.ops.aten.relu.default",
                    "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}}],
                    "outputs": [{"as_tensor": {"name": "out"}}],
                }
            ]
        )
        specs = model["graph_module"]["signature"]["input_specs"]
        specs.append({"user_input": {"arg": {"as_tensor": {"name": "ghost"}}}})
        document = open_pt2(write_archive(tmp_path / "ghost.pt2", model))
        assert any(
            item.code == "pytorch.unknown-user-input" for item in document.diagnostics
        )
        names = [
            document.main_graph.value(value_id).name
            for value_id in document.main_graph.inputs
        ]
        assert "ghost" not in names


# --- safetensors ------------------------------------------------------------


def write_raw_safetensors(path: Path, header: object, data: bytes) -> Path:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


class TestSafetensorsValidation:
    def test_overlapping_tensor_ranges_are_refused(self, tmp_path: Path) -> None:
        header: dict[str, Any] = {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
            "c": {"dtype": "F32", "shape": [2], "data_offsets": [12, 20]},
        }
        path = write_raw_safetensors(tmp_path / "o.safetensors", header, b"\x00" * 20)
        document = open_safetensors(path)
        by_name = {t.id.rsplit("#", 1)[-1]: t for t in document.tensors.values()}
        # Both sides of the overlap are refused: neither can be edited
        # without corrupting the other.
        assert by_name["a"].storage is Storage.ABSENT
        assert by_name["b"].storage is Storage.ABSENT
        assert by_name["c"].storage is Storage.EMBEDDED_RAW
        assert any(
            item.code == "pytorch.tensor-range-overlap" for item in document.diagnostics
        )

    def test_unordered_but_disjoint_ranges_still_read(self, tmp_path: Path) -> None:
        # A spec deviation, not a corruption risk: disclose it, read on.
        header: dict[str, Any] = {
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
        path = write_raw_safetensors(tmp_path / "u.safetensors", header, b"\x00" * 16)
        document = open_safetensors(path)
        assert all(
            tensor.storage is Storage.EMBEDDED_RAW
            for tensor in document.tensors.values()
        )
        assert any(
            item.code == "pytorch.tensor-ranges-unordered"
            for item in document.diagnostics
        )

    def test_bool_dimensions_are_refused(self, tmp_path: Path) -> None:
        header: dict[str, Any] = {
            "a": {"dtype": "F32", "shape": [True, 2], "data_offsets": [0, 8]},
        }
        path = write_raw_safetensors(tmp_path / "b.safetensors", header, b"\x00" * 8)
        document = open_safetensors(path)
        assert not document.tensors
        assert any(
            item.code == "pytorch.malformed-tensor-entry"
            for item in document.diagnostics
        )

    def test_deeply_nested_headers_are_refused(self, tmp_path: Path) -> None:
        path = write_raw_safetensors(tmp_path / "deep.safetensors", b"[" * 100_000, b"")
        with pytest.raises(SafetensorsError):
            open_safetensors(path)

    def test_huge_integer_headers_are_refused(self, tmp_path: Path) -> None:
        path = write_raw_safetensors(
            tmp_path / "bigint.safetensors", b'{"a": ' + b"9" * 5000 + b"}", b""
        )
        with pytest.raises(SafetensorsError):
            open_safetensors(path)


# --- fx ----------------------------------------------------------------------


class TestFxBounds:
    def test_oversized_raw_pickles_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fx_module, "_MAX_PICKLE_BYTES", 16)
        blob = tmp_path / "big.pkl"
        blob.write_bytes(b"\x80\x02N." + b"\x00" * 100)
        with pytest.raises(FxError, match="limit"):
            open_fx_graph_module(blob)

    def test_malformed_pickle_streams_become_fx_errors(self, tmp_path: Path) -> None:
        blob = tmp_path / "bad.pkl"
        blob.write_bytes(b"\x80\x02X\xff\xff\xff\xff")
        with pytest.raises(FxError):
            open_fx_graph_module(blob)


# --- detection ---------------------------------------------------------------


class TestDetection:
    def test_safetensors_needs_a_sane_length_prefix(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.bin"
        bogus.write_bytes(struct.pack("<Q", 2**62) + b"{}")
        with pytest.raises(DetectionError):
            detect_artifact_kind(bogus)

    def test_real_safetensors_still_detect(self, tmp_path: Path) -> None:
        path = write_safetensors(
            tmp_path / "w.safetensors",
            [("t", "float32", (1,), b"\x00\x00\x00\x00")],
        )
        assert detect_artifact_kind(path) is ArtifactKind.SAFETENSORS

    def test_data_pkl_wins_over_a_planted_archive_format(self, tmp_path: Path) -> None:
        path = tmp_path / "both.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", b"\x80\x02}.")
            archive.writestr("archive/archive_format", "pt2")
        assert detect_artifact_kind(path) is ArtifactKind.PYTORCH_STATE_DICT

    def test_commented_stablehlo_detects_by_content(self, tmp_path: Path) -> None:
        path = tmp_path / "commented.txt"
        path.write_text("// produced by jax.export\nmodule @m {\n}\n", encoding="utf-8")
        assert detect_artifact_kind(path) is ArtifactKind.JAX_STABLEHLO
