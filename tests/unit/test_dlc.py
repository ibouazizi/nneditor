"""Container-level Qualcomm DLC inspection tests."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest

from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.qualcomm import dlc as dlc_module
from nneditor.adapters.qualcomm.dlc import DlcError, open_dlc
from nneditor.adapters.registry import default_artifact_adapter_registry
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import Document, JsonValue, Storage

_MODEL_BYTES = b"DLC2\x00\x01serialized-network-stream" * 8
_METADATA_TEXT = (
    "converter-command=snpe-onnx-to-dlc\n"
    "converter-version: 2.22.6\n"
    "model-version=1\n"
    "\n"
    "# comment line\n"
)


def build_dlc(
    path: Path,
    *,
    metadata: bytes | None = _METADATA_TEXT.encode(),
    model_compression: int = zipfile.ZIP_STORED,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo("model"), _MODEL_BYTES, compress_type=model_compression
        )
        if metadata is not None:
            archive.writestr("dlc.metadata", metadata)
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
    return path


def container_of(document: Document) -> dict[str, JsonValue]:
    container = dict(document.extensions)["x-dlc.container"]
    assert isinstance(container, dict)
    return container


def members_of(document: Document) -> list[dict[str, JsonValue]]:
    members = container_of(document)["members"]
    assert isinstance(members, list)
    narrowed: list[dict[str, JsonValue]] = []
    for member in members:
        assert isinstance(member, dict)
        narrowed.append(member)
    return narrowed


def test_zip_with_dlc_metadata_detects_as_dlc(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc")
    assert detect_artifact_kind(path) is ArtifactKind.QUALCOMM_DLC


def test_nested_metadata_member_still_marks_the_container(tmp_path: Path) -> None:
    path = tmp_path / "nested.dlc"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle/dlc.metadata", "model-version=1\n")
        archive.writestr("bundle/model", _MODEL_BYTES)
    assert detect_artifact_kind(path) is ArtifactKind.QUALCOMM_DLC


def test_data_pkl_wins_over_a_planted_dlc_marker(tmp_path: Path) -> None:
    # A hostile checkpoint could append a dlc.metadata member; the torch
    # container markers keep precedence so the reroute cannot happen.
    path = tmp_path / "planted.dlc"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02.")
        archive.writestr("dlc.metadata", "model-version=1\n")
    assert detect_artifact_kind(path) is ArtifactKind.PYTORCH_STATE_DICT


def test_plain_zip_still_matches_no_contract(tmp_path: Path) -> None:
    path = tmp_path / "plain.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "not a model")
    with pytest.raises(DetectionError, match="no recognized model container"):
        detect_artifact_kind(path)


def test_open_dlc_indexes_members_and_metadata(tmp_path: Path) -> None:
    path = build_dlc(
        tmp_path / "model.dlc", extra_members={"aux/quant_params": b"\x01" * 64}
    )
    document = open_dlc(path)

    assert document.artifact_kind is ArtifactKind.QUALCOMM_DLC
    container = container_of(document)
    assert container["member_count"] == 3
    names = {member["name"] for member in members_of(document)}
    assert names == {"model", "dlc.metadata", "aux/quant_params"}
    assert container["metadata"] == {
        "converter-command": "snpe-onnx-to-dlc",
        "converter-version": "2.22.6",
        "model-version": "1",
    }
    assert "dlc.model-stream-not-decoded" in {
        item.code for item in document.diagnostics
    }


def test_stored_member_bytes_are_range_readable(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc")
    document = open_dlc(path)

    tensors = {tensor.id: tensor for tensor in document.tensors.values()}
    model_tensor = next(
        tensor for tensor in tensors.values() if tensor.id.endswith("model")
    )
    assert model_tensor.storage is Storage.EMBEDDED_RAW
    assert model_tensor.element_type == "uint8"
    assert model_tensor.dims == (len(_MODEL_BYTES),)
    assert model_tensor.payload is not None
    raw = path.read_bytes()
    span = raw[
        model_tensor.payload.offset : model_tensor.payload.offset
        + model_tensor.payload.length
    ]
    assert span == _MODEL_BYTES


def test_compressed_members_are_listed_without_bytes(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc", model_compression=zipfile.ZIP_DEFLATED)
    document = open_dlc(path)

    assert not any(tensor.id.endswith("#model") for tensor in document.tensors.values())
    assert "dlc.compressed-member" in {item.code for item in document.diagnostics}
    member = next(item for item in members_of(document) if item["name"] == "model")
    assert member["compression"] == "deflated"
    assert member["byte_size"] == len(_MODEL_BYTES)


def test_capability_contract_declares_the_boundaries(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc")
    document = open_dlc(path)

    availability = {
        status.capability: status.availability
        for status in document.capabilities.values()
    }
    assert availability[Capability.METADATA] is Availability.PARTIAL
    assert availability[Capability.TOPOLOGY] is Availability.PARTIAL
    assert availability[Capability.WEIGHTS] is Availability.PARTIAL
    assert availability[Capability.EDITING] is Availability.UNAVAILABLE
    assert availability[Capability.TRACING] is Availability.UNAVAILABLE
    assert availability[Capability.EXPORT] is Availability.UNAVAILABLE


def test_oversized_metadata_is_skipped_with_a_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dlc_module, "_MAX_METADATA_BYTES", 16)
    path = build_dlc(tmp_path / "model.dlc")
    document = open_dlc(path)

    assert "dlc.metadata-unreadable" in {item.code for item in document.diagnostics}
    assert "metadata" not in container_of(document)


def test_unparseable_metadata_is_carried_as_text(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc", metadata=b"just a free-form sentence\n")
    document = open_dlc(path)

    assert container_of(document)["metadata_text"] == "just a free-form sentence\n"


def test_member_index_is_truncated_at_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dlc_module, "_MAX_MEMBERS", 2)
    path = build_dlc(tmp_path / "model.dlc", extra_members={"a": b"1", "b": b"2"})
    document = open_dlc(path)

    assert "dlc.member-index-truncated" in {item.code for item in document.diagnostics}
    assert container_of(document)["member_count"] == 4
    assert len(members_of(document)) == 2


def test_bad_archives_surface_as_dlc_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.dlc"
    path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(DlcError, match="not a readable archive"):
        open_dlc(path)
    with pytest.raises(DlcError, match="not a readable file"):
        open_dlc(tmp_path / "missing.dlc")


def test_default_registry_opens_a_dlc_end_to_end(tmp_path: Path) -> None:
    path = build_dlc(tmp_path / "model.dlc")
    document = default_artifact_adapter_registry().open(path)
    assert document.artifact_kind is ArtifactKind.QUALCOMM_DLC


def test_versioned_json_metadata_is_parsed(tmp_path: Path) -> None:
    """QAIRT names the member ``dlc.metadata2.1.0`` and writes JSON."""
    metadata = json.dumps(
        {
            "header": {"artifact_type": "DLC_METADATA"},
            "dlcGenerationInfo": [{"converterCommand": {"tool": "qairt-converter"}}],
        }
    )
    path = tmp_path / "model.dlc"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dlc.metadata2.1.0", metadata)
        archive.writestr(zipfile.ZipInfo("model"), _MODEL_BYTES)

    assert detect_artifact_kind(path) is ArtifactKind.QUALCOMM_DLC
    container = container_of(open_dlc(path))
    assert container["metadata_format_version"] == "2.1.0"
    parsed = container["metadata"]
    assert isinstance(parsed, dict)
    header = parsed["header"]
    assert isinstance(header, dict)
    assert header["artifact_type"] == "DLC_METADATA"


def _write_zero_crc_zip(path: Path, members: dict[str, bytes]) -> None:
    """A stored-only zip whose CRC fields are zero, as Qualcomm's writer
    emits: ``zipfile`` can list it but refuses to ``read`` its members."""
    raw = bytearray()
    central = bytearray()
    offsets: list[int] = []
    for name, content in members.items():
        offsets.append(len(raw))
        encoded = name.encode()
        raw += struct.pack(
            "<4sHHHHHIIIHH",
            b"PK\x03\x04",
            20,
            0,
            0,
            0,
            0,
            0,
            len(content),
            len(content),
            len(encoded),
            0,
        )
        raw += encoded + content
    for (name, content), offset in zip(members.items(), offsets, strict=True):
        encoded = name.encode()
        central += struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02",
            20,
            20,
            0,
            0,
            0,
            0,
            0,
            len(content),
            len(content),
            len(encoded),
            0,
            0,
            0,
            0,
            0,
            offset,
        )
        central += encoded
    end = struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06",
        0,
        0,
        len(members),
        len(members),
        len(central),
        len(raw),
        0,
    )
    path.write_bytes(bytes(raw) + bytes(central) + end)


def test_zeroed_crc_members_are_still_readable(tmp_path: Path) -> None:
    """Qualcomm's writer zeroes every CRC-32; range reads must not care."""
    path = tmp_path / "model.dlc"
    _write_zero_crc_zip(
        path,
        {
            "dlc.metadata2.1.0": b"converter-version=2.38.0\n",
            "model": _MODEL_BYTES,
        },
    )
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(zipfile.BadZipFile, match="CRC"):
            archive.read("model")

    assert detect_artifact_kind(path) is ArtifactKind.QUALCOMM_DLC
    document = open_dlc(path)
    container = container_of(document)
    assert container["metadata"] == {"converter-version": "2.38.0"}
    model_tensor = next(
        tensor for tensor in document.tensors.values() if tensor.id.endswith("model")
    )
    assert model_tensor.payload is not None
    raw = path.read_bytes()
    span = raw[
        model_tensor.payload.offset : model_tensor.payload.offset
        + model_tensor.payload.length
    ]
    assert span == _MODEL_BYTES
    assert "dlc.metadata-unreadable" not in {item.code for item in document.diagnostics}


# -- synthetic NETD/NETP topology ---------------------------------------------


class _Fb:
    """A minimal FlatBuffers writer for tests.

    Nodes are ``("table", {index: node})``, ``("vector", [nodes])``,
    ``("string", text)``, or inline scalars ``("u32"|"i32"|"f32", value)``.
    Children serialize after their parents, so every reference offset is
    forward-positive as the schema-less reader expects.
    """

    def __init__(self, identifier: bytes) -> None:
        self.data = bytearray(8)
        self.identifier = identifier

    def finish(self, root: tuple[str, Any]) -> bytes:
        position = self._write(root)
        struct.pack_into("<I", self.data, 0, position)
        self.data[4:8] = self.identifier
        return bytes(self.data)

    def _write(self, node: tuple[str, Any]) -> int:
        kind, payload = node
        if kind == "string":
            position = len(self.data)
            encoded = payload.encode()
            self.data += struct.pack("<I", len(encoded)) + encoded + b"\x00"
            return position
        if kind == "vector":
            position = len(self.data)
            self.data += struct.pack("<I", len(payload))
            patch_positions: list[int | None] = []
            for item in payload:
                if isinstance(item, tuple) and item[0] == "u32":
                    self.data += struct.pack("<I", item[1])
                    patch_positions.append(None)
                else:
                    patch_positions.append(len(self.data))
                    self.data += b"\x00\x00\x00\x00"
            for item, patch in zip(payload, patch_positions, strict=True):
                if patch is not None:
                    target = self._write(item)
                    struct.pack_into("<I", self.data, patch, target - patch)
            return position
        assert kind == "table"
        fields = dict(payload)
        max_index = max(fields) if fields else -1
        table_position = len(self.data)
        self.data += b"\x00\x00\x00\x00"  # soffset patched below
        slots: dict[int, int] = {}
        deferred: list[tuple[int, tuple[str, Any]]] = []
        for index in sorted(fields):
            value = fields[index]
            slots[index] = len(self.data) - table_position
            if value[0] == "u32":
                self.data += struct.pack("<I", value[1])
            elif value[0] == "i32":
                self.data += struct.pack("<i", value[1])
            elif value[0] == "f32":
                self.data += struct.pack("<f", value[1])
            else:
                deferred.append((len(self.data), value))
                self.data += b"\x00\x00\x00\x00"
        table_size = len(self.data) - table_position
        vtable_position = len(self.data)
        vtable_length = 4 + 2 * (max_index + 1)
        self.data += struct.pack("<HH", vtable_length, table_size)
        for index in range(max_index + 1):
            self.data += struct.pack("<H", slots.get(index, 0))
        struct.pack_into(
            "<i", self.data, table_position, table_position - vtable_position
        )
        for patch, value in deferred:
            target = self._write(value)
            struct.pack_into("<I", self.data, patch, target - patch)
        return table_position


def _s(text: str) -> tuple[str, Any]:
    return ("string", text)


def _netd_stream() -> bytes:
    """One graph: input -> Conv (with a static weight and one attr) -> output."""
    weight = (
        "table",
        {
            0: ("u32", 2),
            1: _s("conv.weight"),
            2: ("u32", 4),
            3: ("vector", [("u32", 4), ("u32", 2)]),
            5: (
                "table",
                {
                    0: ("u32", 0),
                    1: ("u32", 1),
                    2: (
                        "table",
                        {
                            0: ("u32", 8),
                            1: ("f32", -1.0),
                            2: ("f32", 1.0),
                            3: ("f32", 0.0078125),
                            4: ("i32", -128),
                        },
                    ),
                },
            ),
            6: ("u32", 0x0308),
            7: ("u32", 0x0232),
        },
    )
    tensors = (
        "vector",
        [
            (
                "table",
                {
                    0: ("u32", 0),
                    1: _s("input"),
                    2: ("u32", 0),
                    3: ("vector", [("u32", 1), ("u32", 4)]),
                    6: ("u32", 0x0232),
                },
            ),
            weight,
            (
                "table",
                {
                    0: ("u32", 1),
                    1: _s("output"),
                    2: ("u32", 1),
                    3: ("vector", [("u32", 1), ("u32", 2)]),
                    6: ("u32", 0x0232),
                },
            ),
        ],
    )
    ops = (
        "vector",
        [
            (
                "table",
                {
                    0: _s("conv0"),
                    1: _s("Conv2d"),
                    2: ("vector", [_s("input"), _s("conv.weight")]),
                    3: ("vector", [_s("output")]),
                    4: (
                        "vector",
                        [
                            (
                                "table",
                                {
                                    0: _s("packageName"),
                                    3: (
                                        "table",
                                        {0: ("u32", 0x0608), 3: _s("qti.aisw")},
                                    ),
                                },
                            ),
                            (
                                "table",
                                {
                                    0: _s("group"),
                                    3: ("table", {0: ("u32", 0x0132), 1: ("u32", 1)}),
                                },
                            ),
                        ],
                    ),
                },
            ),
        ],
    )
    graph = ("table", {0: _s("net"), 1: ops, 2: tensors})
    body = _Fb(b"NETD").finish(("table", {0: ("vector", [graph])}))
    return struct.pack("<II", 0x00040AD5, 1) + body


def _netp_stream(weight_size: int) -> bytes:
    location = ("table", {0: ("u32", 0), 1: ("u32", weight_size)})
    static = ("table", {0: _s("conv.weight"), 3: location})
    op_record: tuple[str, Any] = ("table", {0: ("vector", [])})
    graph = (
        "table",
        {
            0: _s("net"),
            1: ("vector", [static]),
            2: ("vector", [op_record]),
        },
    )
    body = _Fb(b"NETP").finish(("table", {0: ("vector", [graph])}))
    return struct.pack("<II", 0x00040AD5, 1) + body


def _build_topology_dlc(path: Path, *, netp: bool = True) -> tuple[Path, bytes]:
    weight_bytes = bytes(range(8))  # 4*2 int8 elements
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model", _netd_stream())
        if netp:
            archive.writestr("model.params", _netp_stream(len(weight_bytes)))
            archive.writestr("model.params.bin", weight_bytes)
        archive.writestr("dlc.metadata2.1.0", '{"header": {}}')
    return path, weight_bytes


def test_netd_stream_decodes_into_a_graph(tmp_path: Path) -> None:
    path, weight_bytes = _build_topology_dlc(tmp_path / "model.dlc")
    document = open_dlc(path)

    graph = document.graphs[document.entry_graph]
    assert graph.name == "net"
    (node,) = graph.nodes
    assert node.op_type == "Conv2d"
    assert node.domain == "qualcomm.qnn"
    assert node.source_name == "conv0"
    assert [graph.value(v).name for v in node.inputs] == ["input", "conv.weight"]
    assert [graph.value(v).name for v in node.outputs] == ["output"]
    assert {a.name for a in node.attributes} == {"packageName", "group"}
    assert [graph.value(v).name for v in graph.inputs] == ["input"]
    assert [graph.value(v).name for v in graph.outputs] == ["output"]
    assert "dlc.model-stream-not-decoded" not in {
        item.code for item in document.diagnostics
    }

    (tensor,) = document.tensors.values()
    assert tensor.element_type == "int8"
    assert tensor.dims == (4, 2)
    assert tensor.payload is not None
    raw = path.read_bytes()
    span = raw[tensor.payload.offset : tensor.payload.offset + tensor.payload.length]
    assert span == weight_bytes

    quant = dict(document.extensions)["x-dlc.quantization"]
    assert isinstance(quant, dict)
    entry = quant["conv.weight"]
    assert isinstance(entry, dict)
    assert entry["bitwidth"] == 8
    assert entry["offset"] == -128


def test_static_tensor_without_payload_directory_degrades(tmp_path: Path) -> None:
    path, _ = _build_topology_dlc(tmp_path / "model.dlc", netp=False)
    document = open_dlc(path)

    (tensor,) = document.tensors.values()
    assert tensor.storage is Storage.ABSENT
    assert "dlc.payload-unavailable" in {item.code for item in document.diagnostics}


def test_unresolved_op_edge_gets_a_placeholder_value(tmp_path: Path) -> None:
    op = (
        "table",
        {
            0: _s("mul0"),
            1: _s("Eltwise_Binary"),
            2: ("vector", [_s("input"), _s("ghost")]),
            3: ("vector", [_s("output")]),
        },
    )
    tensors = (
        "vector",
        [
            ("table", {1: _s("input"), 2: ("u32", 0), 6: ("u32", 0x0232)}),
            ("table", {1: _s("output"), 2: ("u32", 1), 6: ("u32", 0x0232)}),
        ],
    )
    graph = ("table", {0: _s("net"), 1: ("vector", [op]), 2: tensors})
    body = _Fb(b"NETD").finish(("table", {0: ("vector", [graph])}))
    path = tmp_path / "model.dlc"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model", struct.pack("<II", 0x00040AD5, 1) + body)
        archive.writestr("dlc.metadata", "model-version=1\n")
    document = open_dlc(path)

    graph_ir = document.graphs[document.entry_graph]
    (node,) = graph_ir.nodes
    assert graph_ir.value(node.inputs[1]).name == "ghost"
    assert "dlc.tensor-unresolved" in {item.code for item in document.diagnostics}
