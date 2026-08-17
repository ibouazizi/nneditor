"""Core ML .mlpackage inspection tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from nneditor.adapters.coreml.package import CoremlError, open_coreml_package
from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.registry import default_artifact_adapter_registry
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import Document, JsonValue, Storage
from nneditor.storage.store import TensorStore

_SENTINEL = 0xDEADBEEF


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _feature(name: str) -> bytes:
    return _field(1, name.encode())


def build_mlmodel(
    *,
    spec_version: int = 6,
    inputs: tuple[str, ...] = ("input_ids",),
    outputs: tuple[str, ...] = ("logits",),
    model_type_field: int = 502,
) -> bytes:
    description = b"".join(_field(1, _feature(name)) for name in inputs)
    description += b"".join(_field(10, _feature(name)) for name in outputs)
    return (
        _varint(1 << 3)
        + _varint(spec_version)
        + _field(2, description)
        + _field(model_type_field, b"")
    )


def build_weight_bin(blobs: list[tuple[int, bytes]]) -> bytes:
    """MIL blob-storage v2 bytes from ``(mil_dtype, payload)`` pairs."""
    out = bytearray(struct.pack("<II", len(blobs), 2))
    out += b"\x00" * (64 - len(out))
    for mil_dtype, payload in blobs:
        metadata_offset = len(out)
        data_offset = metadata_offset + 64
        out += struct.pack("<IIQQQ", _SENTINEL, mil_dtype, len(payload), data_offset, 0)
        out += b"\x00" * (64 - 32)
        out += payload
        if len(out) % 64:
            out += b"\x00" * (64 - len(out) % 64)
    return bytes(out)


def build_package(
    root: Path,
    *,
    manifest: bytes | None = None,
    model: bytes | None = None,
    weights: bytes | None = None,
) -> Path:
    package = root / "Model.mlpackage"
    data_dir = package / "Data" / "com.apple.CoreML"
    data_dir.mkdir(parents=True)
    if manifest is None:
        manifest = json.dumps(
            {
                "fileFormatVersion": "1.0.0",
                "rootModelIdentifier": "root-id",
                "itemInfoEntries": {
                    "root-id": {"path": "com.apple.CoreML/model.mlmodel"}
                },
            }
        ).encode()
    (package / "Manifest.json").write_bytes(manifest)
    (data_dir / "model.mlmodel").write_bytes(
        model if model is not None else build_mlmodel()
    )
    if weights is not None:
        weights_dir = data_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "weight.bin").write_bytes(weights)
    return package


def package_info(document: Document) -> dict[str, JsonValue]:
    info = dict(document.extensions)["x-coreml.package"]
    assert isinstance(info, dict)
    return info


def test_mlpackage_directory_detects_as_coreml(tmp_path: Path) -> None:
    package = build_package(tmp_path)
    assert detect_artifact_kind(package) is ArtifactKind.COREML_PACKAGE


def test_other_directories_match_no_contract(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-package"
    plain.mkdir()
    (plain / "readme.txt").write_text("hello")
    with pytest.raises(DetectionError, match="no recognized model package"):
        detect_artifact_kind(plain)


def test_open_reads_header_manifest_and_features(tmp_path: Path) -> None:
    package = build_package(
        tmp_path,
        model=build_mlmodel(
            spec_version=8,
            inputs=("pixel_values", "attention_mask"),
            outputs=("embedding",),
        ),
    )
    document = open_coreml_package(package)

    assert document.artifact_kind is ArtifactKind.COREML_PACKAGE
    info = package_info(document)
    assert info["specification_version"] == 8
    assert info["model_type"] == "mlProgram"
    assert info["inputs"] == ["pixel_values", "attention_mask"]
    assert info["outputs"] == ["embedding"]
    manifest = info["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["fileFormatVersion"] == "1.0.0"
    assert "coreml.mil-not-decoded" in {item.code for item in document.diagnostics}


def test_blobs_become_typed_external_tensors(tmp_path: Path) -> None:
    payload = struct.pack("<8e", *(float(i) for i in range(8)))
    package = build_package(
        tmp_path, weights=build_weight_bin([(1, payload), (2, b"\x00" * 16)])
    )
    document = open_coreml_package(package)

    tensors = list(document.tensors.values())
    assert [tensor.element_type for tensor in tensors] == ["float16", "float32"]
    assert tensors[0].dims == (8,)
    assert tensors[1].dims == (4,)
    assert all(tensor.storage is Storage.EXTERNAL for tensor in tensors)

    with TensorStore(document) as store:
        assert store.read(tensors[0].id) == payload


def test_sub_byte_blobs_expose_raw_bytes(tmp_path: Path) -> None:
    package = build_package(tmp_path, weights=build_weight_bin([(8, b"\x21" * 6)]))
    document = open_coreml_package(package)

    (tensor,) = document.tensors.values()
    assert tensor.element_type == "uint8"
    assert tensor.dims == (6,)
    assert "coreml.blob-raw-bytes" in {item.code for item in document.diagnostics}


def test_bad_blob_sentinel_stops_exposure(tmp_path: Path) -> None:
    weights = bytearray(build_weight_bin([(1, b"\x00" * 4), (1, b"\x00" * 4)]))
    # header (64) + blob-0 metadata (64) + blob-0 data padded to 64 = 192,
    # where blob 1's metadata record and its sentinel begin.
    struct.pack_into("<I", weights, 192, 0)
    package = build_package(tmp_path, weights=bytes(weights))
    document = open_coreml_package(package)

    assert len(document.tensors) == 1
    assert "coreml.blob-invalid" in {item.code for item in document.diagnostics}


def test_unsupported_blob_version_yields_no_tensors(tmp_path: Path) -> None:
    weights = bytearray(build_weight_bin([(1, b"\x00" * 4)]))
    struct.pack_into("<I", weights, 4, 1)  # blob_v1 / Espresso
    package = build_package(tmp_path, weights=bytes(weights))
    document = open_coreml_package(package)

    assert not document.tensors
    assert "coreml.blob-store-unreadable" in {
        item.code for item in document.diagnostics
    }


def test_broken_manifest_degrades_with_a_diagnostic(tmp_path: Path) -> None:
    package = build_package(tmp_path, manifest=b"{not json")
    document = open_coreml_package(package)

    info = package_info(document)
    assert "manifest" not in info
    assert "coreml.manifest-unreadable" in {item.code for item in document.diagnostics}


def test_package_without_weights_opens_weightless(tmp_path: Path) -> None:
    document = open_coreml_package(build_package(tmp_path))
    assert not document.tensors
    assert "weight_blob_count" not in package_info(document)


def test_capability_contract_declares_the_boundaries(tmp_path: Path) -> None:
    document = open_coreml_package(build_package(tmp_path))
    availability = {
        status.capability: status.availability
        for status in document.capabilities.values()
    }
    assert availability[Capability.WEIGHTS] is Availability.PARTIAL
    assert availability[Capability.METADATA] is Availability.PARTIAL
    assert availability[Capability.TOPOLOGY] is Availability.UNAVAILABLE
    assert availability[Capability.TRACING] is Availability.UNAVAILABLE


def test_missing_model_file_surfaces_as_coreml_error(tmp_path: Path) -> None:
    package = tmp_path / "Broken.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("{}")
    with pytest.raises(CoremlError, match=r"model\.mlmodel"):
        open_coreml_package(package)


def test_registry_and_session_open_a_package_directory(tmp_path: Path) -> None:
    package = build_package(tmp_path, weights=build_weight_bin([(2, b"\x00" * 8)]))
    document = default_artifact_adapter_registry().open(package)
    assert document.artifact_kind is ArtifactKind.COREML_PACKAGE

    with ApplicationService() as service:
        session = service.open_model(package)
        assert session.document.artifact_kind is ArtifactKind.COREML_PACKAGE
