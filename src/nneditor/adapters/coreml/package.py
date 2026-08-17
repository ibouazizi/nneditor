"""Core ML ``.mlpackage`` inspection.

A ``.mlpackage`` is a directory bundle: ``Manifest.json`` next to
``Data/com.apple.CoreML/model.mlmodel`` (one serialized ``Model`` protobuf)
and, for ML Programs, ``weights/weight.bin`` — Apple's MIL blob-storage v2
file. The blob store is fully specified in coremltools' C++
(``StorageFormat.hpp``): a 64-byte header, then 64-byte-aligned metadata
records with a ``0xDEADBEEF`` sentinel, a dtype, a size, and the absolute
offset of the raw data.

This adapter reads what those specifications prove: the manifest, the model
protobuf's header fields (specification version, model type, feature names),
and the typed, range-readable weight blobs. The MIL program's operations are
not decoded into a graph; the capability contract says so.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import (
    ArtifactRef,
    Document,
    ExternalRef,
    Graph,
    JsonValue,
    ProvenanceEntry,
    Storage,
    TensorRef,
)
from nneditor.ir.dtypes import dtype_info
from nneditor.ir.identity import ROOT_GRAPH_ID, initializer_id
from nneditor.storage.reader import hash_file

__all__ = ["CoremlError", "open_coreml_package"]

_MANIFEST_NAME: Final = "Manifest.json"
_MODEL_RELATIVE: Final = ("Data", "com.apple.CoreML", "model.mlmodel")
_WEIGHTS_LOCATION: Final = "weights/weight.bin"
_MAX_MANIFEST_BYTES: Final = 1024 * 1024
_MAX_MODEL_HEADER_BYTES: Final = 64 * 1024 * 1024
_BLOB_SENTINEL: Final = 0xDEADBEEF
_BLOB_ALIGNMENT: Final = 64
_BLOB_STORAGE_VERSION: Final = 2

# BlobDataType from coremltools mlmodel/src/MILBlob/Blob/BlobDataType.hpp,
# mapped to the dtype registry's canonical names. ``None`` marks packed
# sub-byte types this build exposes as raw bytes.
_BLOB_DTYPES: Final[dict[int, str | None]] = {
    1: "float16",
    2: "float32",
    3: "uint8",
    4: "int8",
    5: "bfloat16",
    6: "int16",
    7: "uint16",
    8: None,  # int4
    9: None,  # uint1
    10: None,  # uint2
    11: None,  # uint4
    12: None,  # uint3
    13: None,  # uint6
    14: "int32",
    15: "uint32",
    16: "float8e4m3fn",
    17: "float8e5m2",
}
_SUB_BYTE_NAMES: Final[dict[int, str]] = {
    8: "int4",
    9: "uint1",
    10: "uint2",
    11: "uint4",
    12: "uint3",
    13: "uint6",
}

# The Model.Type oneof field numbers this build names; every other number
# renders generically. From coremltools mlmodel/format/Model.proto.
_MODEL_TYPE_NAMES: Final[dict[int, str]] = {
    200: "pipelineClassifier",
    201: "pipelineRegressor",
    202: "pipeline",
    500: "neuralNetwork",
    502: "mlProgram",
    3000: "serializedModel",
}


class CoremlError(ValueError):
    """Raised when a Core ML package cannot be indexed."""


def open_coreml_package(path: Path | str) -> Document:
    """Open a ``.mlpackage`` directory as an inspection document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`CoremlError`; the layered error contract in the session
    depends on adapters never leaking raw exceptions.
    """
    target = Path(path)
    try:
        return _open_package(target)
    except CoremlError:
        raise
    except Exception as error:
        raise CoremlError(
            f"{target.name} could not be opened as a Core ML package: "
            f"{type(error).__name__}: {error}"
        ) from error


# -- protobuf wire walking ---------------------------------------------------


def _walk_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Top-level protobuf fields as ``(number, wire_type, value)``.

    Varints yield ints; length-delimited fields yield their bytes; fixed
    fields yield their raw bytes. Raises :class:`ValueError` on malformed
    input so callers can report one header-unreadable diagnostic.
    """
    i, n = 0, len(data)
    while i < n:
        tag, i = _read_varint(data, i)
        number, wire_type = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("field number 0")
        if wire_type == 0:
            value, i = _read_varint(data, i)
            yield number, wire_type, value
        elif wire_type == 2:
            length, i = _read_varint(data, i)
            if length > n - i:
                raise ValueError("length-delimited field overruns the buffer")
            yield number, wire_type, data[i : i + length]
            i += length
        elif wire_type == 5:
            if i + 4 > n:
                raise ValueError("fixed32 overruns the buffer")
            yield number, wire_type, data[i : i + 4]
            i += 4
        elif wire_type == 1:
            if i + 8 > n:
                raise ValueError("fixed64 overruns the buffer")
            yield number, wire_type, data[i : i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire_type}")


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        if i >= len(data) or shift > 63:
            raise ValueError("truncated varint")
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, i


def _feature_names(description: bytes, field_number: int) -> list[str]:
    """``FeatureDescription.name`` strings from one repeated field."""
    names: list[str] = []
    for number, wire_type, value in _walk_fields(description):
        if number != field_number or wire_type != 2:
            continue
        assert isinstance(value, bytes)
        for inner_number, inner_type, inner in _walk_fields(value):
            if inner_number == 1 and inner_type == 2:
                assert isinstance(inner, bytes)
                names.append(inner.decode("utf-8", errors="replace"))
                break
    return names


def _model_header(model_path: Path, log: DiagnosticLog) -> dict[str, JsonValue]:
    size = model_path.stat().st_size
    if size > _MAX_MODEL_HEADER_BYTES:
        log.add(
            "coreml.model-header-unreadable",
            Severity.WARNING,
            f"model.mlmodel holds {size} bytes, above the "
            f"{_MAX_MODEL_HEADER_BYTES}-byte header ceiling",
            model_path.name,
        )
        return {}
    data = model_path.read_bytes()
    header: dict[str, JsonValue] = {}
    try:
        for number, wire_type, value in _walk_fields(data):
            if number == 1 and wire_type == 0:
                assert isinstance(value, int)
                header["specification_version"] = value
            elif number == 2 and wire_type == 2:
                assert isinstance(value, bytes)
                header["inputs"] = list(_feature_names(value, 1)) or None
                header["outputs"] = list(_feature_names(value, 10)) or None
            elif number >= 200 and wire_type == 2:
                header["model_type"] = _MODEL_TYPE_NAMES.get(
                    number, f"model type field {number}"
                )
                if number == 502:
                    log.add(
                        "coreml.mil-not-decoded",
                        Severity.INFO,
                        "the ML Program's operations are not decoded into a "
                        "graph in this build",
                        model_path.name,
                    )
    except ValueError as error:
        log.add(
            "coreml.model-header-unreadable",
            Severity.WARNING,
            f"model.mlmodel does not walk as protobuf: {error}",
            model_path.name,
        )
        return {}
    # Repeated inputs are merged per call; drop explicit Nones from the
    # carried extension so the JSON stays clean.
    return {key: value for key, value in header.items() if value is not None}


# -- MIL blob storage --------------------------------------------------------


def _index_blobs(weights_path: Path, log: DiagnosticLog) -> list[TensorRef]:
    file_size = weights_path.stat().st_size
    with open(weights_path, "rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            log.add(
                "coreml.blob-store-unreadable",
                Severity.WARNING,
                "weight.bin is shorter than the blob-storage header",
                _WEIGHTS_LOCATION,
            )
            return []
        count, version = struct.unpack("<II", head)
        if version != _BLOB_STORAGE_VERSION:
            log.add(
                "coreml.blob-store-unreadable",
                Severity.WARNING,
                f"weight.bin declares blob-storage version {version}; only "
                f"version {_BLOB_STORAGE_VERSION} is supported",
                _WEIGHTS_LOCATION,
            )
            return []
        tensors: list[TensorRef] = []
        offset = _BLOB_ALIGNMENT
        for index in range(count):
            if offset + 32 > file_size:
                log.add(
                    "coreml.blob-invalid",
                    Severity.ERROR,
                    f"blob {index} metadata lies outside the file; it and "
                    "the records after it are not exposed",
                    _WEIGHTS_LOCATION,
                )
                break
            handle.seek(offset)
            sentinel, mil_dtype, size, data_offset, _padding = struct.unpack(
                "<IIQQQ", handle.read(32)
            )
            if sentinel != _BLOB_SENTINEL or data_offset + size > file_size:
                log.add(
                    "coreml.blob-invalid",
                    Severity.ERROR,
                    f"blob {index} fails its sentinel or bounds check; it "
                    "and the records after it are not exposed",
                    _WEIGHTS_LOCATION,
                )
                break
            element_type = _BLOB_DTYPES.get(mil_dtype)
            if element_type is None:
                declared = _SUB_BYTE_NAMES.get(mil_dtype, f"blob dtype {mil_dtype}")
                log.add(
                    "coreml.blob-raw-bytes",
                    Severity.INFO,
                    f"blob {index} stores {declared} elements; it is "
                    "exposed as raw bytes",
                    _WEIGHTS_LOCATION,
                )
                element_type, dims = "uint8", (size,)
            else:
                info = dtype_info(element_type)
                assert info is not None and info.byte_width is not None
                dims = (size // info.byte_width,)
            tensors.append(
                TensorRef(
                    initializer_id(ROOT_GRAPH_ID, f"weight.bin/blob{index}"),
                    element_type,
                    dims,
                    Storage.EXTERNAL,
                    external=ExternalRef(
                        location=_WEIGHTS_LOCATION,
                        offset=data_offset,
                        length=size,
                    ),
                )
            )
            data_end = data_offset + size
            offset = (
                (data_end + _BLOB_ALIGNMENT - 1) // _BLOB_ALIGNMENT * (_BLOB_ALIGNMENT)
            )
    return tensors


# -- package assembly --------------------------------------------------------


def _read_manifest(package: Path, log: DiagnosticLog) -> JsonValue | None:
    manifest_path = package / _MANIFEST_NAME
    try:
        if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds the size ceiling")
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        log.add(
            "coreml.manifest-unreadable",
            Severity.WARNING,
            f"{_MANIFEST_NAME} is not readable JSON: {error}",
            _MANIFEST_NAME,
        )
        return None
    if not isinstance(parsed, (dict, list)):
        log.add(
            "coreml.manifest-unreadable",
            Severity.WARNING,
            f"{_MANIFEST_NAME} holds a bare value instead of a manifest",
            _MANIFEST_NAME,
        )
        return None
    return parsed


def _open_package(target: Path) -> Document:
    if not target.is_dir():
        raise CoremlError(f"{target} is not a package directory")
    model_path = target.joinpath(*_MODEL_RELATIVE)
    if not model_path.is_file():
        raise CoremlError(f"{target.name} holds no Data/com.apple.CoreML/model.mlmodel")

    log = DiagnosticLog()
    manifest = _read_manifest(target, log)
    header = _model_header(model_path, log)
    weights_path = model_path.parent / "weights" / "weight.bin"
    tensors = _index_blobs(weights_path, log) if weights_path.is_file() else []

    package_info: dict[str, JsonValue] = {"package_name": target.name}
    if manifest is not None:
        package_info["manifest"] = manifest
    package_info.update(header)
    if weights_path.is_file():
        package_info["weight_blob_count"] = len(tensors)

    contract = contract_for(ArtifactKind.COREML_PACKAGE)
    return Document(
        source=ArtifactRef(
            path=str(model_path),
            content_hash=hash_file(model_path),
            byte_size=model_path.stat().st_size,
        ),
        artifact_kind=ArtifactKind.COREML_PACKAGE,
        capabilities=contract.statuses,
        graphs=[Graph(id=ROOT_GRAPH_ID, name="package")],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(("loading_mode", "safe artifact"),),
            )
        ],
        diagnostics=tuple(log),
        extensions=[("x-coreml.package", package_info)],
    )
