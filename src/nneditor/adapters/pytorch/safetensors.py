"""Native safetensors reader and writer (P6.4/P6.5).

The format is deliberately simple — an 8-byte little-endian header length, a
JSON header mapping tensor names to ``{dtype, shape, data_offsets}``, then
raw little-endian data — which makes it the ideal safe artifact: every
tensor is an exact byte range, nothing executes, and a writer can be
byte-deterministic.

Reading produces a weights-only IR document per the safetensors capability
contract (topology unavailable, weights available, export available), so the
whole existing stack — tensor store, statistics, revisions, diff previews —
works on checkpoints unchanged.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.scalar_types import element_width_bytes
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import (
    ArtifactRef,
    Document,
    Graph,
    JsonValue,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, initializer_id
from nneditor.storage.reader import hash_file

__all__ = ["SafetensorsError", "open_safetensors", "write_safetensors"]

_MAX_HEADER_BYTES: Final = 100 * 1024 * 1024
_HEADER_LENGTH = struct.Struct("<Q")

_DTYPE_TO_IR: Final[dict[str, str]] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool",
}
_IR_TO_DTYPE: Final[dict[str, str]] = {
    value: key for key, value in _DTYPE_TO_IR.items()
}


class SafetensorsError(ValueError):
    """Raised when a safetensors artifact cannot be read or written."""


def _read_header(path: Path) -> tuple[dict[str, JsonValue], int]:
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_LENGTH.size)
        if len(prefix) != _HEADER_LENGTH.size:
            raise SafetensorsError(f"{path.name} is too short to be safetensors")
        (header_length,) = _HEADER_LENGTH.unpack(prefix)
        if header_length > _MAX_HEADER_BYTES:
            raise SafetensorsError(
                f"{path.name} declares a {header_length}-byte header; the "
                f"limit is {_MAX_HEADER_BYTES}"
            )
        raw = handle.read(header_length)
    if len(raw) != header_length:
        raise SafetensorsError(f"{path.name} header is truncated")
    try:
        # ValueError covers JSONDecodeError, oversized integer literals, and
        # malformed UTF-8; RecursionError covers pathological nesting depth —
        # all attacker-reachable through the header bytes.
        header = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise SafetensorsError(f"{path.name} header is not valid JSON") from error
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path.name} header must be a JSON object")
    return header, _HEADER_LENGTH.size + header_length


def open_safetensors(path: Path | str) -> Document:
    """Open a safetensors file as a weights-only document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`SafetensorsError`; the layered error contract in the session
    depends on adapters never leaking raw exceptions.
    """
    source = Path(path)
    try:
        return _open_safetensors(source)
    except SafetensorsError:
        raise
    except Exception as error:
        raise SafetensorsError(
            f"{source.name} could not be opened as safetensors: "
            f"{type(error).__name__}: {error}"
        ) from error


def _overlapping_names(header: dict[str, JsonValue]) -> tuple[set[str], bool]:
    """Names whose byte ranges overlap another's, plus whether order is off.

    The spec requires tensor ranges to be disjoint and laid out in header
    order. Overlap is the dangerous half — editing one tensor would silently
    corrupt another — so overlapping tensors are refused. Merely unordered
    (but disjoint) ranges are legal bytes with a spec deviation, so they are
    disclosed and still read.
    """
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            continue
        begin, end = offsets
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or isinstance(begin, bool)
            or isinstance(end, bool)
            or begin < 0
            or end < begin
        ):
            continue
        ranges.append((begin, end, name))
    ordered = sorted(ranges)
    unordered = ordered != ranges
    overlapping: set[str] = set()
    previous_end = 0
    previous_name = ""
    for begin, end, name in ordered:
        if begin < previous_end:
            overlapping.add(name)
            overlapping.add(previous_name)
        if end > previous_end:
            previous_end, previous_name = end, name
    return overlapping, unordered


def _open_safetensors(source: Path) -> Document:
    header, data_start = _read_header(source)
    file_size = source.stat().st_size
    log = DiagnosticLog()

    overlapping, unordered = _overlapping_names(header)
    if unordered:
        log.add(
            "pytorch.tensor-ranges-unordered",
            Severity.WARNING,
            "the header lists tensor ranges out of byte order, which the "
            "safetensors specification does not allow; the bytes are still "
            "read because the ranges are disjoint",
            source.name,
        )

    tensors: list[TensorRef] = []
    metadata: dict[str, str] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            if isinstance(entry, dict):
                metadata = {str(key): str(value) for key, value in entry.items()}
            continue
        if not isinstance(entry, dict):
            log.add(
                "pytorch.malformed-tensor-entry",
                Severity.ERROR,
                f"header entry {name!r} is not an object",
                name,
            )
            continue
        dtype = str(entry.get("dtype", ""))
        element_type = _DTYPE_TO_IR.get(dtype)
        shape_raw = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            element_type is None
            or not isinstance(shape_raw, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            log.add(
                "pytorch.malformed-tensor-entry",
                Severity.ERROR,
                f"header entry {name!r} has an unsupported dtype or malformed "
                f"shape/offsets (dtype={dtype!r})",
                name,
            )
            continue
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (*shape_raw, *offsets)
        ):
            # Bools are ints to isinstance, but `true` is not a dimension.
            log.add(
                "pytorch.malformed-tensor-entry",
                Severity.ERROR,
                f"header entry {name!r} holds non-integer shape or offsets",
                name,
            )
            continue
        dims = tuple(int(item) for item in shape_raw if isinstance(item, int))
        begin = int(offsets[0]) if isinstance(offsets[0], int) else 0
        end = int(offsets[1]) if isinstance(offsets[1], int) else 0
        if any(dim < 0 for dim in dims):
            log.add(
                "pytorch.malformed-tensor-entry",
                Severity.ERROR,
                f"header entry {name!r} holds a negative dimension",
                name,
            )
            continue
        width = element_width_bytes(element_type)
        expected = None
        if width is not None:
            count = 1
            for dim in dims:
                count *= dim
            expected = count * width
        absolute = PayloadRange(data_start + begin, max(0, end - begin))
        tensor_id = initializer_id(ROOT_GRAPH_ID, name)
        if begin < 0 or end < begin or data_start + end > file_size:
            log.add(
                "pytorch.tensor-range-out-of-bounds",
                Severity.ERROR,
                f"tensor {name!r} declares offsets [{begin}, {end}) outside "
                "the data section",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        if name in overlapping:
            log.add(
                "pytorch.tensor-range-overlap",
                Severity.ERROR,
                f"tensor {name!r} declares offsets [{begin}, {end}) that "
                "overlap another tensor's range; editing one would corrupt "
                "the other, so its bytes are not exposed",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        if expected is not None and expected != absolute.length:
            log.add(
                "pytorch.tensor-length-mismatch",
                Severity.ERROR,
                f"tensor {name!r} holds {absolute.length} bytes but its shape "
                f"and dtype imply {expected}",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        tensors.append(
            TensorRef(
                tensor_id,
                element_type,
                dims,
                Storage.EMBEDDED_RAW,
                payload=absolute,
            )
        )

    extensions: list[tuple[str, JsonValue]] = []
    if metadata:
        extensions.append(
            ("x-safetensors.metadata", {key: value for key, value in metadata.items()})
        )
    contract = contract_for(ArtifactKind.SAFETENSORS)
    return Document(
        source=ArtifactRef(
            path=str(source),
            content_hash=hash_file(source),
            byte_size=file_size,
        ),
        artifact_kind=ArtifactKind.SAFETENSORS,
        capabilities=contract.statuses,
        graphs=[Graph(id=ROOT_GRAPH_ID, name="weights")],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(("loading_mode", "safe artifact"),),
                source_artifact=None,
            )
        ],
        diagnostics=tuple(log),
        extensions=extensions,
    )


def write_safetensors(
    destination: Path | str,
    tensors: Iterable[tuple[str, str, tuple[int, ...], bytes]],
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write ``(name, ir dtype, shape, raw bytes)`` tuples atomically.

    The header is deterministic (insertion order, compact separators), so the
    same tensors always produce byte-identical files — golden export tests
    depend on it. The destination is never overwritten.
    """
    target = Path(destination)
    if target.exists():
        raise SafetensorsError(f"{target} already exists; exports never overwrite")
    header: dict[str, JsonValue] = {}
    if metadata:
        header["__metadata__"] = {
            str(key): str(value) for key, value in metadata.items()
        }
    chunks: list[bytes] = []
    cursor = 0
    for name, element_type, dims, raw in tensors:
        dtype = _IR_TO_DTYPE.get(element_type)
        if dtype is None:
            raise SafetensorsError(
                f"tensor {name!r} has element type {element_type!r}, which "
                "safetensors cannot represent"
            )
        width = element_width_bytes(element_type)
        if width is not None:
            count = 1
            for dim in dims:
                count *= dim
            if count * width != len(raw):
                raise SafetensorsError(
                    f"tensor {name!r} holds {len(raw)} bytes but its shape "
                    f"and dtype imply {count * width}"
                )
        if name in header:
            raise SafetensorsError(f"duplicate tensor name {name!r}")
        header[name] = {
            "dtype": dtype,
            "shape": list(dims),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        chunks.append(raw)
        cursor += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with open(partial, "xb") as handle:
            handle.write(_HEADER_LENGTH.pack(len(header_bytes)))
            handle.write(header_bytes)
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target
