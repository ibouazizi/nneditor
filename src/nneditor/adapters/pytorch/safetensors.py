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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.scalar_types import element_width_bytes
from nneditor.cancellation import CancellationToken, OperationCancelled
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

__all__ = [
    "SafetensorSource",
    "SafetensorsError",
    "open_safetensors",
    "write_safetensors",
    "write_safetensors_stream",
]

_MAX_HEADER_BYTES: Final = 100 * 1024 * 1024
_HEADER_LENGTH = struct.Struct("<Q")
_WRITE_CHUNK_BYTES: Final = 8 * 1024 * 1024

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


@dataclass(frozen=True, slots=True)
class SafetensorSource:
    """One tensor whose bytes can be fetched in bounded ranges.

    ``byte_length`` may be omitted when the fixed-width dtype and shape fully
    determine it. The writer resolves every length before opening the staged
    output, then calls ``read(offset, length)`` in bounded chunks.
    """

    name: str
    element_type: str
    dims: tuple[int, ...]
    byte_length: int | None
    read: Callable[[int, int], bytes]


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


def _prepare_safetensor_header(
    tensors: Iterable[SafetensorSource],
    metadata: Mapping[str, str] | None,
) -> tuple[dict[str, JsonValue], tuple[tuple[SafetensorSource, int], ...]]:
    """Validate tensor metadata and resolve the byte ranges in the header."""
    header: dict[str, JsonValue] = {}
    if metadata:
        header["__metadata__"] = {
            str(key): str(value) for key, value in metadata.items()
        }
    prepared: list[tuple[SafetensorSource, int]] = []
    cursor = 0
    for source in tensors:
        dtype = _IR_TO_DTYPE.get(source.element_type)
        if dtype is None:
            raise SafetensorsError(
                f"tensor {source.name!r} has element type "
                f"{source.element_type!r}, which safetensors cannot represent"
            )
        if source.name in header:
            raise SafetensorsError(f"duplicate tensor name {source.name!r}")
        if any(dim < 0 for dim in source.dims):
            raise SafetensorsError(
                f"tensor {source.name!r} has a negative shape dimension"
            )
        width = element_width_bytes(source.element_type)
        if width is None:
            raise SafetensorsError(
                f"tensor {source.name!r} has element type "
                f"{source.element_type!r} without a fixed byte width"
            )
        count = 1
        for dim in source.dims:
            count *= dim
        expected = count * width
        length = expected if source.byte_length is None else source.byte_length
        if length < 0:
            raise SafetensorsError(
                f"tensor {source.name!r} declares a negative byte length"
            )
        if expected != length:
            raise SafetensorsError(
                f"tensor {source.name!r} holds {length} bytes but its shape "
                f"and dtype imply {expected}"
            )
        header[source.name] = {
            "dtype": dtype,
            "shape": list(source.dims),
            "data_offsets": [cursor, cursor + length],
        }
        prepared.append((source, length))
        cursor += length
    return header, tuple(prepared)


def write_safetensors_stream(
    destination: Path | str,
    tensors: Iterable[SafetensorSource],
    *,
    metadata: Mapping[str, str] | None = None,
    token: CancellationToken | None = None,
    chunk_bytes: int | None = None,
) -> Path:
    """Write range-readable tensors atomically without accumulating payloads."""
    target = Path(destination)
    if target.exists():
        raise SafetensorsError(f"{target} already exists; exports never overwrite")
    if token is not None:
        token.raise_if_cancelled()
    chunk_size = _WRITE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    if chunk_size <= 0:
        raise SafetensorsError("safetensors write chunk size must be positive")
    header, prepared = _prepare_safetensor_header(tensors, metadata)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise SafetensorsError(
            f"safetensors header is {len(header_bytes)} bytes; the limit is "
            f"{_MAX_HEADER_BYTES}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with open(partial, "xb") as handle:
            handle.write(_HEADER_LENGTH.pack(len(header_bytes)))
            handle.write(header_bytes)
            for source, length in prepared:
                offset = 0
                while offset < length:
                    if token is not None:
                        token.raise_if_cancelled()
                    requested = min(chunk_size, length - offset)
                    try:
                        chunk = source.read(offset, requested)
                    except OperationCancelled:
                        raise
                    except Exception as error:
                        raise SafetensorsError(
                            f"tensor {source.name!r} could not be read: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    if len(chunk) != requested:
                        raise SafetensorsError(
                            f"tensor {source.name!r} returned {len(chunk)} bytes "
                            f"for range [{offset}, {offset + requested}); expected "
                            f"{requested}"
                        )
                    handle.write(chunk)
                    offset += requested
            if token is not None:
                token.raise_if_cancelled()
            handle.flush()
            os.fsync(handle.fileno())
        if token is not None:
            token.raise_if_cancelled()
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target


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

    def bytes_reader(payload: bytes) -> Callable[[int, int], bytes]:
        def read(offset: int, length: int) -> bytes:
            return payload[offset : offset + length]

        return read

    sources = (
        SafetensorSource(
            name=name,
            element_type=element_type,
            dims=dims,
            byte_length=len(raw),
            read=bytes_reader(raw),
        )
        for name, element_type, dims, raw in tensors
    )
    return write_safetensors_stream(destination, sources, metadata=metadata)
