"""Qualcomm DLC inspection: container index plus decoded NETD topology.

A SNPE/QAIRT ``.dlc`` file is a zip archive holding a serialized ``model``
stream (plus ``model.params``/``model.params.bin`` in QAIRT containers) next
to a metadata member. The metadata member's name carries the container format
version as a suffix — ``dlc.metadata2.1.0`` — and its content is JSON in
current converters, plain records in older ones.

The ``model`` stream is a FlatBuffers buffer in an unpublished schema whose
layout :mod:`nneditor.adapters.qualcomm.netd` decodes through validated,
reverse-engineered field indices. When the stream matches that layout the
document carries the real graph — ops, name-connected edges, tensor shapes,
dtypes, quantization encodings, and static weights resolved to byte ranges in
``model.params.bin``. A stream that does not match falls back to the
container index with raw member bytes; decoding never guesses.

Qualcomm's writer leaves every zip CRC-32 field zeroed, so
:meth:`zipfile.ZipFile.read` rejects the members it wrote. Members are
therefore read through their raw byte ranges, which need no checksum.
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.qualcomm.netd import (
    DlcGraph,
    DlcTensor,
    NetdError,
    PayloadLocation,
    decode_netd,
    decode_netp,
    qnn_dtype_name,
)
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import (
    ArtifactRef,
    Attribute,
    AttrKind,
    Document,
    Graph,
    IrError,
    JsonValue,
    Node,
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.dtypes import dtype_info
from nneditor.ir.identity import (
    ROOT_GRAPH_ID,
    initializer_id,
    node_id,
    value_id,
)
from nneditor.storage.reader import hash_file

__all__ = ["DlcError", "open_dlc"]

_MAX_MEMBERS: Final = 4096
_MAX_METADATA_BYTES: Final = 1024 * 1024
_MAX_STREAM_BYTES: Final = 512 * 1024 * 1024
_MAX_QUANTIZATION_ENTRIES: Final = 8192
_LOCAL_HEADER_SIZE: Final = 30
_LOCAL_HEADER_MAGIC: Final = b"PK\x03\x04"
_METADATA_MEMBER: Final = "dlc.metadata"
_MODEL_MEMBER: Final = "model"
_PARAMS_MEMBER: Final = "model.params"
_PARAMS_BIN_MEMBER: Final = "model.params.bin"
_QNN_DOMAIN: Final = "qualcomm.qnn"

_COMPRESSION_NAMES: Final = {
    zipfile.ZIP_STORED: "stored",
    zipfile.ZIP_DEFLATED: "deflated",
    zipfile.ZIP_BZIP2: "bzip2",
    zipfile.ZIP_LZMA: "lzma",
}


class DlcError(ValueError):
    """Raised when a DLC container cannot be indexed."""


def open_dlc(path: Path | str) -> Document:
    """Open a ``.dlc`` archive, with decoded topology when it matches.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`DlcError`; the layered error contract in the session depends
    on adapters never leaking raw exceptions.
    """
    target = Path(path)
    try:
        return _open_dlc(target)
    except DlcError:
        raise
    except Exception as error:
        raise DlcError(
            f"{target.name} could not be opened as a DLC container: "
            f"{type(error).__name__}: {error}"
        ) from error


# -- container access --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Member:
    info: zipfile.ZipInfo
    data_offset: int | None

    @property
    def stored(self) -> bool:
        return (
            self.info.compress_type == zipfile.ZIP_STORED
            and self.data_offset is not None
        )


def _member_basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _data_offset(handle_path: Path, info: zipfile.ZipInfo) -> int | None:
    """The absolute offset of a member's stored bytes, from its local header.

    The central directory's name and extra lengths may legally differ from
    the local header's, so the local header is authoritative for where the
    data starts. ``None`` means the header is inconsistent.
    """
    with open(handle_path, "rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(_LOCAL_HEADER_SIZE)
    if len(header) != _LOCAL_HEADER_SIZE or not header.startswith(_LOCAL_HEADER_MAGIC):
        return None
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return (
        info.header_offset + _LOCAL_HEADER_SIZE + int(name_length) + int(extra_length)
    )


def _member_bytes(target: Path, member: _Member, ceiling: int) -> bytes | None:
    """A stored member's raw content, tolerating Qualcomm's zeroed CRCs."""
    if not member.stored or member.info.file_size > ceiling:
        return None
    assert member.data_offset is not None
    with open(target, "rb") as handle:
        handle.seek(member.data_offset)
        return handle.read(member.info.file_size)


def _parse_records(text: str) -> dict[str, str]:
    """``key=value`` (or ``key: value``) records from a legacy metadata text."""
    records: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for separator in ("=", ":"):
            key, found, value = stripped.partition(separator)
            if found and key.strip():
                records.setdefault(key.strip(), value.strip())
                break
    return records


def _interpret_metadata(raw: bytes) -> tuple[JsonValue | None, str | None]:
    """The metadata member as parsed structure, or as text when unparsed.

    QAIRT converters write a JSON document; older SNPE metadata is line
    records. Anything else is carried verbatim.
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        return parsed, None
    records = _parse_records(text)
    if records:
        return dict(records), None
    return None, text


# -- decoded topology to IR --------------------------------------------------


@dataclass(slots=True)
class _Topology:
    graphs: list[Graph]
    tensors: list[TensorRef]
    quantization: dict[str, JsonValue]


def _tensor_element_type(tensor: DlcTensor) -> str | None:
    return qnn_dtype_name(tensor.stored_dtype_code)


def _quant_json(tensor: DlcTensor) -> JsonValue | None:
    quant = tensor.quantization
    if quant is None:
        return None
    entry: dict[str, JsonValue] = {
        "bitwidth": quant.bitwidth,
        "scale": quant.scale,
        "offset": quant.offset,
        "minimum": quant.minimum,
        "maximum": quant.maximum,
        "per_axis": quant.per_axis,
    }
    if quant.per_axis:
        entry["axis"] = quant.axis
        entry["channels"] = quant.channel_count
    return entry


def _static_tensor_ref(
    graph_id: str,
    tensor: DlcTensor,
    tensor_name: str,
    location: PayloadLocation | None,
    bin_member: _Member | None,
    file_size: int,
    log: DiagnosticLog,
) -> TensorRef:
    """A typed, range-readable reference for one static tensor.

    Falls back to raw bytes when the dtype is unknown, and to absent storage
    when the payload cannot be located — with a diagnostic either way.
    """
    tensor_id = initializer_id(graph_id, tensor_name)
    element_type = _tensor_element_type(tensor)
    if (
        location is None
        or bin_member is None
        or not bin_member.stored
        or location.offset + location.size > bin_member.info.file_size
    ):
        log.add(
            "dlc.payload-unavailable",
            Severity.WARNING,
            f"static tensor {tensor_name!r} has no readable payload in "
            "model.params.bin",
            tensor_id,
        )
        return TensorRef(
            tensor_id, element_type or "uint8", tensor.shape, Storage.ABSENT
        )
    assert bin_member.data_offset is not None
    absolute = bin_member.data_offset + location.offset
    if absolute + location.size > file_size:
        log.add(
            "dlc.payload-unavailable",
            Severity.WARNING,
            f"static tensor {tensor_name!r} declares a payload outside the container",
            tensor_id,
        )
        return TensorRef(
            tensor_id, element_type or "uint8", tensor.shape, Storage.ABSENT
        )
    payload = PayloadRange(absolute, location.size)
    if element_type is not None:
        info = dtype_info(element_type)
        expected = None
        if info is not None and info.byte_width is not None:
            count = 1
            for dim in tensor.shape:
                count *= dim
            expected = count * info.byte_width
        if expected == location.size:
            return TensorRef(
                tensor_id,
                element_type,
                tensor.shape,
                Storage.EMBEDDED_RAW,
                payload=payload,
            )
    log.add(
        "dlc.payload-unavailable",
        Severity.INFO,
        f"static tensor {tensor_name!r} does not match its declared shape "
        "and dtype; its payload is exposed as raw bytes",
        tensor_id,
    )
    return TensorRef(
        tensor_id,
        "uint8",
        (location.size,),
        Storage.EMBEDDED_RAW,
        payload=payload,
    )


def _graph_to_ir(
    graph_id: str,
    graph: DlcGraph,
    statics: dict[str, PayloadLocation],
    per_op: list[dict[str, PayloadLocation]],
    bin_member: _Member | None,
    file_size: int,
    log: DiagnosticLog,
    quantization: dict[str, JsonValue],
) -> tuple[Graph, list[TensorRef]]:
    values: dict[str, Value] = {}
    tensors: list[TensorRef] = []
    unresolved: set[str] = set()

    for tensor in graph.tensors:
        if tensor.name in values:
            continue
        values[tensor.name] = Value(
            id=value_id(graph_id, tensor.name),
            name=tensor.name,
            element_type=_tensor_element_type(tensor),
            shape=tuple(tensor.shape) or None,
        )
        if tensor.quantization is not None and (
            len(quantization) < _MAX_QUANTIZATION_ENTRIES
        ):
            quantization[tensor.name] = _quant_json(tensor)
        if tensor.is_static:
            tensors.append(
                _static_tensor_ref(
                    graph_id,
                    tensor,
                    tensor.name,
                    statics.get(tensor.name),
                    bin_member,
                    file_size,
                    log,
                )
            )

    def resolve(name: str) -> str:
        if name not in values:
            if name not in unresolved:
                unresolved.add(name)
                log.add(
                    "dlc.tensor-unresolved",
                    Severity.WARNING,
                    f"op edge {name!r} names no tensor in the graph table; "
                    "a placeholder value is shown",
                    name,
                )
            values[name] = Value(id=value_id(graph_id, name), name=name)
        return values[name].id

    nodes: list[Node] = []
    for ordinal, op in enumerate(graph.ops):
        derived_id, stability = node_id(
            graph_id,
            name=op.name,
            first_output=op.outputs[0] if op.outputs else "",
            ordinal=ordinal,
        )
        attributes: list[Attribute] = []
        parameters = per_op[ordinal] if ordinal < len(per_op) else {}
        for attribute in op.attributes:
            if not attribute.name:
                continue
            if attribute.tensor is not None:
                parameter = attribute.tensor
                tensor_ref = _static_tensor_ref(
                    graph_id,
                    parameter,
                    f"{op.name}#{attribute.name}",
                    parameters.get(parameter.name),
                    bin_member,
                    file_size,
                    log,
                )
                tensors.append(tensor_ref)
                attributes.append(
                    Attribute(attribute.name, AttrKind.TENSOR, tensor_ref.id)
                )
            elif attribute.string_value is not None:
                attributes.append(
                    Attribute(attribute.name, AttrKind.STRING, attribute.string_value)
                )
            elif attribute.float_value is not None:
                attributes.append(
                    Attribute(attribute.name, AttrKind.FLOAT, attribute.float_value)
                )
            elif attribute.int_value is not None:
                attributes.append(
                    Attribute(attribute.name, AttrKind.INT, attribute.int_value)
                )
        nodes.append(
            Node(
                id=derived_id,
                id_stability=stability,
                op_type=op.op_type,
                domain=_QNN_DOMAIN,
                inputs=tuple(resolve(name) for name in op.inputs),
                outputs=tuple(resolve(name) for name in op.outputs),
                attributes=tuple(attributes),
                source_name=op.name,
            )
        )

    inputs = [values[tensor.name].id for tensor in graph.tensors if tensor.is_input]
    outputs = [values[tensor.name].id for tensor in graph.tensors if tensor.is_output]
    initializers = [
        initializer_id(graph_id, tensor.name)
        for tensor in graph.tensors
        if tensor.is_static
    ]
    return (
        Graph(
            id=graph_id,
            name=graph.name,
            nodes=nodes,
            values=values.values(),
            inputs=inputs,
            outputs=outputs,
            initializers=initializers,
        ),
        tensors,
    )


def _decode_topology(
    target: Path,
    members: dict[str, _Member],
    file_size: int,
    log: DiagnosticLog,
) -> _Topology | None:
    model = members.get(_MODEL_MEMBER)
    if model is None:
        log.add(
            "dlc.model-stream-not-decoded",
            Severity.WARNING,
            "the container holds no model member, so no topology is shown",
            target.name,
        )
        return None
    try:
        model_bytes = _member_bytes(target, model, _MAX_STREAM_BYTES)
        if model_bytes is None:
            raise NetdError(
                "the model member is compressed or above the stream ceiling"
            )
        graphs = decode_netd(model_bytes)
        directory: dict[
            str, tuple[dict[str, PayloadLocation], list[dict[str, PayloadLocation]]]
        ] = {}
        params = members.get(_PARAMS_MEMBER)
        if params is not None:
            params_bytes = _member_bytes(target, params, _MAX_STREAM_BYTES)
            if params_bytes is not None:
                directory = decode_netp(params_bytes)

        bin_member = members.get(_PARAMS_BIN_MEMBER)
        ir_graphs: list[Graph] = []
        tensors: list[TensorRef] = []
        quantization: dict[str, JsonValue] = {}
        for index, graph in enumerate(graphs):
            graph_id = ROOT_GRAPH_ID if index == 0 else f"g:dlc{index}"
            statics, per_op = directory.get(graph.name, ({}, []))
            ir_graph, graph_tensors = _graph_to_ir(
                graph_id,
                graph,
                statics,
                per_op,
                bin_member,
                file_size,
                log,
                quantization,
            )
            ir_graphs.append(ir_graph)
            tensors.extend(graph_tensors)
    except (NetdError, IrError) as error:
        # IrError covers a decoded stream whose contents violate IR
        # invariants (duplicate names, dangling references): the decode is
        # then untrusted as a whole and the container index is shown.
        log.add(
            "dlc.model-stream-not-decoded",
            Severity.WARNING,
            f"the model stream does not match the validated NETD layout "
            f"({error}); the container index is shown instead",
            model.info.filename,
        )
        return None
    return _Topology(ir_graphs, tensors, quantization)


# -- container assembly ------------------------------------------------------


def _open_dlc(target: Path) -> Document:
    if not target.is_file():
        raise DlcError(f"{target} is not a readable file")
    try:
        archive = zipfile.ZipFile(target)
    except zipfile.BadZipFile as error:
        raise DlcError(f"{target.name} is not a readable archive") from error

    log = DiagnosticLog()
    member_rows: list[dict[str, JsonValue]] = []
    members: dict[str, _Member] = {}
    metadata_value: JsonValue | None = None
    metadata_text: str | None = None
    metadata_format_version: str | None = None
    file_size = target.stat().st_size

    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        total_members = len(infos)
        if total_members > _MAX_MEMBERS:
            log.add(
                "dlc.member-index-truncated",
                Severity.WARNING,
                f"the archive holds {total_members} members, above the "
                f"{_MAX_MEMBERS}-member indexing ceiling; the excess is "
                "not listed",
                target.name,
            )
            infos = infos[:_MAX_MEMBERS]

        for info in infos:
            basename = _member_basename(info.filename)
            compression = _COMPRESSION_NAMES.get(
                info.compress_type, f"method {info.compress_type}"
            )
            member_rows.append(
                {
                    "name": info.filename,
                    "byte_size": info.file_size,
                    "compressed_byte_size": info.compress_size,
                    "compression": compression,
                    "crc32": info.CRC,
                }
            )
            offset = (
                _data_offset(target, info)
                if info.compress_type == zipfile.ZIP_STORED
                else None
            )
            member = _Member(info, offset)
            members.setdefault(basename, member)
            if member.stored and offset is not None:
                if info.compress_size != info.file_size or (
                    offset + info.file_size > file_size
                ):
                    log.add(
                        "dlc.member-header-mismatch",
                        Severity.ERROR,
                        f"{info.filename} declares a local header "
                        "inconsistent with the central directory; its bytes "
                        "are not exposed",
                        info.filename,
                    )
                    members[basename] = _Member(info, None)
            elif info.compress_type != zipfile.ZIP_STORED:
                log.add(
                    "dlc.compressed-member",
                    Severity.INFO,
                    f"{info.filename} is stored {compression}, so its bytes "
                    "are not exposed for bounded reads",
                    info.filename,
                )
            if basename.startswith(_METADATA_MEMBER) and metadata_value is None:
                # The basename carries the container format version as a
                # suffix in current converters: ``dlc.metadata2.1.0``.
                metadata_format_version = basename[len(_METADATA_MEMBER) :] or None
                raw = _member_bytes(target, members[basename], _MAX_METADATA_BYTES)
                if raw is None:
                    log.add(
                        "dlc.metadata-unreadable",
                        Severity.WARNING,
                        f"{info.filename} is compressed, oversized, or "
                        "inconsistent, and is skipped",
                        info.filename,
                    )
                else:
                    metadata_value, metadata_text = _interpret_metadata(raw)

    topology = _decode_topology(target, members, file_size, log)
    if topology is not None:
        graphs = topology.graphs
        tensors = topology.tensors
    else:
        graphs = [Graph(id=ROOT_GRAPH_ID, name="container")]
        tensors = [
            TensorRef(
                initializer_id(ROOT_GRAPH_ID, member.info.filename),
                "uint8",
                (member.info.file_size,),
                Storage.EMBEDDED_RAW,
                payload=PayloadRange(member.data_offset, member.info.file_size),
            )
            for member in members.values()
            if member.stored and member.data_offset is not None
        ]

    container: dict[str, JsonValue] = {
        "member_count": total_members,
        "members": list(member_rows),
    }
    if metadata_format_version is not None:
        container["metadata_format_version"] = metadata_format_version
    if metadata_value is not None:
        container["metadata"] = metadata_value
    elif metadata_text is not None:
        container["metadata_text"] = metadata_text

    extensions: list[tuple[str, JsonValue]] = [("x-dlc.container", container)]
    if topology is not None and topology.quantization:
        extensions.append(("x-dlc.quantization", dict(topology.quantization)))

    contract = contract_for(ArtifactKind.QUALCOMM_DLC)
    return Document(
        source=ArtifactRef(
            path=str(target),
            content_hash=hash_file(target),
            byte_size=file_size,
        ),
        artifact_kind=ArtifactKind.QUALCOMM_DLC,
        capabilities=contract.statuses,
        graphs=graphs,
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
        extensions=extensions,
    )
