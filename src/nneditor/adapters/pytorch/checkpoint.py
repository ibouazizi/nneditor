"""Weights-only PyTorch checkpoint sessions (P6.4).

A ``.pt``/``.pth`` checkpoint is a zip whose ``data.pkl`` names tensors and
whose ``data/<key>`` members hold their storages, uncompressed. The
restricted pickle scan provides names, dtypes, and geometry; the zip member
table provides byte ranges — so a checkpoint opens as a weights-only
document with range-readable tensors and *no* invented topology, exactly as
the artifact contract promises.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.pickle_scan import (
    PickleScanError,
    TensorRecord,
    scan_state_dict,
)
from nneditor.adapters.pytorch.scalar_types import element_width_bytes
from nneditor.adapters.pytorch.zip_store import (
    ZipStoreError,
    read_member,
    zip_members,
)
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

__all__ = ["CheckpointError", "open_checkpoint"]

_MAX_PICKLE_BYTES: Final = 256 * 1024 * 1024
_MAX_FLATTEN_ENTRIES: Final = 1_000_000


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be opened as a weights-only session."""


def _flatten(
    mapping: dict[str, object],
) -> tuple[dict[str, TensorRecord], dict[str, JsonValue]]:
    """Flatten the scanned mapping into dotted tensor and extra names.

    The walk is iterative: pickle memos let the mapping share sub-dicts (a
    DAG) or even reference itself, so a naive tree recursion would be
    exponential on hostile sharing and unbounded on cycles. A dict already on
    the current path is a cycle and is refused; a total-visited budget bounds
    DAG blow-up.
    """
    tensors: dict[str, TensorRecord] = {}
    extras: dict[str, JsonValue] = {}
    visited = 0
    on_path: set[int] = {id(mapping)}
    stack: list[tuple[int, str, Iterator[tuple[object, object]]]] = [
        (id(mapping), "", iter(mapping.items()))
    ]
    while stack:
        container_id, prefix, iterator = stack[-1]
        entry = next(iterator, None)
        if entry is None:
            stack.pop()
            on_path.discard(container_id)
            continue
        key, value = entry
        visited += 1
        if visited > _MAX_FLATTEN_ENTRIES:
            raise CheckpointError(
                "the checkpoint mapping expands past the safety ceiling of "
                f"{_MAX_FLATTEN_ENTRIES} entries"
            )
        name = f"{prefix}.{key}" if prefix else f"{key}"
        if isinstance(value, TensorRecord):
            tensors[name] = value
        elif isinstance(value, dict):
            if id(value) in on_path:
                raise CheckpointError(
                    "the checkpoint mapping references itself; refusing the "
                    "self-referential structure"
                )
            on_path.add(id(value))
            stack.append((id(value), name, iter(value.items())))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            extras[name] = value
        else:
            extras[name] = repr(value)
    return tensors, extras


def open_checkpoint(path: Path | str) -> Document:
    """Open a torch checkpoint as a weights-only document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`CheckpointError`; the layered error contract in the session
    depends on adapters never leaking raw exceptions.
    """
    source = Path(path)
    try:
        return _open_checkpoint(source)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError(
            f"{source.name} could not be opened as a checkpoint: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_checkpoint(source: Path) -> Document:
    try:
        members = zip_members(source)
    except ZipStoreError as error:
        raise CheckpointError(str(error)) from error

    pickle_member = next(
        (
            member
            for name, member in members.items()
            if name.endswith("data.pkl") and name.count("/") <= 1
        ),
        None,
    )
    if pickle_member is None:
        raise CheckpointError(
            f"{source.name} holds no data.pkl; it is not a torch zip checkpoint"
        )
    prefix = pickle_member.name.rsplit("data.pkl", 1)[0]

    try:
        payload = read_member(source, pickle_member, limit=_MAX_PICKLE_BYTES)
        scan = scan_state_dict(payload)
        root = scan.mapping
    except (PickleScanError, ZipStoreError) as error:
        raise CheckpointError(
            f"{source.name} could not be read in safe artifact mode: {error}"
        ) from error

    log = DiagnosticLog()
    records, extras = _flatten(root)
    if not records:
        log.add(
            "pytorch.no-tensors",
            Severity.WARNING,
            "The checkpoint mapping holds no tensors.",
            source.name,
        )

    storage_users: dict[str, int] = {}
    for record in records.values():
        storage_users[record.storage.key] = storage_users.get(record.storage.key, 0) + 1

    tensors: list[TensorRef] = []
    for name, record in records.items():
        tensor_id = initializer_id(ROOT_GRAPH_ID, name)
        element_type = record.storage.element_type
        dims = record.shape
        if record.storage_offset < 0 or any(dim < 0 for dim in dims):
            # Defense in depth: the pickle scan already refuses negative
            # geometry, but a payload pointing before its storage or a
            # negative extent must never reach range arithmetic.
            log.add(
                "pytorch.invalid-tensor-geometry",
                Severity.ERROR,
                f"tensor {name!r} declares a negative storage offset or "
                "dimension and is not readable",
                tensor_id,
            )
            continue
        if element_type is None:
            log.add(
                "pytorch.unsupported-storage-dtype",
                Severity.WARNING,
                f"tensor {name!r} uses a storage class this build cannot type",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, "undefined", dims, Storage.ABSENT))
            continue
        width = element_width_bytes(element_type)
        member = members.get(f"{prefix}data/{record.storage.key}")
        span = member.span if member is not None else None
        if member is None or width is None:
            log.add(
                "pytorch.storage-missing",
                Severity.ERROR,
                f"tensor {name!r} references storage "
                f"{record.storage.key!r}, which the archive does not hold",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        if span is None:
            log.add(
                "pytorch.compressed-storage",
                Severity.WARNING,
                f"tensor {name!r} is stored compressed; range reads are not "
                "possible for it",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        if not record.is_contiguous:
            log.add(
                "pytorch.non-contiguous-tensor",
                Severity.WARNING,
                f"tensor {name!r} is a strided view; its bytes are not a "
                "single range and are not materialized in safe mode",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        count = 1
        for dim in dims:
            count *= dim
        offset_bytes = record.storage_offset * width
        length = count * width
        member_offset, member_length = span
        if offset_bytes + length > member_length:
            log.add(
                "pytorch.tensor-range-out-of-bounds",
                Severity.ERROR,
                f"tensor {name!r} spans past its {member_length}-byte storage",
                tensor_id,
            )
            tensors.append(TensorRef(tensor_id, element_type, dims, Storage.ABSENT))
            continue
        if storage_users[record.storage.key] > 1:
            log.add(
                "pytorch.shared-storage",
                Severity.INFO,
                f"tensor {name!r} shares storage {record.storage.key!r} with "
                "other tensors; editing one edits them all",
                tensor_id,
            )
        tensors.append(
            TensorRef(
                tensor_id,
                element_type,
                dims,
                Storage.EMBEDDED_RAW,
                payload=PayloadRange(member_offset + offset_bytes, length),
            )
        )

    extensions: list[tuple[str, JsonValue]] = []
    if extras:
        extensions.append(("x-torch.checkpoint", dict(sorted(extras.items()))))
    if scan.root_state:
        extensions.append(
            (
                "x-torch.state_dict_metadata",
                {key: repr(value) for key, value in sorted(scan.root_state.items())},
            )
        )
    contract = contract_for(ArtifactKind.PYTORCH_STATE_DICT)
    return Document(
        source=ArtifactRef(
            path=str(source),
            content_hash=hash_file(source),
            byte_size=source.stat().st_size,
        ),
        artifact_kind=ArtifactKind.PYTORCH_STATE_DICT,
        capabilities=contract.statuses,
        graphs=[Graph(id=ROOT_GRAPH_ID, name="state_dict")],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("loading_mode", "safe artifact"),
                    ("reader", "restricted weights-only pickle scan"),
                ),
            )
        ],
        diagnostics=tuple(log),
        extensions=extensions,
    )
