"""Weights-only JAX/Flax/Orbax checkpoint sessions (P7.3).

An Orbax checkpoint directory holds one msgpack/JSON tree plus array data;
the portable, universally readable member is the **safetensors** or ``.npy``
payload many JAX pipelines emit alongside. This adapter opens the directory's
readable tensor files as a weights-only document with the *checkpoint tree*
preserved as a path-keyed structure — it never claims topology, because a
parameter tree describes no computation (the Orbax contract says exactly
that).
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.safetensors import open_safetensors
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import (
    ArtifactRef,
    Document,
    Graph,
    JsonValue,
    ProvenanceEntry,
    TensorRef,
)
from nneditor.ir.identity import ROOT_GRAPH_ID
from nneditor.storage.reader import hash_file

__all__ = ["OrbaxError", "open_orbax_checkpoint"]

_MAX_TREE_FILES: Final = 256
_MAX_TREE_FILE_BYTES: Final = 8 * 1024 * 1024


class OrbaxError(ValueError):
    """Raised when a checkpoint directory yields no readable tensors."""


def open_orbax_checkpoint(path: Path | str) -> Document:
    """Open a checkpoint directory (or single safetensors file) weights-only.

    A directory containing ``*.safetensors`` reads through the native
    safetensors path; ``.npy`` members are read directly. The parameter tree
    (``*.json`` metadata) is carried as an extension without interpretation.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`OrbaxError`; the layered error contract in the session depends
    on adapters never leaking raw exceptions.
    """
    target = Path(path)
    try:
        return _open_orbax_checkpoint(target)
    except OrbaxError:
        raise
    except Exception as error:
        raise OrbaxError(
            f"{target.name} could not be opened as a checkpoint: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_orbax_checkpoint(target: Path) -> Document:
    if target.is_file():
        document = open_safetensors(target)
        return document

    if not target.is_dir():
        raise OrbaxError(f"{target} is neither a file nor a directory")

    log = DiagnosticLog()
    safetensor_files = sorted(target.rglob("*.safetensors"))
    tensors: list[TensorRef] = []
    tree: dict[str, JsonValue] = {}

    # The directory contents are untrusted: both the number of metadata
    # files and each file's size are capped so a hostile checkpoint cannot
    # turn the tree walk into unbounded reads.
    metadata_files = sorted(islice(target.rglob("*.json"), _MAX_TREE_FILES + 1))
    if len(metadata_files) > _MAX_TREE_FILES:
        log.add(
            "jax.tree-metadata-truncated",
            Severity.WARNING,
            f"the checkpoint holds more than {_MAX_TREE_FILES} metadata "
            "files; only the first ones are carried",
            target.name,
        )
        metadata_files = metadata_files[:_MAX_TREE_FILES]
    for metadata in metadata_files:
        try:
            if metadata.stat().st_size > _MAX_TREE_FILE_BYTES:
                log.add(
                    "jax.tree-metadata-too-large",
                    Severity.WARNING,
                    f"{metadata.name} exceeds the {_MAX_TREE_FILE_BYTES}-byte "
                    "metadata limit and is skipped",
                    metadata.name,
                )
                continue
            tree[metadata.relative_to(target).as_posix()] = json.loads(
                metadata.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, RecursionError):
            log.add(
                "jax.unreadable-tree-metadata",
                Severity.WARNING,
                f"{metadata.name} is not readable JSON and is skipped",
                metadata.name,
            )

    if not safetensor_files:
        raise OrbaxError(
            f"{target} holds no safetensors payload; export the checkpoint "
            "with a safetensors writer to inspect its weights"
        )

    primary = safetensor_files[0]
    if len(safetensor_files) > 1:
        log.add(
            "jax.sharded-checkpoint",
            Severity.WARNING,
            f"{len(safetensor_files)} shards found; only {primary.name} is "
            "opened in this build",
            target.name,
        )
    document = open_safetensors(primary)
    tensors = list(document.tensors.values())

    extensions: list[tuple[str, JsonValue]] = list(document.extensions)
    if tree:
        extensions.append(("x-orbax.tree", tree))

    contract = contract_for(ArtifactKind.ORBAX_CHECKPOINT)
    return Document(
        source=ArtifactRef(
            path=str(primary),
            content_hash=hash_file(primary),
            byte_size=primary.stat().st_size,
        ),
        artifact_kind=ArtifactKind.ORBAX_CHECKPOINT,
        capabilities=contract.statuses,
        graphs=[Graph(id=ROOT_GRAPH_ID, name="checkpoint")],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(("loading_mode", "safe artifact"),),
            )
        ],
        diagnostics=(*document.diagnostics, *tuple(log)),
        extensions=extensions,
    )
