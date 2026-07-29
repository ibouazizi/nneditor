"""Artifact-kind detection by content, not by extension (P6.5/P7.5).

Architectural rule 2 makes format support an artifact-and-capability matrix
rather than an extension list, so opening dispatches on what a file *is*:
its magic bytes, and for zip containers the members it holds. Extensions are
only a tie-breaker for formats whose content is genuinely ambiguous (textual
MLIR versus arbitrary text).
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from nneditor.ir.capabilities import ArtifactKind

__all__ = ["DetectionError", "detect_artifact_kind"]

_ZIP_MAGIC = b"PK\x03\x04"
_ONNX_PROTO_HINT = b"\x08"  # ModelProto.ir_version, field 1 varint


class DetectionError(ValueError):
    """Raised when a file matches no supported artifact contract."""


def _zip_kind(path: Path) -> ArtifactKind:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as error:
        raise DetectionError(f"{path.name} is not a readable archive") from error
    # data.pkl wins over archive_format when both are present: a torch
    # checkpoint always holds data.pkl, while anyone can append a spare
    # archive_format member to reroute it into the PT2 parser.
    if any(name.endswith("data.pkl") for name in names):
        # Both plain state dicts and pickled FX modules live here; the FX
        # reader is the one that can say which, so the caller tries the
        # checkpoint path first and falls back on a scan refusal.
        return ArtifactKind.PYTORCH_STATE_DICT
    if any(name.endswith("archive_format") for name in names):
        return ArtifactKind.PYTORCH_EXPORTED_PROGRAM
    raise DetectionError(
        f"{path.name} is a zip archive but holds no recognized model container"
    )


def _leading_content(prefix: bytes) -> bytes:
    """The first content-bearing line, with ``//`` comment lines skipped.

    StableHLO exporters routinely emit header comments before ``module``, so
    the probe must look at the first non-comment line.
    """
    for line in prefix.split(b"\n"):
        stripped = line.lstrip()
        if not stripped or stripped.startswith(b"//"):
            continue
        return stripped
    return b""


def detect_artifact_kind(path: Path | str) -> ArtifactKind:
    """The artifact contract a file satisfies, decided from its content."""
    target = Path(path)
    with open(target, "rb") as handle:
        prefix = handle.read(4096)
    if not prefix:
        raise DetectionError(f"{target.name} is empty")
    if prefix.startswith(_ZIP_MAGIC):
        return _zip_kind(target)
    # safetensors: an 8-byte little-endian header length that fits inside
    # the file, followed immediately by '{'. Without the length check any
    # file with a '{' at offset 8 would claim the format.
    if len(prefix) > 8 and prefix[8:9] == b"{":
        (header_length,) = struct.unpack("<Q", prefix[:8])
        if 8 + header_length <= target.stat().st_size:
            return ArtifactKind.SAFETENSORS
    content = _leading_content(prefix)
    if content.startswith(b"module") or content.startswith(b'"builtin.module"'):
        return ArtifactKind.JAX_STABLEHLO
    if prefix.startswith(_ONNX_PROTO_HINT):
        return ArtifactKind.ONNX_MODEL
    if target.suffix.lower() in {".onnx", ".pb"}:
        return ArtifactKind.ONNX_MODEL
    if target.suffix.lower() in {".mlir", ".stablehlo"}:
        return ArtifactKind.JAX_STABLEHLO
    raise DetectionError(
        f"{target.name} matches no supported artifact contract; see "
        "docs/artifact-capabilities.md for the accepted inputs"
    )
