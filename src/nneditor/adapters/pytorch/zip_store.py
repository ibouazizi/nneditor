"""Range-readable access to zip members (P6.1 foundation).

PyTorch containers — ``.pt2`` archives and ``.pt`` checkpoints — are zip
files whose tensor storages torch writes *uncompressed*. That makes every
storage a contiguous byte range inside the artifact, which is exactly what
the lazy tensor store needs: a member's absolute payload span is derived
from its local file header without reading the payload.

The stdlib ``zipfile`` parses the central directory (pure Python, no code
execution); this module only adds the local-header arithmetic the stdlib
does not expose. Compressed members are still *representable* — they simply
come back without a span, and the caller discloses the full-parse cost the
same way embedded-typed ONNX tensors do.
"""

from __future__ import annotations

import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ZipMember", "ZipStoreError", "read_member", "zip_members"]

_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_MAGIC = b"PK\x03\x04"


class ZipStoreError(ValueError):
    """Raised when an archive member cannot be located or verified."""


@dataclass(frozen=True, slots=True)
class ZipMember:
    """One archive entry and, when stored uncompressed, its payload span."""

    name: str
    size: int
    compressed: bool
    data_offset: int | None
    compressed_size: int

    @property
    def span(self) -> tuple[int, int] | None:
        """Absolute ``(offset, length)`` of the raw payload, if range-readable."""
        if self.compressed or self.data_offset is None:
            return None
        return (self.data_offset, self.size)


def zip_members(path: Path) -> dict[str, ZipMember]:
    """Every member of the archive, keyed by name.

    The data offset comes from each member's *local* header — central
    directory name/extra lengths can legally differ, so trusting them would
    read the wrong bytes.
    """
    members: dict[str, ZipMember] = {}
    try:
        archive_size = path.stat().st_size
        with zipfile.ZipFile(path) as archive, open(path, "rb") as handle:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                handle.seek(info.header_offset)
                raw = handle.read(_LOCAL_HEADER.size)
                if len(raw) != _LOCAL_HEADER.size:
                    raise ZipStoreError(f"truncated local header for {info.filename!r}")
                fields = _LOCAL_HEADER.unpack(raw)
                if fields[0] != _LOCAL_MAGIC:
                    raise ZipStoreError(
                        f"member {info.filename!r} has a malformed local header"
                    )
                name_length, extra_length = fields[9], fields[10]
                data_offset = (
                    info.header_offset + _LOCAL_HEADER.size + name_length + extra_length
                )
                compressed = info.compress_type != zipfile.ZIP_STORED
                # The central directory's sizes are attacker-writable
                # independently of the local header and the file itself, so
                # a member's claimed payload span is verified before anyone
                # range-reads through it.
                if info.file_size < 0 or info.compress_size < 0:
                    raise ZipStoreError(
                        f"member {info.filename!r} declares a negative size"
                    )
                stored_bytes = info.compress_size if compressed else info.file_size
                if data_offset + stored_bytes > archive_size:
                    raise ZipStoreError(
                        f"member {info.filename!r} claims {stored_bytes} bytes "
                        f"at offset {data_offset}, past the end of the "
                        f"{archive_size}-byte archive"
                    )
                if not compressed and info.compress_size != info.file_size:
                    raise ZipStoreError(
                        f"member {info.filename!r} is stored uncompressed but "
                        f"declares {info.compress_size} compressed and "
                        f"{info.file_size} uncompressed bytes"
                    )
                has_data_descriptor = bool(fields[2] & 0x0008)
                local_compress_size, local_file_size = fields[7], fields[8]
                # Data-descriptor members carry their sizes after the payload
                # and zip64 members carry them in the extra field; only a
                # plainly declared local size can be cross-checked.
                uses_zip64 = 0xFFFFFFFF in (local_compress_size, local_file_size)
                if (
                    not has_data_descriptor
                    and not uses_zip64
                    and (
                        local_compress_size != info.compress_size
                        or local_file_size != info.file_size
                    )
                ):
                    raise ZipStoreError(
                        f"member {info.filename!r} has sizes that disagree "
                        "between its local header and the central directory"
                    )
                members[info.filename] = ZipMember(
                    name=info.filename,
                    size=info.file_size,
                    compressed=compressed,
                    data_offset=None if compressed else data_offset,
                    compressed_size=info.compress_size,
                )
    except zipfile.BadZipFile as error:
        raise ZipStoreError(f"{path.name} is not a readable zip archive") from error
    return members


def read_member(path: Path, member: ZipMember, *, limit: int | None = None) -> bytes:
    """Read one member's bytes (decompressing when required).

    ``limit`` guards structural reads: a member claiming to be metadata but
    holding gigabytes is refused before allocation.
    """
    if limit is not None and member.size > limit:
        raise ZipStoreError(
            f"member {member.name!r} is {member.size} bytes; the limit for "
            f"this read is {limit}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read(member.name)
    except (zipfile.BadZipFile, KeyError, RuntimeError, OSError, zlib.error) as error:
        raise ZipStoreError(
            f"member {member.name!r} could not be read: {error}"
        ) from error
