"""Bounded, pickle-free NPY and named NPZ model-input files."""

from __future__ import annotations

import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nneditor.input_generation import (
    MAX_GENERATED_TENSOR_BYTES,
    GeneratedTensor,
    InputGenerationError,
)


def read_input_array(path: Path, input_name: str) -> NDArray[np.generic]:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"))
    if path.suffix.lower() != ".npz":
        raise ValueError("Trace tensor inputs must be safe .npy or .npz arrays.")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 1024:
            raise ValueError("NPZ inputs are limited to 1,024 members.")
        matches = [m for m in members if m.filename == input_name + ".npy"]
        if not matches and len(members) == 1 and members[0].filename.endswith(".npy"):
            matches = members
        if len(matches) != 1:
            raise ValueError(f"NPZ input needs one unambiguous {input_name!r} array.")
        member = matches[0]
        if member.file_size > MAX_GENERATED_TENSOR_BYTES + 64 * 1024:
            raise ValueError("An NPZ input array exceeds the 512 MiB limit.")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("NPZ inputs need NPY version 1 or 2 arrays.")
            if dtype.hasobject or dtype.kind not in "biuf":
                raise ValueError("NPZ inputs must contain real numeric arrays.")
            size = math.prod(shape) * dtype.itemsize
            if size > MAX_GENERATED_TENSOR_BYTES:
                raise ValueError("An NPZ input array exceeds the 512 MiB limit.")
            if size != member.file_size - stream.tell():
                raise ValueError("The NPZ array header and payload sizes disagree.")
        with archive.open(member) as stream:
            return np.asarray(np.load(stream, allow_pickle=False))


def publish_input_archive(
    tensor: GeneratedTensor, destination: Path, input_name: str
) -> GeneratedTensor:
    """Publish a generated NPY as an atomic NPZ with a model-input key."""
    target = destination.expanduser().resolve()
    if target.suffix.lower() != ".npz" or not target.parent.is_dir():
        raise InputGenerationError("Choose an .npz file in an existing directory.")
    array = read_input_array(tensor.path, input_name)
    if array.nbytes > MAX_GENERATED_TENSOR_BYTES:
        raise InputGenerationError("Generated input exceeds the 512 MiB limit.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            with zipfile.ZipFile(handle, "w") as archive:
                with archive.open(input_name + ".npy", "w", force_zip64=True) as member:
                    np.save(member, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return GeneratedTensor(
        target,
        tensor.shape,
        tensor.dtype,
        tensor.byte_size,
        tensor.source + f"; NPZ array {input_name!r}",
    )
