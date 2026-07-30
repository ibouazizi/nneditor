"""PyTorch dtype vocabularies mapped onto IR element-type names (P6.1).

Two distinct enumerations exist in the wild and must not be confused:

* the **serde** ``ScalarType`` used by exported-program JSON
  (``torch._export.serde.schema``), where ``FLOAT`` is 7, and
* the **storage class names** that appear in checkpoints
  (``FloatStorage`` and friends).

Both map onto the same IR element-type names the rest of NNEditor already
understands (the vocabulary the ONNX adapter established), so tensors from
any framework flow through the same store, statistics, and editing code.
"""

from __future__ import annotations

from typing import Final

from nneditor.ir.dtypes import dtype_info

__all__ = [
    "SERDE_SCALAR_TYPES",
    "STORAGE_CLASS_TYPES",
    "element_width_bytes",
]

# torch._export.serde.schema.ScalarType (verified against torch 2.13).
SERDE_SCALAR_TYPES: Final[dict[int, str]] = {
    1: "uint8",
    2: "int8",
    3: "int16",
    4: "int32",
    5: "int64",
    6: "float16",
    7: "float32",
    8: "float64",
    9: "complex32",
    10: "complex64",
    11: "complex128",
    12: "bool",
    13: "bfloat16",
    16: "uint16",
    17: "uint32",
    18: "uint64",
}

# Legacy storage class names as they appear in torch.save pickles.
STORAGE_CLASS_TYPES: Final[dict[str, str]] = {
    "DoubleStorage": "float64",
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
    "BFloat16Storage": "bfloat16",
    "ComplexDoubleStorage": "complex128",
    "ComplexFloatStorage": "complex64",
}

# torch.save with the modern serialization writes UntypedStorage plus a dtype
# carried through _rebuild_tensor_v2; the dtype reprs seen in pickles:
TORCH_DTYPE_NAMES: Final[dict[str, str]] = {
    "torch.float64": "float64",
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.int16": "int16",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.uint16": "uint16",
    "torch.uint32": "uint32",
    "torch.uint64": "uint64",
    "torch.bool": "bool",
    "torch.complex64": "complex64",
    "torch.complex128": "complex128",
}


def element_width_bytes(element_type: str) -> int | None:
    """Whole bytes per element, answered by the shared dtype registry."""
    info = dtype_info(element_type)
    return None if info is None else info.byte_width
