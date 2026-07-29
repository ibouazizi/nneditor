"""A restricted, non-executing scan of torch checkpoint pickles (P6.4).

The security checklist forbids pickle *execution* in safe artifact mode, but
a ``torch.save`` state dict is structurally just data: dict literals,
strings, ints, tuples, storage references, and two constructor calls
(``OrderedDict`` and ``_rebuild_tensor_v2``). This module interprets exactly
that vocabulary with a tiny stack machine:

* Only allow-listed opcodes are interpreted; anything else aborts the scan.
* ``GLOBAL``/``STACK_GLOBAL`` resolve to inert markers from an allowlist —
  an unknown import aborts, it is never resolved against real modules.
* ``REDUCE`` never calls anything: for the two allow-listed constructors it
  builds plain records, and every other callee aborts.

The result is the same information ``torch.load(weights_only=True)`` would
yield — tensor names, dtypes, shapes, strides, and storage references —
obtained without torch and without executing a single artifact-controlled
instruction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final

from nneditor.adapters.pytorch.scalar_types import (
    STORAGE_CLASS_TYPES,
    TORCH_DTYPE_NAMES,
)

__all__ = [
    "PickleScanError",
    "ScanResult",
    "StorageRef",
    "TensorRecord",
    "scan_state_dict",
]


class PickleScanError(ValueError):
    """Raised when the pickle steps outside the weights-only vocabulary."""


@dataclass(frozen=True, slots=True)
class StorageRef:
    """A persistent storage reference: where a tensor's bytes live."""

    key: str
    element_type: str | None
    location: str
    numel: int


@dataclass(frozen=True, slots=True)
class TensorRecord:
    """One rebuilt tensor, described without materializing anything."""

    storage: StorageRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool

    @property
    def is_contiguous(self) -> bool:
        expected = 1
        for dimension, declared in zip(
            reversed(self.shape), reversed(self.stride), strict=True
        ):
            if dimension > 1 and declared != expected:
                return False
            expected *= dimension
        return True


@dataclass(frozen=True, slots=True)
class ScanResult:
    """The interpreted mapping plus any object state applied to it.

    ``torch.save`` attaches a state dict's ``_metadata`` (per-module version
    info) with the ``BUILD`` opcode. It is data, not behaviour, so it is
    carried here and preserved in an extension rather than discarded.
    """

    mapping: dict[str, object]
    root_state: dict[str, object]


class _Global:
    """An inert stand-in for an allow-listed import."""

    __slots__ = ("module", "name")

    def __init__(self, module: str, name: str) -> None:
        self.module = module
        self.name = name


class _Mark:
    __slots__ = ()


_ALLOWED_GLOBALS: Final = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch.serialization", "_get_layout"),
}


def _is_allowed_global(module: str, name: str) -> bool:
    if (module, name) in _ALLOWED_GLOBALS:
        return True
    if module == "torch" and (
        name in STORAGE_CLASS_TYPES
        or name == "UntypedStorage"
        or f"torch.{name}" in TORCH_DTYPE_NAMES
    ):
        return True
    return module == "torch.storage" and name == "TypedStorage"


_MAX_MEMO: Final = 1_000_000
_MAX_STACK: Final = 1_000_000


def scan_state_dict(payload: bytes) -> ScanResult:
    """Interpret a weights-only pickle into plain records.

    Returns the top-level mapping and any state applied to it; values are
    :class:`TensorRecord`, nested dicts/lists, strings, numbers, bools, or
    ``None``. Raises :class:`PickleScanError` the moment anything outside the
    vocabulary appears.
    """
    stack: list[object] = []
    memo: dict[int, object] = {}
    states: list[tuple[object, dict[str, object]]] = []
    position = 0
    size = len(payload)

    def need(count: int) -> bytes:
        nonlocal position
        if position + count > size:
            raise PickleScanError("truncated pickle stream")
        raw = payload[position : position + count]
        position += count
        return raw

    def decode_text(raw: bytes) -> str:
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PickleScanError(
                f"pickle text argument is not valid UTF-8: {error}"
            ) from error

    def read_line() -> str:
        nonlocal position
        end = payload.find(b"\n", position)
        if end < 0:
            raise PickleScanError("unterminated text argument")
        raw = payload[position:end]
        position = end + 1
        return decode_text(raw)

    def push(value: object) -> None:
        if len(stack) > _MAX_STACK:
            raise PickleScanError("pickle stack exceeds the safety ceiling")
        stack.append(value)

    def pop() -> object:
        if not stack:
            raise PickleScanError("pickle stack underflow")
        return stack.pop()

    def top() -> object:
        if not stack:
            raise PickleScanError("pickle stack underflow")
        return stack[-1]

    def memo_get(index: int) -> object:
        if index not in memo:
            raise PickleScanError(
                f"memo index {index} is read before anything was stored there"
            )
        return memo[index]

    def memo_put(index: int) -> None:
        if index > _MAX_MEMO or len(memo) >= _MAX_MEMO:
            raise PickleScanError("memo index exceeds the safety ceiling")
        memo[index] = top()

    def pop_mark() -> list[object]:
        items: list[object] = []
        while stack:
            item = stack.pop()
            if isinstance(item, _Mark):
                items.reverse()
                return items
            items.append(item)
        raise PickleScanError("MARK underflow")

    def reduce(callee: object, args: tuple[object, ...]) -> object:
        if not isinstance(callee, _Global):
            raise PickleScanError("REDUCE of a non-import callee")
        target = (callee.module, callee.name)
        if target == ("collections", "OrderedDict"):
            return {}
        if target == ("torch._utils", "_rebuild_parameter"):
            if not args or not isinstance(args[0], TensorRecord):
                raise PickleScanError("_rebuild_parameter without a tensor")
            return args[0]
        if target == ("torch._utils", "_rebuild_tensor_v2"):
            if len(args) < 5:
                raise PickleScanError("_rebuild_tensor_v2 with too few arguments")
            storage, offset, shape, stride = args[0], args[1], args[2], args[3]
            requires_grad = args[4] if isinstance(args[4], bool) else False
            if not isinstance(storage, StorageRef):
                raise PickleScanError("tensor rebuilt from a non-storage source")
            if (
                not isinstance(offset, int)
                or not isinstance(shape, tuple)
                or not isinstance(stride, tuple)
                or not all(isinstance(item, int) for item in shape)
                or not all(isinstance(item, int) for item in stride)
                or len(shape) != len(stride)
            ):
                raise PickleScanError("tensor rebuilt with malformed geometry")
            if offset < 0 or any(item < 0 for item in shape):
                raise PickleScanError(
                    "tensor rebuilt with a negative storage offset or dimension"
                )
            return TensorRecord(
                storage=storage,
                storage_offset=offset,
                shape=tuple(int(item) for item in shape),
                stride=tuple(int(item) for item in stride),
                requires_grad=requires_grad,
            )
        raise PickleScanError(
            f"refusing to interpret constructor {callee.module}.{callee.name}"
        )

    def persistent(reference: object) -> StorageRef:
        if (
            not isinstance(reference, tuple)
            or len(reference) < 5
            or reference[0] != "storage"
        ):
            raise PickleScanError("unsupported persistent reference")
        marker = reference[1]
        element_type: str | None = None
        if isinstance(marker, _Global):
            element_type = STORAGE_CLASS_TYPES.get(marker.name)
        key, location, numel = reference[2], reference[3], reference[4]
        if (
            not isinstance(key, str)
            or not isinstance(location, str)
            or not isinstance(numel, int)
        ):
            raise PickleScanError("malformed storage reference fields")
        return StorageRef(
            key=key, element_type=element_type, location=location, numel=numel
        )

    while True:
        opcode = need(1)
        if opcode == b"\x80":  # PROTO
            need(1)
        elif opcode == b"\x95":  # FRAME
            need(8)
        elif opcode == b"(":  # MARK
            push(_Mark())
        elif opcode == b"}":  # EMPTY_DICT
            push({})
        elif opcode == b"]":  # EMPTY_LIST
            push([])
        elif opcode == b")":  # EMPTY_TUPLE
            push(())
        elif opcode == b"c":  # GLOBAL
            module, name = read_line(), read_line()
            if not _is_allowed_global(module, name):
                raise PickleScanError(
                    f"refusing global {module}.{name} in safe artifact mode"
                )
            push(_Global(module, name))
        elif opcode == b"\x93":  # STACK_GLOBAL
            stacked_name, stacked_module = pop(), pop()
            if not isinstance(stacked_module, str) or not isinstance(stacked_name, str):
                raise PickleScanError("STACK_GLOBAL without string operands")
            module, name = stacked_module, stacked_name
            if not _is_allowed_global(module, name):
                raise PickleScanError(
                    f"refusing global {module}.{name} in safe artifact mode"
                )
            push(_Global(module, name))
        elif opcode == b"q":  # BINPUT
            (index,) = need(1)
            memo_put(index)
        elif opcode == b"r":  # LONG_BINPUT
            (index,) = struct.unpack("<I", need(4))
            memo_put(index)
        elif opcode == b"\x94":  # MEMOIZE
            memo_put(len(memo))
        elif opcode == b"h":  # BINGET
            (index,) = need(1)
            push(memo_get(index))
        elif opcode == b"j":  # LONG_BINGET
            (index,) = struct.unpack("<I", need(4))
            push(memo_get(index))
        elif opcode == b"X":  # BINUNICODE
            (length,) = struct.unpack("<I", need(4))
            push(decode_text(need(length)))
        elif opcode == b"\x8c":  # SHORT_BINUNICODE
            (length,) = need(1)
            push(decode_text(need(length)))
        elif opcode == b"K":  # BININT1
            push(need(1)[0])
        elif opcode == b"M":  # BININT2
            (value,) = struct.unpack("<H", need(2))
            push(value)
        elif opcode == b"J":  # BININT
            (value,) = struct.unpack("<i", need(4))
            push(value)
        elif opcode == b"\x8a":  # LONG1
            (length,) = need(1)
            push(int.from_bytes(need(length), "little", signed=True))
        elif opcode == b"G":  # BINFLOAT
            (value,) = struct.unpack(">d", need(8))
            push(value)
        elif opcode == b"\x88":  # NEWTRUE
            push(True)
        elif opcode == b"\x89":  # NEWFALSE
            push(False)
        elif opcode == b"N":  # NONE
            push(None)
        elif opcode == b"\x85":  # TUPLE1
            push((pop(),))
        elif opcode == b"\x86":  # TUPLE2
            second, first = pop(), pop()
            push((first, second))
        elif opcode == b"\x87":  # TUPLE3
            third, second, first = pop(), pop(), pop()
            push((first, second, third))
        elif opcode == b"t":  # TUPLE
            push(tuple(pop_mark()))
        elif opcode == b"Q":  # BINPERSID
            push(persistent(pop()))
        elif opcode == b"R":  # REDUCE
            args, callee = pop(), pop()
            if not isinstance(args, tuple):
                raise PickleScanError("REDUCE without a tuple of arguments")
            push(reduce(callee, args))
        elif opcode == b"s":  # SETITEM
            value, key, container = pop(), pop(), top()
            if not isinstance(container, dict):
                raise PickleScanError("SETITEM on a non-dict")
            container[key] = value
        elif opcode == b"u":  # SETITEMS
            items = pop_mark()
            container = top()
            if not isinstance(container, dict) or len(items) % 2:
                raise PickleScanError("SETITEMS on a non-dict")
            for key, value in zip(items[::2], items[1::2], strict=True):
                container[key] = value
        elif opcode == b"a":  # APPEND
            value, container = pop(), top()
            if not isinstance(container, list):
                raise PickleScanError("APPEND on a non-list")
            container.append(value)
        elif opcode == b"e":  # APPENDS
            items = pop_mark()
            container = top()
            if not isinstance(container, list):
                raise PickleScanError("APPENDS on a non-list")
            container.extend(items)
        elif opcode == b"b":  # BUILD
            # Applying state never calls __setstate__ here: the state is
            # recorded as data next to its target and nothing is invoked.
            # The target itself is held (not its id()) so the pairing cannot
            # be confused by id reuse after garbage collection.
            state = pop()
            target = top()
            if isinstance(state, dict) and isinstance(target, dict):
                states.append(
                    (target, {str(key): value for key, value in state.items()})
                )
            elif state is not None:
                raise PickleScanError(
                    "refusing to apply non-mapping object state in safe mode"
                )
        elif opcode == b"2":  # DUP
            push(top())
        elif opcode == b"0":  # POP
            pop()
        elif opcode == b".":  # STOP
            break
        else:
            raise PickleScanError(
                f"refusing pickle opcode {opcode!r} in safe artifact mode"
            )

    if len(stack) != 1:
        raise PickleScanError("pickle stream left a malformed stack")
    result = stack[0]
    if not isinstance(result, dict):
        raise PickleScanError("checkpoint root is not a mapping")
    root_state: dict[str, object] = {}
    for target, state in states:
        if target is result:
            root_state.update(state)
    return ScanResult(mapping=result, root_state=root_state)
