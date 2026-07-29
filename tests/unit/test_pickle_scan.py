"""The restricted pickle vocabulary, opcode by opcode (P6.4).

These exercise the scanner directly with pickles built by the stdlib, which
is the only way to reach protocol variants (long binputs, framing, memoized
strings) that torch's writer does not happen to emit.
"""

from __future__ import annotations

import pickle
import re

import pytest

from nneditor.adapters.pytorch.pickle_scan import (
    PickleScanError,
    StorageRef,
    TensorRecord,
    scan_state_dict,
)


def roundtrip(value: object, protocol: int = 2) -> dict[str, object]:
    return scan_state_dict(pickle.dumps(value, protocol=protocol)).mapping


class TestVocabulary:
    @pytest.mark.parametrize("protocol", [2, 3, 4, 5])
    def test_plain_data_survives_every_protocol(self, protocol: int) -> None:
        payload = {
            "text": "hello",
            "unicode": "héllo",
            "int": 7,
            "big": 2**40,
            "negative": -3,
            "float": 1.5,
            "true": True,
            "false": False,
            "none": None,
            "list": [1, 2, 3],
            "tuple1": (1,),
            "tuple2": (1, 2),
            "tuple3": (1, 2, 3),
            "tuple4": (1, 2, 3, 4),
            "nested": {"inner": {"deep": [1, "two"]}},
        }
        assert roundtrip(payload, protocol) == payload

    def test_empty_containers(self) -> None:
        assert roundtrip({"empty": {}, "list": [], "tuple": ()}) == {
            "empty": {},
            "list": [],
            "tuple": (),
        }

    def test_a_non_mapping_root_is_refused(self) -> None:
        with pytest.raises(PickleScanError, match="not a mapping"):
            scan_state_dict(pickle.dumps([1, 2, 3]))

    def test_truncated_streams_are_refused(self) -> None:
        payload = pickle.dumps({"a": 1})
        with pytest.raises(PickleScanError):
            scan_state_dict(payload[:-1])

    def test_unknown_opcodes_are_refused(self) -> None:
        # `i` (INST) is a legal pickle opcode this vocabulary excludes.
        with pytest.raises(PickleScanError, match="refusing pickle opcode"):
            scan_state_dict(b"\x80\x02}q\x00icopy_reg\n_reconstructor\n.")

    def test_disallowed_globals_are_refused_by_name(self) -> None:
        with pytest.raises(
            PickleScanError, match=re.escape("refusing global os.system")
        ):
            scan_state_dict(b"\x80\x02cos\nsystem\n.")

    def test_build_state_on_a_non_mapping_is_refused(self) -> None:
        # OrderedDict()… then BUILD with a string state.
        stream = (
            b"\x80\x02ccollections\nOrderedDict\nq\x00)Rq\x01X\x03\x00\x00\x00badb."
        )
        with pytest.raises(PickleScanError, match="non-mapping object state"):
            scan_state_dict(stream)


class TestTensorRecords:
    def test_geometry_and_contiguity(self) -> None:
        storage = StorageRef("0", "float32", "cpu", 12)
        contiguous = TensorRecord(storage, 0, (4, 3), (3, 1), False)
        assert contiguous.is_contiguous
        transposed = TensorRecord(storage, 0, (4, 3), (1, 4), False)
        assert not transposed.is_contiguous
        # A length-1 dimension may carry any stride and stays contiguous.
        squeezed = TensorRecord(storage, 0, (1, 12), (999, 1), False)
        assert squeezed.is_contiguous

    def test_rebuild_arguments_are_validated(self) -> None:
        base = (
            b"\x80\x02}q\x00X\x01\x00\x00\x00tq\x01"
            b"ctorch._utils\n_rebuild_tensor_v2\nq\x02"
        )
        # Too few arguments for the constructor.
        with pytest.raises(PickleScanError, match="too few arguments"):
            scan_state_dict(base + b")Rq\x03s.")

    def test_persistent_ids_must_be_storage_tuples(self) -> None:
        stream = b"\x80\x02}q\x00X\x01\x00\x00\x00tq\x01X\x03\x00\x00\x00badQs."
        with pytest.raises(PickleScanError, match="unsupported persistent"):
            scan_state_dict(stream)
