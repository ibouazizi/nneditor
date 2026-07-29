"""Regression tests for verified analysis-layer defects.

Covers: dominator fixed-point sweep order (declaration-order independence),
near-linear structural-region detection with bounded cancellation latency,
two-pass variance accuracy on offset-heavy data, the bfloat16 decoder,
malformed-payload alignment reporting, the heap-based Kahn ready queue, and
cancellable layout.
"""

from __future__ import annotations

import math
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from nneditor.analysis.detectors import (
    StructuralRegionDetector,
    dominators,
    post_dominators,
)
from nneditor.analysis.layout import (
    LayoutSettings,
    _assign_layers,
    _dataflow_edges,
    layout_graph,
)
from nneditor.analysis.statistics import (
    STATISTICS_VERSION,
    compute_statistics,
    decode_packed,
    element_width,
    statistics_from_json,
)
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.core import ExternalRef, Graph, Node, Storage, TensorRef, Value
from nneditor.ir.identity import NodeIdStability
from nneditor.storage.store import TensorStore, TensorUnavailableError
from tests.unit.test_hierarchy import make_document
from tests.unit.test_statistics import store_for_raw
from tests.unit.test_tensor_store import handcrafted_document

_PERF_BUDGET_SECONDS = 2.0
"""Generous wall-clock guard; the quadratic/cubic code paths took 2-20 s."""


def chain_graph(count: int, *, reverse_declared: bool = False) -> Graph:
    """``n0 -> n1 -> ...``, optionally declared last-first."""
    values = tuple(Value(id=f"v{i}", name=f"v{i}") for i in range(count))
    indices = range(count - 1, -1, -1) if reverse_declared else range(count)
    nodes = tuple(
        Node(
            id=f"n{i}",
            id_stability=NodeIdStability.NAMED,
            op_type="Relu",
            inputs=() if i == 0 else (f"v{i - 1}",),
            outputs=(f"v{i}",),
        )
        for i in indices
    )
    return Graph(id="g:main", name="chain", nodes=nodes, values=values)


def residual_chain(blocks: int) -> Graph:
    """``blocks`` residual blocks: v -> Gemm -> Add(v, gemm_out) -> next."""
    values = [Value(id="v0", name="v0")]
    nodes = [
        Node(
            id="src",
            id_stability=NodeIdStability.NAMED,
            op_type="Identity",
            inputs=(),
            outputs=("v0",),
        )
    ]
    previous = "v0"
    for i in range(blocks):
        values.append(Value(id=f"g{i}", name=f"g{i}"))
        values.append(Value(id=f"a{i}", name=f"a{i}"))
        nodes.append(
            Node(
                id=f"gemm{i}",
                id_stability=NodeIdStability.NAMED,
                op_type="Gemm",
                inputs=(previous,),
                outputs=(f"g{i}",),
            )
        )
        nodes.append(
            Node(
                id=f"add{i}",
                id_stability=NodeIdStability.NAMED,
                op_type="Add",
                inputs=(previous, f"g{i}"),
                outputs=(f"a{i}",),
            )
        )
        previous = f"a{i}"
    return Graph(id="g:main", name="residual", nodes=tuple(nodes), values=tuple(values))


class _CountingToken(CancellationToken):
    """Never cancels; records how many checkpoints the work passed."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def raise_if_cancelled(self) -> None:
        self.calls += 1
        super().raise_if_cancelled()


class _CancelAfterToken(CancellationToken):
    """Cancels itself once ``allowed`` checkpoints have passed."""

    def __init__(self, allowed: int) -> None:
        super().__init__()
        self.allowed = allowed

    def raise_if_cancelled(self) -> None:
        if self.allowed <= 0:
            raise OperationCancelled
        self.allowed -= 1


class TestDominatorSweepOrder:
    """Finding 1: fixed-point sweeps must not depend on declaration order."""

    def test_reverse_declared_chain_matches_forward_declaration(self) -> None:
        forward = chain_graph(40)
        backward = chain_graph(40, reverse_declared=True)
        assert dominators(forward) == dominators(backward)
        assert post_dominators(forward) == post_dominators(backward)
        assert dominators(forward)["n39"] == frozenset(f"n{i}" for i in range(40))

    def test_reverse_declared_chain_is_fast(self) -> None:
        graph = chain_graph(800, reverse_declared=True)
        start = time.perf_counter()
        sets = dominators(graph)
        elapsed = time.perf_counter() - start
        assert len(sets["n799"]) == 800
        assert elapsed < _PERF_BUDGET_SECONDS, (
            f"declaration-order-sensitive sweep suspected: {elapsed:.2f} s "
            "for an 800-node reverse-declared chain (was 3.9 s before the "
            "reverse-postorder sweep)"
        )

    def test_dominators_are_cancellable(self) -> None:
        graph = chain_graph(600, reverse_declared=True)
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            StructuralRegionDetector().detect(make_document(graph), graph, token)


class TestStructuralRegionDetector:
    """Finding 2: residual chains must be detected in near-linear time."""

    def test_residual_chain_regions_are_exact(self) -> None:
        graph = residual_chain(2)
        candidates = StructuralRegionDetector().detect(make_document(graph), graph)
        expected = {
            frozenset(("src", "gemm0", "add0")),
            frozenset(("add0", "gemm1", "add1")),
        }
        sese = {c.members for c in candidates if c.label.startswith("Branch")}
        residual = {c.members for c in candidates if c.label == "Residual region"}
        assert sese == expected
        assert residual == expected
        assert len(candidates) == 4

    def test_residual_chain_detection_is_near_linear(self) -> None:
        graph = residual_chain(400)  # 801 nodes
        detector = StructuralRegionDetector()
        start = time.perf_counter()
        candidates = detector.detect(make_document(graph), graph)
        elapsed = time.perf_counter() - start
        assert len(candidates) == 2 * 400, "one SESE and one residual per block"
        assert elapsed < _PERF_BUDGET_SECONDS, (
            f"cubic nearest-common-post-dominator search suspected: "
            f"{elapsed:.2f} s for an 800-node residual chain (was 20.7 s "
            "before the dominator-chain lookup)"
        )

    def test_cancellation_latency_is_bounded_per_node(self) -> None:
        graph = residual_chain(40)
        counting = _CountingToken()
        StructuralRegionDetector().detect(make_document(graph), graph, counting)
        # Both the branch loop and the residual-Add loop must check the token
        # on every node, so a full run passes at least two checkpoints per
        # node — a cancelled token can never wait out a whole loop.
        assert counting.calls >= 2 * len(graph.nodes)
        mid_run = _CancelAfterToken(counting.calls // 2)
        with pytest.raises(OperationCancelled):
            StructuralRegionDetector().detect(make_document(graph), graph, mid_run)


class TestVarianceAccuracy:
    """Finding 3: one-pass variance cancelled catastrophically on offsets."""

    def test_offset_heavy_data_matches_numpy(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(11)
        values = (1e4 + rng.standard_normal(4096) * 1e-2).astype(np.float32)
        store, tensor_id = store_for_raw(
            tmp_path, values.tobytes(), dims=[4096], data_type=1
        )
        with store:
            stats = compute_statistics(store, tensor_id)
        reference = values.astype(np.float64)
        assert stats.mean == pytest.approx(float(reference.mean()), rel=1e-9)
        assert stats.std == pytest.approx(float(reference.std()), rel=1e-4)

    def test_version_was_bumped_so_stale_sidecars_recompute(self) -> None:
        assert STATISTICS_VERSION >= 2, "variance fix must invalidate v1 results"
        assert statistics_from_json({"version": 1, "tensor_id": "t"}) is None


class TestBfloat16Decoding:
    """Finding 4: every adapter emits bfloat16; statistics must decode it."""

    def test_known_bit_patterns_decode_exactly(self) -> None:
        patterns = (0x0000, 0x8000, 0x3F80, 0xBF80, 0x4049, 0x7F80, 0xFF80)
        raw = struct.pack(f"<{len(patterns)}H", *patterns)
        decoded = decode_packed("bfloat16", raw)
        assert decoded is not None
        assert list(decoded) == [
            0.0,
            -0.0,
            1.0,
            -1.0,
            3.140625,
            math.inf,
            -math.inf,
        ]
        assert math.copysign(1.0, decoded[1]) == -1.0, "sign of -0.0 survives"

    def test_nan_and_denormal_patterns_survive_widening(self) -> None:
        raw = struct.pack("<3H", 0x7FC0, 0x0080, 0x0001)
        decoded = decode_packed("bfloat16", raw)
        assert decoded is not None
        nan, smallest_normal, denormal = decoded
        assert math.isnan(nan)
        assert smallest_normal == 2.0**-126
        expected_denormal = struct.unpack("<f", struct.pack("<I", 0x0001 << 16))[0]
        assert denormal == expected_denormal == 2.0**-133

    def test_element_width_and_truncation(self) -> None:
        assert element_width("bfloat16") == 2
        raw = struct.pack("<2H", 0x3F80, 0x4000) + b"\x7f"
        decoded = decode_packed("bfloat16", raw)
        assert decoded is not None
        assert list(decoded) == [1.0, 2.0], "the trailing odd byte is dropped"

    def test_statistics_stream_bfloat16(self, tmp_path: Path) -> None:
        patterns = (0x3F80, 0xC000, 0x0000, 0x4049, 0x7F80, 0x7FC0)
        raw = struct.pack(f"<{len(patterns)}H", *patterns)
        store, tensor_id = store_for_raw(tmp_path, raw, dims=[6], data_type=16)
        with store:
            stats = compute_statistics(store, tensor_id)
        assert stats.element_type == "bfloat16"
        assert stats.element_count == 6
        assert stats.inf_count == 1 and stats.nan_count == 1
        assert stats.minimum == -2.0 and stats.maximum == 3.140625
        assert stats.zero_count == 1
        finite = np.array([1.0, -2.0, 0.0, 3.140625], dtype=np.float64)
        assert stats.mean == pytest.approx(float(finite.mean()))
        assert stats.std == pytest.approx(float(finite.std()))


class TestMalformedPayloads:
    """Finding 5: a misaligned payload must be reported, not a raw ValueError."""

    def test_misaligned_payload_is_reported_clearly(self, tmp_path: Path) -> None:
        (tmp_path / "odd.bin").write_bytes(b"\x00" * 7)
        document = handcrafted_document(
            tmp_path / "model.onnx",
            [
                TensorRef(
                    id="t:odd",
                    element_type="float32",
                    dims=(1,),
                    storage=Storage.EXTERNAL,
                    external=ExternalRef(location="odd.bin", offset=0, length=7),
                )
            ],
        )
        with TensorStore(document) as store:
            with pytest.raises(
                TensorUnavailableError, match="not a multiple of the 4-byte"
            ):
                compute_statistics(store, "t:odd")


def mixed_layout_graph() -> Graph:
    """Isolated nodes interleaved with a diamond and a skip edge."""
    values = tuple(
        Value(id=f"v:{name}", name=name)
        for name in ("z", "a", "b", "i2", "c", "d", "e")
    )
    spec: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("z", ()),
        ("a", ()),
        ("b", ("v:a",)),
        ("i2", ()),
        ("c", ("v:a",)),
        ("d", ("v:b", "v:c")),
        ("e", ("v:d", "v:a")),  # skip edge from the entry
    )
    nodes = tuple(
        Node(
            id=name,
            id_stability=NodeIdStability.NAMED,
            op_type="Relu",
            inputs=inputs,
            outputs=(f"v:{name}",),
        )
        for name, inputs in spec
    )
    return Graph(id="g:main", name="mixed", nodes=nodes, values=values)


class TestLayerAssignment:
    """Finding 6: the heap ready queue must keep the exact old ordering."""

    def test_mixed_graph_layer_structure_is_pinned(self) -> None:
        expected = (("z", "a", "i2"), ("b", "c"), ("d",), ("e",))
        unordered = layout_graph(
            mixed_layout_graph(), LayoutSettings(ordering_passes=0)
        )
        assert unordered.layers == expected
        assert unordered.cyclic_nodes == ()
        ordered = layout_graph(mixed_layout_graph())
        assert ordered.layers == expected, "barycenter passes keep this order"

    def test_layout_is_deterministic_across_runs(self) -> None:
        first = layout_graph(mixed_layout_graph())
        second = layout_graph(mixed_layout_graph())
        assert first.scene.nodes == second.scene.nodes
        assert first.scene.edges == second.scene.edges
        assert first.layers == second.layers

    def test_wide_flat_graph_assigns_layers_fast(self) -> None:
        count = 16_000
        values = tuple(Value(id=f"v{i}", name=f"v{i}") for i in range(count))
        nodes = tuple(
            Node(
                id=f"n{i}",
                id_stability=NodeIdStability.NAMED,
                op_type="Relu",
                inputs=(),
                outputs=(f"v{i}",),
            )
            for i in range(count)
        )
        graph = Graph(id="g:main", name="flat", nodes=nodes, values=values)
        start = time.perf_counter()
        layer, cyclic = _assign_layers(graph, _dataflow_edges(graph))
        elapsed = time.perf_counter() - start
        assert cyclic == ()
        assert set(layer.values()) == {0}
        assert elapsed < 1.0, (
            f"quadratic ready-queue re-sort suspected: {elapsed:.2f} s for "
            "16k independent nodes (was 1.9 s before the heap)"
        )


class TestLayoutCancellation:
    """Finding 7: layout accepts an optional token without changing callers."""

    def test_layout_without_a_token_is_unchanged(self) -> None:
        result = layout_graph(mixed_layout_graph())
        assert result.scene.node_count == 7
        assert layout_graph(mixed_layout_graph(), token=None).scene.node_count == 7

    def test_cancelled_token_stops_layout(self) -> None:
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            layout_graph(mixed_layout_graph(), token=token)

    def test_mid_run_cancellation_stops_layout(self) -> None:
        counting = _CountingToken()
        layout_graph(mixed_layout_graph(), token=counting)
        assert counting.calls >= 2, "layout passes several checkpoints"
        with pytest.raises(OperationCancelled):
            layout_graph(
                mixed_layout_graph(), token=_CancelAfterToken(counting.calls - 1)
            )
