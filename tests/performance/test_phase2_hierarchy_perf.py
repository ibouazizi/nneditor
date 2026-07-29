"""Phase 2 collapse/expand and architecture-overview budgets."""

from __future__ import annotations

import math
import time

import pytest

from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    Group,
    Hierarchy,
)
from nneditor.analysis.lod import DetailLevel, semantic_scene
from nneditor.rendering.synthetic import synthetic_graph

pytestmark = pytest.mark.performance


def test_10k_collapse_expand_meets_phase0_feedback_budget() -> None:
    synthetic = synthetic_graph(10_000, layer_width=50, seed=42)
    evidence = (DetectionEvidence("benchmark.fixture", "fixed 100-node block"),)
    groups = [
        Group(
            id=f"grp:bench:{start}",
            graph_id="g:main",
            label=f"Block {start // 100}",
            members=frozenset(
                node.id for node in synthetic.scene.nodes[start : start + 100]
            ),
            kind=CandidateKind.REPEATED,
            confidence=0.9,
            evidence=evidence,
        )
        for start in range(0, 10_000, 100)
    ]
    hierarchy = Hierarchy("g:main", groups)

    # Warm caches and the allocator before recording the p95 sample.
    semantic_scene(synthetic.scene, hierarchy, DetailLevel.BLOCK)
    semantic_scene(synthetic.scene, hierarchy, DetailLevel.OPERATOR)
    collapse_ms: list[float] = []
    expand_ms: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        collapsed = semantic_scene(synthetic.scene, hierarchy, DetailLevel.BLOCK)
        collapse_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        expanded = semantic_scene(synthetic.scene, hierarchy, DetailLevel.OPERATOR)
        expand_ms.append((time.perf_counter() - started) * 1000)

    p95_index = math.ceil(len(collapse_ms) * 0.95) - 1
    assert sorted(collapse_ms)[p95_index] < 250
    assert sorted(expand_ms)[p95_index] < 250
    assert collapsed.scene.node_count == 100
    assert expanded.scene.node_count == 10_000


def test_10k_architecture_representation_stays_below_shape_budget() -> None:
    synthetic = synthetic_graph(10_000, layer_width=50, seed=43)
    overview = semantic_scene(
        synthetic.scene,
        Hierarchy("g:main"),
        DetailLevel.ARCHITECTURE,
    )
    # One label plus one node shape per overview glyph remains at the renderer's
    # 2,000-shape ceiling; edges are allowed to drop at overview scale.
    assert overview.scene.node_count <= 1000
