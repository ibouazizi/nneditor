"""Phase 2 hierarchy detection, reconciliation, and manual overrides."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nneditor.analysis.detectors import (
    DetectionPipeline,
    GraphTopology,
    PatternDetector,
    RepeatedSubgraphDetector,
    SourceHierarchyDetector,
    StructuralRegionDetector,
    dominators,
    post_dominators,
)
from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    GroupCandidate,
    reconcile_candidates,
)
from nneditor.application.hierarchy import (
    HierarchyController,
    HierarchySidecarStore,
)
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import ArtifactRef, Document, Graph, Node, Value
from nneditor.ir.identity import NodeIdStability


def make_document(graph: Graph, *extra_graphs: Graph) -> Document:
    return Document(
        source=ArtifactRef(
            path="model.onnx",
            content_hash="sha256:" + "1" * 64,
            byte_size=100,
        ),
        artifact_kind=ArtifactKind.ONNX_MODEL,
        capabilities=contract_for(ArtifactKind.ONNX_MODEL).statuses,
        graphs=(graph, *extra_graphs),
        entry_graph=graph.id,
    )


def linear_graph(
    ops: tuple[str, ...],
    *,
    names: tuple[str, ...] | None = None,
    graph_id: str = "g:main",
) -> Graph:
    values = tuple(
        Value(id=f"v{index}", name=f"v{index}", element_type="float32", shape=(4, 4))
        for index in range(len(ops))
    )
    nodes = tuple(
        Node(
            id=f"n{index}",
            id_stability=NodeIdStability.NAMED,
            op_type=op,
            inputs=() if index == 0 else (f"v{index - 1}",),
            outputs=(f"v{index}",),
            source_name=names[index] if names else None,
        )
        for index, op in enumerate(ops)
    )
    return Graph(id=graph_id, name="main", nodes=nodes, values=values)


def branch_graph() -> Graph:
    values = tuple(Value(id=item, name=item) for item in ("v0", "vl", "vr", "vo"))
    nodes = (
        Node(
            id="entry",
            id_stability=NodeIdStability.NAMED,
            op_type="Identity",
            inputs=(),
            outputs=("v0",),
        ),
        Node(
            id="left",
            id_stability=NodeIdStability.NAMED,
            op_type="Gemm",
            inputs=("v0",),
            outputs=("vl",),
        ),
        Node(
            id="right",
            id_stability=NodeIdStability.NAMED,
            op_type="Identity",
            inputs=("v0",),
            outputs=("vr",),
        ),
        Node(
            id="merge",
            id_stability=NodeIdStability.NAMED,
            op_type="Add",
            inputs=("vl", "vr"),
            outputs=("vo",),
        ),
    )
    return Graph(id="g:main", name="branch", nodes=nodes, values=values)


def repeated_graph() -> Graph:
    values = tuple(
        Value(id=f"v{index}", name=f"v{index}", element_type="float32", shape=(8, 8))
        for index in range(6)
    )
    ops = ("MatMul", "Gelu", "MatMul", "MatMul", "Gelu", "MatMul")
    nodes = []
    for index, op in enumerate(ops):
        start = index in (0, 3)
        nodes.append(
            Node(
                id=f"n{index}",
                id_stability=NodeIdStability.NAMED,
                op_type=op,
                inputs=() if start else (f"v{index - 1}",),
                outputs=(f"v{index}",),
            )
        )
    return Graph(id="g:main", name="repeat", nodes=nodes, values=values)


def proof(code: str = "test") -> tuple[DetectionEvidence, ...]:
    return (DetectionEvidence(code, "test evidence"),)


class TestGraphAlgorithms:
    def test_adjacency_reachability_and_distances(self) -> None:
        topology = GraphTopology(branch_graph())
        assert topology.entries == ("entry",)
        assert topology.exits == ("merge",)
        assert topology.successors("entry") == ("left", "right")
        assert topology.predecessors("merge") == ("left", "right")
        assert topology.reachable(("left",)) == {"left", "merge"}
        assert topology.reachable(("merge",), reverse=True) == {
            "entry",
            "left",
            "right",
            "merge",
        }
        assert topology.shortest_distance(("entry",), "merge") == 2
        assert topology.shortest_distance(("merge",), "entry") is None

    def test_dominator_and_post_dominator_sets(self) -> None:
        graph = branch_graph()
        dom = dominators(graph)
        postdom = post_dominators(graph)
        assert dom["merge"] == {"entry", "merge"}
        assert postdom["entry"] == {"entry", "merge"}
        assert "merge" in postdom["left"]

    def test_structural_detector_explains_branch_and_residual(self) -> None:
        graph = branch_graph()
        candidates = StructuralRegionDetector().detect(make_document(graph), graph)
        assert candidates
        assert any(
            item.members == {"entry", "left", "right", "merge"} for item in candidates
        )
        codes = {evidence.code for item in candidates for evidence in item.evidence}
        assert "structure.sese" in codes
        assert "structure.residual-add" in codes

    def test_cycle_only_graph_is_conservative(self) -> None:
        values = (Value("va", "va"), Value("vb", "vb"))
        graph = Graph(
            id="g:main",
            name="cycle",
            values=values,
            nodes=(
                Node(
                    "a",
                    NodeIdStability.NAMED,
                    "Add",
                    ("vb",),
                    ("va",),
                ),
                Node(
                    "b",
                    NodeIdStability.NAMED,
                    "Add",
                    ("va",),
                    ("vb",),
                ),
            ),
        )
        assert dominators(graph) == {"a": {"a"}, "b": {"b"}}
        assert StructuralRegionDetector().detect(make_document(graph), graph) == ()


class TestEvidenceDetectors:
    def test_source_scopes_form_nested_candidates(self) -> None:
        graph = linear_graph(
            ("Conv", "Relu", "Conv", "Relu"),
            names=(
                "encoder/block0/conv",
                "encoder/block0/relu",
                "encoder/block1/conv",
                "encoder/block1/relu",
            ),
        )
        candidates = SourceHierarchyDetector().detect(make_document(graph), graph)
        memberships = {item.label: item.members for item in candidates}
        assert memberships["encoder"] == {"n0", "n1", "n2", "n3"}
        assert memberships["block0"] == {"n0", "n1"}
        assert memberships["block1"] == {"n2", "n3"}
        assert all(item.evidence for item in candidates)

    def test_control_flow_graph_gets_containment_evidence(self) -> None:
        base = linear_graph(("If",))
        child_raw = linear_graph(("Add", "Relu"), graph_id="g:child")
        child = Graph(
            id=child_raw.id,
            name="then",
            nodes=child_raw.nodes,
            values=child_raw.values,
            parent_node="n0",
        )
        document = make_document(base, child)
        candidates = SourceHierarchyDetector().detect(document, child)
        assert len(candidates) == 1
        assert candidates[0].kind is CandidateKind.CONTROL_FLOW
        assert candidates[0].confidence == 1.0

    def test_pattern_library_detects_cnn_attention_and_feed_forward(self) -> None:
        graph = linear_graph(
            (
                "Conv",
                "BatchNormalization",
                "Relu",
                "MatMul",
                "Softmax",
                "MatMul",
                "Gemm",
                "Gelu",
                "Gemm",
            )
        )
        candidates = PatternDetector().detect(make_document(graph), graph)
        labels = {item.label for item in candidates}
        assert "Conv + norm + activation" in labels
        assert "Attention" in labels
        assert "Feed-forward" in labels
        assert all("motif" in item.evidence[0].summary for item in candidates)

    def test_repeated_detector_emits_each_occurrence(self) -> None:
        graph = repeated_graph()
        candidates = RepeatedSubgraphDetector(max_size=3).detect(
            make_document(graph), graph
        )
        assert len(candidates) == 2
        assert {item.members for item in candidates} == {
            frozenset(("n0", "n1", "n2")),
            frozenset(("n3", "n4", "n5")),
        }
        assert all("full signature" in item.evidence[0].summary for item in candidates)

    def test_hash_collision_never_groups_different_signatures(self) -> None:
        graph = linear_graph(("Conv", "Relu", "MatMul", "Softmax"))
        detector = RepeatedSubgraphDetector(
            min_size=2, max_size=2, digest=lambda _signature: "collision"
        )
        assert detector.detect(make_document(graph), graph) == ()


class TestReconciliation:
    def test_exact_memberships_pool_evidence_and_keep_best_kind(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add"))
        first = GroupCandidate(
            graph.id,
            "source",
            1,
            "Scope",
            frozenset(("n0", "n1")),
            CandidateKind.SOURCE,
            0.7,
            proof("source"),
        )
        second = GroupCandidate(
            graph.id,
            "pattern",
            1,
            "Layer",
            frozenset(("n0", "n1")),
            CandidateKind.PATTERN,
            0.9,
            proof("pattern"),
        )
        group = reconcile_candidates(graph, (first, second)).groups[0]
        assert group.label == "Layer"
        assert group.kind is CandidateKind.PATTERN
        assert {item.code for item in group.evidence} == {"source", "pattern"}

    def test_partial_overlap_keeps_higher_confidence(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add"))
        weak = GroupCandidate(
            graph.id,
            "weak",
            1,
            "Weak",
            frozenset(("n0", "n1")),
            CandidateKind.SOURCE,
            0.5,
            proof(),
        )
        strong = GroupCandidate(
            graph.id,
            "strong",
            1,
            "Strong",
            frozenset(("n1", "n2")),
            CandidateKind.STRUCTURAL,
            0.9,
            proof(),
        )
        hierarchy = reconcile_candidates(graph, (weak, strong))
        assert tuple(group.label for group in hierarchy.groups) == ("Strong",)

    def test_nested_groups_get_parent_and_breadcrumbs(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add"))
        outer = GroupCandidate(
            graph.id,
            "outer",
            1,
            "Outer",
            frozenset(("n0", "n1", "n2")),
            CandidateKind.SOURCE,
            0.8,
            proof(),
        )
        inner = GroupCandidate(
            graph.id,
            "inner",
            1,
            "Inner",
            frozenset(("n0", "n1")),
            CandidateKind.PATTERN,
            0.9,
            proof(),
        )
        hierarchy = reconcile_candidates(graph, (outer, inner))
        inner_group = next(
            group for group in hierarchy.groups if group.label == "Inner"
        )
        assert inner_group.parent_id is not None
        assert [item.label for item in hierarchy.breadcrumbs(inner_group.id)] == [
            "Outer",
            "Inner",
        ]
        assert hierarchy.groups_for_node("n0")[0].label == "Inner"
        assert hierarchy.roots[0].label == "Outer"

    def test_pipeline_is_deterministic(self) -> None:
        graph = repeated_graph()
        document = make_document(graph)
        pipeline = DetectionPipeline()
        first = pipeline.detect_graph(document, graph)
        second = pipeline.detect_graph(document, graph)
        assert first.revision == second.revision
        assert first.groups == second.groups

    def test_pipeline_honors_cancellation_before_detector_work(self) -> None:
        graph = repeated_graph()
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            DetectionPipeline().detect_graph(make_document(graph), graph, token)


class TestManualOverrides:
    def test_group_rename_lock_split_and_reset(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm"))
        controller = HierarchyController(
            make_document(graph), pipeline=DetectionPipeline(())
        )
        group = controller.group(graph.id, "Feature", frozenset(("n0", "n1", "n2")))
        renamed = controller.rename(graph.id, group.id, "Encoder")
        assert renamed.label == "Encoder"
        locked = controller.lock(graph.id, renamed.id)
        assert locked.locked
        with pytest.raises(ValueError, match="unlock"):
            controller.split(graph.id, renamed.id)
        controller.lock(graph.id, renamed.id, locked=False)
        pieces = controller.split(
            graph.id,
            renamed.id,
            (frozenset(("n0", "n1")), frozenset(("n2",))),
        )
        assert len(pieces) == 1
        assert pieces[0].members == {"n0", "n1"}
        controller.reset(graph.id)
        assert controller.hierarchy(graph.id).groups == ()

    def test_merge_and_manual_partial_overlap_validation(self) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm"))
        controller = HierarchyController(
            make_document(graph), pipeline=DetectionPipeline(())
        )
        left = controller.group(graph.id, "Left", frozenset(("n0", "n1")))
        right = controller.group(graph.id, "Right", frozenset(("n2", "n3")))
        with pytest.raises(ValueError, match="partially overlap"):
            controller.group(graph.id, "Bad", frozenset(("n1", "n2")))
        merged = controller.merge(graph.id, (left.id, right.id), "Whole")
        assert merged.members == {"n0", "n1", "n2", "n3"}
        assert not merged.automatic
        controller.reject(graph.id, merged.id)
        assert controller.hierarchy(graph.id).groups == ()

    def test_automatic_reject_rename_lock_and_reset(self) -> None:
        graph = linear_graph(
            ("Conv", "Relu", "Add"),
            names=("encoder/conv", "encoder/relu", "encoder/add"),
        )
        controller = HierarchyController(
            make_document(graph),
            pipeline=DetectionPipeline((SourceHierarchyDetector(),)),
        )
        automatic = controller.hierarchy(graph.id).groups[0]
        renamed = controller.rename(graph.id, automatic.id, "Backbone")
        assert renamed.label == "Backbone"
        assert controller.lock(graph.id, renamed.id).locked
        with pytest.raises(ValueError, match="unlock"):
            controller.reject(graph.id, renamed.id)
        controller.lock(graph.id, renamed.id, locked=False)
        controller.reject(graph.id, renamed.id)
        assert controller.hierarchy(graph.id).groups == ()
        controller.reset()
        assert controller.hierarchy(graph.id).groups[0].label == "encoder"

    def test_sidecar_round_trip_and_corrupt_or_newer_discard(
        self, tmp_path: Path
    ) -> None:
        graph = linear_graph(("Conv", "Relu", "Add"))
        document = make_document(graph)
        store = HierarchySidecarStore(tmp_path)
        first = HierarchyController(
            document, pipeline=DetectionPipeline(()), store=store
        )
        first.group(graph.id, "Saved", frozenset(("n0", "n1")))
        restored = HierarchyController(
            document,
            pipeline=DetectionPipeline(()),
            store=HierarchySidecarStore(tmp_path),
        )
        assert restored.hierarchy(graph.id).groups[0].label == "Saved"

        store.path.write_text("{bad", encoding="utf-8")
        assert HierarchySidecarStore(tmp_path).for_artifact("x").is_empty()
        store.path.write_text(
            json.dumps({"version": 999, "artifacts": {}}), encoding="utf-8"
        )
        assert HierarchySidecarStore(tmp_path).for_artifact("x").is_empty()

    def test_two_sessions_mutate_shared_overrides_without_save_races(
        self,
        tmp_path: Path,
    ) -> None:
        graph = linear_graph(("Conv", "Relu", "Add", "Gemm"))
        document = make_document(graph)
        store = HierarchySidecarStore(tmp_path)
        first = HierarchyController(
            document,
            pipeline=DetectionPipeline(()),
            store=store,
        )
        second = HierarchyController(
            document,
            pipeline=DetectionPipeline(()),
            store=store,
        )
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def add(
            controller: HierarchyController,
            label: str,
            members: frozenset[str],
        ) -> None:
            try:
                barrier.wait(timeout=2)
                controller.group(graph.id, label, members)
            except BaseException as error:
                errors.append(error)

        threads = (
            threading.Thread(
                target=add,
                args=(first, "Front", frozenset(("n0", "n1"))),
            ),
            threading.Thread(
                target=add,
                args=(second, "Back", frozenset(("n2", "n3"))),
            ),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        assert not errors
        restored = HierarchyController(
            document,
            pipeline=DetectionPipeline(()),
            store=HierarchySidecarStore(tmp_path),
        )
        assert {group.label for group in restored.hierarchy(graph.id).groups} == {
            "Front",
            "Back",
        }
