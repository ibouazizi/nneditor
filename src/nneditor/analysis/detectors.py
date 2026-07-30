"""Deterministic hierarchy detectors (Phase 2, tasks P2.1-P2.4).

The detectors intentionally share only the small :class:`GroupDetector`
contract. Source naming, graph regions, operator motifs, and repeated
subgraphs are independent evidence sources and can be replaced without
changing reconciliation or the UI.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    GroupCandidate,
    GroupDetector,
    Hierarchy,
    reconcile_candidates,
)
from nneditor.cancellation import CancellationToken
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.core import Attribute, Document, Graph, Node

__all__ = [
    "DEFAULT_DETECTORS",
    "MAX_EXACT_DOMINATOR_NODES",
    "MAX_REPEAT_SCAN_NODES",
    "DetectionPipeline",
    "GraphTopology",
    "Pattern",
    "PatternDetector",
    "RepeatedSubgraphDetector",
    "SourceHierarchyDetector",
    "StructuralRegionDetector",
    "dominators",
    "post_dominators",
]

MAX_EXACT_DOMINATOR_NODES: Final = 2000
"""Avoid quadratic dominator sets on very large flat graphs."""

MAX_REPEAT_SCAN_NODES: Final = 5000
"""Upper bound for the exact windowed structural-signature scan."""


class GraphTopology:
    """Derived node adjacency behind the replaceable graph-algorithm boundary."""

    __slots__ = ("_predecessors", "_successors", "graph", "order")

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.order = {node.id: index for index, node in enumerate(graph.nodes)}
        predecessors: dict[str, set[str]] = {node.id: set() for node in graph.nodes}
        successors: dict[str, set[str]] = {node.id: set() for node in graph.nodes}
        for node in graph.nodes:
            for value_id in node.inputs:
                producer = graph.producer(value_id)
                if producer is None:
                    continue
                source = producer[0]
                predecessors[node.id].add(source)
                successors[source].add(node.id)
        self._predecessors = MappingProxyType(
            {
                key: tuple(sorted(items, key=self.order.__getitem__))
                for key, items in predecessors.items()
            }
        )
        self._successors = MappingProxyType(
            {
                key: tuple(sorted(items, key=self.order.__getitem__))
                for key, items in successors.items()
            }
        )

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        return self._predecessors[node_id]

    def successors(self, node_id: str) -> tuple[str, ...]:
        return self._successors[node_id]

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(
            node.id for node in self.graph.nodes if not self.predecessors(node.id)
        )

    @property
    def exits(self) -> tuple[str, ...]:
        return tuple(
            node.id for node in self.graph.nodes if not self.successors(node.id)
        )

    def reachable(
        self,
        starts: Iterable[str],
        *,
        reverse: bool = False,
        stop: str | None = None,
    ) -> frozenset[str]:
        """All nodes reachable from ``starts``, including the starts."""
        visit = self.predecessors if reverse else self.successors
        pending = deque(starts)
        seen: set[str] = set()
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            if current == stop:
                continue
            pending.extend(visit(current))
        return frozenset(seen)

    def shortest_distance(
        self, starts: Iterable[str], target: str, *, reverse: bool = False
    ) -> int | None:
        visit = self.predecessors if reverse else self.successors
        pending = deque((item, 0) for item in starts)
        seen: set[str] = set()
        while pending:
            current, distance = pending.popleft()
            if current == target:
                return distance
            if current in seen:
                continue
            seen.add(current)
            pending.extend((item, distance + 1) for item in visit(current))
        return None


def _reverse_postorder(
    roots: tuple[str, ...],
    neighbors: Callable[[str], tuple[str, ...]],
) -> list[str]:
    """Reverse postorder of an iterative DFS from ``roots``.

    Deterministic: roots are expanded in the given order and neighbors in
    their (declaration-sorted) adjacency order.
    """
    seen: set[str] = set()
    postorder: list[str] = []
    stack: list[tuple[str, int]] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        stack.append((root, 0))
        while stack:
            node_id, cursor = stack[-1]
            targets = neighbors(node_id)
            if cursor < len(targets):
                stack[-1] = (node_id, cursor + 1)
                target = targets[cursor]
                if target not in seen:
                    seen.add(target)
                    stack.append((target, 0))
            else:
                stack.pop()
                postorder.append(node_id)
    postorder.reverse()
    return postorder


def _fixed_point_dominators(
    topology: GraphTopology,
    *,
    reverse: bool,
    token: CancellationToken | None = None,
) -> dict[str, frozenset[str]]:
    nodes = tuple(node.id for node in topology.graph.nodes)
    if not nodes:
        return {}
    roots = topology.exits if reverse else topology.entries
    # A cycle-only malformed graph has no natural root. Treat every node as a
    # root so analysis terminates conservatively instead of inventing regions.
    if not roots:
        roots = nodes
    root_set = frozenset(roots)
    incoming = topology.successors if reverse else topology.predecessors
    outgoing = topology.predecessors if reverse else topology.successors
    # Sweeping in reverse postorder from the roots visits every parent before
    # its children, so the fixed point converges in a couple of passes no
    # matter how the artifact declared its nodes; declaration order alone
    # needs O(V) passes on a reverse-declared chain. The converged sets are
    # identical either way — the equations have one maximum fixed point.
    sweep = _reverse_postorder(roots, outgoing)
    if len(sweep) != len(nodes):
        reached = set(sweep)
        sweep.extend(node_id for node_id in nodes if node_id not in reached)
    universe = frozenset(nodes)
    values: dict[str, frozenset[str]] = {
        node_id: frozenset((node_id,)) if node_id in root_set else universe
        for node_id in nodes
    }
    changed = True
    while changed:
        changed = False
        for position, node_id in enumerate(sweep):
            if token is not None and position % 256 == 0:
                token.raise_if_cancelled()
            if node_id in root_set:
                continue
            parents = incoming(node_id)
            if not parents:
                updated = frozenset((node_id,))
            else:
                shared = set(values[parents[0]])
                for parent in parents[1:]:
                    shared.intersection_update(values[parent])
                shared.add(node_id)
                updated = frozenset(shared)
            if updated != values[node_id]:
                values[node_id] = updated
                changed = True
    return values


def dominators(graph: Graph) -> dict[str, frozenset[str]]:
    """Node dominator sets for a possibly disconnected graph."""
    return _fixed_point_dominators(GraphTopology(graph), reverse=False)


def post_dominators(graph: Graph) -> dict[str, frozenset[str]]:
    """Node post-dominator sets for a possibly disconnected graph."""
    return _fixed_point_dominators(GraphTopology(graph), reverse=True)


def _nearest_common(
    topology: GraphTopology,
    starts: tuple[str, ...],
    sets: dict[str, frozenset[str]],
    token: CancellationToken | None = None,
) -> str | None:
    """The nearest entry of ``sets`` shared by every start.

    The (post-)dominators of one node form a chain ordered by the dominator
    tree, and a nearer member's set strictly contains a farther member's —
    so the nearest common member is simply the shared entry with the largest
    set, no per-candidate breadth-first searches required. One reachability
    check per start keeps malformed graphs (whose fixed point is not a
    chain) conservative instead of inventing a merge point.
    """
    if not starts:
        return None
    common = set(sets[starts[0]])
    for node_id in starts[1:]:
        common.intersection_update(sets[node_id])
    if not common:
        return None
    nearest = min(common, key=lambda item: (-len(sets[item]), item))
    for start in starts:
        if token is not None:
            token.raise_if_cancelled()
        if topology.shortest_distance((start,), nearest) is None:
            return None
    return nearest


_SCOPE_SPLIT: Final = re.compile(r"[/.:]+")


class SourceHierarchyDetector:
    """Groups nodes sharing exporter-provided scope/name prefixes."""

    name = "source-scope"
    version = 1

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
        log: DiagnosticLog | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document, log
        prefixes: dict[tuple[str, ...], set[str]] = {}
        for node in graph.nodes:
            if token is not None:
                token.raise_if_cancelled()
            source = (node.source_location or node.source_name or "").strip()
            parts = tuple(part for part in _SCOPE_SPLIT.split(source) if part)
            # The final segment is usually the operator instance, not a scope.
            for length in range(1, len(parts)):
                prefixes.setdefault(parts[:length], set()).add(node.id)
        candidates: list[GroupCandidate] = []
        for prefix, members in sorted(prefixes.items()):
            if len(members) < 2:
                continue
            label = prefix[-1]
            candidates.append(
                GroupCandidate(
                    graph_id=graph.id,
                    detector=self.name,
                    detector_version=self.version,
                    label=label,
                    members=frozenset(members),
                    kind=CandidateKind.SOURCE,
                    confidence=min(0.98, 0.72 + 0.04 * len(prefix)),
                    evidence=(
                        DetectionEvidence(
                            "source.shared-scope",
                            f"{len(members)} nodes share exporter scope "
                            f"{'/'.join(prefix)!r}.",
                            tuple(sorted(members)),
                        ),
                    ),
                )
            )
        if graph.parent_node is not None and graph.nodes:
            control_members = frozenset(node.id for node in graph.nodes)
            candidates.append(
                GroupCandidate(
                    graph_id=graph.id,
                    detector=self.name,
                    detector_version=self.version,
                    label=graph.name or "Control-flow body",
                    members=control_members,
                    kind=CandidateKind.CONTROL_FLOW,
                    confidence=1.0,
                    evidence=(
                        DetectionEvidence(
                            "source.control-flow-body",
                            "Nodes are contained by one ONNX control-flow graph.",
                            tuple(sorted(control_members)),
                        ),
                    ),
                )
            )
        return tuple(candidates)


class StructuralRegionDetector:
    """Finds branch/merge SESE regions and residual paths."""

    name = "structural-region"
    version = 1

    def __init__(self, *, max_nodes: int = MAX_EXACT_DOMINATOR_NODES) -> None:
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
        log: DiagnosticLog | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document
        if len(graph.nodes) > self.max_nodes:
            # Say so rather than silently returning nothing: a large model
            # otherwise loses every structural block with no explanation.
            if log is not None:
                log.add(
                    "hierarchy.structural-analysis-skipped",
                    Severity.INFO,
                    f"{len(graph.nodes)} nodes exceed the "
                    f"{self.max_nodes}-node exact dominator limit, so "
                    "branch/merge and residual regions were not detected.",
                    graph.id,
                )
            return ()
        topology = GraphTopology(graph)
        postdom = _fixed_point_dominators(topology, reverse=True, token=token)
        dom = _fixed_point_dominators(topology, reverse=False, token=token)
        candidates: list[GroupCandidate] = []

        for branch in graph.nodes:
            if token is not None:
                token.raise_if_cancelled()
            successors = topology.successors(branch.id)
            if len(successors) < 2:
                continue
            merge = _nearest_common(topology, successors, postdom, token)
            if merge is None or merge == branch.id:
                continue
            forward = topology.reachable(successors, stop=merge)
            backward = topology.reachable((merge,), reverse=True)
            region_members = frozenset(
                {branch.id, merge} | (set(forward) & set(backward))
            )
            if len(region_members) < 3:
                continue
            candidates.append(
                GroupCandidate(
                    graph_id=graph.id,
                    detector=self.name,
                    detector_version=self.version,
                    label=f"Branch → {graph.node(merge).op_type}",
                    members=region_members,
                    kind=CandidateKind.STRUCTURAL,
                    confidence=0.86,
                    evidence=(
                        DetectionEvidence(
                            "structure.sese",
                            f"{branch.id!r} branches into {len(successors)} paths "
                            f"that reconverge at {merge!r}; the region has one "
                            "structural entry and exit.",
                            (branch.id, merge),
                        ),
                        DetectionEvidence(
                            "structure.post-dominator",
                            f"{merge!r} post-dominates every branch successor.",
                            successors,
                        ),
                    ),
                )
            )

        for merge_node in graph.nodes:
            if token is not None:
                token.raise_if_cancelled()
            if merge_node.op_type != "Add":
                continue
            predecessors = topology.predecessors(merge_node.id)
            if len(predecessors) != 2:
                continue
            common = set(dom[predecessors[0]]) & set(dom[predecessors[1]])
            if not common:
                continue
            # The closest shared dominator is the deepest one: it has the
            # largest dominator set.
            entry = max(common, key=lambda item: (len(dom[item]), item))
            between = topology.reachable((entry,)) & topology.reachable(
                (merge_node.id,), reverse=True
            )
            residual_members = frozenset(between)
            if len(residual_members) < 3:
                continue
            candidates.append(
                GroupCandidate(
                    graph_id=graph.id,
                    detector=self.name,
                    detector_version=self.version,
                    label="Residual region",
                    members=residual_members,
                    kind=CandidateKind.STRUCTURAL,
                    confidence=0.9,
                    evidence=(
                        DetectionEvidence(
                            "structure.residual-add",
                            f"Two paths dominated by {entry!r} reconverge at "
                            f"residual Add {merge_node.id!r}.",
                            (entry, *predecessors, merge_node.id),
                        ),
                    ),
                )
            )
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class Pattern:
    """A versioned, human-readable operator motif with optional bridging ops.

    ``stages`` are the ops that define the motif and must all be present.
    ``connectors`` are ops that a real exporter interposes *between* stages
    without changing what the motif is — the scale ``Div``/``Mul`` and mask
    ``Add`` an attention core carries between its score ``MatMul`` and
    ``Softmax``, for instance. Up to ``max_connector_run`` of them may be
    skipped through per stage transition, and any that are traversed join the
    group, because they belong to the motif rather than sitting outside it.

    Without this, matching was a strict chain of immediate successors, so the
    attention motif never fired on ordinary exported transformers and the
    layer level of detail had nothing to show.
    """

    name: str
    label: str
    stages: tuple[frozenset[str], ...]
    confidence: float
    connectors: frozenset[str] = frozenset()
    max_connector_run: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.label or len(self.stages) < 2:
            raise ValueError("a pattern needs a name, label, and at least two stages")
        if any(not stage for stage in self.stages):
            raise ValueError("pattern stages cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("pattern confidence must be in [0, 1]")
        if self.max_connector_run < 0:
            raise ValueError("max_connector_run cannot be negative")
        if self.connectors and self.max_connector_run < 1:
            raise ValueError("connectors need a positive max_connector_run")
        if self.max_connector_run and not self.connectors:
            raise ValueError("max_connector_run needs connector op types")
        for stage in self.stages:
            if stage & self.connectors:
                raise ValueError(
                    "an op type cannot be both a stage and a connector: "
                    f"{sorted(stage & self.connectors)}"
                )


_LINEAR = frozenset(("Gemm", "MatMul"))
_NORMS = frozenset(
    (
        "BatchNormalization",
        "LayerNormalization",
        "GroupNormalization",
        "InstanceNormalization",
    )
)
_ACTIVATIONS = frozenset(("Relu", "LeakyRelu", "Gelu", "Sigmoid", "Tanh", "Clip"))

# Shape/layout ops that move data without changing what a motif computes.
_RESHAPES: Final = frozenset(("Reshape", "Transpose", "Cast", "Squeeze", "Unsqueeze"))
# Elementwise ops an exporter emits for attention scaling and mask application.
_SCALE_MASK: Final = frozenset(("Div", "Mul", "Add", "Sub", "Where"))

DEFAULT_PATTERNS: Final[tuple[Pattern, ...]] = (
    Pattern(
        "conv-norm-activation",
        "Conv + norm + activation",
        (frozenset(("Conv", "ConvTranspose")), _NORMS, _ACTIVATIONS),
        0.96,
        connectors=frozenset(("Add", "Cast")),
        max_connector_run=1,
    ),
    Pattern(
        "attention-core",
        "Attention",
        (_LINEAR, frozenset(("Softmax",)), _LINEAR),
        0.94,
        # Scores are scaled and masked before the softmax and the context
        # product is usually reshaped after it; an exported attention core
        # essentially never has softmax as an immediate successor of MatMul.
        connectors=_SCALE_MASK | _RESHAPES,
        max_connector_run=4,
    ),
    Pattern(
        "feed-forward",
        "Feed-forward",
        (_LINEAR, _ACTIVATIONS, _LINEAR),
        0.94,
        # Gated feed-forward variants multiply the activated branch, and
        # exporters interpose shape ops around the projections.
        connectors=frozenset(("Mul",)) | _RESHAPES,
        max_connector_run=2,
    ),
    Pattern(
        "norm-activation",
        "Normalization",
        (_NORMS, _ACTIVATIONS),
        0.82,
        connectors=frozenset(("Cast",)),
        max_connector_run=1,
    ),
)


class PatternDetector:
    """Matches the deterministic, versioned motif library."""

    name = "operator-pattern"
    # Version 2 added bounded connector traversal, so a match may now include
    # bridging ops. Group ids embed the detector version, which means version 1
    # pattern ids (and any correction that referenced one) no longer resolve.
    version = 2

    def __init__(self, patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS) -> None:
        self.patterns = patterns

    @staticmethod
    def _advance(
        graph: Graph,
        topology: GraphTopology,
        current: str,
        stage: frozenset[str],
        pattern: Pattern,
    ) -> tuple[tuple[str, ...], str] | None:
        """Walk from ``current`` to the next stage across bridging ops.

        Returns the traversed connector ids and the matching stage node, or
        ``None`` when the next stage is absent or the step is ambiguous. Every
        hop demands exactly one candidate so a match stays deterministic and a
        fan-out never silently picks a branch.
        """
        bridge: list[str] = []
        node_id = current
        for _ in range(pattern.max_connector_run + 1):
            successors = topology.successors(node_id)
            direct = [item for item in successors if graph.node(item).op_type in stage]
            if len(direct) == 1:
                return tuple(bridge), direct[0]
            if len(direct) > 1:
                return None
            hops = [
                item
                for item in successors
                if graph.node(item).op_type in pattern.connectors
            ]
            if len(hops) != 1 or hops[0] in bridge:
                return None
            bridge.append(hops[0])
            node_id = hops[0]
        return None

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
        log: DiagnosticLog | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document, log
        topology = GraphTopology(graph)
        candidates: list[GroupCandidate] = []
        for start in graph.nodes:
            if token is not None:
                token.raise_if_cancelled()
            for pattern in self.patterns:
                if start.op_type not in pattern.stages[0]:
                    continue
                matched = [start.id]
                stage_nodes = [start.id]
                current = start.id
                for stage in pattern.stages[1:]:
                    step = self._advance(graph, topology, current, stage, pattern)
                    if step is None:
                        break
                    bridge, current = step
                    matched.extend(bridge)
                    matched.append(current)
                    stage_nodes.append(current)
                if len(stage_nodes) != len(pattern.stages):
                    continue
                members = frozenset(matched)
                candidates.append(
                    GroupCandidate(
                        graph_id=graph.id,
                        detector=f"{self.name}:{pattern.name}",
                        detector_version=self.version,
                        label=pattern.label,
                        members=members,
                        kind=CandidateKind.PATTERN,
                        confidence=pattern.confidence,
                        evidence=(
                            DetectionEvidence(
                                f"pattern.{pattern.name}",
                                f"Operators match the version {self.version} "
                                f"{pattern.label!r} motif: "
                                + " → ".join(
                                    graph.node(item).op_type for item in matched
                                )
                                + "."
                                + (
                                    ""
                                    if len(matched) == len(stage_nodes)
                                    else f" {len(matched) - len(stage_nodes)} "
                                    "bridging operator(s) between motif stages "
                                    "were included in the group."
                                ),
                                tuple(matched),
                            ),
                        ),
                    )
                )
        return tuple(candidates)


def _attribute_signature(attribute: Attribute) -> object:
    value: object = attribute.value
    if isinstance(value, tuple):
        value = list(value)
    return [attribute.name, attribute.kind.value, value]


def _node_signature(graph: Graph, node: Node) -> object:
    inputs = []
    for value_id in node.inputs:
        value = graph.value(value_id)
        inputs.append([value.element_type, list(value.shape) if value.shape else None])
    outputs = []
    for value_id in node.outputs:
        value = graph.value(value_id)
        outputs.append([value.element_type, list(value.shape) if value.shape else None])
    return {
        "op": node.qualified_op_type,
        "attributes": [
            _attribute_signature(item)
            for item in sorted(node.attributes, key=lambda item: item.name)
        ],
        "inputs": inputs,
        "outputs": outputs,
    }


def _window_signature(
    graph: Graph, topology: GraphTopology, node_ids: tuple[str, ...]
) -> str:
    # Order edges by their position *within the window*, never by node id.
    # Sorting the id strings first made the signature depend on how ids happen
    # to collate: a window over "n1".."n16" sorts as n1, n10, n11, ..., n2,
    # while "n17".."n32" sorts numerically, so two structurally identical
    # blocks produced different edge lists and were never grouped.
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    internal_edges = sorted(
        (position[source], position[target])
        for source in node_ids
        for target in topology.successors(source)
        if target in position
    )
    payload = {
        "nodes": [_node_signature(graph, graph.node(item)) for item in node_ids],
        "edges": [[source, target] for source, target in internal_edges],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


DigestFunction = Callable[[str], str]


def _default_digest(signature: str) -> str:
    return hashlib.blake2b(signature.encode(), digest_size=16).hexdigest()


def _block_scan_order(topology: GraphTopology) -> tuple[str, ...]:
    """Depth-first dependency order for repeat scanning.

    Repeat detection compares windows of consecutive nodes, so the ordering
    decides what can be found at all, and raw declaration order fails whenever
    an exporter interleaves the nodes of separate blocks. Depth-first reverse
    postorder follows one path to its end before starting the next, so each
    block's nodes land together and corresponding nodes of identical blocks
    land at corresponding offsets. A breadth-first or index-ordered
    topological sort would not: it interleaves independent parallel chains
    exactly the way the declaration order already did.

    Reverse postorder is also a valid topological order, and nodes left
    unreachable by a malformed cycle are appended in declaration order rather
    than dropped.
    """
    graph = topology.graph
    roots = topology.entries or tuple(node.id for node in graph.nodes)
    ordered = _reverse_postorder(roots, topology.successors)
    if len(ordered) != len(graph.nodes):
        seen = set(ordered)
        ordered.extend(node.id for node in graph.nodes if node.id not in seen)
    return tuple(ordered)


_HASH_BASE: Final = 1_000_003
_HASH_MODULUS: Final = (1 << 61) - 1


def _prefix_hashes(sequence: Sequence[int]) -> tuple[list[int], list[int]]:
    """Polynomial prefix hashes so any window's hash is a constant-time read.

    Bucketing windows by hash keeps the scan linear per window size, which is
    what makes an unbounded window size affordable; the full canonical
    signature is still compared before anything is grouped.
    """
    prefix = [0] * (len(sequence) + 1)
    powers = [1] * (len(sequence) + 1)
    for index, value in enumerate(sequence):
        prefix[index + 1] = (prefix[index] * _HASH_BASE + value + 1) % _HASH_MODULUS
        powers[index + 1] = powers[index] * _HASH_BASE % _HASH_MODULUS
    return prefix, powers


class RepeatedSubgraphDetector:
    """Finds repeated, shape- and attribute-aware windows of the graph.

    Windows are runs of consecutive nodes in *dependency* order (see
    :func:`_stable_topological_order`) rather than declaration order, so a
    block whose nodes an exporter interleaved with its neighbours is still
    found. Window size is bounded only by half the graph, because a real
    transformer layer is far larger than the twelve nodes this detector used to
    allow — a layer was previously fragmented into sub-chunks instead of being
    recognised.

    When a size matches, a smaller size that divides it and explains at least
    as many nodes is preferred, so twenty-four layers are reported as
    twenty-four blocks and not twelve double-layer blocks.

    Hashes are only bucket keys. Full canonical signatures are compared before
    a group is emitted, so a collision can at worst reduce performance; it
    cannot group different structures.
    """

    name = "repeated-subgraph"
    # Version 2 changed the ordering, the size bound, and the period choice, so
    # ids from version 1 (and any correction referencing one) no longer resolve.
    version = 2

    def __init__(
        self,
        *,
        min_size: int = 2,
        max_size: int = 96,
        max_graph_nodes: int = MAX_REPEAT_SCAN_NODES,
        digest: DigestFunction = _default_digest,
    ) -> None:
        if min_size < 2 or max_size < min_size:
            raise ValueError("invalid repeated-subgraph window sizes")
        if max_graph_nodes < 2:
            raise ValueError("max_graph_nodes must be at least two")
        self.min_size = min_size
        self.max_size = max_size
        self.max_graph_nodes = max_graph_nodes
        self._digest = digest

    def _groups_for_size(
        self,
        graph: Graph,
        topology: GraphTopology,
        order: tuple[str, ...],
        size: int,
        prefix: list[int],
        powers: list[int],
        blocked: list[bool],
        position: dict[str, int],
        token: CancellationToken | None,
    ) -> list[tuple[str, list[tuple[str, ...]]]]:
        """Confirmed, non-overlapping repeat groups for one window size."""
        occupied = [0] * (len(order) + 1)
        for index, flag in enumerate(blocked):
            occupied[index + 1] = occupied[index] + (1 if flag else 0)
        buckets: dict[int, list[int]] = {}
        for start in range(len(order) - size + 1):
            if token is not None and start % 512 == 0:
                token.raise_if_cancelled()
            if occupied[start + size] - occupied[start]:
                continue
            window_hash = (
                prefix[start + size] - prefix[start] * powers[size]
            ) % _HASH_MODULUS
            buckets.setdefault(window_hash, []).append(start)
        groups: list[tuple[str, list[tuple[str, ...]]]] = []
        for window_hash in sorted(buckets):
            starts = buckets[window_hash]
            if len(starts) < 2:
                continue
            picked: list[tuple[str, ...]] = []
            used: set[str] = set()
            for start in starts:
                window = order[start : start + size]
                if used.intersection(window):
                    continue
                picked.append(window)
                used.update(window)
            if len(picked) < 2:
                continue
            # Confirm against full canonical signatures: equal hashes are only
            # a bucket key, so anything that disagrees is dropped rather than
            # grouped.
            reference = _window_signature(graph, topology, picked[0])
            confirmed = [
                window
                for window in picked
                if _window_signature(graph, topology, window) == reference
            ]
            if len(confirmed) < 2:
                continue
            groups.append((self._digest(reference), confirmed))

        # Different alignments of the same size land in different buckets and
        # can overlap each other, so accept the alignment that explains the
        # most nodes first. Without this, an arbitrary shifted alignment could
        # win and leave the aligned blocks fragmented.
        selected: list[tuple[str, list[tuple[str, ...]]]] = []
        taken: set[str] = set()
        for digest, windows in sorted(
            groups,
            key=lambda item: (-len(item[1]), position[item[1][0][0]]),
        ):
            kept = [window for window in windows if not taken.intersection(window)]
            if len(kept) < 2:
                continue
            selected.append((digest, kept))
            taken.update(node_id for window in kept for node_id in window)
        return selected

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
        log: DiagnosticLog | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document
        if len(graph.nodes) > self.max_graph_nodes:
            if log is not None:
                log.add(
                    "hierarchy.repeat-analysis-skipped",
                    Severity.INFO,
                    f"{len(graph.nodes)} nodes exceed the "
                    f"{self.max_graph_nodes}-node repeated-structure scan "
                    "limit, so identical blocks were not grouped.",
                    graph.id,
                )
            return ()
        topology = GraphTopology(graph)
        order = _block_scan_order(topology)
        if len(order) < 2 * self.min_size:
            return ()
        sequence: list[int] = []
        seen_signatures: dict[str, int] = {}
        for node_id in order:
            signature = json.dumps(
                _node_signature(graph, graph.node(node_id)),
                sort_keys=True,
                separators=(",", ":"),
            )
            sequence.append(seen_signatures.setdefault(signature, len(seen_signatures)))
        prefix, powers = _prefix_hashes(sequence)
        position = {node_id: index for index, node_id in enumerate(order)}
        blocked = [False] * len(order)
        candidates: list[GroupCandidate] = []
        largest = min(self.max_size, len(order) // 2)
        for size in range(largest, self.min_size - 1, -1):
            if token is not None:
                token.raise_if_cancelled()
            groups = self._groups_for_size(
                graph, topology, order, size, prefix, powers, blocked, position, token
            )
            if not groups:
                continue
            coverage = sum(len(windows) for _, windows in groups) * size
            chosen_size, chosen = size, groups
            # A block that is itself periodic should be reported at its own
            # period: two stacked layers match as one window, but the single
            # layer explains just as many nodes and is the useful block.
            for divisor in range(self.min_size, size):
                if size % divisor:
                    continue
                alternative = self._groups_for_size(
                    graph,
                    topology,
                    order,
                    divisor,
                    prefix,
                    powers,
                    blocked,
                    position,
                    token,
                )
                # Explaining the same nodes is not enough: many *different*
                # small windows can tile the same span, which would demote one
                # meaningful block into a pile of fragments. Only accept a
                # divisor that keeps the grouping at least as consolidated,
                # which is what genuine internal periodicity looks like.
                if (
                    alternative
                    and len(alternative) <= len(groups)
                    and sum(len(windows) for _, windows in alternative) * divisor
                    >= coverage
                ):
                    chosen_size, chosen = divisor, alternative
                    break
            for digest, windows in chosen:
                label_ops = " → ".join(
                    graph.node(item).op_type for item in windows[0][:3]
                )
                if chosen_size > 3:
                    label_ops += " → …"
                for occurrence, window in enumerate(windows, 1):
                    candidates.append(
                        GroupCandidate(
                            graph_id=graph.id,
                            detector=self.name,
                            detector_version=self.version,
                            label=f"Repeated block: {label_ops}",
                            members=frozenset(window),
                            kind=CandidateKind.REPEATED,
                            confidence=min(0.98, 0.84 + chosen_size * 0.01),
                            evidence=(
                                DetectionEvidence(
                                    "repeat.canonical-signature",
                                    f"Occurrence {occurrence} of "
                                    f"{len(windows)} has the same canonical "
                                    "operator, attribute, shape, and internal "
                                    f"edge signature ({digest}); the full "
                                    "signature was compared after hashing.",
                                    window,
                                ),
                            ),
                        )
                    )
                for window in windows:
                    for node_id in window:
                        blocked[position[node_id]] = True
        return tuple(candidates)


DEFAULT_DETECTORS: Final[tuple[GroupDetector, ...]] = (
    SourceHierarchyDetector(),
    StructuralRegionDetector(),
    PatternDetector(),
    RepeatedSubgraphDetector(),
)


class DetectionPipeline:
    """Runs independent detectors and reconciles their evidence per graph."""

    __slots__ = ("detectors",)

    def __init__(
        self, detectors: tuple[GroupDetector, ...] = DEFAULT_DETECTORS
    ) -> None:
        self.detectors = detectors

    def candidates(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
        log: DiagnosticLog | None = None,
    ) -> tuple[GroupCandidate, ...]:
        found: list[GroupCandidate] = []
        for detector in self.detectors:
            if token is not None:
                token.raise_if_cancelled()
            found.extend(detector.detect(document, graph, token, log))
        return tuple(found)

    def detect_graph(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> Hierarchy:
        log = DiagnosticLog()
        candidates = self.candidates(document, graph, token, log)
        hierarchy = reconcile_candidates(graph, candidates)
        return Hierarchy(
            hierarchy.graph_id,
            hierarchy.groups,
            revision=hierarchy.revision,
            diagnostics=tuple(log),
        )

    def detect_document(
        self,
        document: Document,
        token: CancellationToken | None = None,
    ) -> dict[str, Hierarchy]:
        return {
            graph.id: self.detect_graph(document, graph, token)
            for graph in document.graphs.values()
        }
