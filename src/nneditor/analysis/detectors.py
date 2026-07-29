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
from collections.abc import Callable, Iterable
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
    ) -> tuple[GroupCandidate, ...]:
        del document
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

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document
        if len(graph.nodes) > MAX_EXACT_DOMINATOR_NODES:
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
    """A versioned, human-readable linear operator motif."""

    name: str
    label: str
    stages: tuple[frozenset[str], ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.name or not self.label or len(self.stages) < 2:
            raise ValueError("a pattern needs a name, label, and at least two stages")
        if any(not stage for stage in self.stages):
            raise ValueError("pattern stages cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("pattern confidence must be in [0, 1]")


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

DEFAULT_PATTERNS: Final[tuple[Pattern, ...]] = (
    Pattern(
        "conv-norm-activation",
        "Conv + norm + activation",
        (frozenset(("Conv", "ConvTranspose")), _NORMS, _ACTIVATIONS),
        0.96,
    ),
    Pattern(
        "attention-core",
        "Attention",
        (_LINEAR, frozenset(("Softmax",)), _LINEAR),
        0.94,
    ),
    Pattern(
        "feed-forward",
        "Feed-forward",
        (_LINEAR, _ACTIVATIONS, _LINEAR),
        0.94,
    ),
    Pattern(
        "norm-activation",
        "Normalization",
        (_NORMS, _ACTIVATIONS),
        0.82,
    ),
)


class PatternDetector:
    """Matches the deterministic, versioned motif library."""

    name = "operator-pattern"
    version = 1

    def __init__(self, patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS) -> None:
        self.patterns = patterns

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document
        topology = GraphTopology(graph)
        candidates: list[GroupCandidate] = []
        for start in graph.nodes:
            if token is not None:
                token.raise_if_cancelled()
            for pattern in self.patterns:
                if start.op_type not in pattern.stages[0]:
                    continue
                matched = [start.id]
                current = start.id
                for stage in pattern.stages[1:]:
                    successors = topology.successors(current)
                    choices = [
                        item for item in successors if graph.node(item).op_type in stage
                    ]
                    if len(choices) != 1:
                        break
                    current = choices[0]
                    matched.append(current)
                if len(matched) != len(pattern.stages):
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
                                + ".",
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
    member_set = set(node_ids)
    internal_edges = sorted(
        (source, target)
        for source in node_ids
        for target in topology.successors(source)
        if target in member_set
    )
    payload = {
        "nodes": [_node_signature(graph, graph.node(item)) for item in node_ids],
        "edges": [
            [node_ids.index(source), node_ids.index(target)]
            for source, target in internal_edges
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


DigestFunction = Callable[[str], str]


def _default_digest(signature: str) -> str:
    return hashlib.blake2b(signature.encode(), digest_size=16).hexdigest()


class RepeatedSubgraphDetector:
    """Finds repeated, shape- and attribute-aware declaration-order windows.

    Hashes are only bucket keys. Full canonical signatures are compared before
    a group is emitted, so a collision can at worst reduce performance; it
    cannot group different structures.
    """

    name = "repeated-subgraph"
    version = 1

    def __init__(
        self,
        *,
        min_size: int = 2,
        max_size: int = 12,
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

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> tuple[GroupCandidate, ...]:
        del document
        if len(graph.nodes) > self.max_graph_nodes:
            return ()
        topology = GraphTopology(graph)
        order = tuple(node.id for node in graph.nodes)
        claimed: set[str] = set()
        candidates: list[GroupCandidate] = []
        largest = min(self.max_size, len(order) // 2)
        for size in range(largest, self.min_size - 1, -1):
            if token is not None:
                token.raise_if_cancelled()
            buckets: dict[str, dict[str, list[tuple[str, ...]]]] = {}
            for start in range(0, len(order) - size + 1):
                if token is not None and start % 256 == 0:
                    token.raise_if_cancelled()
                window = order[start : start + size]
                if set(window) & claimed:
                    continue
                signature = _window_signature(graph, topology, window)
                digest = self._digest(signature)
                buckets.setdefault(digest, {}).setdefault(signature, []).append(window)
            for digest, by_signature in sorted(buckets.items()):
                # Different signatures in this bucket are an explicit hash
                # collision and stay isolated.
                for _signature, windows in sorted(by_signature.items()):
                    selected: list[tuple[str, ...]] = []
                    used: set[str] = set()
                    for window in windows:
                        if not (set(window) & used):
                            selected.append(window)
                            used.update(window)
                    if len(selected) < 2:
                        continue
                    label_ops = " → ".join(
                        graph.node(item).op_type for item in selected[0][:3]
                    )
                    if size > 3:
                        label_ops += " → …"
                    for occurrence, window in enumerate(selected, 1):
                        members = frozenset(window)
                        candidates.append(
                            GroupCandidate(
                                graph_id=graph.id,
                                detector=self.name,
                                detector_version=self.version,
                                label=f"Repeated block: {label_ops}",
                                members=members,
                                kind=CandidateKind.REPEATED,
                                confidence=min(0.98, 0.84 + size * 0.01),
                                evidence=(
                                    DetectionEvidence(
                                        "repeat.canonical-signature",
                                        f"Occurrence {occurrence} of "
                                        f"{len(selected)} has the same canonical "
                                        "operator, attribute, shape, and internal "
                                        f"edge signature ({digest}); the full "
                                        "signature was compared after hashing.",
                                        window,
                                    ),
                                ),
                            )
                        )
                    claimed.update(used)
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
    ) -> tuple[GroupCandidate, ...]:
        found: list[GroupCandidate] = []
        for detector in self.detectors:
            if token is not None:
                token.raise_if_cancelled()
            found.extend(detector.detect(document, graph, token))
        return tuple(found)

    def detect_graph(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> Hierarchy:
        return reconcile_candidates(graph, self.candidates(document, graph, token))

    def detect_document(
        self,
        document: Document,
        token: CancellationToken | None = None,
    ) -> dict[str, Hierarchy]:
        return {
            graph.id: self.detect_graph(document, graph, token)
            for graph in document.graphs.values()
        }
