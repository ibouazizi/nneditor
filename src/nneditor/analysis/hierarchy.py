"""Hierarchy candidates, reconciled groups, and detector contracts (Phase 2).

Hierarchy is deliberately *view state*, not canonical model state. Detectors
emit immutable candidates with evidence; reconciliation turns compatible
candidates into a nested hierarchy with stable identifiers. Manual corrections
are applied later by :mod:`nneditor.application.hierarchy` and persisted in a
sidecar document.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from nneditor.cancellation import CancellationToken
from nneditor.ir.core import Document, Graph
from nneditor.ir.identity import auto_group_id

__all__ = [
    "CandidateKind",
    "DetectionEvidence",
    "Group",
    "GroupCandidate",
    "GroupDetector",
    "Hierarchy",
    "candidate_priority",
    "hierarchy_revision",
    "reconcile_candidates",
]


class CandidateKind(StrEnum):
    """Why a candidate exists and which semantic level it normally represents."""

    SOURCE = "source"
    STRUCTURAL = "structural"
    PATTERN = "pattern"
    REPEATED = "repeated"
    CONTROL_FLOW = "control-flow"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class DetectionEvidence:
    """One human-readable reason supporting a grouping decision."""

    code: str
    summary: str
    node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("evidence needs a code")
        if not self.summary.strip():
            raise ValueError("evidence needs a summary")


@dataclass(frozen=True, slots=True)
class GroupCandidate:
    """A detector's proposed group before overlap reconciliation."""

    graph_id: str
    detector: str
    detector_version: int
    label: str
    members: frozenset[str]
    kind: CandidateKind
    confidence: float
    evidence: tuple[DetectionEvidence, ...]

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("a group candidate needs a graph id")
        if not self.detector:
            raise ValueError("a group candidate needs a detector")
        if self.detector_version < 1:
            raise ValueError("detector_version must be positive")
        if not self.label.strip():
            raise ValueError("a group candidate needs a label")
        if not self.members:
            raise ValueError("a group candidate needs at least one member")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be in [0, 1]")
        if not self.evidence:
            raise ValueError("a group candidate needs explainable evidence")

    @property
    def id(self) -> str:
        return auto_group_id(self.detector, self.detector_version, self.members)


@runtime_checkable
class GroupDetector(Protocol):
    """Replaceable analysis boundary for automatic hierarchy algorithms."""

    name: str
    version: int

    def detect(
        self,
        document: Document,
        graph: Graph,
        token: CancellationToken | None = None,
    ) -> tuple[GroupCandidate, ...]:
        """Return deterministic candidates for ``graph``."""


@dataclass(frozen=True, slots=True)
class Group:
    """One accepted automatic or manual group."""

    id: str
    graph_id: str
    label: str
    members: frozenset[str]
    kind: CandidateKind
    confidence: float
    evidence: tuple[DetectionEvidence, ...]
    parent_id: str | None = None
    locked: bool = False
    automatic: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.graph_id:
            raise ValueError("a group needs ids")
        if not self.label.strip() or not self.members:
            raise ValueError("a group needs a label and members")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("group confidence must be in [0, 1]")

    @property
    def explanation(self) -> str:
        return " ".join(item.summary for item in self.evidence)


class Hierarchy:
    """A validated forest of nested, non-partially-overlapping groups."""

    __slots__ = ("_children", "_groups", "_node_groups", "graph_id", "revision")

    def __init__(
        self,
        graph_id: str,
        groups: Iterable[Group] = (),
        *,
        revision: str | None = None,
    ) -> None:
        self.graph_id = graph_id
        group_map: dict[str, Group] = {}
        for group in groups:
            if group.graph_id != graph_id:
                raise ValueError(
                    f"group {group.id!r} belongs to {group.graph_id!r}, "
                    f"not hierarchy {graph_id!r}"
                )
            if group.id in group_map:
                raise ValueError(f"duplicate group id {group.id!r}")
            group_map[group.id] = group
        for group in group_map.values():
            if group.parent_id is not None:
                parent = group_map.get(group.parent_id)
                if parent is None:
                    raise ValueError(
                        f"group {group.id!r} has missing parent {group.parent_id!r}"
                    )
                if not group.members < parent.members:
                    raise ValueError(
                        f"group {group.id!r} is not strictly contained by its parent"
                    )
        values = tuple(group_map.values())
        for position, left in enumerate(values):
            for right in values[position + 1 :]:
                overlap = left.members & right.members
                if overlap and not (
                    left.members <= right.members or right.members <= left.members
                ):
                    raise ValueError(
                        f"groups {left.id!r} and {right.id!r} partially overlap"
                    )
        children: dict[str | None, list[str]] = {}
        node_groups: dict[str, list[str]] = {}
        for group in values:
            children.setdefault(group.parent_id, []).append(group.id)
            for member in group.members:
                node_groups.setdefault(member, []).append(group.id)
        for ids in children.values():
            ids.sort(key=lambda item: (len(group_map[item].members), item))
        for ids in node_groups.values():
            ids.sort(key=lambda item: (len(group_map[item].members), item))
        self._groups: Mapping[str, Group] = MappingProxyType(group_map)
        self._children: Mapping[str | None, tuple[str, ...]] = MappingProxyType(
            {key: tuple(ids) for key, ids in children.items()}
        )
        self._node_groups: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {key: tuple(ids) for key, ids in node_groups.items()}
        )
        self.revision = revision or hierarchy_revision(graph_id, values)

    @property
    def groups(self) -> tuple[Group, ...]:
        return tuple(self._groups.values())

    @property
    def roots(self) -> tuple[Group, ...]:
        return tuple(self._groups[item] for item in self._children.get(None, ()))

    def group(self, group_id: str) -> Group:
        return self._groups[group_id]

    def has_group(self, group_id: str) -> bool:
        return group_id in self._groups

    def children(self, group_id: str | None) -> tuple[Group, ...]:
        return tuple(self._groups[item] for item in self._children.get(group_id, ()))

    def groups_for_node(self, node_id: str) -> tuple[Group, ...]:
        """Containing groups, most specific first."""
        return tuple(self._groups[item] for item in self._node_groups.get(node_id, ()))

    def breadcrumbs(self, group_id: str) -> tuple[Group, ...]:
        """Root-to-target path for hierarchy navigation."""
        current = self.group(group_id)
        path = [current]
        seen = {current.id}
        while current.parent_id is not None:
            current = self.group(current.parent_id)
            if current.id in seen:
                raise ValueError("hierarchy contains a parent cycle")
            seen.add(current.id)
            path.append(current)
        return tuple(reversed(path))


_KIND_PRIORITY = {
    CandidateKind.MANUAL: 6,
    CandidateKind.REPEATED: 5,
    CandidateKind.PATTERN: 4,
    CandidateKind.STRUCTURAL: 3,
    CandidateKind.SOURCE: 2,
    CandidateKind.CONTROL_FLOW: 1,
}


def candidate_priority(candidate: GroupCandidate) -> tuple[float, int, int, str]:
    """Deterministic preference for candidates that partially overlap."""
    return (
        candidate.confidence,
        _KIND_PRIORITY[candidate.kind],
        len(candidate.members),
        candidate.id,
    )


def _merge_exact(candidates: Sequence[GroupCandidate]) -> list[GroupCandidate]:
    """Combine independent evidence for identical memberships."""
    by_members: dict[tuple[str, frozenset[str]], list[GroupCandidate]] = {}
    for candidate in candidates:
        by_members.setdefault((candidate.graph_id, candidate.members), []).append(
            candidate
        )
    merged: list[GroupCandidate] = []
    for (_graph_id, _members), items in by_members.items():
        selected = max(items, key=candidate_priority)
        evidence: dict[tuple[str, str, tuple[str, ...]], DetectionEvidence] = {}
        for item in items:
            for proof in item.evidence:
                evidence.setdefault((proof.code, proof.summary, proof.node_ids), proof)
        merged.append(
            GroupCandidate(
                graph_id=selected.graph_id,
                detector=selected.detector,
                detector_version=selected.detector_version,
                label=selected.label,
                members=selected.members,
                kind=selected.kind,
                confidence=max(item.confidence for item in items),
                evidence=tuple(evidence.values()),
            )
        )
    return merged


def reconcile_candidates(
    graph: Graph, candidates: Iterable[GroupCandidate]
) -> Hierarchy:
    """Resolve duplicate, overlapping, and nested candidates deterministically.

    Exact duplicates pool their explanations. Strict nesting is retained.
    Partial overlaps cannot form a hierarchy, so the higher-confidence,
    higher-semantic-priority candidate wins. Candidates naming unknown nodes
    are discarded defensively.
    """
    node_ids = {node.id for node in graph.nodes}
    valid = [
        candidate
        for candidate in candidates
        if candidate.graph_id == graph.id
        and candidate.members <= node_ids
        and (
            len(candidate.members) >= 2 or candidate.kind is CandidateKind.CONTROL_FLOW
        )
    ]
    merged = _merge_exact(valid)
    accepted: list[GroupCandidate] = []
    for candidate in sorted(merged, key=candidate_priority, reverse=True):
        partial = False
        for other in accepted:
            overlap = candidate.members & other.members
            if overlap and not (
                candidate.members <= other.members or other.members <= candidate.members
            ):
                partial = True
                break
        if not partial:
            accepted.append(candidate)

    groups: list[Group] = []
    for candidate in sorted(accepted, key=lambda item: (-len(item.members), item.id)):
        supersets = [other for other in accepted if candidate.members < other.members]
        parent = (
            min(supersets, key=lambda item: (len(item.members), item.id))
            if supersets
            else None
        )
        groups.append(
            Group(
                id=candidate.id,
                graph_id=candidate.graph_id,
                label=candidate.label,
                members=candidate.members,
                kind=candidate.kind,
                confidence=candidate.confidence,
                evidence=candidate.evidence,
                parent_id=parent.id if parent is not None else None,
            )
        )
    return Hierarchy(graph.id, groups)


def hierarchy_revision(graph_id: str, groups: Iterable[Group]) -> str:
    """Content-derived revision used by semantic-layout cache keys."""
    payload = [
        {
            "id": group.id,
            "label": group.label,
            "members": sorted(group.members),
            "kind": group.kind.value,
            "confidence": group.confidence,
            "parent": group.parent_id,
            "locked": group.locked,
            "automatic": group.automatic,
        }
        for group in sorted(groups, key=lambda item: item.id)
    ]
    digest = hashlib.sha256(
        json.dumps(
            {"graph_id": graph_id, "groups": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"hier:{digest}"
