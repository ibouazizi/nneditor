"""Manual hierarchy controls and versioned sidecar persistence (Phase 2).

Automatic hierarchy is reproducible and never serialized into the model.
User corrections are authoritative view state stored under the NNEditor state
directory. Every operation recomputes a valid nested forest and saves
atomically, so a crash cannot leave half an override document.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Final, cast

from nneditor.analysis.detectors import DetectionPipeline
from nneditor.analysis.hierarchy import (
    CandidateKind,
    DetectionEvidence,
    Group,
    GroupCandidate,
    Hierarchy,
    hierarchy_revision,
    reconcile_candidates,
)
from nneditor.application.persistence import default_state_directory
from nneditor.cancellation import CancellationToken
from nneditor.diagnostics import Diagnostic, DiagnosticLog
from nneditor.ir.core import Document
from nneditor.ir.identity import user_group_id

__all__ = [
    "HierarchyController",
    "HierarchyOverrides",
    "HierarchySidecarStore",
    "ManualGroup",
    "OrganizationMode",
]

SIDECAR_VERSION: Final = 2

# Version 1 predates the per-artifact organization mode; those documents are
# fully understood by this code (mode simply defaults to AUTO), so they load
# rather than being discarded like a genuinely newer, unknown version.
_READABLE_SIDECAR_VERSIONS: Final = frozenset((1, SIDECAR_VERSION))


class OrganizationMode(StrEnum):
    """Which detector evidence organizes blocks; declaration order is the
    toolbar cycling order."""

    AUTO = "auto"
    SOURCE = "source"
    REPEATED = "repeated"
    PATTERNS = "patterns"
    STRUCTURAL = "structural"


_MODE_KINDS: Final[dict[OrganizationMode, frozenset[CandidateKind] | None]] = {
    # AUTO pools every detector and reconciles by priority (today's behavior).
    OrganizationMode.AUTO: None,
    OrganizationMode.SOURCE: frozenset(
        (CandidateKind.SOURCE, CandidateKind.CONTROL_FLOW)
    ),
    OrganizationMode.REPEATED: frozenset(
        (CandidateKind.REPEATED, CandidateKind.CONTROL_FLOW)
    ),
    OrganizationMode.PATTERNS: frozenset(
        (CandidateKind.PATTERN, CandidateKind.CONTROL_FLOW)
    ),
    OrganizationMode.STRUCTURAL: frozenset(
        (CandidateKind.STRUCTURAL, CandidateKind.CONTROL_FLOW)
    ),
}


@dataclass(frozen=True, slots=True)
class ManualGroup:
    """One user-authored group persisted in the sidecar."""

    id: str
    graph_id: str
    label: str
    members: frozenset[str]
    locked: bool = False

    def __post_init__(self) -> None:
        if not self.id or not self.graph_id or not self.label.strip():
            raise ValueError("manual group needs ids and a label")
        if len(self.members) < 2:
            raise ValueError("manual groups need at least two members")


@dataclass(slots=True)
class HierarchyOverrides:
    """Corrections for one artifact, partitioned by graph through entity ids."""

    manual_groups: dict[str, ManualGroup] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    renames: dict[str, str] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)
    mode: OrganizationMode = OrganizationMode.AUTO
    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def is_empty(self) -> bool:
        return self.mode is OrganizationMode.AUTO and not (
            self.manual_groups or self.rejected or self.renames or self.locked
        )


def _serialize_overrides(overrides: HierarchyOverrides) -> dict[str, object] | None:
    """Take one internally consistent snapshot for sidecar persistence."""
    with overrides.lock:
        if overrides.is_empty():
            return None
        return {
            "manual_groups": [
                {
                    "id": group.id,
                    "graph_id": group.graph_id,
                    "label": group.label,
                    "members": sorted(group.members),
                    "locked": group.locked,
                }
                for group in sorted(
                    overrides.manual_groups.values(),
                    key=lambda item: item.id,
                )
            ],
            "rejected": sorted(overrides.rejected),
            "renames": dict(sorted(overrides.renames.items())),
            "locked": sorted(overrides.locked),
            "mode": overrides.mode.value,
        }


class HierarchySidecarStore:
    """One disposable, versioned view document for hierarchy overrides."""

    __slots__ = ("_artifacts", "_lock", "path")

    def __init__(self, directory: Path | None = None) -> None:
        base = directory if directory is not None else default_state_directory()
        self.path = base / "hierarchy-view.json"
        self._artifacts = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict[str, HierarchyOverrides]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(raw, dict)
            or raw.get("version") not in _READABLE_SIDECAR_VERSIONS
        ):
            return {}
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, dict):
            return {}
        loaded: dict[str, HierarchyOverrides] = {}
        try:
            for content_hash, data in artifacts.items():
                if not isinstance(data, dict):
                    raise TypeError
                overrides = HierarchyOverrides()
                for item in data.get("manual_groups", []):
                    group = ManualGroup(
                        id=str(item["id"]),
                        graph_id=str(item["graph_id"]),
                        label=str(item["label"]),
                        members=frozenset(str(value) for value in item["members"]),
                        locked=bool(item.get("locked", False)),
                    )
                    overrides.manual_groups[group.id] = group
                overrides.rejected.update(
                    str(item) for item in data.get("rejected", [])
                )
                overrides.renames.update(
                    (str(key), str(value))
                    for key, value in data.get("renames", {}).items()
                )
                overrides.locked.update(str(item) for item in data.get("locked", []))
                # Version-1 documents predate the field; an unknown value
                # (from a newer build) degrades to AUTO instead of discarding
                # the user's manual corrections wholesale.
                try:
                    overrides.mode = OrganizationMode(
                        str(data.get("mode", OrganizationMode.AUTO.value))
                    )
                except ValueError:
                    overrides.mode = OrganizationMode.AUTO
                loaded[str(content_hash)] = overrides
        except (KeyError, TypeError, ValueError):
            return {}
        return loaded

    def for_artifact(self, content_hash: str) -> HierarchyOverrides:
        # The lock keeps this setdefault from resizing the dict while a
        # concurrent save() iterates it (two open jobs race exactly this way).
        with self._lock:
            return self._artifacts.setdefault(content_hash, HierarchyOverrides())

    def save(self) -> None:
        # A unique temp name plus the lock keeps concurrent saves from
        # colliding on one .partial file, which raises PermissionError on
        # Windows when os.replace hits a file another thread holds open.
        partial = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.partial")
        with self._lock:
            artifacts: dict[str, object] = {}
            for content_hash, overrides in sorted(self._artifacts.items()):
                snapshot = _serialize_overrides(overrides)
                if snapshot is not None:
                    artifacts[content_hash] = snapshot
            payload = {
                "version": SIDECAR_VERSION,
                "artifacts": artifacts,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(partial, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=1, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    partial.unlink(missing_ok=True)
                raise


def _manual_as_group(group: ManualGroup) -> Group:
    return Group(
        id=group.id,
        graph_id=group.graph_id,
        label=group.label,
        members=group.members,
        kind=CandidateKind.MANUAL,
        confidence=1.0,
        evidence=(
            DetectionEvidence(
                "manual.user-group",
                "This group was created by the user and overrides automatic "
                "overlap decisions.",
                tuple(sorted(group.members)),
            ),
        ),
        locked=group.locked,
        automatic=False,
    )


def _rank(group: Group) -> tuple[int, int, float, int, str]:
    return (
        int(not group.automatic),
        int(group.locked),
        group.confidence,
        len(group.members),
        group.id,
    )


def _nest(
    graph_id: str,
    groups: list[Group],
    diagnostics: tuple[Diagnostic, ...] = (),
) -> Hierarchy:
    accepted: list[Group] = []
    for candidate in sorted(groups, key=_rank, reverse=True):
        if any(
            candidate.members & other.members
            and not (
                candidate.members <= other.members or other.members <= candidate.members
            )
            for other in accepted
        ):
            continue
        accepted.append(candidate)
    nested: list[Group] = []
    for group in accepted:
        parents = [other for other in accepted if group.members < other.members]
        parent = (
            min(parents, key=lambda item: (len(item.members), item.id))
            if parents
            else None
        )
        nested.append(replace(group, parent_id=parent.id if parent else None))
    return Hierarchy(
        graph_id,
        nested,
        revision=hierarchy_revision(graph_id, nested),
        diagnostics=diagnostics,
    )


def _read_overrides_locked[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Read a shared override object without racing a sibling controller."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        controller = cast("HierarchyController", args[0])
        with controller._overrides.lock:
            return method(*args, **kwargs)

    return guarded


def _mutate_overrides_locked[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Serialize an override mutation, then persist the latest snapshot."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        controller = cast("HierarchyController", args[0])
        with controller._overrides.lock:
            result = method(*args, **kwargs)
        # Never wait for the store lock while holding the override lock:
        # save() snapshots in the opposite order so concurrent sessions cannot
        # deadlock.
        if controller._store is not None:
            controller._store.save()
        return result

    return guarded


class HierarchyController:
    """Automatic hierarchy plus the complete manual correction command set."""

    __slots__ = (
        "_automatic_by_mode",
        "_candidates",
        "_diagnostics",
        "_document",
        "_hierarchies",
        "_overrides",
        "_store",
    )

    def __init__(
        self,
        document: Document,
        *,
        pipeline: DetectionPipeline | None = None,
        store: HierarchySidecarStore | None = None,
        token: CancellationToken | None = None,
    ) -> None:
        self._document = document
        self._store = store
        self._overrides = (
            store.for_artifact(document.source.content_hash)
            if store is not None
            else HierarchyOverrides()
        )
        # Every detector runs exactly once per document; organization modes
        # only filter this pool before reconciliation, so switching modes is
        # a re-reconcile, never a re-detect.
        active = pipeline or DetectionPipeline()
        # Detectors report the analysis they declined to run (a graph over a
        # scan bound, say) so the shell can explain missing blocks instead of
        # showing an unexplained gap.
        logs = {graph.id: DiagnosticLog() for graph in document.graphs.values()}
        self._candidates: dict[str, tuple[GroupCandidate, ...]] = {
            graph.id: active.candidates(document, graph, token, logs[graph.id])
            for graph in document.graphs.values()
        }
        self._diagnostics: dict[str, tuple[Diagnostic, ...]] = {
            graph_id: tuple(log) for graph_id, log in logs.items()
        }
        self._automatic_by_mode: dict[OrganizationMode, dict[str, Hierarchy]] = {}
        self._hierarchies: dict[str, Hierarchy] = {}
        self._recompute()

    @_read_overrides_locked
    def hierarchy(self, graph_id: str) -> Hierarchy:
        return self._hierarchies[graph_id]

    @_read_overrides_locked
    def diagnostics(self, graph_id: str) -> tuple[Diagnostic, ...]:
        """Coded reasons that automatic analysis skipped part of this graph."""
        return self._diagnostics.get(graph_id, ())

    @property
    @_read_overrides_locked
    def hierarchies(self) -> tuple[Hierarchy, ...]:
        return tuple(self._hierarchies.values())

    @property
    @_read_overrides_locked
    def mode(self) -> OrganizationMode:
        """The persisted per-artifact organization mode."""
        return self._overrides.mode

    @_mutate_overrides_locked
    def set_mode(self, mode: OrganizationMode) -> None:
        """Reorganize blocks around one detector's evidence.

        Manual corrections (groups, locks, rejections, renames) are user
        intent, not detector output, and apply in every mode.
        """
        if mode is self._overrides.mode:
            return
        self._overrides.mode = mode
        self._save_recompute()

    def _automatic_for(self, mode: OrganizationMode) -> dict[str, Hierarchy]:
        cached = self._automatic_by_mode.get(mode)
        if cached is None:
            kinds = _MODE_KINDS[mode]
            cached = {
                graph_id: reconcile_candidates(
                    self._document.graphs[graph_id],
                    (
                        candidates
                        if kinds is None
                        else tuple(item for item in candidates if item.kind in kinds)
                    ),
                )
                for graph_id, candidates in self._candidates.items()
            }
            self._automatic_by_mode[mode] = cached
        return cached

    @_mutate_overrides_locked
    def group(
        self,
        graph_id: str,
        label: str,
        members: frozenset[str],
        *,
        locked: bool = False,
    ) -> Group:
        """Create a manual group, resolving partial automatic overlaps."""
        graph = self._document.graphs[graph_id]
        node_ids = {node.id for node in graph.nodes}
        if len(members) < 2:
            raise ValueError("group needs at least two nodes")
        if not members <= node_ids:
            raise ValueError("group contains nodes outside the graph")
        group_id = user_group_id(graph_id, label)
        if group_id in self._overrides.manual_groups:
            raise ValueError(f"manual group label {label!r} is already used")
        for existing in self._overrides.manual_groups.values():
            overlap = members & existing.members
            if overlap and not (
                members <= existing.members or existing.members <= members
            ):
                raise ValueError(
                    "manual groups cannot partially overlap; split or merge the "
                    "existing group first"
                )
        manual = ManualGroup(group_id, graph_id, label, members, locked)
        self._overrides.manual_groups[group_id] = manual
        self._save_recompute()
        return self.hierarchy(graph_id).group(group_id)

    @_mutate_overrides_locked
    def split(
        self,
        graph_id: str,
        group_id: str,
        partitions: tuple[frozenset[str], ...] = (),
    ) -> tuple[Group, ...]:
        """Remove a group and optionally replace it with explicit partitions."""
        hierarchy = self.hierarchy(graph_id)
        target = hierarchy.group(group_id)
        if target.locked:
            raise ValueError("unlock the group before splitting it")
        if partitions:
            union = frozenset().union(*partitions)
            if union != target.members:
                raise ValueError("split partitions must exactly cover the group")
            for position, left in enumerate(partitions):
                if not left:
                    raise ValueError("split partitions cannot be empty")
                for right in partitions[position + 1 :]:
                    if left & right:
                        raise ValueError("split partitions must be disjoint")
        planned: list[tuple[str, str, frozenset[str]]] = []
        for position, members in enumerate(partitions, 1):
            if len(members) < 2:
                continue  # singleton partitions become ungrouped nodes
            label = f"{target.label} {position}"
            group_id_new = user_group_id(graph_id, label)
            # Checked before any mutation: a derived label colliding with an
            # existing manual group must fail cleanly, not overwrite it.
            if (
                group_id_new != target.id
                and group_id_new in self._overrides.manual_groups
            ):
                raise ValueError(f"manual group label {label!r} is already used")
            planned.append((group_id_new, label, members))
        self._remove_group_override(target)
        created_ids: list[str] = []
        for group_id_new, label, members in planned:
            manual = ManualGroup(group_id_new, graph_id, label, members)
            self._overrides.manual_groups[group_id_new] = manual
            created_ids.append(group_id_new)
        self._save_recompute()
        hierarchy = self.hierarchy(graph_id)
        return tuple(hierarchy.group(item) for item in created_ids)

    @_mutate_overrides_locked
    def merge(self, graph_id: str, group_ids: tuple[str, ...], label: str) -> Group:
        if len(set(group_ids)) < 2:
            raise ValueError("merge needs at least two different groups")
        hierarchy = self.hierarchy(graph_id)
        groups = [hierarchy.group(item) for item in group_ids]
        if any(group.locked for group in groups):
            raise ValueError("unlock groups before merging them")
        members = frozenset().union(*(group.members for group in groups))
        group_id = user_group_id(graph_id, label)
        # The derived id must not silently overwrite an unrelated manual
        # group; reusing the label of a group being merged away is fine.
        if group_id in self._overrides.manual_groups and group_id not in group_ids:
            raise ValueError(f"manual group label {label!r} is already used")
        for group in groups:
            self._remove_group_override(group)
        self._overrides.manual_groups[group_id] = ManualGroup(
            group_id, graph_id, label, members
        )
        self._save_recompute()
        return self.hierarchy(graph_id).group(group_id)

    @_mutate_overrides_locked
    def rename(self, graph_id: str, group_id: str, label: str) -> Group:
        if not label.strip():
            raise ValueError("group label cannot be empty")
        target = self.hierarchy(graph_id).group(group_id)
        manual = self._overrides.manual_groups.get(group_id)
        if manual is not None:
            new_id = user_group_id(graph_id, label)
            # Collision check before any mutation: popping first would
            # destroy the group when the new label is taken.
            if new_id != group_id and new_id in self._overrides.manual_groups:
                raise ValueError(f"manual group label {label!r} is already used")
            self._overrides.manual_groups.pop(group_id, None)
            self._overrides.manual_groups[new_id] = replace(
                manual, id=new_id, label=label
            )
            self._overrides.locked.discard(group_id)
            self._save_recompute()
            return self.hierarchy(graph_id).group(new_id)
        if not target.automatic:
            raise KeyError(group_id)
        self._overrides.renames[group_id] = label
        self._save_recompute()
        return self.hierarchy(graph_id).group(group_id)

    @_mutate_overrides_locked
    def lock(self, graph_id: str, group_id: str, *, locked: bool = True) -> Group:
        target = self.hierarchy(graph_id).group(group_id)
        manual = self._overrides.manual_groups.get(group_id)
        if manual is not None:
            self._overrides.manual_groups[group_id] = replace(manual, locked=locked)
        elif locked:
            self._overrides.locked.add(group_id)
        else:
            self._overrides.locked.discard(group_id)
        self._save_recompute()
        return self.hierarchy(graph_id).group(target.id)

    @_mutate_overrides_locked
    def reject(self, graph_id: str, group_id: str) -> None:
        target = self.hierarchy(graph_id).group(group_id)
        if target.locked:
            raise ValueError("unlock the group before rejecting it")
        self._remove_group_override(target)
        self._save_recompute()

    @_mutate_overrides_locked
    def reset(self, graph_id: str | None = None) -> None:
        """Discard all corrections, or only corrections belonging to a graph."""
        if graph_id is None:
            self._overrides.manual_groups.clear()
            self._overrides.rejected.clear()
            self._overrides.renames.clear()
            self._overrides.locked.clear()
        else:
            # Candidate ids cover every group any organization mode can
            # reconcile for this graph, so corrections made under one mode
            # are discarded even when reset runs under another.
            auto_ids = {candidate.id for candidate in self._candidates[graph_id]}
            self._overrides.manual_groups = {
                key: value
                for key, value in self._overrides.manual_groups.items()
                if value.graph_id != graph_id
            }
            self._overrides.rejected.difference_update(auto_ids)
            for key in auto_ids:
                self._overrides.renames.pop(key, None)
            self._overrides.locked.difference_update(auto_ids)
        self._save_recompute()

    def _remove_group_override(self, group: Group) -> None:
        if group.automatic:
            self._overrides.rejected.add(group.id)
            self._overrides.renames.pop(group.id, None)
            self._overrides.locked.discard(group.id)
        else:
            self._overrides.manual_groups.pop(group.id, None)

    def _save_recompute(self) -> None:
        self._recompute()

    def _recompute(self) -> None:
        result: dict[str, Hierarchy] = {}
        for graph_id, automatic in self._automatic_for(self._overrides.mode).items():
            groups: list[Group] = []
            for group in automatic.groups:
                if group.id in self._overrides.rejected:
                    continue
                groups.append(
                    replace(
                        group,
                        label=self._overrides.renames.get(group.id, group.label),
                        locked=group.id in self._overrides.locked,
                        parent_id=None,
                    )
                )
            groups.extend(
                _manual_as_group(group)
                for group in self._overrides.manual_groups.values()
                if group.graph_id == graph_id
            )
            result[graph_id] = _nest(
                graph_id, groups, self._diagnostics.get(graph_id, ())
            )
        self._hierarchies = result
