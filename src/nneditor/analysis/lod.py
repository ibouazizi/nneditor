"""Semantic levels of detail and edge aggregation (Phase 2, task P2.7)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from nneditor.analysis.hierarchy import CandidateKind, Group, Hierarchy
from nneditor.analysis.layout import layered_order
from nneditor.rendering.scene import EdgeGlyph, NodeGlyph, Scene

__all__ = [
    "DetailLevel",
    "SemanticScene",
    "detail_for_scale",
    "semantic_scene",
]


class DetailLevel(StrEnum):
    """The four stable representations exposed to navigation and caches."""

    ARCHITECTURE = "architecture"
    BLOCK = "block"
    LAYER = "layer"
    OPERATOR = "operator"


@dataclass(frozen=True, slots=True)
class SemanticScene:
    """A rendered representation plus its mapping back to operator nodes."""

    scene: Scene
    detail_level: DetailLevel
    members_by_glyph: dict[str, frozenset[str]]

    def representative_for(self, node_id: str) -> str | None:
        for glyph_id, members in self.members_by_glyph.items():
            if node_id in members:
                return glyph_id
        return None


def detail_for_scale(scale: float) -> DetailLevel:
    """Default semantic level at a zoom; explicit navigation may override it."""
    if scale < 0.12:
        return DetailLevel.ARCHITECTURE
    if scale < 0.35:
        return DetailLevel.BLOCK
    if scale < 0.75:
        return DetailLevel.LAYER
    return DetailLevel.OPERATOR


def _filtered_scene(scene: Scene, members: frozenset[str] | None) -> Scene:
    if members is None:
        return scene
    nodes = [node for node in scene.nodes if node.id in members]
    present = {node.id for node in nodes}
    edges = [
        edge
        for edge in scene.edges
        if edge.source in present and edge.target in present
    ]
    return Scene(nodes, edges)


_DRILL_DOMINANCE: Final = 0.8
"""Architecture drill-through: descend while one selected group covers more
than this share of the visible scope. Exporters routinely wrap a whole model
in a single root scope; without drill-through that scope renders as one glyph
covering the entire architecture view."""

_FEEDER_OP_TYPES: Final = frozenset({"Constant", "ConstantOfShape"})
"""Attribute-feeder ops that fold into their consumer at coarse detail.

The structural criteria alone (no producers, one consumer glyph) also match a
model's entry compute nodes — the first ``Conv`` consumes only graph inputs —
and those must keep their own glyph, so folding is limited to ops that exist
purely to feed attributes or constants."""

_GROUP_GLYPH_WIDTH: Final = 180.0
_GROUP_GLYPH_HEIGHT: Final = 64.0
_GROUP_GLYPH_MAX_WIDTH: Final = 420.0
_GROUP_LABEL_CHAR_WIDTH: Final = 7.5
_RELAYOUT_COLUMN_GAP: Final = 48.0
_RELAYOUT_ROW_GAP: Final = 64.0


def _base_op_type(type_label: str) -> str:
    """The unqualified op name behind ``domain::Op:overload`` type labels."""
    tail = type_label.rsplit("::", 1)[-1]
    return tail.split(":", 1)[0]


def _group_rank(group: Group) -> tuple[int, float, int, str]:
    priority = {
        CandidateKind.MANUAL: 6,
        CandidateKind.REPEATED: 5,
        CandidateKind.PATTERN: 4,
        CandidateKind.STRUCTURAL: 3,
        CandidateKind.SOURCE: 2,
        CandidateKind.CONTROL_FLOW: 1,
    }[group.kind]
    return priority, group.confidence, len(group.members), group.id


def _selected_groups(
    hierarchy: Hierarchy,
    detail_level: DetailLevel,
    allowed_members: frozenset[str],
    root: Group | None,
) -> tuple[Group, ...]:
    if detail_level is DetailLevel.OPERATOR:
        return ()
    if detail_level is DetailLevel.ARCHITECTURE:
        # Drilling into a group must reveal its interior, so the coarsest
        # antichain inside a drill root is that root's children, not the
        # hierarchy's own roots (which contain the drill root itself).
        top = hierarchy.roots if root is None else hierarchy.children(root.id)
        candidates = [
            group
            for group in top
            if group.members <= allowed_members and len(group.members) >= 2
        ]
    else:
        if detail_level is DetailLevel.BLOCK:
            kinds = {
                CandidateKind.MANUAL,
                CandidateKind.REPEATED,
                CandidateKind.STRUCTURAL,
                CandidateKind.SOURCE,
                CandidateKind.CONTROL_FLOW,
            }
        else:
            kinds = {CandidateKind.MANUAL, CandidateKind.PATTERN}
        # With a drill root, only groups strictly inside it qualify: selecting
        # a group that spans the whole drill scope (the root itself, or a
        # same-membership twin) would render one glyph containing the root and
        # make drilling a no-op.
        candidates = [
            group
            for group in hierarchy.groups
            if group.kind in kinds
            and group.members <= allowed_members
            and len(group.members) >= 2
            and (root is None or group.members < allowed_members)
        ]

    # Choose an antichain. Manual groups and stronger detector evidence win;
    # architecture prefers the larger (coarser) group, while block/layer
    # prefer the smaller (more specific) one so nested scopes do not collapse
    # every level into the outermost glyph.
    selected: list[Group] = []
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.kind is CandidateKind.MANUAL),
            _group_rank(item)[0],
            len(item.members)
            if detail_level is DetailLevel.ARCHITECTURE
            else -len(item.members),
            item.confidence,
            item.id,
        ),
        reverse=True,
    )
    covered: set[str] = set()
    for group in ordered:
        if not (group.members & covered):
            selected.append(group)
            covered.update(group.members)
    return tuple(selected)


def _drill_into_dominant(
    hierarchy: Hierarchy,
    selected: tuple[Group, ...],
    allowed_members: frozenset[str],
) -> tuple[Group, ...]:
    """Descend through groups that swallow the whole architecture view.

    While a single selected group covers more than :data:`_DRILL_DOMINANCE`
    of the visible scope, replace it with its children and reselect, so the
    architecture view lands on the first level with meaningful fan-out (a
    model wrapped in one exporter scope shows its blocks, not one giant
    rectangle). Stops when no group dominates or the dominant group has no
    children to descend into. Children of one parent never overlap, so the
    selection stays an antichain and every operator keeps exactly one glyph.
    """
    current = list(selected)
    total = len(allowed_members)
    while current:
        dominant = max(current, key=lambda item: (len(item.members), item.id))
        if len(dominant.members) <= _DRILL_DOMINANCE * total:
            break
        children = [
            child
            for child in hierarchy.children(dominant.id)
            if len(child.members) >= 2 and child.members <= allowed_members
        ]
        if not children:
            break
        current = [item for item in current if item.id != dominant.id]
        current.extend(children)
    return tuple(current)


def _collapse(
    scene: Scene,
    representatives: dict[str, tuple[str, str, str, frozenset[str]]],
) -> SemanticScene:
    """Collapse ``(glyph id, name, type, represented members)`` mappings."""
    nodes_by_rep: dict[str, list[NodeGlyph]] = {}
    metadata: dict[str, tuple[str, str, frozenset[str]]] = {}
    for node in scene.nodes:
        glyph_id, label, type_label, members = representatives[node.id]
        nodes_by_rep.setdefault(glyph_id, []).append(node)
        metadata[glyph_id] = (label, type_label, members)

    glyphs: list[NodeGlyph] = []
    members_by_glyph: dict[str, frozenset[str]] = {}
    for glyph_id, nodes in nodes_by_rep.items():
        label, type_label, members = metadata[glyph_id]
        members_by_glyph[glyph_id] = members
        if len(nodes) == 1 and glyph_id == nodes[0].id:
            glyphs.append(nodes[0])
            continue
        bounds = nodes[0].bounds
        for node in nodes[1:]:
            bounds = bounds.union(node.bounds)
        padding = 10.0
        glyphs.append(
            NodeGlyph(
                id=glyph_id,
                x=bounds.min_x - padding,
                y=bounds.min_y - padding,
                width=max(80.0, bounds.width + padding * 2),
                height=max(38.0, bounds.height + padding * 2),
                kind="group",
                label=f"{label} ({len(members)})",
                type_label=type_label,
            )
        )

    glyph_map = {node.id: node for node in glyphs}
    aggregated: dict[tuple[str, str], int] = {}
    for edge in scene.edges:
        source = representatives[edge.source][0]
        target = representatives[edge.target][0]
        if source != target:
            aggregated[(source, target)] = aggregated.get((source, target), 0) + 1
    edges: list[EdgeGlyph] = []
    for (source, target), count in sorted(aggregated.items()):
        source_node = glyph_map[source]
        target_node = glyph_map[target]
        digest = hashlib.blake2b(
            f"{source}\0{target}".encode(), digest_size=8
        ).hexdigest()
        edges.append(
            EdgeGlyph(
                id=f"agg:{digest}:{count}",
                source=source,
                target=target,
                points=(
                    (
                        source_node.x + source_node.width / 2,
                        source_node.y + source_node.height,
                    ),
                    (
                        target_node.x + target_node.width / 2,
                        target_node.y,
                    ),
                ),
            )
        )
    return SemanticScene(Scene(glyphs, edges), DetailLevel.OPERATOR, members_by_glyph)


def _fold_pure_feeders(
    scene: Scene,
    scoped: Scene,
    representatives: dict[str, tuple[str, str, str, frozenset[str]]],
) -> dict[str, tuple[str, str, str, frozenset[str]]]:
    """Fold attribute feeders (Constant and friends) into their consumer glyph.

    At architecture/block detail, a node with no producers that feeds exactly
    one glyph and is not itself in a selected group is presentation noise: a
    real model can carry hundreds of naked ``Constant`` nodes. Folding extends
    the consuming glyph's members, so selection and inspection still resolve
    to the folded node, while layer/operator detail keeps showing it
    individually.

    Whether a node is a pure source is judged on the full ``scene``, not the
    drill crop: a drilled block's head op merely had its producer cropped away
    and must keep its own glyph, while fan-out is judged on ``scoped`` because
    only in-scope consumers are visible.
    """
    incoming: dict[str, int] = {}
    for edge in scene.edges:
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
    fed_glyphs: dict[str, set[str]] = {}
    for edge in scoped.edges:
        fed_glyphs.setdefault(edge.source, set()).add(representatives[edge.target][0])
    folds: dict[str, str] = {}
    extras: dict[str, set[str]] = {}
    for node in scoped.nodes:
        if _base_op_type(node.type_label) not in _FEEDER_OP_TYPES:
            continue  # entry compute nodes are pure sources too; keep them
        if representatives[node.id][0] != node.id:
            continue  # already inside a selected group
        if incoming.get(node.id, 0):
            continue  # consumes another node's output: not a pure source
        targets = fed_glyphs.get(node.id)
        if targets is None or len(targets) != 1:
            continue  # feeds nothing, or fans out to several glyphs
        target = next(iter(targets))
        if target == node.id:
            continue  # defensive: a self-loop must not fold a node into itself
        folds[node.id] = target
        extras.setdefault(target, set()).add(node.id)
    if not folds:
        return representatives
    # A fold target always has an incoming edge (from the folded source), so
    # it is never itself folded and its metadata below is stable.
    metadata: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    for entry in representatives.values():
        metadata.setdefault(entry[0], entry)
    folded: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    for node_id, entry in representatives.items():
        glyph_id = folds.get(node_id, entry[0])
        base = metadata[glyph_id]
        extra = extras.get(glyph_id)
        members = base[3] | frozenset(extra) if extra else base[3]
        folded[node_id] = (glyph_id, base[1], base[2], members)
    return folded


def _relayout(scene: Scene) -> Scene:
    """Give a collapsed scene its own compact layered geometry.

    Collapsed glyphs otherwise inherit the union of their members'
    operator-level bounds — enormous, heavily overlapping rectangles for any
    real model. Treating the glyphs as fixed-size nodes and reusing the
    deterministic layering from :func:`nneditor.analysis.layout.layered_order`
    over the aggregated inter-glyph edges yields clean, non-overlapping
    architecture/block scenes with edges attached to the new positions.
    """
    if scene.node_count == 0:
        return scene
    sizes: dict[str, tuple[float, float]] = {}
    for node in scene.nodes:
        if node.kind == "group":
            width = min(
                _GROUP_GLYPH_MAX_WIDTH,
                max(
                    _GROUP_GLYPH_WIDTH,
                    24.0 + _GROUP_LABEL_CHAR_WIDTH * len(node.label),
                ),
            )
            sizes[node.id] = (width, _GROUP_GLYPH_HEIGHT)
        else:
            sizes[node.id] = (node.width, node.height)
    pairs = sorted({(edge.source, edge.target) for edge in scene.edges})
    rows = layered_order(tuple(node.id for node in scene.nodes), pairs)
    row_widths = [
        sum(sizes[node_id][0] for node_id in row)
        + _RELAYOUT_COLUMN_GAP * max(len(row) - 1, 0)
        for row in rows
    ]
    widest = max(row_widths, default=0.0)
    placed: dict[str, tuple[float, float]] = {}
    y = 0.0
    for row, row_width in zip(rows, row_widths, strict=True):
        x = (widest - row_width) / 2.0
        for node_id in row:
            placed[node_id] = (x, y)
            x += sizes[node_id][0] + _RELAYOUT_COLUMN_GAP
        y += max(sizes[node_id][1] for node_id in row) + _RELAYOUT_ROW_GAP
    nodes = [
        NodeGlyph(
            id=node.id,
            x=placed[node.id][0],
            y=placed[node.id][1],
            width=sizes[node.id][0],
            height=sizes[node.id][1],
            kind=node.kind,
            label=node.label,
            type_label=node.type_label,
        )
        for node in scene.nodes
    ]
    positioned = {node.id: node for node in nodes}
    edges = [
        EdgeGlyph(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            points=(
                (
                    positioned[edge.source].x + positioned[edge.source].width / 2.0,
                    positioned[edge.source].y + positioned[edge.source].height,
                ),
                (
                    positioned[edge.target].x + positioned[edge.target].width / 2.0,
                    positioned[edge.target].y,
                ),
            ),
        )
        for edge in scene.edges
    ]
    return Scene(nodes, edges)


def _overview(semantic: SemanticScene, *, max_nodes: int) -> SemanticScene:
    """Bound architecture overviews even when detectors find little structure."""
    scene = semantic.scene
    if scene.node_count <= max_nodes:
        return semantic
    chunk_size = max(2, math.ceil(scene.node_count / max_nodes))
    ordered = scene.nodes
    representative: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start : start + chunk_size]
        underlying = frozenset(
            member for node in chunk for member in semantic.members_by_glyph[node.id]
        )
        digest = hashlib.blake2b(
            "\0".join(sorted(underlying)).encode(), digest_size=8
        ).hexdigest()
        glyph_id = f"grp:overview:{digest}"
        for node in chunk:
            representative[node.id] = (
                glyph_id,
                "Overview region",
                "Architecture region",
                underlying,
            )
    collapsed = _collapse(scene, representative)
    return SemanticScene(
        collapsed.scene,
        DetailLevel.ARCHITECTURE,
        collapsed.members_by_glyph,
    )


def semantic_scene(
    scene: Scene,
    hierarchy: Hierarchy,
    detail_level: DetailLevel,
    *,
    root_group: str | None = None,
    max_architecture_nodes: int = 1000,
) -> SemanticScene:
    """Build one architecture/block/layer/operator representation."""
    root = hierarchy.group(root_group) if root_group is not None else None
    if detail_level is DetailLevel.OPERATOR and root is None:
        # The unscoped operator representation is already the source scene.
        # Rebuilding it through the generic collapse path validates and
        # reallocates every glyph and aggregates every edge even though no
        # node is collapsed. Reuse the immutable scene and construct only the
        # identity mapping required for semantic selection.
        return SemanticScene(
            scene,
            detail_level,
            {node.id: frozenset((node.id,)) for node in scene.nodes},
        )
    allowed = (
        root.members if root is not None else frozenset(node.id for node in scene.nodes)
    )
    scoped = _filtered_scene(scene, allowed)
    selected = _selected_groups(hierarchy, detail_level, allowed, root)
    if detail_level is DetailLevel.ARCHITECTURE:
        selected = _drill_into_dominant(hierarchy, selected, allowed)
    by_node: dict[str, Group] = {}
    for group in selected:
        for node_id in group.members:
            by_node[node_id] = group
    representatives: dict[str, tuple[str, str, str, frozenset[str]]] = {}
    for node in scoped.nodes:
        selected_group = by_node.get(node.id)
        if selected_group is None:
            representatives[node.id] = (
                node.id,
                node.label,
                node.display_type,
                frozenset((node.id,)),
            )
        else:
            representatives[node.id] = (
                selected_group.id,
                selected_group.label,
                f"{selected_group.kind.value.replace('-', ' ').title()} block",
                selected_group.members,
            )
    compact = detail_level in (DetailLevel.ARCHITECTURE, DetailLevel.BLOCK)
    if compact:
        representatives = _fold_pure_feeders(scene, scoped, representatives)
    collapsed = _collapse(scoped, representatives)
    semantic = SemanticScene(
        collapsed.scene,
        detail_level,
        collapsed.members_by_glyph,
    )
    if detail_level is DetailLevel.ARCHITECTURE:
        semantic = _overview(semantic, max_nodes=max_architecture_nodes)
    if compact:
        semantic = SemanticScene(
            _relayout(semantic.scene),
            detail_level,
            semantic.members_by_glyph,
        )
    return semantic
