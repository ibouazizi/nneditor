"""Renderer-agnostic scene model.

A *scene* is the fully laid-out, flattened set of glyphs a renderer is asked to
draw: nodes as axis-aligned rectangles and edges as polylines, all in world
coordinates. Scenes deliberately know nothing about the IR — graph slicing
(task P1.6) will produce scenes from IR documents, and every renderer candidate
consumes the same scene type, which is what makes the renderer replaceable.

Scenes are immutable. Interactive changes — collapsing a block, expanding a
group — arrive as a :class:`ScenePatch`, so a renderer can update incrementally
instead of rebuilding, and so the benchmark can measure exactly that difference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

__all__ = [
    "Bounds",
    "EdgeGlyph",
    "NodeGlyph",
    "PatchError",
    "Scene",
    "SceneError",
    "ScenePatch",
    "Viewport",
]


class SceneError(ValueError):
    """Raised when glyphs do not form a consistent scene."""


class PatchError(ValueError):
    """Raised when a patch does not apply cleanly to a scene."""


@dataclass(frozen=True, slots=True)
class Bounds:
    """An axis-aligned rectangle in world coordinates, y growing downward."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        # NaN fails every comparison, so "max < min" alone would wave NaN
        # bounds through and detonate deep inside the spatial index instead.
        if not all(
            math.isfinite(value)
            for value in (self.min_x, self.min_y, self.max_x, self.max_y)
        ):
            raise SceneError(f"non-finite bounds: {self}")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise SceneError(f"inverted bounds: {self}")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def intersects(self, other: Bounds) -> bool:
        """Whether the two rectangles share any area (touching counts)."""
        return (
            self.min_x <= other.max_x
            and other.min_x <= self.max_x
            and self.min_y <= other.max_y
            and other.min_y <= self.max_y
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies inside or on the rectangle."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def union(self, other: Bounds) -> Bounds:
        """The smallest rectangle covering both."""
        return Bounds(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )


_EMPTY_BOUNDS: Final = Bounds(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class NodeGlyph:
    """One drawable node: a rectangle with a kind that drives its styling.

    ``kind`` is an open vocabulary ("conv", "attention", "group", ...) rather
    than an enum so that adapters and pattern detection can evolve without
    touching the scene model; renderers map unknown kinds to a default style.
    """

    id: str
    x: float
    y: float
    width: float
    height: float
    kind: str
    label: str
    type_label: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise SceneError("node glyph id must be non-empty")
        if not all(
            math.isfinite(value) for value in (self.x, self.y, self.width, self.height)
        ):
            raise SceneError(
                f"node glyph {self.id!r} has non-finite geometry: "
                f"({self.x}, {self.y}) {self.width}x{self.height}"
            )
        if self.width <= 0 or self.height <= 0:
            raise SceneError(
                f"node glyph {self.id!r} must have positive size, got "
                f"{self.width}x{self.height}"
            )

    @property
    def bounds(self) -> Bounds:
        return Bounds(self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def display_type(self) -> str:
        """The semantic type shown beneath the glyph's name."""
        if self.type_label.strip():
            return self.type_label
        return self.kind.replace("-", " ").title()


@dataclass(frozen=True, slots=True)
class EdgeGlyph:
    """One drawable edge: a polyline from a source glyph to a target glyph."""

    id: str
    source: str
    target: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise SceneError("edge glyph id must be non-empty")
        if len(self.points) < 2:
            raise SceneError(f"edge glyph {self.id!r} needs at least two points")
        for x, y in self.points:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise SceneError(
                    f"edge glyph {self.id!r} has non-finite point ({x}, {y})"
                )

    @property
    def bounds(self) -> Bounds:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return Bounds(min(xs), min(ys), max(xs), max(ys))


class Scene:
    """An immutable, validated collection of node and edge glyphs.

    Validation is strict because a renderer receiving a dangling edge has no
    sensible recovery: every edge must reference nodes present in the scene and
    every id must be unique across both glyph kinds, so a single id space can
    drive selection and hit testing.
    """

    __slots__ = ("_bounds", "_edges", "_nodes")

    def __init__(
        self, nodes: Iterable[NodeGlyph], edges: Iterable[EdgeGlyph] = ()
    ) -> None:
        node_map: dict[str, NodeGlyph] = {}
        for node in nodes:
            if node.id in node_map:
                raise SceneError(f"duplicate node glyph id {node.id!r}")
            node_map[node.id] = node
        edge_map: dict[str, EdgeGlyph] = {}
        for edge in edges:
            if edge.id in edge_map or edge.id in node_map:
                raise SceneError(f"duplicate glyph id {edge.id!r}")
            for endpoint in (edge.source, edge.target):
                if endpoint not in node_map:
                    raise SceneError(
                        f"edge {edge.id!r} references missing node {endpoint!r}"
                    )
            edge_map[edge.id] = edge
        self._nodes = node_map
        self._edges = edge_map
        bounds: Bounds | None = None
        for node in node_map.values():
            bounds = node.bounds if bounds is None else bounds.union(node.bounds)
        self._bounds = bounds if bounds is not None else _EMPTY_BOUNDS

    @property
    def nodes(self) -> tuple[NodeGlyph, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[EdgeGlyph, ...]:
        return tuple(self._edges.values())

    @property
    def bounds(self) -> Bounds:
        """The union of all node bounds; edges may extend slightly past it."""
        return self._bounds

    def node(self, node_id: str) -> NodeGlyph:
        """Return the node glyph with ``node_id`` or raise :class:`KeyError`."""
        return self._nodes[node_id]

    def edge(self, edge_id: str) -> EdgeGlyph:
        """Return the edge glyph with ``edge_id`` or raise :class:`KeyError`."""
        return self._edges[edge_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def apply(self, patch: ScenePatch) -> Scene:
        """Return a new scene with ``patch`` applied.

        Removals of unknown ids are errors rather than no-ops: a patch is
        produced against a specific scene revision, so a miss means the caller
        is patching the wrong scene, which should fail loudly.
        """
        nodes = dict(self._nodes)
        edges = dict(self._edges)
        for node_id in patch.remove_nodes:
            if nodes.pop(node_id, None) is None:
                raise PatchError(f"patch removes unknown node {node_id!r}")
        for edge_id in patch.remove_edges:
            if edges.pop(edge_id, None) is None:
                raise PatchError(f"patch removes unknown edge {edge_id!r}")
        for node in patch.upsert_nodes:
            nodes[node.id] = node
        for edge in patch.upsert_edges:
            edges[edge.id] = edge
        try:
            return Scene(nodes.values(), edges.values())
        except SceneError as error:
            raise PatchError(f"patch produced an invalid scene: {error}") from error


@dataclass(frozen=True, slots=True)
class ScenePatch:
    """An incremental scene change: removals first, then upserts."""

    remove_nodes: frozenset[str] = frozenset()
    remove_edges: frozenset[str] = frozenset()
    upsert_nodes: tuple[NodeGlyph, ...] = ()
    upsert_edges: tuple[EdgeGlyph, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.remove_nodes
            or self.remove_edges
            or self.upsert_nodes
            or self.upsert_edges
        )

    def inverse_for(self, scene: Scene) -> ScenePatch:
        """The patch that undoes this one when applied after it.

        Computed against the scene the patch is *about to be applied to*, since
        that is where the overwritten glyphs still exist.
        """
        restore_nodes: list[NodeGlyph] = []
        restore_edges: list[EdgeGlyph] = []
        drop_nodes: set[str] = set()
        drop_edges: set[str] = set()
        for node_id in self.remove_nodes:
            restore_nodes.append(scene.node(node_id))
        for edge_id in self.remove_edges:
            restore_edges.append(scene.edge(edge_id))
        for node in self.upsert_nodes:
            if scene.has_node(node.id):
                restore_nodes.append(scene.node(node.id))
            else:
                drop_nodes.add(node.id)
        for edge in self.upsert_edges:
            if scene.has_edge(edge.id):
                restore_edges.append(scene.edge(edge.id))
            else:
                drop_edges.add(edge.id)
        return ScenePatch(
            remove_nodes=frozenset(drop_nodes),
            remove_edges=frozenset(drop_edges),
            upsert_nodes=tuple(restore_nodes),
            upsert_edges=tuple(restore_edges),
        )


@dataclass(frozen=True, slots=True)
class Viewport:
    """The visible world-coordinate window and its zoom.

    ``scale`` is the pixels-per-world-unit ratio. It drives level-of-detail
    decisions (labels vanish when glyphs get small on screen) and converts
    between screen and world coordinates for hit testing.
    """

    x: float
    y: float
    width: float
    height: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SceneError(
                f"viewport must have positive size, got {self.width}x{self.height}"
            )
        if self.scale <= 0:
            raise SceneError(f"viewport scale must be positive, got {self.scale}")

    @property
    def bounds(self) -> Bounds:
        return Bounds(self.x, self.y, self.x + self.width, self.y + self.height)

    def expanded(self, factor: float) -> Bounds:
        """Bounds grown by ``factor`` of the size on every side.

        Renderers cull against an expanded viewport so that small pans reveal
        already-drawn glyphs instead of a blank margin.
        """
        margin_x = self.width * factor
        margin_y = self.height * factor
        return Bounds(
            self.x - margin_x,
            self.y - margin_y,
            self.x + self.width + margin_x,
            self.y + self.height + margin_y,
        )

    def to_world(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Map screen pixels (origin at the viewport corner) to world units."""
        return (self.x + screen_x / self.scale, self.y + screen_y / self.scale)

    def to_screen(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Map world units to screen pixels relative to the viewport corner."""
        return ((world_x - self.x) * self.scale, (world_y - self.y) * self.scale)
