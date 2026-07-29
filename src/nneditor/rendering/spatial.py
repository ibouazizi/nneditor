"""Uniform-grid spatial index for viewport culling and hit testing.

Every renderer candidate needs the same two queries — "which glyphs intersect
this rectangle" and "what is under this point" — and both must stay fast at
10,000 nodes, so the index lives outside any concrete renderer. A uniform grid
is chosen over a tree because scenes come from layered graph layouts whose
glyphs are near-uniformly distributed and similarly sized, which is the grid's
best case, and because incremental insert/remove (for patches) is trivial.
"""

from __future__ import annotations

from collections.abc import Mapping

from nneditor.rendering.scene import Bounds, Scene

__all__ = ["SpatialIndex", "scene_indexes"]


class SpatialIndex:
    """Maps item ids to bounds and buckets them into fixed-size grid cells.

    Later insertions are treated as being on top for :meth:`hit`, matching the
    painter's order in which renderers draw them.
    """

    __slots__ = ("_boxes", "_cells", "cell_size")

    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size
        self._boxes: dict[str, Bounds] = {}
        self._cells: dict[tuple[int, int], list[str]] = {}

    def __len__(self) -> int:
        return len(self._boxes)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._boxes

    def _cell_span(self, bounds: Bounds) -> tuple[int, int, int, int]:
        size = self.cell_size
        return (
            int(bounds.min_x // size),
            int(bounds.min_y // size),
            int(bounds.max_x // size),
            int(bounds.max_y // size),
        )

    def insert(self, item_id: str, bounds: Bounds) -> None:
        """Add ``item_id``; inserting an existing id is an error, not an update."""
        if item_id in self._boxes:
            raise ValueError(f"item {item_id!r} is already indexed")
        self._boxes[item_id] = bounds
        first_col, first_row, last_col, last_row = self._cell_span(bounds)
        for col in range(first_col, last_col + 1):
            for row in range(first_row, last_row + 1):
                self._cells.setdefault((col, row), []).append(item_id)

    def remove(self, item_id: str) -> None:
        """Drop ``item_id``; removing an unknown id is an error."""
        bounds = self._boxes.pop(item_id, None)
        if bounds is None:
            raise KeyError(item_id)
        first_col, first_row, last_col, last_row = self._cell_span(bounds)
        for col in range(first_col, last_col + 1):
            for row in range(first_row, last_row + 1):
                cell = self._cells[(col, row)]
                cell.remove(item_id)
                if not cell:
                    del self._cells[(col, row)]

    def replace(self, item_id: str, bounds: Bounds) -> None:
        """Update an existing item's bounds in place."""
        self.remove(item_id)
        self.insert(item_id, bounds)

    def query(self, bounds: Bounds) -> set[str]:
        """Ids of all items whose bounds intersect ``bounds``.

        Two strategies, chosen by whichever is cheaper for this query:

        *Walk the span* — visit each grid cell the rectangle covers. Best
        when the query is small relative to the scene, which is the common
        viewport case.

        *Walk the content* — visit the populated cells instead. A zoomed-out
        viewport over a tall, narrow graph can span millions of mostly-empty
        cells, and walking the span there costs far more than the index even
        holds (this was a 15-second frame on a 14k-node model).
        """
        found: set[str] = set()
        boxes = self._boxes
        first_col, first_row, last_col, last_row = self._cell_span(bounds)
        span_cells = (last_col - first_col + 1) * (last_row - first_row + 1)
        if span_cells > len(self._cells):
            for (col, row), items in self._cells.items():
                if not (first_col <= col <= last_col and first_row <= row <= last_row):
                    continue
                for item_id in items:
                    if item_id not in found and boxes[item_id].intersects(bounds):
                        found.add(item_id)
            return found
        for col in range(first_col, last_col + 1):
            for row in range(first_row, last_row + 1):
                for item_id in self._cells.get((col, row), ()):
                    if item_id not in found and boxes[item_id].intersects(bounds):
                        found.add(item_id)
        return found

    def hit(
        self, x: float, y: float, order: Mapping[str, int] | None = None
    ) -> str | None:
        """The topmost item containing the point, or ``None``.

        Without ``order``, "topmost" means most recently inserted. That breaks
        down after :meth:`replace`, which re-inserts and so silently moves an
        item to the top even though the renderer still paints it in its old
        slot. Renderers therefore pass their painter order, and the hit with
        the highest order wins — pixels and picks then always agree.
        """
        size = self.cell_size
        cell = self._cells.get((int(x // size), int(y // size)))
        if not cell:
            return None
        boxes = self._boxes
        if order is None:
            for item_id in reversed(cell):
                if boxes[item_id].contains_point(x, y):
                    return item_id
            return None
        best: str | None = None
        best_order = 0
        for item_id in cell:
            if not boxes[item_id].contains_point(x, y):
                continue
            item_order = order.get(item_id, -1)
            if best is None or item_order > best_order:
                best = item_id
                best_order = item_order
        return best


def _pick_cell_size(scene: Scene) -> float:
    """A cell a few glyphs wide: big enough to keep cell counts sane, small
    enough that a viewport query skips most of the scene."""
    if scene.node_count == 0:
        return 1.0
    total = 0.0
    for node in scene.nodes:
        total += max(node.width, node.height)
    return 4.0 * (total / scene.node_count)


def scene_indexes(scene: Scene) -> tuple[SpatialIndex, SpatialIndex]:
    """Build ``(node_index, edge_index)`` for a scene.

    Nodes and edges are indexed separately so node hits can take precedence
    before a small-tolerance connection query, and because an edge's bounding
    box is long and thin, which would otherwise pollute node point queries.
    """
    cell_size = _pick_cell_size(scene)
    node_index = SpatialIndex(cell_size)
    edge_index = SpatialIndex(cell_size)
    for node in scene.nodes:
        node_index.insert(node.id, node.bounds)
    for edge in scene.edges:
        edge_index.insert(edge.id, edge.bounds)
    return node_index, edge_index
