"""Renderer-facing graph slices with hierarchy-aware cached layout.

A *slice* is what the renderer receives: a laid-out scene for one graph, or a
derived view of one (a neighborhood, a search result). Layout dominates the
cost, so results are cached under a bounded entry count, keyed by everything
that could change the geometry: the artifact's content hash, the session's
editing revision (graph edits keep the source hash but change the graph), the
graph id, and the layout settings. Reopening the same artifact therefore
reuses layouts, and two documents with the same hash *and revision* share them
safely because layout is deterministic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from nneditor.analysis.hierarchy import Hierarchy
from nneditor.analysis.layout import LayoutResult, LayoutSettings, layout_graph
from nneditor.analysis.lod import DetailLevel, SemanticScene, semantic_scene
from nneditor.application.navigation import NavigationModel, SearchHit
from nneditor.ir.core import Document
from nneditor.rendering.scene import Bounds, Scene
from nneditor.storage.cache import BudgetedCache, CacheSnapshot

__all__ = ["GraphSlice", "GraphSlicer", "SliceKey"]


@dataclass(frozen=True, slots=True)
class SliceKey:
    """Everything a cached layout depends on."""

    content_hash: str
    graph_id: str
    settings: LayoutSettings
    hierarchy_revision: str
    detail_level: DetailLevel
    root_group: str | None
    revision: str | None = None
    """The editing revision the document reflects; ``None`` at base."""


@dataclass(frozen=True, slots=True)
class GraphSlice:
    """One semantic representation returned to the renderer."""

    scene: Scene
    layers: tuple[tuple[str, ...], ...]
    cyclic_nodes: tuple[str, ...]
    detail_level: DetailLevel
    hierarchy_revision: str
    root_group: str | None
    members_by_glyph: dict[str, frozenset[str]]

    def representative_for(self, node_id: str) -> str | None:
        for glyph_id, members in self.members_by_glyph.items():
            if node_id in members:
                return glyph_id
        return None


class GraphSlicer:
    """Produces scenes and scene-derived queries for one or more documents."""

    __slots__ = ("_base_cache", "_cache", "_capacity", "_lock", "_navigation")

    def __init__(self, *, cache_capacity: int = 32) -> None:
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be positive")
        self._capacity = cache_capacity
        self._cache: BudgetedCache[SliceKey, GraphSlice] = BudgetedCache(
            "semantic-slices", cache_capacity
        )
        self._base_cache: BudgetedCache[
            tuple[str, str | None, str, LayoutSettings], LayoutResult
        ] = BudgetedCache("base-layouts", cache_capacity)
        self._lock = threading.RLock()
        self._navigation = NavigationModel()

    def slice_graph(
        self,
        document: Document,
        graph_id: str | None = None,
        settings: LayoutSettings | None = None,
        *,
        revision_id: str | None = None,
        hierarchy: Hierarchy | None = None,
        detail_level: DetailLevel = DetailLevel.OPERATOR,
        root_group: str | None = None,
        viewport: Bounds | None = None,
    ) -> GraphSlice:
        """A semantic graph representation, optionally cropped to a viewport.

        ``revision_id`` is the editing revision the document reflects
        (``None`` at base). Graph edits keep ``document.source`` — and with it
        the content hash — so without the revision in the key, two sessions on
        one artifact at different revisions would serve each other the wrong
        scene.
        """
        resolved_graph = graph_id if graph_id is not None else document.entry_graph
        graph = document.graphs.get(resolved_graph)
        if graph is None:
            raise KeyError(resolved_graph)
        resolved_hierarchy = hierarchy or Hierarchy(resolved_graph)
        if resolved_hierarchy.graph_id != resolved_graph:
            raise ValueError("hierarchy belongs to a different graph")
        resolved_settings = settings or LayoutSettings()
        key = SliceKey(
            content_hash=document.source.content_hash,
            graph_id=resolved_graph,
            settings=resolved_settings,
            hierarchy_revision=resolved_hierarchy.revision,
            detail_level=detail_level,
            root_group=root_group,
            revision=revision_id,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                result = cached
            else:
                base_key = (
                    document.source.content_hash,
                    revision_id,
                    resolved_graph,
                    resolved_settings,
                )
                base = self._base_cache.get(base_key)
                if base is None:
                    base = layout_graph(graph, resolved_settings)
                    self._base_cache.put(base_key, base)
                semantic = semantic_scene(
                    base.scene,
                    resolved_hierarchy,
                    detail_level,
                    root_group=root_group,
                )
                result = self._to_slice(
                    base,
                    semantic,
                    resolved_hierarchy.revision,
                    root_group,
                )
                self._cache.put(key, result)
        return self._crop(result, viewport) if viewport is not None else result

    @property
    def cached_layouts(self) -> int:
        return len(self._cache)

    def cache_snapshots(self) -> tuple[CacheSnapshot, ...]:
        """Observable state of the slicer's caches."""
        return (self._cache.snapshot(), self._base_cache.snapshot())

    def invalidate(self, content_hash: str | None = None) -> None:
        """Drop cached layouts, optionally only one artifact's.

        Runs under the slicer lock so an in-flight :meth:`slice_graph` cannot
        re-populate the caches with pre-invalidation layout mid-drop.
        """
        with self._lock:
            if content_hash is None:
                self._cache.invalidate()
                self._base_cache.invalidate()
                return
            self._cache.invalidate(lambda key: key.content_hash == content_hash)
            self._base_cache.invalidate(lambda key: key[0] == content_hash)

    def search(self, document: Document, text: str) -> tuple[str, ...]:
        """Node ids whose label, source name, or op type contains ``text``.

        Case-insensitive, across every graph in the document, in document
        order, so results are stable for keyboard navigation.
        """
        needle = text.strip().lower()
        if not needle:
            return ()
        found: list[str] = []
        for graph in document.graphs.values():
            for node in graph.nodes:
                haystacks = (node.source_name or "", node.op_type, node.id)
                if any(needle in item.lower() for item in haystacks):
                    found.append(node.id)
        return tuple(found)

    def search_hits(self, document: Document, text: str) -> tuple[SearchHit, ...]:
        """Graph-aware results suitable for cross-graph jump navigation."""
        return self._navigation.search(document, text)

    def neighborhood(
        self,
        document: Document,
        graph_id: str,
        node_id: str,
        *,
        hops: int = 1,
    ) -> frozenset[str]:
        """Node ids within ``hops`` dataflow steps of ``node_id``.

        Used by the UI to focus a selection's surroundings without asking the
        renderer to walk the graph itself.
        """
        if hops < 0:
            raise ValueError("hops must be non-negative")
        graph = document.graphs[graph_id]
        graph.node(node_id)  # KeyError for unknown nodes
        frontier = {node_id}
        seen = {node_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for current in frontier:
                node = graph.node(current)
                for value_id in node.inputs:
                    producer = graph.producer(value_id)
                    if producer is not None and producer[0] not in seen:
                        next_frontier.add(producer[0])
                for value_id in node.outputs:
                    for consumer, _port in graph.consumers(value_id):
                        if consumer not in seen:
                            next_frontier.add(consumer)
            seen.update(next_frontier)
            frontier = next_frontier
        return frozenset(seen)

    @staticmethod
    def _to_slice(
        base: LayoutResult,
        semantic: SemanticScene,
        hierarchy_revision: str,
        root_group: str | None,
    ) -> GraphSlice:
        rows: dict[float, list[str]] = {}
        for node in semantic.scene.nodes:
            rows.setdefault(node.y, []).append(node.id)
        layers = tuple(
            tuple(sorted(rows[y], key=lambda item: semantic.scene.node(item).x))
            for y in sorted(rows)
        )
        cyclic = tuple(
            dict.fromkeys(
                item
                for node_id in base.cyclic_nodes
                if (item := semantic.representative_for(node_id)) is not None
            )
        )
        return GraphSlice(
            scene=semantic.scene,
            layers=layers,
            cyclic_nodes=cyclic,
            detail_level=semantic.detail_level,
            hierarchy_revision=hierarchy_revision,
            root_group=root_group,
            members_by_glyph=semantic.members_by_glyph,
        )

    @staticmethod
    def _crop(result: GraphSlice, viewport: Bounds) -> GraphSlice:
        visible_nodes = [
            node for node in result.scene.nodes if node.bounds.intersects(viewport)
        ]
        ids = {node.id for node in visible_nodes}
        visible_edges = [
            edge
            for edge in result.scene.edges
            if edge.source in ids and edge.target in ids
        ]
        scene = Scene(visible_nodes, visible_edges)
        return GraphSlice(
            scene=scene,
            layers=tuple(
                tuple(item for item in row if item in ids)
                for row in result.layers
                if any(item in ids for item in row)
            ),
            cyclic_nodes=tuple(item for item in result.cyclic_nodes if item in ids),
            detail_level=result.detail_level,
            hierarchy_revision=result.hierarchy_revision,
            root_group=result.root_group,
            members_by_glyph={
                key: value
                for key, value in result.members_by_glyph.items()
                if key in ids
            },
        )
