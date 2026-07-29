"""The minimal renderer contract every candidate must satisfy (task P0.5).

The application talks to a renderer only through :class:`GraphRenderer`, so a
Flet Canvas surface, a custom Flutter control, or a hosted web renderer are
interchangeable. The contract is deliberately small: it covers exactly what the
Phase 0 benchmark exercises — scene replacement, incremental patches, viewport
(pan/zoom) updates, selection, and hit testing — and nothing speculative.

Coordinates crossing the contract are always *world* coordinates; only the
renderer knows about pixels, via the viewport's scale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nneditor.rendering.scene import Scene, ScenePatch, Viewport

__all__ = [
    "GraphRenderer",
    "InteractiveGraphRenderer",
    "RenderStats",
    "RendererFactory",
    "SelectionCallback",
]

SelectionCallback = Callable[[frozenset[str]], None]
"""Invoked with the new selected glyph ids after a user pick."""


@dataclass(frozen=True, slots=True)
class RenderStats:
    """What the renderer last did, so benchmarks and tests can observe it.

    ``build_ms`` is the Python-side cost of producing the drawable output for
    the current viewport. It is a lower bound on frame cost: whatever the UI
    toolkit adds comes on top, so if this alone exceeds the frame budget the
    approach is disqualified without needing a display.
    """

    visible_nodes: int
    visible_edges: int
    shape_count: int
    build_ms: float
    dropped_nodes: int = 0
    """Nodes in the viewport the shape budget refused; non-zero means the
    picture is deliberately incomplete and the shell must say so."""
    shifted: bool = False
    """True when this frame only shifted existing shapes in place (a pan).

    On such frames ``build_ms`` is the shift cost and the other counters still
    describe the last fully rebuilt region, so consumers comparing regimes
    must not read them as fresh cull results."""


@runtime_checkable
class GraphRenderer(Protocol):
    """A drawing surface for one scene at one viewport."""

    def replace_scene(self, scene: Scene, viewport: Viewport) -> None:
        """Show a different scene, dropping all prior state including selection."""

    def apply_patch(self, patch: ScenePatch) -> None:
        """Apply an incremental change to the current scene."""

    def set_viewport(self, viewport: Viewport) -> None:
        """Pan or zoom; the renderer re-culls and redraws as needed."""

    def set_selection(self, ids: frozenset[str]) -> None:
        """Highlight node or edge glyphs, replacing the previous selection."""

    def hit_test(self, world_x: float, world_y: float) -> str | None:
        """The topmost node or nearest connection at a world position."""

    @property
    def scene(self) -> Scene | None:
        """The scene currently shown, if any."""

    @property
    def viewport(self) -> Viewport | None:
        """The viewport last applied, if any."""

    @property
    def stats(self) -> RenderStats:
        """Metrics for the most recent rebuild."""


@runtime_checkable
class InteractiveGraphRenderer(GraphRenderer, Protocol):
    """Renderer features required by the interactive application shell.

    ``control`` is deliberately typed as ``object``: the renderer package does
    not make the core scene contract depend on Flet, while a Flet shell can
    validate and retain the concrete host control at its composition root.
    """

    @property
    def control(self) -> object:
        """Toolkit host control embedded by the shell."""

    @property
    def selection(self) -> frozenset[str]:
        """Current logical glyph selection."""

    def set_additive_selection(self, enabled: bool) -> None:
        """Enable or disable explicit additive selection."""

    def viewport_focused_on(
        self,
        node_id: str,
        *,
        width: float,
        height: float,
        scale: float | None = None,
    ) -> Viewport:
        """Return a viewport centered on one glyph."""


RendererFactory = Callable[[SelectionCallback | None], InteractiveGraphRenderer]
