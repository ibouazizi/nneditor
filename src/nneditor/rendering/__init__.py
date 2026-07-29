"""Graph rendering contracts and adapters (task P0.5).

Architectural rule 5 requires graph rendering to sit behind a replaceable
interface. This package defines the renderer-agnostic scene model and the
minimal :class:`~nneditor.rendering.contract.GraphRenderer` contract, plus the
candidate adapters being benchmarked in Phase 0. Nothing outside this package
may import a concrete renderer directly.
"""

from nneditor.rendering.contract import (
    InteractiveGraphRenderer as InteractiveGraphRenderer,
)
from nneditor.rendering.contract import (
    RendererFactory as RendererFactory,
)
from nneditor.rendering.contract import (
    SelectionCallback as SelectionCallback,
)

__all__ = [
    "InteractiveGraphRenderer",
    "RendererFactory",
    "SelectionCallback",
    "create_flet_renderer",
]


def create_flet_renderer(
    on_select: SelectionCallback | None = None,
) -> InteractiveGraphRenderer:
    """Construct the selected production renderer behind its protocol."""
    from nneditor.rendering.flet_canvas_managed import ManagedCanvasRenderer

    return ManagedCanvasRenderer(on_select=on_select)
