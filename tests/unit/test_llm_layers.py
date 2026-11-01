"""The Layers level opens the weights inside a synthesized LLM block."""

from __future__ import annotations

from pathlib import Path

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from tests.unit.test_gguf import _tiny_llm


def test_layers_level_lists_the_weights_inside_a_block(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "layers.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        hierarchy = session.graph_hierarchy(graph.id)
        block = next(group for group in hierarchy.groups if group.label == "block-000")

        blocks = session.scene(detail_level=DetailLevel.BLOCK, root_group=block.id)
        layers = session.scene(detail_level=DetailLevel.LAYER, root_group=block.id)

    assert [node.label for node in blocks.scene.nodes] == [
        "Attention",
        "Feed-forward",
    ]
    assert [node.label for node in layers.scene.nodes] == [
        "000 · Query",
        "000 · Key",
        "Feed-forward",
    ]
    tops = [node.y for node in layers.scene.nodes]
    assert tops[0] < tops[1] < tops[2]
    query, key = layers.scene.nodes[:2]
    assert layers.members_by_glyph[query.id] == layers.members_by_glyph[key.id]
    assert len(layers.members_by_glyph[query.id]) == 1
    assert any(
        edge.source == query.id and edge.target == key.id for edge in layers.scene.edges
    )
