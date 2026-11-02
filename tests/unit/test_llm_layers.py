"""The Layers level opens the weights inside a synthesized LLM block."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from nneditor.ir.capabilities import Availability, Capability
from tests.unit.test_gguf import _tiny_llm
from tests.unit.test_shell import make_shell


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
        "Block 0 · Query",
        "Block 0 · Key",
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


def test_layers_stay_inside_the_selected_block(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "navigate.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(path))
        assert DetailLevel.OPERATOR.value not in {
            segment.value for segment in shell.detail_segment.segments
        }
        scene = shell.renderer.scene
        assert scene is not None
        block = next(node for node in scene.nodes if "block-000" in node.label)
        shell.renderer.set_selection(frozenset({block.id}))
        shell.detail_segment.selected = [DetailLevel.LAYER.value]
        shell._on_detail_segment_changed(cast(Any, None))
        shown = shell.renderer.scene
        assert shown is not None
        assert "Block 0 · Query" in [node.label for node in shown.nodes]
        assert shell.place_label.value == "Block 0 · Layers"
        shell._on_back_to_parent(cast(Any, None))
        current = shell.current_slice
        assert current is not None
        assert [node.label for node in current.scene.nodes] == [
            "Attention",
            "Feed-forward",
        ]
        assert shell.current_detail is DetailLevel.BLOCK
        assert shell.session is not None
        assert any(
            hit.layer_stem == "attn_q" for hit in shell.session.search_hits("query")
        )


def test_input_button_shows_when_tracing_is_unavailable(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "input.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        assert (
            session.capability(Capability.TRACING).availability
            is Availability.UNAVAILABLE
        )
        assert any(
            isinstance(control.data, str)
            and control.data.startswith("trace-input-action:")
            for control in shell.trace.graph_actions.controls
        )


def test_double_click_without_an_interior_keeps_the_view(tmp_path: Path) -> None:
    path = _tiny_llm(tmp_path / "drill.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(path))
        scene = shell.renderer.scene
        assert scene is not None
        embed = next(node for node in scene.nodes if node.label == "Token embeddings")
        placed = [(node.id, node.x, node.y) for node in scene.nodes]
        shell.renderer.set_selection(frozenset({embed.id}))
        shell._drill_into_selection()
        shown = shell.renderer.scene
        assert shown is not None
        assert shell.current_detail is DetailLevel.ARCHITECTURE
        assert [(node.id, node.x, node.y) for node in shown.nodes] == placed
        block = next(node for node in shown.nodes if "block-000" in node.label)
        shell.renderer.set_selection(frozenset({block.id}))
        shell._drill_into_selection()
        assert cast(DetailLevel, shell.current_detail) is DetailLevel.BLOCK
