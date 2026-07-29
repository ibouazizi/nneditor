"""Graph-first presentation tests for activation tracing."""

from __future__ import annotations

from pathlib import Path

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from nneditor.ui.trace_graph import build_trace_graph
from tests.fixtures.onnx_models import build_embedded_model


def test_trace_graph_adds_selectable_inputs_outputs_and_value_connections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    build_embedded_model(path, elements=4)
    with ApplicationService() as service:
        session = service.open_model(path)
        graph = session.document.main_graph
        layout = session.scene(
            session.document.entry_graph,
            detail_level=DetailLevel.OPERATOR,
        )
        presentation = build_trace_graph(
            layout.scene,
            graph,
            layout.members_by_glyph,
        )

    input_id = next(
        value_id for value_id in graph.inputs if value_id not in graph.initializers
    )
    output_id = graph.outputs[0]
    input_glyph = presentation.input_glyphs[input_id]
    output_glyph = presentation.output_glyphs[output_id]
    assert presentation.scene.node(input_glyph).kind == "graph-input"
    assert presentation.scene.node(output_glyph).kind == "graph-output"
    assert presentation.values_by_glyph[input_glyph] == (input_id,)
    assert presentation.values_by_glyph[output_glyph] == (output_id,)
    assert any(
        values == (output_id,)
        for edge_id, values in presentation.values_by_glyph.items()
        if presentation.scene.has_edge(edge_id)
    )
