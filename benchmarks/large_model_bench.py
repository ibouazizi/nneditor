"""End-to-end scale benchmark on a genuinely large ONNX model.

The Phase 0 renderer benchmark measured *synthetic scenes*; this measures the
real product path a user experiences: import a large artifact, build the
hierarchy, lay out and slice the graph at each detail level, search, and
inspect. It writes JSON to ``benchmarks/results/large_model.json``.

Run: ``uv run python benchmarks/large_model_bench.py --nodes 20000``
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper

from nneditor.analysis.lod import DetailLevel
from nneditor.application.session import ApplicationService
from nneditor.rendering.flet_canvas_managed import ManagedCanvasRenderer
from nneditor.rendering.scene import Viewport

VIEW = (1600.0, 1000.0)


def build_transformer_like(path: Path, *, blocks: int, width: int) -> dict[str, int]:
    """A repeated-block model shaped like a real transformer stack."""
    nodes = []
    initializers = []
    value_info = []
    current = "input"
    rng = np.random.default_rng(0)

    for block in range(blocks):
        for name, op in (
            ("qkv", "MatMul"),
            ("attn_out", "MatMul"),
            ("norm1", "Add"),
            ("ffn_in", "MatMul"),
            ("gelu", "Relu"),
            ("ffn_out", "MatMul"),
            ("norm2", "Add"),
        ):
            output = f"b{block}_{name}"
            if op == "MatMul":
                weight_name = f"b{block}_{name}_w"
                initializers.append(
                    helper.make_tensor(
                        weight_name,
                        TensorProto.FLOAT,
                        [width, width],
                        rng.standard_normal(width * width, dtype=np.float32).tobytes(),
                        raw=True,
                    )
                )
                nodes.append(
                    helper.make_node(op, [current, weight_name], [output], name=output)
                )
            elif op == "Add":
                bias_name = f"b{block}_{name}_b"
                initializers.append(
                    helper.make_tensor(
                        bias_name,
                        TensorProto.FLOAT,
                        [width],
                        rng.standard_normal(width, dtype=np.float32).tobytes(),
                        raw=True,
                    )
                )
                nodes.append(
                    helper.make_node(op, [current, bias_name], [output], name=output)
                )
            else:
                nodes.append(helper.make_node(op, [current], [output], name=output))
            value_info.append(
                helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, width])
            )
            current = output

    graph = helper.make_graph(
        nodes,
        "large",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info(current, TensorProto.FLOAT, [1, width])],
        initializer=initializers,
        value_info=value_info[:-1],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(model, path)
    return {
        "nodes": len(nodes),
        "initializers": len(initializers),
        "bytes": path.stat().st_size,
    }


def timed(label: str, fn: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=14_000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "large_model.json",
    )
    args = parser.parse_args(argv)

    blocks = max(1, args.nodes // 7)
    workdir = Path(__file__).parent / "results" / "_large"
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = workdir / f"large_{blocks}.onnx"

    report: dict[str, Any] = {
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        }
    }

    if not model_path.exists():
        shape = build_transformer_like(model_path, blocks=blocks, width=args.width)
    else:
        shape = {
            "nodes": blocks * 7,
            "initializers": blocks * 6,
            "bytes": model_path.stat().st_size,
        }
    report["model"] = shape
    print(f"model: {shape['nodes']} nodes, {shape['bytes'] / 1e6:.1f} MB")

    gc.collect()
    tracemalloc.start()
    with ApplicationService() as service:
        session, open_ms = timed("open", lambda: service.open_model(model_path))
        report["open_ms"] = round(open_ms, 1)
        _current, peak = tracemalloc.get_traced_memory()
        report["import_peak_mib"] = round(peak / (1024 * 1024), 1)
        print(f"open: {open_ms:.0f} ms, peak {peak / 1e6:.1f} MB")

        hierarchy, hierarchy_ms = timed("hierarchy", lambda: session.graph_hierarchy())
        report["hierarchy_ms"] = round(hierarchy_ms, 1)
        report["groups"] = len(hierarchy.groups)
        print(f"hierarchy: {hierarchy_ms:.0f} ms, {len(hierarchy.groups)} groups")

        for level in DetailLevel:
            slice_result, slice_ms = timed(
                f"slice-{level.value}",
                lambda level=level: session.scene(detail_level=level),
            )
            report[f"slice_{level.value}_ms"] = round(slice_ms, 1)
            report[f"slice_{level.value}_nodes"] = slice_result.scene.node_count
            print(
                f"slice {level.value}: {slice_ms:.0f} ms, "
                f"{slice_result.scene.node_count} glyphs"
            )

        operator_slice = session.scene(detail_level=DetailLevel.OPERATOR)
        renderer = ManagedCanvasRenderer()
        _, replace_ms = timed(
            "replace_scene",
            lambda: renderer.replace_scene(
                operator_slice.scene, Viewport(0, 0, *VIEW, scale=1.0)
            ),
        )
        report["renderer_first_frame_ms"] = round(replace_ms, 1)
        report["renderer_first_frame_shapes"] = renderer.stats.shape_count

        pans = []
        for step in range(60):
            _, pan_ms = timed(
                "pan",
                lambda step=step: renderer.set_viewport(
                    Viewport(step * 12.0, step * 8.0, *VIEW, scale=1.0)
                ),
            )
            pans.append(pan_ms)
        report["pan_p50_ms"] = round(statistics.median(pans), 2)
        report["pan_p95_ms"] = round(sorted(pans)[int(len(pans) * 0.95)], 2)
        report["pan_rebuilds"] = renderer.rebuild_count
        print(
            f"pan: p50 {report['pan_p50_ms']} ms, p95 {report['pan_p95_ms']} ms, "
            f"{renderer.rebuild_count} rebuilds"
        )

        # Zoomed all the way out — the worst case for the LOD cap.
        bounds = operator_slice.scene.bounds
        fit = min(VIEW[0] / max(bounds.width, 1), VIEW[1] / max(bounds.height, 1))
        _, overview_ms = timed(
            "overview",
            lambda: renderer.set_viewport(
                Viewport(
                    bounds.min_x,
                    bounds.min_y,
                    VIEW[0] / fit,
                    VIEW[1] / fit,
                    scale=fit,
                )
            ),
        )
        report["overview_ms"] = round(overview_ms, 1)
        report["overview_shapes"] = renderer.stats.shape_count
        report["overview_visible_nodes"] = renderer.stats.visible_nodes
        print(
            f"overview: {overview_ms:.0f} ms, {renderer.stats.shape_count} shapes "
            f"for {renderer.stats.visible_nodes} visible nodes"
        )

        _, search_ms = timed("search", lambda: session.search("ffn"))
        report["search_ms"] = round(search_ms, 1)

        weight_id = session.document.main_graph.initializers[0]
        _, stats_ms = timed("statistics", lambda: session.compute_statistics(weight_id))
        report["statistics_one_tensor_ms"] = round(stats_ms, 1)
        print(f"search: {search_ms:.0f} ms, statistics: {stats_ms:.0f} ms")

    tracemalloc.stop()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
