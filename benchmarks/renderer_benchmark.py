"""Headless renderer benchmark harness (task P0.5).

Measures the Python-side cost of every renderer-contract operation on the
synthetic 1,000- and 10,000-node graphs, without needing a display. The numbers
are *lower bounds* on frame cost: the UI toolkit's serialization, transport,
and paint come on top. That direction of error is safe for a go/no-go decision
— an approach that fails here fails everywhere — while pass results must be
confirmed interactively with ``benchmarks/renderer_app.py``.

Run with ``uv run python benchmarks/renderer_benchmark.py``; results are
printed as a markdown table and written to ``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import flet.canvas as cv

from nneditor.rendering.flet_canvas import FletCanvasRenderer
from nneditor.rendering.scene import Viewport
from nneditor.rendering.spatial import scene_indexes
from nneditor.rendering.synthetic import (
    SyntheticGraph,
    collapse_layers,
    synthetic_graph,
)

VIEW_WIDTH = 1600.0
VIEW_HEIGHT = 1000.0
PAN_STEPS = 120
HIT_SAMPLES = 1000
COLLAPSE_BLOCK_LAYERS = 10


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _timed(fn: Any) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def _estimate_payload_bytes(shapes: list[cv.Shape]) -> int:
    """A JSON-size proxy for the shape list crossing to the UI toolkit.

    Flet transports msgpack, which is smaller than JSON, so this over-estimates
    slightly; it exists to compare scenarios and approaches, not to model the
    wire exactly.
    """
    encoded: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, cv.Line):
            encoded.append(
                {"t": "l", "a": shape.x1, "b": shape.y1, "c": shape.x2, "d": shape.y2}
            )
        elif isinstance(shape, cv.Rect):
            encoded.append(
                {
                    "t": "r",
                    "x": shape.x,
                    "y": shape.y,
                    "w": shape.width,
                    "h": shape.height,
                    "p": "#000000",
                }
            )
        elif isinstance(shape, cv.Text):
            encoded.append({"t": "t", "x": shape.x, "y": shape.y, "v": shape.value})
    return len(json.dumps(encoded).encode())


def _pan_sweep(
    renderer: FletCanvasRenderer, graph: SyntheticGraph, scale: float
) -> tuple[list[float], list[int]]:
    """Diagonal pan across the whole scene at ``scale``; per-frame costs."""
    bounds = graph.scene.bounds
    samples: list[float] = []
    visible: list[int] = []
    for step in range(PAN_STEPS):
        fraction = step / (PAN_STEPS - 1)
        viewport = Viewport(
            x=bounds.min_x + fraction * max(0.0, bounds.width - VIEW_WIDTH / scale),
            y=bounds.min_y + fraction * max(0.0, bounds.height - VIEW_HEIGHT / scale),
            width=VIEW_WIDTH / scale,
            height=VIEW_HEIGHT / scale,
            scale=scale,
        )
        samples.append(_timed(lambda vp=viewport: renderer.set_viewport(vp)))
        visible.append(renderer.stats.visible_nodes)
    return samples, visible


def bench_scenario(node_count: int, layer_width: int) -> dict[str, Any]:
    result: dict[str, Any] = {"node_count": node_count, "layer_width": layer_width}

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    started = time.perf_counter()
    graph = synthetic_graph(node_count, layer_width=layer_width, seed=42)
    result["scene_build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    started = time.perf_counter()
    node_index, edge_index = scene_indexes(graph.scene)
    result["index_build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    result["scene_and_index_kib"] = int(
        sum(stat.size_diff for stat in after.compare_to(before, "filename")) / 1024
    )
    result["edge_count"] = graph.scene.edge_count
    del node_index, edge_index

    renderer = FletCanvasRenderer()
    scene_bounds = graph.scene.bounds
    initial = Viewport(0.0, 0.0, VIEW_WIDTH, VIEW_HEIGHT, scale=1.0)
    result["replace_scene_ms"] = round(
        _timed(lambda: renderer.replace_scene(graph.scene, initial)), 3
    )

    # Working set: pan across the scene at 1:1 zoom, a few hundred visible.
    samples, visible = _pan_sweep(renderer, graph, scale=1.0)
    result["working_set"] = {
        **_percentiles(samples),
        "visible_nodes_mean": int(statistics.mean(visible)),
        "fps_bound_at_p95": round(1000.0 / max(_percentiles(samples)["p95_ms"], 1e-6)),
        "payload_kib": int(
            _estimate_payload_bytes(list(renderer._canvas.shapes or [])) / 1024
        ),
    }

    # Whole graph visible: the zoomed-out worst case the plan calls out.
    fit_scale = min(
        VIEW_WIDTH / max(scene_bounds.width, 1.0),
        VIEW_HEIGHT / max(scene_bounds.height, 1.0),
    )
    overview = Viewport(
        x=scene_bounds.min_x,
        y=scene_bounds.min_y,
        width=VIEW_WIDTH / fit_scale,
        height=VIEW_HEIGHT / fit_scale,
        scale=fit_scale,
    )
    overview_samples = [
        _timed(lambda: renderer.set_viewport(overview)) for _ in range(10)
    ]
    result["all_visible"] = {
        **_percentiles(overview_samples),
        "visible_nodes": renderer.stats.visible_nodes,
        "visible_edges": renderer.stats.visible_edges,
        "shape_count": renderer.stats.shape_count,
        "payload_kib": int(
            _estimate_payload_bytes(list(renderer._canvas.shapes or [])) / 1024
        ),
    }

    # Hit testing at working-set zoom, sampled across the whole scene.
    renderer.set_viewport(initial)
    rng = random.Random(7)
    hit_samples = []
    hits = 0
    for _ in range(HIT_SAMPLES):
        x = rng.uniform(scene_bounds.min_x, scene_bounds.max_x)
        y = rng.uniform(scene_bounds.min_y, scene_bounds.max_y)
        started = time.perf_counter()
        if renderer.hit_test(x, y) is not None:
            hits += 1
        hit_samples.append((time.perf_counter() - started) * 1000.0)
    result["hit_test"] = {
        **_percentiles(hit_samples),
        "hit_fraction": hits / HIT_SAMPLES,
    }

    # Collapse and expand a block of layers near the current viewport.
    collapse_samples = []
    expand_samples = []
    layer_count = len(graph.layers)
    for start in range(0, min(layer_count - COLLAPSE_BLOCK_LAYERS, 50), 10):
        patch = collapse_layers(graph, start, start + COLLAPSE_BLOCK_LAYERS - 1)
        assert renderer.scene is not None
        inverse = patch.inverse_for(renderer.scene)
        collapse_samples.append(_timed(lambda p=patch: renderer.apply_patch(p)))
        expand_samples.append(_timed(lambda p=inverse: renderer.apply_patch(p)))
    result["collapse"] = _percentiles(collapse_samples)
    result["expand"] = _percentiles(expand_samples)
    return result


def _environment() -> dict[str, Any]:
    import importlib.metadata

    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "flet": importlib.metadata.version("flet"),
        "view_px": [int(VIEW_WIDTH), int(VIEW_HEIGHT)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "flet_canvas_headless.json",
    )
    args = parser.parse_args(argv)

    report = {
        "harness": "renderer_benchmark.py",
        "renderer": "flet-canvas (screen-space rebuild per viewport change)",
        "environment": _environment(),
        "scenarios": [
            bench_scenario(1_000, layer_width=25),
            bench_scenario(10_000, layer_width=50),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"environment: {report['environment']}")
    print()
    print(
        "| nodes | edges | working-set p50/p95 | fps bound | all-visible p95 "
        "| hit p95 | collapse p95 | expand p95 |"
    )
    print("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for scenario in report["scenarios"]:
        ws = scenario["working_set"]
        print(
            f"| {scenario['node_count']} | {scenario['edge_count']} "
            f"| {ws['p50_ms']} / {ws['p95_ms']} ms | {ws['fps_bound_at_p95']} "
            f"| {scenario['all_visible']['p95_ms']} ms "
            f"| {scenario['hit_test']['p95_ms']} ms "
            f"| {scenario['collapse']['p95_ms']} ms "
            f"| {scenario['expand']['p95_ms']} ms |"
        )
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
