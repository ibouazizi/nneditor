"""Interactive renderer benchmark app (task P0.5).

The headless harness bounds Python-side frame cost; this app closes the loop
through Flet's transport and Flutter's paint. Run it on the targets the plan
requires:

* desktop: ``uv run python benchmarks/renderer_app.py --nodes 10000``
* web:     ``uv run flet run --web benchmarks/renderer_app.py``

Drag to pan, scroll to zoom, tap to select. The HUD shows the per-frame update
round trip (Python build + serialize + send) and visible-glyph counts.

``--auto`` runs a scripted pan-and-zoom sweep instead, writes the measured
frame times to ``benchmarks/results/flet_canvas_interactive.json``, and exits.
The measured time covers up to the point Flet has handed the shape list to the
transport; Flutter paints asynchronously after that, so visually confirming
smoothness on real hardware remains part of the acceptance check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import flet as ft

from nneditor.rendering.flet_canvas import FletCanvasRenderer
from nneditor.rendering.scene import Viewport
from nneditor.rendering.synthetic import synthetic_graph

VIEW_WIDTH = 1600.0
VIEW_HEIGHT = 1000.0
AUTO_FRAMES = 120


def _fit_viewport(
    renderer: FletCanvasRenderer, scale: float, x: float, y: float
) -> Viewport:
    return Viewport(
        x=x, y=y, width=VIEW_WIDTH / scale, height=VIEW_HEIGHT / scale, scale=scale
    )


def build_main(node_count: int, auto: bool, out: Path):
    graph = synthetic_graph(
        node_count, layer_width=50 if node_count >= 5000 else 25, seed=42
    )

    def main(page: ft.Page) -> None:
        page.title = f"NNEditor renderer benchmark - {node_count} nodes"
        hud = ft.Text("", size=12, font_family="monospace")
        state = {"scale": 1.0, "x": 0.0, "y": 0.0}
        renderer = FletCanvasRenderer(on_select=lambda ids: page.update())

        def refresh_hud() -> None:
            stats = renderer.stats
            hud.value = (
                f"nodes={stats.visible_nodes} edges={stats.visible_edges} "
                f"shapes={stats.shape_count} build={stats.build_ms:.1f}ms "
                f"scale={state['scale']:.2f}"
            )

        def set_view() -> None:
            renderer.set_viewport(
                _fit_viewport(renderer, state["scale"], state["x"], state["y"])
            )
            refresh_hud()
            page.update()

        def on_pan(event: ft.DragUpdateEvent) -> None:
            state["x"] -= event.local_delta.x / state["scale"]
            state["y"] -= event.local_delta.y / state["scale"]
            set_view()

        def on_scroll(event: ft.ScrollEvent) -> None:
            factor = 0.9 if (event.scroll_delta_y or 0) > 0 else 1.1
            state["scale"] = max(0.02, min(4.0, state["scale"] * factor))
            set_view()

        detector = renderer.control
        assert isinstance(detector, ft.GestureDetector)
        detector.on_pan_update = on_pan
        detector.on_scroll = on_scroll

        page.add(
            ft.Column(
                controls=[
                    hud,
                    ft.Container(content=detector, expand=True, bgcolor="#FAFAFA"),
                ],
                expand=True,
            )
        )
        renderer.replace_scene(graph.scene, _fit_viewport(renderer, 1.0, 0.0, 0.0))
        refresh_hud()
        page.update()

        if not auto:
            return

        def sweep() -> None:
            bounds = graph.scene.bounds
            frames: dict[str, list[float]] = {"pan": [], "zoom": []}
            for step in range(AUTO_FRAMES):
                fraction = step / (AUTO_FRAMES - 1)
                state["x"] = bounds.min_x + fraction * max(
                    0.0, bounds.width - VIEW_WIDTH
                )
                state["y"] = bounds.min_y + fraction * max(
                    0.0, bounds.height - VIEW_HEIGHT
                )
                started = time.perf_counter()
                set_view()
                frames["pan"].append((time.perf_counter() - started) * 1000.0)
            for step in range(AUTO_FRAMES):
                state["scale"] = 1.0 - 0.9 * (step / (AUTO_FRAMES - 1))
                started = time.perf_counter()
                set_view()
                frames["zoom"].append((time.perf_counter() - started) * 1000.0)
            report = {
                "harness": "renderer_app.py --auto",
                "mode": "desktop" if page.web is False else "web",
                "node_count": node_count,
                "frames_per_sweep": AUTO_FRAMES,
            }
            for name, samples in frames.items():
                ordered = sorted(samples)
                report[name] = {
                    "p50_ms": round(statistics.median(ordered), 3),
                    "p95_ms": round(
                        ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3
                    ),
                    "max_ms": round(ordered[-1], 3),
                }
            out.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if out.exists():
                existing = json.loads(out.read_text(encoding="utf-8"))
            existing.append(report)
            out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            # Window.destroy is async in Flet 0.86 and the desktop client has
            # been observed to outlive it; a benchmark run must never hang the
            # calling terminal, so exit the interpreter directly.
            import os

            os._exit(0)

        page.run_thread(sweep)

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument(
        "--web",
        action="store_true",
        help="serve through Flet's web mode in the default browser instead of "
        "the desktop client",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "flet_canvas_interactive.json",
    )
    args = parser.parse_args()
    view = ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP
    ft.run(build_main(args.nodes, args.auto, args.out), view=view)
    sys.exit(0)
