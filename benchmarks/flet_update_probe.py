"""Measures which Flet canvas update strategy can carry a per-frame pan.

The interactive benchmark showed a ~100x gap between building a culled shape
list in Python (~12 ms) and pushing it through ``update()`` to the desktop
client (~1000 ms). This probe isolates where that time goes by timing three
strategies on the same 1,200-shape canvas — the 10,000-node working set size:

``replace``
    Assign a brand-new list of shape objects every frame (what the naive
    adapter does).
``mutate``
    Keep the shape objects and change only their coordinate fields, letting
    Flet's dataclass diffing send small patches.
``offset``
    Leave shapes untouched and move the canvas's container offset — the
    GPU-side pan a production adapter would use between cull rebuilds.

Run: ``uv run python benchmarks/flet_update_probe.py``. Appends results to
``benchmarks/results/flet_update_probe.json`` and exits by itself.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import flet as ft
import flet.canvas as cv

SHAPES = 1200
FRAMES = 40
OUT = Path(__file__).parent / "results" / "flet_update_probe.json"


def _make_shapes(offset: float) -> list[cv.Shape]:
    shapes: list[cv.Shape] = []
    for index in range(SHAPES):
        shapes.append(
            cv.Rect(
                x=(index % 40) * 40.0 + offset,
                y=(index // 40) * 30.0,
                width=32.0,
                height=22.0,
                paint=ft.Paint(color="#4C78A8", style=ft.PaintingStyle.FILL),
            )
        )
    return shapes


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
    }


def main(page: ft.Page) -> None:
    page.title = "Flet update strategy probe"
    canvas = cv.Canvas(shapes=_make_shapes(0.0), width=1600, height=1000)
    container = ft.Container(content=canvas, width=1600, height=1000)
    page.add(container)
    page.update()

    def run() -> None:
        report: dict[str, object] = {
            "harness": "flet_update_probe.py",
            "mode": "web" if page.web else "desktop",
            "shapes": SHAPES,
        }

        samples = []
        for frame in range(FRAMES):
            started = time.perf_counter()
            canvas.shapes = _make_shapes(float(frame))
            canvas.update()
            samples.append((time.perf_counter() - started) * 1000.0)
        report["replace"] = _stats(samples)

        samples = []
        for _frame in range(FRAMES):
            started = time.perf_counter()
            for shape in canvas.shapes:
                assert isinstance(shape, cv.Rect)
                shape.x = shape.x + 1.0
            canvas.update()
            samples.append((time.perf_counter() - started) * 1000.0)
        report["mutate"] = _stats(samples)

        samples = []
        for frame in range(FRAMES):
            started = time.perf_counter()
            container.offset = ft.Offset(frame * 0.001, 0.0)
            container.update()
            samples.append((time.perf_counter() - started) * 1000.0)
        report["offset"] = _stats(samples)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
        existing.append(report)
        OUT.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        os._exit(0)

    page.run_thread(run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--web",
        action="store_true",
        help="measure through Flet's web mode instead of the desktop client",
    )
    args = parser.parse_args()
    view = ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP
    ft.run(main, view=view)
