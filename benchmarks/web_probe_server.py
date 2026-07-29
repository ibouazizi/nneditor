"""Serves the hosted-web-renderer probe and collects its results.

``web_probe/index.html`` is the third candidate in the Phase 0 renderer
comparison: a plain HTML canvas renderer of the same synthetic graphs. The
page benchmarks itself in the browser and POSTs its JSON report back; this
server writes it to ``benchmarks/results/web_probe.json`` and exits, so the
whole measurement is one command:

    uv run python benchmarks/web_probe_server.py

Opens the default browser automatically; pass ``--no-open`` to visit the
printed URL manually (e.g. from another browser to widen the matrix).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PAGE = Path(__file__).parent / "web_probe" / "index.html"
OUT = Path(__file__).parent / "results" / "web_probe.json"


class ProbeHandler(BaseHTTPRequestHandler):
    server: ProbeServer  # type: ignore[assignment]

    def do_GET(self) -> None:
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        report = json.loads(self.rfile.read(length))
        existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
        existing.append(report)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        self.send_response(204)
        self.end_headers()
        print(json.dumps(report, indent=2))
        print(f"written: {OUT}")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args: object) -> None:
        pass


class ProbeServer(HTTPServer):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    server = ProbeServer(("127.0.0.1", args.port), ProbeHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"probe at {url} - waiting for the browser to submit results")
    if not args.no_open:
        webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
