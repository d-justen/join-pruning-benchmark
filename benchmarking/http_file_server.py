from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build_handler(root_dir: Path, latency_ms: int) -> type[BaseHTTPRequestHandler]:
    root_dir = root_dir.resolve()
    latency_s = latency_ms / 1000.0
    stats_lock = threading.Lock()
    stats: dict[str, object] = {}

    def reset_stats() -> None:
        with stats_lock:
            stats.clear()
            stats.update(
                {
                    "requests": 0,
                    "head_requests": 0,
                    "get_requests": 0,
                    "range_get_requests": 0,
                    "bytes_sent": 0,
                    "paths": {},
                }
            )

    reset_stats()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def _serve(self, *, send_body: bool) -> None:
            if self.path == "/__health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", "2")
                self.end_headers()
                if send_body:
                    self.wfile.write(b"ok")
                return
            if self.path == "/__stats":
                with stats_lock:
                    payload = json.dumps(stats, sort_keys=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if send_body:
                    self.wfile.write(payload)
                return
            if self.path == "/__reset_stats":
                reset_stats()
                payload = b"ok"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if send_body:
                    self.wfile.write(payload)
                return

            file_path = self._resolve_path()
            if file_path is None or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            time.sleep(latency_s)
            file_size = file_path.stat().st_size
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            range_header = self.headers.get("Range")
            if range_header is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if send_body:
                    with file_path.open("rb") as file_obj:
                        self.wfile.write(file_obj.read())
                self._record_request(file_size if send_body else 0, ranged=False)
                return

            if not range_header.startswith("bytes="):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start_raw, end_raw = range_header.removeprefix("bytes=").split("-", 1)
            if not start_raw:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start = int(start_raw)
            end = file_size - 1 if not end_raw else min(int(end_raw), file_size - 1)
            if start < 0 or start >= file_size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

            content_length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()
            if not send_body:
                self._record_request(0, ranged=True)
                return
            with file_path.open("rb") as file_obj:
                file_obj.seek(start)
                self.wfile.write(file_obj.read(content_length))
            self._record_request(content_length, ranged=True)

        def _record_request(self, byte_count: int, *, ranged: bool) -> None:
            relative_path = self.path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
            with stats_lock:
                stats["requests"] = int(stats["requests"]) + 1
                method_key = "get_requests" if self.command == "GET" else "head_requests"
                stats[method_key] = int(stats[method_key]) + 1
                if ranged and self.command == "GET":
                    stats["range_get_requests"] = int(stats["range_get_requests"]) + 1
                stats["bytes_sent"] = int(stats["bytes_sent"]) + byte_count
                paths = stats["paths"]
                assert isinstance(paths, dict)
                path_stats = paths.setdefault(
                    relative_path,
                    {"requests": 0, "range_get_requests": 0, "bytes_sent": 0},
                )
                path_stats["requests"] += 1
                path_stats["range_get_requests"] += int(ranged and self.command == "GET")
                path_stats["bytes_sent"] += byte_count

        def _resolve_path(self) -> Path | None:
            relative_path = self.path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
            candidate = (root_dir / relative_path).resolve()
            try:
                candidate.relative_to(root_dir)
            except ValueError:
                return None
            return candidate

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve files with fixed per-request latency.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--latency-ms", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(Path(args.root_dir), args.latency_ms),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
