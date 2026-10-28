"""EC2 host control for disposable DuckLake objects; credentials stay on the host."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from benchmarking.suite_io import write_json


def request_host(operation: str, workload: str, *, catalog: Path | None = None,
                 timeout: float = 86400) -> dict:
    directory = os.environ.get("BENCH_SYNC_DIR")
    if not directory:
        raise RuntimeError("DuckLake S3 storage requires the EC2 host (BENCH_SYNC_DIR); use local/HTTP storage for local tests")
    root = Path(directory)
    token = uuid.uuid4().hex
    request = {"token": token, "operation": operation, "workload": workload}
    if catalog is not None:
        request["catalog"] = str(catalog)
    write_json(root / "s3-request.json", request)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = root / "s3-response.json"
        if response.is_file():
            value = json.loads(response.read_text())
            if value.get("token") == token:
                if value.get("error"):
                    raise RuntimeError("DuckLake S3 host operation failed: " + value["error"])
                return value
        if operation == "upload":
            limit = os.environ.get("BENCH_DEADLINE_EPOCH")
            if limit and time.time() >= float(limit):
                raise InterruptedError("measurement deadline reached during S3 staging")
        time.sleep(.25)
    raise TimeoutError(f"S3 {operation} timed out for {workload}")


def stage(manifest, run: Path) -> str:
    result = request_host("upload", manifest.name, catalog=manifest.parquet_catalog_path)
    write_json(run / "s3" / f"{manifest.name}.json", result)
    return result["base_url"]


def cleanup(manifest, run: Path) -> None:
    result = request_host("delete", manifest.name, timeout=120)
    write_json(run / "s3" / f"{manifest.name}-cleanup.json", result)


@contextmanager
def data_scope(manifest, run: Path):
    if manifest.ducklake_data_source != "s3":
        yield None
        return
    try:
        yield stage(manifest, run)
    finally:
        cleanup(manifest, run)
