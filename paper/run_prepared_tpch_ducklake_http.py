from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking import load_experiment_config
from benchmarking.duckdb_build import resolve_system
from benchmarking.duckdb_runner import (
    benchmark_workload,
    ensure_results_schema,
    start_remote_http_service,
    stop_remote_http_service,
    summarize_results,
    write_summary,
)
from benchmarking.manifests import load_manifest


URL_PATTERN = re.compile(rb"http://127\.0\.0\.1:([0-9]+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run a prepared DuckLake/HTTP experiment without exporting Parquet again."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    experiment_root = args.experiment_root.resolve()
    if len(config.workloads) != 1 or len(config.systems) != 1:
        raise ValueError("prepared rerun requires exactly one workload and one system")

    workload_config = config.workloads[0]
    manifest_path = experiment_root / "workloads" / workload_config.name / "manifest.json"
    workload = load_manifest(manifest_path)
    if workload.storage_mode != "ducklake" or workload.ducklake_metadata_path is None:
        raise ValueError("prepared workload is not DuckLake-backed")

    metadata_bytes = workload.ducklake_metadata_path.read_bytes()
    ports = {int(match.group(1)) for match in URL_PATTERN.finditer(metadata_bytes)}
    if len(ports) != 1:
        raise ValueError(f"expected one HTTP port in DuckLake metadata, found {sorted(ports)}")
    port = ports.pop()

    copied_config_path = experiment_root / "config.json"
    shutil.copy2(config.config_path, copied_config_path)
    results_db = experiment_root / "results.duckdb"
    ensure_results_schema(results_db)
    failed_run_backup = experiment_root / "results.prefetch_failed.duckdb"
    if not failed_run_backup.exists():
        shutil.copy2(results_db, failed_run_backup)
    con = duckdb.connect(str(results_db))
    try:
        con.execute("DELETE FROM timing_runs WHERE experiment_name = ?", [config.experiment_name])
        con.execute("DELETE FROM profile_runs WHERE experiment_name = ?", [config.experiment_name])
    finally:
        con.close()

    system = resolve_system(
        config.systems[0], frozenset({"ducklake", "httpfs"})
    )
    service = start_remote_http_service(workload, preferred_port=port)
    if service is None:
        raise ValueError("prepared workload did not start an HTTP service")
    try:
        benchmark_workload(
            config.experiment_name,
            workload,
            system,
            config.systems[0].variants,
            config.benchmark,
            results_db,
            experiment_root / "sql",
            experiment_root / "profiles",
            service,
        )
    finally:
        stop_remote_http_service(service)

    summary = summarize_results(results_db, config.experiment_name)
    write_summary(experiment_root / "summary.json", summary)
    print(experiment_root)


if __name__ == "__main__":
    main()
