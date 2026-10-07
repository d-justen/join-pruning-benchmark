from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.experiment_config import MicrobenchmarkConfig, parse_microbenchmark_config
from benchmarking.manifests import utc_now_iso


FILTERS = {"bloom", "grafite", "diva", "prf_uncompressed", "prf"}
PRF_RANGE_KINDS = {"empty", "full", "mixed_bridge", "mixed_domain"}
ARTIFACT_SUBDIR = Path("microbenchmarks/filter-probes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty microbenchmark CSV: {path}")
    return rows


def range_series_name(row: dict[str, str]) -> str:
    if row["summary"] == "prf":
        if row["mode"] not in {"compressed", "uncompressed"}:
            raise ValueError(f"unknown PRF range mode: {row['mode']}")
        return "prf" if row["mode"] == "compressed" else "prf_uncompressed"
    return row["summary"]


def _number(row: dict[str, str], key: str, *, integer: bool = False, positive: bool = False) -> float | int:
    value = int(row[key]) if integer else float(row[key])
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid microbenchmark {key}: {row[key]}")
    return value


def validate_results(directory: Path, config: MicrobenchmarkConfig,
                     available: dict[str, list[str]]) -> dict[str, int]:
    clusters = set()
    count = config.min_clusters
    while count <= config.max_clusters:
        clusters.add(count)
        count *= 2
    point = read_csv(directory / "point.csv")
    ranges = read_csv(directory / "range.csv")
    point_keys = set()
    negative_counts = {}
    for row in point:
        rep = _number(row, "repetition", integer=True)
        cluster = _number(row, "cluster_count", integer=True, positive=True)
        key = (row["filter"], cluster, rep)
        if key in point_keys:
            raise ValueError(f"duplicate point measurement: {key}")
        point_keys.add(key)
        if int(row["seed"]) != config.seed + rep or int(row["probe_count"]) != config.probe_count:
            raise ValueError("point seed or probe_count does not match the config")
        negatives = _number(row, "actual_negatives", integer=True)
        fp = _number(row, "false_positives", integer=True)
        fpr = _number(row, "actual_fpr")
        if fp > negatives or negatives > config.probe_count or not math.isclose(
                fpr, fp / negatives if negatives else 0, abs_tol=1e-9):
            raise ValueError("point actual_fpr disagrees with measured counts")
        group = (cluster, rep)
        if group in negative_counts and negative_counts[group] != negatives:
            raise ValueError("filters did not use the same point probes")
        negative_counts[group] = negatives
        for field in ("build_time_ns", "post_processing_time_ns", "summary_bytes"):
            _number(row, field, integer=True)
        _number(row, "probe_throughput_per_sec", positive=True)
    expected_point = {(name, cluster, rep) for name in available["point"]
                      for cluster in clusters for rep in range(config.repetitions)}
    if point_keys != expected_point:
        raise ValueError("point CSV has missing or unexpected filters, clusters, or repetitions")
    range_keys = set()
    for row in ranges:
        rep = _number(row, "repetition", integer=True)
        cluster = _number(row, "cluster_count", integer=True, positive=True)
        key = (range_series_name(row), cluster, rep, row["query_kind"])
        if key in range_keys:
            raise ValueError(f"duplicate range measurement: {key}")
        range_keys.add(key)
        queries = _number(row, "query_count", integer=True, positive=True)
        ns = _number(row, "total_query_time_ns", integer=True, positive=True)
        rate = _number(row, "queries_per_sec", positive=True)
        matches = _number(row, "possibly_overlapping", integer=True)
        if queries != config.queries_per_cluster_count or matches > queries:
            raise ValueError("invalid range query or match count")
        if not math.isclose(rate, queries * 1e9 / ns, abs_tol=0.001):
            raise ValueError("range throughput disagrees with measured time")
        if (float(row["max_false_positive_rate"]) != config.max_fpr
                or int(row["seed"]) != config.seed + rep):
            raise ValueError("range filter settings or seed do not match the config")
        if row["correct_result"] != "true" or (row["query_kind"] != "empty" and matches != queries):
            raise ValueError("range benchmark reported a false negative")
        for field in ("summary_bytes", "post_processing_time_ns", "min_range_width", "max_range_width"):
            _number(row, field, integer=True)
    expected_range = {
        (name, cluster, rep, kind) for name in available["range"] for cluster in clusters
        for rep in range(config.repetitions)
        for kind in (PRF_RANGE_KINDS if name.startswith("prf") else {"empty"})
    }
    if range_keys != expected_range:
        raise ValueError("range CSV has missing or unexpected filters, clusters, repetitions, or query kinds")
    return {"point_rows": len(point), "range_rows": len(ranges)}


def resolve_microbenchmark_binaries(system: ResolvedSystem) -> dict[str, Path]:
    # Binary systems must provide the tools beside the CLI or in its tools/ directory.
    build_dir = system.build_dir or system.binary_path.parent
    binaries = {}
    for kind, target in (("point", "filter_benchmark"), ("range", "range_benchmark")):
        candidates = (build_dir / "tools" / target, build_dir / target)
        binary = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
        if binary is None:
            raise FileNotFoundError(f"missing {target} in the selected DuckDB build: {build_dir}")
        binaries[kind] = binary.resolve()
    return binaries


def _available_filters(binary: Path) -> list[str]:
    result = subprocess.run([str(binary), "--list-filters"], check=True, capture_output=True, text=True)
    names = result.stdout.split()
    if not names or set(names) - FILTERS or len(set(names)) != len(names):
        raise ValueError(f"invalid filter capabilities from {binary}")
    return sorted(names)


def _source_metadata(system: ResolvedSystem) -> dict:
    metadata = {"checkout_dir": str(system.checkout_dir) if system.checkout_dir else None,
                "build_dir": str(system.build_dir) if system.build_dir else None,
                "duckdb_version": system.version, "duckdb_sha256": sha256(system.binary_path)}
    if system.build_dir and (system.build_dir / "CMakeCache.txt").is_file():
        cache_path = system.build_dir / "CMakeCache.txt"
        metadata["cmake_cache_sha256"] = sha256(cache_path)
        keys = {"CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_GENERATOR", "BUILD_DIVA_BENCHMARK", "BUILD_GRAFITE_BENCHMARK"}
        metadata["build_settings"] = {}
        for line in cache_path.read_text().splitlines():
            if ":" in line and "=" in line:
                key = line.split(":", 1)[0]
                if key in keys:
                    metadata["build_settings"][key] = line.split("=", 1)[1]
    if system.checkout_dir:
        for name, args in (("git_commit", ["rev-parse", "HEAD"]),
                           ("git_status", ["status", "--porcelain", "--", "."]),
                           ("git_diff_sha256", ["diff", "--binary", "HEAD", "--", "."])):
            result = subprocess.run(["git", *args], cwd=system.checkout_dir, capture_output=True)
            if result.returncode == 0:
                metadata[name] = (hashlib.sha256(result.stdout).hexdigest() if name.endswith("sha256")
                                  else result.stdout.decode().strip())
    return metadata


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_microbenchmark_manifest(experiment_root: Path) -> dict:
    directory = experiment_root / ARTIFACT_SUBDIR
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("microbenchmarks are incomplete; rerun with --only-microbenchmarks")
    config = parse_microbenchmark_config(manifest["config"])
    for kind in ("point", "range"):
        if sha256(directory / f"{kind}.csv") != manifest["outputs"][kind]["sha256"]:
            raise ValueError(f"{kind} CSV differs from the completed microbenchmark manifest")
    validate_results(directory, config, manifest["available_filters"])
    return manifest


def run_microbenchmarks(config: MicrobenchmarkConfig, system: ResolvedSystem,
                       experiment_root: Path, *, reuse_completed: bool = False) -> Path:
    directory = experiment_root / ARTIFACT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    config_dict = json.loads(json.dumps(asdict(config)))
    try:
        binaries = resolve_microbenchmark_binaries(system)
        identities = {kind: {"path": str(path), "sha256": sha256(path)} for kind, path in binaries.items()}
    except Exception as exc:
        _write_manifest(manifest_path, {"schema_version": 1, "status": "failed", "config": config_dict,
                                        "finished_at": utc_now_iso(), "error": str(exc)})
        raise
    if reuse_completed and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("status") == "complete" and saved["config"] == config_dict and saved["binaries"] == identities:
            load_microbenchmark_manifest(experiment_root)
            return directory
    manifest = {
        "schema_version": 1, "status": "running", "started_at": utc_now_iso(),
        "config": config_dict, "binaries": identities,
        "commands": {}, "outputs": {},
    }
    _write_manifest(manifest_path, manifest)
    try:
        manifest["source"] = _source_metadata(system)
        manifest["host"] = {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count()}
        available = {kind: _available_filters(path) for kind, path in binaries.items()}
        manifest["available_filters"] = available
        manifest["unavailable_filters"] = {kind: sorted((FILTERS - ({"bloom"} if kind == "range" else set()))
                                                       - set(names)) for kind, names in available.items()}
        for kind, names in available.items():
            required = set(config.required_filters) | {"prf", "prf_uncompressed"}
            if kind == "range":
                required.discard("bloom")
            else:
                required.add("bloom")
            missing = required - set(names)
            if missing:
                raise ValueError(f"{kind} benchmark lacks required filters: {', '.join(sorted(missing))}")
        common = [f"--{key.replace('_', '-')}={getattr(config, key)}" for key in (
            "domain_size", "total_keys", "min_clusters", "max_clusters", "repetitions", "seed",
            "grafite_bpk", "diva_bpk", "max_fpr")]
        for kind, binary in binaries.items():
            output = directory / f"{kind}.csv"
            temporary = directory / f"{kind}.csv.tmp"
            count_arg = (f"--probe-count={config.probe_count}" if kind == "point" else
                         f"--queries-per-cluster-count={config.queries_per_cluster_count}")
            command = [str(binary), *common, count_arg, f"--output={temporary.resolve()}"]
            manifest["commands"][kind] = command
            _write_manifest(manifest_path, manifest)
            with (directory / f"{kind}.log").open("w", encoding="utf-8") as log:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            temporary.replace(output)
            manifest["outputs"][kind] = {"path": output.name, "sha256": sha256(output)}
        manifest["validation"] = validate_results(directory, config, available)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_at"] = utc_now_iso()
        _write_manifest(manifest_path, manifest)
    return directory
