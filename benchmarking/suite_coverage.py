"""Compare measurements with the configured experiment, including absent sweeps."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import duckdb

from benchmarking.experiment_config import _parse_cross_product, applies_to_workload
from benchmarking.manifests import load_manifest
from benchmarking.query_inventory import load_queries


def expected_parameters(workload: dict) -> dict[str, dict]:
    if workload["kind"] != "cross_product" or "fixed" not in workload.get("params", {}):
        return {}
    from benchmarking.workloads.cross_product import _resolve_sweep_cases, _case_query_types, _sweep_query_filename
    spec = _parse_cross_product(workload, workload["name"])
    definition = spec.experiment_definition
    return {_sweep_query_filename(case, query, layout): {
                "sweep_kind": case.sweep_kind, "sweep_value": case.sweep_value,
                "query_type": query.kind, "probe_layout": layout}
            for case in _resolve_sweep_cases(definition, storage_kind=spec.storage_kind)
            for query in _case_query_types(definition, case) for layout in query.probe_layouts}


def expected_queries(workload: dict, run: Path) -> list[str]:
    parameters = expected_parameters(workload)
    if parameters:
        return sorted(parameters)
    manifest = run / "workloads" / workload["name"] / "manifest.json"
    if not manifest.is_file():
        return list(workload.get("query_names", []))
    return [q.name for q in load_queries(load_manifest(manifest).queries_dir)]


def result_coverage(raw: dict, run: Path) -> dict:
    workloads = [] if raw["suite"].get("microbenchmarks_only") else raw["workloads"]
    cases = {(w["name"], s["name"], v["label"], q)
             for w in workloads for s in raw["systems"]
             if applies_to_workload(s, w["name"])
             for v in s["variants"] if applies_to_workload(v, w["name"])
             for q in expected_queries(w, run)}
    timings = {(*key, rep) for key in cases for rep in range(1, raw["benchmark"]["repetitions"] + 1)}
    profiles = cases if raw["benchmark"].get("collect_profiles", True) else set()
    report = {"status": "complete", "workloads": sorted(w["name"] for w in workloads)}
    sweep_parameters = {w["name"]: expected_parameters(w) for w in workloads}
    with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
        for table, expected, extra in (("timing_runs", timings, ", repetition"), ("profile_runs", profiles, "")):
            counts = Counter(con.execute(
                f"SELECT workload_name, system_name, variant_label, query_name{extra} FROM {table}").fetchall())
            actual = set(counts)
            missing, unexpected = sorted(expected - actual), sorted(actual - expected)
            duplicate = sorted(key for key, count in counts.items() if count != 1)
            report[table] = {"expected_rows": len(expected), "actual_rows": sum(counts.values()),
                             "missing": missing, "unexpected": unexpected, "duplicate": duplicate}
            metadata_errors = []
            for workload, query, encoded in con.execute(
                    f"SELECT DISTINCT workload_name, query_name, query_parameters FROM {table}").fetchall():
                required = sweep_parameters.get(workload, {}).get(query)
                if not required:
                    continue
                try:
                    parameters = json.loads(encoded or "{}")
                except (TypeError, ValueError):
                    parameters = {}
                if not isinstance(parameters, dict) or any(parameters.get(k) != v for k, v in required.items()):
                    metadata_errors.append([workload, query])
            report[table]["metadata_errors"] = metadata_errors
            if missing or unexpected or duplicate or metadata_errors:
                report["status"] = "partial"
    return report
