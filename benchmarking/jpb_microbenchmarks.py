"""Standalone replay of JPB row-group and point filtering (no SQL join execution)."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from benchmarking.experiment_config import (
    CrossProductWorkloadConfig, ExperimentConfig, JPBMicrobenchmarkConfig,
)
from benchmarking.manifests import utc_now_iso
from benchmarking.microbenchmarks import (BUILD_CONTRACT, _source_metadata, _write_manifest, read_csv, sha256,
                                         validate_build_metadata)
from benchmarking.workloads.cross_product import _resolve_sweep_cases, _selected_rank_query, _sweep_query_filename

ARTIFACT_SUBDIR = Path("microbenchmarks/jpb-simulation")
FILTERS = {"prf_uncompressed", "prf", "grafite", "diva"}
INPUT_VERSION = "jpb-dense-v1-mt19937_64-modulo"


def resolve_jpb_cases(config: ExperimentConfig) -> list[dict]:
    spec = config.jpb_microbenchmarks
    if spec is None:
        return []
    workload = next((w for w in config.workloads if w.name == spec.workload_name), None)
    if not isinstance(workload, CrossProductWorkloadConfig) or workload.experiment_definition is None:
        raise ValueError("jpb_microbenchmarks requires a generated cross_product workload")
    definition = workload.experiment_definition
    if (workload.storage_kind != "duckdb" or workload.remote_http_parquet_latency_ms is not None
            or definition.sweep.kind != "dimension_cardinality"):
        raise ValueError("jpb_microbenchmarks supports native dimension_cardinality sweeps")
    cases = []
    for case in _resolve_sweep_cases(definition, storage_kind=workload.storage_kind):
        if (case.join_count != 1 or case.build_key_multiplicity != 1
                or case.key_domain_size != case.dimension_distinct_keys
                or not 0 <= case.seed < 2**64 or case.fact_rows >= 2**63
                or case.dimension_cardinality >= 2**63 or case.selected_distinct_keys < 2):
            raise ValueError("jpb_microbenchmarks requires one join, a dense integer dimension, at least two selected keys, and a uint64 seed")
        for query in definition.query_types:
            if query.kind not in {"range", "shoulder", "seasonal", "scattered"}:
                raise ValueError("unsupported JPB selection pattern")
            selected_sql = ("SELECT CAST(key_rank + 1 AS UBIGINT) AS key FROM (\n"
                            + _selected_rank_query(case, query, 1) + "\n) selected ORDER BY key")
            for layout in query.probe_layouts:
                if layout not in {"ordered", "uniform"}:
                    raise ValueError("unsupported JPB probe layout")
                cases.append({"query": _sweep_query_filename(case, query, layout),
                              "pattern": (f"seasonal_{query.cluster_count}" if query.kind == "seasonal"
                                          and query.cluster_count != 10 else query.kind), "layout": layout,
                              "dimension_cardinality": case.dimension_cardinality,
                              "probe_rows": case.fact_rows, "row_group_size": case.row_group_size,
                              "seed": case.seed, "build_keys": case.selected_distinct_keys,
                              "selected_sql": selected_sql})
    if not cases or len({c["query"] for c in cases}) != len(cases):
        raise ValueError("JPB simulation needs nonempty, unique query cases")
    if len({(c["dimension_cardinality"], c["layout"], c["pattern"]) for c in cases}) != len(cases):
        raise ValueError("JPB simulation supports one case per dimension/layout/pattern")
    return sorted(cases, key=lambda c: (c["dimension_cardinality"], c["layout"], c["query"]))


def resolve_binary(system) -> Path:
    from benchmarking.benchmark_binary import select_benchmark_binary
    build_dir = system.build_dir or system.binary_path.parent
    for path in (build_dir / "tools/join_pruning_benchmark", build_dir / "join_pruning_benchmark"):
        if path.is_file() and os.access(path, os.X_OK):
            return select_benchmark_binary(path)
    raise FileNotFoundError(f"missing join_pruning_benchmark in {build_dir}")


def available_filters(binary: Path) -> list[str]:
    names = subprocess.run([str(binary), "--list-filters"], check=True,
                           capture_output=True, text=True).stdout.split()
    if not names or set(names) - FILTERS or len(names) != len(set(names)):
        raise ValueError("invalid JPB benchmark capabilities")
    return sorted(names)


def validate_results(path: Path, spec: JPBMicrobenchmarkConfig, cases: list[dict], available: list[str]) -> dict:
    rows = read_csv(path)
    expected = {(c["query"], f, r) for c in cases for f in available for r in range(spec.repetitions)}
    case_map = {c["query"]: c for c in cases}
    seen, identities = set(), {}
    for row in rows:
        def number(key):
            value = int(row[key])
            if value < 0:
                raise ValueError(f"negative JPB {key}")
            return value
        key = (row["query"], row["filter"], number("repetition"))
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate JPB measurement: {key}")
        seen.add(key)
        validate_build_metadata(row, spec, row["filter"])
        case = case_map[key[0]]
        for field in ("dimension_cardinality", "probe_rows", "row_group_size", "seed", "build_keys"):
            if number(field) != case[field]:
                raise ValueError(f"JPB {field} differs from workload")
        if any(row[field] != case[field] for field in ("layout", "pattern")):
            raise ValueError("JPB layout or pattern differs from workload")
        n = case["probe_rows"]
        groups = (n + case["row_group_size"] - 1) // case["row_group_size"]
        if (number("range_probes") != groups
                or sum(number(f"groups_{kind}") for kind in ("rejected", "accepted", "mixed")) != groups
                or sum(number(field) for field in ("rows_rejected", "rows_accepted", "point_probes")) != n):
            raise ValueError("inconsistent JPB pruning counts")
        for kind, field in (("rejected", "rows_rejected"), ("accepted", "rows_accepted"), ("mixed", "point_probes")):
            group_count, row_count = number(f"groups_{kind}"), number(field)
            if not group_count <= row_count <= group_count * case["row_group_size"]:
                raise ValueError("inconsistent JPB group/row counts")
        candidates, positives = number("candidate_rows"), number("true_positives")
        if (number("false_negatives") != 0 or not positives <= candidates <= n
                or number("false_positives") != candidates - positives
                or number("point_candidates") > number("point_probes")
                or candidates != number("rows_accepted") + number("point_candidates")
                or not number("range_false_positives") <= number("empty_ranges") <= groups):
            raise ValueError("JPB correctness or candidate counts failed")
        if number("total_time_ns") != sum(number(f"{phase}_time_ns") for phase in ("build", "range", "point")):
            raise ValueError("JPB total does not equal phase sum")
        if number("finalization_time_ns") > number("build_time_ns"):
            raise ValueError("JPB finalization exceeds build")
        if number("point_probes") == 0 and number("point_time_ns") != 0:
            raise ValueError("JPB zero point work must have zero time")
        for field in ("grafite_bpk", "diva_bpk", "max_fpr"):
            if not math.isfinite(float(row[field])) or float(row[field]) != getattr(spec, field):
                raise ValueError("JPB filter parameters differ from config")
        number("summary_bytes")
        identity = tuple(number(field) for field in ("build_checksum", "probe_checksum", "true_positives", "empty_ranges"))
        if key[0] in identities and identities[key[0]] != identity:
            raise ValueError("JPB filters did not receive identical inputs")
        identities[key[0]] = identity
    if seen != expected:
        raise ValueError("missing JPB measurements")
    return {"measurement_rows": len(rows), "query_count": len(cases), "false_negatives": 0}


def load_jpb_manifest(root: Path, config: ExperimentConfig) -> dict:
    directory = root / ARTIFACT_SUBDIR
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
        raise ValueError("JPB microbenchmarks are incomplete; rerun with --only-microbenchmarks")
    if manifest.get("build_contract") != BUILD_CONTRACT:
        raise ValueError("saved construction contract differs; rerun the microbenchmarks")
    if (manifest["config"] != json.loads(json.dumps(asdict(config.jpb_microbenchmarks)))
            or manifest["cases"] != resolve_jpb_cases(config) or manifest["input_version"] != INPUT_VERSION):
        raise ValueError("saved JPB workload/configuration differs; rerun the microbenchmarks")
    if set(manifest["outputs"]) != {"cases", "measurements", *[f"input_{i}" for i in range(len(manifest["cases"]))]}:
        raise ValueError("incomplete JPB artifact manifest")
    for output in manifest["outputs"].values():
        if sha256(directory / output["path"]) != output["sha256"]:
            raise ValueError("JPB artifact differs from its manifest")
    available = manifest["available_filters"]
    required = set(config.jpb_microbenchmarks.required_filters) | {"prf", "prf_uncompressed"}
    if set(available) - FILTERS or not required <= set(available) or len(set(available)) != len(available):
        raise ValueError("invalid saved JPB capabilities")
    validate_results(directory / "measurements.csv", config.jpb_microbenchmarks, manifest["cases"], available)
    return manifest


def run_jpb_microbenchmarks(config: ExperimentConfig, system, root: Path, *, reuse_completed: bool = False,
                            checkpoint_cases: bool = False, checkpoint_callback=None) -> Path:
    spec = config.jpb_microbenchmarks
    cases = resolve_jpb_cases(config)
    directory = root / ARTIFACT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = {"schema_version": 2, "build_contract": BUILD_CONTRACT, "status": "running", "started_at": utc_now_iso(),
                "config": json.loads(json.dumps(asdict(spec))), "cases": cases, "input_version": INPUT_VERSION,
                "scope": "single-threaded row-group filtering; dense non-null integer keys; no Bloom fallback",
                "input_order": "build keys supplied sorted; probes ordered or synthetic uniform",
                "outputs": {}}
    try:
        from benchmarking.benchmark_binary import binary_backend
        binary = resolve_binary(system)
        identity = {"path": str(binary), "sha256": sha256(binary), "diva_backend": binary_backend(binary)}
        manifest["binary"] = identity
        if reuse_completed and manifest_path.is_file():
            saved = json.loads(manifest_path.read_text())
            if saved.get("status") == "complete" and saved.get("binary") == identity:
                load_jpb_manifest(root, config)
                return directory
        _write_manifest(manifest_path, manifest)
        names = available_filters(binary)
        manifest["available_filters"] = names
        manifest["unavailable_filters"] = sorted(FILTERS - set(names))
        missing = (set(spec.required_filters) | {"prf", "prf_uncompressed"}) - set(names)
        if missing:
            raise ValueError(f"JPB benchmark lacks required filters: {', '.join(sorted(missing))}")
        manifest["source"] = _source_metadata(system)
        manifest["host"] = {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count()}
        inputs = directory / "inputs"
        inputs.mkdir(exist_ok=True)
        case_path = directory / "cases.tsv"
        with case_path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for i, case in enumerate(cases):
                sql_path = inputs / f"{i}.sql"
                resource_policy = json.loads(os.environ.get("BENCH_PREPARATION_CONFIG", "{}"))
                settings = []
                # The simulation is deliberately single threaded. Its input SQL
                # still obeys the suite's memory/spill budget; C++ vectors are
                # bounded by the surrounding container's memory limit.
                for setting in ("memory_limit", "temp_directory", "max_temp_directory_size"):
                    if setting in resource_policy:
                        value = str(resource_policy[setting]).replace("'", "''")
                        settings.append(f"SET {setting}='{value}';")
                sql_path.write_text("\n".join(settings + [case["selected_sql"] + ";"]) + "\n")
                manifest["outputs"][f"input_{i}"] = {"path": str(sql_path.relative_to(directory)), "sha256": sha256(sql_path)}
                writer.writerow([case[field] for field in ("query", "pattern", "layout", "dimension_cardinality",
                                 "probe_rows", "row_group_size", "seed", "build_keys")] + [f"inputs/{i}.sql"])
        manifest["outputs"]["cases"] = {"path": case_path.name, "sha256": sha256(case_path)}
        temporary = directory / "measurements.csv.tmp"
        command = [str(binary), f"--cases={case_path.resolve()}", f"--output={temporary.resolve()}",
                   *[f"--{key.replace('_', '-')}={getattr(spec, key)}" for key in
                     ("repetitions", "warmups", "order_seed", "construction_seed", "grafite_bpk", "diva_bpk", "max_fpr")]]
        manifest["command"] = command
        _write_manifest(manifest_path, manifest)
        if checkpoint_cases:
            from benchmarking.case_checkpoints import run_case, combine
            chunks = []
            for i, case in enumerate(cases):
                chunk = directory / "checkpoints" / f"case-{i}.csv"
                run_case([arg for arg in command if not arg.startswith("--output=")] + [f"--case-index={i}"],
                         chunk, {"binary": identity, "config": manifest["config"], "cases": cases, "index": i},
                         checkpoint_callback)
                validate_results(chunk, spec, [case], names)
                chunks.append(chunk)
            combine(chunks, temporary)
            manifest["checkpoint_protocol"] = "case-v1; original per-case filter order; fresh probes per process"
        else:
            with (directory / "benchmark.log").open("w") as log:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["validation"] = validate_results(temporary, spec, cases, names)
        output = directory / "measurements.csv"
        temporary.replace(output)
        manifest["outputs"]["measurements"] = {"path": output.name, "sha256": sha256(output)}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        # A successful reuse must not overwrite the original manifest.
        if manifest["status"] != "running" or not reuse_completed:
            manifest["finished_at"] = utc_now_iso()
            _write_manifest(manifest_path, manifest)
    return directory
