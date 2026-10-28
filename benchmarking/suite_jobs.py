"""Split SQL experiments without changing their definitions; combine verified jobs."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from benchmarking.suite_io import fingerprint, sha256, write_json
from benchmarking.experiment_config import applies_to_workload


def job_configs(raw: dict) -> list[dict]:
    if any(k in raw for k in ("microbenchmarks", "jpb_microbenchmarks")):
        raise ValueError("split a SQL-only configuration; run microbenchmarks separately")
    jobs = []
    for workload in raw["workloads"]:
        job = copy.deepcopy(raw)
        name = workload["name"]
        job["experiment_name"] = raw["experiment_name"] + "-" + name
        job["workloads"] = [copy.deepcopy(workload)]
        job["systems"] = [s for s in job["systems"] if "workload_names" not in s or name in s["workload_names"]]
        if not job["systems"]:
            raise ValueError(f"no configured system measures {name}")
        for system in job["systems"]:
            system["variants"] = [v for v in system["variants"] if applies_to_workload(v, name)]
            for variant in system["variants"]:
                if "workload_names" in variant:
                    variant["workload_names"] = [name]
            if "workload_names" in system:
                system["workload_names"] = [name]
        job["systems"] = [s for s in job["systems"] if s["variants"]]
        if not job["systems"]:
            raise ValueError(f"no configured variant measures {name}")
        job["suite"].update(required_workloads=[name], required_figures=[],
                            job_parent_sha256=fingerprint(raw))
        jobs.append(job)
    return jobs


def split_jobs(config: Path, output: Path) -> list[Path]:
    raw = json.loads(config.read_text())
    jobs = job_configs(raw)
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for job in jobs:
        path = output / (job["workloads"][0]["name"] + ".json")
        write_json(path, job)
        paths.append(path)
    write_json(output / "jobs.json", {"config_sha256": fingerprint(raw), "jobs": [p.name for p in paths],
                                      "required_figures": raw["suite"].get("required_figures", [])})
    return paths


def combine_jobs(config: Path, sources: list[Path], destination: Path) -> None:
    """Keep original measurements immutable; only aggregation rewrites profile paths."""
    raw = json.loads(config.read_text())
    expected = {j["workloads"][0]["name"]: j for j in job_configs(raw)}
    source_jobs = []
    seen = set()
    identity = None
    for source in sources:
        source = source.resolve()
        request = json.loads((source / "request.json").read_text())
        names = [w["name"] for w in request["workloads"]]
        if len(names) != 1 or names[0] in seen or request != expected.get(names[0]):
            raise ValueError(f"duplicate, incompatible or unexpected job: {source}")
        seen.add(names[0])
        state = json.loads((source / "run.json").read_text())
        coverage = json.loads((source / "coverage.json").read_text())
        if state["status"] != "complete" or coverage["status"] != "complete":
            raise ValueError(f"job is incomplete: {source}")
        candidate = {k: state[k] for k in ("source", "binaries", "runtime")}
        if identity is not None and candidate != identity:
            raise ValueError("job runner, binaries or runtime differ; do not mix measurements")
        identity = candidate
        for status in (source / "units").glob("*/status.json"):
            unit = json.loads(status.read_text())
            if unit["action"] == "measure":
                data = status.parent / "results.duckdb"
                if unit["status"] != "complete" or sha256(data) != unit["sha256"]:
                    raise ValueError(f"invalid measurement: {data}")
                for name, digest in unit["artifacts"].items():
                    path = (source / name).resolve()
                    if not path.is_relative_to(source) or sha256(path) != digest:
                        raise ValueError(f"invalid measurement artifact: {name}")
        source_jobs.append((source, state))
    if seen != expected.keys():
        raise ValueError("missing jobs: " + ", ".join(sorted(expected.keys() - seen)))
    destination.mkdir(parents=True, exist_ok=False)
    for source, _ in source_jobs:
        for name in ("units", "profiles", "sql", "workloads"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name, dirs_exist_ok=True)
    resolved = {**raw, "experiment_name": destination.name, "output_root": str(destination.parent)}
    write_json(destination / "config.json", resolved)
    write_json(destination / "request.json", raw)
    write_json(destination / "combined.json", {
        "parent_config_sha256": fingerprint(raw), "identity": identity,
        "sources": [{"path": str(p), "config_sha256": state["config_sha256"]} for p, state in source_jobs],
    })
    write_json(destination / "run.json", {"status": "combined", "config_sha256": fingerprint(raw), **identity})


def combine_dimension_cases(config: Path, sources: list[Path], destination: Path) -> None:
    """Combine disjoint native dimension sweeps, including a finished workload
    from a run whose other workloads are still running. Source evidence stays
    immutable; the combined directory supports analysis/export only.
    """
    from benchmarking.suite import aggregate, measurement_key, valid_measurement, matches_unit_identity
    from benchmarking.suite_coverage import expected_parameters, result_coverage

    raw = json.loads(config.read_text())
    if len(raw["workloads"]) != 1 or any(k in raw for k in ("microbenchmarks", "jpb_microbenchmarks")):
        raise ValueError("case combination requires one SQL-only workload")
    workload = raw["workloads"][0]
    name = workload["name"]
    if (workload["kind"] != "cross_product" or workload["params"]["sweep"]["kind"] != "dimension_cardinality"
            or workload["params"].get("storage", {}).get("kind", "duckdb") != "duckdb"):
        raise ValueError("case combination requires a native dimension-cardinality sweep")

    def recipe(value):
        value = copy.deepcopy(value)
        value.pop("estimated_peak_gib", None)
        value["params"]["sweep"].pop("values")
        return value

    def selected_systems(value):
        systems = copy.deepcopy([s for s in value["systems"] if applies_to_workload(s, name)])
        for system in systems:
            system["variants"] = [v for v in system["variants"] if applies_to_workload(v, name)]
        return systems

    systems = selected_systems(raw)
    expected_queries = set(expected_parameters(workload))
    expected_units = {measurement_key(name, s["name"], v["label"], query)
                      for s in systems for v in s["variants"] for query in expected_queries}
    source_records, units, catalog, cases = [], {}, {}, {}
    identity = None
    snapshot_template = None
    for source in sources:
        source = source.resolve()
        request = json.loads((source / "request.json").read_text())
        selected = [w for w in request["workloads"] if w["name"] == name]
        if len(selected) != 1 or recipe(selected[0]) != recipe(workload):
            raise ValueError(f"workload definitions differ: {source}")
        if request["benchmark"] != raw["benchmark"] or selected_systems(request) != systems:
            raise ValueError(f"benchmark settings or variants differ: {source}")
        for setting in ("container_cpus", "container_memory", "preparation", "diva_backend"):
            if request["suite"].get(setting) != raw["suite"].get(setting):
                raise ValueError(f"resource setting differs: {setting}")
        state = json.loads((source / "run.json").read_text())
        if state["config_sha256"] != fingerprint(request):
            raise ValueError(f"saved request checksum differs: {source}")
        hardware = set()
        for launch in (source / "cloud").glob("*/launch.json"):
            value = json.loads(launch.read_text())
            hardware.add((value["instance_type"], value["ami"]))
        if len(hardware) != 1:
            raise ValueError(f"one consistent EC2 instance type and AMI are required: {source}")
        candidate = {k: state[k] for k in ("source", "binaries", "runtime")}
        candidate["hardware"] = sorted(hardware)
        if identity is not None and candidate != identity:
            raise ValueError("runner, binaries, runtime or EC2 hardware differ")
        identity = candidate
        queries = set(expected_parameters(selected[0]))
        if not queries <= expected_queries or queries & catalog.keys():
            raise ValueError("unexpected or overlapping dimension cases")
        snapshot = source / "workloads" / name
        source_catalog = json.loads((snapshot / "queries/query_catalog.json").read_text())["queries"]
        if set(source_catalog) != queries:
            raise ValueError(f"query snapshot differs from its configuration: {source}")
        catalog.update(source_catalog)
        for case in json.loads((snapshot / "cases.json").read_text())["cases"]:
            if case["case_id"] in cases:
                raise ValueError("overlapping dimension case metadata")
            cases[case["case_id"]] = case
        source_expected = {measurement_key(name, s["name"], v["label"], query)
                           for s in systems for v in s["variants"] for query in queries}
        found = set()
        for status in (source / "units").glob("*/status.json"):
            unit = json.loads(status.read_text())
            if unit.get("action") != "measure" or unit.get("workload") != name:
                continue
            key = status.parent.name
            payload = json.loads((status.parent / "payload.json").read_text())
            if (key not in source_expected or key in units
                    or key != measurement_key(name, payload["system"], payload["variant"], payload["query"])):
                raise ValueError(f"unexpected or duplicate unit: {status}")
            if not matches_unit_identity(payload, unit, status.parent) or not valid_measurement(status.parent, unit, source):
                raise ValueError(f"incomplete or corrupt measurement: {status}")
            for relative in unit.get("artifacts", {}):
                if not (source / relative).resolve().is_relative_to(source):
                    raise ValueError("artifact path escapes source")
            units[key] = (source, status.parent, unit)
            found.add(key)
        if found != source_expected:
            raise ValueError(f"missing measurement units: {source}")
        source_records.append({"path": str(source), "run_id": source.name,
                               "config_sha256": state["config_sha256"],
                               "dimension_cardinalities": selected[0]["params"]["sweep"]["values"]})
        snapshot_template = snapshot_template or snapshot
    if set(units) != expected_units or set(catalog) != expected_queries:
        raise ValueError("missing dimension partitions")

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for record in source_records:
        source = Path(record["path"])
        evidence = destination / "sources" / source.name
        evidence.mkdir(parents=True)
        for filename in ("request.json", "config.json", "run.json", "host.json", "container-image.json"):
            if (source / filename).exists():
                shutil.copy2(source / filename, evidence / filename)
        shutil.copytree(source / "cloud", evidence / "cloud")
        shutil.copytree(source / "workloads" / name, evidence / "workload")
    for source, unit, state in units.values():
        shutil.copytree(unit, destination / "units" / unit.name)
        for relative in state.get("artifacts", {}):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
    snapshot = destination / "workloads" / name
    shutil.copytree(snapshot_template, snapshot)
    for record in source_records:
        for query in (Path(record["path"]) / "workloads" / name / "queries").glob("*.sql"):
            shutil.copy2(query, snapshot / "queries" / query.name)
    definition = json.loads((snapshot / "definition.json").read_text())
    definition["sweep"]["values"] = workload["params"]["sweep"]["values"]
    write_json(snapshot / "definition.json", definition)
    write_json(snapshot / "cases.json", {"schema_version": 1, "cases": sorted(cases.values(), key=lambda c: c["dimension_cardinality"])})
    write_json(snapshot / "queries/query_catalog.json", {"schema_version": 1, "queries": catalog})
    metadata = json.loads((snapshot / "queries/metadata.json").read_text())
    metadata.update(case_count=len(cases), query_count=len(catalog), db_path="../benchmark.duckdb",
                    queries_dir=".", definition_path="../definition.json", cases_path="../cases.json",
                    query_catalog_path="query_catalog.json")
    write_json(snapshot / "queries/metadata.json", metadata)
    resolved = {**raw, "experiment_name": destination.name, "output_root": str(destination.parent)}
    write_json(destination / "config.json", resolved)
    write_json(destination / "request.json", raw)
    write_json(destination / "combined.json", {"kind": "dimension-case-partitions", "analysis_only": True,
               "parent_config_sha256": fingerprint(raw), "identity": identity, "sources": source_records})
    write_json(destination / "run.json", {"status": "combined", "config_sha256": fingerprint(raw), **identity})
    aggregate(destination)
    coverage = result_coverage(resolved, destination)
    write_json(destination / "coverage.json", coverage)
    if coverage["status"] != "complete":
        raise ValueError("combined case coverage is incomplete; see coverage.json")
