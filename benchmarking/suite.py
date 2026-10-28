"""Prepare, measure, analyze and export experiments in isolated, resumable units."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import dataclasses
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from benchmarking import datasets
from benchmarking.artifact_layout import ExperimentPaths
from benchmarking.duckdb_build import resolve_system
from benchmarking.duckdb_runner import (
    benchmark_workload, bootstrap_ducklake_workload, ensure_results_schema,
    label_to_path_component, read_remote_http_stats, start_remote_http_service,
    stop_remote_http_service, summarize_results,
)
from benchmarking.experiment_config import load_experiment_config, applies_to_workload
from benchmarking.experiment_runner import prepare_workloads
from benchmarking.manifests import load_manifest
from benchmarking.resources import size_bytes
from benchmarking.s3_data import data_scope
from benchmarking.suite_coverage import result_coverage
from benchmarking.suite_io import disk_preflight, fingerprint, lock, sha256, tree_bytes, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "run", "analyze", "export", "all")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DeadlineReached(InterruptedError):
    pass


def check_deadline() -> None:
    deadline = os.environ.get("BENCH_DEADLINE_EPOCH")
    if deadline and time.time() >= float(deadline):
        raise DeadlineReached("run deadline reached")


def checkpoint() -> None:
    """Pause between units while the host copies closed artifacts to S3."""
    directory = os.environ.get("BENCH_SYNC_DIR")
    if not directory:
        return
    root = Path(directory)
    token = uuid.uuid4().hex
    write_json(root / "request.json", {"token": token})
    while True:
        check_deadline()
        ack = root / "ack.json"
        if ack.exists() and json.loads(ack.read_text()).get("token") == token:
            return
        time.sleep(.25)


def query_units(raw: dict, workload: dict, run: Path) -> list[str | None]:
    if not raw["suite"].get("checkpoint_queries", False):
        return [None]
    snapshot = run / "workloads" / workload["name"] / "manifest.json"
    if not snapshot.exists():
        return []
    queries = [p.name for p in sorted(load_manifest(snapshot).queries_dir.glob("*.sql"))]
    if not queries:
        raise ValueError("prepared workload has no queries: " + workload["name"])
    return queries


def measurement_key(workload: str, system: str, variant: str, query: str | None) -> str:
    return unit_name("measure", workload, system, variant, *([query] if query else []))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_identity() -> str:
    return fingerprint({str(p.relative_to(ROOT)): sha256(p)
                        for p in sorted((ROOT / "benchmarking").rglob("*.py"))})


def unit_name(*parts: str) -> str:
    return fingerprint(parts)[:24]


def validate(raw: dict) -> None:
    suite = raw["suite"]
    for name in ("threads", "memory_limit", "max_temp_directory_size"):
        if name not in raw["benchmark"]:
            raise ValueError(f"benchmark.{name} is required")
    if raw["benchmark"]["threads"] <= 0:
        raise ValueError("threads must be positive")
    for name in ("timeout_seconds", "prepare_timeout_seconds", "disk_reserve_gib"):
        if suite[name] <= 0:
            raise ValueError(f"suite.{name} must be positive")
    names = [w["name"] for w in raw["workloads"]]
    names += [s["name"] for s in raw["systems"]]
    if any(not SAFE_NAME.fullmatch(n) for n in names):
        raise ValueError("system and workload names must be safe directory names")
    if len(set(w["name"] for w in raw["workloads"])) != len(raw["workloads"]):
        raise ValueError("workload names must be unique")
    missing = set(suite.get("required_workloads", [])) - {w["name"] for w in raw["workloads"]}
    if missing:
        raise ValueError("required workloads missing from configuration: " + ", ".join(sorted(missing)))
    if suite.get("required_figures") and not suite.get("paper_figures"):
        raise ValueError("required_figures needs paper_figures=true")
    if "parameter_sweep_overview" in suite.get("required_figures", []):
        from paper.figure_specs import PARAMETER_SWEEP_PANELS
        if not {p.workload for p in PARAMETER_SWEEP_PANELS}.issubset(w["name"] for w in raw["workloads"]):
            raise ValueError("parameter_sweep_overview requires all six sweep workloads")
    if suite.get("microbenchmarks_only") and not any(k in raw for k in ("microbenchmarks", "jpb_microbenchmarks")):
        raise ValueError("microbenchmarks_only needs configured microbenchmarks")
    if suite.get("compare_diva_backends") and ("microbenchmarks" not in raw or
                                              "diva" not in raw["microbenchmarks"].get("required_filters", [])):
        raise ValueError("compare_diva_backends requires microbenchmarks with Diva required")
    for workload in raw["workloads"]:
        if workload["kind"] == "external" and workload.get("dataset") not in raw["datasets"]:
            raise ValueError(f"{workload['name']} needs a dataset reference")
    for system in raw["systems"]:
        if system["kind"] != "binary":
            raise ValueError("suite images require prebuilt binary systems")
        labels = [v["label"] for v in system["variants"]]
        if len({label_to_path_component(label) for label in labels}) != len(labels):
            raise ValueError(f"duplicate variant in {system['name']}")


def measurement_workloads(raw: dict) -> list[dict]:
    return [] if raw["suite"].get("microbenchmarks_only") else raw["workloads"]


def preflight_microbenchmarks(config_path: Path, raw: dict, run: Path) -> dict:
    from benchmarking.benchmark_binary import binary_backend, select_benchmark_binary
    from benchmarking.microbenchmarks import resolve_microbenchmark_binaries, _available_filters
    from benchmarking.jpb_microbenchmarks import resolve_binary, available_filters
    if not any(key in raw for key in ("microbenchmarks", "jpb_microbenchmarks")):
        return {}
    config = load_experiment_config(config_path)
    report = {}
    for section in ("microbenchmarks", "jpb_microbenchmarks"):
        if section not in raw:
            continue
        spec = raw[section]
        system = resolve_system(next(s for s in config.systems if s.name == spec["system_name"]),
                                run / "systems" / spec["system_name"])
        binaries = (resolve_microbenchmark_binaries(system) if section == "microbenchmarks"
                    else {"simulation": resolve_binary(system)})
        for kind, path in binaries.items():
            filters = (_available_filters(path) if section == "microbenchmarks" else available_filters(path))
            required = set(spec.get("required_filters", [])) | {"prf", "prf_uncompressed"}
            if kind == "range":
                required.discard("bloom")
            if required - set(filters):
                raise ValueError(f"{path} lacks required filters: {sorted(required - set(filters))}")
            report[f"{section}/{kind}"] = {"path": str(path), "sha256": sha256(path),
                                          "diva_backend": binary_backend(path), "filters": filters}
        if raw["suite"].get("validate_diva"):
            for name in ("diva_bit_tests", "range_filter_adapter_tests"):
                if name in report:
                    continue
                path = select_benchmark_binary(next(iter(binaries.values())).parent / name)
                test = subprocess.run([str(path)], check=True, capture_output=True, text=True, timeout=120)
                report[name] = {"path": str(path), "sha256": sha256(path), "output": test.stdout.strip()}
    return report


def _resources(raw: dict, spill: Path, workload: dict | None = None) -> dict:
    spill.mkdir(parents=True, exist_ok=True)
    policy = {k: raw["benchmark"][k] for k in ("threads", "memory_limit", "max_temp_directory_size")}
    # Preparation has its own settings, capped by the suite's resource budget.
    overrides = dict(raw["suite"].get("preparation", {}))
    if workload:
        overrides.update(workload.get("params", {}).get("preparation", {}))
        overrides.update(workload.get("preparation", {}))
    for key in policy.keys() & overrides.keys():
        value = overrides[key]
        if key == "threads":
            if not isinstance(value, int) or value <= 0:
                raise ValueError("preparation.threads must be a positive integer")
            policy[key] = min(policy[key], value)
        else:
            if size_bytes(value) <= 0:
                raise ValueError(f"preparation.{key} must be positive")
            policy[key] = min((policy[key], value), key=size_bytes)
    return {**policy, "temp_directory": str(spill), "storage_compatibility_version": "v1.2.0"}


def _workload_identity(raw: dict, workload: dict) -> dict:
    identity = {"source": source_identity(), "workload": workload, "benchmark": raw["benchmark"],
                "preparation": raw["suite"].get("preparation", {}), "duckdb_python": duckdb.__version__}
    if "dataset" in workload:
        identity["dataset"] = datasets.recipe(raw["datasets"][workload["dataset"]])
        query_dir = ROOT / workload["source_dir"]
        identity["queries"] = {str(p.relative_to(query_dir)): sha256(p)
                               for p in sorted(query_dir.rglob("*.sql"))}
    if workload["kind"] == "string_joins":
        identity["airports"] = datasets.input_lock()["files"]["airports.csv"]
    return identity


def snapshot_workload(prepared: Path, destination: Path) -> None:
    """Keep query definitions and metadata on the durable results volume."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in prepared.rglob("*"):
        if path.is_file() and path.suffix in (".sql", ".json", ".csv") and path.stat().st_size < 16 * 1024**2:
            target = destination / path.relative_to(prepared)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest = json.loads((prepared / "manifest.json").read_text())
    for key, value in list(manifest.items()):
        if key.endswith("_path") or key.endswith("_dir") or key == "source_root":
            if value and Path(value).is_relative_to(prepared):
                manifest[key] = os.path.relpath(destination / Path(value).relative_to(prepared), destination)
        elif key.endswith("_paths"):
            manifest[key] = [os.path.relpath(destination / Path(p).relative_to(prepared), destination)
                             if Path(p).is_relative_to(prepared) else p for p in value]
    write_json(destination / "manifest.json", manifest)


def prepare_one(payload: dict) -> None:
    raw, workload = payload["config"], payload["workload"]
    cache, run = Path(payload["cache"]), Path(payload["run"])
    identity = _workload_identity(raw, workload)
    directory = cache / "workloads" / fingerprint(identity)
    marker = directory / "complete.json"
    with lock(cache / "locks" / (directory.name + ".lock")):
        existing = False
        if marker.is_file() and (directory / "manifest.json").is_file():
            saved = json.loads(marker.read_text())
            manifest = load_manifest(directory / "manifest.json")
            data = (manifest.parquet_catalog_path if manifest.storage_mode == "ducklake"
                    else manifest.remote_parquet_catalog_path if manifest.storage_mode == "remote_http_parquet"
                    else manifest.db_path)
            existing = data is not None and data.is_file()
            if existing:
                existing = all((directory / p).is_file() and sha256(directory / p) == digest
                               for p, digest in saved.get("data_files", {}).items())
            if existing and "dataset" in workload:
                spec = raw["datasets"][workload["dataset"]]
                existing = datasets.valid_dataset(datasets.dataset_path(cache, spec), datasets.recipe(spec))
        if not existing:
            if payload.get("existing_only"):
                raise FileNotFoundError(f"workload must be prepared first: {workload['name']}")
            progress = directory / "preparation-identity.json"
            density = workload.get("params", {}).get("sweep", {}).get("cardinality_mode") == "fixed_selected_build_and_probe"
            resume_density = (density and not marker.exists() and progress.is_file()
                              and json.loads(progress.read_text()) == identity
                              and (directory / "benchmark.duckdb").is_file())
            if directory.exists() and not resume_density:
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(progress, identity)
            write_json(directory / "preparation-resources.json", json.loads(os.environ.get("BENCH_PREPARATION_CONFIG", "{}")))
            resolved_workload = json.loads(json.dumps(workload))
            if "dataset" in workload:
                spec = raw["datasets"][workload["dataset"]]
                dataset = datasets.prepare_dataset(cache, spec)
                source = directory / "source"
                source.mkdir()
                shutil.copytree(ROOT / workload["source_dir"] / "queries", source / "queries")
                write_json(source / "source.json", {
                    "benchmark_name": workload["name"], "queries_dir": "queries",
                    "database_path": str(dataset / "benchmark.duckdb"), "database_handling": "reuse",
                })
                resolved_workload["source_dir"] = str(source)
                shutil.copy2(dataset / "complete.json", directory / "dataset.json")
            elif workload["kind"] == "string_joins":
                resolved_workload["params"]["airport_csv_path"] = str(datasets.airports(cache))
            worker_config = {**raw, "workloads": [resolved_workload],
                             "microbenchmarks": None, "jpb_microbenchmarks": None}
            # Other workloads can appear in system selection lists; preparation
            # only needs this workload and no system resolution.
            worker_config["systems"] = [{k: v for k, v in s.items() if k != "workload_names"}
                                         for s in raw["systems"]]
            for system in worker_config["systems"]:
                system["variants"] = [{k: v for k, v in variant.items() if k != "workload_names"}
                                      for variant in system["variants"]]
            worker_config.pop("microbenchmarks")
            worker_config.pop("jpb_microbenchmarks")
            config_path = directory / "preparation-config.json"
            write_json(config_path, worker_config)
            config = load_experiment_config(config_path)
            class SinglePaths:
                def workload_dir(self, _name):
                    return directory
            # A manifest without complete.json may precede a crash during hashing.
            # Resume through the generator's transactional progress validation.
            if resume_density:
                (directory / "manifest.json").unlink(missing_ok=True)
            manifest = prepare_workloads(config, SinglePaths(), resume_preparation=resume_density)[0]
            data_files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
                          if p.is_file() and p.suffix in (".duckdb", ".parquet")}
            write_json(marker, {"identity": identity, "status": "complete", "prepared_at": now(),
                                "data_files": data_files})
        snapshot_workload(directory, run / "workloads" / workload["name"])
        write_json(Path(payload["unit_dir"]) / "prepared.json", {"manifest": str(directory / "manifest.json")})


def worker(payload: dict) -> None:
    raw = payload["config"]
    os.environ["BENCH_PREPARATION_CONFIG"] = json.dumps(_resources(
        raw, Path(payload["spill"]), payload.get("workload") if payload["action"] == "prepare" else None))
    unit_dir = Path(payload["unit_dir"])
    action = payload["action"]
    if action == "prepare":
        prepare_one(payload)
        return
    config = load_experiment_config(Path(payload["run"]) / "config.json")
    system_config = next(s for s in config.systems if s.name == payload["system"])
    extensions = frozenset()
    if action == "measure":
        manifest = load_manifest(Path(payload["manifest"]))
        manifest = dataclasses.replace(manifest, ducklake_s3_url=payload.get("s3_base_url"))
        if payload.get("query"):
            # Keep original query names and repetition/warmup semantics.
            selected = unit_dir / "queries"
            selected.mkdir(exist_ok=True)
            for old in selected.glob("*.sql"):
                old.unlink()
            shutil.copy2(manifest.queries_dir / payload["query"], selected / payload["query"])
            # load_queries reads this catalog beside the selected SQL. Without
            # it checkpointed rows lose sweep_kind/value and cannot be plotted.
            catalog = manifest.queries_dir / "query_catalog.json"
            if catalog.is_file():
                shutil.copy2(catalog, selected / catalog.name)
            else:
                (selected / "query_catalog.json").unlink(missing_ok=True)
            manifest = dataclasses.replace(manifest, queries_dir=selected)
        if manifest.storage_mode == "ducklake":
            extensions = frozenset(("ducklake", "httpfs") if manifest.ducklake_http_latency_ms is not None else ("ducklake",))
            if manifest.ducklake_data_source == "s3":
                extensions = frozenset(("ducklake", "httpfs", "aws", "json"))
    system = resolve_system(system_config, Path(payload["run"]) / "systems" / system_config.name,
                            required_extensions=extensions)
    write_json(unit_dir / "system.json", {k: str(v) if isinstance(v, Path) else v
                                         for k, v in dataclasses.asdict(system).items()})
    if action == "diva_compare":
        from benchmarking.diva_validation import validate_backends
        validate_backends(config.microbenchmarks, system, Path(payload["run"]) / "diva-comparison", checkpoint)
        return
    if action == "micro":
        from benchmarking.microbenchmarks import run_microbenchmarks
        run_microbenchmarks(config.microbenchmarks, system, Path(payload["run"]),
                           checkpoint_cases=raw["suite"].get("checkpoint_microbenchmarks", False),
                           checkpoint_callback=checkpoint)
        return
    if action == "jpb_micro":
        from benchmarking.jpb_microbenchmarks import run_jpb_microbenchmarks
        run_jpb_microbenchmarks(config, system, Path(payload["run"]),
                               checkpoint_cases=raw["suite"].get("checkpoint_microbenchmarks", False),
                               checkpoint_callback=checkpoint)
        return
    benchmark = dataclasses.replace(config.benchmark, temp_directory=payload["spill"])
    variant = next(v for v in system_config.variants if v.label == payload["variant"])
    service = start_remote_http_service(manifest)
    try:
        if manifest.storage_mode == "ducklake":
            # Each isolated worker may get a different HTTP port. Rebuild the
            # small metadata catalog against the same immutable Parquet files.
            manifest.ducklake_metadata_path.unlink(missing_ok=True)
            Path(str(manifest.ducklake_metadata_path) + ".wal").unlink(missing_ok=True)
        bootstrap_ducklake_workload(system.binary_path, manifest, service, benchmark)
        if manifest.ducklake_data_source == "s3":
            shutil.copy2(manifest.artifact_dir / "ducklake_catalog_validation.json", unit_dir / "ducklake_catalog_validation.json")
        # RPT+ must also be able to append to the result database created by Python.
        with duckdb.connect(str(unit_dir / "results.duckdb"),
                            config={"storage_compatibility_version": "v1.2.0"}) as con:
            con.execute("CREATE TABLE suite_format(version INTEGER)")
            con.execute("DROP TABLE suite_format")
        suffix = unit_dir.name if payload.get("query") else ""
        output = benchmark_workload(config.experiment_name, manifest, system, (variant,), benchmark,
                                    unit_dir / "results.duckdb", Path(payload["run"]) / "sql" / suffix,
                                    Path(payload["run"]) / "profiles" / suffix, service)
        if raw["suite"].get("verify_http_requests", False):
            if service is None:
                raise RuntimeError("HTTP smoke verification requires an HTTP workload")
            stats = read_remote_http_stats(service)
            if stats["requests"] <= 0 or stats["bytes_sent"] <= 0:
                raise RuntimeError("HTTP smoke verification observed no data transfer; choose a query that scans data")
        with duckdb.connect(str(unit_dir / "results.duckdb"), read_only=True) as con:
            timings = con.execute("SELECT count(*) FROM timing_runs").fetchone()[0]
            profiles = con.execute("SELECT count(*) FROM profile_runs").fetchone()[0]
        if output.query_count == 0 or timings != output.query_count * benchmark.repetitions:
            raise RuntimeError(f"incomplete timing output: {timings} rows")
        if benchmark.collect_profiles and profiles != output.query_count:
            raise RuntimeError(f"incomplete profiles: {profiles} rows")
        if manifest.ducklake_data_source == "s3":
            from benchmarking.ducklake_validation import validate_remote_reads
            validate_remote_reads(unit_dir / "results.duckdb", manifest, unit_dir / "s3-read-validation.json")
    finally:
        stop_remote_http_service(service)


def process_metrics(group: int) -> dict:
    """Linux process-group RSS/IO, including CLI children; no psutil dependency."""
    rss = read = written = 0
    available = Path("/proc/self/stat").exists()
    if available:
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) != group:  # pgrp, after state and ppid
                    continue
                rss += int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
                counters = dict(line.split(": ", 1) for line in (p / "io").read_text().splitlines())
                read += int(counters["read_bytes"])
                written += int(counters["write_bytes"])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return {"rss_bytes": rss if available else None,
            "read_bytes": read if available else None, "write_bytes": written if available else None}


def valid_measurement(directory: Path, state: dict, run: Path) -> bool:
    database = directory / "results.duckdb"
    return (state.get("status") == "complete" and database.is_file()
            and sha256(database) == state.get("sha256")
            and all((run / path).is_file() and sha256(run / path) == digest
                    for path, digest in state.get("artifacts", {}).items()))


def unit_identity(payload: dict) -> str:
    ignored = {"existing_only"}
    workload = payload.get("workload")
    storage = workload.get("params", {}).get("storage", {}) if isinstance(workload, dict) else {}
    if payload["action"] == "measure" and storage.get("kind") == "ducklake" and storage.get("data_source") == "s3":
        # The host regenerates the same configured data in a fresh attempt prefix.
        # Its transport location is not part of the measurement's identity.
        ignored.add("s3_base_url")
    return fingerprint({k: v for k, v in payload.items() if k not in ignored})


def matches_unit_identity(payload: dict, previous: dict, directory: Path) -> bool:
    identity = unit_identity(payload)
    if previous.get("identity") == identity:
        return True
    # Older images included the attempt-specific S3 URL. Authenticate their
    # saved payload against that identity before comparing normalized inputs.
    try:
        saved = json.loads((directory / "payload.json").read_text())
        legacy = fingerprint({k: v for k, v in saved.items() if k != "existing_only"})
        return previous.get("identity") == legacy and unit_identity(saved) == identity
    except (OSError, ValueError, AttributeError, KeyError):
        return False


def run_unit(payload: dict, timeout: float) -> bool:
    check_deadline()
    directory = Path(payload["unit_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "status.json"
    identity = unit_identity(payload)
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get("status") == "complete" and matches_unit_identity(payload, previous, directory):
        if payload["action"] == "measure":
            if valid_measurement(directory, previous, Path(payload["run"])):
                return True
        elif payload["action"] in ("micro", "jpb_micro"):
            # Microbenchmark manifests carry checksums for all their output files.
            from benchmarking.microbenchmarks import load_microbenchmark_manifest
            from benchmarking.jpb_microbenchmarks import load_jpb_manifest
            config = load_experiment_config(Path(payload["run"]) / "config.json")
            try:
                if payload["action"] == "micro":
                    load_microbenchmark_manifest(Path(payload["run"]))
                else:
                    load_jpb_manifest(Path(payload["run"]), config)
                return True
            except (OSError, ValueError, KeyError):
                pass  # Retry this unit instead of trusting corrupt/missing outputs.
        elif payload["action"] == "diva_compare":
            from benchmarking.diva_validation import valid_comparison
            if valid_comparison(Path(payload["run"]) / "diva-comparison"):
                return True
    attempt = previous.get("attempt", 0) + 1
    for path in (directory / "results.duckdb", directory / "results.duckdb.wal"):
        path.unlink(missing_ok=True)
    (directory / "process-rusage.json").unlink(missing_ok=True)
    spill = Path(payload["spill"])
    if spill.exists():
        shutil.rmtree(spill)
    spill.mkdir(parents=True)
    write_json(directory / "payload.json", payload)
    state = {"status": "running", "identity": identity, "action": payload["action"],
             "workload": payload.get("workload", {}).get("name") if isinstance(payload.get("workload"), dict) else payload.get("workload"),
             "system": payload.get("system"), "variant": payload.get("variant"),
             "attempt": attempt, "started_at": now()}
    write_json(state_path, state)
    start = time.monotonic()
    print(f"{state['action']}: {state.get('workload') or state.get('system')}", flush=True)
    metrics = {"peak_rss_bytes": None, "peak_spill_bytes": 0, "max_observed_read_bytes": None,
               "max_observed_write_bytes": None, "sampling_interval_seconds": 0.5}
    with (directory / f"attempt-{attempt}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "benchmarking.suite", "_worker",
                                    str(directory / "payload.json")],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        failure = None
        def terminate(signum, _frame):
            raise InterruptedError(f"received signal {signum}")
        handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while process.poll() is None:
                check_deadline()
                values = process_metrics(process.pid)
                for src, dest in (("rss_bytes", "peak_rss_bytes"), ("read_bytes", "max_observed_read_bytes"),
                                  ("write_bytes", "max_observed_write_bytes")):
                    if values[src] is not None:
                        metrics[dest] = max(metrics[dest] or 0, values[src])
                metrics["peak_spill_bytes"] = max(metrics["peak_spill_bytes"], tree_bytes(spill))
                if time.monotonic() - start > timeout:
                    failure = "timeout"
                    break
                if shutil.disk_usage(spill).free < payload["config"]["suite"]["disk_reserve_gib"] * 1024**3:
                    failure = "disk reserve exhausted"
                    break
                time.sleep(0.5)
        except BaseException as exc:
            failure = str(exc)
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state.update({"status": "failed" if failure or process.returncode else "complete",
                          "failure": failure, "exit_code": process.returncode,
                          "finished_at": now(), "elapsed_seconds": time.monotonic() - start,
                          "metrics": metrics})
            if (directory / "process-rusage.json").exists():
                state["process_rusage"] = json.loads((directory / "process-rusage.json").read_text())
            if state["status"] == "complete" and (directory / "results.duckdb").is_file():
                state["sha256"] = sha256(directory / "results.duckdb")
                run = Path(payload["run"])
                workload, system = payload["workload"]["name"], payload["system"]
                variant = label_to_path_component(payload["variant"])
                suffix = directory.name if payload.get("query") else ""
                artifacts = list((run / "profiles" / suffix / workload / system).rglob("*.json"))
                artifacts += [run / "sql" / suffix / workload / system / (variant + ".sql")]
                state["artifacts"] = {str(p.relative_to(run)): sha256(p) for p in artifacts if p.is_file()}
            write_json(state_path, state)
            shutil.rmtree(spill)
    checkpoint()
    return state["status"] == "complete"


def aggregate(run: Path) -> None:
    target = run / "results.next.duckdb"
    target.unlink(missing_ok=True)
    Path(str(target) + ".wal").unlink(missing_ok=True)
    ensure_results_schema(target)
    with duckdb.connect(str(target)) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        for status in sorted((run / "units").glob("*/status.json")):
            state = json.loads(status.read_text())
            source = status.parent / "results.duckdb"
            if state["status"] != "complete" or state["action"] != "measure":
                continue
            if not valid_measurement(status.parent, state, run):
                raise ValueError(f"completed result checksum changed: {source}")
            con.execute(f"ATTACH {datasets.literal(source)} AS unit (READ_ONLY)")
            for table in tables:
                con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM unit."{table}"')
            con.execute("DETACH unit")
        # Profiles are durable artifacts, and must still work after export/move.
        for original, in con.execute("SELECT DISTINCT profile_path FROM profile_runs").fetchall():
            if "/profiles/" in original:
                con.execute("UPDATE profile_runs SET profile_path=? WHERE profile_path=?",
                            [str(run / "profiles" / original.split("/profiles/", 1)[1]), original])
        if (run / "combined.json").exists():
            for table in tables:
                con.execute(f'UPDATE "{table}" SET experiment_name=?', [run.name])
    target.replace(run / "results.duckdb")


def analyze(run: Path, *, paper: bool) -> None:
    config = json.loads((run / "config.json").read_text())
    if "groups" in config:
        from benchmarking.paper_suite import analyze as analyze_paper
        return analyze_paper(run, config)
    aggregate(run)
    summary = summarize_results(run / "results.duckdb", config["experiment_name"])
    states = [json.loads(p.read_text()) for p in sorted((run / "units").glob("*/status.json"))]
    summary["failed_units"] = [s for s in states if s["status"] != "complete"]
    expected = [measurement_key(w["name"], s["name"], v["label"], q)
                for w in measurement_workloads(config) for s in config["systems"]
                if applies_to_workload(s, w["name"]) for v in s["variants"]
                if applies_to_workload(v, w["name"])
                for q in query_units(config, w, run)]
    expected += [unit_name(action) for section, action in
                 (("microbenchmarks", "micro"), ("jpb_microbenchmarks", "jpb_micro")) if section in config]
    if config["suite"].get("compare_diva_backends"):
        expected.append(unit_name("diva_compare"))
    summary["missing_units"] = [key for key in expected if not (run / "units" / key / "status.json").exists()]
    summary["missing_units"] += ["prepare:" + w["name"] for w in measurement_workloads(config)
                                  if not (run / "workloads" / w["name"] / "manifest.json").exists()]
    summary["status"] = "partial" if summary["failed_units"] or summary["missing_units"] else "complete"
    coverage = result_coverage(config, run)
    write_json(run / "coverage.json", coverage)
    summary["coverage_status"] = coverage["status"]
    if coverage["status"] != "complete":
        summary["status"] = "partial"
    write_json(run / "summary.json", summary)
    with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
        for table in ("timing_runs", "profile_runs"):
            con.execute(f"COPY {table} TO ? (HEADER, DELIMITER ',')", [str(run / (table + ".csv"))])
        rows = con.execute("""SELECT workload_name, system_name, variant_label, median(elapsed_ns)/1e6
                              FROM timing_runs GROUP BY 1,2,3 ORDER BY 1,2,3""").fetchall()
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, max(3, len(rows) * .25)))
        axis.barh([f"{w} / {s} / {v}" for w, s, v, _ in rows], [r[3] for r in rows])
        axis.set_xlabel("Median query time (ms; descriptive overview)")
        axis.set_title("Completed units" if summary["status"] == "complete" else "Partial run: completed units only")
        axis.set_xscale("log")
        figure.tight_layout()
        figure.savefig(run / "overview.pdf")
        plt.close(figure)
    if paper:
        if config["suite"].get("required_figures") and summary["status"] != "complete":
            raise ValueError("required paper figures need complete measurements; see coverage.json and summary.json")
        from paper.figures.build_all import build_figures
        build_figures(run, None)
        figures = json.loads((run / "paper/summary.json").read_text())["figures"]
        missing = [name for name in config["suite"].get("required_figures", [])
                   if name not in figures or not Path(figures[name]).is_file()]
        if missing:
            raise ValueError("required figures missing: " + ", ".join(missing))
    if coverage["status"] != "complete":
        raise ValueError("measurement coverage is incomplete; see coverage.json")
    write_json(run / "analysis.json", {"status": "complete", "at": now(), "paper_figures": paper})


def export(run: Path) -> Path:
    # Include failed-unit evidence too. Datasets/spill/binaries live outside run.
    files = [p for p in sorted(run.rglob("*")) if p.is_file() and p.name not in (".lock", "SHA256SUMS")]
    (run / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(run)}\n" for p in files))
    archive = run.parent / (run.name + ".tar.gz")
    temporary = archive.with_suffix(".gz.part")
    with tarfile.open(temporary, "w:gz") as tar:
        tar.add(run, arcname=run.name, filter=lambda info: None if info.name.endswith("/.lock") else info)
    temporary.replace(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    print(archive, flush=True)
    return archive


def execute(config_path: Path, stage: str, run: Path, cache: Path, resume: bool) -> int:
    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        from benchmarking.suite_plan import load_plan
        raw = load_plan(config_path)
        if "groups" in raw:
            from benchmarking.paper_suite import execute as execute_paper
            return execute_paper(raw, stage, run, cache, resume)
        return _execute(config_path, stage, run, cache, resume)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _execute(config_path: Path, stage: str, run: Path, cache: Path, resume: bool,
             *, package: bool = True, cache_locked: bool = False) -> int:
    raw = json.loads(config_path.read_text())
    validate(raw)
    if "diva_backend" in raw["suite"]:
        os.environ["BENCH_DIVA_BACKEND"] = raw["suite"]["diva_backend"]
    run, cache = run.resolve(), cache.resolve()
    if (run / "combined.json").exists():
        raise ValueError("combined results support analysis/export only; resume the original jobs")
    run.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with lock(run / ".lock"), (nullcontext() if cache_locked else lock(cache / ".suite.lock")):
        request = fingerprint(raw)
        runtime = {"python": sys.version, "duckdb_python": duckdb.__version__,
                   "image_id": json.loads(os.environ.get("BENCH_IMAGE_METADATA", "{}")).get("id"),
                   "microbenchmark_binaries": preflight_microbenchmarks(config_path, raw, run)}
        state_path = run / "run.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if not resume:
                raise ValueError(f"run already exists: {run}; use --resume")
            if previous["config_sha256"] != request or previous["source"] != source_identity():
                raise ValueError("config or runner code changed; start a new run")
            if previous.get("runtime") != runtime:
                raise ValueError("image or Python runtime changed; start a new run")
        elif any(p.name != ".lock" for p in run.iterdir()):
            raise ValueError(f"refusing nonempty run directory without run.json: {run}")
        binaries = {s["name"]: sha256(Path(s["binary_path"])) for s in raw["systems"]}
        if state_path.exists() and previous["binaries"] != binaries:
            raise ValueError("binary changed; start a new run")
        state = {"config_sha256": request, "source": source_identity(), "binaries": binaries, "runtime": runtime,
                 "status": "running", "stage": stage, "updated_at": now(),
                 "platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version, "duckdb_python": duckdb.__version__}
        write_json(state_path, state)
        if runtime["microbenchmark_binaries"]:
            write_json(run / "capabilities.json", runtime["microbenchmark_binaries"])
        session = run / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        write_json(session / "start.json", {**state, "deadline": os.environ.get("BENCH_DEADLINE_EPOCH")})
        if "BENCH_IMAGE_METADATA" in os.environ:
            write_json(run / "container-image.json", json.loads(os.environ["BENCH_IMAGE_METADATA"]))
        write_json(run / "host.json", {
            "cpu_count_visible": os.cpu_count(), "uname": list(platform.uname()),
            "linux": {p: Path(p).read_text() for p in (
                "/proc/cpuinfo", "/proc/meminfo", "/sys/fs/cgroup/cpu.max",
                "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.swap.max")
                if Path(p).is_file()},
        })
        shutil.copy2(run / "host.json", session / "host.json")
        resolved = {**copy.deepcopy(raw), "experiment_name": run.name, "output_root": str(run.parent), "overwrite_output": False}
        for workload in resolved["workloads"]:
            if "source_dir" in workload:
                workload["source_dir"] = str((ROOT / workload["source_dir"]).resolve())
        write_json(run / "config.json", resolved)
        write_json(run / "request.json", raw)
        if (ROOT / "docker/provenance.json").exists():
            shutil.copy2(ROOT / "docker/provenance.json", run / "image-provenance.json")
        failures = []
        try:
            if stage in ("prepare", "run", "all"):
                retained_datasets = set()
                for workload in measurement_workloads(raw):
                    key = unit_name("prepare", workload["name"])
                    base = {"config": resolved, "cache": str(cache), "run": str(run),
                            "workload": workload, "action": "prepare",
                            "unit_dir": str(run / "units" / key), "spill": str(cache / "spill" / key)}
                    # Completed measurements don't need their evicted data again.
                    measured = [(s, v) for s in raw["systems"]
                                if applies_to_workload(s, workload["name"])
                                for v in s["variants"] if applies_to_workload(v, workload["name"])]
                    def measure_payload(s, v, query):
                        key = measurement_key(workload["name"], s["name"], v["label"], query)
                        return {**base, "action": "measure", "system": s["name"], "variant": v["label"],
                                **({"query": query} if query else {}),
                                "unit_dir": str(run / "units" / key), "spill": str(cache / "spill" / key)}
                    def complete_measurement(s, v, query):
                        unit = Path(measure_payload(s, v, query)["unit_dir"])
                        if not (unit / "status.json").exists() or not (unit / "results.duckdb").exists():
                            return False
                        status = json.loads((unit / "status.json").read_text())
                        return valid_measurement(unit, status, run)
                    queries = query_units(raw, workload, run)
                    all_done = bool(measured and queries) and all(complete_measurement(s, v, q)
                                                                 for s, v in measured for q in queries)
                    if stage != "prepare" and all_done:
                        continue
                    disk_preflight(cache, workload.get("estimated_peak_gib", 1), raw["suite"]["disk_reserve_gib"])
                    base["existing_only"] = stage == "run"
                    if not run_unit(base, raw["suite"]["prepare_timeout_seconds"]):
                        failures.append(f"prepare:{workload['name']}")
                        if "dataset" in workload:
                            retained_datasets.add(workload["dataset"])
                        continue
                    prepared = json.loads((Path(base["unit_dir"]) / "prepared.json").read_text())
                    if stage == "prepare":
                        continue
                    workload_complete = True
                    with data_scope(load_manifest(Path(prepared["manifest"])), run) as s3_url:
                        for system, variant, query in [(s, v, q) for s, v in measured
                                                         for q in query_units(raw, workload, run)]:
                            payload = {**measure_payload(system, variant, query), "manifest": prepared["manifest"],
                                       "s3_base_url": s3_url}
                            if not run_unit(payload, raw["suite"]["timeout_seconds"]):
                                failures.append(f"{workload['name']}:{system['name']}:{variant['label']}")
                                workload_complete = False
                    if not workload_complete and "dataset" in workload:
                        retained_datasets.add(workload["dataset"])
                    if stage == "all" and workload_complete and raw["suite"].get("evict_datasets", True):
                        directory = Path(prepared["manifest"]).parent
                        shutil.rmtree(directory)
                        if "dataset" in workload:
                            # Recreate for later workload aliases only when necessary;
                            # retaining this shared base saves generation for intro plots.
                            position = raw["workloads"].index(workload)
                            if workload["dataset"] not in retained_datasets and not any(
                                w.get("dataset") == workload["dataset"] for w in raw["workloads"][position+1:]
                            ):
                                dataset = datasets.dataset_path(cache, raw["datasets"][workload["dataset"]])
                                shutil.rmtree(dataset, ignore_errors=True)
                if stage != "prepare":
                    actions = [("microbenchmarks", "micro"), ("jpb_microbenchmarks", "jpb_micro")]
                    if raw["suite"].get("compare_diva_backends"):
                        actions.append(("microbenchmarks", "diva_compare"))
                    for section, action in actions:
                        if section not in raw:
                            continue
                        key = unit_name(action)
                        payload = {"config": resolved, "cache": str(cache), "run": str(run), "action": action,
                                   "system": raw[section]["system_name"], "unit_dir": str(run / "units" / key),
                                   "spill": str(cache / "spill" / key)}
                        if not run_unit(payload, raw["suite"]["timeout_seconds"]):
                            failures.append(action)
                    aggregate(run)
            if stage in ("analyze", "all"):
                try:
                    analyze(run, paper=raw["suite"].get("paper_figures", False))
                except Exception as exc:
                    failures.append(f"analysis: {exc}")
                    write_json(run / "analysis.json", {"status": "failed", "error": str(exc)})
        except (InterruptedError, KeyboardInterrupt) as exc:
            failures.append(str(exc) or "interrupted")
            state["interruption"] = "deadline" if isinstance(exc, DeadlineReached) else "signal"
            # Finalize outside the measurement deadline; the host owns the
            # independent finalization deadline and shutdown watchdog.
            aggregate(run)
        except BaseException as exc:
            state.update(status="failed", error=str(exc), updated_at=now())
            write_json(state_path, state)
            if package:
                export(run)
            raise
        state.update(status="partial" if failures else "complete", failures=failures, updated_at=now())
        write_json(state_path, state)
        write_json(session / "finish.json", state)
        if package and stage in ("export", "all"):
            export(run)
        return 1 if failures else 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        payload = json.loads(Path(sys.argv[2]).read_text())
        try:
            worker(payload)
        finally:
            usage = {}
            for label, who in (("worker", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN)):
                stats = resource.getrusage(who)
                usage[label] = {"max_rss_bytes": stats.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                                "user_seconds": stats.ru_utime, "system_seconds": stats.ru_stime,
                                "input_blocks": stats.ru_inblock, "output_blocks": stats.ru_oublock}
            write_json(Path(payload["unit_dir"]) / "process-rusage.json", usage)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/work"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Analysis/export can be repeated on an extracted archive without binaries
    # or source datasets, even on a different computer.
    if args.stage in ("analyze", "export") and args.config is None:
        with lock(args.run / ".lock"):
            if args.stage == "analyze":
                raw = json.loads((args.run / "config.json").read_text())
                analyze(args.run.resolve(), paper=raw["suite"].get("paper_figures", False))
            else:
                export(args.run.resolve())
        return 0
    if args.config is None:
        parser.error("--config is required for prepare/run/all")
    return execute(args.config, args.stage, args.run, args.cache, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
