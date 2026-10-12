#!/usr/bin/env python3
"""Interleave PRF libraries on existing paper workloads, without modifying their databases.

Example:
  python3 benchmarking/compare_prf_candidates.py --baseline /tmp/baseline.dylib \
      --candidate blocks=/tmp/blocks.dylib --workloads job string jpb --output out/comparison.json

Each library has a persistent worker. Queries execute sequentially in randomized
mode order; only the C API query call is timed (including planning). Warmups,
result extraction, IPC and profiling are excluded. Every library is measured with
both uncompressed and compressed PRF.
"""

import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import statistics
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "job": "join-order-benchmark",
    "tpch": "tpch-skew",
    "jpb": "join-pruning-workload",
    "string": "string-joins",
    "ssb": "ssb-ordered",
}
SETUP = """
SET enable_prefix_range_filter=true;
SET enable_join_min_max_filter_pushdown=true;
SET enable_join_bloom_filter_pushdown=true;
SET enable_perfect_hash_join_filter_pushdown=false;
SET dynamic_or_filter_threshold=0;
SET disabled_optimizers='compressed_materialization';
"""


def load_api():
    spec = importlib.util.spec_from_file_location(
        "prf_api", ROOT / "duckdb/scripts/benchmark_join_filter_profiling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(library, threads, pipe):
    db = None
    try:
        api = load_api()
        db = api.Database(library)
        lib = db.lib
        result_ptr = ctypes.POINTER(api.Result)
        for name in ("duckdb_column_count", "duckdb_row_count"):
            function = getattr(lib, name)
            function.argtypes = [result_ptr]
            function.restype = ctypes.c_uint64
        lib.duckdb_column_type.argtypes = [result_ptr, ctypes.c_uint64]
        lib.duckdb_column_type.restype = ctypes.c_int
        pipe.send({"ready": True})
        while True:
            request = pipe.recv()
            if request is None:
                break
            if "database" in request:
                db.close()
                db = api.Database(library)
                escaped = request["database"].replace("'", "''")
                db.query(f"ATTACH '{escaped}' AS benchmark (READ_ONLY); USE benchmark;")
                db.query(SETUP + f"SET threads={threads};")
                pipe.send({"ready": True})
                continue
            db.query("SET enable_prefix_range_filter_compression=" +
                     ("true" if request["compressed"] else "false") + ";")
            if request.get("profile"):
                _, profile = db.query("EXPLAIN (ANALYZE, FORMAT JSON) " + request["sql"], 1)
                pipe.send({"profile": json.loads(profile)})
                continue
            result = api.Result()
            try:
                start = time.perf_counter_ns()
                status = lib.duckdb_query(db.connection, request["sql"].encode(), ctypes.byref(result))
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if status:
                    raise RuntimeError(lib.duckdb_result_error(ctypes.byref(result)).decode())
                values = None
                if request.get("verify"):
                    rows = []
                    for row in range(lib.duckdb_row_count(ctypes.byref(result))):
                        cells = []
                        for col in range(lib.duckdb_column_count(ctypes.byref(result))):
                            value = lib.duckdb_value_varchar(ctypes.byref(result), col, row)
                            try:
                                cells.append(ctypes.string_at(value).decode() if value else None)
                            finally:
                                if value:
                                    lib.duckdb_free(value)
                        rows.append(cells)
                    values = {"types": [lib.duckdb_column_type(ctypes.byref(result), col)
                                        for col in range(lib.duckdb_column_count(ctypes.byref(result)))], "rows": rows}
                pipe.send({"ms": elapsed, "values": values})
            finally:
                lib.duckdb_destroy_result(ctypes.byref(result))
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        if db is not None:
            db.close()


class Worker:
    def __init__(self, library, threads):
        ctx = multiprocessing.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=worker, args=(str(library), threads, child))
        self.process.start()
        child.close()
        self.receive()

    def receive(self):
        if not self.pipe.poll(300):
            raise TimeoutError("PRF worker did not respond within 300 seconds")
        result = self.pipe.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def request(self, **request):
        self.pipe.send(request)
        return self.receive()

    def close(self):
        if self.process.is_alive():
            self.pipe.send(None)
        self.process.join(10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.pipe.close()


def equivalent(left, right):
    if left["types"] != right["types"] or len(left["rows"]) != len(right["rows"]):
        return False
    # SQL permits arbitrary order within ORDER BY ties (for example TPC-H Q18).
    if Counter(map(tuple, left["rows"])) == Counter(map(tuple, right["rows"])):
        return True
    for a, b in zip(left["rows"], right["rows"]):
        if len(a) != len(b):
            return False
        for column_type, x, y in zip(left["types"], a, b):
            if x == y:
                continue
            try:
                # FLOAT/DOUBLE aggregates can vary with parallel reduction order.
                if column_type in (10, 11) and math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-8):
                    continue
            except (TypeError, ValueError):
                pass
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="LABEL=LIBRARY")
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=["job", "tpch", "jpb", "string", "ssb"])
    parser.add_argument("--queries", nargs="*", help="Optional query filenames for exploratory runs")
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmups < 1 or args.threads < 1:
        parser.error("Require repetitions>=3, warmups>=1 and threads>=1")
    libraries = {"baseline": args.baseline.resolve(strict=True)}
    for candidate in args.candidate:
        if "=" not in candidate:
            parser.error("Candidates must have the form LABEL=LIBRARY")
        label, path = candidate.split("=", 1)
        if (not label or label in libraries or label in ("original", "uncompressed")
                or label.endswith("-uncompressed")):
            parser.error(f"Duplicate, reserved or empty candidate label: {label}")
        libraries[label] = Path(path).resolve(strict=True)
    modes = {"uncompressed": ("baseline", False), "original": ("baseline", True)}
    modes.update({label: (label, True) for label in libraries if label != "baseline"})
    modes.update({label + "-uncompressed": (label, False) for label in libraries if label != "baseline"})
    data = {"metadata": {"platform": platform.platform(), "threads": args.threads,
                         "repetitions": args.repetitions, "warmups": args.warmups, "seed": args.seed,
                         "libraries": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                                       for k, v in libraries.items()},
                         "timing": "Sequential, shuffled modes per repetition; C API planning and execution only",
                         "verification": "All output cells; COUNT(*) substitute for nondeterministic ANY_VALUE queries"},
            "queries": {}, "totals_ms": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    workers = {}
    try:
        for label, library in libraries.items():
            workers[label] = Worker(library, args.threads)
        for workload in args.workloads:
            manifest_path = ROOT / "out/experiments/paper-local/workloads" / WORKLOADS[workload] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            queries = sorted(Path(manifest["queries_dir"]).glob("*.sql"))
            queries = [q for q in queries if args.queries is None or q.name in args.queries]
            if not queries:
                raise ValueError(f"No matching queries in {workload}")
            for w in workers.values():
                w.request(database=manifest["db_path"])
            for query in queries:
                sql = query.read_text().strip().rstrip(";")
                verification_sql = sql
                if "ANY_VALUE(" in sql.upper():
                    verification_sql = "SELECT COUNT(*) " + sql[sql.upper().index("FROM "):]
                answers = {}
                for mode, (label, compressed) in modes.items():
                    answers[mode] = workers[label].request(sql=verification_sql, compressed=compressed, verify=True)["values"]
                for mode, answer in answers.items():
                    if not equivalent(answers["uncompressed"], answer):
                        args.output.with_suffix(".mismatch.json").write_text(
                            json.dumps({"query": f"{workload}/{query.name}", "answers": answers}, indent=2) + "\n")
                        raise AssertionError(f"Different results for {workload}/{query.name}: {mode}")
                samples = {mode: [] for mode in modes}
                for repetition in range(-args.warmups, args.repetitions):
                    order = list(modes)
                    rng.shuffle(order)
                    for mode in order:
                        label, compressed = modes[mode]
                        result = workers[label].request(sql=sql, compressed=compressed)
                        if repetition >= 0:
                            samples[mode].append(result["ms"])
                entry = {"samples_ms": samples, "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                         "verified": True, "query_sha256": hashlib.sha256(sql.encode()).hexdigest()}
                if args.profiles:
                    entry["profiles"] = {mode: workers[label].request(sql=sql, compressed=compressed, profile=True)["profile"]
                                         for mode, (label, compressed) in modes.items()}
                data["queries"][f"{workload}/{query.name}"] = entry
                data["totals_ms"][workload] = {
                    mode: sum(v["median_ms"][mode] for k, v in data["queries"].items() if k.startswith(workload + "/"))
                    for mode in modes
                }
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(data, indent=2) + "\n")
                temporary.replace(args.output)
                print(workload, query.name, {k: round(v, 3) for k, v in entry["median_ms"].items()}, flush=True)
            print("TOTAL", workload, data["totals_ms"][workload], flush=True)
    finally:
        for w in workers.values():
            w.close()


if __name__ == "__main__":
    main()
