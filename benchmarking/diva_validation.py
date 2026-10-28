"""Compare Diva backends sequentially on one host, outside paper measurements."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

from benchmarking.benchmark_binary import supports_sve_bitperm
from benchmarking.microbenchmarks import run_microbenchmarks
from benchmarking.suite_io import sha256, write_json


def compare_results(root: Path) -> dict:
    comparisons = []
    for kind, label, timings in (
        ("point", "filter", {"build_time_ns", "post_processing_time_ns", "probe_throughput_per_sec"}),
        ("range", "summary", {"total_query_time_ns", "queries_per_sec", "post_processing_time_ns"}),
    ):
        rows = {}
        for backend in ("portable", "sve2"):
            path = root / backend / "microbenchmarks/filter-probes" / (kind + ".csv")
            with path.open(newline="") as handle:
                rows[backend] = {tuple(row[k] for k in ("repetition", "cluster_count") +
                                      (("range_count",) if kind == "range" else ())): row
                                 for row in csv.DictReader(handle) if row[label] == "diva"}
        if not rows["portable"] or rows["portable"].keys() != rows["sve2"].keys():
            raise ValueError("Diva backend comparison has missing measurements")
        for key, baseline in rows["portable"].items():
            optimized = rows["sve2"][key]
            if {k: v for k, v in baseline.items() if k not in timings} != {
                    k: v for k, v in optimized.items() if k not in timings}:
                raise ValueError(f"Diva backends disagree on inputs, memory or answers: {kind}/{key}")
        for clusters in sorted({r["cluster_count"] for r in rows["portable"].values()}, key=int):
            for metric in ("build_time_ns", "probe_throughput_per_sec") if kind == "point" else ("queries_per_sec",):
                medians = {b: statistics.median(float(r[metric]) for r in rs.values() if r["cluster_count"] == clusters)
                           for b, rs in rows.items()}
                comparisons.append({"kind": kind, "clusters": int(clusters), "metric": metric, **medians,
                                    "sve2_over_portable": medians["sve2"] / medians["portable"]
                                    if medians["portable"] else None})
    return {"status": "complete", "identical_inputs_memory_and_answers": True, "comparisons": comparisons,
            "scope": "Sequential same-host backend comparison; not an x86 performance comparison"}


def validate_backends(spec, system, root: Path, checkpoint_callback) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not supports_sve_bitperm():
        write_json(root / "comparison.json", {"status": "unavailable", "reason": "host lacks SVE2 bit permutation"})
        return
    previous = os.environ.get("BENCH_DIVA_BACKEND")
    try:
        for backend in ("portable", "sve2"):
            os.environ["BENCH_DIVA_BACKEND"] = backend
            run_microbenchmarks(spec, system, root / backend, reuse_completed=True,
                               checkpoint_cases=True, checkpoint_callback=checkpoint_callback)
    finally:
        if previous is None:
            os.environ.pop("BENCH_DIVA_BACKEND", None)
        else:
            os.environ["BENCH_DIVA_BACKEND"] = previous
    report = compare_results(root)
    report["files"] = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                       if p.is_file() and p.name != "comparison.json"}
    write_json(root / "comparison.json", report)


def valid_comparison(root: Path) -> bool:
    path = root / "comparison.json"
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    return report.get("status") in {"complete", "unavailable"} and all(
        (root / p).is_file() and sha256(root / p) == digest for p, digest in report.get("files", {}).items())
