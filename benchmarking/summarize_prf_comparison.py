#!/usr/bin/env python3
"""Summarize paired PRF measurements as per-query CSV and workload totals.

Confidence intervals resample repetitions within each fixed query, keeping the
mode measurements paired. They quantify timing variation in this run, not
uncertainty from different machines, datasets or worker-process placement.
"""

import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def changes(totals, candidate):
    result = {
        "vs_original_percent": 100 * (totals[candidate] / totals["original"] - 1),
        "vs_uncompressed_percent": 100 * (totals[candidate] / totals["uncompressed"] - 1),
    }
    control = candidate + "-uncompressed"
    if control in totals:
        result["vs_own_uncompressed_percent"] = 100 * (totals[candidate] / totals[control] - 1)
        result["control_adjusted_vs_original_percent"] = 100 * (
            (totals[candidate] / totals[control]) / (totals["original"] / totals["uncompressed"]) - 1
        )
    return result


def has_prf(node):
    if isinstance(node, dict):
        if node.get("join_filter", {}).get("prf_build_count", 0) > 0:
            return True
        return any(has_prf(value) for value in node.values())
    if isinstance(node, list):
        return any(has_prf(value) for value in node)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--candidate", default="tuned")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    groups = {}
    rows = []
    for query, entry in data["queries"].items():
        workload = query.split("/", 1)[0]
        groups.setdefault(workload, []).append(entry)
        rows.append({"query": query, **{k + "_ms": v for k, v in entry["median_ms"].items()},
                     **changes(entry["median_ms"], args.candidate)})
    groups["all"] = list(data["queries"].values())
    # Expose workloads where a total-time difference is dominated by queries without a PRF.
    for workload, entries in list(groups.items()):
        if workload == "all" or not all("profiles" in entry for entry in entries):
            continue
        active = [entry for entry in entries if has_prf(entry["profiles"]["original"])]
        if active and len(active) != len(entries):
            groups[workload + "_with_prf"] = active
            groups[workload + "_without_prf"] = [entry for entry in entries if not has_prf(entry["profiles"]["original"])]
    rng = random.Random(args.seed)
    report = {"source": str(args.input), "candidate": args.candidate, "bootstrap_samples": args.bootstrap,
              "ci_scope": "Paired repetition bootstrap for this fixed workload/run; excludes machine/process systematic effects",
              "workloads": {}}
    for workload, entries in groups.items():
        modes = list(entries[0]["median_ms"])
        totals = {mode: sum(e["median_ms"][mode] for e in entries) for mode in modes}
        point = changes(totals, args.candidate)
        replicates = {k: [] for k in point}
        for _ in range(args.bootstrap):
            sampled = dict.fromkeys(modes, 0.0)
            for entry in entries:
                n = len(entry["samples_ms"][modes[0]])
                indices = [rng.randrange(n) for _ in range(n)]
                for mode in modes:
                    sampled[mode] += statistics.median(entry["samples_ms"][mode][i] for i in indices)
            for key, value in changes(sampled, args.candidate).items():
                replicates[key].append(value)
        intervals = {}
        for key, values in replicates.items():
            values.sort()
            if values:
                intervals[key] = [values[int(len(values) * .025)], values[min(len(values) - 1, int(len(values) * .975))]]
        report["workloads"][workload] = {"queries": len(entries), "totals_ms": totals,
                                            "changes": point, "ci95": intervals,
                                            "all_results_verified": all(e["verified"] for e in entries)}
        print(workload, {k: round(v, 3) for k, v in totals.items()}, {k: round(v, 2) for k, v in point.items()})
    output = args.input.with_suffix(".summary.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    with args.input.with_suffix(".csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
