"""Additive build/range/point timing bars for the standalone JPB replay."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from benchmarking.microbenchmarks import read_csv

FILTER_ORDER = ("grafite", "diva", "prf_uncompressed", "prf")
FILTER_CODES = {"grafite": "G", "diva": "D", "prf_uncompressed": "U", "prf": "P"}
PATTERNS = ("range", "shoulder", "seasonal", "scattered")
PHASES = (("build", "Build", "#4c78a8"), ("range", "Range probes", "#e6a23c"),
          ("point", "Point probes", "#38978a"))
CAPTION = ("Mean simulated filtering time; stacks include construction/finalization, row-group range probes, "
           "and point probes on surviving mixed groups. Error bars show one sample standard deviation of total "
           "time across repetitions. Panels use independent linear scales starting at zero. "
           "Build keys are supplied sorted. Inputs and correctness checks are untimed. "
           "Single thread; no scan I/O, decompression, hash join, "
           "aggregation, column-segment pruning, or automatic Bloom fallback. Uniform probes are synthetic.")


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["query"], row["filter"]
        identity = (*key, int(row["repetition"]))
        if identity in seen:
            raise ValueError("duplicate JPB figure sample")
        seen.add(identity)
        values = [float(row[f"{phase}_time_ns"]) for phase in ("build", "range", "point", "total")]
        if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(sum(values[:3]), values[3]):
            raise ValueError("invalid JPB phase timings")
        groups[key].append(row)
    series = []
    for (query, name), samples in groups.items():
        first = samples[0]
        total = [int(row["total_time_ns"]) / 1e6 for row in samples]
        item = {"query": query, "filter": name, "pattern": first["pattern"], "layout": first["layout"],
                "dimension_cardinality": int(first["dimension_cardinality"]), "sample_count": len(samples),
                **{f"{phase}_ms": statistics.mean(int(row[f"{phase}_time_ns"]) / 1e6 for row in samples)
                   for phase in ("build", "range", "point")},
                "total_ms": statistics.mean(total), "total_stddev_ms": statistics.stdev(total) if len(total) > 1 else 0,
                "median_total_ms": statistics.median(total)}
        for field in ("summary_bytes", "measured_bpk", "groups_rejected", "groups_accepted", "groups_mixed",
                      "point_probes", "candidate_rows", "true_positives", "false_positives", "false_negatives",
                      "empty_ranges", "range_false_positives"):
            item[f"mean_{field}"] = statistics.mean(float(row[field]) for row in samples)
        point_rates = [
            int(row["false_positives"]) / (int(row["probe_rows"]) - int(row["true_positives"]))
            for row in samples if int(row["probe_rows"]) > int(row["true_positives"])]
        range_rates = [int(row["range_false_positives"]) / int(row["empty_ranges"])
                       for row in samples if int(row["empty_ranges"])]
        item["mean_false_positive_rate"] = statistics.mean(point_rates) if point_rates else None
        item["mean_range_false_positive_rate"] = statistics.mean(range_rates) if range_rates else None
        if not math.isclose(sum(item[f"{phase}_ms"] for phase, _, _ in PHASES), item["total_ms"]):
            raise ValueError("JPB figure stack does not equal mean total")
        series.append(item)
    return sorted(series, key=lambda row: (row["dimension_cardinality"], row["layout"],
                                          PATTERNS.index(row["pattern"]), FILTER_ORDER.index(row["filter"])))


def build_jpb_artifacts(csv_path: Path, output_dir: Path, figures_dir: Path,
                        unavailable_filters: list[str]) -> dict:
    series = aggregate_rows(read_csv(csv_path))
    dimensions = sorted({row["dimension_cardinality"] for row in series})
    layouts = [layout for layout in ("ordered", "uniform") if any(row["layout"] == layout for row in series)]
    filters = [name for name in FILTER_ORDER if any(row["filter"] == name for row in series)]
    lookup = {(row["dimension_cardinality"], row["layout"], row["pattern"], row["filter"]): row for row in series}
    height = 2.45 * len(layouts) + 0.65
    fig, axes = plt.subplots(len(layouts), len(dimensions), squeeze=False,
                             figsize=(max(3.8, 3.65 * len(dimensions)), height))
    fig.subplots_adjust(left=0.095, right=0.985, top=1 - 0.5 / height, bottom=0.9 / height,
                        hspace=0.65, wspace=0.3)
    bar_count = 0
    for li, layout in enumerate(layouts):
        for di, dimension in enumerate(dimensions):
            ax = axes[li, di]
            ticks, labels = [], []
            patterns = [pattern for pattern in PATTERNS if (dimension, layout, pattern, filters[0]) in lookup]
            for pi, pattern in enumerate(patterns):
                for fi, name in enumerate(filters):
                    row = lookup[(dimension, layout, pattern, name)]
                    x = pi + (fi - (len(filters) - 1) / 2) * 0.19
                    bottom = 0
                    for phase, _, color in PHASES:
                        value = row[f"{phase}_ms"]
                        ax.bar(x, value, width=0.17, bottom=bottom, color=color, linewidth=0,
                               rasterized=False, zorder=3)
                        bottom += value
                    if row["total_stddev_ms"]:
                        ax.errorbar(x, bottom, yerr=row["total_stddev_ms"], color="#202020", fmt="none",
                                    linewidth=0.6, capsize=1.5, zorder=4)
                    ticks.append(x)
                    labels.append(FILTER_CODES[name])
                    bar_count += 1
                ax.text(pi, -0.19, pattern.capitalize(), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7)
            ax.set_xticks(ticks, labels, fontsize=6)
            ax.tick_params(axis="x", length=0, pad=3)
            ax.tick_params(axis="y", labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(-0.55, max(0.55, len(patterns) - 0.45))
            ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{layout.capitalize()} · {dimension:,} dimension keys", fontsize=8, pad=7)
            if not patterns:
                ax.text(0.5, 0.5, "No configured cases", transform=ax.transAxes, ha="center", fontsize=8)
    fig.supylabel("Simulated filtering time (ms)", fontsize=9, x=0.015)
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in PHASES],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.53, 0.99))
    note = "G: Grafite   D: Diva   U: PRF uncompressed   P: PRF compressed"
    if unavailable_filters:
        note += "\nUnavailable in this build: " + ", ".join(unavailable_filters)
    fig.text(0.54, 0.025, note + "\nMean of repetition totals; error bars: ±1 SD. Independent panel scales.",
             ha="center", va="bottom", fontsize=6.5, linespacing=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf = figures_dir / "jpb_simulated_filtering.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, format="png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    series_path = output_dir / "jpb_simulated_filtering_series.csv"
    with series_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    summary_path = output_dir / "jpb_simulated_filtering_summary.json"
    payload = {"figure_path": str(pdf), "preview_path": str(png), "series_path": str(series_path),
               "summary_path": str(summary_path), "caption": CAPTION, "unavailable_filters": unavailable_filters,
               "verification": {"bar_count": bar_count, "panel_count": len(layouts) * len(dimensions),
                                "phase_order": [phase for phase, _, _ in PHASES], "aggregate": "mean",
                                "zero_baseline": True, "stacks_equal_mean_totals": True}, "series": series}
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
