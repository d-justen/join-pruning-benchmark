"""Compact total-time bars for the standalone JPB replay; phase data stay in exports."""
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
from paper.figures.filter_benchmark import FILTER_COLORS

FILTER_ORDER = ("grafite", "diva", "prf_uncompressed", "prf")
FILTER_CODES = {"grafite": "G", "diva": "D", "prf_uncompressed": "U", "prf": "P"}
PATTERNS = ("range", "shoulder", "seasonal", "scattered")


def pattern_order(pattern: str) -> tuple[int, int]:
    if pattern.startswith("seasonal_"):
        return (2, int(pattern.split("_")[1]))
    return (PATTERNS.index(pattern), 10 if pattern == "seasonal" else 0)
PHASES = (("build", "Build", "#4c78a8"), ("range", "Range probes", "#e6a23c"),
          ("point", "Point probes", "#38978a"))
CAPTION = ("Mean total simulated filtering time, including construction/finalization, row-group range probes, "
           "and point probes on surviving mixed groups. Error bars show one sample standard deviation of "
           "repetition totals. Build keys are supplied sorted and distinct; ordered/uniform describe the probes. "
           "Inputs and correctness checks are untimed. Single thread; no scan I/O, decompression, hash join, "
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
            raise ValueError("JPB phase means do not equal mean total")
        # Preserve representation/budget provenance; never average hash seeds.
        for field in ("adapter_representation", "requested_bpk", "construction_seed", "build_seed",
                      "hash_a", "hash_b", "reduced_universe", "build_input_order"):
            if field in first:
                item[field] = ";".join(sorted({row[field] for row in samples}))
        series.append(item)
    return sorted(series, key=lambda row: (row["dimension_cardinality"], row["layout"],
                                          pattern_order(row["pattern"]), FILTER_ORDER.index(row["filter"])))


def build_jpb_artifacts(csv_path: Path, output_dir: Path, figures_dir: Path,
                        unavailable_filters: list[str], *, y_scale: str = "log") -> dict:
    return render_jpb_series(aggregate_rows(read_csv(csv_path)), output_dir, figures_dir,
                             unavailable_filters, y_scale=y_scale)


def render_jpb_series(series: list[dict], output_dir: Path, figures_dir: Path,
                      unavailable_filters: list[str], *, y_scale: str = "log") -> dict:
    """Render exported means/SDs as well as newly aggregated measurements."""
    if not series:
        raise ValueError("empty JPB timing series")
    if any(not math.isfinite(row[field]) or row[field] < 0 for row in series
           for field in ("total_ms", "total_stddev_ms")):
        raise ValueError("invalid JPB total timing statistics")
    if y_scale not in {"linear", "log"}:
        raise ValueError("JPB y_scale must be linear or log")
    if y_scale == "log" and any(row["total_ms"] <= 0 for row in series):
        raise ValueError("logarithmic JPB figures require positive total times; use y_scale='linear'")
    dimensions = sorted({row["dimension_cardinality"] for row in series})
    layouts = [layout for layout in ("ordered", "uniform") if any(row["layout"] == layout for row in series)]
    filters = [name for name in FILTER_ORDER if any(row["filter"] == name for row in series)]
    lookup = {(row["dimension_cardinality"], row["layout"], row["pattern"], row["filter"]): row for row in series}
    height = 1.7 * len(layouts) + (1.0 if len(dimensions) == 1 else 0.65)
    fig, axes = plt.subplots(len(layouts), len(dimensions), squeeze=False, sharey=True,
                             figsize=(max(5.6, 3.65 * len(dimensions)), height))
    fig.subplots_adjust(left=0.14 if len(dimensions) == 1 else 0.095,
                        right=0.99, top=1 - 0.5 / height, bottom=0.48 / height,
                        hspace=0.62, wspace=0.13)
    low = min(row["total_ms"] for row in series) * 0.45 if y_scale == "log" else 0
    high = max(row["total_ms"] + row["total_stddev_ms"] for row in series) * (1.6 if y_scale == "log" else 1.1)
    bar_count, clipped_errors = 0, 0
    for li, layout in enumerate(layouts):
        for di, dimension in enumerate(dimensions):
            ax = axes[li, di]
            patterns = sorted({row["pattern"] for row in series
                               if row["dimension_cardinality"] == dimension and row["layout"] == layout},
                              key=pattern_order)
            for pi, pattern in enumerate(patterns):
                for fi, name in enumerate(filters):
                    row = lookup[(dimension, layout, pattern, name)]
                    x = pi + (fi - (len(filters) - 1) / 2) * 0.19
                    total, sd = row["total_ms"], row["total_stddev_ms"]
                    ax.bar(x, total, width=0.17, color=FILTER_COLORS[name], linewidth=0, zorder=3)
                    if sd:
                        lower_error = min(sd, total - low) if y_scale == "log" else sd
                        clipped_errors += lower_error < sd
                        ax.errorbar(x, total, yerr=[[lower_error], [sd]], color="#202020", fmt="none",
                                    linewidth=0.6, capsize=1.5, zorder=4)
                    bar_count += 1
            ax.set_xticks(range(len(patterns)), [p.capitalize().replace("_", "\n") for p in patterns], fontsize=7)
            ax.tick_params(axis="x", length=0, pad=4)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_yscale(y_scale)
            if y_scale == "linear":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylim(low, high if high else 1)
            ax.set_xlim(-0.55, max(0.55, len(patterns) - 0.45))
            ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{layout.capitalize()} probes · {dimension:,} dimension keys", fontsize=8, pad=7)
            if not patterns:
                ax.text(0.5, 0.5, "No configured cases", transform=ax.transAxes, ha="center", fontsize=8)
    fig.supylabel("Total simulated filtering time\n(ms)", fontsize=7, x=0.015)
    exact_fallback = any("exact_elias_fano" in row.get("adapter_representation", "") for row in series)
    labels = {"grafite": "Grafite (+ exact EF)" if exact_fallback else "Grafite", "diva": "Diva", "prf_uncompressed": "PRF uncompressed", "prf": "PRF compressed"}
    fig.legend(handles=[Patch(facecolor=FILTER_COLORS[name], label=labels[name]) for name in filters],
               loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.54, 0.99))
    note = f"Mean total ±1 SD · shared {y_scale} scale"
    if unavailable_filters:
        note += " · Unavailable: " + ", ".join(unavailable_filters)
    fig.text(0.54, 0.012, note, ha="center", va="bottom", fontsize=6.5)
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
    caption = CAPTION + f" Panels share a {y_scale} scale."
    if exact_fallback:
        caption += " Grafite uses exact Elias–Fano when it is cheaper than the requested approximate budget."
    if clipped_errors:
        caption += " Lower error bars extending below the displayed log range are clipped at the axis limit."
    payload = {"figure_path": str(pdf), "preview_path": str(png), "series_path": str(series_path),
               "summary_path": str(summary_path), "caption": caption, "unavailable_filters": unavailable_filters,
               "verification": {"bar_count": bar_count, "panel_count": len(layouts) * len(dimensions),
                                "phase_order": [phase for phase, _, _ in PHASES], "aggregate": "mean",
                                "bar_value": "total_ms", "error_value": "total_stddev_ms", "stacked": False,
                                "y_scale": y_scale, "shared_y_axis": True, "clipped_error_bars": clipped_errors,
                                "zero_baseline": y_scale == "linear", "phase_means_equal_mean_totals": True},
               "series": series}
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
