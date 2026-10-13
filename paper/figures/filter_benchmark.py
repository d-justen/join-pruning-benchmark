from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FILTER_ORDER = (
    "bloom",
    "grafite",
    "diva",
    "prf_uncompressed",
    "prf",
)

FILTER_LABELS = {
    "bloom": "Bloom",
    "grafite": "Grafite",
    "diva": "Diva",
    "prf_uncompressed": "Prefix Range Filter (uncompressed)",
    "prf": "Prefix Range Filter",
}

FILTER_COLORS = {
    "bloom": "#2563eb",
    "grafite": "#b91c1c",
    "diva": "#111827",
    "prf_uncompressed": "#d97706",
    "prf": "#0f766e",
}

FILTER_LINESTYLES = {
    "bloom": "-",
    "grafite": "--",
    "diva": ":",
    "prf_uncompressed": "-.",
    "prf": "-",
}

PAGE_SIZES = {
    "one-column": (3.35, 2.45),
    "two-column": (7.0, 3.15),
}

FILTER_BENCHMARK_EXTRA_HEIGHT = 2.20
FILTER_BENCHMARK_LABEL_FONT_SIZE = 7.5
FILTER_BENCHMARK_TICK_FONT_SIZE = 6.5
FILTER_BENCHMARK_LEGEND_FONT_SIZE = 6.5
CACHE_REFERENCE_COLOR = "#6b7280"
CACHE_REFERENCE_LABEL_FONT_SIZE = 6.3
CACHE_REFERENCE_LINES = (
    (49_152, "L1d · 48 KiB"),
    (2 * 1_024 * 1_024, "L2 · 2 MiB"),
)


@dataclass(frozen=True)
class FilterBenchmarkRow:
    filter_name: str
    cluster_count: int
    build_time_ns: int
    post_processing_time_ns: int
    summary_bytes: int
    probe_throughput_per_sec: float
    false_positives: int
    actual_fpr: float
    actual_negatives: int | None = None

    @property
    def total_build_time_ns(self) -> int:
        return self.build_time_ns + self.post_processing_time_ns


@dataclass(frozen=True)
class BenchmarkGenerationConfig:
    domain_size: int
    total_keys: int
    probe_count: int
    seed: int


def _display_label(filter_name: str) -> str:
    return FILTER_LABELS.get(filter_name, filter_name.replace("_", " "))


def _row_major_legend_order(labels: list[str], ncol: int) -> list[str]:
    if ncol <= 1 or len(labels) <= 1:
        return labels
    rows = (len(labels) + ncol - 1) // ncol
    grid: list[list[str | None]] = []
    index = 0
    for _ in range(rows):
        row: list[str | None] = []
        for _ in range(ncol):
            row.append(labels[index] if index < len(labels) else None)
            index += 1
        grid.append(row)

    ordered: list[str] = []
    for column in range(ncol):
        for row in range(rows):
            label = grid[row][column]
            if label is not None:
                ordered.append(label)
    return ordered


def _format_millions(value: float, _: object) -> str:
    return f"{value / 1_000_000:.3g}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def _format_ms(value: float) -> str:
    return f"{value:.2f}"


def _format_compact_bytes(value: float, _: object) -> str:
    if value < 1_000:
        return f"{value:.0f}"
    if value < 1_000_000:
        return f"{value / 1_000:.0f}K"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.0f}M"
    return f"{value / 1_000_000_000:.0f}G"


def _aggregate(values: list[float], aggregate: str) -> float:
    if aggregate == "mean":
        return statistics.fmean(values)
    if aggregate == "median":
        return statistics.median(values)
    raise ValueError(f"unsupported aggregate {aggregate!r}")


def load_filter_benchmark_rows(csv_path: Path) -> list[FilterBenchmarkRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "filter",
            "cluster_count",
            "build_time_ns",
            "post_processing_time_ns",
            "summary_bytes",
            "probe_throughput_per_sec",
            "false_positives",
            "actual_fpr",
        }
        missing = sorted(required_columns.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing required CSV columns: {', '.join(missing)}")
        rows = [
            FilterBenchmarkRow(
                filter_name=row["filter"],
                cluster_count=int(row["cluster_count"]),
                build_time_ns=int(row["build_time_ns"]),
                post_processing_time_ns=int(row["post_processing_time_ns"]),
                summary_bytes=int(row["summary_bytes"]),
                probe_throughput_per_sec=float(row["probe_throughput_per_sec"]),
                false_positives=int(row["false_positives"]),
                actual_fpr=float(row["actual_fpr"]),
                actual_negatives=int(row["actual_negatives"]) if row.get("actual_negatives") else None,
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    for row in rows:
        if row.filter_name not in FILTER_ORDER:
            raise ValueError(f"unsupported filter: {row.filter_name}")
        if not math.isfinite(row.actual_fpr) or not 0 <= row.actual_fpr <= 1:
            raise ValueError("actual_fpr must be finite and between 0 and 1")
        if row.actual_negatives is not None:
            if not 0 <= row.false_positives <= row.actual_negatives:
                raise ValueError("invalid measured false-positive/negative counts")
            expected = row.false_positives / row.actual_negatives if row.actual_negatives else 0
            if not math.isclose(row.actual_fpr, expected, abs_tol=1e-9):
                raise ValueError("actual_fpr disagrees with measured counts")
        if (row.cluster_count <= 0 or not math.isfinite(row.probe_throughput_per_sec)
                or row.probe_throughput_per_sec <= 0 or min(row.build_time_ns,
                row.post_processing_time_ns, row.summary_bytes, row.false_positives) < 0):
            raise ValueError("invalid point benchmark measurement")
    return rows


def build_throughput_series(
    rows: list[FilterBenchmarkRow],
    *,
    aggregate: str,
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.filter_name, {}).setdefault(row.cluster_count, []).append(
            row.probe_throughput_per_sec
        )
    return {
        filter_name: [
            (cluster_count, _aggregate(values, aggregate))
            for cluster_count, values in sorted(cluster_rows.items())
        ]
        for filter_name, cluster_rows in grouped.items()
    }


def build_summary_size_series(
    rows: list[FilterBenchmarkRow],
    *,
    aggregate: str,
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.filter_name, {}).setdefault(row.cluster_count, []).append(
            float(row.summary_bytes)
        )
    return {
        filter_name: [
            (cluster_count, _aggregate(values, aggregate))
            for cluster_count, values in sorted(cluster_rows.items())
        ]
        for filter_name, cluster_rows in grouped.items()
    }


def build_build_time_series(
    rows: list[FilterBenchmarkRow],
    *,
    aggregate: str,
) -> dict[str, list[tuple[int, float]]]:
    """Aggregate total filter construction time, expressed in milliseconds."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.filter_name, {}).setdefault(row.cluster_count, []).append(
            row.total_build_time_ns / 1_000_000.0
        )
    return {
        filter_name: [
            (cluster_count, _aggregate(values, aggregate))
            for cluster_count, values in sorted(cluster_rows.items())
        ]
        for filter_name, cluster_rows in grouped.items()
    }


def build_false_positive_rate_series(
    rows: list[FilterBenchmarkRow], *, aggregate: str,
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.filter_name, {}).setdefault(row.cluster_count, []).append(row.actual_fpr)
    return {name: [(cluster, _aggregate(values, aggregate)) for cluster, values in sorted(points.items())]
            for name, points in grouped.items()}


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {"min": min(values), "max": max(values), "avg": statistics.fmean(values)}


def summarize_filter_benchmark(rows: list[FilterBenchmarkRow]) -> dict[str, object]:
    systems = []
    for name in FILTER_ORDER:
        selected = [row for row in rows if row.filter_name == name]
        if not selected:
            continue
        systems.append({
            "filter": name, "label": _display_label(name), "row_count": len(selected),
            "cluster_count_min": min(row.cluster_count for row in selected),
            "cluster_count_max": max(row.cluster_count for row in selected),
            "actual_negatives": _stats([row.actual_negatives for row in selected if row.actual_negatives is not None]),
            "true_negatives": _stats([row.actual_negatives - row.false_positives for row in selected
                                      if row.actual_negatives is not None]),
            "build_time_ms": _stats([row.total_build_time_ns / 1_000_000 for row in selected]),
            "false_positive_rate": _stats([row.actual_fpr for row in selected]),
            "point_probes_per_sec": _stats([row.probe_throughput_per_sec for row in selected]),
        })
    return {"actual_negative_probe_counts": _stats([row.actual_negatives for row in rows
                                                    if row.actual_negatives is not None]),
            "system_count": len(systems), "systems": systems}


def write_filter_benchmark_summary_markdown(
    path: Path, *, csv_path: Path, aggregate: str, summary: dict[str, object],
) -> None:
    lines = ["# Filter Benchmark Summary", "", f"- Point CSV: `{csv_path}`",
             f"- Aggregation for figure line plots: `{aggregate}`", "",
             "| System | Build min (ms) | Build max (ms) | Build avg (ms) | FPR min | FPR max | FPR avg | TN min | TN max | TN avg |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for system in summary["systems"]:
        values = [system["label"]]
        for metric, formatter in (("build_time_ms", _format_ms), ("false_positive_rate", _format_percent),
                                   ("true_negatives", lambda value: f"{value:.3f}")):
            values.extend("—" if system[metric][key] is None else formatter(system[metric][key])
                          for key in ("min", "max", "avg"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Build time is `build_time_ns + post_processing_time_ns`.",
                  "FPR uses each row's measured `actual_fpr` (FP / actual negatives).",
                  "Each repetition uses seed + repetition; all point filters share that repetition's probe set.",
                  "True negatives use measured counts; unavailable counts in legacy CSVs remain unknown.", ""])
    if "range_queries_per_sec" in summary:
        lines.extend(["Range throughput is aggregated separately for each query kind. The overview uses empty ranges.", "",
                      "| System | Query kind | Min queries/s | Max queries/s | Avg queries/s |",
                      "|---|---|---:|---:|---:|"])
        for kind, systems in summary["range_queries_per_sec"].items():
            for name, stats in systems.items():
                lines.append(f"| {_display_label(name)} | {kind} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['avg']:.3f} |")
        lines.append("")
    if "construction_policy" in summary:
        lines.extend([summary["construction_policy"], ""] )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_filter_benchmark_plot(
    output_path: Path,
    *,
    throughput_series: dict[str, list[tuple[int, float]]],
    summary_size_series: dict[str, list[tuple[int, float]]],
    build_time_series: dict[str, list[tuple[int, float]]],
    false_positive_rate_series: dict[str, list[tuple[int, float]]],
    aggregate: str,
    range_throughput_series: dict[str, list[tuple[int, float]]] | None = None,
    grafite_exact_fallback: bool = False,
) -> dict[str, object]:
    width, base_height = PAGE_SIZES["one-column"]
    panel_count = 5 if range_throughput_series is not None else 4
    height = base_height + FILTER_BENCHMARK_EXTRA_HEIGHT + (1.05 if panel_count == 5 else 0)
    fig, axes = plt.subplots(
        panel_count, 1, figsize=(width, height), sharex=True,
        gridspec_kw={"hspace": 0.10}, constrained_layout=False,
    )
    throughput_ax = axes[0]
    size_ax, build_time_ax, false_positive_rate_ax = axes[-3:]
    if range_throughput_series is not None:
        range_ax = axes[1]
        for name in FILTER_ORDER:
            if name not in range_throughput_series:
                continue
            points = range_throughput_series[name]
            range_ax.plot([point[0] for point in points], [point[1] for point in points],
                          color=FILTER_COLORS[name], linestyle=FILTER_LINESTYLES[name], linewidth=1.7)
        range_ax.set_ylabel("Range throughput\n(M queries/s)", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
        range_ax.set_yscale("log")
        range_ax.yaxis.set_major_formatter(FuncFormatter(_format_millions))
        range_ax.yaxis.set_minor_formatter(NullFormatter())
    fig.patch.set_facecolor("white")
    throughput_ax.set_facecolor("white")
    size_ax.set_facecolor("white")
    build_time_ax.set_facecolor("white")
    false_positive_rate_ax.set_facecolor("white")

    plotted_filters = [filter_name for filter_name in FILTER_ORDER if filter_name in throughput_series]
    throughput_values = [value for points in throughput_series.values() for _, value in points if value > 0]
    summary_size_values = [value for points in summary_size_series.values() for _, value in points if value > 0]
    build_time_values = [value for points in build_time_series.values() for _, value in points if value > 0]
    false_positive_rate_values = [
        value for points in false_positive_rate_series.values() for _, value in points if value > 0
    ]
    throughput_ymin = min(throughput_values) * 0.95 if throughput_values else 1.0
    throughput_ymax = max(throughput_values) * 1.08 if throughput_values else 10.0
    summary_ymin = min(summary_size_values) * 0.95 if summary_size_values else 1.0
    summary_ymax = max(summary_size_values) * 1.08 if summary_size_values else 10.0
    build_time_ymin = 0.0
    build_time_ymax = max(build_time_values) * 1.08 if build_time_values else 10.0
    false_positive_rate_ymax = max(false_positive_rate_values) * 1.08 if false_positive_rate_values else 0.01

    for filter_name in plotted_filters:
        throughput_points = throughput_series[filter_name]
        xs = [cluster_count for cluster_count, _ in throughput_points]
        throughput_ys = [throughput for _, throughput in throughput_points]
        size_points = summary_size_series[filter_name]
        size_ys = [summary_bytes for _, summary_bytes in size_points]
        build_time_points = build_time_series[filter_name]
        build_time_ys = [build_time_ms for _, build_time_ms in build_time_points]
        false_positive_rate_points = false_positive_rate_series[filter_name]
        false_positive_rate_ys = [rate for _, rate in false_positive_rate_points]
        throughput_ax.plot(
            xs,
            throughput_ys,
            label=_display_label(filter_name),
            color=FILTER_COLORS[filter_name],
            linestyle=FILTER_LINESTYLES[filter_name],
            linewidth=1.7,
        )
        size_ax.plot(
            xs,
            size_ys,
            color=FILTER_COLORS[filter_name],
            linestyle=FILTER_LINESTYLES[filter_name],
            linewidth=1.7,
        )
        build_time_ax.plot(
            xs,
            build_time_ys,
            color=FILTER_COLORS[filter_name],
            linestyle=FILTER_LINESTYLES[filter_name],
            linewidth=1.7,
        )
        false_positive_rate_ax.plot(
            xs,
            false_positive_rate_ys,
            color=FILTER_COLORS[filter_name],
            linestyle=FILTER_LINESTYLES[filter_name],
            linewidth=1.7,
        )

    throughput_ax.set_ylabel(
        "Point throughput\n(M probes/s)",
        fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE,
    )
    throughput_ax.set_xscale("log")
    throughput_ax.set_yscale("log")
    throughput_ax.yaxis.set_major_formatter(FuncFormatter(_format_millions))
    throughput_ax.yaxis.set_minor_formatter(NullFormatter())
    throughput_ax.set_ylim(throughput_ymin, throughput_ymax)

    size_ax.set_ylabel("Summary size\n(bytes)", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    size_ax.set_xscale("log")
    size_ax.set_yscale("log")
    size_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
    size_ax.yaxis.set_major_formatter(FuncFormatter(_format_compact_bytes))
    size_ax.yaxis.set_minor_formatter(NullFormatter())
    size_ax.set_ylim(summary_ymin, summary_ymax)
    for cache_size, label in CACHE_REFERENCE_LINES:
        if not summary_ymin <= cache_size <= summary_ymax:
            continue
        size_ax.axhline(
            cache_size,
            color=CACHE_REFERENCE_COLOR,
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.8,
            zorder=1,
        )
        size_ax.annotate(
            label,
            xy=(0.99, cache_size),
            xycoords=("axes fraction", "data"),
            xytext=(0, -1.5),
            textcoords="offset points",
            ha="right",
            va="top",
            color=CACHE_REFERENCE_COLOR,
            fontsize=CACHE_REFERENCE_LABEL_FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
            zorder=4,
        )

    build_time_ax.set_ylabel("Build time\n(ms)", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    build_time_ax.set_xscale("log")
    build_time_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    build_time_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    build_time_ax.set_ylim(build_time_ymin, build_time_ymax)

    false_positive_rate_ax.set_xlabel("Number of clusters", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    false_positive_rate_ax.set_ylabel("False-positive\nrate", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    false_positive_rate_ax.set_xscale("log", base=2)
    clusters = sorted({cluster for points in throughput_series.values() for cluster, _ in points})
    tick_step = max(1, math.ceil((len(clusters) - 1) / 4))
    ticks = sorted(set(clusters[::tick_step] + [clusters[-1]]))
    false_positive_rate_ax.xaxis.set_major_locator(FixedLocator(ticks))
    false_positive_rate_ax.xaxis.set_major_formatter(FuncFormatter(_format_compact_bytes))
    false_positive_rate_ax.xaxis.set_minor_locator(NullLocator())
    false_positive_rate_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f}%"))
    false_positive_rate_ax.set_ylim(0.0, false_positive_rate_ymax)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=FILTER_BENCHMARK_TICK_FONT_SIZE)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.grid(axis="x", which="major", color="#f3f4f6", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")

    legend_columns = 2
    reordered_filters = _row_major_legend_order(plotted_filters, legend_columns)
    handles, labels = throughput_ax.get_legend_handles_labels()
    handle_by_label = {label: handle for handle, label in zip(handles, labels)}
    reordered_labels = [_display_label(filter_name) for filter_name in reordered_filters]
    reordered_handles = [handle_by_label[label] for label in reordered_labels]

    fig.legend(
        reordered_handles,
        [label.replace("Prefix Range Filter (uncompressed)", "PRF (uncompressed)").replace(
            "Grafite", "Grafite + exact EF" if grafite_exact_fallback else "Grafite") for label in reordered_labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=legend_columns,
        frameon=False,
        fontsize=FILTER_BENCHMARK_LEGEND_FONT_SIZE,
        handlelength=2.0,
        columnspacing=1.2,
        alignment="left",
    )
    fig.subplots_adjust(left=0.20, right=0.995, bottom=0.09, top=0.87)
    fig.align_ylabels(axes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "panel_order": (["point_throughput", "range_throughput", "summary_size", "build_time", "point_fpr"]
                        if panel_count == 5 else ["point_throughput", "summary_size", "build_time", "point_fpr"]),
        "range_filters": list(range_throughput_series or {}),
        "range_query_kind": "empty" if range_throughput_series is not None else None,
        "throughput_aggregate": aggregate,
        "summary_size_aggregate": aggregate,
        "build_time_aggregate": aggregate,
        "false_positive_rate_aggregate": aggregate,
        "plotted_filters": plotted_filters,
        "grafite_exact_fallback": grafite_exact_fallback,
        "point_count_per_filter": {
            filter_name: len(throughput_series[filter_name]) for filter_name in plotted_filters
        },
    }


def build_filter_benchmark_artifacts(
    *, csv_path: Path, output_dir: Path, aggregate: str = "median",
    range_csv_path: Path | None = None, figures_dir: Path | None = None,
    generation_config: BenchmarkGenerationConfig | None = None,
) -> dict[str, object]:
    rows = load_filter_benchmark_rows(csv_path)
    series = build_throughput_series(rows, aggregate=aggregate)
    summary_size_series = build_summary_size_series(rows, aggregate=aggregate)
    build_time_series = build_build_time_series(rows, aggregate=aggregate)
    false_positive_rate_series = build_false_positive_rate_series(rows, aggregate=aggregate)
    summary = summarize_filter_benchmark(rows)
    with csv_path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    metadata_fields = ("filter", "cluster_count", "repetition", "build_checksum", "construction_seed", "build_seed",
                       "build_input_order", "adapter_representation", "requested_bpk", "hash_a", "hash_b", "reduced_universe")
    construction = [{key: row[key] for key in metadata_fields} for row in raw_rows
                    if all(key in row for key in metadata_fields)]
    exact_fallback = any(row["adapter_representation"] == "exact_elias_fano" for row in construction)
    if construction:
        summary["construction"] = construction
        summary["construction_policy"] = ("Sorted distinct build keys; Grafite uses exact SUX Elias–Fano when cheaper "
                                           "than the requested approximate budget. Seeds derive from build inputs and repetition.")
    range_series = None
    range_rows = []
    all_range_series = {}
    if range_csv_path is not None:
        from paper.figures.range_benchmark import load_range_benchmark_rows, build_range_throughput_series
        range_rows = load_range_benchmark_rows(range_csv_path)
        all_range_series = {kind: build_range_throughput_series(range_rows, aggregate=aggregate, query_kind=kind)
                            for kind in sorted({row.query_kind for row in range_rows})}
        if "empty" not in all_range_series:
            raise ValueError("range CSV must include empty ranges for the comparison panel")
        range_series = all_range_series["empty"]
        expected = {name: {cluster for cluster, _ in points} for name, points in series.items() if name != "bloom"}
        actual = {name: {cluster for cluster, _ in points} for name, points in range_series.items()}
        if expected != actual:
            raise ValueError("point and empty-range series must use the same filters and cluster counts")
        summary["range_queries_per_sec"] = {
            kind: {name: _stats([row.queries_per_sec for row in range_rows
                                if row.query_kind == kind and row.series_name == name]) for name in kind_series}
            for kind, kind_series in all_range_series.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = (figures_dir or output_dir) / "filter_benchmark_overview.pdf"
    markdown_path = output_dir / "filter_benchmark_summary.md"
    summary_path = output_dir / "filter_benchmark_summary.json"
    series_path = output_dir / "filter_benchmark_series.csv"
    verification = draw_filter_benchmark_plot(
        figure_path, throughput_series=series, range_throughput_series=range_series,
        summary_size_series=summary_size_series, build_time_series=build_time_series,
        false_positive_rate_series=false_positive_rate_series, aggregate=aggregate,
        grafite_exact_fallback=exact_fallback,
    )
    with series_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filter", "cluster_count", "metric", "query_kind", "value", "unit", "sample_count"])
        for metric, unit, values in (("point_throughput", "probes/s", series),
                                     ("summary_size", "bytes", summary_size_series),
                                     ("build_time", "ms", build_time_series),
                                     ("point_fpr", "fraction", false_positive_rate_series)):
            for name, points in values.items():
                for cluster, value in points:
                    n = sum(row.filter_name == name and row.cluster_count == cluster for row in rows)
                    writer.writerow([name, cluster, metric, "point", value, unit, n])
        for kind, values in all_range_series.items():
            for name, points in values.items():
                for cluster, value in points:
                    n = sum(row.series_name == name and row.cluster_count == cluster and row.query_kind == kind
                            for row in range_rows)
                    writer.writerow([name, cluster, "range_throughput", kind, value, "queries/s", n])
    write_filter_benchmark_summary_markdown(markdown_path, csv_path=csv_path, aggregate=aggregate, summary=summary)
    payload = {
        "csv_path": str(csv_path.resolve()),
        "range_csv_path": str(range_csv_path.resolve()) if range_csv_path else None,
        "figure_path": str(figure_path.resolve()), "series_csv_path": str(series_path.resolve()),
        "summary_markdown_path": str(markdown_path.resolve()), "aggregate": aggregate,
        "summary": summary, "verification": verification,
    }
    if generation_config is not None:
        # Retain caller-supplied legacy metadata without reconstructing any measurements.
        payload["generation_config"] = asdict(generation_config)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render paper-style filter benchmark figures from a raw CSV export."
    )
    parser.add_argument("--csv", required=True, type=Path, help="path to point benchmark CSV")
    parser.add_argument("--range-csv", type=Path, help="add range throughput as the second panel")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/paper/filter_benchmark"),
        help="directory for the PDF and summary artifacts",
    )
    parser.add_argument(
        "--domain-size",
        type=int,
        help="legacy metadata (unused): benchmark domain size used to generate clustered keys and sampled probes",
    )
    parser.add_argument(
        "--total-keys",
        type=int,
        help="legacy metadata (unused): number of build-side keys used in the benchmark",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        help="legacy metadata (unused): number of sampled probes per cluster count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="legacy metadata (unused); measured FPR is read directly from CSV",
    )
    parser.add_argument(
        "--throughput-aggregate",
        choices=("median", "mean"),
        default="median",
        help="aggregate repetitions with this statistic before plotting each metric",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_filter_benchmark_artifacts(
        csv_path=args.csv,
        output_dir=args.output_dir,
        range_csv_path=args.range_csv,
        generation_config=BenchmarkGenerationConfig(
            domain_size=args.domain_size,
            total_keys=args.total_keys,
            probe_count=args.probe_count,
            seed=args.seed,
        ) if all(value is not None for value in (args.domain_size, args.total_keys, args.probe_count)) else None,
        aggregate=args.throughput_aggregate,
    )
    print(payload["figure_path"])
    print(payload["summary_markdown_path"])
    print(payload["summary"])


if __name__ == "__main__":
    main()
