from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

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
    return f"{value / 1_000_000:.0f}"


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
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
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
    rows: list[FilterBenchmarkRow],
    *,
    actual_negatives_by_cluster: dict[int, int],
    aggregate: str,
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row.filter_name, {}).setdefault(row.cluster_count, []).append(
            row.false_positives / actual_negatives_by_cluster[row.cluster_count]
        )
    return {
        filter_name: [
            (cluster_count, _aggregate(values, aggregate))
            for cluster_count, values in sorted(cluster_rows.items())
        ]
        for filter_name, cluster_rows in grouped.items()
    }


def derive_actual_negatives_by_cluster(
    rows: list[FilterBenchmarkRow],
    *,
    generation_config: BenchmarkGenerationConfig,
) -> dict[int, int]:
    cluster_counts = sorted({row.cluster_count for row in rows})
    helper_code = f"""
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using idx_t = std::size_t;

struct ClusterLayout {{
    std::vector<uint64_t> keys;
    idx_t cluster_length = 0;
    idx_t cluster_length_remainder = 0;
    idx_t base_gap = 0;
    idx_t gap_remainder = 0;
}};

ClusterLayout GenerateClusteredKeys(uint64_t domain_size, idx_t total_keys, idx_t cluster_count) {{
    ClusterLayout layout;
    layout.cluster_length = total_keys / cluster_count;
    layout.cluster_length_remainder = total_keys % cluster_count;
    const auto free_values = static_cast<idx_t>(domain_size - total_keys);
    layout.base_gap = free_values / cluster_count;
    layout.gap_remainder = free_values % cluster_count;
    layout.keys.reserve(total_keys);
    uint64_t cursor = 0;
    for (idx_t cluster_idx = 0; cluster_idx < cluster_count; cluster_idx++) {{
        const idx_t cluster_len = layout.cluster_length + (cluster_idx < layout.cluster_length_remainder ? 1 : 0);
        for (idx_t i = 0; i < cluster_len; i++) {{
            layout.keys.push_back(cursor + i);
        }}
        cursor += cluster_len;
        const idx_t gap = layout.base_gap + (cluster_idx < layout.gap_remainder ? 1 : 0);
        cursor += gap;
    }}
    return layout;
}}

std::vector<uint64_t> GenerateProbeKeys(uint64_t domain_size, idx_t probe_count, uint64_t seed) {{
    std::vector<uint64_t> probes;
    probes.reserve(probe_count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, domain_size - 1);
    for (idx_t i = 0; i < probe_count; i++) {{
        probes.push_back(dist(rng));
    }}
    return probes;
}}

int main() {{
    const uint64_t domain_size = {generation_config.domain_size}ULL;
    const idx_t total_keys = {generation_config.total_keys}ULL;
    const idx_t probe_count = {generation_config.probe_count}ULL;
    const uint64_t base_seed = {generation_config.seed}ULL;
    const idx_t cluster_counts[] = {{{", ".join(str(value) + "ULL" for value in cluster_counts)}}};

    std::cout << "cluster_count,actual_negatives\\n";
    for (auto cluster_count : cluster_counts) {{
        auto layout = GenerateClusteredKeys(domain_size, total_keys, cluster_count);
        std::vector<uint8_t> in_build(domain_size, 0);
        for (auto key : layout.keys) {{
            in_build[static_cast<idx_t>(key)] = 1;
        }}
        auto probes = GenerateProbeKeys(domain_size, probe_count, base_seed + cluster_count);
        idx_t positives = 0;
        for (auto probe : probes) {{
            positives += in_build[static_cast<idx_t>(probe)] != 0;
        }}
        std::cout << cluster_count << "," << (probe_count - positives) << "\\n";
    }}
}}
"""
    with tempfile.TemporaryDirectory() as temporary_dir:
        source_path = Path(temporary_dir) / "actual_negatives.cpp"
        binary_path = Path(temporary_dir) / "actual_negatives"
        source_path.write_text(helper_code, encoding="utf-8")
        subprocess.run(
            ["clang++", "-std=c++17", "-O2", str(source_path), "-o", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = subprocess.run(
            [str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return {
        int(row["cluster_count"]): int(row["actual_negatives"])
        for row in csv.DictReader(output.splitlines())
    }


def summarize_filter_benchmark(
    rows: list[FilterBenchmarkRow],
    *,
    actual_negatives_by_cluster: dict[int, int],
) -> dict[str, object]:
    filters_present = {row.filter_name for row in rows}
    systems: list[dict[str, object]] = []
    for filter_name in FILTER_ORDER:
        if filter_name not in filters_present:
            continue
        filter_rows = [row for row in rows if row.filter_name == filter_name]
        total_build_times_ms = [row.total_build_time_ns / 1_000_000.0 for row in filter_rows]
        false_positive_rates = [
            row.false_positives / actual_negatives_by_cluster[row.cluster_count]
            for row in filter_rows
        ]
        true_negatives = [
            actual_negatives_by_cluster[row.cluster_count] - row.false_positives
            for row in filter_rows
        ]
        systems.append(
            {
                "filter": filter_name,
                "label": _display_label(filter_name),
                "row_count": len(filter_rows),
                "cluster_count_min": min(row.cluster_count for row in filter_rows),
                "cluster_count_max": max(row.cluster_count for row in filter_rows),
                "actual_negatives": {
                    "min": min(actual_negatives_by_cluster[row.cluster_count] for row in filter_rows),
                    "max": max(actual_negatives_by_cluster[row.cluster_count] for row in filter_rows),
                    "avg": statistics.fmean(actual_negatives_by_cluster[row.cluster_count] for row in filter_rows),
                },
                "true_negatives": {
                    "min": min(true_negatives),
                    "max": max(true_negatives),
                    "avg": statistics.fmean(true_negatives),
                },
                "build_time_ms": {
                    "min": min(total_build_times_ms),
                    "max": max(total_build_times_ms),
                    "avg": statistics.fmean(total_build_times_ms),
                },
                "false_positive_rate": {
                    "min": min(false_positive_rates),
                    "max": max(false_positive_rates),
                    "avg": statistics.fmean(false_positive_rates),
                },
            }
        )
    return {
        "actual_negative_probe_counts": {
            "min": min(actual_negatives_by_cluster.values()),
            "max": max(actual_negatives_by_cluster.values()),
            "avg": statistics.fmean(actual_negatives_by_cluster.values()),
        },
        "system_count": len(systems),
        "systems": systems,
    }


def write_filter_benchmark_summary_markdown(
    path: Path,
    *,
    csv_path: Path,
    aggregate: str,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Filter Benchmark Summary",
        "",
        f"- CSV: `{csv_path}`",
        f"- Aggregation for figure line plots: `{aggregate}`",
        f"- FPR denominator range: `{summary['actual_negative_probe_counts']['min']}` to "
        f"`{summary['actual_negative_probe_counts']['max']}` sampled negatives per run",
        "",
        "| System | Build min (ms) | Build max (ms) | Build avg (ms) | FPR min | FPR max | FPR avg | TN min | TN max | TN avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in summary["systems"]:
        build_stats = system["build_time_ms"]
        fp_stats = system["false_positive_rate"]
        tn_stats = system["true_negatives"]
        lines.append(
            f"| {system['label']} | {_format_ms(build_stats['min'])} | {_format_ms(build_stats['max'])} | "
            f"{_format_ms(build_stats['avg'])} | {_format_percent(fp_stats['min'])} | "
            f"{_format_percent(fp_stats['max'])} | {_format_percent(fp_stats['avg'])} | "
            f"{tn_stats['min']:.0f} | {tn_stats['max']:.0f} | {tn_stats['avg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Build time is `build_time_ns + post_processing_time_ns`.",
            "For each `cluster_count`, the benchmark uses one deterministic sampled probe set shared across systems and repetitions.",
            "FPR is `false_positives / actual_negatives(cluster_count)`, i.e. `FP / (FP + TN)`, on each raw CSV row.",
            "True negatives are `actual_negatives(cluster_count) - false_positives`.",
            "",
        ]
    )
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
) -> dict[str, object]:
    width, base_height = PAGE_SIZES["one-column"]
    height = base_height + FILTER_BENCHMARK_EXTRA_HEIGHT
    fig, (throughput_ax, size_ax, build_time_ax, false_positive_rate_ax) = plt.subplots(
        4,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.95, 0.95, 0.95], "hspace": 0.08},
        constrained_layout=False,
    )
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
        "Probe throughput\n(M tuples/s)",
        fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE,
    )
    throughput_ax.set_xscale("log")
    throughput_ax.set_yscale("log")
    throughput_ax.yaxis.set_major_formatter(FuncFormatter(_format_millions))
    throughput_ax.yaxis.set_minor_formatter(LogFormatterMathtext(base=10))
    throughput_ax.set_ylim(throughput_ymin, throughput_ymax)

    size_ax.set_ylabel("Summary size\n(bytes)", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    size_ax.set_xscale("log")
    size_ax.set_yscale("log")
    size_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
    size_ax.yaxis.set_major_formatter(FuncFormatter(_format_compact_bytes))
    size_ax.yaxis.set_minor_formatter(NullFormatter())
    size_ax.set_ylim(summary_ymin, summary_ymax)
    for cache_size, label in CACHE_REFERENCE_LINES:
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
    build_time_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    build_time_ax.set_ylim(build_time_ymin, build_time_ymax)

    false_positive_rate_ax.set_xlabel("Number of clusters", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    false_positive_rate_ax.set_ylabel("False-positive\nrate", fontsize=FILTER_BENCHMARK_LABEL_FONT_SIZE)
    false_positive_rate_ax.set_xscale("log")
    false_positive_rate_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f}%"))
    false_positive_rate_ax.set_ylim(0.0, false_positive_rate_ymax)

    for ax in (throughput_ax, size_ax, build_time_ax, false_positive_rate_ax):
        ax.tick_params(axis="both", labelsize=FILTER_BENCHMARK_TICK_FONT_SIZE)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.grid(axis="x", which="major", color="#f3f4f6", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")

    legend_columns = 3
    reordered_filters = _row_major_legend_order(plotted_filters, legend_columns)
    handles, labels = throughput_ax.get_legend_handles_labels()
    handle_by_label = {label: handle for handle, label in zip(handles, labels)}
    reordered_labels = [_display_label(filter_name) for filter_name in reordered_filters]
    reordered_handles = [handle_by_label[label] for label in reordered_labels]

    fig.legend(
        reordered_handles,
        reordered_labels,
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
    fig.align_ylabels((throughput_ax, size_ax, build_time_ax, false_positive_rate_ax))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "throughput_aggregate": aggregate,
        "summary_size_aggregate": aggregate,
        "build_time_aggregate": aggregate,
        "false_positive_rate_aggregate": aggregate,
        "plotted_filters": plotted_filters,
        "point_count_per_filter": {
            filter_name: len(throughput_series[filter_name]) for filter_name in plotted_filters
        },
    }


def build_filter_benchmark_artifacts(
    *,
    csv_path: Path,
    output_dir: Path,
    generation_config: BenchmarkGenerationConfig,
    aggregate: str,
) -> dict[str, object]:
    rows = load_filter_benchmark_rows(csv_path)
    series = build_throughput_series(rows, aggregate=aggregate)
    summary_size_series = build_summary_size_series(rows, aggregate=aggregate)
    build_time_series = build_build_time_series(rows, aggregate=aggregate)
    actual_negatives_by_cluster = derive_actual_negatives_by_cluster(
        rows,
        generation_config=generation_config,
    )
    summary = summarize_filter_benchmark(
        rows,
        actual_negatives_by_cluster=actual_negatives_by_cluster,
    )
    false_positive_rate_series = build_false_positive_rate_series(
        rows,
        actual_negatives_by_cluster=actual_negatives_by_cluster,
        aggregate=aggregate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "filter_benchmark_overview.pdf"
    markdown_path = output_dir / "filter_benchmark_summary.md"
    summary_path = output_dir / "filter_benchmark_summary.json"

    verification = draw_filter_benchmark_plot(
        figure_path,
        throughput_series=series,
        summary_size_series=summary_size_series,
        build_time_series=build_time_series,
        false_positive_rate_series=false_positive_rate_series,
        aggregate=aggregate,
    )
    write_filter_benchmark_summary_markdown(
        markdown_path,
        csv_path=csv_path,
        aggregate=aggregate,
        summary=summary,
    )
    payload = {
        "csv_path": str(csv_path.resolve()),
        "figure_path": str(figure_path.resolve()),
        "summary_markdown_path": str(markdown_path.resolve()),
        "generation_config": {
            "domain_size": generation_config.domain_size,
            "total_keys": generation_config.total_keys,
            "probe_count": generation_config.probe_count,
            "seed": generation_config.seed,
        },
        "summary": summary,
        "verification": verification,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render paper-style filter benchmark figures from a raw CSV export."
    )
    parser.add_argument("--csv", required=True, type=Path, help="path to filter_benchmark_result.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/paper/filter_benchmark"),
        help="directory for the PDF and summary artifacts",
    )
    parser.add_argument(
        "--domain-size",
        type=int,
        required=True,
        help="benchmark domain size used to generate clustered keys and sampled probes",
    )
    parser.add_argument(
        "--total-keys",
        type=int,
        required=True,
        help="number of build-side keys used in the benchmark",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        required=True,
        help="number of sampled probes per cluster count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="benchmark seed; the benchmark probes with seed + cluster_count",
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
        generation_config=BenchmarkGenerationConfig(
            domain_size=args.domain_size,
            total_keys=args.total_keys,
            probe_count=args.probe_count,
            seed=args.seed,
        ),
        aggregate=args.throughput_aggregate,
    )
    print(payload["figure_path"])
    print(payload["summary_markdown_path"])
    print(payload["summary"])


if __name__ == "__main__":
    main()
