from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
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
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.filter_benchmark import FILTER_COLORS


RANGE_SERIES_ORDER = (
    "grafite",
    "diva",
    "prf_uncompressed",
    "prf",
)

RANGE_SERIES_LABELS = {
    "grafite": "Grafite",
    "diva": "Diva",
    "prf_uncompressed": "Prefix Range Filter (uncompressed)",
    "prf": "Prefix Range Filter",
}

RANGE_SERIES_LINESTYLES = {
    "grafite": "--",
    "diva": ":",
    "prf_uncompressed": "-.",
    "prf": "-",
}

PAGE_SIZE = (3.35, 1.55)
LABEL_FONT_SIZE = 7.5
TICK_FONT_SIZE = 6.5
LEGEND_FONT_SIZE = 6.5


@dataclass(frozen=True)
class RangeBenchmarkRow:
    summary: str
    mode: str
    cluster_count: int
    queries_per_sec: float
    query_kind: str = "empty"

    @property
    def series_name(self) -> str:
        if self.summary != "prf":
            return self.summary
        if self.mode == "compressed":
            return "prf"
        if self.mode == "uncompressed":
            return "prf_uncompressed"
        raise ValueError(f"unsupported PRF mode {self.mode!r}")


def _aggregate(values: list[float], aggregate: str) -> float:
    if aggregate == "median":
        return statistics.median(values)
    if aggregate == "mean":
        return statistics.fmean(values)
    raise ValueError(f"unsupported aggregate {aggregate!r}")


def load_range_benchmark_rows(csv_path: Path) -> list[RangeBenchmarkRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"summary", "mode", "queries_per_sec"}
        missing = sorted(required_columns.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing required CSV columns: {', '.join(missing)}")
        fieldnames = set(reader.fieldnames or [])
        if "cluster_count" not in fieldnames:
            raise ValueError("missing required CSV column: cluster_count")
        if "query_kind" not in fieldnames and not {"gap_selection", "gap_count"} <= fieldnames:
            raise ValueError("missing range query_kind (or legacy gap-only schema)")
        rows = [
            RangeBenchmarkRow(
                summary=row["summary"],
                mode=row["mode"],
                cluster_count=int(row["cluster_count"]),
                queries_per_sec=float(row["queries_per_sec"]),
                query_kind=row.get("query_kind", "empty"),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    return rows


def build_latency_series(
    rows: list[RangeBenchmarkRow], *, aggregate: str, query_kind: str = "empty"
) -> dict[str, list[tuple[float, float]]]:
    """Aggregate individual query latency by cluster count, expressed in microseconds."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row.query_kind != query_kind:
            continue
        if row.cluster_count <= 0:
            raise ValueError(f"cluster_count must be positive, got {row.cluster_count}")
        if not math.isfinite(row.queries_per_sec) or row.queries_per_sec <= 0:
            raise ValueError(f"queries_per_sec must be positive, got {row.queries_per_sec}")
        grouped.setdefault(row.series_name, {}).setdefault(row.cluster_count, []).append(
            1_000_000.0 / row.queries_per_sec
        )
    unknown_series = sorted(set(grouped).difference(RANGE_SERIES_ORDER))
    if unknown_series:
        raise ValueError(f"unsupported range benchmark summaries: {', '.join(unknown_series)}")
    if not grouped:
        raise ValueError(f"no range measurements for query_kind={query_kind!r}")
    return {
        series_name: [
            (cluster_count, _aggregate(latencies_us, aggregate))
            for cluster_count, latencies_us in sorted(points.items())
        ]
        for series_name, points in grouped.items()
    }


def build_range_throughput_series(
    rows: list[RangeBenchmarkRow], *, aggregate: str, query_kind: str = "empty"
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row.query_kind != query_kind:
            continue
        if row.cluster_count <= 0 or not math.isfinite(row.queries_per_sec) or row.queries_per_sec <= 0:
            raise ValueError("range cluster counts and throughput must be finite and positive")
        if row.series_name not in RANGE_SERIES_ORDER:
            raise ValueError(f"unsupported range benchmark summary: {row.series_name}")
        grouped.setdefault(row.series_name, {}).setdefault(row.cluster_count, []).append(row.queries_per_sec)
    if not grouped:
        raise ValueError(f"no range measurements for query_kind={query_kind!r}")
    return {name: [(cluster, _aggregate(values, aggregate)) for cluster, values in sorted(points.items())]
            for name, points in grouped.items()}


def _format_cluster_count(value: float, _: object) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _cluster_count_ticks(latency_series: dict[str, list[tuple[int, float]]]) -> list[float]:
    cluster_counts = [cluster_count for points in latency_series.values() for cluster_count, _ in points]
    minimum_exponent = math.floor(math.log2(min(cluster_counts)))
    maximum_exponent = math.ceil(math.log2(max(cluster_counts)))
    exponent_step = max(1, math.ceil((maximum_exponent - minimum_exponent) / 4))
    exponents = list(range(minimum_exponent, maximum_exponent + 1, exponent_step))
    if exponents[-1] != maximum_exponent:
        exponents.append(maximum_exponent)
    return [2 ** exponent for exponent in exponents]


def draw_range_latency_plot(
    output_path: Path, *, latency_series: dict[str, list[tuple[int, float]]]
) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=PAGE_SIZE, constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted_series = [name for name in RANGE_SERIES_ORDER if name in latency_series]
    for series_name in plotted_series:
        points = latency_series[series_name]
        ax.plot(
            [cluster_count for cluster_count, _ in points],
            [latency_us for _, latency_us in points],
            label=RANGE_SERIES_LABELS[series_name],
            color=FILTER_COLORS[series_name],
            linestyle=RANGE_SERIES_LINESTYLES[series_name],
            linewidth=1.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of clusters", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Range query latency (µs)", fontsize=LABEL_FONT_SIZE)
    ax.xaxis.set_major_locator(FixedLocator(_cluster_count_ticks(latency_series)))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_cluster_count))
    ax.yaxis.set_minor_formatter(LogFormatterMathtext(base=10))
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="y", which="both", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="x", which="major", color="#f3f4f6", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.18, top=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "x_scale": "log2",
        "y_scale": "log10",
        "latency_unit": "microseconds",
        "plotted_series": plotted_series,
        "point_count_per_series": {
            series_name: len(latency_series[series_name]) for series_name in plotted_series
        },
    }


def build_range_benchmark_artifacts(
    *, csv_path: Path, output_dir: Path, aggregate: str = "median"
) -> dict[str, object]:
    rows = load_range_benchmark_rows(csv_path)
    latency_series = build_latency_series(rows, aggregate=aggregate)
    figure_path = output_dir / "range_query_latency.pdf"
    summary_path = output_dir / "range_query_latency.json"
    verification = draw_range_latency_plot(figure_path, latency_series=latency_series)
    payload = {
        "csv_path": str(csv_path.resolve()),
        "figure_path": str(figure_path.resolve()),
        "aggregate": aggregate,
        "query_kind": "empty",
        "verification": verification,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render range query latency figure from a raw CSV export."
    )
    parser.add_argument("--csv", required=True, type=Path, help="path to range_results.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/output"), help="directory for PDF and JSON artifacts"
    )
    parser.add_argument(
        "--aggregate", choices=("median", "mean"), default="median", help="aggregation for repeated points"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_range_benchmark_artifacts(
        csv_path=args.csv, output_dir=args.output_dir, aggregate=args.aggregate
    )
    print(payload["figure_path"])
    print(Path(payload["figure_path"]).with_suffix(".json"))


if __name__ == "__main__":
    main()
