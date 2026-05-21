from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figure_specs import FIGURES, VARIANT_ORDER
from paper.figures.data import (
    build_absolute_table,
    load_metric_rows,
    load_system_name,
    select_queries,
    summarize_end_to_end,
    summarize_section,
    write_summary,
)

PAGE_SIZES = {
    "one-column": (3.35, 2.45),
    "two-column": (7.0, 3.15),
}

COLORS = {
    "no prefilter": "#9ca3af",
    "sip": "#2563eb",
    "sip + min/max": "#d97706",
    "prf with fallback": "#0f766e",
}


def _format_axis_value(metric: str):
    if metric == "runtime_ms":
        return FuncFormatter(lambda value, _: f"{value:.0f}")
    return FuncFormatter(lambda value, _: _format_large_number(value))


def _format_large_number(value: float) -> str:
    if value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        return f"{scaled:.0f}B" if scaled >= 100 else f"{scaled:.1f}B"
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}M" if scaled >= 100 else f"{scaled:.1f}M"
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.0f}K" if scaled >= 100 else f"{scaled:.1f}K"
    return f"{value:.0f}"


def _label_rotation(width_key: str, count: int) -> int:
    if width_key == "one-column":
        return 45
    return 30 if count > 8 else 0


def _annotate_bars(ax, rects, metric: str, ymax: float) -> None:
    if metric != "runtime_ms":
        return
    for rect in rects:
        height = rect.get_height()
        if height <= 0:
            continue
        label = f"{height:.1f}" if metric == "runtime_ms" and ymax <= 250 else _format_large_number(height)
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#111827",
            rotation=90 if len(label) > 4 else 0,
            clip_on=False,
        )


def _should_annotate_bars(width_key: str, count: int, metric: str, workload_labels: bool) -> bool:
    if metric != "runtime_ms":
        return False
    if workload_labels:
        return False
    if width_key == "one-column" and count > 6:
        return False
    if width_key == "two-column" and count > 8:
        return False
    return True


def _verify_layout(fig, ax) -> dict[str, bool]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_bboxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_text()]
    overlap = False
    for left, right in zip(tick_bboxes, tick_bboxes[1:]):
        if left.x1 > right.x0:
            overlap = True
            break
    fig_bbox = fig.get_window_extent(renderer)
    outside = False
    for artist in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < fig_bbox.x0 or bbox.x1 > fig_bbox.x1 or bbox.y0 < fig_bbox.y0 or bbox.y1 > fig_bbox.y1:
            outside = True
            break
    return {
        "tick_label_overlap": overlap,
        "text_outside_figure": outside,
    }


def _draw_vertical_grouped_bars(
    output_path: Path,
    *,
    entries: list[dict[str, object]],
    width_key: str,
    metric: str,
    y_label: str,
    workload_labels: bool,
) -> dict[str, bool]:
    width, height = PAGE_SIZES[width_key]
    count = len(entries)
    if count > 8:
        height += 0.35
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_positions = list(range(count))
    bar_width = 0.18 if count <= 9 else 0.16
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    ymax = 0.0
    for entry in entries:
        for item in entry["series"]:
            if item["value"] is not None:
                ymax = max(ymax, float(item["value"]))
    ymax = ymax * 1.18 if ymax > 0 else 1.0
    annotate_bars = _should_annotate_bars(width_key, count, metric, workload_labels)

    for series_index, variant_label in enumerate(VARIANT_ORDER):
        values = []
        for entry in entries:
            match = next((item for item in entry["series"] if item["variant_label"] == variant_label), None)
            values.append(float(match["value"]) if match and match["value"] is not None else 0.0)
        rects = ax.bar(
            [x + offsets[series_index] for x in x_positions],
            values,
            width=bar_width,
            color=COLORS[variant_label],
            edgecolor="white",
            linewidth=0.5,
            label=variant_label,
            zorder=3,
        )
        if annotate_bars:
            _annotate_bars(ax, rects, metric, ymax)

    xticklabels = []
    for entry in entries:
        label = entry["query_label"]
        if workload_labels:
            label = entry["workload_label"]
        xticklabels.append(label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        xticklabels,
        rotation=_label_rotation(width_key, count),
        ha="right" if _label_rotation(width_key, count) else "center",
        fontsize=7,
    )
    ax.set_ylabel(y_label, fontsize=8)
    ax.yaxis.set_major_formatter(_format_axis_value(metric))
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", pad=2)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.set_xlim(-0.6, count - 0.4)
    ax.set_ylim(0, ymax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(
        left=0.16 if width_key == "one-column" else 0.10,
        right=0.995,
        bottom=0.29 if count <= 9 else 0.34,
        top=0.76,
    )
    verification = _verify_layout(fig, ax)
    if verification["tick_label_overlap"] or verification["text_outside_figure"]:
        fig.set_size_inches(width, height + 0.45, forward=True)
        fig.subplots_adjust(
            left=0.17 if width_key == "one-column" else 0.11,
            right=0.995,
            bottom=0.35 if count <= 9 else 0.40,
            top=0.72,
        )
        verification = _verify_layout(fig, ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return verification


def _end_to_end_entries(rows: list) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        workload_totals = grouped.setdefault(row.workload_name, {})
        workload_totals[row.variant_label] = workload_totals.get(row.variant_label, 0.0) + row.median_ms
    entries: list[dict[str, object]] = []
    for workload_name, totals in grouped.items():
        series = []
        for variant_label in VARIANT_ORDER:
            total = totals.get(variant_label)
            if total is None:
                continue
            series.append(
                {
                    "variant_label": variant_label,
                    "value": total,
                    "median_ms": total,
                    "probe_rows_scanned": None,
                    "total_rows_scanned": None,
                }
            )
        entries.append(
            {
                "workload_name": workload_name,
                "workload_label": workload_name.replace("-", " ").title(),
                "query_name": workload_name,
                "query_label": workload_name.replace("-", " ").title(),
                "series": series,
            }
        )
    return entries


def build_figures(experiment_root: Path, system_name: str | None) -> Path:
    experiment_root = experiment_root.resolve()
    resolved_system_name = load_system_name(experiment_root, system_name)
    rows = load_metric_rows(experiment_root, resolved_system_name)
    output_dir = experiment_root / "paper"
    figures_dir = output_dir / "figures"
    summary: dict[str, object] = {
        "experiment_root": str(experiment_root),
        "system_name": resolved_system_name,
        "figures": {},
        "sections": {},
        "verification": {},
    }
    for figure in FIGURES:
        if figure.slug == "end_to_end_runtime":
            entries = _end_to_end_entries([row for row in rows if row.workload_name in figure.workloads])
            section_summary = summarize_end_to_end([row for row in rows if row.workload_name in figure.workloads])
        else:
            entries = []
            for workload_name in figure.workloads:
                query_names = select_queries(rows, workload_name, figure.queries_by_workload.get(workload_name, ()))
                entries.extend(build_absolute_table(rows, workload_name, query_names, figure.metric))
            section_summary = summarize_section(entries, figure.metric)
        figure_path = figures_dir / f"{figure.slug}.pdf"
        verification = _draw_vertical_grouped_bars(
            figure_path,
            entries=entries,
            width_key=figure.width,
            metric=figure.metric,
            y_label=figure.y_label,
            workload_labels=figure.query_workload_labels,
        )
        summary["figures"][figure.slug] = str(figure_path)
        summary["verification"][figure.slug] = verification
        summary["sections"].setdefault(figure.summary_key, {}).update({figure.slug: section_summary})
    summary["sections"]["end_to_end"] = summarize_end_to_end(
        [row for row in rows if row.workload_name in FIGURES[-1].workloads]
    )
    write_summary(output_dir / "summary.json", summary)
    _write_summary_markdown(output_dir / "summary.md", summary)
    return output_dir


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Paper Summary",
        "",
        f"- experiment root: `{summary['experiment_root']}`",
        f"- system: `{summary['system_name']}`",
        "",
    ]
    for section_name, section_value in summary["sections"].items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section_value, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["verification"], indent=2))
    lines.append("```")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper figures and summaries from one experiment root.")
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--system-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = build_figures(args.experiment_root, args.system_name)
    print(output_dir)


if __name__ == "__main__":
    main()
