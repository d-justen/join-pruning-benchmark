from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figure_specs import (
    FIGURES,
    JPB_ALL_QUERIES,
    STRING_JOIN_QUERIES,
    TPCH_SELECTED_QUERIES,
    VARIANT_ORDER,
    WORKLOAD_LABELS,
)
from paper.figures.data import (
    build_absolute_table,
    build_benchmark_runtime_distributions,
    build_runtime_plan_work_table,
    build_runtime_probe_table,
    build_relative_probe_cardinality_outliers,
    build_relative_probe_cardinality_values,
    load_metric_rows,
    load_plan_metric_rows,
    load_probe_scan_rows,
    load_system_name,
    select_queries,
    summarize_relative_probe_cardinality,
    summarize_benchmark_runtime,
    summarize_benchmark_runtime_distributions,
    summarize_benchmark_metrics,
    summarize_end_to_end,
    summarize_section,
    write_summary,
)

PAGE_SIZES = {
    "one-column": (3.35, 2.45),
    "two-column": (7.0, 3.15),
}

TITLE_FONT_SIZE = 8.0
LABEL_FONT_SIZE = 8.0
TICK_FONT_SIZE = 7.0
LEGEND_FONT_SIZE = 7.0
ANNOTATION_FONT_SIZE = 6.5

COLORS = {
    "no prefilter": "#9ca3af",
    "sip": "#2563eb",
    "sip + min/max": "#d97706",
    "prf with fallback": "#0f766e",
}

PROBE_CARDINALITY_VARIANTS = ("sip", "sip + min/max", "prf with fallback")
PROBE_CARDINALITY_OUTLIER_THRESHOLD = 10.0
PROBE_OUTPUT_HATCH = "////"
PROBE_FILTERED_HATCH = "...."
ROWS_SCANNED_HATCH = "////"
BENCHMARK_RUNTIME_BOXPLOT_GROUPS = (
    (
        "join-pruning-workload",
        "JPB",
        JPB_ALL_QUERIES,
    ),
    (
        "tpch-skew",
        "TPC-H Skew",
        None,
    ),
    (
        "join-order-benchmark",
        "JOB",
        None,
    ),
)
VARIANT_DISPLAY_LABELS = {
    "no prefilter": "No pre-filter",
    "sip": "Bloom",
    "sip + min/max": "Bloom + min/max",
    "prf with fallback": "Prefix Range Filter",
}
VARIANT_PANEL_LABELS = {
    "no prefilter": "No\npre-\nfilter",
    "sip": "Bloom",
    "sip + min/max": "Bloom +\nmin/max",
    "prf with fallback": "Prefix\nRange\nFilter",
}
MULTI_JOIN_QUERY_PANEL_SIZE = 24


def _available_workload_names(rows: list) -> set[str]:
    return {row.workload_name for row in rows}


def _figure_is_available(figure, available_workloads: set[str]) -> bool:
    if len(figure.workloads) > 1:
        return all(workload_name in available_workloads for workload_name in figure.workloads)
    return any(workload_name in available_workloads for workload_name in figure.workloads)


def _present_benchmark_groups(rows: list) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    available_workloads = _available_workload_names(rows)
    return tuple(
        group for group in BENCHMARK_RUNTIME_BOXPLOT_GROUPS
        if group[0] in available_workloads
    )


def _variant_display_label(label: str) -> str:
    return VARIANT_DISPLAY_LABELS.get(label, label)


def _variant_panel_label(label: str) -> str:
    return VARIANT_PANEL_LABELS.get(label, _variant_display_label(label))


def _format_axis_value(metric: str):
    if metric == "runtime_ms":
        return FuncFormatter(lambda value, _: f"{value:.0f}")
    return FuncFormatter(lambda value, _: _format_large_number(value))


def _format_large_number(value: float) -> str:
    if value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        return f"{scaled:.0f}B" if float(scaled).is_integer() else f"{scaled:.1f}B"
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}M" if float(scaled).is_integer() else f"{scaled:.1f}M"
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.0f}K" if float(scaled).is_integer() else f"{scaled:.1f}K"
    return f"{value:.0f}"


def _log_axis_bounds(
    values: list[float],
    *,
    min_floor: float = 1.0,
    upper_padding: float = 1.15,
) -> tuple[float, float] | None:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return None
    lower_power = math.floor(math.log10(min(positive_values)))
    max_value = max(positive_values)
    lower = max(min_floor, 10 ** lower_power)
    upper = max(lower * 1.01, max_value * upper_padding)
    return lower, upper


def _render_log_value(value: float, floor: float) -> float:
    if value <= 0:
        return floor
    return max(value, floor)


def _drop_seam_tick(ax) -> None:
    ticks = list(ax.get_yticks())
    if len(ticks) <= 1:
        return
    ax.set_yticks(ticks[1:])


def _draw_axis_break_marks(upper_ax, lower_ax) -> None:
    break_kwargs = dict(color="#6b7280", clip_on=False, linewidth=0.8)
    upper_ax.plot((-0.012, +0.012), (-0.015, +0.015), transform=upper_ax.transAxes, **break_kwargs)
    lower_ax.plot((-0.012, +0.012), (1.0 - 0.015, 1.0 + 0.015), transform=lower_ax.transAxes, **break_kwargs)


def _label_rotation(width_key: str, count: int) -> int:
    if width_key == "one-column":
        return 45
    return 30 if count > 8 else 0


def _text_scale(width_key: str, placement_width_key: str) -> float:
    return PAGE_SIZES[width_key][0] / PAGE_SIZES[placement_width_key][0]


def _capped_text_scale(width_key: str, placement_width_key: str, *, maximum: float = 1.45) -> float:
    return min(_text_scale(width_key, placement_width_key), maximum)


def _log_ticks(lower: float, upper: float) -> list[float]:
    lower_power = int(math.floor(math.log10(lower)))
    upper_power = int(math.floor(math.log10(upper)))
    return [10 ** power for power in range(lower_power, upper_power + 1)]


def _scaled_font_size(size: float, scale: float) -> float:
    return round(size * scale, 2)


def _wrap_tick_label(label: str, text_scale: float) -> str:
    if text_scale < 1.5 or " " not in label:
        return label
    words = label.split()
    if len(words) == 2:
        return "\n".join(words)
    midpoint = math.ceil(len(words) / 2)
    return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])


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
    placement_width_key: str,
    metric: str,
    y_label: str,
    workload_labels: bool,
) -> dict[str, bool]:
    width, height = PAGE_SIZES[width_key]
    text_scale = _text_scale(width_key, placement_width_key)
    tick_font_size = _scaled_font_size(TICK_FONT_SIZE, text_scale)
    label_font_size = _scaled_font_size(LABEL_FONT_SIZE, text_scale)
    legend_font_size = _scaled_font_size(LEGEND_FONT_SIZE, text_scale)
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
            label=_variant_display_label(variant_label),
            zorder=3,
        )

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
        fontsize=tick_font_size,
    )
    ax.set_ylabel(y_label, fontsize=label_font_size)
    ax.yaxis.set_major_formatter(_format_axis_value(metric))
    ax.tick_params(axis="y", labelsize=tick_font_size)
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
        fontsize=legend_font_size,
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
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return verification


def _draw_runtime_probe_bars(
    output_path: Path,
    *,
    entries: list[dict[str, object]],
    width_key: str,
    placement_width_key: str,
    show_query_labels: bool,
    annotate_runtime_values: bool,
    runtime_log_axis: bool,
    rows_log_axis: bool,
) -> dict[str, bool]:
    width, height = PAGE_SIZES[width_key]
    text_scale = _capped_text_scale(width_key, placement_width_key)
    compressed_column_layout = placement_width_key == "one-column" and width_key != placement_width_key
    tick_font_size = _scaled_font_size(TICK_FONT_SIZE, text_scale)
    label_font_size = _scaled_font_size(LABEL_FONT_SIZE, text_scale)
    legend_font_size = _scaled_font_size(LEGEND_FONT_SIZE, text_scale)
    annotation_font_size = _scaled_font_size(ANNOTATION_FONT_SIZE - 0.5, text_scale)
    has_bar_labels = annotate_runtime_values
    height += 0.95 if compressed_column_layout else 1.0
    if has_bar_labels:
        height += 0.12
    mirrored_log_layout = runtime_log_axis or rows_log_axis
    panel_hspace = 0.055 if compressed_column_layout else (0.035 if mirrored_log_layout else 0.0)
    fig, (runtime_ax, rows_ax) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.15],
            "hspace": panel_hspace,
        },
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    runtime_ax.set_facecolor("white")
    rows_ax.set_facecolor("white")

    count = len(entries)
    x_positions = list(range(count))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    runtime_max = 0.0
    rows_max = 0
    positive_runtime_values: list[float] = []
    positive_row_values: list[float] = []

    series_data: list[tuple[str, list[float], list[int], list[int]]] = []
    for variant_label in VARIANT_ORDER:
        runtime_values: list[float] = []
        output_values: list[int] = []
        filtered_values: list[int] = []
        for entry in entries:
            match = next(
                (item for item in entry["series"] if item["variant_label"] == variant_label),
                None,
            )
            runtime = float(match["median_ms"]) if match else 0.0
            output_rows = int(match["probe_output_cardinality"]) if match else 0
            filtered_rows = int(match["probe_rows_filtered_out"]) if match else 0
            runtime_values.append(runtime)
            output_values.append(output_rows)
            filtered_values.append(filtered_rows)
            runtime_max = max(runtime_max, runtime)
            rows_max = max(rows_max, output_rows + filtered_rows)
            if runtime > 0:
                positive_runtime_values.append(runtime)
            if output_rows > 0:
                positive_row_values.append(float(output_rows))
            if output_rows + filtered_rows > 0:
                positive_row_values.append(float(output_rows + filtered_rows))
        series_data.append((variant_label, runtime_values, output_values, filtered_values))

    runtime_bounds = _log_axis_bounds(positive_runtime_values)
    runtime_floor = runtime_bounds[0] if runtime_bounds else 1.0
    row_bounds = _log_axis_bounds(positive_row_values)
    row_floor = row_bounds[0] if row_bounds else 1.0

    for series_index, (variant_label, runtime_values, output_values, filtered_values) in enumerate(series_data):
        positions = [x + offsets[series_index] for x in x_positions]

        if runtime_log_axis:
            rendered_runtime_values = [
                _render_log_value(value, runtime_floor)
                for value in runtime_values
            ]
            runtime_ax.bar(
                positions,
                [value - runtime_floor for value in rendered_runtime_values],
                width=bar_width,
                bottom=runtime_floor,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        else:
            runtime_ax.bar(
                positions,
                runtime_values,
                width=bar_width,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        if annotate_runtime_values:
            label_values = rendered_runtime_values if runtime_log_axis else runtime_values
            for position, value, label_value in zip(positions, runtime_values, label_values):
                if value <= 0:
                    continue
                runtime_ax.annotate(
                    f"{value:.0f}",
                    xy=(position, label_value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_font_size,
                    color="#111827",
                    clip_on=False,
                    zorder=5,
                )

        if rows_log_axis:
            rendered_output_values = [
                _render_log_value(value, row_floor)
                for value in output_values
            ]
            rendered_total_values = [
                _render_log_value(output + filtered, row_floor)
                for output, filtered in zip(output_values, filtered_values)
            ]
            rows_ax.bar(
                positions,
                [top - row_floor for top in rendered_output_values],
                width=bar_width,
                bottom=row_floor,
                color=COLORS[variant_label],
                edgecolor="#ffffff",
                linewidth=0.45,
                hatch=PROBE_OUTPUT_HATCH,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                [top - bottom for top, bottom in zip(rendered_total_values, rendered_output_values)],
                width=bar_width,
                bottom=rendered_output_values,
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_FILTERED_HATCH,
                zorder=3,
            )
        else:
            rows_ax.bar(
                positions,
                output_values,
                width=bar_width,
                color=COLORS[variant_label],
                edgecolor="#ffffff",
                linewidth=0.45,
                hatch=PROBE_OUTPUT_HATCH,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                filtered_values,
                width=bar_width,
                bottom=output_values,
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_FILTERED_HATCH,
                zorder=3,
            )

    runtime_ax.set_ylabel(
        "Time (ms)" if compressed_column_layout else "Execution time (ms)",
        fontsize=label_font_size,
        labelpad=10,
    )
    if runtime_log_axis:
        runtime_bounds = _log_axis_bounds(
            positive_runtime_values,
            upper_padding=1.35 if annotate_runtime_values else 1.15,
        )
        if runtime_bounds:
            runtime_floor, runtime_upper = runtime_bounds
            runtime_ax.set_yscale("log")
            runtime_ax.set_ylim(runtime_floor, runtime_upper)
            runtime_ax.set_yticks(_log_ticks(runtime_floor, runtime_upper))
            runtime_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            _drop_seam_tick(runtime_ax)
        else:
            runtime_ax.set_ylim(1.0, 10.0)
    else:
        runtime_ax.yaxis.set_major_formatter(_format_axis_value("runtime_ms"))
        runtime_ax.set_ylim(0, runtime_max * 1.15 if runtime_max > 0 else 1.0)
    rows_ax.set_ylabel(
        "Probe rows",
        fontsize=label_font_size,
        labelpad=10,
    )
    if rows_log_axis:
        row_bounds = _log_axis_bounds(positive_row_values)
        if row_bounds:
            row_floor, row_upper = row_bounds
            rows_ax.set_yscale("log")
            rows_ax.set_ylim(row_floor, row_upper)
            rows_ax.set_yticks(_log_ticks(row_floor, row_upper))
            rows_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            _drop_seam_tick(rows_ax)
        else:
            rows_ax.set_ylim(1.0, 10.0)
    else:
        rows_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_large_number(value)))
        rows_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        rows_ax.set_ylim(0, rows_max * 1.03 if rows_max > 0 else 1.0)
    rows_ax.invert_yaxis()

    for ax in (runtime_ax, rows_ax):
        ax.tick_params(axis="y", labelsize=tick_font_size)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
    runtime_ax.spines["top"].set_visible(False)
    runtime_ax.spines["bottom"].set_color("#6b7280")
    rows_ax.spines["top"].set_color("#6b7280")
    rows_ax.spines["bottom"].set_visible(False)
    if mirrored_log_layout:
        _draw_axis_break_marks(runtime_ax, rows_ax)

    has_wrapped_xticklabels = False
    if show_query_labels:
        xticklabels = [
            _wrap_tick_label(str(entry["query_label"]), text_scale)
            for entry in entries
        ]
        has_wrapped_xticklabels = any("\n" in label for label in xticklabels)
        rows_ax.set_xticks(x_positions)
        rows_ax.set_xticklabels(
            xticklabels,
            fontsize=tick_font_size,
        )
    else:
        rows_ax.set_xticks([])
    rows_ax.tick_params(axis="x", pad=4, length=0)
    rows_ax.set_xlim(-0.6, count - 0.4)

    variant_handles = [
        Patch(facecolor=COLORS[label], edgecolor="none", label=_variant_display_label(label))
        for label in VARIANT_ORDER
    ]
    semantic_handles = [
        Patch(
            facecolor="white",
            edgecolor="#4b5563",
            hatch=PROBE_OUTPUT_HATCH,
            label="Join-probed tuples",
        ),
        Patch(
            facecolor="white",
            edgecolor="#4b5563",
            hatch=PROBE_FILTERED_HATCH,
            label="Pre-filtered tuples",
        ),
    ]
    legend_handles = variant_handles + semantic_handles
    if width_key == "one-column":
        # Matplotlib fills multi-column legends column-first.
        legend_handles = [
            variant_handles[0],
            variant_handles[2],
            semantic_handles[0],
            variant_handles[1],
            variant_handles[3],
            semantic_handles[1],
        ]
    if has_bar_labels:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.89),
            ncol=3,
            frameon=False,
            fontsize=legend_font_size,
            handlelength=1.5,
            columnspacing=1.0,
        )
    else:
        runtime_ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.40 if compressed_column_layout else (1.42 if width_key == "one-column" else 1.31)),
            ncol=3 if compressed_column_layout else (2 if width_key == "one-column" else 3),
            frameon=False,
            fontsize=legend_font_size,
            handlelength=1.5,
            columnspacing=1.1,
        )

    fig.subplots_adjust(
        left=0.26 if width_key == "one-column" else (0.17 if compressed_column_layout else 0.10),
        right=0.992 if width_key == "one-column" else 0.997,
        bottom=0.08 if not show_query_labels else (0.18 if has_wrapped_xticklabels else (0.12 if compressed_column_layout else 0.11)),
        top=0.74 if has_bar_labels else (0.72 if compressed_column_layout else (0.75 if width_key == "one-column" else 0.83)),
    )
    runtime_ax.yaxis.set_label_coords(-0.11, 0.5)
    rows_ax.yaxis.set_label_coords(-0.11, 0.5)
    runtime_verification = _verify_layout(fig, runtime_ax)
    rows_verification = _verify_layout(fig, rows_ax)
    verification = {
        "tick_label_overlap": (
            runtime_verification["tick_label_overlap"]
            or rows_verification["tick_label_overlap"]
        ),
        "text_outside_figure": (
            runtime_verification["text_outside_figure"]
            or rows_verification["text_outside_figure"]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format=output_path.suffix.removeprefix(".") or "pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return verification


def _draw_runtime_total_rows_bars(
    output_path: Path,
    *,
    entries: list[dict[str, object]],
    width_key: str,
    placement_width_key: str,
    show_query_labels: bool,
    show_runtime_labels: bool,
    workload_labels: bool,
    runtime_log_axis: bool,
    rows_log_axis: bool,
) -> dict[str, bool]:
    width, height = PAGE_SIZES[width_key]
    text_scale = _capped_text_scale(width_key, placement_width_key)
    compressed_column_layout = placement_width_key == "one-column" and width_key != placement_width_key
    tick_font_size = _scaled_font_size(TICK_FONT_SIZE, text_scale)
    label_font_size = _scaled_font_size(LABEL_FONT_SIZE, text_scale)
    legend_font_size = _scaled_font_size(LEGEND_FONT_SIZE, text_scale)
    annotation_font_size = _scaled_font_size(ANNOTATION_FONT_SIZE, text_scale)
    height += 1.0 if compressed_column_layout else 0.85
    if show_runtime_labels:
        height += 0.10
    mirrored_log_layout = runtime_log_axis or rows_log_axis
    fig, (runtime_ax, rows_ax) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.05],
            "hspace": 0.035 if mirrored_log_layout else 0.0,
        },
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    runtime_ax.set_facecolor("white")
    rows_ax.set_facecolor("white")

    count = len(entries)
    x_positions = list(range(count))
    bar_width = 0.15 if show_runtime_labels else 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    runtime_max = 0.0
    rows_max = 0.0
    positive_runtime_values: list[float] = []
    positive_row_values: list[float] = []

    series_data: list[tuple[str, list[float], list[float]]] = []
    for variant_label in VARIANT_ORDER:
        runtime_values: list[float] = []
        row_values: list[float] = []
        for entry in entries:
            match = next(
                (item for item in entry["series"] if item["variant_label"] == variant_label),
                None,
            )
            runtime = float(match["median_ms"]) if match else 0.0
            rows_scanned = float(match["rows_scanned"]) if match else 0.0
            runtime_values.append(runtime)
            row_values.append(rows_scanned)
            runtime_max = max(runtime_max, runtime)
            rows_max = max(rows_max, rows_scanned)
            if runtime > 0:
                positive_runtime_values.append(runtime)
            if rows_scanned > 0:
                positive_row_values.append(rows_scanned)
        series_data.append((variant_label, runtime_values, row_values))

    runtime_bounds = _log_axis_bounds(positive_runtime_values)
    runtime_floor = runtime_bounds[0] if runtime_bounds else 1.0
    row_bounds = _log_axis_bounds(positive_row_values)
    row_floor = row_bounds[0] if row_bounds else 1.0

    for series_index, (variant_label, runtime_values, row_values) in enumerate(series_data):
        positions = [x + offsets[series_index] for x in x_positions]
        if runtime_log_axis:
            rendered_runtime_values = [
                _render_log_value(value, runtime_floor)
                for value in runtime_values
            ]
            runtime_ax.bar(
                positions,
                [value - runtime_floor for value in rendered_runtime_values],
                width=bar_width,
                bottom=runtime_floor,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        else:
            runtime_ax.bar(
                positions,
                runtime_values,
                width=bar_width,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        if show_runtime_labels:
            label_values = rendered_runtime_values if runtime_log_axis else runtime_values
            for position, value, label_value in zip(positions, runtime_values, label_values):
                if value <= 0:
                    continue
                runtime_ax.annotate(
                    f"{value:.0f}",
                    xy=(position, label_value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_font_size,
                    color="#111827",
                    clip_on=False,
                    zorder=5,
                )
        if rows_log_axis:
            rendered_row_values = [
                _render_log_value(value, row_floor)
                for value in row_values
            ]
            rows_ax.bar(
                positions,
                [value - row_floor for value in rendered_row_values],
                width=bar_width,
                bottom=row_floor,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        else:
            rows_ax.bar(
                positions,
                row_values,
                width=bar_width,
                color=COLORS[variant_label],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    runtime_ax.set_ylabel("Time (ms)", fontsize=label_font_size)
    if runtime_log_axis:
        runtime_bounds = _log_axis_bounds(
            positive_runtime_values,
            upper_padding=1.35 if show_runtime_labels else 1.15,
        )
        if runtime_bounds:
            runtime_floor, runtime_upper = runtime_bounds
            runtime_ax.set_yscale("log")
            runtime_ax.set_ylim(runtime_floor, runtime_upper)
            runtime_ax.set_yticks(_log_ticks(runtime_floor, runtime_upper))
            runtime_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            _drop_seam_tick(runtime_ax)
        else:
            runtime_ax.set_ylim(1.0, 10.0)
    else:
        runtime_ax.yaxis.set_major_formatter(_format_axis_value("runtime_ms"))
        runtime_ax.set_ylim(0, runtime_max * 1.15 if runtime_max > 0 else 1.0)

    rows_ax.set_ylabel("Rows scanned", fontsize=label_font_size)
    if rows_log_axis:
        if row_bounds:
            row_floor, row_upper = row_bounds
            rows_ax.set_yscale("log")
            rows_ax.set_ylim(row_floor, row_upper)
            rows_ax.set_yticks(_log_ticks(row_floor, row_upper))
            rows_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            _drop_seam_tick(rows_ax)
        else:
            rows_ax.set_ylim(1.0, 10.0)
    else:
        rows_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_large_number(value)))
        rows_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        rows_ax.set_ylim(0, rows_max * 1.03 if rows_max > 0 else 1.0)
    rows_ax.invert_yaxis()

    for ax in (runtime_ax, rows_ax):
        ax.tick_params(axis="y", labelsize=tick_font_size)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
    runtime_ax.spines["top"].set_visible(False)
    runtime_ax.spines["bottom"].set_color("#6b7280")
    rows_ax.spines["top"].set_color("#6b7280")
    rows_ax.spines["bottom"].set_visible(False)
    if mirrored_log_layout:
        _draw_axis_break_marks(runtime_ax, rows_ax)

    has_wrapped_xticklabels = False
    if show_query_labels:
        xticklabels = []
        for entry in entries:
            label = str(entry["query_label"])
            if workload_labels:
                label = str(entry["workload_label"])
            xticklabels.append(_wrap_tick_label(label, text_scale))
        has_wrapped_xticklabels = any("\n" in label for label in xticklabels)
        rows_ax.set_xticks(x_positions)
        rows_ax.set_xticklabels(xticklabels, fontsize=7)
        for tick in rows_ax.get_xticklabels():
            tick.set_fontsize(tick_font_size)
    else:
        rows_ax.set_xticks([])
    rows_ax.tick_params(axis="x", pad=4, length=0)
    rows_ax.set_xlim(-0.6, count - 0.4)

    total_rows_legend_handles = [
        Patch(
            facecolor=COLORS[label],
            edgecolor="none",
            label=_variant_display_label(label),
        )
        for label in VARIANT_ORDER
    ]
    if show_runtime_labels:
        fig.legend(
            handles=total_rows_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.89),
            ncol=2,
            frameon=False,
            fontsize=legend_font_size,
            handlelength=1.5,
            columnspacing=1.0,
        )
    else:
        runtime_ax.legend(
            handles=total_rows_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.36 if compressed_column_layout else (1.40 if width_key == "one-column" else 1.30)),
            ncol=2 if width_key == "one-column" else (2 if compressed_column_layout else 4),
            frameon=False,
            fontsize=legend_font_size,
            handlelength=1.5,
            columnspacing=1.1,
        )

    fig.subplots_adjust(
        left=0.24 if width_key == "one-column" else (0.14 if compressed_column_layout else 0.10),
        right=0.992 if width_key == "one-column" else 0.997,
        bottom=0.08 if not show_query_labels else (0.18 if has_wrapped_xticklabels else (0.13 if compressed_column_layout else 0.11)),
        top=0.74 if show_runtime_labels else (0.70 if compressed_column_layout else (0.74 if width_key == "one-column" else 0.83)),
    )
    runtime_ax.yaxis.set_label_coords(-0.14 if show_runtime_labels else -0.11, 0.5)
    rows_ax.yaxis.set_label_coords(-0.14 if show_runtime_labels else -0.11, 0.5)
    runtime_verification = _verify_layout(fig, runtime_ax)
    rows_verification = _verify_layout(fig, rows_ax)
    verification = {
        "tick_label_overlap": (
            runtime_verification["tick_label_overlap"]
            or rows_verification["tick_label_overlap"]
        ),
        "text_outside_figure": (
            runtime_verification["text_outside_figure"]
            or rows_verification["text_outside_figure"]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format=output_path.suffix.removeprefix(".") or "pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return verification


def _draw_runtime_plan_work_bars(
    output_path: Path,
    *,
    entries: list[dict[str, object]],
    width_key: str,
    show_query_labels: bool,
    log_lower_axis: bool = True,
) -> dict[str, bool]:
    width, height = PAGE_SIZES[width_key]
    if len(entries) > 16:
        height += 0.55
    else:
        height += 0.90
    mirrored_log_layout = log_lower_axis
    fig, (runtime_ax, work_ax) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.2],
            "hspace": 0.035 if mirrored_log_layout else 0.0,
        },
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    runtime_ax.set_facecolor("white")
    work_ax.set_facecolor("white")

    count = len(entries)
    x_positions = list(range(count))
    variant_step = 0.18 if count <= 24 else 0.16
    measure_width = 0.06 if count <= 24 else 0.05
    variant_offsets = [-1.5 * variant_step, -0.5 * variant_step, 0.5 * variant_step, 1.5 * variant_step]
    runtime_max = 0.0
    positive_work_values: list[float] = []

    for series_index, variant_label in enumerate(VARIANT_ORDER):
        runtime_values: list[float] = []
        join_probe_values: list[int] = []
        rows_scanned_values: list[int] = []
        for entry in entries:
            match = next(
                (item for item in entry["series"] if item["variant_label"] == variant_label),
                None,
            )
            runtime = float(match["median_ms"]) if match else 0.0
            join_probe_tuples = int(match["join_probe_tuples"]) if match else 0
            rows_scanned = int(match["rows_scanned"]) if match else 0
            runtime_values.append(runtime)
            join_probe_values.append(join_probe_tuples)
            rows_scanned_values.append(rows_scanned)
            runtime_max = max(runtime_max, runtime)
            if join_probe_tuples > 0:
                positive_work_values.append(join_probe_tuples)
            if rows_scanned > 0:
                positive_work_values.append(rows_scanned)

        variant_centers = [x + variant_offsets[series_index] for x in x_positions]
        runtime_ax.bar(
            variant_centers,
            runtime_values,
            width=variant_step * 0.92,
            color=COLORS[variant_label],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        work_ax.bar(
            [center - measure_width / 2 for center in variant_centers],
            join_probe_values,
            width=measure_width,
            color=COLORS[variant_label],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        work_ax.bar(
            [center + measure_width / 2 for center in variant_centers],
            rows_scanned_values,
            width=measure_width,
            color="white",
            edgecolor=COLORS[variant_label],
            linewidth=0.7,
            hatch=ROWS_SCANNED_HATCH,
            zorder=3,
        )

    runtime_ax.set_ylabel("Execution time (ms)", fontsize=8)
    runtime_ax.yaxis.set_major_formatter(_format_axis_value("runtime_ms"))
    runtime_ax.set_ylim(0, runtime_max * 1.15 if runtime_max > 0 else 1.0)
    work_ax.set_ylabel("Tuples (log)", fontsize=8 if log_lower_axis else 8)
    work_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_large_number(value)))
    if log_lower_axis and positive_work_values:
        lower_power = math.floor(math.log10(min(positive_work_values)))
        upper_power = math.ceil(math.log10(max(positive_work_values)))
        work_ax.set_yscale("log")
        work_ax.set_ylim(10 ** lower_power, 10 ** upper_power)
        work_ax.set_yticks([10 ** power for power in range(lower_power, upper_power + 1)])
        work_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        _drop_seam_tick(work_ax)
    else:
        ymax = max(positive_work_values) if positive_work_values else 1.0
        work_ax.set_ylim(0, ymax * 1.05)
        work_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    work_ax.invert_yaxis()

    for ax in (runtime_ax, work_ax):
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", which="major", color="#e5e7eb", linewidth=0.8, zorder=0)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
    runtime_ax.spines["top"].set_visible(False)
    runtime_ax.spines["bottom"].set_color("#6b7280")
    work_ax.spines["top"].set_color("#6b7280")
    work_ax.spines["bottom"].set_visible(False)
    if mirrored_log_layout:
        _draw_axis_break_marks(runtime_ax, work_ax)

    if show_query_labels:
        work_ax.set_xticks(x_positions)
        work_ax.set_xticklabels(
            [str(entry["query_label"]) for entry in entries],
            rotation=_label_rotation(width_key, count),
            ha="right" if _label_rotation(width_key, count) else "center",
            fontsize=7,
        )
    else:
        work_ax.set_xticks([])
    work_ax.tick_params(axis="x", pad=4, length=0)
    work_ax.set_xlim(-0.7, count - 0.3)

    variant_handles = [
        Patch(facecolor=COLORS[label], edgecolor="none", label=_variant_display_label(label))
        for label in VARIANT_ORDER
    ]
    semantic_handles = [
        Patch(facecolor="white", edgecolor="#4b5563", label="Join-probed tuples"),
        Patch(facecolor="white", edgecolor="#4b5563", hatch=ROWS_SCANNED_HATCH, label="Rows scanned"),
    ]
    runtime_ax.legend(
        handles=variant_handles + semantic_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=1.5,
        columnspacing=1.1,
    )

    fig.subplots_adjust(
        left=0.10 if width_key == "two-column" else 0.15,
        right=0.997,
        bottom=0.18 if show_query_labels else 0.09,
        top=0.83,
    )
    runtime_verification = _verify_layout(fig, runtime_ax)
    work_verification = _verify_layout(fig, work_ax)
    verification = {
        "tick_label_overlap": runtime_verification["tick_label_overlap"] or work_verification["tick_label_overlap"],
        "text_outside_figure": runtime_verification["text_outside_figure"] or work_verification["text_outside_figure"],
    }
    if verification["tick_label_overlap"] or verification["text_outside_figure"]:
        fig.set_size_inches(width, height + 0.35, forward=True)
        fig.subplots_adjust(
            left=0.12 if width_key == "two-column" else 0.17,
            right=0.995,
            bottom=0.26 if show_query_labels else 0.12,
            top=0.80,
        )
        runtime_verification = _verify_layout(fig, runtime_ax)
        work_verification = _verify_layout(fig, work_ax)
        verification = {
            "tick_label_overlap": runtime_verification["tick_label_overlap"] or work_verification["tick_label_overlap"],
            "text_outside_figure": runtime_verification["text_outside_figure"] or work_verification["text_outside_figure"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format=output_path.suffix.removeprefix(".") or "pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return verification


def _draw_relative_probe_cardinality_boxplot(
    output_path: Path,
    *,
    relative_values: dict[str, list[float]],
) -> dict[str, bool]:
    width, height = PAGE_SIZES["two-column"]
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    data = [relative_values[label] for label in PROBE_CARDINALITY_VARIANTS]
    positions = [1, 2, 3]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.0},
        boxprops={"linewidth": 0.9},
        whiskerprops={"linewidth": 0.9, "color": "#6b7280"},
        capprops={"linewidth": 0.9, "color": "#6b7280"},
    )
    for patch, label in zip(bp["boxes"], PROBE_CARDINALITY_VARIANTS):
        patch.set_facecolor(COLORS[label])
        patch.set_edgecolor("#374151")
        patch.set_alpha(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels([_variant_display_label(label) for label in PROBE_CARDINALITY_VARIANTS], fontsize=7)
    ax.set_ylabel("Relative scan cardinality (log)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    all_values = sorted(value for values in data for value in values if value >= 1.0)
    if all_values:
        q99_index = min(len(all_values) - 1, int(0.99 * (len(all_values) - 1)))
        ymax = all_values[q99_index]
    else:
        ymax = 1.0
    ax.set_yscale("log")
    upper = max(1.2, ymax * 1.15)
    ax.set_ylim(1.0, upper)
    max_power = max(0, math.ceil(math.log10(upper)))
    ax.set_yticks([10 ** power for power in range(0, max_power + 1)])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.18, top=0.96)
    verification = _verify_layout(fig, ax)
    if verification["tick_label_overlap"] or verification["text_outside_figure"]:
        fig.set_size_inches(width, height + 0.35, forward=True)
        fig.subplots_adjust(left=0.21, right=0.995, bottom=0.22, top=0.96)
        verification = _verify_layout(fig, ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return verification


def _draw_benchmark_runtime_boxplot(
    output_path: Path,
    *,
    distributions: list[dict[str, object]],
    placement_width_key: str,
) -> dict[str, bool]:
    width_key = "two-column"
    width, height = PAGE_SIZES[width_key]
    text_scale = _capped_text_scale(width_key, placement_width_key)
    compressed_column_layout = placement_width_key == "one-column"
    tick_font_size = _scaled_font_size(TICK_FONT_SIZE, text_scale)
    title_font_size = _scaled_font_size(TITLE_FONT_SIZE, text_scale)
    label_font_size = _scaled_font_size(LABEL_FONT_SIZE, text_scale)
    legend_font_size = _scaled_font_size(LEGEND_FONT_SIZE, text_scale)
    fig, axes = plt.subplots(
        1,
        len(distributions),
        figsize=(width, height + (0.55 if compressed_column_layout else 0.35)),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = [axes]

    positions = (1, 2, 3, 4)
    for index, (ax, distribution) in enumerate(zip(axes, distributions)):
        ax.set_facecolor("white")
        panel_values = []
        for variant_index, variant_label in enumerate(VARIANT_ORDER):
            values = distribution["values_by_variant"][variant_label]
            panel_values.extend(values)
            boxplot = ax.boxplot(
                [values],
                positions=[positions[variant_index]],
                widths=0.68,
                patch_artist=True,
                showfliers=True,
                whis=1.5,
                medianprops={"color": "#111827", "linewidth": 1.0},
                boxprops={"linewidth": 0.8, "edgecolor": "#374151"},
                whiskerprops={"linewidth": 0.8, "color": "#6b7280"},
                capprops={"linewidth": 0.8, "color": "#6b7280"},
                flierprops={
                    "marker": "o",
                    "markersize": 2.4,
                    "markerfacecolor": COLORS[variant_label],
                    "markeredgecolor": COLORS[variant_label],
                    "alpha": 0.55,
                },
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(COLORS[variant_label])
                patch.set_alpha(0.9)

        ymax = max(panel_values) * 1.08 if panel_values else 1.0
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(_format_axis_value("runtime_ms"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax.tick_params(axis="y", labelsize=tick_font_size)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", which="major", color="#e5e7eb", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")
        ax.set_xlim(0.35, 4.65)
        ax.set_title(
            f"{distribution['benchmark_label']}\n(n={distribution['query_count']})",
            fontsize=title_font_size,
            pad=8,
        )
        if index == 0:
            ax.set_ylabel(
                "Time (ms)" if compressed_column_layout else "Query execution time (ms)",
                fontsize=label_font_size,
            )
        else:
            ax.set_ylabel("")

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS[label],
                edgecolor="#374151",
                label=_variant_display_label(label),
            )
            for label in VARIANT_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        fontsize=legend_font_size,
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(
        left=0.12 if compressed_column_layout else 0.08,
        right=0.997,
        bottom=0.11,
        top=0.79 if compressed_column_layout else 0.86,
        wspace=0.22 if compressed_column_layout else 0.18,
    )
    verifications = [_verify_layout(fig, ax) for ax in axes]
    verification = {
        "tick_label_overlap": any(item["tick_label_overlap"] for item in verifications),
        "text_outside_figure": any(item["text_outside_figure"] for item in verifications),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format=output_path.suffix.removeprefix(".") or "pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
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


def _chunk_query_names(query_names: tuple[str, ...], chunk_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        query_names[index:index + chunk_size]
        for index in range(0, len(query_names), chunk_size)
    )


def _benchmark_summaries(rows: list) -> dict[str, object]:
    job_queries = tuple(
        sorted({row.query_name for row in rows if row.workload_name == "join-order-benchmark"})
    )
    summaries: dict[str, object] = {}
    available_workloads = _available_workload_names(rows)
    if "join-pruning-workload" in available_workloads:
        summaries["join_pruning_benchmark"] = summarize_benchmark_runtime(
            rows,
            workload_name="join-pruning-workload",
            query_names=JPB_ALL_QUERIES,
            benchmark_label=WORKLOAD_LABELS["join-pruning-workload"],
        )
    if "tpch-skew" in available_workloads:
        summaries["tpch_skew_selected"] = summarize_benchmark_runtime(
            rows,
            workload_name="tpch-skew",
            query_names=TPCH_SELECTED_QUERIES,
            benchmark_label=WORKLOAD_LABELS["tpch-skew"],
        )
    if "string-joins" in available_workloads:
        summaries["string_join_benchmark"] = summarize_benchmark_runtime(
            rows,
            workload_name="string-joins",
            query_names=STRING_JOIN_QUERIES,
            benchmark_label=WORKLOAD_LABELS["string-joins"],
        )
    if "join-order-benchmark" in available_workloads:
        summaries["join_order_benchmark"] = summarize_benchmark_runtime(
            rows,
            workload_name="join-order-benchmark",
            query_names=job_queries,
            benchmark_label=WORKLOAD_LABELS["join-order-benchmark"],
        )
    return summaries


def _format_duration_ms(value: float) -> str:
    return f"{value / 1000.0:,.2f} s"


def _format_tuple_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,}"


def _format_relative(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _bold_if_best(text: str, value: float | int | None, best: float | int | None) -> str:
    if value is None or best is None or value != best:
        return text
    return f"**{text}**"


def _write_benchmark_summary_markdown(
    path: Path,
    *,
    experiment_root: Path,
    system_name: str,
    benchmark_summary: dict[str, object],
) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"- experiment root: `{experiment_root}`",
        f"- system: `{system_name}`",
        "",
        "| Benchmark | Variant | Queries | Total time | Relative time | Rows scanned | Relative rows | Join-probe tuples | Relative probes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in benchmark_summary["benchmarks"]:
        variants = benchmark["variants"]
        metric_keys = (
            "total_runtime_ms",
            "relative_runtime_geomean",
            "total_rows_scanned",
            "relative_rows_scanned_geomean",
            "total_join_probe_tuples",
            "relative_join_probe_geomean",
        )
        best_values = {
            key: min(
                value
                for variant in variants.values()
                if (value := variant[key]) is not None
            )
            for key in metric_keys
        }
        for variant_index, variant_label in enumerate(VARIANT_ORDER):
            values = variants[variant_label]
            benchmark_cell = (
                benchmark["benchmark_label"]
                if variant_index == 0
                else ""
            )
            variant_cell = _variant_display_label(variant_label)
            total_runtime = _bold_if_best(
                _format_duration_ms(values["total_runtime_ms"]),
                values["total_runtime_ms"],
                best_values["total_runtime_ms"],
            )
            relative_runtime = _bold_if_best(
                _format_relative(values["relative_runtime_geomean"]),
                values["relative_runtime_geomean"],
                best_values["relative_runtime_geomean"],
            )
            total_rows = _bold_if_best(
                _format_tuple_count(values["total_rows_scanned"]),
                values["total_rows_scanned"],
                best_values["total_rows_scanned"],
            )
            relative_rows = _bold_if_best(
                _format_relative(values["relative_rows_scanned_geomean"]),
                values["relative_rows_scanned_geomean"],
                best_values["relative_rows_scanned_geomean"],
            )
            total_probes = _bold_if_best(
                _format_tuple_count(values["total_join_probe_tuples"]),
                values["total_join_probe_tuples"],
                best_values["total_join_probe_tuples"],
            )
            relative_probes = _bold_if_best(
                _format_relative(values["relative_join_probe_geomean"]),
                values["relative_join_probe_geomean"],
                best_values["relative_join_probe_geomean"],
            )
            lines.append(
                f"| {benchmark_cell} | {variant_cell} | {benchmark['query_count']} | "
                f"{total_runtime} | {relative_runtime} | {total_rows} | "
                f"{relative_rows} | {total_probes} | {relative_probes} |"
            )
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- **Total time:** sum of per-query median execution times.",
            "- **Relative time:** geometric mean of each query runtime divided by the best variant runtime for that query.",
            "- **Rows scanned:** sum of profile-level `cumulative_rows_scanned` across all queries.",
            "- **Relative rows:** geometric mean of each query's cumulative rows scanned divided by the best variant for that query.",
            "- **Join-probe tuples:** sum of the output cardinality of distinct `TABLE_SCAN` operators that appear anywhere in the left subtree of at least one `*_JOIN` operator in a query plan. Each qualifying scan is counted once per query.",
            "- **Relative probes:** geometric mean of each query's join-probe tuples divided by the best variant for that query.",
            "- Lower relative values are better; `1.00x` is best.",
            "",
            "## Coverage",
            "",
        ]
    )
    for benchmark in benchmark_summary["benchmarks"]:
        relative_counts = benchmark["relative_query_counts"]
        excluded_counts = benchmark["relative_excluded_query_counts"]
        lines.append(
            f"- **{benchmark['benchmark_label']}:** {benchmark['query_count']} queries; "
            f"relative time uses {relative_counts['runtime_ms']} "
            f"(excluded {excluded_counts['runtime_ms']}); "
            f"relative rows use {relative_counts['rows_scanned']} "
            f"(excluded {excluded_counts['rows_scanned']}); "
            f"relative probes use {relative_counts['join_probe_tuples']} "
            f"(excluded {excluded_counts['join_probe_tuples']}). "
            f"{benchmark['join_free_query_count']} queries contain no join operator."
        )
    lines.extend(
        [
            "",
            "Relative geometric means exclude a query when any variant reports a non-positive value for that metric, because normalization against zero is undefined. Absolute totals still include those values.",
        ]
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_figures(experiment_root: Path, system_name: str | None) -> Path:
    experiment_root = experiment_root.resolve()
    resolved_system_name = load_system_name(experiment_root, system_name)
    rows = load_metric_rows(experiment_root, resolved_system_name)
    plan_metric_rows = load_plan_metric_rows(experiment_root, resolved_system_name)
    probe_scan_rows = load_probe_scan_rows(experiment_root, resolved_system_name)
    output_dir = experiment_root / "paper"
    figures_dir = output_dir / "figures"
    available_workloads = _available_workload_names(rows)
    summary: dict[str, object] = {
        "experiment_root": str(experiment_root),
        "system_name": resolved_system_name,
        "figures": {},
        "sections": {},
        "verification": {},
        "skipped_figures": [],
    }
    for figure in FIGURES:
        if not _figure_is_available(figure, available_workloads):
            summary["skipped_figures"].append(figure.slug)
            continue
        if figure.metric == "runtime_and_probe_rows":
            workload_name = figure.workloads[0]
            query_names = figure.queries_by_workload[workload_name]
            if figure.probe_table is None:
                raise ValueError(
                    f"runtime-and-probe figure {figure.slug} must define probe_table"
                )
            entries = build_runtime_probe_table(
                rows,
                probe_scan_rows,
                workload_name,
                query_names,
                figure.probe_table,
            )
            runtime_summary = summarize_section(
                [
                    {
                        **entry,
                        "series": [
                            {**item, "value": item["median_ms"]}
                            for item in entry["series"]
                        ],
                    }
                    for entry in entries
                ],
                "runtime_ms",
            )
            rows_summary = summarize_section(
                [
                    {
                        **entry,
                        "series": [
                            {**item, "value": item["probe_rows_scanned"]}
                            for item in entry["series"]
                        ],
                    }
                    for entry in entries
                ],
                "total_rows",
            )
            section_summary = {
                "runtime": runtime_summary,
                "probe_rows": rows_summary,
            }
        elif figure.metric == "runtime_and_total_rows":
            entries = []
            for workload_name in figure.workloads:
                query_names = select_queries(
                    rows,
                    workload_name,
                    figure.queries_by_workload.get(workload_name, ()),
                )
                entries.extend(
                    build_runtime_plan_work_table(
                        rows,
                        plan_metric_rows,
                        workload_name,
                        query_names,
                    )
                )
            runtime_summary = summarize_section(
                [
                    {
                        **entry,
                        "series": [
                            {**item, "value": item["median_ms"]}
                            for item in entry["series"]
                        ],
                    }
                    for entry in entries
                ],
                "runtime_ms",
            )
            rows_summary = summarize_section(
                [
                    {
                        **entry,
                        "series": [
                            {**item, "value": item["rows_scanned"]}
                            for item in entry["series"]
                        ],
                    }
                    for entry in entries
                ],
                "total_rows",
            )
            section_summary = {
                "runtime": runtime_summary,
                "rows_scanned": rows_summary,
            }
        elif figure.slug == "end_to_end_runtime":
            entries = _end_to_end_entries([row for row in rows if row.workload_name in figure.workloads])
            section_summary = summarize_end_to_end([row for row in rows if row.workload_name in figure.workloads])
        else:
            entries = []
            for workload_name in figure.workloads:
                query_names = select_queries(rows, workload_name, figure.queries_by_workload.get(workload_name, ()))
                entries.extend(build_absolute_table(rows, workload_name, query_names, figure.metric))
            section_summary = summarize_section(entries, figure.metric)
        figure_path = figures_dir / f"{figure.slug}.pdf"
        if figure.metric == "runtime_and_probe_rows":
            verification = _draw_runtime_probe_bars(
                figure_path,
                entries=entries,
                width_key=figure.width,
                placement_width_key=figure.placement_width,
                show_query_labels=figure.show_query_labels,
                annotate_runtime_values=figure.annotate_runtime_values,
                runtime_log_axis=figure.runtime_log_axis,
                rows_log_axis=figure.rows_log_axis,
            )
        elif figure.metric == "runtime_and_total_rows":
            verification = _draw_runtime_total_rows_bars(
                figure_path,
                entries=entries,
                width_key=figure.width,
                placement_width_key=figure.placement_width,
                show_query_labels=figure.show_query_labels,
                show_runtime_labels=figure.show_runtime_labels,
                workload_labels=figure.query_workload_labels,
                runtime_log_axis=figure.runtime_log_axis,
                rows_log_axis=figure.rows_log_axis,
            )
        else:
            verification = _draw_vertical_grouped_bars(
                figure_path,
                entries=entries,
                width_key=figure.width,
                placement_width_key=figure.placement_width,
                metric=figure.metric,
                y_label=figure.y_label,
                workload_labels=figure.query_workload_labels,
            )
        summary["figures"][figure.slug] = str(figure_path)
        summary["verification"][figure.slug] = verification
        summary["sections"].setdefault(figure.summary_key, {}).update({figure.slug: section_summary})

    def summarize_plan_work_entries(entries: list[dict[str, object]]) -> dict[str, object]:
        runtime_summary = summarize_section(
            [
                {
                    **entry,
                    "series": [
                        {**item, "value": item["median_ms"]}
                        for item in entry["series"]
                    ],
                }
                for entry in entries
            ],
            "runtime_ms",
        )
        rows_summary = summarize_section(
            [
                {
                    **entry,
                    "series": [
                        {**item, "value": item["rows_scanned"]}
                        for item in entry["series"]
                    ],
                }
                for entry in entries
            ],
            "total_rows",
        )
        join_probe_summary = summarize_section(
            [
                {
                    **entry,
                    "series": [
                        {**item, "value": item["join_probe_tuples"]}
                        for item in entry["series"]
                    ],
                }
                for entry in entries
            ],
            "total_rows",
        )
        return {
            "runtime": runtime_summary,
            "rows_scanned": rows_summary,
            "join_probe_tuples": join_probe_summary,
        }

    if "tpch-skew" in available_workloads:
        tpch_queries = select_queries(rows, "tpch-skew", ())
        tpch_entries = build_runtime_plan_work_table(
            rows,
            plan_metric_rows,
            "tpch-skew",
            tpch_queries,
        )
        tpch_path = figures_dir / "tpch_skew_plan_work.pdf"
        tpch_verification = _draw_runtime_plan_work_bars(
            tpch_path,
            entries=tpch_entries,
            width_key="two-column",
            show_query_labels=True,
        )
        summary["figures"]["tpch_skew_plan_work"] = str(tpch_path)
        summary["verification"]["tpch_skew_plan_work"] = tpch_verification
        summary["sections"].setdefault("tpch_skew_plan_work", {}).update(
            {"tpch_skew_plan_work": summarize_plan_work_entries(tpch_entries)}
        )

    if "join-order-benchmark" in available_workloads:
        job_queries = select_queries(rows, "join-order-benchmark", ())
        job_chunks = _chunk_query_names(job_queries, MULTI_JOIN_QUERY_PANEL_SIZE)
        job_sections: dict[str, object] = {}
        for index, chunk in enumerate(job_chunks, start=1):
            slug = f"job_plan_work_part{index}"
            chunk_entries = build_runtime_plan_work_table(
                rows,
                plan_metric_rows,
                "join-order-benchmark",
                chunk,
            )
            chunk_path = figures_dir / f"{slug}.pdf"
            verification = _draw_runtime_plan_work_bars(
                chunk_path,
                entries=chunk_entries,
                width_key="two-column",
                show_query_labels=True,
            )
            summary["figures"][slug] = str(chunk_path)
            summary["verification"][slug] = verification
            job_sections[slug] = {
                "query_count": len(chunk),
                "query_names": list(chunk),
                **summarize_plan_work_entries(chunk_entries),
            }
        summary["sections"]["job_plan_work"] = {
            "panel_count": len(job_chunks),
            "panels": job_sections,
        }
    end_to_end_rows = [row for row in rows if row.workload_name in FIGURES[-1].workloads]
    if end_to_end_rows:
        summary["sections"]["end_to_end"] = summarize_end_to_end(end_to_end_rows)
    aggregate_probe_scan_rows = [
        row for row in probe_scan_rows if row.workload_name != "ssb-ordered"
    ]
    if aggregate_probe_scan_rows:
        probe_cardinality_values = build_relative_probe_cardinality_values(
            aggregate_probe_scan_rows
        )
        probe_cardinality_outliers = build_relative_probe_cardinality_outliers(
            aggregate_probe_scan_rows,
            threshold=PROBE_CARDINALITY_OUTLIER_THRESHOLD,
        )
        probe_cardinality_path = figures_dir / "relative_probe_cardinality_boxplot.pdf"
        probe_cardinality_verification = _draw_relative_probe_cardinality_boxplot(
            probe_cardinality_path,
            relative_values=probe_cardinality_values,
        )
        outlier_log_path = output_dir / "relative_probe_cardinality_outliers.json"
        write_summary(outlier_log_path, {"threshold": PROBE_CARDINALITY_OUTLIER_THRESHOLD, "outliers": probe_cardinality_outliers})
        summary["figures"]["relative_probe_cardinality_boxplot"] = str(probe_cardinality_path)
        summary["verification"]["relative_probe_cardinality_boxplot"] = probe_cardinality_verification
        summary["sections"]["relative_probe_cardinality"] = {
            **summarize_relative_probe_cardinality(aggregate_probe_scan_rows),
            "outlier_threshold_ratio": PROBE_CARDINALITY_OUTLIER_THRESHOLD,
            "outlier_count": len(probe_cardinality_outliers),
            "outlier_log_path": str(outlier_log_path),
            "largest_outliers": probe_cardinality_outliers[:10],
        }
    present_benchmark_groups = _present_benchmark_groups(rows)
    if present_benchmark_groups:
        benchmark_runtime_distributions = build_benchmark_runtime_distributions(
            rows,
            present_benchmark_groups,
        )
        benchmark_runtime_boxplot_path = figures_dir / "benchmark_runtime_boxplot.pdf"
        benchmark_runtime_boxplot_verification = _draw_benchmark_runtime_boxplot(
            benchmark_runtime_boxplot_path,
            distributions=benchmark_runtime_distributions,
            placement_width_key="one-column",
        )
        summary["figures"]["benchmark_runtime_boxplot"] = str(
            benchmark_runtime_boxplot_path
        )
        summary["verification"]["benchmark_runtime_boxplot"] = (
            benchmark_runtime_boxplot_verification
        )
        summary["sections"]["benchmark_runtime_boxplot"] = (
            summarize_benchmark_runtime_distributions(
                benchmark_runtime_distributions
            )
        )
        benchmark_metric_summary = summarize_benchmark_metrics(
            rows,
            plan_metric_rows,
            present_benchmark_groups,
        )
        benchmark_summary_path = output_dir / "benchmark_summary.md"
        _write_benchmark_summary_markdown(
            benchmark_summary_path,
            experiment_root=experiment_root,
            system_name=resolved_system_name,
            benchmark_summary=benchmark_metric_summary,
        )
        summary["benchmark_summary_path"] = str(benchmark_summary_path)
        summary["sections"]["benchmark_metrics"] = benchmark_metric_summary
    summary["benchmark_summaries"] = _benchmark_summaries(rows)
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
    lines.append("## Benchmark Summaries")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["benchmark_summaries"], indent=2))
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
