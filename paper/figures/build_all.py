from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking import load_experiment_config
from benchmarking.microbenchmarks import ARTIFACT_SUBDIR, load_microbenchmark_manifest
from benchmarking.jpb_microbenchmarks import ARTIFACT_SUBDIR as JPB_ARTIFACT_SUBDIR, load_jpb_manifest
from paper.figures.filter_benchmark import build_filter_benchmark_artifacts
from paper.figures.jpb_microbenchmark import build_jpb_artifacts
from paper.figure_specs import (
    FIGURES,
    JPB_ALL_QUERIES,
    JPB_NATIVE_ALL_QUERIES,
    PARAMETER_SWEEP_PANELS,
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
    load_system_names,
    select_queries,
    summarize_relative_probe_cardinality,
    summarize_benchmark_runtime,
    summarize_benchmark_runtime_distributions,
    summarize_benchmark_metrics,
    summarize_end_to_end,
    summarize_section,
    write_summary,
    write_filter_phase_report,
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
    "prf uncompressed": "#06b6d4",
    "prf with fallback": "#0f766e",
    "rpt none": "#6b7280",
    "rpt cascade": "#9333ea",
    "rpt prf": "#dc2626",
}

PROBE_CARDINALITY_VARIANTS = (
    "sip",
    "sip + min/max",
    "prf uncompressed",
    "prf with fallback",
    "rpt cascade",
    "rpt prf",
)
PROBE_CARDINALITY_OUTLIER_THRESHOLD = 10.0
PROBE_FILTERED_HATCH = "...."
PROBE_SKIPPED_HATCH = "////"
ROWS_SCANNED_HATCH = "////"
FILTER_BUILD_HATCH = "\\\\\\\\"
PRF_ANALYSIS_HATCH = "xxxx"
PRF_ANALYSIS_COLOR = "#7c3aed"
PRF_COMPRESSION_HATCH = "...."
PRF_COMPRESSION_COLOR = "#0f766e"
BENCHMARK_RUNTIME_BOXPLOT_GROUPS = (
    (
        "join-pruning-workload",
        "JPB",
        JPB_NATIVE_ALL_QUERIES,
    ),
    (
        "join-pruning-workload-ducklake",
        "JPB DuckLake",
        JPB_ALL_QUERIES,
    ),
    (
        "join-pruning-workload-ducklake-http-10ms",
        "JPB DuckLake +10 ms",
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
    "prf uncompressed": "PRF (uncompressed)",
    "prf with fallback": "Prefix Range Filter",
    "rpt none": "RPT+ disabled",
    "rpt cascade": "RPT+ Cascade",
    "rpt prf": "RPT+ PRF",
}
VARIANT_PANEL_LABELS = {
    "no prefilter": "No\npre-\nfilter",
    "sip": "Bloom",
    "sip + min/max": "Bloom +\nmin/max",
    "prf uncompressed": "PRF\n(uncompressed)",
    "prf with fallback": "Prefix\nRange\nFilter",
    "rpt none": "RPT+\ndisabled",
    "rpt cascade": "RPT+\nCascade",
    "rpt prf": "RPT+\nPRF",
}
MULTI_JOIN_QUERY_PANEL_SIZE = 24

SWEEP_LINE_STYLES = {
    "shoulder": "-",
    "scattered": ":",
    "seasonal": "-",
}

SWEEP_MARKERS = {
    "no prefilter": "o",
    "sip": "s",
    "sip + min/max": "^",
    "prf uncompressed": "h",
    "prf with fallback": "D",
    "rpt none": "v",
    "rpt cascade": "P",
    "rpt prf": "X",
}


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


def _present_entry_variants(entries: list[dict[str, object]]) -> tuple[str, ...]:
    present = {
        item["variant_label"]
        for entry in entries
        for item in entry["series"]
    }
    return tuple(label for label in VARIANT_ORDER if label in present)


def _variant_bar_geometry(
    preferred_width: float,
    variant_count: int | None = None,
) -> tuple[float, list[float]]:
    count = variant_count if variant_count is not None else len(VARIANT_ORDER)
    if count <= 0:
        raise ValueError("variant_count must be positive")
    bar_width = min(preferred_width, 0.72 / count)
    center = (count - 1) / 2
    return bar_width, [
        (index - center) * bar_width for index in range(count)
    ]


FILTER_PHASE_STYLES = (
    ("join_finalize_build_ms", "Join/filter build", FILTER_BUILD_HATCH, "#111827"),
    ("prf_analysis_ms", "PRF analysis", PRF_ANALYSIS_HATCH, PRF_ANALYSIS_COLOR),
    ("prf_compression_ms", "PRF compression", PRF_COMPRESSION_HATCH, PRF_COMPRESSION_COLOR),
)


def _filter_phase_legend_handles() -> list[Patch]:
    return [Patch(facecolor="none", edgecolor=color, hatch=hatch, label=label)
            for _, label, hatch, color in FILTER_PHASE_STYLES]


def _annotate_filter_phases(fig, legend, font_size: float) -> float:
    fig.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    separator = "\n" if fig.get_figwidth() < 4 else " "
    note = fig.text(
        0.5, legend_bottom - 0.025,
        "Join/filter phases: one profile; hatches at bar tops." + separator + "Bar heights: timed medians.",
        ha="center", va="top", fontsize=font_size, color="#4b5563",
    )
    fig.canvas.draw()
    return note.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.04


def _draw_filter_phase_overlays(
    ax,
    entries: list[dict[str, object]],
    present_variants: tuple[str, ...],
    offsets: list[float],
    bar_width: float,
    runtime_floor: float,
) -> dict[str, int]:
    """Annotate median bars with comparable elapsed phases from one profile.

    Includes Bloom-only and uncompressed PRF variants. Missing telemetry is
    unknown; explicit zero analysis/compression is valid. Overlapping join
    spans can exceed query latency, in which case no decomposition is drawn.
    """
    marked = overflow = 0
    for variant_index, variant_label in enumerate(present_variants):
        for query_index, entry in enumerate(entries):
            item = next((part for part in entry["series"] if part["variant_label"] == variant_label), None)
            if item is None or any(item.get(name) is None for name, *_ in FILTER_PHASE_STYLES):
                continue
            runtime = float(item["median_ms"])
            durations = [float(item[name]) for name, *_ in FILTER_PHASE_STYLES]
            if sum(durations) > runtime:
                overflow += 1
                continue
            position = query_index + offsets[variant_index]
            bottom = runtime - sum(durations)
            for index, ((_, _, hatch, color), duration) in enumerate(zip(FILTER_PHASE_STYLES, durations)):
                top = bottom + duration
                clipped_bottom, clipped_top = max(bottom, runtime_floor), max(top, runtime_floor)
                if clipped_top > clipped_bottom:
                    ax.bar(position, clipped_top - clipped_bottom, bottom=clipped_bottom,
                           width=bar_width, facecolor="none", edgecolor=color,
                           hatch=hatch, linewidth=0.5, zorder=4 + index)
                if duration > 0:
                    ax.plot(position + (index - 1) * bar_width * 0.25, clipped_top,
                            marker="_", markersize=4, color=color, zorder=8)
                bottom = top
            marked += 1
    return {"filter_phase_bars": marked, "filter_phase_overflows": overflow}


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


def _artist_overlaps_axes(fig, artist, axes) -> bool:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artist_bbox = artist.get_window_extent(renderer)
    return any(artist_bbox.overlaps(ax.get_window_extent(renderer)) for ax in axes)


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
    present_variants = _present_entry_variants(entries)
    bar_width, offsets = _variant_bar_geometry(
        0.18 if count <= 9 else 0.16,
        len(present_variants),
    )
    ymax = 0.0
    for entry in entries:
        for item in entry["series"]:
            if item["value"] is not None:
                ymax = max(ymax, float(item["value"]))
    ymax = ymax * 1.18 if ymax > 0 else 1.0
    for series_index, variant_label in enumerate(present_variants):
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
    split_scan_pruning: bool = False,
    show_filter_phases: bool = False,
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
    if show_filter_phases and width_key == "one-column":
        height += 1.25
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
    present_variants = _present_entry_variants(entries)
    bar_width, offsets = _variant_bar_geometry(0.18, len(present_variants))
    runtime_max = 0.0
    rows_max = 0
    positive_runtime_values: list[float] = []
    positive_row_values: list[float] = []

    series_data: list[
        tuple[str, list[float], list[int], list[int], list[int]]
    ] = []
    for variant_label in present_variants:
        runtime_values: list[float] = []
        probe_values: list[int] = []
        filtered_values: list[int] = []
        skipped_values: list[int] = []
        for entry in entries:
            match = next(
                (item for item in entry["series"] if item["variant_label"] == variant_label),
                None,
            )
            runtime = float(match["median_ms"]) if match else 0.0
            probe_rows = int(match["join_probe_rows"]) if match else 0
            filtered_rows = (
                int(match["probe_rows_filtered_after_scan"]) if match else 0
            )
            skipped_rows = (
                int(match["probe_rows_skipped_at_scan"])
                if match and split_scan_pruning
                else 0
            )
            runtime_values.append(runtime)
            probe_values.append(probe_rows)
            filtered_values.append(filtered_rows)
            skipped_values.append(skipped_rows)
            runtime_max = max(runtime_max, runtime)
            rows_max = max(rows_max, probe_rows + filtered_rows + skipped_rows)
            if runtime > 0:
                positive_runtime_values.append(runtime)
            if probe_rows > 0:
                positive_row_values.append(float(probe_rows))
            if probe_rows + filtered_rows > 0:
                positive_row_values.append(float(probe_rows + filtered_rows))
            if probe_rows + filtered_rows + skipped_rows > 0:
                positive_row_values.append(
                    float(probe_rows + filtered_rows + skipped_rows)
                )
        series_data.append(
            (
                variant_label,
                runtime_values,
                probe_values,
                filtered_values,
                skipped_values,
            )
        )

    runtime_bounds = _log_axis_bounds(positive_runtime_values)
    runtime_floor = runtime_bounds[0] if runtime_bounds else 1.0
    row_bounds = _log_axis_bounds(positive_row_values)
    row_floor = row_bounds[0] if row_bounds else 1.0

    for series_index, (
        variant_label,
        runtime_values,
        probe_values,
        filtered_values,
        skipped_values,
    ) in enumerate(series_data):
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
            rendered_probe_values = [
                _render_log_value(value, row_floor)
                for value in probe_values
            ]
            rendered_scanned_values = [
                _render_log_value(probe + filtered, row_floor)
                for probe, filtered in zip(probe_values, filtered_values)
            ]
            rendered_total_values = [
                _render_log_value(probe + filtered + skipped, row_floor)
                for probe, filtered, skipped in zip(
                    probe_values, filtered_values, skipped_values
                )
            ]
            rows_ax.bar(
                positions,
                [top - row_floor for top in rendered_probe_values],
                width=bar_width,
                bottom=row_floor,
                color=COLORS[variant_label],
                edgecolor="#ffffff",
                linewidth=0.45,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                [
                    top - bottom
                    for top, bottom in zip(
                        rendered_scanned_values, rendered_probe_values
                    )
                ],
                width=bar_width,
                bottom=rendered_probe_values,
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_FILTERED_HATCH,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                [
                    top - bottom
                    for top, bottom in zip(
                        rendered_total_values, rendered_scanned_values
                    )
                ],
                width=bar_width,
                bottom=rendered_scanned_values,
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_SKIPPED_HATCH,
                zorder=3,
            )
        else:
            rows_ax.bar(
                positions,
                probe_values,
                width=bar_width,
                color=COLORS[variant_label],
                edgecolor="#ffffff",
                linewidth=0.45,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                filtered_values,
                width=bar_width,
                bottom=probe_values,
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_FILTERED_HATCH,
                zorder=3,
            )
            rows_ax.bar(
                positions,
                skipped_values,
                width=bar_width,
                bottom=[
                    probe + filtered
                    for probe, filtered in zip(probe_values, filtered_values)
                ],
                color="white",
                edgecolor=COLORS[variant_label],
                linewidth=0.7,
                hatch=PROBE_SKIPPED_HATCH,
                zorder=3,
            )

    phase_verification = (
        _draw_filter_phase_overlays(
            runtime_ax, entries, present_variants, offsets, bar_width,
            runtime_floor if runtime_log_axis else 0.0,
        )
        if show_filter_phases
        else {"filter_phase_bars": 0, "filter_phase_overflows": 0}
    )
    runtime_ax.set_ylabel(
        "Time (ms)" if compressed_column_layout or width_key == "one-column" else "Execution time (ms)",
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
        "Fact tuples",
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
        for label in present_variants
    ]
    semantic_handles = [
        Patch(
            facecolor="#6b7280",
            edgecolor="none",
            label="Join-probed tuples",
        ),
        Patch(
            facecolor="white",
            edgecolor="#4b5563",
            hatch=PROBE_FILTERED_HATCH,
            label=(
                "Tuple-filtered after scan"
                if split_scan_pruning
                else "Pre-filtered tuples"
            ),
        ),
    ]
    if split_scan_pruning:
        semantic_handles.append(
            Patch(
                facecolor="white",
                edgecolor="#4b5563",
                hatch=PROBE_SKIPPED_HATCH,
                label="Skipped at scan",
            )
        )
    if show_filter_phases and phase_verification["filter_phase_bars"]:
        semantic_handles.extend(_filter_phase_legend_handles())
    variant_legend = fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2 if width_key == "one-column" else 4,
        frameon=False,
        fontsize=legend_font_size,
        handlelength=1.5,
        columnspacing=1.0,
    )
    semantic_legend = fig.legend(
        handles=semantic_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78 if width_key == "one-column" else 0.87),
        ncol=1 if width_key == "one-column" else 3,
        frameon=False,
        fontsize=legend_font_size,
        handlelength=1.5,
        columnspacing=1.0,
    )
    plot_top = 0.50 if width_key == "one-column" else (0.72 if show_filter_phases else 0.75)
    if show_filter_phases and phase_verification["filter_phase_bars"]:
        plot_top = min(plot_top, _annotate_filter_phases(fig, semantic_legend, annotation_font_size))

    fig.subplots_adjust(
        left=0.26 if width_key == "one-column" else (0.17 if compressed_column_layout else 0.10),
        right=0.992 if width_key == "one-column" else 0.997,
        bottom=0.08 if not show_query_labels else (0.18 if has_wrapped_xticklabels else (0.12 if compressed_column_layout else 0.11)),
        top=plot_top,
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
            or _artist_overlaps_axes(
                fig,
                variant_legend,
                (runtime_ax, rows_ax),
            )
            or _artist_overlaps_axes(
                fig,
                semantic_legend,
                (runtime_ax, rows_ax),
            )
        ),
        **phase_verification,
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
    rows_metric_key: str = "rows_scanned",
    rows_y_label: str = "Rows scanned",
    rows_percent: bool = False,
    show_filter_phases: bool = False,
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
    present_variants = _present_entry_variants(entries)
    bar_width, offsets = _variant_bar_geometry(
        0.15 if show_runtime_labels else 0.18,
        len(present_variants),
    )
    runtime_max = 0.0
    rows_max = 0.0
    positive_runtime_values: list[float] = []
    positive_row_values: list[float] = []

    series_data: list[tuple[str, list[float], list[float]]] = []
    for variant_label in present_variants:
        runtime_values: list[float] = []
        row_values: list[float] = []
        for entry in entries:
            match = next(
                (item for item in entry["series"] if item["variant_label"] == variant_label),
                None,
            )
            runtime = float(match["median_ms"]) if match else 0.0
            rows_scanned = (
                float(match[rows_metric_key])
                if match and match.get(rows_metric_key) is not None
                else 0.0
            )
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

    phase_verification = (
        _draw_filter_phase_overlays(
            runtime_ax, entries, present_variants, offsets, bar_width,
            runtime_floor if runtime_log_axis else 0.0,
        )
        if show_filter_phases
        else {"filter_phase_bars": 0, "filter_phase_overflows": 0}
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

    rows_ax.set_ylabel(rows_y_label, fontsize=label_font_size)
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
        rows_ax.yaxis.set_major_formatter(
            FuncFormatter(
                (lambda value, _: f"{value * 100:.0f}%")
                if rows_percent
                else (lambda value, _: _format_large_number(value))
            )
        )
        rows_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        rows_ax.set_ylim(
            0,
            1.03 if rows_percent else (rows_max * 1.03 if rows_max > 0 else 1.0),
        )
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
        for label in present_variants
    ]
    if show_filter_phases and phase_verification["filter_phase_bars"]:
        total_rows_legend_handles.extend(_filter_phase_legend_handles())
    plot_top = 0.74 if show_runtime_labels else (0.70 if compressed_column_layout else (0.74 if width_key == "one-column" else 0.83))
    if show_filter_phases and phase_verification["filter_phase_bars"]:
        phase_legend = fig.legend(
            handles=total_rows_legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
            ncol=2 if width_key == "one-column" else 3, frameon=False,
            fontsize=legend_font_size, handlelength=1.5, columnspacing=1.0,
        )
        plot_top = min(plot_top, _annotate_filter_phases(fig, phase_legend, annotation_font_size))
    elif show_runtime_labels:
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
        top=plot_top,
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
        **phase_verification,
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
    present_variants = _present_entry_variants(entries)
    variant_step, variant_offsets = _variant_bar_geometry(
        0.18 if count <= 24 else 0.16,
        len(present_variants),
    )
    measure_width = variant_step / 3
    runtime_max = 0.0
    positive_work_values: list[float] = []

    for series_index, variant_label in enumerate(present_variants):
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
        for label in present_variants
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

    present_variants = [
        label
        for label in PROBE_CARDINALITY_VARIANTS
        if label in relative_values and relative_values[label]
    ]
    if not present_variants:
        ax.text(
            0.5,
            0.5,
            "No comparable filtered scans",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#4b5563",
        )
        ax.set_axis_off()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        return {"tick_label_overlap": False, "text_outside_figure": False}
    data = [relative_values[label] for label in present_variants]
    positions = list(range(1, len(present_variants) + 1))
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
    for patch, label in zip(bp["boxes"], present_variants):
        patch.set_facecolor(COLORS[label])
        patch.set_edgecolor("#374151")
        patch.set_alpha(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels([_variant_display_label(label) for label in present_variants], fontsize=7)
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

    positions = tuple(range(1, len(VARIANT_ORDER) + 1))
    for index, (ax, distribution) in enumerate(zip(axes, distributions)):
        ax.set_facecolor("white")
        panel_values = []
        for variant_index, variant_label in enumerate(VARIANT_ORDER):
            values = distribution["values_by_variant"][variant_label]
            if not values:
                continue
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
        ax.set_xlim(0.35, len(VARIANT_ORDER) + 0.65)
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

    present_variants = [
        label
        for label in VARIANT_ORDER
        if any(distribution["values_by_variant"][label] for distribution in distributions)
    ]
    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS[label],
                edgecolor="#374151",
                label=_variant_display_label(label),
            )
            for label in present_variants
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(4, len(present_variants)),
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
            query_names=JPB_NATIVE_ALL_QUERIES,
            benchmark_label=WORKLOAD_LABELS["join-pruning-workload"],
        )
    if "join-pruning-workload-ducklake" in available_workloads:
        summaries["join_pruning_benchmark_ducklake"] = summarize_benchmark_runtime(
            rows,
            workload_name="join-pruning-workload-ducklake",
            query_names=JPB_ALL_QUERIES,
            benchmark_label=WORKLOAD_LABELS["join-pruning-workload-ducklake"],
        )
    if "join-pruning-workload-ducklake-http-10ms" in available_workloads:
        summaries["join_pruning_benchmark_ducklake_http_10ms"] = (
            summarize_benchmark_runtime(
                rows,
                workload_name="join-pruning-workload-ducklake-http-10ms",
                query_names=JPB_ALL_QUERIES,
                benchmark_label=WORKLOAD_LABELS[
                    "join-pruning-workload-ducklake-http-10ms"
                ],
            )
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


def _format_filter_phase_ms(value: float | None, covered: int, queries: int) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f} ms ({covered}/{queries})"


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
        "| Benchmark | Variant | Queries | Total time | Join/filter build | PRF analysis | PRF compression | PRF/Bloom builds | Relative time | Rows scanned | Relative rows | Join-probe tuples | Relative probes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        present_variants = [
            label for label in VARIANT_ORDER if label in variants
        ]
        for variant_index, variant_label in enumerate(present_variants):
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
            phases = [
                _format_filter_phase_ms(values[name], values[name.removesuffix("_ms") + "_profile_count"],
                                        benchmark["query_count"])
                for name, *_ in FILTER_PHASE_STYLES
            ]
            builds = ("—" if values["prf_build_count"] is None or values["bloom_build_count"] is None
                      else f"{values['prf_build_count']}/{values['bloom_build_count']}")
            lines.append(
                f"| {benchmark_cell} | {variant_cell} | {benchmark['query_count']} | "
                f"{total_runtime} | {phases[0]} | {phases[1]} | {phases[2]} | {builds} | {relative_runtime} | {total_rows} | "
                f"{relative_rows} | {total_probes} | {relative_probes} |"
            )
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- **Total time:** sum of per-query median execution times.",
            "- **Join/filter build:** elapsed hash-join finalization, including hash-table work and PRF/Bloom construction, excluding the separately measured PRF postprocessing. This is not an isolated filter-build cost.",
            "- **PRF analysis / compression:** separate elapsed phases. Uncompressed PRF can require analysis to choose a fallback; its compression time is zero. Compression includes its own bitmap analysis. Bloom-only builds record zero for both PRF phases.",
            "- Phase values sum operator spans from one JSON profile per query. Parentheses give queries with recorded phases / all queries. A dash means unavailable, including legacy PRF telemetry and RPT+ profiles; explicit zero means no work in that phase.",
            "- **PRF/Bloom builds:** actual build counts. A PRF followed by Bloom fallback increments both counts; its build time is included only once. Baseline hash joins also record finalization even when both counts are zero.",
            "- Profile phases come from a separate execution from the timed median. Multi-join phase spans can overlap, so their sums are not an additive decomposition of query latency. External joins report initial finalization only, excluding later source-side rounds. Worker times are not used.",
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


def _sweep_tick_formatter(sweep_kind: str):
    if sweep_kind == "selectivity":
        return FuncFormatter(lambda value, _: f"{value * 100:g}%")
    if sweep_kind == "join_count":
        return FuncFormatter(lambda value, _: f"{value:.0f}")
    return FuncFormatter(lambda value, _: _format_large_number(value))


def _parameter_sweep_series(
    rows: list,
) -> tuple[dict[str, dict[tuple[str, str], list[tuple[float, float]]]], dict[str, object]]:
    result: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {}
    summary: dict[str, object] = {}
    for panel in PARAMETER_SWEEP_PANELS:
        grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
        point_summary: list[dict[str, object]] = []
        for row in rows:
            if row.workload_name != panel.workload:
                continue
            parameters = row.query_parameters
            if parameters is None:
                raise ValueError(
                    f"{panel.workload} results do not contain query_parameters"
                )
            if parameters.get("sweep_kind") != panel.sweep_kind:
                raise ValueError(
                    f"{panel.workload} expected sweep kind {panel.sweep_kind!r}, "
                    f"got {parameters.get('sweep_kind')!r}"
                )
            sweep_value = parameters.get("sweep_value")
            query_type = parameters.get("query_type")
            if not isinstance(sweep_value, (int, float)) or isinstance(sweep_value, bool):
                raise ValueError(f"{row.query_name} has no numeric sweep_value")
            if not isinstance(query_type, str):
                raise ValueError(f"{row.query_name} has no query_type")
            point = (float(sweep_value), float(row.median_ms))
            grouped.setdefault((row.variant_label, query_type), []).append(point)
            point_summary.append(
                {
                    "query_name": row.query_name,
                    "variant_label": row.variant_label,
                    "query_type": query_type,
                    "sweep_value": sweep_value,
                    "median_ms": row.median_ms,
                    "query_parameters": parameters,
                    **row.filter_phases.to_dict(),
                }
            )
        for points in grouped.values():
            points.sort(key=lambda point: point[0])
            if len({point[0] for point in points}) != len(points):
                raise ValueError(f"duplicate sweep points found for {panel.workload}")
        expected_query_types = (
            {"seasonal"}
            if panel.sweep_kind == "cluster_count"
            else {"shoulder", "scattered"}
        )
        present_query_types = {query_type for _, query_type in grouped}
        if present_query_types != expected_query_types:
            raise ValueError(
                f"{panel.workload} expected query types "
                f"{sorted(expected_query_types)}, got {sorted(present_query_types)}"
            )
        expected_values = {
            point[0]
            for points in grouped.values()
            for point in points
        }
        present_variants = {variant for variant, _ in grouped}
        for variant in present_variants:
            for query_type in expected_query_types:
                actual_values = {
                    point[0] for point in grouped.get((variant, query_type), [])
                }
                if actual_values != expected_values:
                    raise ValueError(
                        f"incomplete {panel.workload} series for "
                        f"{variant}/{query_type}: expected {sorted(expected_values)}, "
                        f"got {sorted(actual_values)}"
                    )
        result[panel.workload] = grouped
        summary[panel.workload] = {
            "sweep_kind": panel.sweep_kind,
            "point_count": len(point_summary),
            "points": sorted(
                point_summary,
                key=lambda point: (
                    str(point["variant_label"]),
                    str(point["query_type"]),
                    float(point["sweep_value"]),
                ),
            ),
        }
    return result, summary


def _draw_parameter_sweep_overview(
    output_path: Path,
    *,
    rows: list,
) -> tuple[dict[str, bool], dict[str, object]]:
    panel_series, summary = _parameter_sweep_series(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    present_variants = tuple(
        variant
        for variant in VARIANT_ORDER
        if any(
            any(key[0] == variant for key in grouped)
            for grouped in panel_series.values()
        )
    )
    for ax, panel in zip(axes.flat, PARAMETER_SWEEP_PANELS):
        ax.set_facecolor("white")
        grouped = panel_series[panel.workload]
        for variant in present_variants:
            query_types = ("seasonal",) if panel.sweep_kind == "cluster_count" else (
                "shoulder",
                "scattered",
            )
            for query_type in query_types:
                points = grouped.get((variant, query_type), [])
                if not points:
                    continue
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=COLORS[variant],
                    marker=SWEEP_MARKERS[variant],
                    markersize=3.0,
                    markeredgewidth=0.4,
                    linewidth=1.15,
                    linestyle=SWEEP_LINE_STYLES[query_type],
                    zorder=3,
                )
        if panel.x_scale == "log":
            ax.set_xscale("log", base=panel.log_base)
        ax.set_yscale("log")
        if panel.x_ticks:
            ax.xaxis.set_major_locator(FixedLocator(panel.x_ticks))
        ax.xaxis.set_major_formatter(_sweep_tick_formatter(panel.sweep_kind))
        ax.set_title(panel.title, fontsize=TITLE_FONT_SIZE, pad=4)
        ax.set_xlabel(panel.x_label, fontsize=LABEL_FONT_SIZE, labelpad=2)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, pad=1.5)
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")
    fig.supylabel("Median execution time (ms)", fontsize=LABEL_FONT_SIZE, x=0.016)
    method_handles = [
        Line2D(
            [],
            [],
            color=COLORS[variant],
            marker=SWEEP_MARKERS[variant],
            linestyle="None",
            markersize=4,
            label=_variant_display_label(variant),
        )
        for variant in present_variants
    ]
    query_handles = [
        Line2D([], [], color="#111827", linestyle="-", linewidth=1.2, label="Shoulder"),
        Line2D([], [], color="#111827", linestyle=":", linewidth=1.2, label="Scattered"),
    ]
    legend = fig.legend(
        handles=method_handles + query_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.7,
        columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.975,
        top=0.955,
        bottom=0.205,
        wspace=0.34,
        hspace=0.42,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.get_window_extent(renderer)
    verifications = []
    for ax in axes.flat:
        axes_bbox = ax.get_window_extent(renderer)
        visible_x_tick_boxes = []
        visible_tick_boxes = []
        for tick in ax.get_xticklabels():
            if not tick.get_text():
                continue
            bbox = tick.get_window_extent(renderer)
            if axes_bbox.x0 - 1 <= (bbox.x0 + bbox.x1) / 2 <= axes_bbox.x1 + 1:
                visible_x_tick_boxes.append(bbox)
                visible_tick_boxes.append(bbox)
        for tick in ax.get_yticklabels():
            if not tick.get_text():
                continue
            bbox = tick.get_window_extent(renderer)
            if axes_bbox.y0 - 1 <= (bbox.y0 + bbox.y1) / 2 <= axes_bbox.y1 + 1:
                visible_tick_boxes.append(bbox)
        visible_x_tick_boxes.sort(key=lambda bbox: bbox.x0)
        verifications.append(
            {
                "tick_label_overlap": any(
                    left.x1 > right.x0
                    for left, right in zip(
                        visible_x_tick_boxes,
                        visible_x_tick_boxes[1:],
                    )
                ),
                "text_outside_figure": any(
                    bbox.x0 < figure_bbox.x0
                    or bbox.x1 > figure_bbox.x1
                    or bbox.y0 < figure_bbox.y0
                    or bbox.y1 > figure_bbox.y1
                    for bbox in visible_tick_boxes
                ),
            }
        )
    verification = {
        "tick_label_overlap": any(
            item["tick_label_overlap"] for item in verifications
        ),
        "text_outside_figure": any(
            item["text_outside_figure"] for item in verifications
        )
        or _artist_overlaps_axes(fig, legend, axes.flat),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return verification, summary


def build_figures(experiment_root: Path, system_name: str | None, *, jpb_y_scale: str = "log") -> Path:
    experiment_root = experiment_root.resolve()
    experiment_config = load_experiment_config(experiment_root / "config.json")
    resolved_system_names = load_system_names(experiment_root, system_name)
    micro_manifest = None
    jpb_manifest = None
    if (experiment_config.microbenchmarks is not None
            and experiment_config.microbenchmarks.system_name in resolved_system_names):
        micro_manifest = load_microbenchmark_manifest(experiment_root)
        if micro_manifest["config"] != json.loads(json.dumps(asdict(experiment_config.microbenchmarks))):
            raise ValueError("saved microbenchmark measurements do not match config.json; rerun the microbenchmarks")
    if (experiment_config.jpb_microbenchmarks is not None
            and experiment_config.jpb_microbenchmarks.system_name in resolved_system_names):
        jpb_manifest = load_jpb_manifest(experiment_root, experiment_config)
    has_sql_results = (experiment_root / "results.duckdb").is_file()
    if not has_sql_results and micro_manifest is None and jpb_manifest is None:
        raise FileNotFoundError("no SQL results or selected completed microbenchmarks in the experiment")
    rows = load_metric_rows(experiment_root, resolved_system_names) if has_sql_results else []
    plan_metric_rows = load_plan_metric_rows(experiment_root, resolved_system_names) if has_sql_results else []
    probe_scan_rows = load_probe_scan_rows(experiment_root, resolved_system_names) if has_sql_results else []
    output_dir = experiment_root / "paper"
    phase_report = output_dir / "filter_phases.csv"
    write_filter_phase_report(phase_report, rows)
    figures_dir = output_dir / "figures"
    available_workloads = _available_workload_names(rows)
    summary: dict[str, object] = {
        "experiment_root": str(experiment_root),
        "system_names": list(resolved_system_names),
        "figures": {},
        "sections": {},
        "verification": {},
        "skipped_figures": [],
        "filter_phases_csv": str(phase_report),
    }
    if micro_manifest is not None:
        micro_dir = experiment_root / ARTIFACT_SUBDIR
        artifacts = build_filter_benchmark_artifacts(
            csv_path=micro_dir / "point.csv", range_csv_path=micro_dir / "range.csv",
            output_dir=output_dir, figures_dir=figures_dir,
            aggregate=micro_manifest["config"]["aggregate"],
        )
        summary["figures"]["filter_benchmark_overview"] = artifacts["figure_path"]
        summary["verification"]["filter_benchmark_overview"] = artifacts["verification"]
        summary["sections"]["filter_microbenchmarks"] = {
            **artifacts, "manifest_path": str(micro_dir / "manifest.json"),
            "unavailable_filters": micro_manifest["unavailable_filters"],
        }
    if jpb_manifest is not None:
        jpb_dir = experiment_root / JPB_ARTIFACT_SUBDIR
        artifacts = build_jpb_artifacts(jpb_dir / "measurements.csv", output_dir, figures_dir,
                                        jpb_manifest["unavailable_filters"], y_scale=jpb_y_scale)
        summary["figures"]["jpb_simulated_filtering"] = artifacts["figure_path"]
        summary["verification"]["jpb_simulated_filtering"] = artifacts["verification"]
        summary["sections"]["jpb_microbenchmarks"] = {**artifacts, "manifest_path": str(jpb_dir / "manifest.json")}
    for figure in FIGURES:
        if not _figure_is_available(figure, available_workloads):
            summary["skipped_figures"].append(figure.slug)
            continue
        if figure.metric in {"runtime_and_probe_rows", "runtime_and_files_read"}:
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
                logical_probe_rows=(
                    next(
                        (
                            workload.fact_rows
                            for workload in experiment_config.workloads
                            if workload.name == workload_name
                            and hasattr(workload, "fact_rows")
                        ),
                        None,
                    )
                    if workload_name == "join-pruning-workload"
                    and figure.probe_table == "fact"
                    else None
                ),
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
            detail_metric_key = (
                "files_read_fraction"
                if figure.metric == "runtime_and_files_read"
                else "physical_rows_scanned"
            )
            rows_summary = summarize_section(
                [
                    {
                        **entry,
                        "series": [
                            {**item, "value": item[detail_metric_key]}
                            for item in entry["series"]
                        ],
                    }
                    for entry in entries
                ],
                detail_metric_key,
            )
            section_summary = {
                "runtime": runtime_summary,
                (
                    "fact_files_opened_fraction"
                    if figure.metric == "runtime_and_files_read"
                    else "probe_rows"
                ): (
                    rows_summary
                    if figure.metric == "runtime_and_files_read"
                    else {
                        metric_key: summarize_section(
                            [
                                {
                                    **entry,
                                    "series": [
                                        {**item, "value": item[metric_key]}
                                        for item in entry["series"]
                                    ],
                                }
                                for entry in entries
                            ],
                            metric_key,
                        )
                        for metric_key in (
                            "logical_probe_rows",
                            "physical_rows_scanned",
                            "join_probe_rows",
                            "probe_rows_filtered_after_scan",
                            "probe_rows_skipped_at_scan",
                        )
                    }
                ),
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
                split_scan_pruning=(
                    figure.workloads == ("join-pruning-workload",)
                ),
                show_filter_phases=figure.workloads[0].startswith("join-pruning-workload"),
            )
        elif figure.metric == "runtime_and_files_read":
            verification = _draw_runtime_total_rows_bars(
                figure_path,
                entries=entries,
                width_key=figure.width,
                placement_width_key=figure.placement_width,
                show_query_labels=figure.show_query_labels,
                show_runtime_labels=figure.show_runtime_labels,
                workload_labels=figure.query_workload_labels,
                runtime_log_axis=figure.runtime_log_axis,
                rows_log_axis=False,
                rows_metric_key="files_read_fraction",
                rows_y_label="Fact files opened",
                rows_percent=True,
                show_filter_phases=figure.workloads[0].startswith("join-pruning-workload"),
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

    sweep_workloads = {panel.workload for panel in PARAMETER_SWEEP_PANELS}
    if sweep_workloads.issubset(available_workloads):
        sweep_path = figures_dir / "parameter_sweep_overview.pdf"
        sweep_verification, sweep_summary = _draw_parameter_sweep_overview(
            sweep_path,
            rows=[row for row in rows if row.workload_name in sweep_workloads],
        )
        summary["figures"]["parameter_sweep_overview"] = str(sweep_path)
        summary["verification"]["parameter_sweep_overview"] = sweep_verification
        summary["sections"]["parameter_sweeps"] = sweep_summary
    elif sweep_workloads.intersection(available_workloads):
        summary["skipped_figures"].append("parameter_sweep_overview")

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
            system_name=", ".join(resolved_system_names),
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
        f"- systems: `{', '.join(summary['system_names'])}`",
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
    parser.add_argument("--jpb-y-scale", choices=("log", "linear"), default="log")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = build_figures(args.experiment_root, args.system_name, jpb_y_scale=args.jpb_y_scale)
    print(output_dir)


if __name__ == "__main__":
    main()
