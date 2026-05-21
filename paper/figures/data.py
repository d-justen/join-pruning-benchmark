from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking import load_experiment_config
from paper.figure_specs import QUERY_LABELS, VARIANT_ORDER, WORKLOAD_LABELS


@dataclass(frozen=True)
class MetricRow:
    workload_name: str
    query_name: str
    variant_label: str
    median_ms: float
    probe_rows_scanned: int | None
    total_rows_scanned: int | None


def geometric_mean(values: list[float]) -> float | None:
    filtered = [value for value in values if value > 0]
    if not filtered:
        return None
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))


def workload_label(name: str) -> str:
    return WORKLOAD_LABELS.get(name, name)


def display_label(query_name: str) -> str:
    return QUERY_LABELS.get(query_name, query_name.removesuffix(".sql"))


def load_system_name(experiment_root: Path, requested: str | None) -> str:
    config = load_experiment_config(experiment_root / "config.json")
    system_names = [system.name for system in config.systems]
    if requested is not None:
        if requested not in system_names:
            raise ValueError(f"unknown system {requested!r}; available: {', '.join(system_names)}")
        return requested
    if len(system_names) != 1:
        raise ValueError("multiple systems configured; pass --system-name explicitly")
    return system_names[0]


def load_metric_rows(experiment_root: Path, system_name: str) -> list[MetricRow]:
    config = load_experiment_config(experiment_root / "config.json")
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        rows = con.execute(
            """
            WITH timing AS (
                SELECT
                    workload_name,
                    query_name,
                    variant_label,
                    median(elapsed_ns) / 1000000.0 AS median_ms
                FROM timing_runs
                WHERE experiment_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3
            ),
            profiles AS (
                SELECT
                    workload_name,
                    query_name,
                    variant_label,
                    max(probe_rows_scanned) AS probe_rows_scanned,
                    max(total_rows_scanned) AS total_rows_scanned
                FROM profile_runs
                WHERE experiment_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3
            )
            SELECT
                timing.workload_name,
                timing.query_name,
                timing.variant_label,
                timing.median_ms,
                profiles.probe_rows_scanned,
                profiles.total_rows_scanned
            FROM timing
            LEFT JOIN profiles
              ON profiles.workload_name = timing.workload_name
             AND profiles.query_name = timing.query_name
             AND profiles.variant_label = timing.variant_label
            ORDER BY timing.workload_name, timing.query_name, timing.variant_label
            """,
            [config.experiment_name, system_name, config.experiment_name, system_name],
        ).fetchall()
    finally:
        con.close()
    return [
        MetricRow(
            workload_name=row[0],
            query_name=row[1],
            variant_label=row[2],
            median_ms=row[3],
            probe_rows_scanned=row[4],
            total_rows_scanned=row[5],
        )
        for row in rows
    ]


def select_queries(rows: list[MetricRow], workload_name: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    if requested:
        return requested
    present = sorted({row.query_name for row in rows if row.workload_name == workload_name})
    return tuple(present)


def _metric_value(row: MetricRow, metric: str) -> float | int | None:
    if metric == "runtime_ms":
        return row.median_ms
    if metric == "total_rows":
        return row.total_rows_scanned
    raise ValueError(f"unsupported metric: {metric}")


def build_absolute_table(
    rows: list[MetricRow],
    workload_name: str,
    query_names: tuple[str, ...],
    metric: str,
) -> list[dict[str, object]]:
    by_key = {(row.workload_name, row.query_name, row.variant_label): row for row in rows}
    result: list[dict[str, object]] = []
    for query_name in query_names:
        series: list[dict[str, object]] = []
        for variant_label in VARIANT_ORDER:
            row = by_key.get((workload_name, query_name, variant_label))
            if row is None:
                continue
            series.append(
                {
                    "variant_label": variant_label,
                    "value": _metric_value(row, metric),
                    "median_ms": row.median_ms,
                    "probe_rows_scanned": row.probe_rows_scanned,
                    "total_rows_scanned": row.total_rows_scanned,
                }
            )
        if series:
            result.append(
                {
                    "workload_name": workload_name,
                    "workload_label": workload_label(workload_name),
                    "query_name": query_name,
                    "query_label": display_label(query_name),
                    "series": series,
                }
            )
    return result


def summarize_section(entries: list[dict[str, object]], metric: str) -> dict[str, object]:
    relative_values = {label: [] for label in VARIANT_ORDER}
    absolute_totals = {label: 0.0 for label in VARIANT_ORDER}
    for entry in entries:
        metric_values = [
            item["value"]
            for item in entry["series"]
            if item["value"] is not None and item["value"] > 0
        ]
        if not metric_values:
            continue
        fastest_value = min(metric_values)
        for item in entry["series"]:
            value = item["value"]
            if value is None or value <= 0:
                continue
            relative_values[item["variant_label"]].append(value / fastest_value)
            absolute_totals[item["variant_label"]] += float(value)
    summary_key = "absolute_runtime_totals_ms" if metric == "runtime_ms" else "absolute_cumulative_rows_totals"
    return {
        "geometric_means_fastest_equals_one": {
            label: geometric_mean(values)
            for label, values in relative_values.items()
        },
        summary_key: absolute_totals,
    }


def summarize_end_to_end(rows: list[MetricRow]) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, MetricRow]] = {}
    absolute_totals = {label: 0.0 for label in VARIANT_ORDER}
    by_variant = {label: [] for label in VARIANT_ORDER}
    for row in rows:
        grouped.setdefault((row.workload_name, row.query_name), {})[row.variant_label] = row
        absolute_totals[row.variant_label] += row.median_ms
    max_runtime = {"variant": None, "query": None, "value": 0.0}
    min_runtime = {"variant": None, "query": None, "value": None}
    for (workload_name, query_name), variant_rows in sorted(grouped.items()):
        query_times = [row.median_ms for row in variant_rows.values() if row.median_ms > 0]
        if not query_times:
            continue
        fastest = min(query_times)
        for variant_label, row in variant_rows.items():
            ratio = row.median_ms / fastest
            by_variant[variant_label].append(ratio)
            if row.median_ms > max_runtime["value"]:
                max_runtime = {
                    "variant": variant_label,
                    "query": f"{workload_label(workload_name)} / {display_label(query_name)}",
                    "value": row.median_ms,
                }
            if min_runtime["value"] is None or row.median_ms < min_runtime["value"]:
                min_runtime = {
                    "variant": variant_label,
                    "query": f"{workload_label(workload_name)} / {display_label(query_name)}",
                    "value": row.median_ms,
                }
    return {
        "absolute_runtime_totals_ms": absolute_totals,
        "geometric_means_fastest_equals_one": {
            label: geometric_mean(values)
            for label, values in by_variant.items()
        },
        "max_observed_runtime_ms": max_runtime,
        "min_observed_runtime_ms": min_runtime,
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
