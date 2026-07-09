from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking import load_experiment_config
from paper.figure_specs import FIGURES, QUERY_LABELS, VARIANT_ORDER, WORKLOAD_LABELS


@dataclass(frozen=True)
class MetricRow:
    workload_name: str
    query_name: str
    variant_label: str
    median_ms: float
    probe_rows_scanned: int | None
    total_rows_scanned: int | None


@dataclass(frozen=True)
class ProbeScanRow:
    workload_name: str
    query_name: str
    variant_label: str
    scan_key: str
    table_name: str
    operator_rows_scanned: int
    operator_cardinality: int
    dynamic_filters: str


@dataclass(frozen=True)
class PlanMetricRow:
    workload_name: str
    query_name: str
    variant_label: str
    cumulative_rows_scanned: int
    cumulative_join_probe_tuples: int
    join_operator_count: int


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


def _normalize_dynamic_filters(raw: object) -> str:
    if isinstance(raw, list):
        return " ".join(str(item) for item in raw)
    if raw is None:
        return ""
    return str(raw)


def _has_optional_join_filter(dynamic_filters: str) -> bool:
    if "optional:" not in dynamic_filters:
        return False
    return any(token in dynamic_filters for token in (" IN BF(", " IN PRF(", ">=", "<="))


def _table_basename(table_name: str) -> str:
    return table_name.rsplit(".", 1)[-1].strip('"')


def _configured_probe_tables() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for figure in FIGURES:
        if figure.probe_table is None:
            continue
        for workload_name in figure.workloads:
            result.setdefault(workload_name, set()).add(figure.probe_table)
    return result


def _extract_probe_scan_rows(
    profile_data: dict,
    *,
    workload_name: str,
    query_name: str,
    variant_label: str,
    probe_tables: set[str],
) -> list[ProbeScanRow]:
    extracted: list[ProbeScanRow] = []

    def visit(node: dict, path: tuple[int, ...]) -> None:
        extra_info = node.get("extra_info", {})
        if node.get("operator_type") == "TABLE_SCAN" and isinstance(extra_info, dict):
            dynamic_filters = _normalize_dynamic_filters(extra_info.get("Dynamic Filters"))
            operator_rows_scanned = node.get("operator_rows_scanned")
            operator_cardinality = node.get("operator_cardinality")
            table_name = str(extra_info.get("Table", ""))
            is_probe_table = _table_basename(table_name) in probe_tables
            if (
                isinstance(operator_rows_scanned, int)
                and isinstance(operator_cardinality, int)
                and table_name
                and (
                    is_probe_table
                    or _has_optional_join_filter(dynamic_filters)
                )
            ):
                if (
                    is_probe_table
                    and not 0 <= operator_cardinality <= operator_rows_scanned
                ):
                    raise ValueError(
                        f"invalid scan cardinality for {workload_name}/{query_name}/"
                        f"{variant_label}/{table_name}: {operator_cardinality} output rows "
                        f"from {operator_rows_scanned} scanned rows"
                    )
                scan_key = f"{table_name}::{'/'.join(str(part) for part in path)}"
                extracted.append(
                    ProbeScanRow(
                        workload_name=workload_name,
                        query_name=query_name,
                        variant_label=variant_label,
                        scan_key=scan_key,
                        table_name=table_name,
                        operator_rows_scanned=operator_rows_scanned,
                        operator_cardinality=operator_cardinality,
                        dynamic_filters=dynamic_filters,
                    )
                )
        for index, child in enumerate(node.get("children", [])):
            visit(child, path + (index,))

    visit(profile_data, ())
    return extracted


def load_probe_scan_rows(experiment_root: Path, system_name: str) -> list[ProbeScanRow]:
    config = load_experiment_config(experiment_root / "config.json")
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT workload_name, query_name, variant_label, profile_path
            FROM profile_runs
            WHERE experiment_name = ?
              AND system_name = ?
            ORDER BY workload_name, query_name, variant_label
            """,
            [config.experiment_name, system_name],
        ).fetchall()
    finally:
        con.close()

    extracted: list[ProbeScanRow] = []
    probe_tables_by_workload = _configured_probe_tables()
    for workload_name, query_name, variant_label, profile_path in rows:
        profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        extracted.extend(
            _extract_probe_scan_rows(
                profile_data,
                workload_name=workload_name,
                query_name=query_name,
                variant_label=variant_label,
                probe_tables=probe_tables_by_workload.get(workload_name, set()),
            )
        )
    return extracted


def _extract_join_probe_metrics(profile_data: dict) -> tuple[int, int]:
    left_subtree_scan_cardinalities: dict[tuple[int, ...], int] = {}
    join_operator_count = 0

    def collect_scan_cardinalities(node: dict, path: tuple[int, ...]) -> None:
        if node.get("operator_type") == "TABLE_SCAN":
            operator_cardinality = node.get("operator_cardinality")
            if isinstance(operator_cardinality, int):
                left_subtree_scan_cardinalities[path] = operator_cardinality
        for index, child in enumerate(node.get("children", [])):
            collect_scan_cardinalities(child, path + (index,))

    def visit(node: dict, path: tuple[int, ...]) -> None:
        nonlocal join_operator_count
        children = node.get("children", [])
        operator_type = str(node.get("operator_type", ""))
        if operator_type.endswith("_JOIN"):
            join_operator_count += 1
            if children:
                collect_scan_cardinalities(children[0], path + (0,))
        for index, child in enumerate(children):
            visit(child, path + (index,))

    visit(profile_data, ())
    return sum(left_subtree_scan_cardinalities.values()), join_operator_count


def load_plan_metric_rows(
    experiment_root: Path,
    system_name: str,
) -> list[PlanMetricRow]:
    config = load_experiment_config(experiment_root / "config.json")
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                workload_name,
                query_name,
                variant_label,
                profile_path,
                total_rows_scanned
            FROM profile_runs
            WHERE experiment_name = ?
              AND system_name = ?
            ORDER BY workload_name, query_name, variant_label
            """,
            [config.experiment_name, system_name],
        ).fetchall()
    finally:
        con.close()

    result: list[PlanMetricRow] = []
    for workload_name, query_name, variant_label, profile_path, total_rows_scanned in rows:
        profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        profile_total = profile_data.get("cumulative_rows_scanned")
        cumulative_rows_scanned = (
            total_rows_scanned
            if isinstance(total_rows_scanned, int)
            else profile_total
        )
        if not isinstance(cumulative_rows_scanned, int):
            raise ValueError(
                f"missing cumulative rows scanned for {workload_name}/{query_name}/"
                f"{variant_label}"
            )
        cumulative_join_probe_tuples, join_operator_count = (
            _extract_join_probe_metrics(profile_data)
        )
        result.append(
            PlanMetricRow(
                workload_name=workload_name,
                query_name=query_name,
                variant_label=variant_label,
                cumulative_rows_scanned=cumulative_rows_scanned,
                cumulative_join_probe_tuples=cumulative_join_probe_tuples,
                join_operator_count=join_operator_count,
            )
        )
    return result


def build_runtime_probe_table(
    rows: list[MetricRow],
    scan_rows: list[ProbeScanRow],
    workload_name: str,
    query_names: tuple[str, ...],
    probe_table: str,
) -> list[dict[str, object]]:
    metric_rows = {
        (row.query_name, row.variant_label): row
        for row in rows
        if row.workload_name == workload_name
    }
    probe_scans: dict[tuple[str, str], ProbeScanRow] = {}
    for scan_row in scan_rows:
        if (
            scan_row.workload_name != workload_name
            or _table_basename(scan_row.table_name) != probe_table
        ):
            continue
        key = (scan_row.query_name, scan_row.variant_label)
        if key in probe_scans:
            raise ValueError(
                f"multiple {probe_table} scans found for {workload_name}/{scan_row.query_name}/"
                f"{scan_row.variant_label}"
            )
        probe_scans[key] = scan_row

    entries: list[dict[str, object]] = []
    for query_name in query_names:
        series: list[dict[str, object]] = []
        for variant_label in VARIANT_ORDER:
            metric_row = metric_rows.get((query_name, variant_label))
            scan_row = probe_scans.get((query_name, variant_label))
            if metric_row is None:
                raise ValueError(
                    f"missing timing row for {workload_name}/{query_name}/{variant_label}"
                )
            if scan_row is None:
                raise ValueError(
                    f"missing {probe_table} scan profile for "
                    f"{workload_name}/{query_name}/{variant_label}"
                )
            series.append(
                {
                    "variant_label": variant_label,
                    "median_ms": metric_row.median_ms,
                    "probe_rows_scanned": scan_row.operator_rows_scanned,
                    "probe_output_cardinality": scan_row.operator_cardinality,
                    "probe_rows_filtered_out": (
                        scan_row.operator_rows_scanned - scan_row.operator_cardinality
                    ),
                }
            )
        entries.append(
            {
                "workload_name": workload_name,
                "workload_label": workload_label(workload_name),
                "query_name": query_name,
                "query_label": display_label(query_name),
                "series": series,
            }
        )
    return entries


def build_runtime_plan_work_table(
    rows: list[MetricRow],
    plan_rows: list[PlanMetricRow],
    workload_name: str,
    query_names: tuple[str, ...],
) -> list[dict[str, object]]:
    metric_rows = {
        (row.query_name, row.variant_label): row
        for row in rows
        if row.workload_name == workload_name
    }
    plan_by_key = {
        (row.query_name, row.variant_label): row
        for row in plan_rows
        if row.workload_name == workload_name
    }
    entries: list[dict[str, object]] = []
    for query_name in query_names:
        series: list[dict[str, object]] = []
        for variant_label in VARIANT_ORDER:
            metric_row = metric_rows.get((query_name, variant_label))
            plan_row = plan_by_key.get((query_name, variant_label))
            if metric_row is None:
                raise ValueError(
                    f"missing timing row for {workload_name}/{query_name}/{variant_label}"
                )
            if plan_row is None:
                raise ValueError(
                    f"missing plan metrics for {workload_name}/{query_name}/{variant_label}"
                )
            series.append(
                {
                    "variant_label": variant_label,
                    "median_ms": metric_row.median_ms,
                    "rows_scanned": plan_row.cumulative_rows_scanned,
                    "join_probe_tuples": plan_row.cumulative_join_probe_tuples,
                    "join_operator_count": plan_row.join_operator_count,
                }
            )
        entries.append(
            {
                "workload_name": workload_name,
                "workload_label": workload_label(workload_name),
                "query_name": query_name,
                "query_label": display_label(query_name),
                "series": series,
            }
        )
    return entries


def build_benchmark_runtime_distributions(
    rows: list[MetricRow],
    benchmark_queries: tuple[tuple[str, str, tuple[str, ...] | None], ...],
) -> list[dict[str, object]]:
    distributions: list[dict[str, object]] = []
    for workload_name, benchmark_label, requested_queries in benchmark_queries:
        available_queries = {
            row.query_name
            for row in rows
            if row.workload_name == workload_name
        }
        query_names = (
            tuple(sorted(available_queries))
            if requested_queries is None
            else requested_queries
        )
        by_key = {
            (row.query_name, row.variant_label): row.median_ms
            for row in rows
            if row.workload_name == workload_name
        }
        values_by_variant = {label: [] for label in VARIANT_ORDER}
        for query_name in query_names:
            missing_variants = [
                variant_label
                for variant_label in VARIANT_ORDER
                if (query_name, variant_label) not in by_key
            ]
            if missing_variants:
                raise ValueError(
                    f"incomplete runtime data for {workload_name}/{query_name}: "
                    f"missing {', '.join(missing_variants)}"
                )
            for variant_label in VARIANT_ORDER:
                runtime_ms = by_key[(query_name, variant_label)]
                if runtime_ms <= 0:
                    raise ValueError(
                        f"non-positive runtime for {workload_name}/{query_name}/"
                        f"{variant_label}: {runtime_ms}"
                    )
                values_by_variant[variant_label].append(runtime_ms)
        distributions.append(
            {
                "workload_name": workload_name,
                "benchmark_label": benchmark_label,
                "query_count": len(query_names),
                "values_by_variant": values_by_variant,
            }
        )
    return distributions


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_benchmark_runtime_distributions(
    distributions: list[dict[str, object]],
) -> dict[str, object]:
    benchmarks: dict[str, object] = {}
    for distribution in distributions:
        variants: dict[str, object] = {}
        values_by_variant = distribution["values_by_variant"]
        for variant_label in VARIANT_ORDER:
            values = values_by_variant[variant_label]
            variants[variant_label] = {
                "query_count": len(values),
                "minimum_ms": min(values) if values else None,
                "q1_ms": _percentile(values, 0.25),
                "median_ms": _percentile(values, 0.5),
                "q3_ms": _percentile(values, 0.75),
                "maximum_ms": max(values) if values else None,
                "geometric_mean_ms": geometric_mean(values),
            }
        benchmarks[distribution["workload_name"]] = {
            "benchmark_label": distribution["benchmark_label"],
            "query_count": distribution["query_count"],
            "variants": variants,
        }
    return {"benchmarks": benchmarks}


def summarize_benchmark_metrics(
    rows: list[MetricRow],
    plan_rows: list[PlanMetricRow],
    benchmark_queries: tuple[tuple[str, str, tuple[str, ...] | None], ...],
) -> dict[str, object]:
    timing_by_key = {
        (row.workload_name, row.query_name, row.variant_label): row.median_ms
        for row in rows
    }
    plan_by_key = {
        (row.workload_name, row.query_name, row.variant_label): row
        for row in plan_rows
    }
    benchmarks: list[dict[str, object]] = []
    for workload_name, benchmark_label, requested_queries in benchmark_queries:
        available_queries = {
            row.query_name
            for row in rows
            if row.workload_name == workload_name
        }
        query_names = (
            tuple(sorted(available_queries))
            if requested_queries is None
            else requested_queries
        )
        metric_values = {
            "runtime_ms": {label: [] for label in VARIANT_ORDER},
            "rows_scanned": {label: [] for label in VARIANT_ORDER},
            "join_probe_tuples": {label: [] for label in VARIANT_ORDER},
        }
        relative_values = {
            metric: {label: [] for label in VARIANT_ORDER}
            for metric in metric_values
        }
        relative_query_counts = {metric: 0 for metric in metric_values}
        join_free_query_count = 0

        for query_name in query_names:
            query_metrics = {
                "runtime_ms": {},
                "rows_scanned": {},
                "join_probe_tuples": {},
            }
            query_has_join = False
            for variant_label in VARIANT_ORDER:
                key = (workload_name, query_name, variant_label)
                runtime_ms = timing_by_key.get(key)
                plan_row = plan_by_key.get(key)
                if runtime_ms is None or plan_row is None:
                    raise ValueError(
                        f"incomplete benchmark summary data for {workload_name}/"
                        f"{query_name}/{variant_label}"
                    )
                query_metrics["runtime_ms"][variant_label] = runtime_ms
                query_metrics["rows_scanned"][variant_label] = (
                    plan_row.cumulative_rows_scanned
                )
                query_metrics["join_probe_tuples"][variant_label] = (
                    plan_row.cumulative_join_probe_tuples
                )
                query_has_join = query_has_join or plan_row.join_operator_count > 0

            if not query_has_join:
                join_free_query_count += 1
            for metric, values_by_variant in query_metrics.items():
                for variant_label, value in values_by_variant.items():
                    metric_values[metric][variant_label].append(value)
                values = list(values_by_variant.values())
                if any(value <= 0 for value in values):
                    continue
                best = min(values)
                relative_query_counts[metric] += 1
                for variant_label, value in values_by_variant.items():
                    relative_values[metric][variant_label].append(value / best)

        variants: dict[str, object] = {}
        for variant_label in VARIANT_ORDER:
            variants[variant_label] = {
                "total_runtime_ms": sum(
                    metric_values["runtime_ms"][variant_label]
                ),
                "relative_runtime_geomean": geometric_mean(
                    relative_values["runtime_ms"][variant_label]
                ),
                "total_rows_scanned": sum(
                    metric_values["rows_scanned"][variant_label]
                ),
                "relative_rows_scanned_geomean": geometric_mean(
                    relative_values["rows_scanned"][variant_label]
                ),
                "total_join_probe_tuples": sum(
                    metric_values["join_probe_tuples"][variant_label]
                ),
                "relative_join_probe_geomean": geometric_mean(
                    relative_values["join_probe_tuples"][variant_label]
                ),
            }
        benchmarks.append(
            {
                "workload_name": workload_name,
                "benchmark_label": benchmark_label,
                "query_count": len(query_names),
                "join_free_query_count": join_free_query_count,
                "relative_query_counts": relative_query_counts,
                "relative_excluded_query_counts": {
                    metric: len(query_names) - count
                    for metric, count in relative_query_counts.items()
                },
                "variants": variants,
            }
        )
    return {"benchmarks": benchmarks}


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


def summarize_benchmark_runtime(
    rows: list[MetricRow],
    *,
    workload_name: str,
    query_names: tuple[str, ...],
    benchmark_label: str,
) -> dict[str, object]:
    grouped: dict[str, dict[str, MetricRow]] = {}
    for row in rows:
        if row.workload_name != workload_name or row.query_name not in query_names:
            continue
        grouped.setdefault(row.query_name, {})[row.variant_label] = row

    absolute_totals = {label: 0.0 for label in VARIANT_ORDER}
    relative_values = {label: [] for label in VARIANT_ORDER}
    max_speedup = {
        label: {"query": None, "factor": None}
        for label in VARIANT_ORDER
        if label != "no prefilter"
    }
    max_slowdown = {
        label: {"query": None, "factor": None}
        for label in VARIANT_ORDER
        if label != "no prefilter"
    }

    for query_name in query_names:
        variant_rows = grouped.get(query_name, {})
        query_times = [
            row.median_ms
            for row in variant_rows.values()
            if row.median_ms > 0
        ]
        if not query_times:
            continue
        fastest = min(query_times)
        for variant_label, row in variant_rows.items():
            absolute_totals[variant_label] += row.median_ms
            relative_values[variant_label].append(row.median_ms / fastest)

        baseline = variant_rows.get("no prefilter")
        if baseline is None or baseline.median_ms <= 0:
            continue
        for variant_label in VARIANT_ORDER:
            if variant_label == "no prefilter":
                continue
            row = variant_rows.get(variant_label)
            if row is None or row.median_ms <= 0:
                continue
            speedup_factor = baseline.median_ms / row.median_ms
            slowdown_factor = row.median_ms / baseline.median_ms
            if speedup_factor > 1.0:
                current = max_speedup[variant_label]["factor"]
                if current is None or speedup_factor > current:
                    max_speedup[variant_label] = {
                        "query": display_label(query_name),
                        "factor": speedup_factor,
                    }
            if slowdown_factor > 1.0:
                current = max_slowdown[variant_label]["factor"]
                if current is None or slowdown_factor > current:
                    max_slowdown[variant_label] = {
                        "query": display_label(query_name),
                        "factor": slowdown_factor,
                    }

    return {
        "benchmark_label": benchmark_label,
        "query_count": len(grouped),
        "absolute_runtime_totals_ms": absolute_totals,
        "geometric_means_fastest_equals_one": {
            label: geometric_mean(values)
            for label, values in relative_values.items()
        },
        "largest_speedup_vs_no_prefilter": max_speedup,
        "largest_slowdown_vs_no_prefilter": max_slowdown,
    }


def summarize_relative_probe_cardinality(scan_rows: list[ProbeScanRow]) -> dict[str, object]:
    variants = tuple(label for label in VARIANT_ORDER if label != "no prefilter")
    grouped = _group_relative_probe_scans(scan_rows, variants)

    relative_values = {label: [] for label in variants}
    included_count = 0
    excluded_missing_variants = 0
    for variant_rows in grouped.values():
        if any(label not in variant_rows for label in variants):
            excluded_missing_variants += 1
            continue
        cardinalities = [variant_rows[label].operator_cardinality for label in variants if variant_rows[label].operator_cardinality > 0]
        if len(cardinalities) != len(variants):
            continue
        best = min(cardinalities)
        included_count += 1
        for label in variants:
            relative_values[label].append(variant_rows[label].operator_cardinality / best)

    return {
        "included_probe_scans": included_count,
        "excluded_probe_scans_missing_variants": excluded_missing_variants,
        "sample_counts": {label: len(values) for label, values in relative_values.items()},
        "geometric_means_fastest_equals_one": {
            label: geometric_mean(values)
            for label, values in relative_values.items()
        },
    }


def build_relative_probe_cardinality_values(scan_rows: list[ProbeScanRow]) -> dict[str, list[float]]:
    variants = tuple(label for label in VARIANT_ORDER if label != "no prefilter")
    grouped = _group_relative_probe_scans(scan_rows, variants)

    relative_values = {label: [] for label in variants}
    for variant_rows in grouped.values():
        if any(label not in variant_rows for label in variants):
            continue
        cardinalities = [variant_rows[label].operator_cardinality for label in variants if variant_rows[label].operator_cardinality > 0]
        if len(cardinalities) != len(variants):
            continue
        best = min(cardinalities)
        for label in variants:
            relative_values[label].append(variant_rows[label].operator_cardinality / best)
    return relative_values


def _group_relative_probe_scans(
    scan_rows: list[ProbeScanRow],
    variants: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, ProbeScanRow]]:
    grouped: dict[tuple[str, str, str], dict[str, ProbeScanRow]] = {}
    for row in scan_rows:
        if row.variant_label not in variants:
            continue
        grouped.setdefault((row.workload_name, row.query_name, row.scan_key), {})[row.variant_label] = row
    return grouped


def build_relative_probe_cardinality_outliers(
    scan_rows: list[ProbeScanRow],
    *,
    threshold: float = 10.0,
) -> list[dict[str, object]]:
    variants = tuple(label for label in VARIANT_ORDER if label != "no prefilter")
    grouped = _group_relative_probe_scans(scan_rows, variants)
    outliers: list[dict[str, object]] = []
    for (workload_name, query_name, scan_key), variant_rows in grouped.items():
        if any(label not in variant_rows for label in variants):
            continue
        cardinalities = {
            label: variant_rows[label].operator_cardinality
            for label in variants
            if variant_rows[label].operator_cardinality > 0
        }
        if len(cardinalities) != len(variants):
            continue
        minimum = min(cardinalities.values())
        maximum = max(cardinalities.values())
        if minimum <= 0:
            continue
        ratio = maximum / minimum
        if ratio < threshold:
            continue
        table_name = next(iter(variant_rows.values())).table_name
        outliers.append(
            {
                "workload_name": workload_name,
                "workload_label": workload_label(workload_name),
                "query_name": query_name,
                "query_label": display_label(query_name),
                "scan_key": scan_key,
                "table_name": table_name,
                "ratio_max_to_min": ratio,
                "min_cardinality": minimum,
                "max_cardinality": maximum,
                "cardinalities": cardinalities,
                "dynamic_filters": {
                    label: variant_rows[label].dynamic_filters
                    for label in variants
                },
            }
        )
    outliers.sort(key=lambda item: (item["ratio_max_to_min"], item["max_cardinality"]), reverse=True)
    return outliers


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
