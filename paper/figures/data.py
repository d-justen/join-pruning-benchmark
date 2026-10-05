from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb

from benchmarking import load_experiment_config
from benchmarking.manifests import load_manifest
from paper.figure_specs import FIGURES, QUERY_LABELS, VARIANT_ORDER, WORKLOAD_LABELS


@dataclass(frozen=True)
class MetricRow:
    workload_name: str
    query_name: str
    variant_label: str
    median_ms: float
    probe_rows_scanned: int | None
    total_rows_scanned: int | None
    query_parameters: dict[str, object] | None = None
    prf_build_phase_ms: float | None = None
    prf_analysis_phase_ms: float | None = None
    prf_filter_count: int = 0


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
    join_probe_cardinality: int | None = None
    total_files_read: int | None = None
    total_files_skipped: int | None = None
    table_file_count: int | None = None


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


def load_system_names(experiment_root: Path, requested: str | None) -> tuple[str, ...]:
    config = load_experiment_config(experiment_root / "config.json")
    system_names = [system.name for system in config.systems]
    if requested is not None:
        if requested not in system_names:
            raise ValueError(f"unknown system {requested!r}; available: {', '.join(system_names)}")
        selected = (requested,)
    else:
        selected = tuple(system_names)
    selected_labels = [
        variant.label
        for system in config.systems
        if system.name in selected
        for variant in system.variants
    ]
    duplicates = sorted(
        label for label in set(selected_labels) if selected_labels.count(label) > 1
    )
    if duplicates:
        raise ValueError(
            "paper figures require unique variant labels across selected systems: "
            + ", ".join(duplicates)
        )
    return selected


def load_system_name(experiment_root: Path, requested: str | None) -> str:
    """Compatibility wrapper for callers that require exactly one system."""
    selected = load_system_names(experiment_root, requested)
    if len(selected) != 1:
        raise ValueError("multiple systems configured; pass --system-name explicitly")
    return selected[0]


def _system_predicate(system_names: tuple[str, ...]) -> tuple[str, list[str]]:
    if not system_names:
        raise ValueError("at least one system must be selected")
    return ", ".join("?" for _ in system_names), list(system_names)


def _runtime_ms_with_profile_fallback(
    median_ms: float,
    profile_path: str | None,
) -> float:
    if median_ms > 0:
        return median_ms
    if profile_path:
        profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        latency_seconds = profile_data.get("latency")
        if (
            isinstance(latency_seconds, (int, float))
            and not isinstance(latency_seconds, bool)
            and latency_seconds > 0
        ):
            return float(latency_seconds) * 1000.0
    # A zero from a millisecond-granularity clock means the query completed
    # within one tick. Keep it positive for logarithmic and ratio plots.
    return 0.5


def _prf_phase_metrics(profile_data: dict) -> tuple[float | None, float | None, int]:
    """Sum per-filter wall-clock spans from one query profile, never worker time."""
    filters = []
    stack = [profile_data]
    while stack:
        node = stack.pop()
        stack.extend(node.get("children", []))
        if "prefix_range_filter" in node:
            filters.append(node["prefix_range_filter"])
    if not filters:
        return None, None, 0

    def total_ms(field: str) -> float | None:
        values = [item.get(field) for item in filters]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            return None
        return sum(values) * 1000.0

    return (
        total_ms("build_phase_elapsed_seconds"),
        total_ms("compression_phase_elapsed_seconds"),
        len(filters),
    )


def load_metric_rows(experiment_root: Path, system_names: tuple[str, ...]) -> list[MetricRow]:
    config = load_experiment_config(experiment_root / "config.json")
    placeholders, system_parameters = _system_predicate(system_names)
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        timing_columns = {
            row[1] for row in con.execute("PRAGMA table_info('timing_runs')").fetchall()
        }
        query_parameters_projection = (
            "any_value(query_parameters) AS query_parameters"
            if "query_parameters" in timing_columns
            else "NULL::VARCHAR AS query_parameters"
        )
        rows = con.execute(
            f"""
            WITH timing AS (
                SELECT
                    workload_name,
                    query_name,
                    variant_label,
                    median(elapsed_ns) / 1000000.0 AS median_ms,
                    {query_parameters_projection}
                FROM timing_runs
                WHERE experiment_name = ?
                  AND system_name IN ({placeholders})
                GROUP BY 1, 2, 3
            ),
            profiles AS (
                SELECT
                    workload_name,
                    query_name,
                    variant_label,
                    max(probe_rows_scanned) AS probe_rows_scanned,
                    max(total_rows_scanned) AS total_rows_scanned,
                    max(profile_path) AS profile_path
                FROM profile_runs
                WHERE experiment_name = ?
                  AND system_name IN ({placeholders})
                GROUP BY 1, 2, 3
            )
            SELECT
                timing.workload_name,
                timing.query_name,
                timing.variant_label,
                timing.median_ms,
                timing.query_parameters,
                profiles.probe_rows_scanned,
                profiles.total_rows_scanned,
                profiles.profile_path
            FROM timing
            LEFT JOIN profiles
              ON profiles.workload_name = timing.workload_name
             AND profiles.query_name = timing.query_name
             AND profiles.variant_label = timing.variant_label
            ORDER BY timing.workload_name, timing.query_name, timing.variant_label
            """,
            [
                config.experiment_name,
                *system_parameters,
                config.experiment_name,
                *system_parameters,
            ],
        ).fetchall()
    finally:
        con.close()
    result: list[MetricRow] = []
    for row in rows:
        profile_path = row[7]
        prf_build_ms = prf_analysis_ms = None
        prf_filter_count = 0
        if profile_path:
            profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            prf_build_ms, prf_analysis_ms, prf_filter_count = _prf_phase_metrics(profile_data)
        result.append(
            MetricRow(
                workload_name=row[0],
                query_name=row[1],
                variant_label=row[2],
                # Older DuckDB builds expose CURRENT_TIMESTAMP at millisecond
                # precision. Use the high-resolution profile latency when their
                # repetition median is zero.
                median_ms=_runtime_ms_with_profile_fallback(float(row[3]), profile_path),
                probe_rows_scanned=row[5],
                total_rows_scanned=row[6],
                query_parameters=None if row[4] is None else json.loads(row[4]),
                prf_build_phase_ms=prf_build_ms,
                prf_analysis_phase_ms=prf_analysis_ms,
                prf_filter_count=prf_filter_count,
            )
        )
    return result


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


def _optional_int(raw: object) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _configured_probe_tables() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for figure in FIGURES:
        if figure.probe_table is None:
            continue
        for workload_name in figure.workloads:
            result.setdefault(workload_name, set()).add(figure.probe_table)
    return result


_PROBE_PATH_OPERATORS = {
    "FILTER",
    "PROJECTION",
    "TABLE_SCAN",
    "USE_BF",
}


def _join_probe_cardinalities_by_scan_path(
    profile_data: dict,
) -> dict[tuple[int, ...], int]:
    """Map direct scan-fed join probes to the rows entering their join."""
    result: dict[tuple[int, ...], int] = {}

    def find_scan_path(node: dict, path: tuple[int, ...]) -> tuple[int, ...] | None:
        operator_type = str(node.get("operator_type", ""))
        if operator_type == "TABLE_SCAN":
            return path
        children = node.get("children", [])
        if operator_type not in _PROBE_PATH_OPERATORS or len(children) != 1:
            return None
        return find_scan_path(children[0], path + (0,))

    def visit(node: dict, path: tuple[int, ...]) -> None:
        children = node.get("children", [])
        operator_type = str(node.get("operator_type", ""))
        if operator_type.endswith("_JOIN") and children:
            probe_root = children[0]
            scan_path = find_scan_path(probe_root, path + (0,))
            probe_cardinality = probe_root.get("operator_cardinality")
            if scan_path is not None and isinstance(probe_cardinality, int):
                result[scan_path] = probe_cardinality
        for index, child in enumerate(children):
            visit(child, path + (index,))

    visit(profile_data, ())
    return result


def _extract_probe_scan_rows(
    profile_data: dict,
    *,
    workload_name: str,
    query_name: str,
    variant_label: str,
    probe_tables: set[str],
    table_file_counts: dict[str, int],
) -> list[ProbeScanRow]:
    extracted: list[ProbeScanRow] = []
    join_probe_cardinalities = _join_probe_cardinalities_by_scan_path(profile_data)

    def visit(
        node: dict,
        path: tuple[int, ...],
        inherited_runtime_filters: tuple[str, ...] = (),
    ) -> None:
        extra_info = node.get("extra_info", {})
        operator_type = str(node.get("operator_type", ""))
        runtime_filters = inherited_runtime_filters
        if operator_type == "USE_BF" and isinstance(extra_info, dict):
            filter_name = str(extra_info.get("Filter", "runtime filter"))
            placement = str(extra_info.get("Tuple Placement", "operator"))
            runtime_filters = runtime_filters + (
                f"RPT {filter_name} ({placement})",
            )
        elif operator_type not in _PROBE_PATH_OPERATORS:
            runtime_filters = ()
        if operator_type == "TABLE_SCAN" and isinstance(extra_info, dict):
            scan_dynamic_filters = _normalize_dynamic_filters(
                extra_info.get("Dynamic Filters")
            )
            dynamic_filters = " ".join(
                value
                for value in (
                    scan_dynamic_filters,
                    " ".join(runtime_filters),
                )
                if value
            )
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
                    or runtime_filters
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
                        join_probe_cardinality=join_probe_cardinalities.get(path),
                        total_files_read=_optional_int(extra_info.get("Total Files Read")),
                        total_files_skipped=_optional_int(
                            extra_info.get("Total Files Skipped")
                        ),
                        table_file_count=table_file_counts.get(
                            _table_basename(table_name)
                        ),
                    )
                )
        for index, child in enumerate(node.get("children", [])):
            visit(child, path + (index,), runtime_filters)

    visit(profile_data, ())
    return extracted


def load_probe_scan_rows(experiment_root: Path, system_names: tuple[str, ...]) -> list[ProbeScanRow]:
    config = load_experiment_config(experiment_root / "config.json")
    placeholders, system_parameters = _system_predicate(system_names)
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT workload_name, query_name, variant_label, profile_path
            FROM profile_runs
            WHERE experiment_name = ?
              AND system_name IN ({placeholders})
            ORDER BY workload_name, query_name, variant_label
            """,
            [config.experiment_name, *system_parameters],
        ).fetchall()
    finally:
        con.close()

    table_file_counts_by_workload: dict[str, dict[str, int]] = {}
    for manifest_path in sorted((experiment_root / "workloads").glob("*/manifest.json")):
        manifest = load_manifest(manifest_path)
        catalog_path = manifest.parquet_catalog_path
        if catalog_path is None and manifest.storage_mode == "ducklake":
            catalog_path = manifest.remote_parquet_catalog_path
        if catalog_path is None:
            continue
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        table_file_counts_by_workload[manifest.name] = {
            table_name: len(file_paths)
            for table_name, file_paths in catalog.items()
        }

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
                table_file_counts=table_file_counts_by_workload.get(workload_name, {}),
            )
        )

    # COPY can leave a trailing, zero-row Parquet file on a size boundary.
    # DuckLake ignores that file when adding data, so prefer the number of
    # registered files observed by the scan operator over the export-path count.
    observed_file_counts: dict[tuple[str, str], int] = {}
    for row in extracted:
        if row.total_files_read is None:
            continue
        observed_count = row.total_files_read + (row.total_files_skipped or 0)
        key = (row.workload_name, _table_basename(row.table_name))
        observed_file_counts[key] = max(
            observed_count,
            observed_file_counts.get(key, 0),
        )
    return [
        replace(
            row,
            table_file_count=observed_file_counts.get(
                (row.workload_name, _table_basename(row.table_name)),
                row.table_file_count,
            ),
        )
        for row in extracted
    ]


def _extract_join_probe_metrics(profile_data: dict) -> tuple[int, int]:
    join_probe_tuples = 0
    join_operator_count = 0

    def visit(node: dict, path: tuple[int, ...]) -> None:
        nonlocal join_operator_count, join_probe_tuples
        children = node.get("children", [])
        operator_type = str(node.get("operator_type", ""))
        if operator_type.endswith("_JOIN"):
            join_operator_count += 1
            if children:
                probe_cardinality = children[0].get("operator_cardinality")
                if isinstance(probe_cardinality, int):
                    join_probe_tuples += probe_cardinality
        for index, child in enumerate(children):
            visit(child, path + (index,))

    visit(profile_data, ())
    return join_probe_tuples, join_operator_count


def load_plan_metric_rows(
    experiment_root: Path,
    system_names: tuple[str, ...],
) -> list[PlanMetricRow]:
    config = load_experiment_config(experiment_root / "config.json")
    placeholders, system_parameters = _system_predicate(system_names)
    con = duckdb.connect(str(experiment_root / "results.duckdb"), read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT
                workload_name,
                query_name,
                variant_label,
                profile_path,
                total_rows_scanned
            FROM profile_runs
            WHERE experiment_name = ?
              AND system_name IN ({placeholders})
            ORDER BY workload_name, query_name, variant_label
            """,
            [config.experiment_name, *system_parameters],
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
    logical_probe_rows: int | None = None,
) -> list[dict[str, object]]:
    metric_rows = {
        (row.query_name, row.variant_label): row
        for row in rows
        if row.workload_name == workload_name
    }
    probe_scans: dict[tuple[str, str], list[ProbeScanRow]] = {}
    for scan_row in scan_rows:
        if (
            scan_row.workload_name != workload_name
            or _table_basename(scan_row.table_name) != probe_table
        ):
            continue
        key = (scan_row.query_name, scan_row.variant_label)
        probe_scans.setdefault(key, []).append(scan_row)

    entries: list[dict[str, object]] = []
    for query_name in query_names:
        series: list[dict[str, object]] = []
        for variant_label in VARIANT_ORDER:
            metric_row = metric_rows.get((query_name, variant_label))
            matching_scans = probe_scans.get((query_name, variant_label))
            if metric_row is None:
                continue
            if not matching_scans:
                raise ValueError(
                    f"missing {probe_table} scan profile for "
                    f"{workload_name}/{query_name}/{variant_label}"
                )
            probe_rows_scanned = sum(row.operator_rows_scanned for row in matching_scans)
            join_probe_rows = sum(
                row.join_probe_cardinality
                if row.join_probe_cardinality is not None
                else row.operator_cardinality
                for row in matching_scans
            )
            base_probe_rows = (
                logical_probe_rows
                if logical_probe_rows is not None
                else probe_rows_scanned
            )
            if not 0 <= join_probe_rows <= probe_rows_scanned <= base_probe_rows:
                raise ValueError(
                    f"invalid probe accounting for {workload_name}/{query_name}/"
                    f"{variant_label}/{probe_table}: join probe={join_probe_rows}, "
                    f"physical scan={probe_rows_scanned}, logical input={base_probe_rows}"
                )
            files_read = (
                None
                if any(row.total_files_read is None for row in matching_scans)
                else sum(int(row.total_files_read) for row in matching_scans)
            )
            table_file_count = (
                None
                if any(row.table_file_count is None for row in matching_scans)
                else sum(int(row.table_file_count) for row in matching_scans)
            )
            series.append(
                {
                    "variant_label": variant_label,
                    "median_ms": metric_row.median_ms,
                    "prf_build_phase_ms": metric_row.prf_build_phase_ms,
                    "prf_analysis_phase_ms": metric_row.prf_analysis_phase_ms,
                    "logical_probe_rows": base_probe_rows,
                    "physical_rows_scanned": probe_rows_scanned,
                    "join_probe_rows": join_probe_rows,
                    "probe_rows_skipped_at_scan": (
                        base_probe_rows - probe_rows_scanned
                    ),
                    "probe_rows_filtered_after_scan": (
                        probe_rows_scanned - join_probe_rows
                    ),
                    # Compatibility aliases for consumers outside the JPB figures.
                    "probe_rows_scanned": probe_rows_scanned,
                    "probe_output_cardinality": join_probe_rows,
                    "probe_rows_filtered_out": (
                        probe_rows_scanned - join_probe_rows
                    ),
                    "files_read": files_read,
                    "table_file_count": table_file_count,
                    "files_read_fraction": (
                        None
                        if files_read is None or table_file_count in (None, 0)
                        else files_read / table_file_count
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
                continue
            if plan_row is None:
                raise ValueError(
                    f"missing plan metrics for {workload_name}/{query_name}/{variant_label}"
                )
            series.append(
                {
                    "variant_label": variant_label,
                    "median_ms": metric_row.median_ms,
                    "prf_build_phase_ms": metric_row.prf_build_phase_ms,
                    "prf_analysis_phase_ms": metric_row.prf_analysis_phase_ms,
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
        active_variants = tuple(
            label
            for label in VARIANT_ORDER
            if any((query_name, label) in by_key for query_name in query_names)
        )
        values_by_variant = {label: [] for label in VARIANT_ORDER}
        for query_name in query_names:
            missing_variants = [
                variant_label
                for variant_label in active_variants
                if (query_name, variant_label) not in by_key
            ]
            if missing_variants:
                raise ValueError(
                    f"incomplete runtime data for {workload_name}/{query_name}: "
                    f"missing {', '.join(missing_variants)}"
                )
            for variant_label in active_variants:
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
        active_variants = tuple(
            label
            for label in VARIANT_ORDER
            if any(
                (workload_name, query_name, label) in timing_by_key
                for query_name in query_names
            )
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
            for variant_label in active_variants:
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
        for variant_label in active_variants:
            phase_rows = [
                row for row in rows
                if row.workload_name == workload_name
                and row.query_name in query_names
                and row.variant_label == variant_label
            ]
            build_values = [
                row.prf_build_phase_ms for row in phase_rows
                if row.prf_build_phase_ms is not None
            ]
            analysis_values = [
                row.prf_analysis_phase_ms for row in phase_rows
                if row.prf_analysis_phase_ms is not None
            ]
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
                "prf_build_phase_ms": sum(build_values) if build_values else None,
                "prf_analysis_phase_ms": sum(analysis_values) if analysis_values else None,
                "prf_build_profile_count": len(build_values),
                "prf_analysis_profile_count": len(analysis_values),
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
    active_variants = tuple(
        label
        for label in VARIANT_ORDER
        if any(
            item["variant_label"] == label
            for entry in entries
            for item in entry["series"]
        )
    )
    relative_values = {label: [] for label in active_variants}
    absolute_totals = {label: 0.0 for label in active_variants}
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
    summary_key = {
        "runtime_ms": "absolute_runtime_totals_ms",
        "files_read_fraction": "sum_of_opened_file_fractions",
    }.get(metric, "absolute_cumulative_rows_totals")
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

    active_variants = tuple(
        label
        for label in VARIANT_ORDER
        if any(label in variant_rows for variant_rows in grouped.values())
    )
    absolute_totals = {label: 0.0 for label in active_variants}
    relative_values = {label: [] for label in active_variants}
    max_speedup = {
        label: {"query": None, "factor": None}
        for label in active_variants
        if label != "no prefilter"
    }
    max_slowdown = {
        label: {"query": None, "factor": None}
        for label in active_variants
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
        for variant_label in active_variants:
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
    variants = _relative_probe_variants(scan_rows)
    grouped = _group_relative_probe_scans(scan_rows, variants)

    relative_values = {label: [] for label in variants}
    included_count = 0
    excluded_missing_variants = 0
    for variant_rows in grouped.values():
        if any(label not in variant_rows for label in variants):
            excluded_missing_variants += 1
            continue
        cardinalities = [
            _effective_probe_cardinality(variant_rows[label])
            for label in variants
            if _effective_probe_cardinality(variant_rows[label]) > 0
        ]
        if len(cardinalities) != len(variants):
            continue
        best = min(cardinalities)
        included_count += 1
        for label in variants:
            relative_values[label].append(
                _effective_probe_cardinality(variant_rows[label]) / best
            )

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
    variants = _relative_probe_variants(scan_rows)
    grouped = _group_relative_probe_scans(scan_rows, variants)

    relative_values = {label: [] for label in variants}
    for variant_rows in grouped.values():
        if any(label not in variant_rows for label in variants):
            continue
        cardinalities = [
            _effective_probe_cardinality(variant_rows[label])
            for label in variants
            if _effective_probe_cardinality(variant_rows[label]) > 0
        ]
        if len(cardinalities) != len(variants):
            continue
        best = min(cardinalities)
        for label in variants:
            relative_values[label].append(
                _effective_probe_cardinality(variant_rows[label]) / best
            )
    return relative_values


def _group_relative_probe_scans(
    scan_rows: list[ProbeScanRow],
    variants: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, ProbeScanRow]]:
    grouped: dict[tuple[str, str, str], dict[str, ProbeScanRow]] = {}
    for row in scan_rows:
        if row.variant_label not in variants:
            continue
        logical_scan_key = (
            row.workload_name,
            row.query_name,
            _table_basename(row.table_name),
        )
        variant_rows = grouped.setdefault(logical_scan_key, {})
        existing = variant_rows.get(row.variant_label)
        if existing is None:
            variant_rows[row.variant_label] = replace(
                row,
                scan_key=f"{logical_scan_key[2]}::aggregate",
                table_name=logical_scan_key[2],
            )
            continue
        variant_rows[row.variant_label] = replace(
            existing,
            operator_rows_scanned=(
                existing.operator_rows_scanned + row.operator_rows_scanned
            ),
            operator_cardinality=(
                existing.operator_cardinality + row.operator_cardinality
            ),
            join_probe_cardinality=(
                None
                if existing.join_probe_cardinality is None
                or row.join_probe_cardinality is None
                else existing.join_probe_cardinality + row.join_probe_cardinality
            ),
            dynamic_filters=" | ".join(
                value
                for value in (existing.dynamic_filters, row.dynamic_filters)
                if value
            ),
        )
    return grouped


def _effective_probe_cardinality(row: ProbeScanRow) -> int:
    return (
        row.join_probe_cardinality
        if row.join_probe_cardinality is not None
        else row.operator_cardinality
    )


def _relative_probe_variants(scan_rows: list[ProbeScanRow]) -> tuple[str, ...]:
    present = {row.variant_label for row in scan_rows}
    return tuple(
        label
        for label in VARIANT_ORDER
        if label in present and label not in {"no prefilter", "rpt none"}
    )


def build_relative_probe_cardinality_outliers(
    scan_rows: list[ProbeScanRow],
    *,
    threshold: float = 10.0,
) -> list[dict[str, object]]:
    variants = _relative_probe_variants(scan_rows)
    grouped = _group_relative_probe_scans(scan_rows, variants)
    outliers: list[dict[str, object]] = []
    for (workload_name, query_name, scan_key), variant_rows in grouped.items():
        if any(label not in variant_rows for label in variants):
            continue
        cardinalities = {
            label: _effective_probe_cardinality(variant_rows[label])
            for label in variants
            if _effective_probe_cardinality(variant_rows[label]) > 0
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
