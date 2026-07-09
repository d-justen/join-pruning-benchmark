from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking.duckdb_build import resolve_existing_system
from benchmarking.duckdb_runner import _extract_profile_metrics
from benchmarking.experiment_config import ExperimentConfig, VariantConfig, load_experiment_config
from benchmarking.manifests import PreparedWorkload, load_manifest
from benchmarking.query_inventory import load_queries
from benchmarking.svg_charts import PALETTE, grouped_bar_chart, line_chart, svg_document, text


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path

    @property
    def comparisons_dir(self) -> Path:
        return self.root / "comparisons"

    @property
    def charts_dir(self) -> Path:
        return self.root / "charts"

    @property
    def summary_json(self) -> Path:
        return self.root / "summary.json"


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _ensure_profile_runs_table(results_db: Path) -> None:
    con = duckdb.connect(str(results_db))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_runs (
                run_id VARCHAR,
                experiment_name VARCHAR,
                workload_name VARCHAR,
                workload_kind VARCHAR,
                system_name VARCHAR,
                system_kind VARCHAR,
                system_version VARCHAR,
                system_git_commit VARCHAR,
                system_repo VARCHAR,
                system_ref VARCHAR,
                variant_label VARCHAR,
                variant_setup_sql VARCHAR,
                query_name VARCHAR,
                query_path VARCHAR,
                join_variant VARCHAR,
                cluster_count INTEGER,
                profile_path VARCHAR,
                cumulative_rows_scanned BIGINT,
                profiled_at TIMESTAMP
            );
            """
        )
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS total_rows_scanned BIGINT;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS probe_rows_scanned BIGINT;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS build_rows_scanned BIGINT;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS total_bytes_read BIGINT;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS total_bytes_written BIGINT;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS table_scan_file_count BIGINT;")
    finally:
        con.close()


def _load_workloads(experiment_root: Path) -> dict[str, PreparedWorkload]:
    workloads: dict[str, PreparedWorkload] = {}
    for manifest_path in sorted((experiment_root / "workloads").glob("*/manifest.json")):
        manifest = load_manifest(manifest_path)
        workloads[manifest.name] = manifest
    return workloads


def _variant_map(config: ExperimentConfig) -> dict[tuple[str, str], VariantConfig]:
    mapping: dict[tuple[str, str], VariantConfig] = {}
    for system in config.systems:
        for variant in system.variants:
            mapping[(system.name, variant.label)] = variant
    return mapping


def _profile_sql(
    query_text: str,
    profile_path: Path,
    variant: VariantConfig,
    threads: int | None,
    workload_setup_sql: tuple[str, ...],
) -> str:
    statements = ["SET enable_progress_bar=false;"]
    if threads is not None:
        statements.append(f"SET threads = {threads};")
    statements.extend(workload_setup_sql)
    statements.extend(variant.setup_sql)
    statements.append("SET enable_profiling='json';")
    escaped_profile_path = str(profile_path).replace("'", "''")
    statements.append(f"SET profiling_output='{escaped_profile_path}';")
    statements.append(query_text.strip().rstrip(";") + ";")
    statements.append("SET enable_profiling='no_output';")
    return "\n".join(statements) + "\n"


def backfill_profiles(experiment_root: Path, config: ExperimentConfig, *, results_db: Path) -> int:
    _ensure_profile_runs_table(results_db)
    con = duckdb.connect(str(results_db))
    try:
        missing_rows = con.execute(
            """
            SELECT DISTINCT
                t.run_id,
                t.experiment_name,
                t.workload_name,
                t.workload_kind,
                t.system_name,
                t.system_kind,
                t.system_version,
                t.system_git_commit,
                t.system_repo,
                t.system_ref,
                t.variant_label,
                t.variant_setup_sql,
                t.query_name,
                t.query_path,
                t.join_variant,
                t.cluster_count
            FROM timing_runs AS t
            LEFT JOIN profile_runs AS p
              ON p.run_id = t.run_id
             AND p.workload_name = t.workload_name
             AND p.system_name = t.system_name
             AND p.variant_label = t.variant_label
             AND p.query_name = t.query_name
            WHERE t.experiment_name = ?
              AND p.query_name IS NULL
            ORDER BY 4, 5, 11, 13
            """,
            [config.experiment_name],
        ).fetchall()
        if not missing_rows:
            return 0
        workloads = _load_workloads(experiment_root)
        variant_by_key = _variant_map(config)
        systems_by_name = {system.name: system for system in config.systems}
        inserted = 0
        for row in missing_rows:
            (
                run_id,
                experiment_name,
                workload_name,
                workload_kind,
                system_name,
                system_kind,
                system_version,
                system_git_commit,
                system_repo,
                system_ref,
                variant_label,
                variant_setup_sql,
                query_name,
                query_path,
                join_variant,
                cluster_count,
            ) = row
            workload = workloads[workload_name]
            system_config = systems_by_name[system_name]
            resolved_system = resolve_existing_system(system_config, experiment_root / "systems" / system_name)
            variant = variant_by_key[(system_name, variant_label)]
            profile_dir = experiment_root / "analysis" / "profiles" / workload_name / system_name / variant_label
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = (profile_dir / query_name).with_suffix(".json")
            temp_sql_path = profile_dir / (Path(query_name).stem + ".sql")
            query_text = Path(query_path).read_text(encoding="utf-8")
            workload_setup_sql = tuple(
                path.read_text(encoding="utf-8").strip() for path in workload.setup_sql_paths
            )
            temp_sql_path.write_text(
                _profile_sql(
                    query_text,
                    profile_path,
                    variant,
                    config.benchmark.threads,
                    workload_setup_sql,
                ),
                encoding="utf-8",
            )
            _run_command([str(resolved_system.binary_path), str(workload.db_path), "-f", str(temp_sql_path)])
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            metrics = _extract_profile_metrics(profile_data)
            con.execute(
                """
                INSERT INTO profile_runs (
                    run_id, experiment_name, workload_name, workload_kind,
                    system_name, system_kind, system_version, system_git_commit, system_repo, system_ref,
                    variant_label, variant_setup_sql, query_name, query_path, join_variant, cluster_count,
                    profile_path, cumulative_rows_scanned, total_rows_scanned, probe_rows_scanned, build_rows_scanned,
                    total_bytes_read, total_bytes_written, table_scan_file_count, profiled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """,
                [
                    run_id,
                    experiment_name,
                    workload_name,
                    workload_kind,
                    system_name,
                    system_kind,
                    system_version,
                    system_git_commit,
                    system_repo,
                    system_ref,
                    variant_label,
                    variant_setup_sql,
                    query_name,
                    query_path,
                    join_variant,
                    cluster_count,
                    str(profile_path),
                    metrics.total_rows_scanned,
                    metrics.total_rows_scanned,
                    metrics.probe_rows_scanned,
                    metrics.build_rows_scanned,
                    metrics.total_bytes_read,
                    metrics.total_bytes_written,
                    metrics.table_scan_file_count,
                ],
            )
            inserted += 1
        return inserted
    finally:
        con.close()


def _default_baselines(config: ExperimentConfig) -> dict[str, str]:
    return {system.name: system.variants[0].label for system in config.systems}


def _load_query_metadata(workload: PreparedWorkload) -> dict[str, dict]:
    if workload.kind == "single_join_clusters":
        queries = load_queries(workload.queries_dir)
        return {
            query.name: {
                "cluster_count": query.cluster_count,
                "join_variant": query.join_variant,
            }
            for query in queries
        }
    if workload.kind == "cross_product":
        metadata: dict[str, dict] = {}
        for query_path in sorted(workload.queries_dir.glob("*.sql")):
            stem = query_path.stem
            build_part, selectivity_part, predicate_type, probe_clustering = stem.split("_", 3)
            metadata[query_path.name] = {
                "build_size_label": build_part[1:],
                "selectivity_label": selectivity_part,
                "predicate_type": predicate_type,
                "probe_clustering": probe_clustering,
            }
        return metadata
    return {
        query_path.name: {"query_group": query_path.stem}
        for query_path in sorted(workload.queries_dir.glob("*.sql"))
    }


def _comparison_rows(
    results_db: Path,
    experiment_name: str,
    workload_name: str,
    system_name: str,
    baseline_variant: str,
) -> list[dict]:
    con = duckdb.connect(str(results_db))
    try:
        rows = con.execute(
            """
            WITH timing AS (
                SELECT
                    workload_name,
                    system_name,
                    variant_label,
                    query_name,
                    join_variant,
                    cluster_count,
                    median(elapsed_ns) / 1000000.0 AS median_ms,
                    avg(elapsed_ns) / 1000000.0 AS avg_ms
                FROM timing_runs
                WHERE experiment_name = ?
                  AND workload_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3, 4, 5, 6
            ),
            profiles AS (
                SELECT
                    workload_name,
                    system_name,
                    variant_label,
                    query_name,
                    join_variant,
                    cluster_count,
                    max(total_rows_scanned) AS total_rows_scanned,
                    max(probe_rows_scanned) AS probe_rows_scanned,
                    max(total_bytes_read) AS total_bytes_read
                FROM profile_runs
                WHERE experiment_name = ?
                  AND workload_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3, 4, 5, 6
            )
            SELECT
                candidate.variant_label,
                candidate.query_name,
                candidate.join_variant,
                candidate.cluster_count,
                baseline.median_ms,
                candidate.median_ms,
                baseline.avg_ms,
                candidate.avg_ms,
                coalesce(baseline_profile.probe_rows_scanned, baseline_profile.total_rows_scanned),
                coalesce(candidate_profile.probe_rows_scanned, candidate_profile.total_rows_scanned),
                baseline_profile.total_bytes_read,
                candidate_profile.total_bytes_read
            FROM timing AS candidate
            JOIN timing AS baseline
              ON baseline.query_name = candidate.query_name
             AND baseline.join_variant IS NOT DISTINCT FROM candidate.join_variant
             AND baseline.cluster_count IS NOT DISTINCT FROM candidate.cluster_count
             AND baseline.variant_label = ?
            LEFT JOIN profiles AS baseline_profile
              ON baseline_profile.query_name = candidate.query_name
             AND baseline_profile.join_variant IS NOT DISTINCT FROM candidate.join_variant
             AND baseline_profile.cluster_count IS NOT DISTINCT FROM candidate.cluster_count
             AND baseline_profile.variant_label = ?
            LEFT JOIN profiles AS candidate_profile
              ON candidate_profile.query_name = candidate.query_name
             AND candidate_profile.join_variant IS NOT DISTINCT FROM candidate.join_variant
             AND candidate_profile.cluster_count IS NOT DISTINCT FROM candidate.cluster_count
             AND candidate_profile.variant_label = candidate.variant_label
            WHERE candidate.variant_label <> ?
            ORDER BY candidate.variant_label, candidate.query_name
            """,
            [
                experiment_name,
                workload_name,
                system_name,
                experiment_name,
                workload_name,
                system_name,
                baseline_variant,
                baseline_variant,
                baseline_variant,
            ],
        ).fetchall()
        return [
            {
                "variant_label": row[0],
                "query_name": row[1],
                "join_variant": row[2],
                "cluster_count": row[3],
                "baseline_median_ms": row[4],
                "variant_median_ms": row[5],
                "baseline_avg_ms": row[6],
                "variant_avg_ms": row[7],
                "baseline_probe_or_total_rows_scanned": row[8],
                "variant_probe_or_total_rows_scanned": row[9],
                "baseline_total_bytes_read": row[10],
                "variant_total_bytes_read": row[11],
                "speedup_vs_variant": None if row[5] in (None, 0) else row[4] / row[5],
                "rows_scanned_ratio": None if row[9] in (None, 0) else row[8] / row[9] if row[8] is not None else None,
                "bytes_read_ratio": None if row[11] in (None, 0) else row[10] / row[11] if row[10] is not None else None,
            }
            for row in rows
        ]
    finally:
        con.close()


def _absolute_metric_rows(
    results_db: Path,
    experiment_name: str,
    workload_name: str,
    system_name: str,
) -> list[dict]:
    con = duckdb.connect(str(results_db))
    try:
        rows = con.execute(
            """
            WITH timing AS (
                SELECT
                    workload_name,
                    system_name,
                    variant_label,
                    query_name,
                    join_variant,
                    cluster_count,
                    median(elapsed_ns) / 1000000.0 AS median_ms
                FROM timing_runs
                WHERE experiment_name = ?
                  AND workload_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3, 4, 5, 6
            ),
            profiles AS (
                SELECT
                    workload_name,
                    system_name,
                    variant_label,
                    query_name,
                    join_variant,
                    cluster_count,
                    max(total_rows_scanned) AS total_rows_scanned,
                    max(probe_rows_scanned) AS probe_rows_scanned,
                    max(total_bytes_read) AS total_bytes_read,
                    max(total_bytes_written) AS total_bytes_written,
                    max(table_scan_file_count) AS table_scan_file_count
                FROM profile_runs
                WHERE experiment_name = ?
                  AND workload_name = ?
                  AND system_name = ?
                GROUP BY 1, 2, 3, 4, 5, 6
            )
            SELECT
                timing.variant_label,
                timing.query_name,
                timing.join_variant,
                timing.cluster_count,
                timing.median_ms,
                profiles.total_rows_scanned,
                profiles.probe_rows_scanned,
                profiles.total_bytes_read,
                profiles.total_bytes_written,
                profiles.table_scan_file_count
            FROM timing
            LEFT JOIN profiles
              ON profiles.variant_label = timing.variant_label
             AND profiles.query_name = timing.query_name
             AND profiles.join_variant IS NOT DISTINCT FROM timing.join_variant
             AND profiles.cluster_count IS NOT DISTINCT FROM timing.cluster_count
            ORDER BY timing.query_name, timing.variant_label
            """,
            [
                experiment_name,
                workload_name,
                system_name,
                experiment_name,
                workload_name,
                system_name,
            ],
        ).fetchall()
        return [
            {
                "variant_label": row[0],
                "query_name": row[1],
                "join_variant": row[2],
                "cluster_count": row[3],
                "median_ms": row[4],
                "total_rows_scanned": row[5],
                "probe_rows_scanned": row[6],
                "total_bytes_read": row[7],
                "total_bytes_written": row[8],
                "table_scan_file_count": row[9],
                "display_rows_scanned": row[6] if row[6] is not None else row[5],
            }
            for row in rows
        ]
    finally:
        con.close()


def _single_join_chart(workload_name: str, systems: list[str], rows_by_system: dict[str, list[dict]]) -> str:
    width = 1200
    per_system_height = 500
    elements = [text(24, 30, f"{workload_name}: runtime and rows scanned by cluster count", size=20, weight="700")]
    for system_index, system_name in enumerate(systems):
        top = 50 + system_index * per_system_height
        rows = rows_by_system[system_name]
        x_values = sorted({row["cluster_count"] for row in rows if row["cluster_count"] is not None})
        combos = sorted({(row["variant_label"], row["join_variant"]) for row in rows})
        series_runtime = []
        series_rows = []
        for combo_index, (variant_label, join_variant) in enumerate(combos):
            color = PALETTE[combo_index % len(PALETTE)]
            dash = "6,4" if join_variant == "uniform" else None
            points_runtime = []
            points_rows = []
            for cluster_count in x_values:
                match = next(
                    item for item in rows
                    if item["variant_label"] == variant_label
                    and item["join_variant"] == join_variant
                    and item["cluster_count"] == cluster_count
                )
                points_runtime.append((cluster_count, match["median_ms"]))
                points_rows.append((cluster_count, match["display_rows_scanned"] or 0))
            label = f"{variant_label} / {join_variant}"
            series_runtime.append({"label": label, "color": color, "dash": dash, "points": points_runtime})
            series_rows.append({"label": label, "color": color, "dash": dash, "points": points_rows})
        elements.append(text(24, top + 18, system_name, size=15, weight="600"))
        elements.extend(
            line_chart(
                20,
                top + 30,
                560,
                420,
                title="Median runtime (ms)",
                x_label="cluster count",
                y_label="ms",
                series=series_runtime,
                x_values=x_values,
                log_x=True,
            )
        )
        elements.extend(
            line_chart(
                600,
                top + 30,
                560,
                420,
                title="Probe rows scanned",
                x_label="cluster count",
                y_label="rows",
                series=series_rows,
                x_values=x_values,
                log_x=True,
            )
        )
    return svg_document(width, 70 + per_system_height * len(systems), elements)


def _bar_chart_svg(workload_name: str, title_suffix: str, systems: list[str], rows_by_system: dict[str, list[dict]]) -> str:
    width = 1400
    system_blocks = []
    total_height = 60
    for system_name in systems:
        rows = rows_by_system[system_name]
        categories = sorted({row["query_name"] for row in rows})
        variant_count = max(1, len({row["variant_label"] for row in rows}))
        per_query_height = variant_count * 14 + 10
        chart_height = max(320, 60 + per_query_height * len(categories))
        totals_height = 26
        block_height = chart_height + totals_height + 70
        system_blocks.append((system_name, categories, block_height, chart_height))
        total_height += block_height + 20
    elements = [text(24, 30, f"{workload_name}: {title_suffix}", size=20, weight="700")]
    top = 50
    for system_name, categories, block_height, chart_height in system_blocks:
        rows = rows_by_system[system_name]
        variants = sorted({row["variant_label"] for row in rows})
        runtime_series = []
        rows_series = []
        for variant_index, variant_label in enumerate(variants):
            color = PALETTE[variant_index % len(PALETTE)]
            runtime_values = []
            row_scan_values = []
            for category in categories:
                match = next(item for item in rows if item["variant_label"] == variant_label and item["query_name"] == category)
                runtime_values.append(match["median_ms"])
                row_scan_values.append(match["display_rows_scanned"])
            runtime_series.append({"label": variant_label, "color": color, "values": runtime_values})
            rows_series.append({"label": variant_label, "color": color, "values": row_scan_values})
        runtime_totals = sorted(
            (
                (series["label"], sum(value for value in series["values"] if value is not None))
                for series in runtime_series
            ),
            key=lambda item: item[1],
        )
        elements.append(text(24, top + 18, system_name, size=15, weight="600"))
        elements.extend(
            grouped_bar_chart(
                20,
                top + 30,
                660,
                chart_height,
                title="Median runtime (ms)",
                categories=categories,
                series=runtime_series,
                x_label="ms",
            )
        )
        elements.extend(
            grouped_bar_chart(
                710,
                top + 30,
                660,
                chart_height,
                title="Probe rows scanned",
                categories=categories,
                series=rows_series,
                x_label="rows",
            )
        )
        totals_text = "Total runtime (ascending): " + " | ".join(
            f"{label}: {value:.2f} ms" for label, value in runtime_totals
        )
        elements.append(text(24, top + 30 + chart_height + 28, totals_text, size=10))
        top += block_height + 20
    return svg_document(width, total_height, elements)


def write_workload_chart(
    output_path: Path,
    workload: PreparedWorkload,
    systems: list[str],
    rows_by_system: dict[str, list[dict]],
) -> None:
    if workload.kind == "single_join_clusters":
        svg = _single_join_chart(workload.name, systems, rows_by_system)
    else:
        svg = _bar_chart_svg(workload.name, "variant comparison", systems, rows_by_system)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def analyze_experiment(experiment_root: Path, baseline_variant: str | None) -> AnalysisPaths:
    experiment_root = experiment_root.resolve()
    config = load_experiment_config(experiment_root / "config.json")
    results_db = experiment_root / "results.duckdb"
    analysis_paths = AnalysisPaths(root=experiment_root / "analysis")
    analysis_paths.root.mkdir(parents=True, exist_ok=True)
    analysis_paths.comparisons_dir.mkdir(parents=True, exist_ok=True)
    analysis_paths.charts_dir.mkdir(parents=True, exist_ok=True)
    backfill_count = backfill_profiles(experiment_root, config, results_db=results_db)
    workloads = _load_workloads(experiment_root)
    baselines = _default_baselines(config)
    if baseline_variant is not None:
        baselines = {system_name: baseline_variant for system_name in baselines}
    summary: dict[str, object] = {
        "experiment_name": config.experiment_name,
        "backfilled_profiles": backfill_count,
        "workloads": [],
    }
    for workload_name, workload in sorted(workloads.items()):
        workload_entry = {"workload_name": workload_name, "workload_kind": workload.kind, "systems": []}
        rows_for_chart: dict[str, list[dict]] = {}
        system_names = [system.name for system in config.systems]
        query_metadata = _load_query_metadata(workload)
        for system_name in system_names:
            absolute_rows = _absolute_metric_rows(results_db, config.experiment_name, workload_name, system_name)
            for row in absolute_rows:
                row.update(query_metadata.get(row["query_name"], {}))
            rows_for_chart[system_name] = absolute_rows
            comparison_rows = _comparison_rows(
                results_db,
                config.experiment_name,
                workload_name,
                system_name,
                baselines[system_name],
            )
            for row in comparison_rows:
                row.update(query_metadata.get(row["query_name"], {}))
            comparison_path = analysis_paths.comparisons_dir / workload_name / f"{system_name}.json"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_path.write_text(json.dumps(comparison_rows, indent=2) + "\n", encoding="utf-8")
            workload_entry["systems"].append(
                {
                    "system_name": system_name,
                    "baseline_variant": baselines[system_name],
                    "comparison_path": str(comparison_path),
                    "comparison_count": len(comparison_rows),
                    "faster_queries": sum(1 for row in comparison_rows if (row["speedup_vs_variant"] or 0) > 1),
                    "slower_queries": sum(1 for row in comparison_rows if (row["speedup_vs_variant"] or 0) < 1),
                    "fewer_rows_scanned": sum(1 for row in comparison_rows if (row["rows_scanned_ratio"] or 0) > 1),
                    "more_rows_scanned": sum(1 for row in comparison_rows if (row["rows_scanned_ratio"] or 0) < 1),
                    "fewer_bytes_read": sum(1 for row in comparison_rows if (row["bytes_read_ratio"] or 0) > 1),
                    "more_bytes_read": sum(1 for row in comparison_rows if (row["bytes_read_ratio"] or 0) < 1),
                }
            )
        chart_path = analysis_paths.charts_dir / f"{workload_name}.svg"
        write_workload_chart(chart_path, workload, system_names, rows_for_chart)
        workload_entry["chart_path"] = str(chart_path)
        summary["workloads"].append(workload_entry)
    analysis_paths.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return analysis_paths
