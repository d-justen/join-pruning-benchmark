from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.experiment_config import BenchmarkOptions, VariantConfig
from benchmarking.manifests import PreparedWorkload
from benchmarking.query_inventory import QuerySpec, load_queries


LABEL_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def label_to_path_component(label: str) -> str:
    normalized = LABEL_COMPONENT_RE.sub("_", label.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"variant label cannot be normalized into a path component: {label!r}")
    return normalized


def run_command(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cwd: {cwd or Path.cwd()}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def ensure_results_schema(results_db: Path) -> None:
    con = duckdb.connect(str(results_db))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS timing_runs (
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
                repetition INTEGER,
                elapsed_ns BIGINT,
                measured_at TIMESTAMP
            );
            """
        )
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
    finally:
        con.close()


@dataclass(frozen=True)
class ProfileMetrics:
    total_rows_scanned: int | None
    probe_rows_scanned: int | None
    build_rows_scanned: int | None


def _extract_profile_metrics(profile_data: dict) -> ProfileMetrics:
    total_rows_scanned = profile_data.get("cumulative_rows_scanned")
    probe_rows_scanned: int | None = None
    build_rows_scanned = 0

    stack = list(profile_data.get("children", []))
    while stack:
        node = stack.pop()
        stack.extend(node.get("children", []))
        if node.get("operator_type") != "TABLE_SCAN":
            continue
        extra_info = node.get("extra_info", {})
        table_name = str(extra_info.get("Table", ""))
        rows_scanned = node.get("operator_rows_scanned")
        if not isinstance(rows_scanned, int):
            continue
        if table_name.endswith(".fact"):
            probe_rows_scanned = rows_scanned
        else:
            build_rows_scanned += rows_scanned

    if build_rows_scanned == 0:
        build_rows_scanned = None
    return ProfileMetrics(
        total_rows_scanned=total_rows_scanned,
        probe_rows_scanned=probe_rows_scanned,
        build_rows_scanned=build_rows_scanned,
    )


def render_query_execution_block(
    experiment_name: str,
    workload: PreparedWorkload,
    system: ResolvedSystem,
    query: QuerySpec,
    *,
    run_id: str,
    variant: VariantConfig,
    repetition_count: int,
) -> str:
    query_sql = query.path.read_text(encoding="utf-8").strip().rstrip(";")
    join_variant_sql = sql_literal(query.join_variant) if query.join_variant is not None else "NULL"
    cluster_count_sql = str(query.cluster_count) if query.cluster_count is not None else "NULL"
    variant_setup_sql = sql_literal("\n".join(variant.setup_sql))
    statements: list[str] = [query_sql + ";"]
    for repetition in range(1, repetition_count + 1):
        statements.extend(
            [
                "SET VARIABLE benchmark_t0_ns = epoch_ns(current_timestamp);",
                query_sql + ";",
                "INSERT INTO benchmark_results.timing_runs VALUES (",
                f"    {sql_literal(run_id)},",
                f"    {sql_literal(experiment_name)},",
                f"    {sql_literal(workload.name)},",
                f"    {sql_literal(workload.kind)},",
                f"    {sql_literal(system.name)},",
                f"    {sql_literal(system.kind)},",
                f"    {sql_literal(system.version)},",
                "    NULL," if system.git_commit is None else f"    {sql_literal(system.git_commit)},",
                "    NULL," if system.repo is None else f"    {sql_literal(system.repo)},",
                "    NULL," if system.ref is None else f"    {sql_literal(system.ref)},",
                f"    {sql_literal(variant.label)},",
                f"    {variant_setup_sql},",
                f"    {sql_literal(query.name)},",
                f"    {sql_literal(str(query.path))},",
                f"    {join_variant_sql},",
                f"    {cluster_count_sql},",
                f"    {repetition},",
                "    epoch_ns(current_timestamp) - getvariable('benchmark_t0_ns'),",
                "    CAST(current_timestamp AS TIMESTAMP)",
                ");",
            ]
        )
    return "\n".join(statements)


def render_profile_execution_block(
    experiment_name: str,
    workload: PreparedWorkload,
    system: ResolvedSystem,
    query: QuerySpec,
    *,
    run_id: str,
    variant: VariantConfig,
    profile_path: Path,
) -> str:
    query_sql = query.path.read_text(encoding="utf-8").strip().rstrip(";")
    join_variant_sql = sql_literal(query.join_variant) if query.join_variant is not None else "NULL"
    cluster_count_sql = str(query.cluster_count) if query.cluster_count is not None else "NULL"
    variant_setup_sql = sql_literal("\n".join(variant.setup_sql))
    statements = [
        "SET enable_profiling = 'json';",
        f"SET profiling_output = {sql_literal(str(profile_path))};",
        query_sql + ";",
        "SET enable_profiling = 'no_output';",
        "INSERT INTO benchmark_results.profile_runs (",
        "    run_id, experiment_name, workload_name, workload_kind,",
        "    system_name, system_kind, system_version, system_git_commit, system_repo, system_ref,",
        "    variant_label, variant_setup_sql, query_name, query_path, join_variant, cluster_count,",
        "    profile_path, cumulative_rows_scanned, total_rows_scanned, probe_rows_scanned, build_rows_scanned, profiled_at",
        ") VALUES (",
        f"    {sql_literal(run_id)},",
        f"    {sql_literal(experiment_name)},",
        f"    {sql_literal(workload.name)},",
        f"    {sql_literal(workload.kind)},",
        f"    {sql_literal(system.name)},",
        f"    {sql_literal(system.kind)},",
        f"    {sql_literal(system.version)},",
        "    NULL," if system.git_commit is None else f"    {sql_literal(system.git_commit)},",
        "    NULL," if system.repo is None else f"    {sql_literal(system.repo)},",
        "    NULL," if system.ref is None else f"    {sql_literal(system.ref)},",
        f"    {sql_literal(variant.label)},",
        f"    {variant_setup_sql},",
        f"    {sql_literal(query.name)},",
        f"    {sql_literal(str(query.path))},",
        f"    {join_variant_sql},",
        f"    {cluster_count_sql},",
        f"    {sql_literal(str(profile_path))},",
        "    NULL,",
        "    NULL,",
        "    NULL,",
        "    NULL,",
        "    CAST(current_timestamp AS TIMESTAMP)",
        ");",
    ]
    return "\n".join(statements)


def render_workload_sql(
    experiment_name: str,
    workload: PreparedWorkload,
    system: ResolvedSystem,
    variant: VariantConfig,
    queries: list[QuerySpec],
    results_db: Path,
    benchmark: BenchmarkOptions,
    run_id: str,
    profile_dir: Path,
) -> str:
    statements = [
        "-- Generated by run_benchmarks.py",
        f"ATTACH {sql_literal(str(results_db))} AS benchmark_results;",
        "SET enable_progress_bar=false;",
    ]
    if benchmark.threads is not None:
        statements.append(f"SET threads = {benchmark.threads};")
    statements.extend(variant.setup_sql)
    for query in queries:
        statements.append(
            render_query_execution_block(
                experiment_name,
                workload,
                system,
                query,
                run_id=run_id,
                variant=variant,
                repetition_count=benchmark.repetitions,
            )
        )
        if benchmark.collect_profiles:
            profile_path = (profile_dir / label_to_path_component(variant.label) / query.relative_path).with_suffix(".json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            statements.append(
                render_profile_execution_block(
                    experiment_name,
                    workload,
                    system,
                    query,
                    run_id=run_id,
                    variant=variant,
                    profile_path=profile_path,
                )
            )
    return "\n\n".join(statements) + "\n"


def summarize_results(results_db: Path, experiment_name: str) -> dict:
    con = duckdb.connect(str(results_db))
    try:
        workload_rows = con.execute(
            """
            SELECT
                workload_name,
                system_name,
                variant_label,
                COUNT(*) AS timing_rows,
                median(elapsed_ns) / 1000000.0 AS median_ms
            FROM timing_runs
            WHERE experiment_name = ?
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """,
            [experiment_name],
        ).fetchall()
        summary = {
            "experiment_name": experiment_name,
            "variants": [
                {
                    "workload_name": row[0],
                    "system_name": row[1],
                    "variant_label": row[2],
                    "timing_rows": row[3],
                    "median_ms": row[4],
                }
                for row in workload_rows
            ],
        }
        return summary
    finally:
        con.close()


@dataclass(frozen=True)
class RunOutput:
    run_id: str
    query_count: int


def execute_workload(binary: Path, db_path: Path, sql_path: Path) -> None:
    run_command([str(binary), str(db_path), "-f", str(sql_path)])


def populate_profile_metrics(results_db: Path, run_id: str) -> None:
    con = duckdb.connect(str(results_db))
    try:
        rows = con.execute(
            """
            SELECT variant_label, query_name, profile_path
            FROM profile_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchall()
        for variant_label, query_name, profile_path in rows:
            profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            metrics = _extract_profile_metrics(profile_data)
            con.execute(
                """
                UPDATE profile_runs
                SET cumulative_rows_scanned = ?,
                    total_rows_scanned = ?,
                    probe_rows_scanned = ?,
                    build_rows_scanned = ?
                WHERE run_id = ? AND variant_label = ? AND query_name = ? AND profile_path = ?
                """,
                [
                    metrics.total_rows_scanned,
                    metrics.total_rows_scanned,
                    metrics.probe_rows_scanned,
                    metrics.build_rows_scanned,
                    run_id,
                    variant_label,
                    query_name,
                    profile_path,
                ],
            )
    finally:
        con.close()


def benchmark_workload(
    experiment_name: str,
    workload: PreparedWorkload,
    system: ResolvedSystem,
    variants: tuple[VariantConfig, ...],
    benchmark: BenchmarkOptions,
    results_db: Path,
    sql_dir: Path,
    profiles_dir: Path,
) -> RunOutput:
    ensure_results_schema(results_db)
    queries = load_queries(workload.queries_dir)
    run_id = uuid.uuid4().hex
    print(
        f"Benchmarking workload={workload.name} system={system.name} queries={len(queries)}",
        file=sys.stderr,
    )
    for variant in variants:
        variant_dir = sql_dir / workload.name / system.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        sql_path = variant_dir / f"{label_to_path_component(variant.label)}.sql"
        workload_profile_dir = profiles_dir / workload.name / system.name
        sql_path.write_text(
            render_workload_sql(
                experiment_name,
                workload,
                system,
                variant,
                queries,
                results_db.resolve(),
                benchmark,
                run_id,
                workload_profile_dir,
            ),
            encoding="utf-8",
        )
        execute_workload(system.binary_path, workload.db_path, sql_path)
    if benchmark.collect_profiles:
        populate_profile_metrics(results_db, run_id)
    return RunOutput(run_id=run_id, query_count=len(queries))


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
