from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb


QUERY_FILENAME_RE = re.compile(r"^(ordered|uniform)_cc_(\d+)\.sql$")


@dataclass(frozen=True)
class QuerySpec:
    name: str
    path: Path
    join_variant: str
    cluster_count: int


@dataclass(frozen=True)
class EngineSpec:
    label: str
    binary_path: Path
    version: str
    git_commit: str | None


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the system DuckDB CLI with a DuckDB build from the latest "
            "commit on main using the benchmark queries in a directory."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("out/join-prune.duckdb"),
        help="benchmark database file to open with each DuckDB binary",
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=Path("out/queries"),
        help="directory containing benchmark query .sql files",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=Path("out/duckdb-compare-results.duckdb"),
        help="DuckDB file that receives benchmark timing rows",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=Path("out/compare-sql"),
        help="directory that receives generated workload SQL files",
    )
    parser.add_argument(
        "--duckdb-clone-dir",
        type=Path,
        default=Path("out/duckdb-main"),
        help="location for the DuckDB git checkout used for the main build",
    )
    parser.add_argument(
        "--duckdb-repo",
        default="https://github.com/duckdb/duckdb.git",
        help="DuckDB git repository URL",
    )
    parser.add_argument(
        "--system-duckdb",
        type=Path,
        default=Path("duckdb"),
        help="path to the pre-installed DuckDB CLI to compare against",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="number of measured repetitions per query after one warmup run",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="set DuckDB PRAGMA threads to this value; omitted means leave unset",
    )
    parser.add_argument(
        "--main-branch",
        default="main",
        help="branch to fetch and build for the comparison build",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse an existing DuckDB checkout/build instead of fetching and running make",
    )
    return parser


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
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


def find_queries(queries_dir: Path) -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    for path in sorted(queries_dir.glob("*.sql")):
        match = QUERY_FILENAME_RE.match(path.name)
        if not match:
            continue
        join_variant = match.group(1)
        cluster_count = int(match.group(2))
        queries.append(
            QuerySpec(
                name=path.stem,
                path=path.resolve(),
                join_variant=join_variant,
                cluster_count=cluster_count,
            )
        )
    if not queries:
        raise FileNotFoundError(f"no benchmark queries found in {queries_dir}")
    return queries


def read_query_text(path: Path) -> str:
    query_sql = path.read_text(encoding="utf-8").strip()
    if not query_sql:
        raise ValueError(f"query file is empty: {path}")
    return query_sql.rstrip().rstrip(";")


def render_query_execution_block(
    query: QuerySpec,
    *,
    run_id: str,
    engine: EngineSpec,
    repetition_count: int,
    profile_dir: Path,
) -> str:
    query_sql = read_query_text(query.path)
    profile_path = (
        profile_dir / engine.label / f"{query.name}.json"
    ).resolve()
    rendered_statements: list[str] = [
        f"-- Warmup for {query.name}",
        query_sql,
        ";",
    ]

    for repetition in range(1, repetition_count + 1):
        rendered_statements.extend(
            [
                "",
                f"-- Measured repetition {repetition} for {query.name}",
                "SET VARIABLE benchmark_t0_ms = epoch_ms(current_timestamp);",
                query_sql,
                ";",
                "INSERT INTO benchmark_results.timing_runs VALUES (",
                f"    {sql_literal(run_id)},",
                f"    {sql_literal(engine.label)},",
                f"    {sql_literal(str(engine.binary_path))},",
                f"    {sql_literal(engine.version)},",
                (
                    f"    {sql_literal(engine.git_commit)},"
                    if engine.git_commit is not None
                    else "    NULL,"
                ),
                f"    {sql_literal(query.name)},",
                f"    {sql_literal(str(query.path))},",
                f"    {sql_literal(query.join_variant)},",
                f"    {query.cluster_count},",
                f"    {repetition},",
                "    epoch_ms(current_timestamp) - getvariable('benchmark_t0_ms'),",
                "    current_timestamp",
                ");",
            ]
        )

    rendered_statements.extend(
        [
            "",
            f"-- Profile run for {query.name}",
            "SET enable_profiling = 'json';",
            f"SET profiling_output = {sql_literal(str(profile_path))};",
            query_sql,
            ";",
            "SET enable_profiling = 'no_output';",
            "INSERT INTO benchmark_results.profile_runs VALUES (",
            f"    {sql_literal(run_id)},",
            f"    {sql_literal(engine.label)},",
            f"    {sql_literal(str(engine.binary_path))},",
            f"    {sql_literal(engine.version)},",
            (
                f"    {sql_literal(engine.git_commit)},"
                if engine.git_commit is not None
                else "    NULL,"
            ),
            f"    {sql_literal(query.name)},",
            f"    {sql_literal(str(query.path))},",
            f"    {sql_literal(query.join_variant)},",
            f"    {query.cluster_count},",
            f"    {sql_literal(str(profile_path))},",
            "    NULL,",
            "    current_timestamp",
            ");",
        ]
    )

    return "\n".join(rendered_statements)


def render_workload_sql(
    *,
    run_id: str,
    engine: EngineSpec,
    results_db: Path,
    queries: list[QuerySpec],
    repetitions: int,
    threads: int | None,
    profile_dir: Path,
) -> str:
    statements = [
        "-- Generated by compare_duckdb.py",
        f"ATTACH {sql_literal(str(results_db))} AS benchmark_results;",
        """
CREATE TABLE IF NOT EXISTS benchmark_results.timing_runs (
    run_id VARCHAR,
    engine VARCHAR,
    binary_path VARCHAR,
    engine_version VARCHAR,
    git_commit VARCHAR,
    query_name VARCHAR,
    query_path VARCHAR,
    join_variant VARCHAR,
    cluster_count INTEGER,
    repetition INTEGER,
    elapsed_ms BIGINT,
    measured_at TIMESTAMP WITH TIME ZONE
);
""".strip(),
        """
CREATE TABLE IF NOT EXISTS benchmark_results.profile_runs (
    run_id VARCHAR,
    engine VARCHAR,
    binary_path VARCHAR,
    engine_version VARCHAR,
    git_commit VARCHAR,
    query_name VARCHAR,
    query_path VARCHAR,
    join_variant VARCHAR,
    cluster_count INTEGER,
    profile_path VARCHAR,
    cumulative_rows_scanned BIGINT,
    profiled_at TIMESTAMP WITH TIME ZONE
);
""".strip(),
    ]

    if threads is not None:
        statements.append(f"SET threads = {threads};")

    for query in queries:
        statements.append("")
        statements.append(
            render_query_execution_block(
                query,
                run_id=run_id,
                engine=engine,
                repetition_count=repetitions,
                profile_dir=profile_dir,
            )
        )

    return "\n".join(statements) + "\n"


def resolve_binary(binary: Path) -> Path:
    if binary.is_absolute() or binary.parent != Path("."):
        resolved = binary.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"DuckDB binary not found: {binary}")
        return resolved

    candidate = shutil.which(str(binary))
    if candidate is None:
        raise FileNotFoundError(f"cannot find DuckDB binary on PATH: {binary}")
    return Path(candidate).resolve()


def get_binary_version(binary: Path) -> str:
    return run_command([str(binary), "--version"])


def ensure_duckdb_checkout(clone_dir: Path, repo: str, branch: str) -> None:
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            ["git", "clone", "--branch", branch, "--depth", "1", repo, str(clone_dir)]
        )
        return

    git_dir = clone_dir / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"existing clone path is not a git checkout: {clone_dir}")

    run_command(["git", "fetch", "origin", branch, "--depth", "1"], cwd=clone_dir)
    run_command(["git", "switch", "--detach", "FETCH_HEAD"], cwd=clone_dir)


def build_duckdb(clone_dir: Path) -> None:
    build_env = dict(os.environ)
    build_env["GEN"] = "ninja"
    run_command(["make"], cwd=clone_dir, env=build_env)


def resolve_built_duckdb_binary(clone_dir: Path) -> Path:
    candidates = [
        clone_dir / "build" / "release" / "duckdb",
        clone_dir / "build" / "debug" / "duckdb",
        clone_dir / "build" / "duckdb",
        clone_dir / "duckdb",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate compiled DuckDB CLI after build; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def get_git_commit(clone_dir: Path) -> str:
    return run_command(["git", "rev-parse", "HEAD"], cwd=clone_dir)


def write_workload_sql(path: Path, sql_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql_text, encoding="utf-8")


def execute_workload(binary: Path, db_path: Path, sql_path: Path) -> None:
    run_command([str(binary), str(db_path), "-f", str(sql_path)])


def populate_profile_metrics(results_db: Path, run_id: str) -> None:
    con = duckdb.connect(str(results_db))
    try:
        rows = con.execute(
            """
            SELECT engine, query_name, profile_path
            FROM profile_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchall()
        for engine, query_name, profile_path in rows:
            with open(profile_path, encoding="utf-8") as profile_file:
                profile_data = json.load(profile_file)
            cumulative_rows_scanned = profile_data.get("cumulative_rows_scanned")
            con.execute(
                """
                UPDATE profile_runs
                SET cumulative_rows_scanned = ?
                WHERE run_id = ? AND engine = ? AND query_name = ? AND profile_path = ?
                """,
                [
                    cumulative_rows_scanned,
                    run_id,
                    engine,
                    query_name,
                    profile_path,
                ],
            )
    finally:
        con.close()


def summarize_results(
    results_db: Path, run_id: str
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    con = duckdb.connect(str(results_db))
    try:
        timing_rows = con.execute(
            """
            WITH per_query AS (
                SELECT
                    engine,
                    query_name,
                    join_variant,
                    cluster_count,
                    median(elapsed_ms) AS median_ms,
                    avg(elapsed_ms) AS avg_ms
                FROM timing_runs
                WHERE run_id = ?
                GROUP BY 1, 2, 3, 4
            )
            SELECT
                query_name,
                join_variant,
                cluster_count,
                MAX(CASE WHEN engine = 'system' THEN median_ms END) AS system_median_ms,
                MAX(CASE WHEN engine = 'main' THEN median_ms END) AS main_median_ms,
                MAX(CASE WHEN engine = 'system' THEN avg_ms END) AS system_avg_ms,
                MAX(CASE WHEN engine = 'main' THEN avg_ms END) AS main_avg_ms,
                MAX(CASE WHEN engine = 'system' THEN median_ms END)
                    / NULLIF(MAX(CASE WHEN engine = 'main' THEN median_ms END), 0) AS speedup_vs_main
            FROM per_query
            GROUP BY 1, 2, 3
            ORDER BY cluster_count, join_variant
            """,
            [run_id],
        ).fetchall()

        engine_rows = con.execute(
            """
            SELECT
                engine,
                engine_version,
                git_commit,
                COUNT(*) AS timing_rows,
                median(elapsed_ms) AS overall_median_ms,
                avg(elapsed_ms) AS overall_avg_ms
            FROM timing_runs
            WHERE run_id = ?
            GROUP BY 1, 2, 3
            ORDER BY engine
            """,
            [run_id],
        ).fetchall()

        profile_rows = con.execute(
            """
            SELECT
                query_name,
                join_variant,
                cluster_count,
                MAX(CASE WHEN engine = 'system' THEN cumulative_rows_scanned END) AS system_cumulative_rows_scanned,
                MAX(CASE WHEN engine = 'main' THEN cumulative_rows_scanned END) AS main_cumulative_rows_scanned,
                MAX(CASE WHEN engine = 'system' THEN cumulative_rows_scanned END)::DOUBLE
                    / NULLIF(MAX(CASE WHEN engine = 'main' THEN cumulative_rows_scanned END), 0) AS rows_scanned_ratio
            FROM profile_runs
            WHERE run_id = ?
            GROUP BY 1, 2, 3
            ORDER BY cluster_count, join_variant
            """,
            [run_id],
        ).fetchall()
    finally:
        con.close()
    return engine_rows, timing_rows, profile_rows


def format_table(headers: list[str], rows: list[tuple]) -> str:
    rendered_rows = [
        [("" if value is None else str(value)) for value in row] for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in rendered_rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def load_metadata(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be positive when set")
    if not args.db.exists():
        raise FileNotFoundError(f"benchmark database does not exist: {args.db}")
    if not args.queries_dir.exists():
        raise FileNotFoundError(f"queries directory does not exist: {args.queries_dir}")

    db_path = args.db.resolve()
    results_db = args.results_db.resolve()
    sql_dir = args.sql_dir.resolve()
    clone_dir = args.duckdb_clone_dir.resolve()
    queries = find_queries(args.queries_dir.resolve())

    metadata = load_metadata(args.queries_dir / "metadata.json")
    if metadata is not None:
        print(
            f"Loaded benchmark metadata from {args.queries_dir / 'metadata.json'} "
            f"with dim_rows={metadata.get('dim_rows')} fact_rows={metadata.get('fact_rows')}",
            file=sys.stderr,
        )

    system_binary = resolve_binary(args.system_duckdb)
    system_engine = EngineSpec(
        label="system",
        binary_path=system_binary,
        version=get_binary_version(system_binary),
        git_commit=None,
    )

    if not args.skip_build:
        print(f"Fetching DuckDB {args.main_branch} into {clone_dir}", file=sys.stderr)
        ensure_duckdb_checkout(clone_dir, args.duckdb_repo, args.main_branch)
        print(f"Building DuckDB with make in {clone_dir}", file=sys.stderr)
        build_duckdb(clone_dir)

    main_binary = resolve_built_duckdb_binary(clone_dir)
    main_engine = EngineSpec(
        label="main",
        binary_path=main_binary,
        version=get_binary_version(main_binary),
        git_commit=get_git_commit(clone_dir),
    )

    run_id = uuid.uuid4().hex
    print(f"Benchmark run_id={run_id}", file=sys.stderr)
    profile_dir = (sql_dir.parent / "profiles" / run_id).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    for engine_label in ("system", "main"):
        (profile_dir / engine_label).mkdir(parents=True, exist_ok=True)

    workloads = {
        "system": sql_dir / "system_benchmark.sql",
        "main": sql_dir / "main_benchmark.sql",
    }
    write_workload_sql(
        workloads["system"],
        render_workload_sql(
            run_id=run_id,
            engine=system_engine,
            results_db=results_db,
            queries=queries,
            repetitions=args.repetitions,
            threads=args.threads,
            profile_dir=profile_dir,
        ),
    )
    write_workload_sql(
        workloads["main"],
        render_workload_sql(
            run_id=run_id,
            engine=main_engine,
            results_db=results_db,
            queries=queries,
            repetitions=args.repetitions,
            threads=args.threads,
            profile_dir=profile_dir,
        ),
    )

    print(f"Executing system workload with {system_binary}", file=sys.stderr)
    execute_workload(system_binary, db_path, workloads["system"])
    print(f"Executing main workload with {main_binary}", file=sys.stderr)
    execute_workload(main_binary, db_path, workloads["main"])
    populate_profile_metrics(results_db, run_id)

    engine_rows, timing_rows, profile_rows = summarize_results(results_db, run_id)

    print()
    print("Engines")
    print(
        format_table(
            [
                "engine",
                "engine_version",
                "git_commit",
                "timing_rows",
                "overall_median_ms",
                "overall_avg_ms",
            ],
            engine_rows,
        )
    )
    print()
    print("Per-query Timing Summary")
    print(
        format_table(
            [
                "query_name",
                "join_variant",
                "cluster_count",
                "system_median_ms",
                "main_median_ms",
                "system_avg_ms",
                "main_avg_ms",
                "speedup_vs_main",
            ],
            timing_rows,
        )
    )
    print()
    print("Per-query Profile Summary")
    print(
        format_table(
            [
                "query_name",
                "join_variant",
                "cluster_count",
                "system_cumulative_rows_scanned",
                "main_cumulative_rows_scanned",
                "rows_scanned_ratio",
            ],
            profile_rows,
        )
    )
    print()
    print(f"Results DB: {results_db}")
    print(f"System SQL: {workloads['system']}")
    print(f"Main SQL: {workloads['main']}")


if __name__ == "__main__":
    main()
