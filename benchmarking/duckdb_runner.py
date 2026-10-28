from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

import duckdb

from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.experiment_config import BenchmarkOptions, VariantConfig, applies_to_workload
from benchmarking.filter_profiles import (
    ensure_filter_profile_columns, extract_join_filter_metrics, store_join_filter_metrics,
)
from benchmarking.manifests import (
    DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER,
    DUCKLAKE_S3_BASE_URL_PLACEHOLDER,
    PreparedWorkload,
)
from benchmarking.query_inventory import QuerySpec, load_queries
LABEL_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def label_to_path_component(label: str) -> str:
    normalized = LABEL_COMPONENT_RE.sub("_", label.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"variant label cannot be normalized into a path component: {label!r}")
    return normalized


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


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
                query_parameters VARCHAR,
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
                query_parameters VARCHAR,
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
        for name in ("remote_bytes_read", "fact_remote_bytes_read", "remote_get_requests"):
            con.execute(f"ALTER TABLE timing_runs ADD COLUMN IF NOT EXISTS {name} BIGINT;")
        con.execute("ALTER TABLE timing_runs ADD COLUMN IF NOT EXISTS remote_read_details VARCHAR;")
        con.execute("ALTER TABLE timing_runs ADD COLUMN IF NOT EXISTS query_parameters VARCHAR;")
        con.execute("ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS query_parameters VARCHAR;")
        ensure_filter_profile_columns(con)
    finally:
        con.close()


@dataclass(frozen=True)
class ProfileMetrics:
    total_rows_scanned: int | None
    probe_rows_scanned: int | None
    build_rows_scanned: int | None
    total_bytes_read: int | None
    total_bytes_written: int | None
    table_scan_file_count: int | None


def _extract_profile_metrics(profile_data: dict) -> ProfileMetrics:
    total_rows_scanned = profile_data.get("cumulative_rows_scanned")
    probe_rows_scanned: int | None = None
    build_rows_scanned = 0
    total_bytes_read = profile_data.get("total_bytes_read")
    total_bytes_written = profile_data.get("total_bytes_written")
    table_scan_file_count = 0

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
        files_read = extra_info.get("Total Files Read")
        if isinstance(files_read, str) and files_read.isdigit():
            table_scan_file_count += int(files_read)
        elif isinstance(files_read, int):
            table_scan_file_count += files_read
        if table_name.endswith(".fact"):
            probe_rows_scanned = rows_scanned
        else:
            build_rows_scanned += rows_scanned

    if build_rows_scanned == 0:
        build_rows_scanned = None
    if table_scan_file_count == 0:
        table_scan_file_count = None
    return ProfileMetrics(
        total_rows_scanned=total_rows_scanned,
        probe_rows_scanned=probe_rows_scanned,
        build_rows_scanned=build_rows_scanned,
        total_bytes_read=total_bytes_read if isinstance(total_bytes_read, int) else None,
        total_bytes_written=total_bytes_written if isinstance(total_bytes_written, int) else None,
        table_scan_file_count=table_scan_file_count,
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
    query_parameters_sql = sql_literal(json.dumps(query.parameters, sort_keys=True))
    variant_setup_sql = sql_literal("\n".join(variant.setup_sql))
    statements: list[str] = [query_sql + ";"]
    measure_bytes = workload.ducklake_data_source == "s3"
    if measure_bytes and not workload.ducklake_s3_url:
        raise ValueError("S3 workload has not been uploaded")
    for repetition in range(1, repetition_count + 1):
        if measure_bytes:
            statements.append("SELECT httpfs_reset_transfer_stats();")
        statements.extend(
            [
                "SET VARIABLE benchmark_t0_ns = epoch_ns(current_timestamp);",
                query_sql + ";",
                *(["SET VARIABLE benchmark_elapsed_ns = epoch_ns(current_timestamp) - getvariable('benchmark_t0_ns');",
                   remote_bytes_snapshot_sql(workload)] if measure_bytes else []),
                "INSERT INTO benchmark_results.timing_runs (",
                "    run_id, experiment_name, workload_name, workload_kind,",
                "    system_name, system_kind, system_version, system_git_commit, system_repo, system_ref,",
                "    variant_label, variant_setup_sql, query_name, query_path, join_variant, cluster_count,",
                "    query_parameters, repetition, elapsed_ns, measured_at,",
                "    remote_bytes_read, fact_remote_bytes_read, remote_get_requests, remote_read_details",
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
                f"    {query_parameters_sql},",
                f"    {repetition},",
                ("    getvariable('benchmark_elapsed_ns')," if measure_bytes else
                 "    epoch_ns(current_timestamp) - getvariable('benchmark_t0_ns'),"),
                "    CAST(current_timestamp AS TIMESTAMP),",
                ("    getvariable('benchmark_io').total_bytes, getvariable('benchmark_io').fact_bytes, "
                 "getvariable('benchmark_io').requests, getvariable('benchmark_io').details"
                 if measure_bytes else "    NULL, NULL, NULL, NULL"),
                ");",
            ]
        )
    return "\n".join(statements)


def remote_bytes_snapshot_sql(workload: PreparedWorkload) -> str:
    prefix = urlsplit(workload.ducklake_s3_url).path.rstrip('/') + '/'
    return f"""SET VARIABLE benchmark_io = (
        SELECT struct_pack(
            total_bytes := coalesce(sum(bytes_received), 0)::BIGINT,
            fact_bytes := coalesce(sum(bytes_received) FILTER (WHERE contains(url, {sql_literal(prefix + 'fact/')})), 0)::BIGINT,
            requests := coalesce(sum(get_requests), 0)::BIGINT,
            details := coalesce(to_json(list(struct_pack(url := url, bytes_received := bytes_received,
                                                        get_requests := get_requests) ORDER BY url)), '[]'))
        FROM httpfs_transfer_stats() WHERE contains(url, {sql_literal(prefix)})
    );"""


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
    query_parameters_sql = sql_literal(json.dumps(query.parameters, sort_keys=True))
    variant_setup_sql = sql_literal("\n".join(variant.setup_sql))
    statements = [
        "SET profiling_mode = 'standard';",
        "SET enable_profiling = 'json';",
        f"SET profiling_output = {sql_literal(str(profile_path))};",
        query_sql + ";",
        "SET enable_profiling = 'no_output';",
        "INSERT INTO benchmark_results.profile_runs (",
        "    run_id, experiment_name, workload_name, workload_kind,",
        "    system_name, system_kind, system_version, system_git_commit, system_repo, system_ref,",
        "    variant_label, variant_setup_sql, query_name, query_path, join_variant, cluster_count, query_parameters,",
        "    profile_path, cumulative_rows_scanned, total_rows_scanned, probe_rows_scanned, build_rows_scanned,",
        "    total_bytes_read, total_bytes_written, table_scan_file_count, profiled_at",
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
        f"    {query_parameters_sql},",
        f"    {sql_literal(str(profile_path))},",
        "    NULL,",
        "    NULL,",
        "    NULL,",
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
    workload_setup_sql: tuple[str, ...],
) -> str:
    statements = [
        "-- Generated by run_benchmarks.py",
        f"ATTACH {sql_literal(str(results_db))} AS benchmark_results;",
        "SET enable_progress_bar=false;",
    ]
    statements.extend(workload_setup_sql)
    statements.extend(variant.setup_sql)
    statements.extend(workload.post_variant_setup_sql)
    # Apply the experiment policy after variant setup so all variants have the
    # same resource budget, including workloads that spill beyond RAM.
    if benchmark.threads is not None:
        statements.append(f"SET threads = {benchmark.threads};")
    for setting in ("memory_limit", "temp_directory", "max_temp_directory_size"):
        value = getattr(benchmark, setting)
        if value is not None:
            statements.append(f"SET {setting} = {sql_literal(value)};")
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


@dataclass(frozen=True)
class RemoteHttpService:
    process: subprocess.Popen[str]
    setup_sql: tuple[str, ...]
    base_url: str


def execute_workload(binary: Path, db_path: Path, sql_path: Path) -> None:
    run_command([str(binary), str(db_path), "-f", str(sql_path)])


def bootstrap_ducklake_workload(
    binary: Path,
    workload: PreparedWorkload,
    remote_service: RemoteHttpService | None = None,
    benchmark: BenchmarkOptions | None = None,
) -> None:
    if workload.storage_mode != "ducklake":
        return
    bootstrap_sql_path = workload.ducklake_bootstrap_sql_path
    if bootstrap_sql_path is None:
        raise ValueError(f"DuckLake workload has no bootstrap SQL: {workload.name}")
    bootstrap_sql = bootstrap_sql_path.read_text(encoding="utf-8")
    if benchmark is not None:
        settings = []
        if benchmark.threads is not None:
            settings.append(f"SET threads={benchmark.threads};")
        for name in ("memory_limit", "temp_directory", "max_temp_directory_size"):
            value = getattr(benchmark, name)
            if value is not None:
                settings.append(f"SET {name}={sql_literal(value)};")
        bootstrap_sql = "\n".join(settings) + "\n" + bootstrap_sql
    if workload.ducklake_http_latency_ms is not None:
        if remote_service is None:
            raise ValueError(
                f"DuckLake HTTP workload has no running HTTP service: {workload.name}"
            )
        bootstrap_sql = bootstrap_sql.replace(
            DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER,
            remote_service.base_url,
        )
    if DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER in bootstrap_sql:
        raise ValueError(
            f"DuckLake bootstrap SQL still contains an HTTP URL placeholder: {workload.name}"
        )
    if workload.ducklake_data_source == "s3":
        bootstrap_sql = remote_s3_setup(workload) + "\n" + bootstrap_sql.replace(
            DUCKLAKE_S3_BASE_URL_PLACEHOLDER, workload.ducklake_s3_url)
    run_command([str(binary), str(workload.db_path), "-c", bootstrap_sql])
    if workload.ducklake_data_source == "s3":
        from benchmarking.ducklake_validation import validate_catalog
        validate_catalog(binary, workload, load_workload_setup_sql(workload))


def load_workload_setup_sql(workload: PreparedWorkload) -> tuple[str, ...]:
    statements = tuple(path.read_text(encoding="utf-8").strip() for path in workload.setup_sql_paths)
    if workload.ducklake_data_source == "s3":
        return (remote_s3_setup(workload),) + tuple(
            sql.replace(DUCKLAKE_S3_BASE_URL_PLACEHOLDER, workload.ducklake_s3_url) for sql in statements)
    return statements


def remote_s3_setup(workload: PreparedWorkload) -> str:
    if not workload.ducklake_s3_url or not workload.ducklake_s3_url.startswith("s3://"):
        raise ValueError("S3 workload requires a staged s3:// URL")
    region = os.environ.get("AWS_REGION")
    if not region:
        raise ValueError("S3 workload requires AWS_REGION from the EC2 host")
    return ("LOAD httpfs; LOAD aws; LOAD json;\nCREATE OR REPLACE SECRET jpb_s3 "
            f"(TYPE s3, PROVIDER credential_chain, CHAIN 'instance', REFRESH 'auto', REGION {sql_literal(region)}, "
            f"SCOPE {sql_literal(workload.ducklake_s3_url + '/')});")


def start_remote_http_service(
    workload: PreparedWorkload, *, preferred_port: int | None = None
) -> RemoteHttpService | None:
    if workload.ducklake_http_latency_ms is not None:
        latency_ms = workload.ducklake_http_latency_ms
        catalog_path = workload.parquet_catalog_path
        root_dir_name = "parquet"
        create_parquet_views = False
    else:
        latency_ms = workload.remote_http_parquet_latency_ms
        catalog_path = workload.remote_parquet_catalog_path
        root_dir_name = "remote_parquet"
        create_parquet_views = True
    if latency_ms is None or catalog_path is None:
        return None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", preferred_port or 0))
        host, port = sock.getsockname()

    root_dir = catalog_path.parent / root_dir_name
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarking.http_file_server",
            "--root-dir",
            str(root_dir),
            "--host",
            str(host),
            "--port",
            str(port),
            "--latency-ms",
            str(latency_ms),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    health_url = f"http://{host}:{port}/__health"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"remote HTTP server exited before startup\nstderr:\n{stderr}")
        try:
            with urlopen(health_url, timeout=0.2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError("remote HTTP server did not become healthy within 5 seconds")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    base_url = f"http://{host}:{port}"
    setup_sql = ["LOAD httpfs;"]
    if create_parquet_views:
        for table_name in sorted(catalog):
            file_urls = [
                f"{base_url}/{relative_path}"
                for relative_path in catalog[table_name]
            ]
            file_list_sql = ", ".join(sql_literal(url) for url in file_urls)
            setup_sql.append(
                f"CREATE OR REPLACE TEMP VIEW {quote_identifier(table_name)} AS "
                f"SELECT * FROM read_parquet([{file_list_sql}]);"
            )
    return RemoteHttpService(
        process=process,
        setup_sql=tuple(setup_sql),
        base_url=base_url,
    )


def stop_remote_http_service(service: RemoteHttpService | None) -> None:
    if service is None or service.process.poll() is not None:
        return
    service.process.terminate()
    try:
        service.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        service.process.kill()
        service.process.wait(timeout=5)


def reset_remote_http_stats(service: RemoteHttpService) -> None:
    with urlopen(f"{service.base_url}/__reset_stats", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"failed to reset HTTP stats: status {response.status}")


def read_remote_http_stats(service: RemoteHttpService) -> dict:
    with urlopen(f"{service.base_url}/__stats", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"failed to read HTTP stats: status {response.status}")
        return json.loads(response.read().decode("utf-8"))


def populate_profile_metrics(results_db: Path, run_id: str) -> None:
    con = duckdb.connect(str(results_db))
    try:
        ensure_filter_profile_columns(con)
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
            store_join_filter_metrics(con, run_id, profile_path, extract_join_filter_metrics(profile_data))
            con.execute(
                """
                UPDATE profile_runs
                SET cumulative_rows_scanned = ?,
                    total_rows_scanned = ?,
                    probe_rows_scanned = ?,
                    build_rows_scanned = ?,
                    total_bytes_read = ?,
                    total_bytes_written = ?,
                    table_scan_file_count = ?
                WHERE run_id = ? AND variant_label = ? AND query_name = ? AND profile_path = ?
                """,
                [
                    metrics.total_rows_scanned,
                    metrics.total_rows_scanned,
                    metrics.probe_rows_scanned,
                    metrics.build_rows_scanned,
                    metrics.total_bytes_read,
                    metrics.total_bytes_written,
                    metrics.table_scan_file_count,
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
    remote_service: RemoteHttpService | None = None,
) -> RunOutput:
    ensure_results_schema(results_db)
    queries = load_queries(workload.queries_dir)
    run_id = uuid.uuid4().hex
    static_setup_sql = load_workload_setup_sql(workload)
    print(
        f"Benchmarking workload={workload.name} system={system.name} queries={len(queries)}",
        file=sys.stderr,
    )
    owns_remote_service = remote_service is None
    active_remote_service = remote_service or start_remote_http_service(workload)
    workload_setup_sql = static_setup_sql
    if active_remote_service is not None:
        workload_setup_sql += active_remote_service.setup_sql
    try:
        for variant in variants:
            if not applies_to_workload(variant, workload.name):
                continue
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
                    workload_setup_sql,
                ),
                encoding="utf-8",
            )
            if active_remote_service is not None:
                reset_remote_http_stats(active_remote_service)
            execute_workload(system.binary_path, workload.db_path, sql_path)
            if active_remote_service is not None:
                http_stats_dir = workload_profile_dir / "http_stats"
                http_stats_dir.mkdir(parents=True, exist_ok=True)
                http_stats_path = http_stats_dir / (
                    label_to_path_component(variant.label) + ".json"
                )
                http_stats_path.write_text(
                    json.dumps(
                        read_remote_http_stats(active_remote_service), indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        if owns_remote_service:
            stop_remote_http_service(active_remote_service)
    if benchmark.collect_profiles:
        populate_profile_metrics(results_db, run_id)
    return RunOutput(run_id=run_id, query_count=len(queries))


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
