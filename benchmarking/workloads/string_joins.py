from __future__ import annotations

import json
from pathlib import Path

import duckdb

from benchmarking.experiment_config import StringJoinsWorkloadConfig
from benchmarking.manifests import PreparedWorkload, utc_now_iso, write_manifest
from benchmarking.workloads.external import export_remote_parquet_catalog


UUID_MASK_COLUMN = "mask_gapped"
UUID_V7_MAX_TIMESTAMP_MS = (1 << 48) - 1


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


def uuid_key_expr(timestamp_start_ms: int, timestamp_step_ms: int, seed: int) -> str:
    timestamp_ms = f"({timestamp_start_ms} + (d_key - 1) * {timestamp_step_ms})"
    ordered_prefix = f"printf('%012x', {timestamp_ms})"
    suffix_hash = f"md5(CAST(d_key AS VARCHAR) || {sql_literal(f':uuid-seed:{seed}')})"
    return (
        "lower("
        f"substr({ordered_prefix}, 1, 8) || '-' || "
        f"substr({ordered_prefix}, 9, 4) || '-' || "
        f"'7' || substr({suffix_hash}, 1, 3) || '-' || "
        f"'8' || substr({suffix_hash}, 4, 3) || '-' || "
        f"substr({suffix_hash}, 8, 12)"
        ")"
    )


def _validate(spec: StringJoinsWorkloadConfig) -> None:
    if not spec.airport_csv_path.exists():
        raise FileNotFoundError(f"airport CSV not found: {spec.airport_csv_path}")
    if spec.uuid_dim_size <= 0:
        raise ValueError("uuid_dim_size must be positive")
    if spec.uuid_timestamp_start_ms < 0:
        raise ValueError("uuid_timestamp_start_ms cannot be negative")
    if spec.uuid_timestamp_step_ms <= 0:
        raise ValueError("uuid_timestamp_step_ms must be positive")
    final_timestamp = (
        spec.uuid_timestamp_start_ms
        + (spec.uuid_dim_size - 1) * spec.uuid_timestamp_step_ms
    )
    if final_timestamp > UUID_V7_MAX_TIMESTAMP_MS:
        raise ValueError("UUIDv7 timestamp exceeds the 48-bit timestamp field")
    if not spec.uuid_selected_ranges:
        raise ValueError("uuid_selected_ranges must not be empty")
    previous_end = 0
    selected_rows = 0
    for index, (start, end) in enumerate(spec.uuid_selected_ranges, start=1):
        if start < 1 or end > spec.uuid_dim_size:
            raise ValueError(
                f"uuid_selected_ranges[{index}] must be within [1, uuid_dim_size]"
            )
        if start > end:
            raise ValueError(f"uuid_selected_ranges[{index}] start must not exceed end")
        if start <= previous_end:
            raise ValueError("uuid_selected_ranges must be sorted and non-overlapping")
        selected_rows += end - start + 1
        previous_end = end
    if selected_rows <= 50:
        raise ValueError(
            "uuid_selected_ranges must select more than 50 rows to avoid a direct IN filter"
        )
    if spec.fact_rows <= 0:
        raise ValueError("fact_rows must be positive")
    if spec.dummy_string_cols < 0:
        raise ValueError("dummy_string_cols cannot be negative")
    if spec.dummy_string_len <= 0:
        raise ValueError("dummy_string_len must be positive")


def uuid_mask_expr(selected_ranges: tuple[tuple[int, int], ...]) -> str:
    return " OR ".join(
        f"(d_key BETWEEN {start} AND {end})" for start, end in selected_ranges
    )


def _create_airport_dim(con: duckdb.DuckDBPyConnection, spec: StringJoinsWorkloadConfig) -> dict:
    con.execute(
        """
        CREATE TABLE airport_raw AS
        SELECT *
        FROM read_csv_auto(?, HEADER = TRUE)
        """,
        [str(spec.airport_csv_path)],
    )
    con.execute(
        """
        CREATE TABLE airport_dim AS
        SELECT *
        FROM airport_raw
        WHERE ident IS NOT NULL AND ident <> ''
        """
    )
    con.execute("DROP TABLE airport_raw")
    con.execute(
        """
        CREATE TEMP TABLE airport_domain AS
        SELECT row_number() OVER (ORDER BY ident) AS d_key, ident
        FROM (SELECT DISTINCT ident FROM airport_dim)
        ORDER BY ident
        """
    )
    airport_rows = con.execute("SELECT COUNT(*) FROM airport_dim").fetchone()[0]
    airport_ident_count = con.execute("SELECT COUNT(*) FROM airport_domain").fetchone()[0]
    antarctica_rows = con.execute(
        "SELECT COUNT(*) FROM airport_dim WHERE continent = 'AN'"
    ).fetchone()[0]
    if airport_ident_count <= 0:
        raise ValueError("airport CSV contains no non-empty ident values")
    return {
        "airport_rows": airport_rows,
        "airport_ident_count": airport_ident_count,
        "antarctica_rows": antarctica_rows,
    }


def _create_uuid_dim(con: duckdb.DuckDBPyConnection, spec: StringJoinsWorkloadConfig) -> dict:
    selected_rows = sum(end - start + 1 for start, end in spec.uuid_selected_ranges)
    dummy_exprs = [
        build_dummy_column_expr(index, spec.dummy_string_len, spec.seed)
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_list = ",\n            ".join(
        [
            "d_key",
            (
                f"{uuid_key_expr(spec.uuid_timestamp_start_ms, spec.uuid_timestamp_step_ms, spec.seed)} "
                "AS uuid_key"
            ),
        ]
        + dummy_exprs
        + [f"({uuid_mask_expr(spec.uuid_selected_ranges)}) AS {UUID_MASK_COLUMN}"]
    )
    con.execute(
        f"""
        CREATE TABLE uuid_dim AS
        SELECT
            {select_list}
        FROM range(1, {spec.uuid_dim_size + 1}) t(d_key)
        """
    )
    first_selected_key = spec.uuid_selected_ranges[0][0]
    last_selected_key = spec.uuid_selected_ranges[-1][1]
    selected_prefixes = con.execute(
        f"""
        SELECT COUNT(DISTINCT substr(uuid_key, 1, 4))
        FROM uuid_dim
        WHERE {UUID_MASK_COLUMN}
        """
    ).fetchone()[0]
    prefix_covered_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM uuid_dim
        WHERE substr(uuid_key, 1, 4) IN (
            SELECT DISTINCT substr(uuid_key, 1, 4)
            FROM uuid_dim
            WHERE {UUID_MASK_COLUMN}
        )
        """
    ).fetchone()[0]
    min_max_covered_rows = last_selected_key - first_selected_key + 1
    if prefix_covered_rows >= min_max_covered_rows:
        raise ValueError(
            "uuid_selected_ranges do not create prefix-visible gaps beyond min/max"
        )
    return {
        "uuid_dim_size": spec.uuid_dim_size,
        "uuid_timestamp_start_ms": spec.uuid_timestamp_start_ms,
        "uuid_timestamp_step_ms": spec.uuid_timestamp_step_ms,
        "uuid_selected_ranges": [list(item) for item in spec.uuid_selected_ranges],
        "uuid_selected_rows": selected_rows,
        "uuid_selected_fraction": selected_rows / spec.uuid_dim_size,
        "uuid_min_max_covered_rows": min_max_covered_rows,
        "uuid_min_max_coverage": min_max_covered_rows / spec.uuid_dim_size,
        "uuid_selected_prefixes": selected_prefixes,
        "uuid_prefix_covered_rows": prefix_covered_rows,
        "uuid_prefix_coverage": prefix_covered_rows / spec.uuid_dim_size,
        "uuid_mask_column": UUID_MASK_COLUMN,
    }


def _create_fact_table(
    con: duckdb.DuckDBPyConnection,
    spec: StringJoinsWorkloadConfig,
    airport_ident_count: int,
) -> None:
    con.execute(
        f"""
        CREATE TABLE fact AS
        WITH fact_keys AS (
            SELECT
                i AS fact_id,
                CAST(FLOOR(((i - 1) * {airport_ident_count})::DOUBLE / {spec.fact_rows}) + 1 AS BIGINT)
                    AS airport_ordered_key,
                CAST(FLOOR(random() * {airport_ident_count}) + 1 AS BIGINT)
                    AS airport_uniform_key,
                CAST(FLOOR(((i - 1) * {spec.uuid_dim_size})::DOUBLE / {spec.fact_rows}) + 1 AS BIGINT)
                    AS uuid_ordered_key,
                CAST(FLOOR(random() * {spec.uuid_dim_size}) + 1 AS BIGINT)
                    AS uuid_uniform_key,
                i * 17 AS measure
            FROM range(1, {spec.fact_rows + 1}) t(i)
        )
        SELECT
            f.fact_id,
            ordered_airport.ident AS airport_ident_ordered,
            uniform_airport.ident AS airport_ident_uniform,
            ordered_uuid.uuid_key AS uuid_v7_like_ordered,
            uniform_uuid.uuid_key AS uuid_v4_like_uniform,
            f.measure
        FROM fact_keys AS f
        JOIN airport_domain AS ordered_airport
          ON f.airport_ordered_key = ordered_airport.d_key
        JOIN airport_domain AS uniform_airport
          ON f.airport_uniform_key = uniform_airport.d_key
        JOIN uuid_dim AS ordered_uuid
          ON f.uuid_ordered_key = ordered_uuid.d_key
        JOIN uuid_dim AS uniform_uuid
          ON f.uuid_uniform_key = uniform_uuid.d_key
        """
    )


def _write_database(spec: StringJoinsWorkloadConfig, db_path: Path, queries_dir: Path) -> dict:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        seed_value = ((spec.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")
        airport_metadata = _create_airport_dim(con, spec)
        uuid_metadata = _create_uuid_dim(con, spec)
        _create_fact_table(con, spec, airport_metadata["airport_ident_count"])
    finally:
        con.close()
    metadata = {
        "db_path": str(db_path),
        "queries_dir": str(queries_dir),
        "airport_csv_path": str(spec.airport_csv_path),
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        **airport_metadata,
        **uuid_metadata,
        "queries": [
            "airport_ordered_antarctica",
            "airport_uniform_antarctica",
            "uuid_v7_like_ordered",
            "uuid_v4_like_uniform",
        ],
    }
    metadata_path = queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _uuid_select_clause(spec: StringJoinsWorkloadConfig) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d.dummy_s{index}) AS dummy_s{index}"
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    return ",\n".join(payload_exprs) if payload_exprs else "    COUNT(*) AS row_count"


def _airport_select_clause() -> str:
    return ",\n".join(
        [
            "    ANY_VALUE(a.ident) AS ident",
            "    ANY_VALUE(a.name) AS name",
            "    ANY_VALUE(a.municipality) AS municipality",
        ]
    )


def _write_queries(spec: StringJoinsWorkloadConfig, queries_dir: Path) -> list[Path]:
    queries = {
        "airport_ordered_antarctica.sql": f"""SELECT
{_airport_select_clause()}
FROM fact AS f
JOIN airport_dim AS a
  ON f.airport_ident_ordered = a.ident
WHERE a.continent = 'AN';
""",
        "airport_uniform_antarctica.sql": f"""SELECT
{_airport_select_clause()}
FROM fact AS f
JOIN airport_dim AS a
  ON f.airport_ident_uniform = a.ident
WHERE a.continent = 'AN';
""",
        "uuid_v7_like_ordered.sql": f"""SELECT
{_uuid_select_clause(spec)}
FROM fact AS f
JOIN uuid_dim AS d
  ON f.uuid_v7_like_ordered = d.uuid_key
WHERE d.{UUID_MASK_COLUMN};
""",
        "uuid_v4_like_uniform.sql": f"""SELECT
{_uuid_select_clause(spec)}
FROM fact AS f
JOIN uuid_dim AS d
  ON f.uuid_v4_like_uniform = d.uuid_key
WHERE d.{UUID_MASK_COLUMN};
""",
    }
    written: list[Path] = []
    for file_name, sql in queries.items():
        path = queries_dir / file_name
        path.write_text(sql, encoding="utf-8")
        written.append(path)
    return written


def prepare_string_joins_workload(
    spec: StringJoinsWorkloadConfig,
    artifact_dir: Path,
) -> PreparedWorkload:
    _validate(spec)
    queries_dir = artifact_dir / "queries"
    db_path = artifact_dir / "benchmark.duckdb"
    _write_database(spec, db_path, queries_dir)
    _write_queries(spec, queries_dir)
    metadata_path = queries_dir / "metadata.json"
    runtime_db_path = db_path.resolve()
    remote_parquet_catalog_path: Path | None = None
    if spec.remote_http_parquet_latency_ms is not None:
        remote_parquet_catalog_path = export_remote_parquet_catalog(runtime_db_path, artifact_dir)
        runtime_db_path = (artifact_dir / "benchmark.duckdb").resolve()
        if runtime_db_path.exists():
            runtime_db_path.unlink()
    manifest = PreparedWorkload(
        name=spec.name,
        kind=spec.kind,
        source_root=artifact_dir.resolve(),
        artifact_dir=artifact_dir.resolve(),
        db_path=runtime_db_path,
        queries_dir=queries_dir.resolve(),
        metadata_path=metadata_path.resolve(),
        init_sql_paths=(),
        remote_http_parquet_latency_ms=spec.remote_http_parquet_latency_ms,
        remote_parquet_catalog_path=remote_parquet_catalog_path,
        prepared_at=utc_now_iso(),
    )
    write_manifest(artifact_dir / "manifest.json", manifest)
    return manifest
