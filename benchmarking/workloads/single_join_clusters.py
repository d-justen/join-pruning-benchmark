from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking.experiment_config import SingleJoinClustersWorkloadConfig
from benchmarking.manifests import PreparedWorkload, utc_now_iso, write_manifest
from benchmarking.workloads.external import export_remote_parquet_catalog


def mask_column_name(cluster_count: int) -> str:
    return f"mask_c{cluster_count:05d}"


@dataclass(frozen=True)
class ClusterLayout:
    dim_rows: int
    selected_rows: int
    cluster_count: int
    intervals: tuple[tuple[int, int], ...]
    range_sizes: tuple[int, ...]
    gap_sizes: tuple[int, ...]


def build_cluster_layout(dim_rows: int, selected_rows: int, cluster_count: int) -> ClusterLayout:
    range_base, range_remainder = divmod(selected_rows, cluster_count)
    gap_base, gap_remainder = divmod(dim_rows - selected_rows, cluster_count - 1)

    intervals: list[tuple[int, int]] = []
    range_sizes: list[int] = []
    gap_sizes: list[int] = []
    current = 1
    for range_index in range(cluster_count):
        range_size = range_base + (1 if range_index < range_remainder else 0)
        start = current
        end = current + range_size - 1
        intervals.append((start, end))
        range_sizes.append(range_size)
        current = end + 1
        if range_index + 1 < cluster_count:
            gap_size = gap_base + (1 if range_index < gap_remainder else 0)
            gap_sizes.append(gap_size)
            current += gap_size

    return ClusterLayout(
        dim_rows=dim_rows,
        selected_rows=selected_rows,
        cluster_count=cluster_count,
        intervals=tuple(intervals),
        range_sizes=tuple(range_sizes),
        gap_sizes=tuple(gap_sizes),
    )


def selected_keys_table_name(cluster_count: int) -> str:
    return f"selected_keys_c{cluster_count:05d}"


def selected_keys_query(layout: ClusterLayout) -> str:
    build_size = layout.selected_rows
    cluster_count = layout.cluster_count
    dim_rows = layout.dim_rows
    range_base, range_remainder = divmod(build_size, cluster_count)
    gap_base, gap_remainder = divmod(dim_rows - build_size, cluster_count - 1)
    long_range_cutoff = range_remainder * (range_base + 1)
    return f"""
        SELECT
            1
            + range_idx * {range_base}
            + LEAST(range_idx, {range_remainder})
            + range_idx * {gap_base}
            + LEAST(range_idx, {gap_remainder})
            + offset_in_range AS d_key
        FROM (
            SELECT
                CASE
                    WHEN r < {long_range_cutoff} THEN CAST(FLOOR(r::DOUBLE / {range_base + 1}) AS BIGINT)
                    ELSE {range_remainder} + CAST(FLOOR((r - {long_range_cutoff})::DOUBLE / {range_base}) AS BIGINT)
                END AS range_idx,
                CASE
                    WHEN r < {long_range_cutoff} THEN r % {range_base + 1}
                    ELSE (r - {long_range_cutoff}) % {range_base}
                END AS offset_in_range
            FROM range(0, {build_size}) t(r)
        ) generated_keys
    """


def layout_metadata(layout: ClusterLayout) -> dict:
    entry = {
        "mask_column": mask_column_name(layout.cluster_count),
        "cluster_count": layout.cluster_count,
        "interval_count": len(layout.intervals),
        "range_size_min": min(layout.range_sizes),
        "range_size_max": max(layout.range_sizes),
        "gap_size_min": min(layout.gap_sizes) if layout.gap_sizes else 0,
        "gap_size_max": max(layout.gap_sizes) if layout.gap_sizes else 0,
    }
    if len(layout.intervals) <= 64:
        entry["intervals"] = list(layout.intervals)
    return entry


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


def _validate(spec: SingleJoinClustersWorkloadConfig) -> None:
    if spec.build_size <= 0:
        raise ValueError("build_size must be positive")
    if spec.domain_multiplier <= 0:
        raise ValueError("domain_multiplier must be positive")
    if spec.fact_rows <= 0:
        raise ValueError("fact_rows must be positive")
    if spec.dummy_string_cols < 0:
        raise ValueError("dummy_string_cols cannot be negative")
    if spec.dummy_string_len <= 0:
        raise ValueError("dummy_string_len must be positive")
    dim_rows = spec.build_size * spec.domain_multiplier
    for cluster_count in spec.cluster_counts:
        if cluster_count < 2:
            raise ValueError("cluster_counts must be at least 2")
        if cluster_count > spec.build_size:
            raise ValueError(
                f"cluster_count {cluster_count} exceeds build_size {spec.build_size}"
            )
    if dim_rows < spec.build_size:
        raise ValueError("dimension cardinality must be at least build size")


def render_query(spec: SingleJoinClustersWorkloadConfig, join_column: str, cluster_count: int) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d.dummy_s{index}) AS dummy_s{index}"
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_clause = ",\n".join(payload_exprs) if payload_exprs else "    COUNT(*) AS row_count"
    return f"""SELECT
{select_clause}
FROM fact AS f
JOIN dim AS d
  ON f.{join_column} = d.d_key
WHERE d.{mask_column_name(cluster_count)};
"""


def _write_queries(spec: SingleJoinClustersWorkloadConfig, queries_dir: Path) -> list[Path]:
    written: list[Path] = []
    for cluster_count in spec.cluster_counts:
        for join_variant, join_column in (("ordered", "fk_ordered"), ("uniform", "fk_uniform")):
            path = queries_dir / f"{join_variant}_cc_{cluster_count:05d}.sql"
            path.write_text(
                render_query(spec, join_column=join_column, cluster_count=cluster_count),
                encoding="utf-8",
            )
            written.append(path)
    return written


def _write_database(
    spec: SingleJoinClustersWorkloadConfig,
    db_path: Path,
    queries_dir: Path,
) -> dict:
    dim_rows = spec.build_size * spec.domain_multiplier
    layouts = {
        cluster_count: build_cluster_layout(dim_rows, spec.build_size, cluster_count)
        for cluster_count in spec.cluster_counts
    }
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        seed_value = ((spec.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")
        dummy_exprs = [
            build_dummy_column_expr(index, spec.dummy_string_len, spec.seed)
            for index in range(1, spec.dummy_string_cols + 1)
        ]
        dim_base_select = ",\n    ".join(
            ["i AS d_key"] + dummy_exprs if dummy_exprs else ["i AS d_key"]
        )
        con.execute(
            f"""
            CREATE TABLE dim_base AS
            SELECT
                {dim_base_select}
            FROM range(1, {dim_rows + 1}) t(i)
            """
        )

        join_clauses: list[str] = []
        mask_selects: list[str] = []
        for cluster_count, layout in layouts.items():
            table_name = selected_keys_table_name(cluster_count)
            con.execute(f"CREATE TEMP TABLE {table_name} AS {selected_keys_query(layout)}")
            join_clauses.append(
                f"LEFT JOIN {table_name} AS {table_name} ON b.d_key = {table_name}.d_key"
            )
            mask_selects.append(
                f"({table_name}.d_key IS NOT NULL) AS {mask_column_name(cluster_count)}"
            )
        select_list = ",\n    ".join(
            ["b.d_key"]
            + [f"dummy_s{i}" for i in range(1, spec.dummy_string_cols + 1)]
            + mask_selects
        )
        con.execute(
            f"""
            CREATE TABLE dim AS
            SELECT
                {select_list}
            FROM dim_base AS b
            {' '.join(join_clauses)}
            """
        )
        con.execute("DROP TABLE dim_base")
        con.execute(
            f"""
            CREATE TABLE fact AS
            SELECT
                i AS fact_id,
                CAST(FLOOR(((i - 1) * {dim_rows})::DOUBLE / {spec.fact_rows}) + 1 AS BIGINT) AS fk_ordered,
                CAST(FLOOR(random() * {dim_rows}) + 1 AS BIGINT) AS fk_uniform,
                i * 17 AS measure
            FROM range(1, {spec.fact_rows + 1}) t(i)
            """
        )
    finally:
        con.close()
    metadata = {
        "db_path": str(db_path),
        "queries_dir": str(queries_dir),
        "build_size": spec.build_size,
        "domain_multiplier": spec.domain_multiplier,
        "dim_rows": dim_rows,
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        "cluster_layouts": {
            str(cluster_count): layout_metadata(layout)
            for cluster_count, layout in layouts.items()
        },
    }
    metadata_path = queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def prepare_single_join_clusters_workload(
    spec: SingleJoinClustersWorkloadConfig,
    artifact_dir: Path,
) -> PreparedWorkload:
    _validate(spec)
    queries_dir = artifact_dir / "queries"
    db_path = artifact_dir / "benchmark.duckdb"
    metadata = _write_database(spec, db_path, queries_dir)
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
