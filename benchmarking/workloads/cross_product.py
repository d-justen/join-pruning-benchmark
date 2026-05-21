from __future__ import annotations

import json
from pathlib import Path

import duckdb

from benchmarking.experiment_config import CrossProductWorkloadConfig
from benchmarking.manifests import PreparedWorkload, utc_now_iso, write_manifest
from benchmarking.workloads.external import export_remote_parquet_catalog


def build_size_label(build_size: int) -> str:
    if build_size % 1_000_000 == 0:
        return f"{build_size // 1_000_000}m"
    if build_size % 1_000 == 0:
        return f"{build_size // 1_000}k"
    return str(build_size)


def selectivity_label(selectivity: float) -> str:
    scaled = int(round(selectivity * 1_000))
    return f"s{scaled:04d}"


def mask_column_name(selectivity: float, predicate_type: str) -> str:
    return f"mask_{selectivity_label(selectivity)}_{predicate_type}"


def dimension_table_name(build_size: int) -> str:
    return f"dim_{build_size_label(build_size)}"


def fact_fk_column_name(build_size: int, probe_clustering: str) -> str:
    return f"fk_{build_size_label(build_size)}_{probe_clustering}"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


def _validate(spec: CrossProductWorkloadConfig) -> None:
    if not spec.build_sizes:
        raise ValueError("build_sizes must not be empty")
    if len(set(spec.build_sizes)) != len(spec.build_sizes):
        raise ValueError("build_sizes must be distinct")
    if not spec.selectivities:
        raise ValueError("selectivities must not be empty")
    if len(set(spec.selectivities)) != len(spec.selectivities):
        raise ValueError("selectivities must be distinct")
    if not spec.predicate_types:
        raise ValueError("predicate_types must not be empty")
    if not spec.probe_clusterings:
        raise ValueError("probe_clusterings must not be empty")
    allowed_predicate_types = {"range", "shoulder", "seasonal", "scattered"}
    allowed_probe_clusterings = {"ordered", "uniform"}
    for build_size in spec.build_sizes:
        if build_size <= 0:
            raise ValueError("build_sizes must be positive")
    for selectivity in spec.selectivities:
        if not 0 < selectivity <= 1:
            raise ValueError("selectivities must be in (0, 1]")
    for predicate_type in spec.predicate_types:
        if predicate_type not in allowed_predicate_types:
            raise ValueError(f"unsupported predicate_type: {predicate_type}")
    for probe_clustering in spec.probe_clusterings:
        if probe_clustering not in allowed_probe_clusterings:
            raise ValueError(f"unsupported probe_clustering: {probe_clustering}")
    if spec.fact_rows <= 0:
        raise ValueError("fact_rows must be positive")
    if spec.dummy_string_cols < 0:
        raise ValueError("dummy_string_cols cannot be negative")
    if spec.dummy_string_len <= 0:
        raise ValueError("dummy_string_len must be positive")


def _selected_rows(build_size: int, selectivity: float) -> int:
    return int(round(build_size * selectivity))


def _selected_key_rows(build_size: int, selected_rows: int, predicate_type: str, seed: int) -> str:
    if predicate_type == "range":
        return f"SELECT i AS d_key FROM range(1, {selected_rows + 1}) t(i)"
    if predicate_type == "shoulder":
        leading_rows = selected_rows // 2
        trailing_rows = selected_rows - leading_rows
        trailing_start = build_size - trailing_rows + 1
        if leading_rows == 0:
            return f"SELECT i AS d_key FROM range({trailing_start}, {build_size + 1}) t(i)"
        return (
            f"SELECT i AS d_key FROM range(1, {leading_rows + 1}) t(i)\n"
            f"UNION ALL\n"
            f"SELECT i AS d_key FROM range({trailing_start}, {build_size + 1}) t(i)"
        )
    if predicate_type == "seasonal":
        interval_count = min(10, selected_rows)
        rows_per_interval = selected_rows // interval_count
        remainder = selected_rows % interval_count
        gap_space = max(0, build_size - selected_rows)
        gap_base = gap_space // max(1, interval_count - 1)
        gap_remainder = gap_space % max(1, interval_count - 1) if interval_count > 1 else 0
        parts: list[str] = []
        current = 1
        for interval_index in range(interval_count):
            interval_size = rows_per_interval + (1 if interval_index < remainder else 0)
            end = current + interval_size
            parts.append(f"SELECT i AS d_key FROM range({current}, {end}) t(i)")
            current = end
            if interval_index + 1 < interval_count:
                current += gap_base + (1 if interval_index < gap_remainder else 0)
        return "\nUNION ALL\n".join(parts)
    return f"""
        SELECT
            1 + (({seed + build_size + selected_rows} + i * 7919) % {build_size}) AS d_key
        FROM range(0, {selected_rows}) t(i)
    """


def _create_dimension_table(
    con: duckdb.DuckDBPyConnection,
    spec: CrossProductWorkloadConfig,
    build_size: int,
    dimension_metadata: dict[str, dict],
) -> None:
    base_table_name = f"{dimension_table_name(build_size)}_base"
    final_table_name = dimension_table_name(build_size)
    dummy_exprs = [
        build_dummy_column_expr(index, spec.dummy_string_len, spec.seed)
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_list = ",\n                ".join(
        ["i AS d_key"] + dummy_exprs if dummy_exprs else ["i AS d_key"]
    )
    con.execute(
        f"""
        CREATE TABLE {base_table_name} AS
        SELECT
            {select_list}
        FROM range(1, {build_size + 1}) t(i)
        """
    )

    join_clauses: list[str] = []
    mask_selects: list[str] = []
    dimension_entry: dict[str, dict] = {}
    created_temp_tables: list[str] = []
    for selectivity in spec.selectivities:
        for predicate_type in spec.predicate_types:
            selected_rows = _selected_rows(build_size, selectivity)
            table_name = (
                f"selected_keys_{build_size_label(build_size)}_"
                f"{selectivity_label(selectivity)}_{predicate_type}"
            )
            column_name = mask_column_name(selectivity, predicate_type)
            con.execute(
                f"CREATE TEMP TABLE {table_name} AS "
                f"{_selected_key_rows(build_size, selected_rows, predicate_type, spec.seed)}"
            )
            created_temp_tables.append(table_name)
            join_clauses.append(
                f"LEFT JOIN {table_name} AS {table_name} ON b.d_key = {table_name}.d_key"
            )
            mask_selects.append(f"({table_name}.d_key IS NOT NULL) AS {column_name}")
            dimension_entry[column_name] = {
                "build_size": build_size,
                "selectivity": selectivity,
                "predicate_type": predicate_type,
                "selected_rows": selected_rows,
            }

    materialized_selects = ",\n            ".join(
        ["b.d_key"]
        + [f"b.dummy_s{index}" for index in range(1, spec.dummy_string_cols + 1)]
        + mask_selects
    )
    con.execute(
        f"""
        CREATE TABLE {final_table_name} AS
        SELECT
            {materialized_selects}
        FROM {base_table_name} AS b
            {' '.join(join_clauses)}
        """
    )
    con.execute(f"DROP TABLE {base_table_name}")
    for table_name in created_temp_tables:
        con.execute(f"DROP TABLE {table_name}")
    dimension_metadata[final_table_name] = dimension_entry


def _create_fact_table(con: duckdb.DuckDBPyConnection, spec: CrossProductWorkloadConfig) -> None:
    fk_exprs: list[str] = []
    for build_size in spec.build_sizes:
        ordered_column = fact_fk_column_name(build_size, "ordered")
        uniform_column = fact_fk_column_name(build_size, "uniform")
        fk_exprs.append(
            "CAST(FLOOR(((i - 1) * {build_size})::DOUBLE / {fact_rows}) + 1 AS BIGINT) "
            "AS {column_name}".format(
                build_size=build_size,
                fact_rows=spec.fact_rows,
                column_name=ordered_column,
            )
        )
        fk_exprs.append(
            f"CAST(FLOOR(random() * {build_size}) + 1 AS BIGINT) AS {uniform_column}"
        )
    select_list = ",\n            ".join(["i AS fact_id"] + fk_exprs + ["i * 17 AS measure"])
    con.execute(
        f"""
        CREATE TABLE fact AS
        SELECT
            {select_list}
        FROM range(1, {spec.fact_rows + 1}) t(i)
        """
    )


def _write_database(spec: CrossProductWorkloadConfig, db_path: Path, queries_dir: Path) -> dict:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        seed_value = ((spec.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")
        dimension_metadata: dict[str, dict[str, dict]] = {}
        for build_size in spec.build_sizes:
            _create_dimension_table(con, spec, build_size, dimension_metadata)
        _create_fact_table(con, spec)
    finally:
        con.close()
    metadata = {
        "db_path": str(db_path),
        "queries_dir": str(queries_dir),
        "build_sizes": list(spec.build_sizes),
        "selectivities": list(spec.selectivities),
        "predicate_types": list(spec.predicate_types),
        "probe_clusterings": list(spec.probe_clusterings),
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        "dimensions": dimension_metadata,
    }
    metadata_path = queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _render_query(
    spec: CrossProductWorkloadConfig,
    build_size: int,
    selectivity: float,
    predicate_type: str,
    probe_clustering: str,
) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d.dummy_s{index}) AS dummy_s{index}"
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_clause = ",\n".join(payload_exprs) if payload_exprs else "    COUNT(*) AS row_count"
    return f"""SELECT
{select_clause}
FROM fact AS f
JOIN {dimension_table_name(build_size)} AS d
  ON f.{fact_fk_column_name(build_size, probe_clustering)} = d.d_key
WHERE d.{mask_column_name(selectivity, predicate_type)};
"""


def _write_queries(spec: CrossProductWorkloadConfig, queries_dir: Path) -> list[Path]:
    written: list[Path] = []
    for build_size in spec.build_sizes:
        for selectivity in spec.selectivities:
            for predicate_type in spec.predicate_types:
                for probe_clustering in spec.probe_clusterings:
                    path = queries_dir / (
                        f"b{build_size_label(build_size)}_"
                        f"{selectivity_label(selectivity)}_"
                        f"{predicate_type}_{probe_clustering}.sql"
                    )
                    path.write_text(
                        _render_query(
                            spec,
                            build_size=build_size,
                            selectivity=selectivity,
                            predicate_type=predicate_type,
                            probe_clustering=probe_clustering,
                        ),
                        encoding="utf-8",
                    )
                    written.append(path)
    return written


def prepare_cross_product_workload(
    spec: CrossProductWorkloadConfig,
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
