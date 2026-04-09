from __future__ import annotations

import json

import duckdb

from benchmark_spec import BenchmarkSpec
from layout import ClusterLayout, build_cluster_layout, mask_column_name


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
    else:
        entry["intervals_preview"] = {
            "first": list(layout.intervals[:5]),
            "last": list(layout.intervals[-5:]),
        }
    return entry


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


def write_database(spec: BenchmarkSpec) -> dict:
    if spec.db_path.exists():
        if not spec.force:
            raise FileExistsError(
                f"database already exists: {spec.db_path} (pass --force to overwrite)"
            )
        spec.db_path.unlink()

    spec.db_path.parent.mkdir(parents=True, exist_ok=True)
    spec.queries_dir.mkdir(parents=True, exist_ok=True)

    layouts = {
        cluster_count: build_cluster_layout(
            spec.dim_rows, spec.build_size, cluster_count
        )
        for cluster_count in spec.cluster_counts
    }

    con = duckdb.connect(str(spec.db_path))
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
        print("Creating dimension table...")
        con.execute(
            f"""
            CREATE TABLE dim_base AS
            SELECT
                {dim_base_select}
            FROM range(1, {spec.dim_rows + 1}) t(i)
            """
        )

        join_clauses: list[str] = []
        mask_selects: list[str] = []
        for cluster_count, layout in layouts.items():
            table_name = selected_keys_table_name(cluster_count)
            print(f"Preparing selected keys for cluster_count={cluster_count}...")
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
        joins_sql = "\n            ".join(join_clauses)
        con.execute(
            f"""
            CREATE TABLE dim AS
            SELECT
                {select_list}
            FROM dim_base AS b
            {joins_sql}
            """
        )
        con.execute("DROP TABLE dim_base")
        print("Done.")

        print("Creating fact table...")
        con.execute(
            f"""
            CREATE TABLE fact AS
            SELECT
                i AS fact_id,
                CAST(FLOOR(((i - 1) * {spec.dim_rows})::DOUBLE / {spec.fact_rows}) + 1 AS BIGINT) AS fk_ordered,
                CAST(FLOOR(random() * {spec.dim_rows}) + 1 AS BIGINT) AS fk_uniform,
                i * 17 AS measure
            FROM range(1, {spec.fact_rows + 1}) t(i)
            """
        )

        print("Done.")
    finally:
        con.close()

    metadata = {
        "db_path": str(spec.db_path),
        "queries_dir": str(spec.queries_dir),
        "build_size": spec.build_size,
        "domain_multiplier": spec.domain_multiplier,
        "dim_rows": spec.dim_rows,
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        "cluster_layouts": {
            str(cluster_count): layout_metadata(layout)
            for cluster_count, layout in layouts.items()
        },
    }
    metadata_path = spec.queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
