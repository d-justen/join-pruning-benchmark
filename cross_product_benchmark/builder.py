from __future__ import annotations

import json

import duckdb

from .layouts import (
    IntervalLayout,
    ScatteredLayout,
    build_predicate_layout,
    predicate_layout_metadata,
)
from .spec import (
    CrossProductBenchmarkSpec,
    build_size_label,
    dimension_table_name,
    fact_fk_column_name,
    mask_column_name,
    selectivity_label,
)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


def selected_keys_table_name(build_size: int, selectivity: float, predicate_type: str) -> str:
    return (
        f"selected_keys_{build_size_label(build_size)}_"
        f"{selectivity_label(selectivity)}_{predicate_type}"
    )


def selected_keys_query(layout: IntervalLayout | ScatteredLayout) -> str:
    if isinstance(layout, IntervalLayout):
        parts = []
        for start, end in layout.intervals:
            parts.append(
                f"SELECT i AS d_key FROM range({start}, {end + 1}) t(i)"
            )
        return "\nUNION ALL\n".join(parts)

    return f"""
        SELECT
            1 + (({layout.offset} + i * {layout.stride}) % {layout.build_size}) AS d_key
        FROM range(0, {layout.selected_rows}) t(i)
    """


def create_dimension_table(
    con: duckdb.DuckDBPyConnection,
    spec: CrossProductBenchmarkSpec,
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

    print(f"Creating {final_table_name} base rows...")
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
    selected_key_tables: list[str] = []
    build_key = build_size_label(build_size)
    dimension_entry: dict[str, dict] = {}

    for selectivity in spec.selectivities:
        for predicate_type in spec.predicate_types:
            selected_rows = spec.selected_rows(build_size, selectivity)
            layout_seed = spec.seed + build_size + int(round(selectivity * 1_000_000))
            layout = build_predicate_layout(
                predicate_type,
                build_size,
                selected_rows,
                layout_seed,
            )
            table_name = selected_keys_table_name(build_size, selectivity, predicate_type)
            column_name = mask_column_name(selectivity, predicate_type)
            print(
                "Preparing selected keys for "
                f"{final_table_name} {column_name}..."
            )
            con.execute(f"CREATE TEMP TABLE {table_name} AS {selected_keys_query(layout)}")
            selected_key_tables.append(table_name)
            join_clauses.append(
                f"LEFT JOIN {table_name} AS {table_name} ON b.d_key = {table_name}.d_key"
            )
            mask_selects.append(f"({table_name}.d_key IS NOT NULL) AS {column_name}")
            dimension_entry[column_name] = {
                "build_size": build_size,
                "build_size_label": build_key,
                "selectivity": selectivity,
                "predicate_type": predicate_type,
                "mask_column": column_name,
                **predicate_layout_metadata(layout),
            }

    materialized_selects = ",\n            ".join(
        ["b.d_key"]
        + [f"b.dummy_s{index}" for index in range(1, spec.dummy_string_cols + 1)]
        + mask_selects
    )
    joins_sql = "\n            ".join(join_clauses)
    print(f"Materializing {final_table_name}...")
    con.execute(
        f"""
        CREATE TABLE {final_table_name} AS
        SELECT
            {materialized_selects}
        FROM {base_table_name} AS b
            {joins_sql}
        """
    )
    con.execute(f"DROP TABLE {base_table_name}")
    for table_name in selected_key_tables:
        con.execute(f"DROP TABLE {table_name}")
    dimension_metadata[final_table_name] = dimension_entry


def create_fact_table(con: duckdb.DuckDBPyConnection, spec: CrossProductBenchmarkSpec) -> None:
    print("Creating fact table...")
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

    select_list = ",\n            ".join(
        ["i AS fact_id"] + fk_exprs + ["i * 17 AS measure"]
    )
    con.execute(
        f"""
        CREATE TABLE fact AS
        SELECT
            {select_list}
        FROM range(1, {spec.fact_rows + 1}) t(i)
        """
    )


def write_database(spec: CrossProductBenchmarkSpec) -> dict:
    if spec.db_path.exists():
        if not spec.force:
            raise FileExistsError(
                f"database already exists: {spec.db_path} (pass --force to overwrite)"
            )
        spec.db_path.unlink()

    spec.db_path.parent.mkdir(parents=True, exist_ok=True)
    spec.queries_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(spec.db_path))
    try:
        seed_value = ((spec.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")

        dimension_metadata: dict[str, dict[str, dict]] = {}
        for build_size in spec.build_sizes:
            create_dimension_table(con, spec, build_size, dimension_metadata)

        create_fact_table(con, spec)
        print("Done.")
    finally:
        con.close()

    metadata = {
        "db_path": str(spec.db_path),
        "queries_dir": str(spec.queries_dir),
        "build_sizes": list(spec.build_sizes),
        "selectivities": list(spec.selectivities),
        "predicate_types": list(spec.predicate_types),
        "probe_clusterings": list(spec.probe_clusterings),
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        "dimensions": dimension_metadata,
        "query_count": (
            len(spec.build_sizes)
            * len(spec.selectivities)
            * len(spec.predicate_types)
            * len(spec.probe_clusterings)
        ),
    }
    metadata_path = spec.queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
