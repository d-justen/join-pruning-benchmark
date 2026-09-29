#!/usr/bin/env python3
"""Measure synthetic 122,880-row-group ranges for benchmark join columns."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "out" / "analysis" / "join-column-ranges"
WORKLOADS = {
    "join-pruning-workload": {
        "database": ROOT / "out/experiments/paper-local/workloads/join-pruning-workload/benchmark.duckdb",
        "queries": ROOT / "out/experiments/paper-local/workloads/join-pruning-workload/queries",
    },
    "join-order-benchmark": {
        "database": ROOT / "external_benchmarks/join-order-benchmark/benchmark.duckdb",
        "queries": ROOT / "external_benchmarks/join-order-benchmark/queries",
    },
    "tpch-skew": {
        "database": ROOT / "external_benchmarks/tpch-skew/tpch_sf100_z2.duckdb",
        "queries": ROOT / "external_benchmarks/tpch-skew/queries",
    },
}
JPB_COLUMNS = {
    "dim_100k": ["d_key"],
    "dim_10m": ["d_key"],
    "fact": ["fk_100k_ordered", "fk_100k_uniform", "fk_10m_ordered", "fk_10m_uniform"],
}
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
EQUALITY_RE = re.compile(rf"(?<![<>!])\b({IDENT}(?:\.{IDENT})?)\s*=\s*({IDENT}(?:\.{IDENT})?)\b")


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def database_schema(path: Path) -> dict[str, set[str]]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name, column_name FROM duckdb_columns() WHERE schema_name='main'"
        ).fetchall()
    finally:
        con.close()
    result: dict[str, set[str]] = defaultdict(set)
    for table, column in rows:
        result[table.lower()].add(column.lower())
    return dict(result)


def strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def query_tables_and_aliases(sql: str, schema: dict[str, set[str]]) -> tuple[set[str], dict[str, str]]:
    tables = {table for table in schema if re.search(rf"\b{re.escape(table)}\b", sql, re.I)}
    aliases = {table: table for table in tables}
    for table in tables:
        # Benchmark SQL consistently uses either `table AS alias` or `table alias`.
        match = re.search(
            rf"\b{re.escape(table)}\b\s+(?:AS\s+)?({IDENT})\b", sql, re.I
        )
        if match and match.group(1).lower() not in {
            "where", "join", "on", "group", "order", "limit", "union"
        }:
            aliases[match.group(1).lower()] = table
    return tables, aliases


def resolve_column(
    token: str, tables: set[str], aliases: dict[str, str], schema: dict[str, set[str]]
) -> tuple[str, str] | None:
    pieces = token.lower().split(".")
    if len(pieces) == 2:
        table = aliases.get(pieces[0])
        if table and pieces[1] in schema[table]:
            return table, pieces[1]
        return None
    candidates = [(table, pieces[0]) for table in tables if pieces[0] in schema[table]]
    return candidates[0] if len(candidates) == 1 else None


def extract_join_columns(query_dir: Path, schema: dict[str, set[str]]) -> dict[str, list[str]]:
    selected: set[tuple[str, str]] = set()
    for path in sorted(query_dir.glob("*.sql")):
        sql = strip_literals_and_comments(path.read_text(encoding="utf-8"))
        tables, aliases = query_tables_and_aliases(sql, schema)
        for left, right in EQUALITY_RE.findall(sql):
            left_col = resolve_column(left, tables, aliases, schema)
            right_col = resolve_column(right, tables, aliases, schema)
            if left_col and right_col and left_col != right_col:
                selected.add(left_col)
                selected.add(right_col)
    result: dict[str, list[str]] = defaultdict(list)
    for table, column in sorted(selected):
        result[table].append(column)
    return dict(result)


def build_inventory() -> dict[str, dict[str, list[str]]]:
    inventory = {"join-pruning-workload": JPB_COLUMNS}
    for workload in ("join-order-benchmark", "tpch-skew"):
        spec = WORKLOADS[workload]
        schema = database_schema(spec["database"])
        inventory[workload] = extract_join_columns(spec["queries"], schema)
    return inventory


def aggregate_table(
    source: duckdb.DuckDBPyConnection, table: str, columns: list[str], group_size: int
) -> list[tuple]:
    aggregates = []
    for column in columns:
        q = quote_ident(column)
        aggregates.extend((f"min({q})", f"max({q})"))
    sql = (
        f"SELECT floor(rowid / {group_size})::BIGINT + 1 AS row_group_index, "
        + ", ".join(aggregates)
        + f" FROM {quote_ident(table)} GROUP BY 1 ORDER BY 1"
    )
    return source.execute(sql).fetchall()


def run(output_dir: Path, group_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()
    (output_dir / "join_columns.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "group_size": group_size,
        "row_group_index_base": 1,
        "row_group_definition": f"floor(rowid / {group_size}) + 1",
        "workloads": {
            workload: {
                "database": str(spec["database"]),
                "database_size_bytes": spec["database"].stat().st_size,
                "query_directory": str(spec["queries"]),
                "query_count": len(list(spec["queries"].glob("*.sql"))),
                "join_column_count": sum(len(columns) for columns in inventory[workload].values()),
            }
            for workload, spec in WORKLOADS.items()
        },
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    result_db = output_dir / "join_column_ranges.duckdb"
    if result_db.exists():
        result_db.unlink()
    out = duckdb.connect(str(result_db))
    out.execute("""
        CREATE TABLE join_column_ranges (
            workload_name VARCHAR, table_name VARCHAR, column_name VARCHAR,
            row_group_index BIGINT, min_value BIGINT, max_value BIGINT, distance HUGEINT
        )
    """)
    insert_sql = "INSERT INTO join_column_ranges VALUES (?, ?, ?, ?, ?, ?, ?)"
    try:
        for workload, tables in inventory.items():
            db_path = WORKLOADS[workload]["database"]
            source = duckdb.connect(str(db_path), read_only=True)
            try:
                for table, columns in sorted(tables.items()):
                    print(f"{workload}: scanning {table} ({', '.join(columns)})", flush=True)
                    grouped = aggregate_table(source, table, columns, group_size)
                    rows = []
                    for values in grouped:
                        group_index = values[0]
                        for index, column in enumerate(columns):
                            minimum, maximum = values[1 + index * 2 : 3 + index * 2]
                            if minimum is not None:
                                rows.append((workload, table, column, group_index, minimum, maximum, maximum - minimum))
                    out.executemany(insert_sql, rows)
            finally:
                source.close()
        out.execute("""
            CREATE TABLE top_10_join_column_ranges AS
            SELECT * FROM join_column_ranges
            ORDER BY distance DESC, workload_name, table_name, column_name, row_group_index
            LIMIT 10
        """)
        for table, filename in (
            ("join_column_ranges", "all_join_column_ranges.csv"),
            ("top_10_join_column_ranges", "top_10_join_column_ranges.csv"),
        ):
            destination = (output_dir / filename).as_posix().replace("'", "''")
            out.execute(f"COPY {table} TO '{destination}' (HEADER, DELIMITER ',')")
        print("\nTop 10:")
        out.sql("SELECT * FROM top_10_join_column_ranges").show(max_width=160)
    finally:
        out.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-size", type=int, default=122_880)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    run(args.output_dir.resolve(), args.group_size)


if __name__ == "__main__":
    main()
