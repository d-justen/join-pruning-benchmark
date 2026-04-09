from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_spec import BenchmarkSpec, parse_cluster_counts
from duckdb_builder import write_database
from query_gen import write_queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DuckDB benchmark for single-join semi-join reduction / "
            "sideways information passing experiments."
        )
    )
    parser.add_argument("--db", required=True, type=Path, help="output DuckDB file")
    parser.add_argument(
        "--queries-dir",
        required=True,
        type=Path,
        help="directory that will receive generated .sql files",
    )
    parser.add_argument(
        "--build-size",
        required=True,
        type=int,
        help="number of dimension rows selected by each mask",
    )
    parser.add_argument(
        "--domain-multiplier",
        required=True,
        type=int,
        help="dimension rows = build_size * domain_multiplier",
    )
    parser.add_argument(
        "--fact-rows",
        required=True,
        type=int,
        help="number of rows in the fact table",
    )
    parser.add_argument(
        "--cluster-counts",
        required=True,
        type=parse_cluster_counts,
        help="comma-separated cluster counts, e.g. 2,4,8,16",
    )
    parser.add_argument(
        "--dummy-string-cols",
        type=int,
        default=2,
        help="number of dummy string columns in the dimension table",
    )
    parser.add_argument(
        "--dummy-string-len",
        type=int,
        default=24,
        help="length of each dummy string column",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used for reproducible random foreign keys",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing DuckDB file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spec = BenchmarkSpec(
        db_path=args.db,
        queries_dir=args.queries_dir,
        build_size=args.build_size,
        domain_multiplier=args.domain_multiplier,
        fact_rows=args.fact_rows,
        cluster_counts=tuple(args.cluster_counts),
        dummy_string_cols=args.dummy_string_cols,
        dummy_string_len=args.dummy_string_len,
        seed=args.seed,
        force=args.force,
    )
    spec.validate()

    metadata = write_database(spec)
    query_paths = write_queries(spec)

    summary = {
        "db_path": str(spec.db_path),
        "queries_dir": str(spec.queries_dir),
        "query_count": len(query_paths),
        "cluster_counts": list(spec.cluster_counts),
        "dim_rows": spec.dim_rows,
        "fact_rows": spec.fact_rows,
        "metadata_path": str(spec.queries_dir / "metadata.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

