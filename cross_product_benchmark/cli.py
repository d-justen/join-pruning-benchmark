from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import write_database
from .query_gen import write_queries
from .spec import CrossProductBenchmarkSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DuckDB benchmark with the cross-product of build size, "
            "selectivity, predicate type, and probe clustering."
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
        "--fact-rows",
        type=int,
        default=100_000_000,
        help="number of rows in the fact table",
    )
    parser.add_argument(
        "--dummy-string-cols",
        type=int,
        default=2,
        help="number of dummy string columns in the dimension tables",
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
        help="seed used for reproducible scattered keys and uniform foreign keys",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing DuckDB file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = CrossProductBenchmarkSpec(
        db_path=args.db,
        queries_dir=args.queries_dir,
        fact_rows=args.fact_rows,
        dummy_string_cols=args.dummy_string_cols,
        dummy_string_len=args.dummy_string_len,
        seed=args.seed,
        force=args.force,
    )
    spec.validate()

    metadata = write_database(spec)
    query_paths = write_queries(spec)
    metadata["queries"] = [path.name for path in query_paths]
    metadata_path = spec.queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    summary = {
        "db_path": str(spec.db_path),
        "queries_dir": str(spec.queries_dir),
        "query_count": len(query_paths),
        "build_sizes": list(spec.build_sizes),
        "selectivities": list(spec.selectivities),
        "predicate_types": list(spec.predicate_types),
        "probe_clusterings": list(spec.probe_clusterings),
        "fact_rows": spec.fact_rows,
        "metadata_path": str(metadata_path),
        "dimension_tables": sorted(metadata["dimensions"].keys()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
