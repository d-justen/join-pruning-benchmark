from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.workloads.external import DEFAULT_EXTERNAL_DB_FILENAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a local DuckDB snapshot for the join-order benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_EXTERNAL_DB_FILENAME,
        help="path to the output DuckDB database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output database if it already exists",
    )
    return parser


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def _parse_load_statements(load_sql_path: Path) -> list[tuple[str, str]]:
    statements: list[tuple[str, str]] = []
    for raw_line in load_sql_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        prefix = "INSERT INTO "
        middle = " SELECT * FROM '"
        suffix = "';"
        if not line.startswith(prefix) or middle not in line or not line.endswith(suffix):
            raise ValueError(f"unsupported load statement: {line}")
        table_name, remainder = line[len(prefix) :].split(middle, 1)
        url = remainder[: -len(suffix)]
        statements.append((table_name.strip(), url))
    if not statements:
        raise ValueError(f"no load statements found in {load_sql_path}")
    return statements


def _initialize_schema(con: duckdb.DuckDBPyConnection, schema_sql_path: Path) -> None:
    con.execute(schema_sql_path.read_text(encoding="utf-8"))


def _load_parquet_tables(
    con: duckdb.DuckDBPyConnection,
    load_sql_path: Path,
    download_dir: Path,
) -> None:
    for table_name, url in _parse_load_statements(load_sql_path):
        parquet_path = download_dir / f"{table_name}.parquet"
        print(f"downloading {table_name} from {url}", flush=True)
        _download_file(url, parquet_path)
        print(f"loading {table_name}", flush=True)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM read_parquet(?)", [str(parquet_path)])


def main() -> None:
    args = build_parser().parse_args()
    workload_dir = Path(__file__).resolve().parent
    output_path = args.output.resolve()
    if output_path.exists():
        if not args.force:
            raise SystemExit(
                f"output database already exists: {output_path}\n"
                "rerun with --force to overwrite it"
            )
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output_path))
    try:
        _initialize_schema(con, workload_dir / "schema.sql")
        with TemporaryDirectory(prefix="job-parquet-") as temporary_dir:
            _load_parquet_tables(
                con,
                workload_dir / "load.sql",
                Path(temporary_dir),
            )
    finally:
        con.close()
    print(output_path)


if __name__ == "__main__":
    main()
