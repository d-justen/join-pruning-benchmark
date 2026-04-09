from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from benchmark_manifest import BenchmarkManifest, utc_now_iso, write_manifest


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


def resolve_binary(binary: Path) -> Path:
    if binary.is_absolute() or binary.parent != Path("."):
        resolved = binary.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"DuckDB binary not found: {binary}")
        return resolved

    candidate = shutil.which(str(binary))
    if candidate is None:
        raise FileNotFoundError(f"cannot find DuckDB binary on PATH: {binary}")
    return Path(candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a benchmark manifest under out/benchmarks/<name>."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/benchmarks"),
        help="directory that receives prepared benchmark manifests and databases",
    )
    parser.add_argument(
        "--system-duckdb",
        type=Path,
        default=Path("duckdb"),
        help="DuckDB CLI used to initialize external benchmark databases",
    )
    parser.add_argument(
        "--force-init",
        action="store_true",
        help="recreate the prepared database for external benchmarks",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    generated = subparsers.add_parser("generated", help="prepare an existing generated benchmark")
    generated.add_argument("--benchmark-name", required=True)
    generated.add_argument("--db", required=True, type=Path)
    generated.add_argument("--queries-dir", required=True, type=Path)
    generated.add_argument("--metadata-path", type=Path)

    external = subparsers.add_parser("external", help="prepare an external benchmark source")
    external.add_argument("--external-benchmark-dir", required=True, type=Path)

    return parser


def discover_external_layout(external_dir: Path) -> tuple[str, Path, Path | None, list[Path]]:
    config_path = external_dir / "source.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        benchmark_name = raw.get("benchmark_name", external_dir.name)
        queries_dir = resolve_source_path(config_path.parent, raw["queries_dir"])
        metadata_path = resolve_source_optional_path(
            config_path.parent, raw.get("metadata_path")
        )
        init_sql_paths = [
            resolve_source_path(config_path.parent, raw_path)
            for raw_path in raw.get("init_sql_paths", [])
        ]
        return benchmark_name, queries_dir, metadata_path, init_sql_paths

    benchmark_name = external_dir.name
    queries_dir = external_dir / "queries"
    if not queries_dir.exists():
        raise FileNotFoundError(
            f"external benchmark {external_dir} must contain queries/ or source.json"
        )

    metadata_path = external_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path = None

    init_sql_paths: list[Path] = []
    schema_candidates = [external_dir / "schema.sql", external_dir / "init" / "schema.sql"]
    load_candidates = [external_dir / "load.sql", external_dir / "init" / "load.sql"]

    for candidate in schema_candidates:
        if candidate.exists():
            init_sql_paths.append(candidate.resolve())
            break
    for candidate in load_candidates:
        if candidate.exists():
            init_sql_paths.append(candidate.resolve())
            break
    if not init_sql_paths:
        raise FileNotFoundError(
            f"external benchmark {external_dir} must provide load.sql, init/load.sql, or source.json"
        )

    return benchmark_name, queries_dir.resolve(), metadata_path.resolve() if metadata_path else None, init_sql_paths


def resolve_source_path(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_source_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return resolve_source_path(base_dir, raw)


def initialize_external_db(
    duckdb_binary: Path,
    db_path: Path,
    init_sql_paths: list[Path],
    *,
    force_init: bool,
) -> None:
    if db_path.exists():
        if not force_init:
            return
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for sql_path in init_sql_paths:
        run_command([str(duckdb_binary), str(db_path), "-f", str(sql_path)], cwd=sql_path.parent)


def prepare_generated(args: argparse.Namespace) -> Path:
    benchmark_name = args.benchmark_name
    output_dir = (args.output_root / benchmark_name).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = BenchmarkManifest(
        benchmark_name=benchmark_name,
        source_type="generated",
        source_root=args.db.resolve().parent,
        db_path=args.db.resolve(),
        queries_dir=args.queries_dir.resolve(),
        metadata_path=args.metadata_path.resolve() if args.metadata_path else None,
        init_sql_paths=(),
        prepared_at=utc_now_iso(),
    )
    write_manifest(manifest_path, manifest)
    return manifest_path


def prepare_external(args: argparse.Namespace) -> Path:
    external_dir = args.external_benchmark_dir.resolve()
    benchmark_name, queries_dir, metadata_path, init_sql_paths = discover_external_layout(
        external_dir
    )
    output_dir = (args.output_root / benchmark_name).resolve()
    manifest_path = output_dir / "manifest.json"
    db_path = output_dir / "benchmark.duckdb"
    duckdb_binary = resolve_binary(args.system_duckdb)

    initialize_external_db(
        duckdb_binary,
        db_path,
        init_sql_paths,
        force_init=args.force_init,
    )

    manifest = BenchmarkManifest(
        benchmark_name=benchmark_name,
        source_type="external",
        source_root=external_dir,
        db_path=db_path,
        queries_dir=queries_dir,
        metadata_path=metadata_path,
        init_sql_paths=tuple(path.resolve() for path in init_sql_paths),
        prepared_at=utc_now_iso(),
    )
    write_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "generated":
        manifest_path = prepare_generated(args)
    else:
        manifest_path = prepare_external(args)
    print(manifest_path)


if __name__ == "__main__":
    main()
