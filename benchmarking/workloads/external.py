from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb

from benchmarking.resources import connect

from benchmarking.experiment_config import ExternalWorkloadConfig
from benchmarking.manifests import (
    DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER,
    PreparedWorkload,
    utc_now_iso,
    write_manifest,
)


DEFAULT_REMOTE_PARQUET_FILE_SIZE_MB = 200
DEFAULT_REMOTE_PARQUET_COMPRESSION = "zstd"
DEFAULT_EXTERNAL_DB_FILENAME = "benchmark.duckdb"


@dataclass(frozen=True)
class ExternalSourceLayout:
    benchmark_name: str
    queries_dir: Path
    metadata_path: Path | None
    init_sql_paths: tuple[Path, ...]
    source_db_path: Path | None
    db_handling: str | None
    remote_http_parquet_latency_ms: int | None


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parse_remote_http_parquet(raw: object, config_path: Path) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"remote_http_parquet in {config_path} must be an object")
    latency_ms = raw.get("latency_ms")
    if not isinstance(latency_ms, int) or isinstance(latency_ms, bool):
        raise ValueError(f"remote_http_parquet.latency_ms in {config_path} must be an integer")
    if latency_ms < 0:
        raise ValueError(f"remote_http_parquet.latency_ms in {config_path} must be non-negative")
    return latency_ms


def _resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return (base_dir / raw).resolve()


def _discover_default_source_db_path(external_dir: Path) -> Path | None:
    candidate = (external_dir / DEFAULT_EXTERNAL_DB_FILENAME).resolve()
    if candidate.exists():
        return candidate
    return None


def list_base_tables(source_db_path: Path) -> list[str]:
    con = connect(str(source_db_path))
    try:
        return [
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        ]
    finally:
        con.close()


def discover_external_layout(external_dir: Path) -> ExternalSourceLayout:
    config_path = external_dir / "source.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        benchmark_name = raw.get("benchmark_name", external_dir.name)
        queries_dir = (config_path.parent / raw["queries_dir"]).resolve()
        metadata_path = _resolve_optional_path(config_path.parent, raw.get("metadata_path"))
        init_sql_paths = tuple(
            (config_path.parent / raw_path).resolve()
            for raw_path in raw.get("init_sql_paths", [])
        )
        source_db_path = _resolve_optional_path(config_path.parent, raw.get("database_path"))
        db_handling = raw.get("database_handling")
        remote_http_parquet_latency_ms = _parse_remote_http_parquet(
            raw.get("remote_http_parquet"),
            config_path,
        )
        default_source_db_path = _discover_default_source_db_path(config_path.parent)
        if db_handling not in (None, "reuse", "copy"):
            raise ValueError(
                f"unsupported database_handling in {config_path}: {db_handling!r}"
            )
        if source_db_path is not None and not source_db_path.exists():
            raise FileNotFoundError(
                f"external workload database not found: {source_db_path}"
            )
        if source_db_path is not None and init_sql_paths:
            raise ValueError(
                f"external workload {config_path} cannot define both database_path and init_sql_paths"
            )
        if source_db_path is not None:
            return ExternalSourceLayout(
                benchmark_name=benchmark_name,
                queries_dir=queries_dir,
                metadata_path=metadata_path,
                init_sql_paths=(),
                source_db_path=source_db_path,
                db_handling=db_handling or "reuse",
                remote_http_parquet_latency_ms=remote_http_parquet_latency_ms,
            )
        if default_source_db_path is not None:
            return ExternalSourceLayout(
                benchmark_name=benchmark_name,
                queries_dir=queries_dir,
                metadata_path=metadata_path,
                init_sql_paths=(),
                source_db_path=default_source_db_path,
                db_handling="reuse",
                remote_http_parquet_latency_ms=remote_http_parquet_latency_ms,
            )
        if not init_sql_paths:
            raise FileNotFoundError(
                f"external workload {external_dir} must define database_path or init_sql_paths"
            )
        return ExternalSourceLayout(
            benchmark_name=benchmark_name,
            queries_dir=queries_dir,
            metadata_path=metadata_path,
            init_sql_paths=init_sql_paths,
            source_db_path=None,
            db_handling=None,
            remote_http_parquet_latency_ms=remote_http_parquet_latency_ms,
        )
    queries_dir = (external_dir / "queries").resolve()
    if not queries_dir.exists():
        raise FileNotFoundError(
            f"external workload {external_dir} must contain queries/ or source.json"
        )
    metadata_path = (external_dir / "metadata.json").resolve()
    if not metadata_path.exists():
        metadata_path = None
    init_sql_paths: list[Path] = []
    for candidate in (
        external_dir / "schema.sql",
        external_dir / "init" / "schema.sql",
        external_dir / "load.sql",
        external_dir / "init" / "load.sql",
    ):
        if candidate.exists():
            init_sql_paths.append(candidate.resolve())
    if not init_sql_paths:
        raise FileNotFoundError(f"external workload {external_dir} has no init SQL")
    return ExternalSourceLayout(
        benchmark_name=external_dir.name,
        queries_dir=queries_dir,
        metadata_path=metadata_path,
        init_sql_paths=tuple(init_sql_paths),
        source_db_path=None,
        db_handling=None,
        remote_http_parquet_latency_ms=None,
    )


def initialize_external_db(
    db_path: Path,
    init_sql_paths: list[Path],
) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(str(db_path))
    try:
        for sql_path in init_sql_paths:
            script = sql_path.read_text(encoding="utf-8")
            previous_cwd = Path.cwd()
            try:
                os.chdir(sql_path.parent)
                con.execute(script)
            except Exception as exc:
                message = str(exc)
                if "http" in message.lower():
                    message = (
                        f"{message}\n"
                        f"hint: place a prebuilt DuckDB snapshot at "
                        f"{sql_path.parent / DEFAULT_EXTERNAL_DB_FILENAME} "
                        "to reuse the workload offline"
                    )
                raise RuntimeError(
                    "command failed\n"
                    f"cwd: {sql_path.parent}\n"
                    f"command: python duckdb execute {sql_path}\n"
                    f"stderr:\n{message}"
                ) from exc
            finally:
                os.chdir(previous_cwd)
    finally:
        con.close()


def materialize_existing_db(source_db_path: Path, artifact_dir: Path, db_handling: str) -> Path:
    if db_handling == "reuse":
        return source_db_path.resolve()
    if db_handling != "copy":
        raise ValueError(f"unsupported database handling mode: {db_handling}")
    db_path = artifact_dir / "benchmark.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db_path, db_path)
    return db_path.resolve()


def export_parquet_catalog(
    source_db_path: Path,
    artifact_dir: Path,
    *,
    root_dir_name: str,
    catalog_filename: str,
    file_size_mb: int = DEFAULT_REMOTE_PARQUET_FILE_SIZE_MB,
    exclude_empty_files: bool = False,
    export_threads: int | None = None,
    row_groups_per_file: int | None = None,
    row_group_size: int = 122_880,
) -> Path:
    parquet_root = artifact_dir / root_dir_name
    if parquet_root.exists():
        shutil.rmtree(parquet_root)
    parquet_root.mkdir(parents=True, exist_ok=True)

    con = connect(str(source_db_path))
    try:
        if export_threads is not None:
            if export_threads <= 0:
                raise ValueError("export_threads must be positive")
            con.execute(f"SET threads = {export_threads}")
        con.execute("SET preserve_insertion_order = true")
        tables = [
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        if not tables:
            raise ValueError(f"external workload database has no base tables: {source_db_path}")
        catalog: dict[str, list[str]] = {}
        for table_name in tables:
            table_dir = parquet_root / table_name
            layout = (f"ROW_GROUP_SIZE {row_group_size}, ROW_GROUPS_PER_FILE {row_groups_per_file}"
                      if row_groups_per_file is not None else f"FILE_SIZE_BYTES '{file_size_mb}MB'")
            con.execute(
                f"""
                COPY {_quote_identifier(table_name)}
                TO ?
                (FORMAT parquet, COMPRESSION {DEFAULT_REMOTE_PARQUET_COMPRESSION}, {layout})
                """,
                [str(table_dir)],
            )
            exported_paths = sorted(table_dir.rglob("*.parquet"))
            retained_paths = set(exported_paths)
            if exclude_empty_files:
                nonempty_paths = {
                    Path(row[0]).resolve()
                    for row in con.execute(
                        "SELECT DISTINCT file_name FROM parquet_metadata(?)",
                        [str(table_dir.resolve() / "*.parquet")],
                    ).fetchall()
                }
                retained_paths = {
                    path for path in exported_paths if path.resolve() in nonempty_paths
                }
                for path in exported_paths:
                    if path not in retained_paths:
                        path.unlink()
            parquet_files = [
                path.relative_to(parquet_root).as_posix()
                for path in exported_paths
                if path in retained_paths
            ]
            if not parquet_files:
                raise ValueError(f"no Parquet files exported for table {table_name}")
            catalog[table_name] = parquet_files
    finally:
        con.close()

    catalog_path = artifact_dir / catalog_filename
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog_path.resolve()


def read_table_schemas(db_path: Path) -> dict[str, list[tuple[str, str]]]:
    con = connect(str(db_path), read_only=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        return {
            table_name: [
                (row[0], row[1])
                for row in con.execute(
                    f"DESCRIBE {_quote_identifier(table_name)}"
                ).fetchall()
            ]
            for table_name in tables
        }
    finally:
        con.close()


def write_ducklake_sql(
    artifact_dir: Path,
    parquet_catalog_path: Path,
    table_schemas: dict[str, list[tuple[str, str]]],
    *,
    use_http_data: bool,
) -> tuple[Path, Path, Path, Path, Path, str]:
    catalog_alias = "ducklake"
    metadata_path = (artifact_dir / "ducklake_metadata.duckdb").resolve()
    data_path = (artifact_dir / "ducklake_data").resolve()
    runtime_db_path = (artifact_dir / "ducklake_runtime.duckdb").resolve()
    bootstrap_path = (artifact_dir / "ducklake_bootstrap.sql").resolve()
    setup_path = (artifact_dir / "ducklake_setup.sql").resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(parquet_catalog_path.read_text(encoding="utf-8"))
    parquet_root = parquet_catalog_path.parent / "parquet"
    attach_sql = (
        f"ATTACH {_sql_literal('ducklake:' + str(metadata_path))} "
        f"AS {_quote_identifier(catalog_alias)} "
        f"(DATA_PATH {_sql_literal(str(data_path))}, DATA_INLINING_ROW_LIMIT 0);"
    )
    http_setup = (
        [
            "LOAD httpfs;",
            "SET enable_external_file_cache=false;",
            "SET enable_http_metadata_cache=false;",
        ]
        if use_http_data
        else []
    )
    bootstrap_statements = [
        "-- Generated by run_benchmarks.py",
        *http_setup,
        "LOAD ducklake;",
        attach_sql,
    ]
    setup_statements = [
        "-- Generated by run_benchmarks.py",
        *http_setup,
        "LOAD ducklake;",
        attach_sql,
    ]
    for table_name in sorted(catalog):
        if table_name not in table_schemas:
            raise ValueError(f"missing schema for exported table {table_name}")
        column_sql = ", ".join(
            f"{_quote_identifier(column_name)} {column_type}"
            for column_name, column_type in table_schemas[table_name]
        )
        bootstrap_statements.append(
            f"CREATE TABLE {_quote_identifier(catalog_alias)}.{_quote_identifier(table_name)} "
            f"({column_sql});"
        )
        data_files = [
            (
                f"{DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER}/{relative_path}"
                if use_http_data
                else str((parquet_root / relative_path).resolve())
            )
            for relative_path in catalog[table_name]
        ]
        file_list_sql = ", ".join(_sql_literal(path) for path in data_files)
        bootstrap_statements.append(
            f"CALL ducklake_add_data_files({_sql_literal(catalog_alias)}, "
            f"{_sql_literal(table_name)}, [{file_list_sql}]);"
        )
        setup_statements.append(
            f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(table_name)} AS "
            f"SELECT * FROM {_quote_identifier(catalog_alias)}.{_quote_identifier(table_name)};"
        )
    bootstrap_path.write_text("\n".join(bootstrap_statements) + "\n", encoding="utf-8")
    setup_path.write_text("\n".join(setup_statements) + "\n", encoding="utf-8")
    return (
        runtime_db_path,
        bootstrap_path,
        setup_path,
        metadata_path,
        data_path,
        catalog_alias,
    )


def export_remote_parquet_catalog(source_db_path: Path, artifact_dir: Path) -> Path:
    return export_parquet_catalog(
        source_db_path,
        artifact_dir,
        root_dir_name="remote_parquet",
        catalog_filename="remote_parquet_catalog.json",
    )


def prepare_external_workload(
    spec: ExternalWorkloadConfig,
    artifact_dir: Path,
) -> PreparedWorkload:
    layout = discover_external_layout(spec.source_dir)
    source_db_path = layout.source_db_path
    db_handling = layout.db_handling
    if source_db_path is not None:
        db_path = materialize_existing_db(source_db_path, artifact_dir, db_handling)
        init_sql_paths: tuple[Path, ...] = ()
    else:
        db_path = artifact_dir / "benchmark.duckdb"
        initialize_external_db(db_path, list(layout.init_sql_paths))
        db_path = db_path.resolve()
        init_sql_paths = layout.init_sql_paths
    queries_dir = layout.queries_dir.resolve()
    if spec.query_names is not None:
        query_paths: list[Path] = []
        for query_name in spec.query_names:
            query_path = layout.queries_dir / query_name
            if not query_path.is_file():
                raise ValueError(
                    f"query not found for external workload {spec.name}: {query_name}"
                )
            query_paths.append(query_path)
        subset_dir = artifact_dir / "queries"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for query_path in query_paths:
            shutil.copy2(query_path, subset_dir / query_path.name)
        queries_dir = subset_dir.resolve()
    setup_sql_paths: tuple[Path, ...] = ()
    if spec.setup_sql:
        setup_sql_path = (artifact_dir / "setup.sql").resolve()
        setup_sql_path.write_text("\n".join(spec.setup_sql) + "\n", encoding="utf-8")
        setup_sql_paths = (setup_sql_path,)
    remote_parquet_catalog_path: Path | None = None
    parquet_catalog_path: Path | None = None
    ducklake_bootstrap_sql_path: Path | None = None
    ducklake_metadata_path: Path | None = None
    ducklake_data_path: Path | None = None
    ducklake_catalog_alias: str | None = None
    runtime_db_path = db_path
    storage_mode = "duckdb"
    if spec.storage_kind == "ducklake":
        if layout.remote_http_parquet_latency_ms is not None:
            raise ValueError(
                "external workload cannot combine source.json remote_http_parquet "
                "with config storage.kind='ducklake'"
            )
        table_schemas = read_table_schemas(db_path)
        parquet_catalog_path = export_parquet_catalog(
            db_path,
            artifact_dir,
            root_dir_name="parquet",
            catalog_filename="parquet_catalog.json",
            file_size_mb=spec.parquet_file_size_mb,
            exclude_empty_files=True,
            # A single export thread preserves the source table order, which is
            # part of the physical-layout experiment for TPC-H Skew.
            export_threads=1,
        )
        (
            runtime_db_path,
            ducklake_bootstrap_sql_path,
            ducklake_setup_sql_path,
            ducklake_metadata_path,
            ducklake_data_path,
            ducklake_catalog_alias,
        ) = write_ducklake_sql(
            artifact_dir,
            parquet_catalog_path,
            table_schemas,
            use_http_data=spec.ducklake_http_latency_ms is not None,
        )
        setup_sql_paths = (ducklake_setup_sql_path,) + setup_sql_paths
        storage_mode = "ducklake"
    elif layout.remote_http_parquet_latency_ms is not None:
        remote_parquet_catalog_path = export_remote_parquet_catalog(db_path, artifact_dir)
        runtime_db_path = (artifact_dir / "benchmark.duckdb").resolve()
        if runtime_db_path.exists():
            runtime_db_path.unlink()
        storage_mode = "remote_http_parquet"
    manifest = PreparedWorkload(
        name=spec.name,
        kind=spec.kind,
        source_root=spec.source_dir.resolve(),
        artifact_dir=artifact_dir.resolve(),
        db_path=runtime_db_path,
        source_db_path=source_db_path.resolve() if source_db_path is not None else None,
        queries_dir=queries_dir,
        metadata_path=layout.metadata_path.resolve() if layout.metadata_path is not None else None,
        init_sql_paths=tuple(path.resolve() for path in init_sql_paths),
        db_handling=db_handling,
        setup_sql_paths=setup_sql_paths,
        storage_mode=storage_mode,
        remote_http_parquet_latency_ms=layout.remote_http_parquet_latency_ms,
        remote_parquet_catalog_path=remote_parquet_catalog_path,
        parquet_catalog_path=parquet_catalog_path,
        ducklake_bootstrap_sql_path=ducklake_bootstrap_sql_path,
        ducklake_metadata_path=ducklake_metadata_path,
        ducklake_data_path=ducklake_data_path,
        ducklake_catalog_alias=ducklake_catalog_alias,
        ducklake_http_latency_ms=spec.ducklake_http_latency_ms,
        prepared_at=utc_now_iso(),
    )
    write_manifest(artifact_dir / "manifest.json", manifest)
    return manifest
