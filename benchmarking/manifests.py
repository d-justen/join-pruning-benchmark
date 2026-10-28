from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER = "__DUCKLAKE_HTTP_BASE_URL__"
DUCKLAKE_S3_BASE_URL_PLACEHOLDER = "__DUCKLAKE_S3_BASE_URL__"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PreparedWorkload:
    name: str
    kind: str
    source_root: Path
    artifact_dir: Path
    db_path: Path
    queries_dir: Path
    metadata_path: Path | None
    init_sql_paths: tuple[Path, ...]
    prepared_at: str
    source_db_path: Path | None = None
    db_handling: str | None = None
    setup_sql_paths: tuple[Path, ...] = ()
    post_variant_setup_sql: tuple[str, ...] = ()
    storage_mode: str = "duckdb"
    remote_http_parquet_latency_ms: int | None = None
    remote_parquet_catalog_path: Path | None = None
    parquet_catalog_path: Path | None = None
    ducklake_bootstrap_sql_path: Path | None = None
    ducklake_metadata_path: Path | None = None
    ducklake_data_path: Path | None = None
    ducklake_catalog_alias: str | None = None
    ducklake_http_latency_ms: int | None = None
    definition_path: Path | None = None
    cases_path: Path | None = None
    query_catalog_path: Path | None = None
    ducklake_data_source: str = "local"
    ducklake_s3_url: str | None = None


def manifest_to_dict(manifest: PreparedWorkload) -> dict:
    data = asdict(manifest)
    data["source_root"] = str(manifest.source_root)
    data["artifact_dir"] = str(manifest.artifact_dir)
    data["db_path"] = str(manifest.db_path)
    data["source_db_path"] = (
        str(manifest.source_db_path) if manifest.source_db_path is not None else None
    )
    data["queries_dir"] = str(manifest.queries_dir)
    data["metadata_path"] = (
        str(manifest.metadata_path) if manifest.metadata_path is not None else None
    )
    data["init_sql_paths"] = [str(path) for path in manifest.init_sql_paths]
    data["db_handling"] = manifest.db_handling
    data["setup_sql_paths"] = [str(path) for path in manifest.setup_sql_paths]
    data["storage_mode"] = manifest.storage_mode
    data["remote_http_parquet_latency_ms"] = manifest.remote_http_parquet_latency_ms
    data["remote_parquet_catalog_path"] = (
        str(manifest.remote_parquet_catalog_path)
        if manifest.remote_parquet_catalog_path is not None
        else None
    )
    for field_name in (
        "parquet_catalog_path",
        "ducklake_bootstrap_sql_path",
        "ducklake_metadata_path",
        "ducklake_data_path",
        "definition_path",
        "cases_path",
        "query_catalog_path",
    ):
        field_value = getattr(manifest, field_name)
        data[field_name] = str(field_value) if field_value is not None else None
    return data


def write_manifest(path: Path, manifest: PreparedWorkload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_optional_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw).resolve()


def load_manifest(path: Path) -> PreparedWorkload:
    manifest_path = path.resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Suite exports use paths relative to this manifest; historical absolute
    # manifests remain supported.
    for key, value in raw.items():
        if key.endswith("_path") or key.endswith("_dir") or key in ("source_root",):
            if value is not None and not Path(value).is_absolute():
                raw[key] = str(manifest_path.parent / value)
        elif key.endswith("_paths"):
            raw[key] = [str(manifest_path.parent / p) if not Path(p).is_absolute() else p for p in value]
    return PreparedWorkload(
        name=raw["name"],
        kind=raw["kind"],
        source_root=Path(raw["source_root"]).resolve(),
        artifact_dir=Path(raw["artifact_dir"]).resolve(),
        db_path=Path(raw["db_path"]).resolve(),
        source_db_path=_resolve_optional_path(raw.get("source_db_path")),
        queries_dir=Path(raw["queries_dir"]).resolve(),
        metadata_path=_resolve_optional_path(raw.get("metadata_path")),
        init_sql_paths=tuple(Path(item).resolve() for item in raw.get("init_sql_paths", [])),
        db_handling=raw.get("db_handling"),
        setup_sql_paths=tuple(Path(item).resolve() for item in raw.get("setup_sql_paths", [])),
        post_variant_setup_sql=tuple(raw.get("post_variant_setup_sql", [])),
        storage_mode=raw.get("storage_mode", "duckdb"),
        remote_http_parquet_latency_ms=raw.get("remote_http_parquet_latency_ms"),
        remote_parquet_catalog_path=_resolve_optional_path(raw.get("remote_parquet_catalog_path")),
        parquet_catalog_path=_resolve_optional_path(raw.get("parquet_catalog_path")),
        ducklake_bootstrap_sql_path=_resolve_optional_path(
            raw.get("ducklake_bootstrap_sql_path")
        ),
        ducklake_metadata_path=_resolve_optional_path(raw.get("ducklake_metadata_path")),
        ducklake_data_path=_resolve_optional_path(raw.get("ducklake_data_path")),
        ducklake_catalog_alias=raw.get("ducklake_catalog_alias"),
        ducklake_http_latency_ms=raw.get("ducklake_http_latency_ms"),
        ducklake_data_source=raw.get("ducklake_data_source", "local"),
        ducklake_s3_url=raw.get("ducklake_s3_url"),
        definition_path=_resolve_optional_path(raw.get("definition_path")),
        cases_path=_resolve_optional_path(raw.get("cases_path")),
        query_catalog_path=_resolve_optional_path(raw.get("query_catalog_path")),
        prepared_at=raw["prepared_at"],
    )
