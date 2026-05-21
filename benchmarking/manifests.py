from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    remote_http_parquet_latency_ms: int | None = None
    remote_parquet_catalog_path: Path | None = None


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
    data["remote_http_parquet_latency_ms"] = manifest.remote_http_parquet_latency_ms
    data["remote_parquet_catalog_path"] = (
        str(manifest.remote_parquet_catalog_path)
        if manifest.remote_parquet_catalog_path is not None
        else None
    )
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
        remote_http_parquet_latency_ms=raw.get("remote_http_parquet_latency_ms"),
        remote_parquet_catalog_path=_resolve_optional_path(raw.get("remote_parquet_catalog_path")),
        prepared_at=raw["prepared_at"],
    )
