from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_name: str
    source_type: str
    source_root: Path
    db_path: Path
    queries_dir: Path
    metadata_path: Path | None
    init_sql_paths: tuple[Path, ...]
    prepared_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return resolve_path(base_dir, raw)


def resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def manifest_to_dict(manifest: BenchmarkManifest) -> dict:
    data = asdict(manifest)
    data["source_root"] = str(manifest.source_root)
    data["db_path"] = str(manifest.db_path)
    data["queries_dir"] = str(manifest.queries_dir)
    data["metadata_path"] = (
        str(manifest.metadata_path) if manifest.metadata_path is not None else None
    )
    data["init_sql_paths"] = [str(path) for path in manifest.init_sql_paths]
    return data


def write_manifest(path: Path, manifest: BenchmarkManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> BenchmarkManifest:
    manifest_path = path.resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    return BenchmarkManifest(
        benchmark_name=raw["benchmark_name"],
        source_type=raw["source_type"],
        source_root=resolve_path(base_dir, raw["source_root"]),
        db_path=resolve_path(base_dir, raw["db_path"]),
        queries_dir=resolve_path(base_dir, raw["queries_dir"]),
        metadata_path=resolve_optional_path(base_dir, raw.get("metadata_path")),
        init_sql_paths=tuple(
            resolve_path(base_dir, raw_path) for raw_path in raw.get("init_sql_paths", [])
        ),
        prepared_at=raw["prepared_at"],
    )
