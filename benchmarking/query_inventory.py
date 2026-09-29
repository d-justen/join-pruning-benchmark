from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


GENERATED_QUERY_RE = re.compile(r"^(ordered|uniform)_cc_(\d+)\.sql$")


@dataclass(frozen=True)
class QuerySpec:
    name: str
    path: Path
    relative_path: Path
    join_variant: str | None
    cluster_count: int | None
    parameters: dict


def load_queries(queries_dir: Path) -> list[QuerySpec]:
    catalog_path = queries_dir / "query_catalog.json"
    catalog: dict[str, dict] = {}
    if catalog_path.exists():
        raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = raw_catalog.get("queries", raw_catalog)
        if not isinstance(catalog, dict):
            raise ValueError(f"query catalog must contain an object: {catalog_path}")
    queries: list[QuerySpec] = []
    for path in sorted(queries_dir.rglob("*.sql")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(queries_dir)
        match = GENERATED_QUERY_RE.match(path.name)
        parameters = catalog.get(relative_path.as_posix(), catalog.get(path.name, {}))
        if not isinstance(parameters, dict):
            raise ValueError(f"query parameters must be an object for {relative_path}")
        join_variant = parameters.get("probe_layout", match.group(1) if match else None)
        cluster_count_raw = parameters.get(
            "cluster_count",
            int(match.group(2)) if match else None,
        )
        cluster_count = None if cluster_count_raw is None else int(cluster_count_raw)
        queries.append(
            QuerySpec(
                name=relative_path.as_posix(),
                path=path.resolve(),
                relative_path=relative_path,
                join_variant=join_variant,
                cluster_count=cluster_count,
                parameters=parameters,
            )
        )
    if not queries:
        raise FileNotFoundError(f"no benchmark queries found in {queries_dir}")
    return queries
