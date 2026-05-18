from __future__ import annotations

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


def load_queries(queries_dir: Path) -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    for path in sorted(queries_dir.rglob("*.sql")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(queries_dir)
        match = GENERATED_QUERY_RE.match(path.name)
        join_variant = match.group(1) if match else None
        cluster_count = int(match.group(2)) if match else None
        queries.append(
            QuerySpec(
                name=relative_path.as_posix(),
                path=path.resolve(),
                relative_path=relative_path,
                join_variant=join_variant,
                cluster_count=cluster_count,
            )
        )
    if not queries:
        raise FileNotFoundError(f"no benchmark queries found in {queries_dir}")
    return queries
