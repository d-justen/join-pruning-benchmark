from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def parse_cluster_counts(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"invalid cluster count: {item!r}") from exc
    if not values:
        raise ValueError("at least one cluster count is required")
    return sorted(set(values))


@dataclass(frozen=True)
class BenchmarkSpec:
    db_path: Path
    queries_dir: Path
    build_size: int
    domain_multiplier: int
    fact_rows: int
    cluster_counts: tuple[int, ...]
    dummy_string_cols: int
    dummy_string_len: int
    seed: int
    force: bool = False

    @property
    def dim_rows(self) -> int:
        return self.build_size * self.domain_multiplier

    def validate(self) -> None:
        if self.build_size <= 0:
            raise ValueError("--build-size must be positive")
        if self.domain_multiplier <= 0:
            raise ValueError("--domain-multiplier must be positive")
        if self.fact_rows <= 0:
            raise ValueError("--fact-rows must be positive")
        if self.dummy_string_cols < 0:
            raise ValueError("--dummy-string-cols cannot be negative")
        if self.dummy_string_len <= 0:
            raise ValueError("--dummy-string-len must be positive")
        if self.dim_rows < self.build_size:
            raise ValueError("dimension cardinality must be at least build size")
        for cluster_count in self.cluster_counts:
            if cluster_count < 2:
                raise ValueError("cluster counts must be at least 2")
            if cluster_count > self.build_size:
                raise ValueError(
                    f"cluster count {cluster_count} exceeds build size {self.build_size}"
                )

