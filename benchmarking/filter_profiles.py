"""Comparable elapsed join/filter phases from standard DuckDB JSON profiles.

The build interval includes hash-table finalization and any PRF/Bloom build,
but excludes the separately timed PRF analysis/compression. It is not an
isolated filter cost. Worker counters and legacy PRF spans have different
boundaries and must never be substituted for these timings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


PHASE_FIELDS = {
    "join_finalize_ms": "join_finalize_elapsed_seconds",
    "join_finalize_build_ms": "join_finalize_build_elapsed_seconds",
    "prf_analysis_ms": "prf_analysis_elapsed_seconds",
    "prf_compression_ms": "prf_compression_elapsed_seconds",
}
COUNT_FIELDS = {
    "join_finalize_count": "finalize_count",
    "prf_build_count": "prf_build_count",
    "bloom_build_count": "bloom_build_count",
    "perfect_hash_join_count": "perfect_hash_join_count",
    "external_finalize_count": "external_finalize_count",
}
FILTER_PROFILE_COLUMNS = {
    **{field: "DOUBLE" for field in PHASE_FIELDS},
    **{field: "BIGINT" for field in COUNT_FIELDS},
}


@dataclass(frozen=True)
class JoinFilterMetrics:
    join_finalize_ms: float | None = None
    join_finalize_build_ms: float | None = None
    prf_analysis_ms: float | None = None
    prf_compression_ms: float | None = None
    join_finalize_count: int | None = None
    prf_build_count: int | None = None
    bloom_build_count: int | None = None
    perfect_hash_join_count: int | None = None
    external_finalize_count: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_join_filter_metrics(profile: dict) -> JoinFilterMetrics:
    """Sum completed operator snapshots, including repeated finalizations.

    Missing/invalid values remain unknown, including in a partially recorded
    multi-join profile. An explicit zero means the phase did not execute.
    A PRF with Bloom fallback contributes one build interval and both counts.
    """
    records = []
    stack = [profile]
    while stack:
        node = stack.pop()
        stack.extend(node.get("children", []))
        if "join_filter" in node:
            record = node["join_filter"]
            records.append(record if isinstance(record, dict) else {})
    if not records:
        return JoinFilterMetrics()

    def total(field: str, count: bool = False):
        values = [record.get(field) for record in records]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0
               or (count and value != int(value)) for value in values):
            return None
        return int(sum(values)) if count else math.fsum(values) * 1000.0

    return JoinFilterMetrics(
        **{name: total(field) for name, field in PHASE_FIELDS.items()},
        **{name: total(field, count=True) for name, field in COUNT_FIELDS.items()},
    )


def ensure_filter_profile_columns(con) -> None:
    for name, sql_type in FILTER_PROFILE_COLUMNS.items():
        con.execute(f"ALTER TABLE profile_runs ADD COLUMN IF NOT EXISTS {name} {sql_type}")


def store_join_filter_metrics(con, run_id: str, profile_path: str, metrics: JoinFilterMetrics) -> None:
    values = metrics.to_dict()
    assignments = ", ".join(f"{name} = ?" for name in values)
    con.execute(f"UPDATE profile_runs SET {assignments} WHERE run_id = ? AND profile_path = ?",
                [*values.values(), run_id, profile_path])
