from __future__ import annotations

from pathlib import Path

from benchmark_spec import BenchmarkSpec
from layout import mask_column_name


def render_query(spec: BenchmarkSpec, join_column: str, cluster_count: int) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d.dummy_s{index}) AS dummy_s{index}"
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_clause = ",\n".join(payload_exprs)
    mask_column = mask_column_name(cluster_count)
    return f"""SELECT
{select_clause}
FROM fact AS f
JOIN dim AS d
  ON f.{join_column} = d.d_key
WHERE d.{mask_column};
"""


def write_queries(spec: BenchmarkSpec) -> list[Path]:
    written: list[Path] = []
    for cluster_count in spec.cluster_counts:
        for join_variant, join_column in (
            ("ordered", "fk_ordered"),
            ("uniform", "fk_uniform"),
        ):
            filename = f"{join_variant}_cc_{cluster_count:05d}.sql"
            path = spec.queries_dir / filename
            path.write_text(
                render_query(spec, join_column=join_column, cluster_count=cluster_count),
                encoding="utf-8",
            )
            written.append(path)
    return written
