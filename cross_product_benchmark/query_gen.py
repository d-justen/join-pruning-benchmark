from __future__ import annotations

from pathlib import Path

from .spec import (
    CrossProductBenchmarkSpec,
    build_size_label,
    dimension_table_name,
    fact_fk_column_name,
    mask_column_name,
    selectivity_label,
)


def render_query(
    spec: CrossProductBenchmarkSpec,
    build_size: int,
    selectivity: float,
    predicate_type: str,
    probe_clustering: str,
) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d.dummy_s{index}) AS dummy_s{index}"
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_clause = ",\n".join(payload_exprs) if payload_exprs else "    COUNT(*) AS row_count"
    return f"""SELECT
{select_clause}
FROM fact AS f
JOIN {dimension_table_name(build_size)} AS d
  ON f.{fact_fk_column_name(build_size, probe_clustering)} = d.d_key
WHERE d.{mask_column_name(selectivity, predicate_type)};
"""


def query_filename(build_size: int, selectivity: float, predicate_type: str, probe_clustering: str) -> str:
    return (
        f"b{build_size_label(build_size)}_"
        f"{selectivity_label(selectivity)}_"
        f"{predicate_type}_"
        f"{probe_clustering}.sql"
    )


def write_queries(spec: CrossProductBenchmarkSpec) -> list[Path]:
    written: list[Path] = []
    for build_size in spec.build_sizes:
        for selectivity in spec.selectivities:
            for predicate_type in spec.predicate_types:
                for probe_clustering in spec.probe_clusterings:
                    path = spec.queries_dir / query_filename(
                        build_size,
                        selectivity,
                        predicate_type,
                        probe_clustering,
                    )
                    path.write_text(
                        render_query(
                            spec,
                            build_size=build_size,
                            selectivity=selectivity,
                            predicate_type=predicate_type,
                            probe_clustering=probe_clustering,
                        ),
                        encoding="utf-8",
                    )
                    written.append(path)
    return written

