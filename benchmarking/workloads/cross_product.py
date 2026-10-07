from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from benchmarking.experiment_config import (
    CrossProductExperimentDefinition,
    CrossProductQueryTypeConfig,
    CrossProductWorkloadConfig,
    DensityPreparationOptions,
)
from benchmarking.manifests import (
    DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER,
    PreparedWorkload,
    utc_now_iso,
    write_manifest,
)
from benchmarking.workloads.external import (
    export_parquet_catalog,
    export_remote_parquet_catalog,
)


def build_size_label(build_size: int) -> str:
    if build_size % 1_000_000 == 0:
        return f"{build_size // 1_000_000}m"
    if build_size % 1_000 == 0:
        return f"{build_size // 1_000}k"
    return str(build_size)


def selectivity_label(selectivity: float) -> str:
    scaled = int(round(selectivity * 1_000))
    return f"s{scaled:04d}"


def mask_column_name(selectivity: float, predicate_type: str) -> str:
    return f"mask_{selectivity_label(selectivity)}_{predicate_type}"


def dimension_table_name(build_size: int) -> str:
    return f"dim_{build_size_label(build_size)}"


def fact_fk_column_name(build_size: int, probe_clustering: str) -> str:
    return f"fk_{build_size_label(build_size)}_{probe_clustering}"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def build_dummy_column_expr(index: int, string_len: int, seed: int) -> str:
    salt = f"dummy_{index}:"
    return (
        f"substr(md5({sql_literal(salt)} || CAST(d_key AS VARCHAR) || "
        f"{sql_literal(f':seed:{seed}')}), 1, {string_len}) AS dummy_s{index}"
    )


CROSS_PRODUCT_PARAMETER_DESCRIPTIONS = {
    "fact_rows": "Number of rows in the fact/probe table.",
    "dimension_cardinality": "Total dimension rows before its query predicate.",
    "key_domain_size": (
        "Inclusive numeric key domain size; may be fixed, dimension_cardinality, "
        "or the derived dimension_distinct_keys."
    ),
    "key_density": "Selected distinct build keys divided by key_domain_size.",
    "selectivity": "Fraction of distinct dimension keys selected by each join predicate.",
    "join_count": "Number of homogeneous star-schema dimension joins.",
    "build_key_multiplicity": "Rows per distinct dimension key.",
    "probe_layout": "Physical fact-key layout: ordered or uniform.",
    "row_group_size": "Native DuckDB row-group size for the fact table.",
    "dummy_string_cols": "Number of payload columns materialized on each dimension.",
    "dummy_string_len": "Length of each deterministic dimension payload string.",
    "seed": "Seed used for deterministic key selection and fact-key generation.",
    "dimension_distinct_keys": "Derived as dimension_cardinality / build_key_multiplicity.",
    "selected_distinct_keys": "Derived selected key count after the dimension predicate.",
    "selected_build_rows": "Selected distinct keys multiplied by build-key multiplicity.",
    "dimension_predicate_selectivity": "Fraction of dimension keys selected by its predicate.",
    "probe_selectivity": "Expected fraction of fact rows matching one filtered dimension, before duplicate expansion.",
    "expected_query_selectivity": "Probe selectivity raised to join_count.",
    "expected_join_output_rows": "Expected matching fact rows multiplied by the build-key multiplicities; realized counts may differ.",
    "reference_domain_size": "Anchor domain whose per-row membership is preserved by fixed-output density sweeps.",
}

CROSS_PRODUCT_QUERY_TYPE_DESCRIPTIONS = {
    "range": "One contiguous range of dimension-key ranks.",
    "shoulder": (
        "Selected keys form contiguous ranges at the low and high domain boundaries."
    ),
    "seasonal": "Selected ranks split into an explicit number of separated ranges.",
    "scattered": "An exact-size deterministic scattered selection of key ranks.",
}

CROSS_PRODUCT_EXPERIMENT_CONFIGURATIONS = {
    "dimension_cardinality": {
        "query_types": True,
        "description": "Sweep dimension rows while holding all other parameters fixed.",
    },
    "selectivity": {
        "query_types": True,
        "description": "Sweep the fraction of distinct dimension keys selected.",
    },
    "key_domain_size": {
        "query_types": True,
        "description": "Sweep key density; optionally fix selected build keys and matching probe rows using per-query foreign keys.",
    },
    "build_key_multiplicity": {
        "query_types": True,
        "description": "Sweep duplicate rows per key while dimension cardinality stays fixed.",
        "constraints": ["single join", "fixed_dimension_cardinality"],
    },
    "join_count": {
        "query_types": True,
        "description": "Sweep homogeneous independently filtered star joins.",
        "constraints": ["distinct dimension keys", "uniform independent fact keys"],
    },
    "cluster_count": {
        "query_types": False,
        "description": "Sweep two or more separated build-key ranges.",
        "constraints": ["single join", "dense distinct dimension keys"],
    },
    "row_group_size": {
        "query_types": True,
        "description": "Sweep native DuckDB fact-table row-group size.",
        "constraints": ["DuckDB storage", "multiple of 2048"],
    },
}


@dataclass(frozen=True)
class ResolvedSweepCase:
    case_id: str
    sweep_kind: str
    sweep_value: int | float
    fact_rows: int
    dimension_cardinality: int
    dimension_distinct_keys: int
    key_domain_size: int
    selectivity: float
    selected_distinct_keys: int
    selected_build_rows: int
    key_density: float
    dimension_key_density: float
    join_count: int
    build_key_multiplicity: int
    row_group_size: int
    seed: int
    legacy_build_label: str | None = None
    probe_selectivity: float | None = None
    reference_domain_size: int | None = None


def _fixed_density(definition: CrossProductExperimentDefinition) -> bool:
    return definition.sweep.cardinality_mode == "fixed_selected_build_and_probe"


def _case_label(kind: str, value: int | float) -> str:
    if kind == "dimension_cardinality":
        return f"b{build_size_label(int(value))}"
    if kind == "selectivity":
        normalized = format(float(value), ".12g").replace("-", "m").replace(".", "p")
        return f"s{normalized}"
    prefixes = {
        "key_domain_size": "d",
        "build_key_multiplicity": "m",
        "join_count": "j",
        "cluster_count": "c",
        "row_group_size": "rg",
    }
    return f"{prefixes[kind]}{int(value)}"


def _resolve_sweep_cases(
    definition: CrossProductExperimentDefinition,
    *,
    storage_kind: str,
) -> list[ResolvedSweepCase]:
    fixed = definition.fixed
    sweep = definition.sweep
    if sweep.kind == "cluster_count":
        if definition.query_types:
            raise ValueError("query_types cannot be set for a cluster_count sweep")
    elif not definition.query_types:
        raise ValueError("query_types must not be empty")
    cases: list[ResolvedSweepCase] = []
    if _fixed_density(definition):
        if (sweep.kind != "key_domain_size" or fixed.build_key_multiplicity != 1
                or fixed.join_count != 1 or fixed.key_domain_size not in (None, fixed.dimension_cardinality)
                or storage_kind != "duckdb" or fixed.row_group_size != 122_880
                or not 0 < fixed.selectivity < 1
                or any(q.kind not in ("shoulder", "scattered") or q.probe_layouts != ("uniform",)
                       for q in definition.query_types)):
            raise ValueError("fixed density requires a dense distinct reference, one join, shoulder/scattered uniform queries, and default native row groups")
    for raw_value in sweep.values:
        dimension_cardinality = fixed.dimension_cardinality
        selectivity = fixed.selectivity
        domain_size = fixed.key_domain_size
        join_count = fixed.join_count
        multiplicity = fixed.build_key_multiplicity
        row_group_size = fixed.row_group_size
        if sweep.kind == "dimension_cardinality":
            dimension_cardinality = int(raw_value)
        elif sweep.kind == "selectivity":
            selectivity = float(raw_value)
        elif sweep.kind == "key_domain_size":
            domain_size = int(raw_value)
        elif sweep.kind == "build_key_multiplicity":
            multiplicity = int(raw_value)
        elif sweep.kind == "join_count":
            join_count = int(raw_value)
        elif sweep.kind == "row_group_size":
            row_group_size = int(raw_value)

        if _fixed_density(definition):
            selected = int(round(fixed.dimension_cardinality * fixed.selectivity))
            if int(domain_size) <= selected:
                raise ValueError("density domain must exceed selected build cardinality")
            dimension_cardinality = int(domain_size)
            selectivity = selected / dimension_cardinality

        if dimension_cardinality <= 0:
            raise ValueError("dimension_cardinality must be positive")
        if multiplicity <= 0 or dimension_cardinality % multiplicity != 0:
            raise ValueError(
                "dimension_cardinality must be evenly divisible by build_key_multiplicity"
            )
        distinct_keys = dimension_cardinality // multiplicity
        if domain_size is None:
            resolved_domain_size = dimension_cardinality
        elif domain_size == "dimension_distinct_keys":
            resolved_domain_size = distinct_keys
        else:
            resolved_domain_size = domain_size
        if resolved_domain_size < distinct_keys:
            raise ValueError(
                "key_domain_size must be at least the number of distinct dimension keys"
            )
        if not 0 < selectivity <= 1:
            raise ValueError("selectivity must be in (0, 1]")
        selected_distinct_keys = int(round(distinct_keys * selectivity))
        if selected_distinct_keys <= 0:
            raise ValueError("selectivity selects no dimension keys")
        if join_count <= 0:
            raise ValueError("join_count must be positive")
        if row_group_size <= 0 or row_group_size % 2048 != 0:
            raise ValueError("row_group_size must be a positive multiple of 2048")
        if sweep.kind == "build_key_multiplicity" and join_count != 1:
            raise ValueError("build_key_multiplicity sweeps currently support one join")
        if sweep.kind == "join_count":
            if multiplicity != 1:
                raise ValueError("join_count sweeps require distinct dimension table keys")
            layouts = {
                layout
                for query_type in definition.query_types
                for layout in query_type.probe_layouts
            }
            if layouts != {"uniform"}:
                raise ValueError(
                    "join_count sweeps require uniform probe layouts so join predicates are independent"
                )
        if sweep.kind == "cluster_count":
            cluster_count = int(raw_value)
            if cluster_count < 2:
                raise ValueError("cluster_count sweep values must be at least 2")
            if (
                multiplicity != 1
                or join_count != 1
                or resolved_domain_size != distinct_keys
            ):
                raise ValueError(
                    "cluster_count sweeps require one join and a dense, distinct dimension key domain"
                )
            if cluster_count > selected_distinct_keys:
                raise ValueError("cluster_count cannot exceed selected distinct keys")
            if distinct_keys - selected_distinct_keys < cluster_count - 1:
                raise ValueError("cluster_count sweep needs at least one gap between clusters")
        if sweep.kind == "row_group_size" and storage_kind != "duckdb":
            raise ValueError("row_group_size sweeps require native DuckDB storage")
        if fixed.fact_rows <= 0:
            raise ValueError("fact_rows must be positive")
        legacy_label = None
        if (
            sweep.kind == "dimension_cardinality"
            and fixed.key_domain_size is None
            and multiplicity == 1
            and join_count == 1
        ):
            legacy_label = build_size_label(dimension_cardinality)
        cases.append(
            ResolvedSweepCase(
                case_id=_case_label(sweep.kind, raw_value),
                sweep_kind=sweep.kind,
                sweep_value=raw_value,
                fact_rows=fixed.fact_rows,
                dimension_cardinality=dimension_cardinality,
                dimension_distinct_keys=distinct_keys,
                key_domain_size=resolved_domain_size,
                selectivity=selectivity,
                selected_distinct_keys=selected_distinct_keys,
                selected_build_rows=selected_distinct_keys * multiplicity,
                key_density=selected_distinct_keys / resolved_domain_size,
                dimension_key_density=distinct_keys / resolved_domain_size,
                join_count=join_count,
                build_key_multiplicity=multiplicity,
                row_group_size=row_group_size,
                seed=fixed.seed,
                legacy_build_label=legacy_label,
                probe_selectivity=fixed.selectivity if _fixed_density(definition) else selectivity,
                reference_domain_size=fixed.dimension_cardinality if _fixed_density(definition) else None,
            )
        )
    return cases


def _validate(spec: CrossProductWorkloadConfig) -> None:
    if spec.experiment_definition is not None:
        _resolve_sweep_cases(spec.experiment_definition, storage_kind=spec.storage_kind)
        fixed = spec.experiment_definition.fixed
        uses_separate_native_fact = (
            spec.experiment_definition.sweep.kind == "row_group_size"
            or fixed.row_group_size != 122_880
        )
        if uses_separate_native_fact and (
            spec.storage_kind != "duckdb"
            or spec.remote_http_parquet_latency_ms is not None
        ):
            raise ValueError(
                "native fact row-group configuration cannot be combined with DuckLake "
                "or remote Parquet storage"
            )
        if fixed.dummy_string_cols < 0:
            raise ValueError("dummy_string_cols cannot be negative")
        if fixed.dummy_string_len <= 0:
            raise ValueError("dummy_string_len must be positive")
        return
    if not spec.build_sizes:
        raise ValueError("build_sizes must not be empty")
    if len(set(spec.build_sizes)) != len(spec.build_sizes):
        raise ValueError("build_sizes must be distinct")
    if not spec.selectivities:
        raise ValueError("selectivities must not be empty")
    if len(set(spec.selectivities)) != len(spec.selectivities):
        raise ValueError("selectivities must be distinct")
    if not spec.predicate_types:
        raise ValueError("predicate_types must not be empty")
    if not spec.probe_clusterings:
        raise ValueError("probe_clusterings must not be empty")
    allowed_predicate_types = {"range", "shoulder", "seasonal", "scattered"}
    allowed_probe_clusterings = {"ordered", "uniform"}
    for build_size in spec.build_sizes:
        if build_size <= 0:
            raise ValueError("build_sizes must be positive")
    for selectivity in spec.selectivities:
        if not 0 < selectivity <= 1:
            raise ValueError("selectivities must be in (0, 1]")
    for predicate_type in spec.predicate_types:
        if predicate_type not in allowed_predicate_types:
            raise ValueError(f"unsupported predicate_type: {predicate_type}")
    for probe_clustering in spec.probe_clusterings:
        if probe_clustering not in allowed_probe_clusterings:
            raise ValueError(f"unsupported probe_clustering: {probe_clustering}")
    if spec.fact_rows <= 0:
        raise ValueError("fact_rows must be positive")
    if spec.dummy_string_cols < 0:
        raise ValueError("dummy_string_cols cannot be negative")
    if spec.dummy_string_len <= 0:
        raise ValueError("dummy_string_len must be positive")


def _spread_key_expr(rank_expr: str, distinct_keys: int, domain_size: int) -> str:
    if distinct_keys == 1:
        return "1::BIGINT"
    if distinct_keys == domain_size:
        return f"CAST({rank_expr} AS BIGINT) + 1"
    return (
        "CAST(1 + ((CAST({rank} AS HUGEINT) * "
        "CAST({domain_minus_one} AS HUGEINT)) // "
        "CAST({key_minus_one} AS HUGEINT)) AS BIGINT)"
    ).format(
        rank=rank_expr,
        domain_minus_one=domain_size - 1,
        key_minus_one=distinct_keys - 1,
    )


def _shoulder_layout(
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase] | tuple[ResolvedSweepCase, ...],
    case: ResolvedSweepCase,
) -> tuple[int, int] | None:
    if not any(
        query_type.kind == "shoulder"
        for query_type in _case_query_types(definition, case)
    ):
        return None

    compatible_cases = [
        candidate
        for candidate in cases
        if candidate.dimension_distinct_keys == case.dimension_distinct_keys
        and candidate.key_domain_size == case.key_domain_size
    ]
    leading = max(candidate.selected_distinct_keys // 2 for candidate in compatible_cases)
    trailing = max(
        candidate.selected_distinct_keys - candidate.selected_distinct_keys // 2
        for candidate in compatible_cases
    )
    return leading, trailing


def _shoulder_preserving_key_expr(
    rank_expr: str,
    distinct_keys: int,
    domain_size: int,
    leading: int,
    trailing: int,
) -> str:
    if domain_size == distinct_keys:
        return _spread_key_expr(rank_expr, distinct_keys, domain_size)

    middle_keys = distinct_keys - leading - trailing
    high_rank_start = distinct_keys - trailing
    high_key_start = domain_size - trailing + 1
    parts = [
        f"WHEN {rank_expr} < {leading} THEN 1 + {rank_expr}",
        (
            f"WHEN {rank_expr} >= {high_rank_start} THEN {high_key_start} + "
            f"({rank_expr} - {high_rank_start})"
        ),
    ]
    if middle_keys == 1:
        middle_expr = str(leading + 1)
    else:
        middle_domain = domain_size - leading - trailing
        middle_expr = (
            f"{leading + 1} + ((CAST(({rank_expr} - {leading}) AS HUGEINT) * "
            f"CAST({middle_domain - 1} AS HUGEINT)) // "
            f"CAST({middle_keys - 1} AS HUGEINT))"
        )
    return f"CAST(CASE {' '.join(parts)} ELSE {middle_expr} END AS BIGINT)"


def _sweep_key_expr(
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase] | tuple[ResolvedSweepCase, ...],
    case: ResolvedSweepCase,
    rank_expr: str,
    query_type: CrossProductQueryTypeConfig | None = None,
) -> str:
    layout = (_shoulder_layout(definition, cases, case)
              if query_type is None or query_type.kind == "shoulder" else None)
    if layout is None:
        return _spread_key_expr(
            rank_expr,
            case.dimension_distinct_keys,
            case.key_domain_size,
        )
    return _shoulder_preserving_key_expr(
        rank_expr,
        case.dimension_distinct_keys,
        case.key_domain_size,
        *layout,
    )


def _query_type_id(query_type: CrossProductQueryTypeConfig) -> str:
    if query_type.kind == "seasonal":
        return f"seasonal_c{query_type.cluster_count:05d}"
    return query_type.kind


def _case_query_types(
    definition: CrossProductExperimentDefinition,
    case: ResolvedSweepCase,
) -> tuple[CrossProductQueryTypeConfig, ...]:
    if case.sweep_kind == "cluster_count":
        return (
            CrossProductQueryTypeConfig(
                kind="seasonal",
                cluster_count=int(case.sweep_value),
                probe_layouts=(definition.fixed.probe_layout,),
            ),
        )
    return definition.query_types


def _selected_rank_query(
    case: ResolvedSweepCase,
    query_type: CrossProductQueryTypeConfig,
    join_index: int,
) -> str:
    key_count = case.dimension_distinct_keys
    selected = case.selected_distinct_keys
    if query_type.kind == "range":
        available_starts = key_count - selected + 1
        start = ((join_index - 1) * max(1, selected)) % available_starts
        return f"SELECT r AS key_rank FROM range({start}, {start + selected}) t(r)"
    if query_type.kind == "shoulder":
        leading = selected // 2
        trailing = selected - leading
        parts: list[str] = []
        if leading:
            parts.append(f"SELECT r AS key_rank FROM range(0, {leading}) t(r)")
        if trailing:
            parts.append(
                f"SELECT r AS key_rank FROM range({key_count - trailing}, {key_count}) t(r)"
            )
        return "\nUNION ALL\n".join(parts)
    if query_type.kind == "seasonal":
        cluster_count = query_type.cluster_count
        assert cluster_count is not None
        if cluster_count > selected:
            raise ValueError(
                f"seasonal cluster_count {cluster_count} exceeds selected distinct keys {selected}"
            )
        if key_count - selected < cluster_count - 1:
            raise ValueError(
                f"seasonal cluster_count {cluster_count} needs at least one gap between clusters"
            )
        range_base, range_remainder = divmod(selected, cluster_count)
        gap_base, gap_remainder = divmod(key_count - selected, cluster_count - 1)
        if case.legacy_build_label is not None:
            # Preserve the established paper workload's construction path.
            # New cluster-count sweeps use the scalable relational form below.
            parts = []
            current = 0
            for cluster_index in range(cluster_count):
                range_size = range_base + (
                    1 if cluster_index < range_remainder else 0
                )
                parts.append(
                    f"SELECT r AS key_rank FROM range({current}, "
                    f"{current + range_size}) t(r)"
                )
                current += range_size
                if cluster_index + 1 < cluster_count:
                    current += gap_base + (
                        1 if cluster_index < gap_remainder else 0
                    )
            return "\nUNION ALL\n".join(parts)
        # Generate clusters relationally rather than emitting one UNION branch
        # per cluster.  The latter becomes prohibitively large for the paper
        # sweep, whose singleton-cluster endpoint contains 100,000 clusters.
        return f"""
            SELECT
                cluster_start + cluster_offset AS key_rank
            FROM (
                SELECT
                    cluster_index * {range_base}
                        + least(cluster_index, {range_remainder})
                        + cluster_index * {gap_base}
                        + least(cluster_index, {gap_remainder}) AS cluster_start,
                    {range_base}
                        + CASE WHEN cluster_index < {range_remainder} THEN 1 ELSE 0 END
                        AS cluster_size
                FROM range(0, {cluster_count}) clusters(cluster_index)
            ) cluster_layout
            CROSS JOIN LATERAL range(0, cluster_size) offsets(cluster_offset)
        """
    if case.legacy_build_label is not None:
        return (
            "SELECT "
            f"(({case.seed + key_count + selected} + r * 7919) % {key_count}) "
            f"AS key_rank FROM range(0, {selected}) t(r)"
        )
    salt = case.seed + join_index * 104_729
    return (
        f"SELECT r AS key_rank FROM range(0, {key_count}) t(r) "
        f"ORDER BY hash(r + {salt}), r LIMIT {selected}"
    )


def _sweep_mask_column(
    case: ResolvedSweepCase,
    query_type: CrossProductQueryTypeConfig,
) -> str:
    if case.legacy_build_label is not None and (
        query_type.kind != "seasonal" or query_type.cluster_count == 10
    ):
        return mask_column_name(case.selectivity, query_type.kind)
    scaled = int(round(case.selectivity * 1_000))
    if scaled > 0 and abs(case.selectivity - scaled / 1_000) < 1e-12:
        selectivity_id = selectivity_label(case.selectivity)
    else:
        # The legacy three-decimal label maps sub-0.05% selectivities to
        # s0000. Preserve legacy names where they are unambiguous and use an
        # exact identifier for the lower end of selectivity sweeps.
        selectivity_id = _case_label("selectivity", case.selectivity)
    return f"mask_{selectivity_id}_{_query_type_id(query_type)}"


def _key_layout_suffix(case: ResolvedSweepCase, query_type: CrossProductQueryTypeConfig | None) -> str:
    if query_type is None or case.legacy_build_label is not None or case.key_domain_size == case.dimension_distinct_keys:
        return ""
    return "_shoulder" if query_type.kind == "shoulder" else "_spread"


def _sweep_dimension_table_name(case: ResolvedSweepCase, join_index: int,
                              query_type: CrossProductQueryTypeConfig | None = None) -> str:
    if case.legacy_build_label is not None and join_index == 1:
        return f"dim_{case.legacy_build_label}"
    shared_ids = {
        "selectivity": "selectivity",
        "join_count": "joins",
        "cluster_count": "clusters",
        "row_group_size": "row_groups",
    }
    relation_id = shared_ids.get(case.sweep_kind, case.case_id)
    return f"dim_{relation_id}_j{join_index:02d}{_key_layout_suffix(case, query_type)}"


def _sweep_fk_column_name(
    case: ResolvedSweepCase,
    join_index: int,
    probe_layout: str,
    query_type: CrossProductQueryTypeConfig | None = None,
) -> str:
    if case.legacy_build_label is not None and join_index == 1:
        return f"fk_{case.legacy_build_label}_{probe_layout}"
    shared_ids = {
        "selectivity": "selectivity",
        "join_count": "joins",
        "cluster_count": "clusters",
    }
    relation_id = shared_ids.get(case.sweep_kind, case.case_id)
    suffix = (f"_{query_type.kind}" if case.reference_domain_size is not None and query_type is not None
              else _key_layout_suffix(case, query_type))
    return f"fk_{relation_id}_j{join_index:02d}_{probe_layout}{suffix}"


def _create_sweep_dimension_table(
    con: duckdb.DuckDBPyConnection,
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase],
    join_index: int,
    key_query_type: CrossProductQueryTypeConfig | None = None,
) -> dict:
    case = cases[0]
    table_name = _sweep_dimension_table_name(case, join_index, key_query_type)
    base_table_name = f"{table_name}_base"
    key_expr = _sweep_key_expr(definition, cases, case, "key_rank", key_query_type)
    if case.legacy_build_label is not None:
        dummy_exprs = [
            build_dummy_column_expr(
                index,
                definition.fixed.dummy_string_len,
                case.seed,
            )
            for index in range(1, definition.fixed.dummy_string_cols + 1)
        ]
    else:
        dummy_exprs = [
            (
                f"substr(md5('dummy_{index}:' || CAST(d_key AS VARCHAR) || ':dup:' || "
                f"CAST(duplicate_id AS VARCHAR) || ':seed:{case.seed}'), 1, "
                f"{definition.fixed.dummy_string_len}) AS dummy_s{index}"
            )
            for index in range(1, definition.fixed.dummy_string_cols + 1)
        ]
    select_list = ",\n            ".join(
        ["d_key", "key_rank", "duplicate_id"] + dummy_exprs
    )
    con.execute(
        f"""
        CREATE TABLE {base_table_name} AS
        SELECT
            {select_list}
        FROM (
            SELECT
                {key_expr} AS d_key,
                key_rank,
                duplicate_id
            FROM range(0, {case.dimension_distinct_keys}) keys(key_rank)
            CROSS JOIN range(0, {case.build_key_multiplicity}) duplicates(duplicate_id)
        ) generated_dimension
        """
    )

    joins: list[str] = []
    masks: list[str] = []
    mask_metadata: dict[str, dict] = {}
    seen_mask_names: set[str] = set()
    temp_tables: list[str] = []
    scattered_rank_table: str | None = None
    for mask_case in cases:
        for query_type in _case_query_types(definition, mask_case):
            if _key_layout_suffix(mask_case, query_type) != _key_layout_suffix(case, key_query_type):
                continue
            mask_name = _sweep_mask_column(mask_case, query_type)
            if mask_name in seen_mask_names:
                continue
            seen_mask_names.add(mask_name)
            if mask_case.legacy_build_label is None and query_type.kind == "range":
                available_starts = (
                    mask_case.dimension_distinct_keys
                    - mask_case.selected_distinct_keys
                    + 1
                )
                start = (
                    (join_index - 1) * max(1, mask_case.selected_distinct_keys)
                ) % available_starts
                masks.append(
                    f"(b.key_rank >= {start} AND b.key_rank < "
                    f"{start + mask_case.selected_distinct_keys}) AS {mask_name}"
                )
            elif mask_case.legacy_build_label is None and query_type.kind == "shoulder":
                leading = mask_case.selected_distinct_keys // 2
                trailing = mask_case.selected_distinct_keys - leading
                masks.append(
                    f"(b.key_rank < {leading} OR b.key_rank >= "
                    f"{mask_case.dimension_distinct_keys - trailing}) AS {mask_name}"
                )
            elif mask_case.legacy_build_label is None and query_type.kind == "scattered":
                if scattered_rank_table is None:
                    scattered_rank_table = f"ranked_{table_name}_scattered"
                    salt = mask_case.seed + join_index * 104_729
                    con.execute(
                        f"""
                        CREATE TEMP TABLE {scattered_rank_table} AS
                        SELECT
                            r AS key_rank,
                            row_number() OVER (ORDER BY hash(r + {salt}), r) - 1
                                AS selection_rank
                        FROM range(0, {mask_case.dimension_distinct_keys}) keys(r)
                        """
                    )
                    temp_tables.append(scattered_rank_table)
                    joins.append(
                        f"LEFT JOIN {scattered_rank_table} AS {scattered_rank_table} "
                        f"ON b.key_rank = {scattered_rank_table}.key_rank"
                    )
                masks.append(
                    f"({scattered_rank_table}.selection_rank < "
                    f"{mask_case.selected_distinct_keys}) AS {mask_name}"
                )
            else:
                selected_table = f"selected_{table_name}_{mask_name}"
                con.execute(
                    f"CREATE TEMP TABLE {selected_table} AS "
                    f"{_selected_rank_query(mask_case, query_type, join_index)}"
                )
                temp_tables.append(selected_table)
                joins.append(
                    f"LEFT JOIN {selected_table} AS {selected_table} "
                    f"ON b.key_rank = {selected_table}.key_rank"
                )
                masks.append(
                    f"({selected_table}.key_rank IS NOT NULL) AS {mask_name}"
                )
            mask_metadata[mask_name] = {
                "query_type": query_type.kind,
                "cluster_count": query_type.cluster_count,
                "selected_distinct_keys": mask_case.selected_distinct_keys,
                "selected_build_rows": mask_case.selected_build_rows,
            }
    materialized_columns = ["b.d_key"] + [
        f"b.dummy_s{index}"
        for index in range(1, definition.fixed.dummy_string_cols + 1)
    ] + masks
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            {', '.join(materialized_columns)}
        FROM {base_table_name} AS b
        {' '.join(joins)}
        """
    )
    con.execute(f"DROP TABLE {base_table_name}")
    for temp_table in temp_tables:
        con.execute(f"DROP TABLE {temp_table}")
    return {
        "table_name": table_name,
        "join_index": join_index,
        "row_count": case.dimension_cardinality,
        "distinct_keys": case.dimension_distinct_keys,
        "key_domain_size": case.key_domain_size,
        "key_multiplicity": case.build_key_multiplicity,
        "masks": mask_metadata,
    }


def _fact_key_expr(
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase] | tuple[ResolvedSweepCase, ...],
    case: ResolvedSweepCase,
    rank_expr: str,
    query_type: CrossProductQueryTypeConfig | None = None,
) -> str:
    return _sweep_key_expr(definition, cases, case, rank_expr, query_type)


def _fact_column_expr(
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase] | tuple[ResolvedSweepCase, ...],
    case: ResolvedSweepCase,
    join_index: int,
    probe_layout: str,
    column_name: str,
    query_type: CrossProductQueryTypeConfig | None = None,
) -> str:
    if probe_layout == "ordered":
        rank_expr = (
            f"CAST(FLOOR(((i - 1) * {case.dimension_distinct_keys})::DOUBLE / "
            f"{case.fact_rows}) AS BIGINT)"
        )
    elif case.legacy_build_label is not None:
        # Preserve the established paper workload exactly.
        rank_expr = (
            f"CAST(FLOOR(random() * {case.dimension_distinct_keys}) AS BIGINT)"
        )
    else:
        salt = case.seed + join_index * 104_729
        rank_expr = (
            f"CAST(hash(i + {salt}) % {case.dimension_distinct_keys} AS BIGINT)"
        )
    return f"{_fact_key_expr(definition, cases, case, rank_expr, query_type)} AS {column_name}"


def _case_probe_layouts(
    definition: CrossProductExperimentDefinition,
    case: ResolvedSweepCase,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                layout
                for query_type in _case_query_types(definition, case)
                for layout in query_type.probe_layouts
            }
        )
    )


def _density_generation_start(
    con: duckdb.DuckDBPyConnection,
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase],
) -> int:
    """Validate resumable input before changing it, including pre-batching databases."""
    tables = {row[0] for row in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE NOT temporary "
        "AND database_name = current_database() AND schema_name = 'main'"
    ).fetchall()}
    dimensions = {_sweep_dimension_table_name(case, 1): case for case in cases}
    unexpected = tables - dimensions.keys() - {"fact", "_density_generation_state"}
    if unexpected:
        raise ValueError(f"unexpected tables in density database: {sorted(unexpected)}")
    signature = json.dumps(asdict(definition), sort_keys=True)
    next_id = 1
    if "_density_generation_state" in tables:
        state = con.execute("SELECT definition_json, next_fact_id FROM _density_generation_state").fetchall()
        if len(state) != 1 or state[0][0] != signature:
            raise ValueError("density generation definition differs from the saved progress")
        next_id = state[0][1]
        if not 1 <= next_id <= definition.fixed.fact_rows + 1:
            raise ValueError("invalid saved density fact position")
    elif "fact" in tables:
        raise ValueError("cannot resume an existing fact table without density generation progress")
    for table, case in dimensions.items():
        if table not in tables:
            continue
        expected = [("d_key", "BIGINT")]
        expected += [(f"dummy_s{j}", "VARCHAR") for j in range(1, definition.fixed.dummy_string_cols + 1)]
        expected += [(_sweep_mask_column(case, q), "BOOLEAN") for q in definition.query_types]
        actual = [(row[0], row[1]) for row in con.execute(f"DESCRIBE {table}").fetchall()]
        if actual != expected or con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != case.key_domain_size:
            raise ValueError(f"incomplete or incompatible density dimension: {table}")
    if "fact" in tables:
        count, low, high = con.execute("SELECT count(*), min(fact_id), max(fact_id) FROM fact").fetchone()
        if count != next_id - 1 or (count and (low != 1 or high != count)):
            raise ValueError("density fact rows do not match saved progress")
    elif next_id != 1:
        raise ValueError("density fact table missing despite saved progress")
    if "_density_generation_state" not in tables:
        con.execute("CREATE TABLE _density_generation_state AS "
                    "SELECT ?::VARCHAR AS definition_json, 1::BIGINT AS next_fact_id", [signature])
    return next_id


def _insert_density_fact_batches(
    con: duckdb.DuckDBPyConnection,
    select_sql: str,
    fact_rows: int,
    batch_rows: int,
    next_id: int,
) -> None:
    """Commit rows and their restart position together; never sort all fact rows."""
    empty_select = select_sql.format(start=1, stop=1)
    con.execute("CREATE TABLE IF NOT EXISTS fact AS " + empty_select)
    expected = [(r[0], r[1]) for r in con.execute("DESCRIBE " + empty_select).fetchall()]
    actual = [(r[0], r[1]) for r in con.execute("DESCRIBE fact").fetchall()]
    if actual != expected:
        raise ValueError("density fact schema differs from the requested definition")
    for start in range(next_id, fact_rows + 1, batch_rows):
        stop = min(start + batch_rows, fact_rows + 1)
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("INSERT INTO fact " + select_sql.format(start=start, stop=stop))
            con.execute("UPDATE _density_generation_state SET next_fact_id = ?", [stop])
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


def _create_fixed_density_tables(
    con: duckdb.DuckDBPyConnection,
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase],
    preparation: DensityPreparationOptions = DensityPreparationOptions(),
) -> dict[tuple[str, int], dict]:
    """Preserve reference membership with small maps and bounded fact-side sorts.

    The complement map stores gaps between selected keys, not every rejected
    key. An ASOF lookup maps a uniform complement ordinal to its actual key.
    Fact generation batches consecutive IDs so ASOF and order-restoration sorts
    buffer at most one batch. Completed dimensions and committed batches can be
    reused, including dimensions left by the original monolithic generator.
    """
    if preparation.batch_rows <= 0 or preparation.batch_rows % 122_880:
        raise ValueError("preparation.batch_rows must be a positive multiple of 122880")
    next_id = _density_generation_start(con, definition, cases)
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    cases_by_domain = {case.key_domain_size: case for case in cases}
    fixed = definition.fixed
    reference = fixed.dimension_cardinality
    selected = int(round(reference * fixed.selectivity))
    leading, trailing = selected // 2, selected - selected // 2
    salt = fixed.seed + 104_729
    scattered = any(q.kind == "scattered" for q in definition.query_types)
    domains = sorted({reference, *(case.key_domain_size for case in cases)})
    if scattered:
        for domain in domains:
            # Keep the same exact, hash-ranked selection as the ordinary sweeps.
            case = cases_by_domain.get(domain)
            table = _sweep_dimension_table_name(case, 1) if case is not None else None
            if table in existing:
                q = next(q for q in definition.query_types if q.kind == "scattered")
                chosen = f"SELECT d_key - 1 AS key_rank FROM {table} WHERE {_sweep_mask_column(case, q)}"
            else:
                chosen = (f"SELECT r AS key_rank FROM range({domain}) keys(r) "
                          f"ORDER BY hash(r + {salt}), r LIMIT {selected}")
            con.execute(f"""
                CREATE TEMP TABLE density_selected_{domain} AS
                SELECT key_rank, row_number() OVER (ORDER BY hash(key_rank + {salt}), key_rank) - 1
                       AS selection_rank
                FROM ({chosen}) chosen
            """)
            if con.execute(f"SELECT count(*) FROM density_selected_{domain}").fetchone()[0] != selected:
                raise ValueError(f"incorrect selected-key count in density dimension {domain}")
            con.execute(f"""
                CREATE TEMP TABLE density_gaps_{domain} AS
                SELECT gap_start, max(skipped) AS skipped FROM (
                    SELECT key_rank - (row_number() OVER (ORDER BY key_rank) - 1) AS gap_start,
                           row_number() OVER (ORDER BY key_rank) AS skipped
                    FROM density_selected_{domain}
                    UNION ALL SELECT 0, 0
                ) gaps GROUP BY gap_start
            """)

    dimensions = {}
    columns = ["i AS fact_id"]
    joins = ([f"LEFT JOIN density_selected_{reference} ref ON anchor_rank = ref.key_rank"]
             if scattered else [])
    gap_joins = []
    for case in cases:
        domain = case.key_domain_size
        table = _sweep_dimension_table_name(case, 1)
        masks = []
        mask_metadata = {}
        for q in definition.query_types:
            mask = _sweep_mask_column(case, q)
            condition = (f"(b.key_rank < {leading} OR b.key_rank >= {domain - trailing})"
                         if q.kind == "shoulder" else "(selected_keys.key_rank IS NOT NULL)")
            masks.append(f"{condition} AS {mask}")
            mask_metadata[mask] = {"query_type": q.kind, "cluster_count": None,
                                   "selected_distinct_keys": selected, "selected_build_rows": selected}
            column = _sweep_fk_column_name(case, 1, "uniform", q)
            if domain == reference:
                expression = "anchor_rank + 1"
            elif q.kind == "shoulder":
                positive = f"(anchor_rank < {leading} OR anchor_rank >= {reference - trailing})"
                rank = f"CAST(pick_positive % {selected} AS BIGINT)"
                selected_key = f"CASE WHEN {rank} < {leading} THEN {rank} + 1 ELSE {domain - selected} + {rank} + 1 END"
                rejected_key = f"{leading + 1} + CAST(pick_negative % {domain - selected} AS BIGINT)"
                expression = f"CASE WHEN {positive} THEN {selected_key} ELSE {rejected_key} END"
            else:
                joins.append(f"LEFT JOIN density_selected_{domain} sel_{domain} "
                             f"ON CAST(pick_positive % {selected} AS BIGINT) = sel_{domain}.selection_rank")
                gap_joins.append(f"ASOF LEFT JOIN density_gaps_{domain} gap_{domain} "
                                 f"ON CAST(pick_negative % {domain - selected} AS BIGINT) >= gap_{domain}.gap_start")
                expression = (f"CASE WHEN ref.key_rank IS NOT NULL THEN sel_{domain}.key_rank + 1 "
                              f"ELSE CAST(pick_negative % {domain - selected} AS BIGINT) + gap_{domain}.skipped + 1 END")
            columns.append(f"CAST({expression} AS BIGINT) AS {column}")
        payload = [f"substr(md5('dummy_{j}:' || CAST(d_key AS VARCHAR) || ':dup:0:seed:{fixed.seed}'), "
                   f"1, {fixed.dummy_string_len}) AS dummy_s{j}"
                   for j in range(1, fixed.dummy_string_cols + 1)]
        selected_join = (f"LEFT JOIN density_selected_{domain} selected_keys "
                         "ON b.key_rank = selected_keys.key_rank" if scattered else "")
        if table not in existing:
            con.execute(f"""
                CREATE TABLE {table} AS SELECT {', '.join(['d_key'] + payload + masks)}
                FROM (SELECT r AS key_rank, r + 1 AS d_key FROM range({domain}) keys(r)) b
                {selected_join}
            """)
        dimensions[(table, 1)] = {"table_name": table, "join_index": 1,
                                  "row_count": domain, "distinct_keys": domain,
                                  "key_domain_size": domain, "key_multiplicity": 1,
                                  "masks": mask_metadata}
    columns.append("i * 17 AS measure")
    # Restore order inside each consecutive batch, preserving uniform probes
    # without a final sort over the entire fact table.
    select_sql = f"""
        SELECT {', '.join(columns)}
        FROM (SELECT i, CAST(hash(i + {salt}) % {reference} AS BIGINT) AS anchor_rank,
                     hash(i + {salt + 1_000_003}) AS pick_positive,
                     hash(i + {salt + 2_000_033}) AS pick_negative
              FROM range({{start}}, {{stop}}) rows(i)) f
        {' '.join(joins + gap_joins)}
        ORDER BY i
    """
    _insert_density_fact_batches(con, select_sql, fixed.fact_rows, preparation.batch_rows, next_id)
    return dimensions


def _create_shared_sweep_fact_table(
    con: duckdb.DuckDBPyConnection,
    definition: CrossProductExperimentDefinition,
    cases: list[ResolvedSweepCase],
) -> None:
    columns: dict[str, str] = {}
    for case in cases:
        for join_index in range(1, case.join_count + 1):
            for query_type in _case_query_types(definition, case):
                for probe_layout in query_type.probe_layouts:
                    column_name = _sweep_fk_column_name(case, join_index, probe_layout, query_type)
                    expression = _fact_column_expr(definition, cases, case, join_index,
                                                   probe_layout, column_name, query_type)
                    previous = columns.setdefault(column_name, expression)
                    if previous != expression:
                        raise ValueError(f"shared fact column {column_name} resolves to different expressions")
    select_list = ",\n            ".join(
        ["i AS fact_id"] + list(columns.values()) + ["i * 17 AS measure"]
    )
    con.execute(
        f"""
        CREATE TABLE fact AS
        SELECT
            {select_list}
        FROM range(1, {definition.fixed.fact_rows + 1}) t(i)
        """
    )


def _create_row_group_fact_database(
    definition: CrossProductExperimentDefinition,
    case: ResolvedSweepCase,
    db_path: Path,
) -> tuple[str, dict]:
    if db_path.exists():
        db_path.unlink()
    alias = f"fact_{case.case_id}"
    coordinator = duckdb.connect()
    try:
        seed_value = ((case.seed % 10_000_000) + 0.5) / 10_000_001
        coordinator.execute(f"SELECT setseed({seed_value})")
        coordinator.execute(
            f"ATTACH {sql_literal(str(db_path.resolve()))} AS {quote_identifier(alias)} "
            f"(ROW_GROUP_SIZE {case.row_group_size}, STORAGE_VERSION 'v1.2.0')"
        )
        columns: dict[str, str] = {}
        for join_index in range(1, case.join_count + 1):
            for query_type in _case_query_types(definition, case):
                for probe_layout in query_type.probe_layouts:
                    column_name = f"fk_j{join_index:02d}_{probe_layout}{_key_layout_suffix(case, query_type)}"
                    columns[column_name] = _fact_column_expr(definition, (case,), case,
                                                           join_index, probe_layout, column_name, query_type)
        select_list = ",\n                ".join(
            ["i AS fact_id"] + list(columns.values()) + ["i * 17 AS measure"]
        )
        coordinator.execute(
            f"""
            CREATE TABLE {quote_identifier(alias)}.fact AS
            SELECT
                {select_list}
            FROM range(1, {case.fact_rows + 1}) t(i)
            """
        )
        actual_row_groups = coordinator.execute(
            f"SELECT COUNT(DISTINCT row_group_id) FROM pragma_storage_info("
            f"{sql_literal(alias + '.fact')})"
        ).fetchone()[0]
        effective_size = coordinator.execute(
            "SELECT options['row_group_size'] FROM duckdb_databases() WHERE database_name = ?",
            [alias],
        ).fetchone()[0]
        coordinator.execute(f"DETACH {quote_identifier(alias)}")
    finally:
        coordinator.close()
    return alias, {
        "db_path": str(db_path.resolve()),
        "row_group_size": int(effective_size),
        "row_group_count": int(actual_row_groups),
    }


def _selected_rows(build_size: int, selectivity: float) -> int:
    return int(round(build_size * selectivity))


def _selected_key_rows(build_size: int, selected_rows: int, predicate_type: str, seed: int) -> str:
    if predicate_type == "range":
        return f"SELECT i AS d_key FROM range(1, {selected_rows + 1}) t(i)"
    if predicate_type == "shoulder":
        leading_rows = selected_rows // 2
        trailing_rows = selected_rows - leading_rows
        trailing_start = build_size - trailing_rows + 1
        if leading_rows == 0:
            return f"SELECT i AS d_key FROM range({trailing_start}, {build_size + 1}) t(i)"
        return (
            f"SELECT i AS d_key FROM range(1, {leading_rows + 1}) t(i)\n"
            f"UNION ALL\n"
            f"SELECT i AS d_key FROM range({trailing_start}, {build_size + 1}) t(i)"
        )
    if predicate_type == "seasonal":
        interval_count = min(10, selected_rows)
        rows_per_interval = selected_rows // interval_count
        remainder = selected_rows % interval_count
        gap_space = max(0, build_size - selected_rows)
        gap_base = gap_space // max(1, interval_count - 1)
        gap_remainder = gap_space % max(1, interval_count - 1) if interval_count > 1 else 0
        parts: list[str] = []
        current = 1
        for interval_index in range(interval_count):
            interval_size = rows_per_interval + (1 if interval_index < remainder else 0)
            end = current + interval_size
            parts.append(f"SELECT i AS d_key FROM range({current}, {end}) t(i)")
            current = end
            if interval_index + 1 < interval_count:
                current += gap_base + (1 if interval_index < gap_remainder else 0)
        return "\nUNION ALL\n".join(parts)
    return f"""
        SELECT
            1 + (({seed + build_size + selected_rows} + i * 7919) % {build_size}) AS d_key
        FROM range(0, {selected_rows}) t(i)
    """


def _create_dimension_table(
    con: duckdb.DuckDBPyConnection,
    spec: CrossProductWorkloadConfig,
    build_size: int,
    dimension_metadata: dict[str, dict],
) -> None:
    base_table_name = f"{dimension_table_name(build_size)}_base"
    final_table_name = dimension_table_name(build_size)
    dummy_exprs = [
        build_dummy_column_expr(index, spec.dummy_string_len, spec.seed)
        for index in range(1, spec.dummy_string_cols + 1)
    ]
    select_list = ",\n                ".join(
        ["i AS d_key"] + dummy_exprs if dummy_exprs else ["i AS d_key"]
    )
    con.execute(
        f"""
        CREATE TABLE {base_table_name} AS
        SELECT
            {select_list}
        FROM range(1, {build_size + 1}) t(i)
        """
    )

    join_clauses: list[str] = []
    mask_selects: list[str] = []
    dimension_entry: dict[str, dict] = {}
    created_temp_tables: list[str] = []
    for selectivity in spec.selectivities:
        for predicate_type in spec.predicate_types:
            selected_rows = _selected_rows(build_size, selectivity)
            table_name = (
                f"selected_keys_{build_size_label(build_size)}_"
                f"{selectivity_label(selectivity)}_{predicate_type}"
            )
            column_name = mask_column_name(selectivity, predicate_type)
            con.execute(
                f"CREATE TEMP TABLE {table_name} AS "
                f"{_selected_key_rows(build_size, selected_rows, predicate_type, spec.seed)}"
            )
            created_temp_tables.append(table_name)
            join_clauses.append(
                f"LEFT JOIN {table_name} AS {table_name} ON b.d_key = {table_name}.d_key"
            )
            mask_selects.append(f"({table_name}.d_key IS NOT NULL) AS {column_name}")
            dimension_entry[column_name] = {
                "build_size": build_size,
                "selectivity": selectivity,
                "predicate_type": predicate_type,
                "selected_rows": selected_rows,
            }

    materialized_selects = ",\n            ".join(
        ["b.d_key"]
        + [f"b.dummy_s{index}" for index in range(1, spec.dummy_string_cols + 1)]
        + mask_selects
    )
    con.execute(
        f"""
        CREATE TABLE {final_table_name} AS
        SELECT
            {materialized_selects}
        FROM {base_table_name} AS b
            {' '.join(join_clauses)}
        """
    )
    con.execute(f"DROP TABLE {base_table_name}")
    for table_name in created_temp_tables:
        con.execute(f"DROP TABLE {table_name}")
    dimension_metadata[final_table_name] = dimension_entry


def _create_fact_table(con: duckdb.DuckDBPyConnection, spec: CrossProductWorkloadConfig) -> None:
    fk_exprs: list[str] = []
    for build_size in spec.build_sizes:
        ordered_column = fact_fk_column_name(build_size, "ordered")
        uniform_column = fact_fk_column_name(build_size, "uniform")
        fk_exprs.append(
            "CAST(FLOOR(((i - 1) * {build_size})::DOUBLE / {fact_rows}) + 1 AS BIGINT) "
            "AS {column_name}".format(
                build_size=build_size,
                fact_rows=spec.fact_rows,
                column_name=ordered_column,
            )
        )
        fk_exprs.append(
            f"CAST(FLOOR(random() * {build_size}) + 1 AS BIGINT) AS {uniform_column}"
        )
    select_list = ",\n            ".join(["i AS fact_id"] + fk_exprs + ["i * 17 AS measure"])
    con.execute(
        f"""
        CREATE TABLE fact AS
        SELECT
            {select_list}
        FROM range(1, {spec.fact_rows + 1}) t(i)
        """
    )


def _write_database(spec: CrossProductWorkloadConfig, db_path: Path, queries_dir: Path) -> dict:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        seed_value = ((spec.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")
        dimension_metadata: dict[str, dict[str, dict]] = {}
        for build_size in spec.build_sizes:
            _create_dimension_table(con, spec, build_size, dimension_metadata)
        _create_fact_table(con, spec)
    finally:
        con.close()
    metadata = {
        "db_path": str(db_path),
        "queries_dir": str(queries_dir),
        "build_sizes": list(spec.build_sizes),
        "selectivities": list(spec.selectivities),
        "predicate_types": list(spec.predicate_types),
        "probe_clusterings": list(spec.probe_clusterings),
        "fact_rows": spec.fact_rows,
        "dummy_string_cols": spec.dummy_string_cols,
        "dummy_string_len": spec.dummy_string_len,
        "seed": spec.seed,
        "dimensions": dimension_metadata,
    }
    metadata_path = queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _render_query(
    spec: CrossProductWorkloadConfig,
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


def _write_queries(spec: CrossProductWorkloadConfig, queries_dir: Path) -> list[Path]:
    written: list[Path] = []
    for build_size in spec.build_sizes:
        for selectivity in spec.selectivities:
            for predicate_type in spec.predicate_types:
                for probe_clustering in spec.probe_clusterings:
                    path = queries_dir / (
                        f"b{build_size_label(build_size)}_"
                        f"{selectivity_label(selectivity)}_"
                        f"{predicate_type}_{probe_clustering}.sql"
                    )
                    path.write_text(
                        _render_query(
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


def _sweep_query_filename(
    case: ResolvedSweepCase,
    query_type: CrossProductQueryTypeConfig,
    probe_layout: str,
) -> str:
    if case.legacy_build_label is not None:
        query_type_label = (
            query_type.kind
            if query_type.kind != "seasonal" or query_type.cluster_count == 10
            else _query_type_id(query_type)
        )
        return (
            f"b{case.legacy_build_label}_{selectivity_label(case.selectivity)}_"
            f"{query_type_label}_{probe_layout}.sql"
        )
    return f"{case.case_id}_{_query_type_id(query_type)}_{probe_layout}.sql"


def _render_sweep_query(
    definition: CrossProductExperimentDefinition,
    case: ResolvedSweepCase,
    query_type: CrossProductQueryTypeConfig,
    probe_layout: str,
    *,
    fact_relation: str,
    row_group_fact: bool,
) -> str:
    payload_exprs = [
        f"    ANY_VALUE(d{join_index}.dummy_s{column_index}) AS "
        f"d{join_index}_dummy_s{column_index}"
        for join_index in range(1, case.join_count + 1)
        for column_index in range(1, definition.fixed.dummy_string_cols + 1)
    ]
    select_clause = ",\n".join(payload_exprs) if payload_exprs else "    COUNT(*) AS row_count"
    joins: list[str] = []
    predicates: list[str] = []
    for join_index in range(1, case.join_count + 1):
        dimension_table = _sweep_dimension_table_name(case, join_index, query_type)
        fk_column = (
            f"fk_j{join_index:02d}_{probe_layout}{_key_layout_suffix(case, query_type)}"
            if row_group_fact
            else _sweep_fk_column_name(case, join_index, probe_layout, query_type)
        )
        joins.append(
            f"JOIN {dimension_table} AS d{join_index}\n"
            f"  ON f.{fk_column} = d{join_index}.d_key"
        )
        predicates.append(
            f"d{join_index}.{_sweep_mask_column(case, query_type)}"
        )
    return f"""SELECT
{select_clause}
FROM {fact_relation} AS f
{' '.join(joins)}
WHERE {' AND '.join(predicates)};
"""


def _write_sweep_workload(
    spec: CrossProductWorkloadConfig,
    db_path: Path,
    queries_dir: Path,
    artifact_dir: Path,
    *,
    resume_density: bool = False,
) -> tuple[dict, tuple[Path, ...], Path, Path, Path]:
    definition = spec.experiment_definition
    assert definition is not None
    cases = _resolve_sweep_cases(definition, storage_kind=spec.storage_kind)
    if resume_density and not _fixed_density(definition):
        raise ValueError("partial database recovery is only supported for fixed density")
    if db_path.exists() and not resume_density:
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    connection_options = {}
    if _fixed_density(definition):
        preparation = spec.preparation
        connection_options = {"threads": preparation.threads, "preserve_insertion_order": True}
        for name in ("memory_limit", "max_temp_directory_size"):
            value = getattr(preparation, name)
            if value is not None:
                connection_options[name] = value
        if preparation.temp_directory is not None:
            connection_options["temp_directory"] = str((artifact_dir / preparation.temp_directory).resolve())
    con = duckdb.connect(str(db_path), config=connection_options)
    dimensions: dict[str, list[dict]] = {}
    try:
        seed_value = ((definition.fixed.seed % 10_000_000) + 0.5) / 10_000_001
        con.execute(f"SELECT setseed({seed_value})")
        dimension_groups: dict[tuple[str, int], list[ResolvedSweepCase]] = {}
        key_queries: dict[tuple[str, int], CrossProductQueryTypeConfig] = {}
        for case in cases:
            for join_index in range(1, case.join_count + 1):
                for query_type in _case_query_types(definition, case):
                    group_key = (_sweep_dimension_table_name(case, join_index, query_type), join_index)
                    group_cases = dimension_groups.setdefault(group_key, [])
                    if case not in group_cases:
                        group_cases.append(case)
                    key_queries[group_key] = query_type
        if _fixed_density(definition):
            physical_dimensions = _create_fixed_density_tables(con, definition, cases, spec.preparation)
        else:
            physical_dimensions = {
                group_key: _create_sweep_dimension_table(con, definition, group_cases,
                                                        group_key[1], key_queries[group_key])
                for group_key, group_cases in dimension_groups.items()
            }
        for case in cases:
            keys = dict.fromkeys((_sweep_dimension_table_name(case, j, q), j)
                                 for j in range(1, case.join_count + 1)
                                 for q in _case_query_types(definition, case))
            dimensions[case.case_id] = [physical_dimensions[key] for key in keys]
        separate_fact_databases = (
            definition.sweep.kind == "row_group_size"
            or definition.fixed.row_group_size != 122_880
        )
        if not separate_fact_databases and not _fixed_density(definition):
            _create_shared_sweep_fact_table(con, definition, cases)
    finally:
        con.close()

    setup_sql_paths: tuple[Path, ...] = ()
    row_group_storage: dict[str, dict] = {}
    row_group_aliases: dict[str, str] = {}
    if separate_fact_databases:
        setup_statements = ["-- Generated native DuckDB fact row-group variants"]
        for case in cases:
            fact_db_path = artifact_dir / f"fact_{case.case_id}.duckdb"
            alias, storage_metadata = _create_row_group_fact_database(
                definition,
                case,
                fact_db_path,
            )
            row_group_aliases[case.case_id] = alias
            row_group_storage[case.case_id] = storage_metadata
            setup_statements.append(
                f"ATTACH {sql_literal(str(fact_db_path.resolve()))} "
                f"AS {quote_identifier(alias)};"
            )
        setup_path = artifact_dir / "row_group_setup.sql"
        setup_path.write_text("\n".join(setup_statements) + "\n", encoding="utf-8")
        setup_sql_paths = (setup_path.resolve(),)

    query_catalog: dict[str, dict] = {}
    for case in cases:
        query_types = _case_query_types(definition, case)
        for query_type in query_types:
            for probe_layout in query_type.probe_layouts:
                filename = _sweep_query_filename(case, query_type, probe_layout)
                if filename in query_catalog:
                    raise ValueError(f"query definitions generate duplicate filename: {filename}")
                query_path = queries_dir / filename
                row_group_fact = separate_fact_databases
                fact_relation = (
                    f"{quote_identifier(row_group_aliases[case.case_id])}.fact"
                    if row_group_fact
                    else "fact"
                )
                query_path.write_text(
                    _render_sweep_query(
                        definition,
                        case,
                        query_type,
                        probe_layout,
                        fact_relation=fact_relation,
                        row_group_fact=row_group_fact,
                    ),
                    encoding="utf-8",
                )
                expected_selectivity = case.probe_selectivity ** case.join_count
                query_catalog[filename] = {
                    "case_id": case.case_id,
                    "sweep_kind": case.sweep_kind,
                    "sweep_value": case.sweep_value,
                    "cardinality_mode": definition.sweep.cardinality_mode,
                    "query_type": query_type.kind,
                    "cluster_count": query_type.cluster_count,
                    "probe_layout": probe_layout,
                    "join_count": case.join_count,
                    "dimension_cardinality": case.dimension_cardinality,
                    "dimension_distinct_keys": case.dimension_distinct_keys,
                    "selected_distinct_keys": case.selected_distinct_keys,
                    "selected_build_rows": case.selected_build_rows,
                    "build_key_multiplicity": case.build_key_multiplicity,
                    "key_domain_size": case.key_domain_size,
                    "key_density": case.key_density,
                    "dimension_key_density": case.dimension_key_density,
                    "individual_join_selectivity": case.probe_selectivity,
                    "dimension_predicate_selectivity": case.selectivity,
                    "probe_selectivity": case.probe_selectivity,
                    "expected_query_selectivity": expected_selectivity,
                    "expected_output_multiplier": case.build_key_multiplicity ** case.join_count,
                    "expected_join_output_rows": (case.fact_rows * expected_selectivity
                                                  * case.build_key_multiplicity ** case.join_count),
                    "reference_domain_size": case.reference_domain_size,
                    "probe_key_distribution": "baseline_membership_conditioned" if _fixed_density(definition) else probe_layout,
                    "key_layout": "shoulder" if query_type.kind == "shoulder" else "spread",
                    "key_domain_min": 1,
                    "key_domain_max": case.key_domain_size,
                    "dimension_tables": [_sweep_dimension_table_name(case, j, query_type)
                                         for j in range(1, case.join_count + 1)],
                    "row_group_size": case.row_group_size,
                }

    definition_path = artifact_dir / "definition.json"
    cases_path = artifact_dir / "cases.json"
    query_catalog_path = queries_dir / "query_catalog.json"
    normalized_definition = {
        "schema_version": 1,
        "workload_kind": "cross_product",
        "parameter_descriptions": CROSS_PRODUCT_PARAMETER_DESCRIPTIONS,
        "query_type_descriptions": CROSS_PRODUCT_QUERY_TYPE_DESCRIPTIONS,
        "supported_experiment_configurations": CROSS_PRODUCT_EXPERIMENT_CONFIGURATIONS,
        "fixed": asdict(definition.fixed),
        "key_domain_size_mode": (
            "dimension_cardinality"
            if definition.fixed.key_domain_size is None
            else definition.fixed.key_domain_size
            if definition.fixed.key_domain_size == "dimension_distinct_keys"
            else "fixed"
        ),
        "sweep": asdict(definition.sweep),
        "query_types": [asdict(query_type) for query_type in definition.query_types],
    }
    definition_path.write_text(
        json.dumps(normalized_definition, indent=2) + "\n",
        encoding="utf-8",
    )
    case_entries = []
    for case in cases:
        entry = asdict(case)
        entry["dimensions"] = dimensions[case.case_id]
        if case.case_id in row_group_storage:
            entry["fact_storage"] = row_group_storage[case.case_id]
        case_entries.append(entry)
    cases_path.write_text(
        json.dumps({"schema_version": 1, "cases": case_entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    query_catalog_path.write_text(
        json.dumps({"schema_version": 1, "queries": query_catalog}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "db_path": str(db_path),
        "queries_dir": str(queries_dir),
        "definition_path": str(definition_path),
        "cases_path": str(cases_path),
        "query_catalog_path": str(query_catalog_path),
        "sweep_kind": definition.sweep.kind,
        "case_count": len(cases),
        "query_count": len(query_catalog),
        "seed": definition.fixed.seed,
    }
    metadata_path = queries_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return (
        metadata,
        setup_sql_paths,
        definition_path.resolve(),
        cases_path.resolve(),
        query_catalog_path.resolve(),
    )


def _read_table_schemas(db_path: Path) -> dict[str, list[tuple[str, str]]]:
    con = duckdb.connect(str(db_path), read_only=True)
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
                    f"DESCRIBE {quote_identifier(table_name)}"
                ).fetchall()
            ]
            for table_name in tables
        }
    finally:
        con.close()


def _write_ducklake_sql(
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
        f"ATTACH {sql_literal('ducklake:' + str(metadata_path))} "
        f"AS {quote_identifier(catalog_alias)} "
        f"(DATA_PATH {sql_literal(str(data_path))}, DATA_INLINING_ROW_LIMIT 0);"
    )
    bootstrap_statements = [
        "-- Generated by run_benchmarks.py",
        *(
            [
                "LOAD httpfs;",
                "SET enable_external_file_cache=false;",
                "SET enable_http_metadata_cache=false;",
            ]
            if use_http_data
            else []
        ),
        "LOAD ducklake;",
        attach_sql,
    ]
    setup_statements = [
        "-- Generated by run_benchmarks.py",
        *(
            [
                "LOAD httpfs;",
                "SET enable_external_file_cache=false;",
                "SET enable_http_metadata_cache=false;",
            ]
            if use_http_data
            else []
        ),
        "LOAD ducklake;",
        attach_sql,
    ]
    for table_name in sorted(catalog):
        if table_name not in table_schemas:
            raise ValueError(f"missing schema for exported table {table_name}")
        column_sql = ", ".join(
            f"{quote_identifier(column_name)} {column_type}"
            for column_name, column_type in table_schemas[table_name]
        )
        bootstrap_statements.append(
            f"CREATE TABLE {quote_identifier(catalog_alias)}.{quote_identifier(table_name)} "
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
        file_list_sql = ", ".join(sql_literal(path) for path in data_files)
        bootstrap_statements.append(
            f"CALL ducklake_add_data_files({sql_literal(catalog_alias)}, "
            f"{sql_literal(table_name)}, [{file_list_sql}]);"
        )
        setup_statements.append(
            f"CREATE OR REPLACE TEMP VIEW {quote_identifier(table_name)} AS "
            f"SELECT * FROM {quote_identifier(catalog_alias)}.{quote_identifier(table_name)};"
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


def prepare_cross_product_workload(
    spec: CrossProductWorkloadConfig,
    artifact_dir: Path,
    *,
    resume_density: bool = False,
) -> PreparedWorkload:
    _validate(spec)
    queries_dir = artifact_dir / "queries"
    db_path = artifact_dir / "benchmark.duckdb"
    definition_path: Path | None = None
    cases_path: Path | None = None
    query_catalog_path: Path | None = None
    generated_setup_sql_paths: tuple[Path, ...] = ()
    if spec.experiment_definition is None:
        _write_database(spec, db_path, queries_dir)
        _write_queries(spec, queries_dir)
    else:
        (
            _,
            generated_setup_sql_paths,
            definition_path,
            cases_path,
            query_catalog_path,
        ) = _write_sweep_workload(spec, db_path, queries_dir, artifact_dir,
                                 resume_density=resume_density)
    metadata_path = queries_dir / "metadata.json"
    runtime_db_path = db_path.resolve()
    remote_parquet_catalog_path: Path | None = None
    parquet_catalog_path: Path | None = None
    ducklake_bootstrap_sql_path: Path | None = None
    ducklake_metadata_path: Path | None = None
    ducklake_data_path: Path | None = None
    ducklake_catalog_alias: str | None = None
    setup_sql_paths: tuple[Path, ...] = generated_setup_sql_paths
    storage_mode = "duckdb"
    if spec.storage_kind == "ducklake":
        if generated_setup_sql_paths:
            raise ValueError("DuckLake storage cannot be combined with native row-group variants")
        table_schemas = _read_table_schemas(runtime_db_path)
        parquet_catalog_path = export_parquet_catalog(
            runtime_db_path,
            artifact_dir,
            root_dir_name="parquet",
            catalog_filename="parquet_catalog.json",
            file_size_mb=spec.parquet_file_size_mb,
            exclude_empty_files=True,
        )
        (
            runtime_db_path,
            ducklake_bootstrap_sql_path,
            setup_sql_path,
            ducklake_metadata_path,
            ducklake_data_path,
            ducklake_catalog_alias,
        ) = _write_ducklake_sql(
            artifact_dir,
            parquet_catalog_path,
            table_schemas,
            use_http_data=spec.ducklake_http_latency_ms is not None,
        )
        setup_sql_paths = (setup_sql_path,)
        staging_db_path = (artifact_dir / "benchmark.duckdb").resolve()
        if staging_db_path.exists():
            staging_db_path.unlink()
        storage_mode = "ducklake"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["storage"] = {
            "kind": "ducklake",
            "parquet_file_size_mb": spec.parquet_file_size_mb,
            "parquet_catalog_path": str(parquet_catalog_path),
            "http_latency_ms": spec.ducklake_http_latency_ms,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    elif spec.remote_http_parquet_latency_ms is not None:
        remote_parquet_catalog_path = export_remote_parquet_catalog(runtime_db_path, artifact_dir)
        runtime_db_path = (artifact_dir / "benchmark.duckdb").resolve()
        if runtime_db_path.exists():
            runtime_db_path.unlink()
        storage_mode = "remote_http_parquet"
    manifest = PreparedWorkload(
        name=spec.name,
        kind=spec.kind,
        source_root=artifact_dir.resolve(),
        artifact_dir=artifact_dir.resolve(),
        db_path=runtime_db_path,
        queries_dir=queries_dir.resolve(),
        metadata_path=metadata_path.resolve(),
        init_sql_paths=(),
        setup_sql_paths=setup_sql_paths,
        post_variant_setup_sql=spec.post_variant_setup_sql,
        storage_mode=storage_mode,
        remote_http_parquet_latency_ms=spec.remote_http_parquet_latency_ms,
        remote_parquet_catalog_path=remote_parquet_catalog_path,
        parquet_catalog_path=parquet_catalog_path,
        ducklake_bootstrap_sql_path=ducklake_bootstrap_sql_path,
        ducklake_metadata_path=ducklake_metadata_path,
        ducklake_data_path=ducklake_data_path,
        ducklake_catalog_alias=ducklake_catalog_alias,
        ducklake_http_latency_ms=spec.ducklake_http_latency_ms,
        definition_path=definition_path,
        cases_path=cases_path,
        query_catalog_path=query_catalog_path,
        prepared_at=utc_now_iso(),
    )
    write_manifest(artifact_dir / "manifest.json", manifest)
    return manifest
