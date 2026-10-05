from __future__ import annotations

from dataclasses import dataclass


VARIANT_ORDER = (
    "no prefilter",
    "sip",
    "sip + min/max",
    "prf uncompressed",
    "prf with fallback",
    "rpt none",
    "rpt cascade",
    "rpt prf",
)

WORKLOAD_LABELS = {
    "join-pruning-workload": "Join Pruning Benchmark",
    "join-pruning-workload-ducklake": "Join Pruning Benchmark (DuckLake)",
    "join-pruning-workload-ducklake-http-10ms": "Join Pruning Benchmark (DuckLake, +10 ms/request)",
    "string-joins": "String Join Benchmark",
    "ssb-ordered": "SSB Ordered",
    "ssb-ordered-intro": "SSB Ordered",
    "tpch-skew": "TPC-H Skew",
    "tpch-skew-intro": "TPC-H Skew",
    "tpch-skew-ducklake-http-10ms": "TPC-H Skew (DuckLake, +10 ms/request)",
    "join-order-benchmark": "Join Order Benchmark",
    "jpb-sweep-joins": "Join count",
    "jpb-sweep-multiplicity": "Build-key multiplicity",
    "jpb-sweep-selectivity": "Selectivity",
    "jpb-sweep-density": "Key domain",
    "jpb-sweep-clusters": "Cluster count",
    "jpb-sweep-row-groups": "Row-group size",
}

QUERY_LABELS = {
    "b100k_s0010_range_ordered.sql": "R-S",
    "b10m_s0010_range_ordered.sql": "R-L",
    "b100k_s0010_shoulder_ordered.sql": "Sh-S",
    "b10m_s0010_shoulder_ordered.sql": "Sh-L",
    "b100k_s0010_seasonal_ordered.sql": "Se-S",
    "b10m_s0010_seasonal_ordered.sql": "Se-L",
    "b100k_s0010_scattered_ordered.sql": "Sc-S",
    "b10m_s0010_scattered_ordered.sql": "Sc-L",
    "b100k_s0010_range_uniform.sql": "R-S",
    "b10m_s0010_range_uniform.sql": "R-L",
    "b100k_s0010_shoulder_uniform.sql": "Sh-S",
    "b10m_s0010_shoulder_uniform.sql": "Sh-L",
    "b100k_s0010_seasonal_uniform.sql": "Se-S",
    "b10m_s0010_seasonal_uniform.sql": "Se-L",
    "b100k_s0010_scattered_uniform.sql": "Sc-S",
    "b10m_s0010_scattered_uniform.sql": "Sc-L",
    "b100m_s0010_range_ordered.sql": "R-L",
    "b100m_s0010_shoulder_ordered.sql": "Sh-L",
    "b100m_s0010_seasonal_ordered.sql": "Se-L",
    "b100m_s0010_scattered_ordered.sql": "Sc-L",
    "b100m_s0010_range_uniform.sql": "R-L",
    "b100m_s0010_shoulder_uniform.sql": "Sh-L",
    "b100m_s0010_seasonal_uniform.sql": "Se-L",
    "b100m_s0010_scattered_uniform.sql": "Sc-L",
    "q01.sql": "Q1",
    "q02.sql": "Q2",
    "q03.sql": "Q3",
    "q04.sql": "Q4",
    "q05.sql": "Q5",
    "q06.sql": "Q6",
    "q07.sql": "Q7",
    "q08.sql": "Q8",
    "q09.sql": "Q9",
    "q10.sql": "Q10",
    "q11.sql": "Q11",
    "q12.sql": "Q12",
    "q13.sql": "Q13",
    "q14.sql": "Q14",
    "q15.sql": "Q15",
    "q16.sql": "Q16",
    "q17.sql": "Q17",
    "q18.sql": "Q18",
    "q19.sql": "Q19",
    "q20.sql": "Q20",
    "q21.sql": "Q21",
    "q22.sql": "Q22",
    "airport_ordered_antarctica.sql": "Airport ordered",
    "airport_uniform_antarctica.sql": "Airport uniform",
    "uuid_v7_like_ordered.sql": "UUIDv7-like",
    "uuid_v4_like_uniform.sql": "UUIDv4-like",
}

JPB_ORDERED_QUERIES = (
    "b100k_s0010_range_ordered.sql",
    "b100k_s0010_shoulder_ordered.sql",
    "b100k_s0010_seasonal_ordered.sql",
    "b100k_s0010_scattered_ordered.sql",
    "b10m_s0010_range_ordered.sql",
    "b10m_s0010_shoulder_ordered.sql",
    "b10m_s0010_seasonal_ordered.sql",
    "b10m_s0010_scattered_ordered.sql",
)

JPB_UNIFORM_QUERIES = (
    "b100k_s0010_range_uniform.sql",
    "b100k_s0010_shoulder_uniform.sql",
    "b100k_s0010_seasonal_uniform.sql",
    "b100k_s0010_scattered_uniform.sql",
    "b10m_s0010_range_uniform.sql",
    "b10m_s0010_shoulder_uniform.sql",
    "b10m_s0010_seasonal_uniform.sql",
    "b10m_s0010_scattered_uniform.sql",
)

JPB_ALL_QUERIES = JPB_ORDERED_QUERIES + JPB_UNIFORM_QUERIES

# Native paper experiments use a 100-million-row large dimension; the
# separate DuckLake experiments retain their 10-million-row dimension.
JPB_NATIVE_ORDERED_QUERIES = tuple(
    query.replace("b10m_", "b100m_") for query in JPB_ORDERED_QUERIES
)
JPB_NATIVE_UNIFORM_QUERIES = tuple(
    query.replace("b10m_", "b100m_") for query in JPB_UNIFORM_QUERIES
)
JPB_NATIVE_ALL_QUERIES = JPB_NATIVE_ORDERED_QUERIES + JPB_NATIVE_UNIFORM_QUERIES

TPCH_ROW_GROUP_QUERIES = (
    "q02.sql",
    "q04.sql",
    "q07.sql",
    "q08.sql",
    "q09.sql",
    "q11.sql",
    "q16.sql",
    "q17.sql",
    "q19.sql",
)

TPCH_ROW_WISE_QUERIES = (
    "q03.sql",
    "q05.sql",
    "q10.sql",
    "q13.sql",
    "q14.sql",
    "q15.sql",
    "q18.sql",
    "q20.sql",
    "q22.sql",
)

TPCH_SELECTED_QUERIES = TPCH_ROW_GROUP_QUERIES + TPCH_ROW_WISE_QUERIES

TPCH_ALL_QUERIES = tuple(f"q{query_number:02d}.sql" for query_number in range(1, 23))

STRING_JOIN_QUERIES = (
    "airport_ordered_antarctica.sql",
    "airport_uniform_antarctica.sql",
    "uuid_v7_like_ordered.sql",
    "uuid_v4_like_uniform.sql",
)

SSB_ORDERED_QUERIES = ("q01.sql",)


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    metric: str
    workloads: tuple[str, ...]
    queries_by_workload: dict[str, tuple[str, ...]]
    width: str
    placement_width: str
    summary_key: str
    y_label: str
    query_workload_labels: bool = False
    probe_table: str | None = None
    show_query_labels: bool = True
    show_runtime_labels: bool = False
    annotate_runtime_values: bool = False
    runtime_log_axis: bool = False
    rows_log_axis: bool = False


@dataclass(frozen=True)
class SweepPanelSpec:
    workload: str
    sweep_kind: str
    title: str
    x_label: str
    x_scale: str
    log_base: int = 10
    x_ticks: tuple[int | float, ...] = ()


PARAMETER_SWEEP_PANELS = (
    SweepPanelSpec(
        workload="jpb-sweep-joins",
        sweep_kind="join_count",
        title="(a) Number of joins",
        x_label="Joins",
        x_scale="linear",
        x_ticks=(1, 2, 3, 4, 5),
    ),
    SweepPanelSpec(
        workload="jpb-sweep-multiplicity",
        sweep_kind="build_key_multiplicity",
        title="(b) Build-key multiplicity",
        x_label="Rows per key",
        x_scale="log",
        log_base=2,
        x_ticks=(1, 4, 16, 64, 128),
    ),
    SweepPanelSpec(
        workload="jpb-sweep-selectivity",
        sweep_kind="selectivity",
        title="(c) Selectivity",
        x_label="Dimension selectivity",
        x_scale="log",
        log_base=10,
        x_ticks=(0.0001, 0.001, 0.01, 0.1, 0.9),
    ),
    SweepPanelSpec(
        workload="jpb-sweep-density",
        sweep_kind="key_domain_size",
        title="(d) Key density",
        x_label="Key-domain size",
        x_scale="log",
        log_base=10,
        x_ticks=(5_000_000, 30_000_000, 100_000_000, 1_000_000_000),
    ),
    SweepPanelSpec(
        workload="jpb-sweep-clusters",
        sweep_kind="cluster_count",
        title="(e) Clustering",
        x_label="Clusters",
        x_scale="log",
        log_base=2,
        x_ticks=(2, 16, 128, 1_024, 8_192, 65_536),
    ),
    SweepPanelSpec(
        workload="jpb-sweep-row-groups",
        sweep_kind="row_group_size",
        title="(f) Row-group size",
        x_label="Rows per group",
        x_scale="log",
        log_base=2,
        x_ticks=(2_048, 122_880, 2_097_152),
    ),
)


JOIN_PRUNING_ORDERED = FigureSpec(
    slug="join_pruning_ordered",
    metric="runtime_and_probe_rows",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_NATIVE_ORDERED_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="row_group_pruning",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
    rows_log_axis=False,
)

INTRO_MOTIVATION = FigureSpec(
    slug="intro_motivation",
    metric="runtime_and_total_rows",
    workloads=("ssb-ordered-intro", "tpch-skew-intro"),
    queries_by_workload={
        "ssb-ordered-intro": SSB_ORDERED_QUERIES,
        "tpch-skew-intro": ("q02.sql",),
    },
    width="one-column",
    placement_width="one-column",
    summary_key="motivation",
    y_label="",
    query_workload_labels=True,
    show_query_labels=True,
    show_runtime_labels=True,
    runtime_log_axis=True,
    rows_log_axis=True,
)

ROW_GROUP_RUNTIME_EXTERNAL = FigureSpec(
    slug="row_group_runtime_external",
    metric="runtime_ms",
    workloads=("tpch-skew",),
    queries_by_workload={"tpch-skew": TPCH_ROW_GROUP_QUERIES},
    width="two-column",
    placement_width="two-column",
    summary_key="row_group_pruning",
    y_label="Execution time (ms)",
)

ROW_GROUP_ROWS_EXTERNAL = FigureSpec(
    slug="row_group_rows_external",
    metric="total_rows",
    workloads=("tpch-skew",),
    queries_by_workload=ROW_GROUP_RUNTIME_EXTERNAL.queries_by_workload,
    width="two-column",
    placement_width="two-column",
    summary_key="row_group_pruning",
    y_label="Cumulative rows scanned",
)

JOIN_PRUNING_UNIFORM = FigureSpec(
    slug="join_pruning_uniform",
    metric="runtime_and_probe_rows",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_NATIVE_UNIFORM_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="row_wise_filtering",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
    rows_log_axis=False,
)

JOIN_PRUNING_ORDERED_DUCKLAKE = FigureSpec(
    slug="join_pruning_ordered_ducklake",
    metric="runtime_and_files_read",
    workloads=("join-pruning-workload-ducklake",),
    queries_by_workload={"join-pruning-workload-ducklake": JPB_ORDERED_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="ducklake_file_pruning",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
)

JOIN_PRUNING_UNIFORM_DUCKLAKE = FigureSpec(
    slug="join_pruning_uniform_ducklake",
    metric="runtime_and_files_read",
    workloads=("join-pruning-workload-ducklake",),
    queries_by_workload={"join-pruning-workload-ducklake": JPB_UNIFORM_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="ducklake_file_pruning",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
)

JOIN_PRUNING_ORDERED_DUCKLAKE_HTTP_10MS = FigureSpec(
    slug="join_pruning_ordered_ducklake_http_10ms",
    metric="runtime_and_files_read",
    workloads=("join-pruning-workload-ducklake-http-10ms",),
    queries_by_workload={
        "join-pruning-workload-ducklake-http-10ms": JPB_ORDERED_QUERIES
    },
    width="two-column",
    placement_width="one-column",
    summary_key="ducklake_http_10ms_file_pruning",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
)

JOIN_PRUNING_UNIFORM_DUCKLAKE_HTTP_10MS = FigureSpec(
    slug="join_pruning_uniform_ducklake_http_10ms",
    metric="runtime_and_files_read",
    workloads=("join-pruning-workload-ducklake-http-10ms",),
    queries_by_workload={
        "join-pruning-workload-ducklake-http-10ms": JPB_UNIFORM_QUERIES
    },
    width="two-column",
    placement_width="one-column",
    summary_key="ducklake_http_10ms_file_pruning",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
)

ROW_WISE_RUNTIME_EXTERNAL = FigureSpec(
    slug="row_wise_runtime_external",
    metric="runtime_ms",
    workloads=("tpch-skew",),
    queries_by_workload={"tpch-skew": TPCH_ROW_WISE_QUERIES},
    width="two-column",
    placement_width="two-column",
    summary_key="row_wise_filtering",
    y_label="Execution time (ms)",
)

TPCH_SKEW_DUCKLAKE_HTTP_MOTIVATION = FigureSpec(
    slug="tpch_skew_ducklake_http_motivation",
    metric="runtime_and_files_read",
    workloads=("tpch-skew-ducklake-http-10ms",),
    queries_by_workload={"tpch-skew-ducklake-http-10ms": ("q02.sql",)},
    width="one-column",
    placement_width="one-column",
    summary_key="tpch_skew_ducklake_http",
    y_label="",
    probe_table="partsupp",
    query_workload_labels=True,
    show_query_labels=True,
    show_runtime_labels=True,
    runtime_log_axis=True,
    rows_log_axis=False,
)

TPCH_SKEW_DUCKLAKE_HTTP_RUNTIME = FigureSpec(
    slug="tpch_skew_ducklake_http_runtime",
    metric="runtime_ms",
    workloads=("tpch-skew-ducklake-http-10ms",),
    queries_by_workload={"tpch-skew-ducklake-http-10ms": TPCH_ALL_QUERIES},
    width="two-column",
    placement_width="two-column",
    summary_key="tpch_skew_ducklake_http",
    y_label="Execution time (ms)",
)

STRING_JOIN = FigureSpec(
    slug="string_join",
    metric="runtime_and_probe_rows",
    workloads=("string-joins",),
    queries_by_workload={"string-joins": STRING_JOIN_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="string_joins",
    y_label="",
    probe_table="fact",
    annotate_runtime_values=True,
    runtime_log_axis=True,
    rows_log_axis=False,
)

END_TO_END_RUNTIME = FigureSpec(
    slug="end_to_end_runtime",
    metric="runtime_ms",
    workloads=("join-pruning-workload", "tpch-skew", "join-order-benchmark"),
    queries_by_workload={},
    width="two-column",
    placement_width="two-column",
    summary_key="end_to_end",
    y_label="Execution time (ms)",
    query_workload_labels=True,
)

FIGURES = (
    INTRO_MOTIVATION,
    TPCH_SKEW_DUCKLAKE_HTTP_MOTIVATION,
    JOIN_PRUNING_ORDERED,
    ROW_GROUP_RUNTIME_EXTERNAL,
    ROW_GROUP_ROWS_EXTERNAL,
    JOIN_PRUNING_UNIFORM,
    JOIN_PRUNING_ORDERED_DUCKLAKE,
    JOIN_PRUNING_UNIFORM_DUCKLAKE,
    JOIN_PRUNING_ORDERED_DUCKLAKE_HTTP_10MS,
    JOIN_PRUNING_UNIFORM_DUCKLAKE_HTTP_10MS,
    ROW_WISE_RUNTIME_EXTERNAL,
    TPCH_SKEW_DUCKLAKE_HTTP_RUNTIME,
    STRING_JOIN,
    END_TO_END_RUNTIME,
)
