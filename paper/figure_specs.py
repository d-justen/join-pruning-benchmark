from __future__ import annotations

from dataclasses import dataclass


VARIANT_ORDER = (
    "no prefilter",
    "sip",
    "sip + min/max",
    "prf with fallback",
)

WORKLOAD_LABELS = {
    "join-pruning-workload": "Join Pruning Benchmark",
    "string-joins": "String Join Benchmark",
    "ssb-ordered": "SSB Ordered",
    "ssb-ordered-intro": "SSB Ordered",
    "tpch-skew": "TPC-H Skew",
    "tpch-skew-intro": "TPC-H Skew",
    "join-order-benchmark": "Join Order Benchmark",
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


JOIN_PRUNING_ORDERED = FigureSpec(
    slug="join_pruning_ordered",
    metric="runtime_and_probe_rows",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_ORDERED_QUERIES},
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
    queries_by_workload={"join-pruning-workload": JPB_UNIFORM_QUERIES},
    width="two-column",
    placement_width="one-column",
    summary_key="row_wise_filtering",
    y_label="",
    probe_table="fact",
    runtime_log_axis=True,
    rows_log_axis=False,
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
    JOIN_PRUNING_ORDERED,
    ROW_GROUP_RUNTIME_EXTERNAL,
    ROW_GROUP_ROWS_EXTERNAL,
    JOIN_PRUNING_UNIFORM,
    ROW_WISE_RUNTIME_EXTERNAL,
    STRING_JOIN,
    END_TO_END_RUNTIME,
)
