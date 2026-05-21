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
    "tpch-skew": "TPC-H Skew",
    "join-order-benchmark": "Join Order Benchmark",
}

QUERY_LABELS = {
    "b100k_s0005_range_ordered.sql": "R-S",
    "b10m_s0050_range_ordered.sql": "R-L",
    "b100k_s0005_shoulder_ordered.sql": "Sh-S",
    "b10m_s0050_shoulder_ordered.sql": "Sh-L",
    "b100k_s0005_seasonal_ordered.sql": "Se-S",
    "b10m_s0050_seasonal_ordered.sql": "Se-L",
    "b100k_s0005_scattered_ordered.sql": "Sc-S",
    "b10m_s0050_scattered_ordered.sql": "Sc-L",
    "b100k_s0005_range_uniform.sql": "R-S",
    "b10m_s0050_range_uniform.sql": "R-L",
    "b100k_s0005_shoulder_uniform.sql": "Sh-S",
    "b10m_s0050_shoulder_uniform.sql": "Sh-L",
    "b100k_s0005_seasonal_uniform.sql": "Se-S",
    "b10m_s0050_seasonal_uniform.sql": "Se-L",
    "b100k_s0005_scattered_uniform.sql": "Sc-S",
    "b10m_s0050_scattered_uniform.sql": "Sc-L",
    "q02.sql": "Q2",
    "q03.sql": "Q3",
    "q04.sql": "Q4",
    "q05.sql": "Q5",
    "q07.sql": "Q7",
    "q08.sql": "Q8",
    "q09.sql": "Q9",
    "q10.sql": "Q10",
    "q11.sql": "Q11",
    "q13.sql": "Q13",
    "q14.sql": "Q14",
    "q15.sql": "Q15",
    "q16.sql": "Q16",
    "q17.sql": "Q17",
    "q18.sql": "Q18",
    "q19.sql": "Q19",
    "q20.sql": "Q20",
    "q22.sql": "Q22",
}

JPB_ORDERED_QUERIES = (
    "b100k_s0005_range_ordered.sql",
    "b10m_s0050_range_ordered.sql",
    "b100k_s0005_shoulder_ordered.sql",
    "b10m_s0050_shoulder_ordered.sql",
    "b100k_s0005_seasonal_ordered.sql",
    "b10m_s0050_seasonal_ordered.sql",
    "b100k_s0005_scattered_ordered.sql",
    "b10m_s0050_scattered_ordered.sql",
)

JPB_UNIFORM_QUERIES = (
    "b100k_s0005_range_uniform.sql",
    "b10m_s0050_range_uniform.sql",
    "b100k_s0005_shoulder_uniform.sql",
    "b10m_s0050_shoulder_uniform.sql",
    "b100k_s0005_seasonal_uniform.sql",
    "b10m_s0050_seasonal_uniform.sql",
    "b100k_s0005_scattered_uniform.sql",
    "b10m_s0050_scattered_uniform.sql",
)


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    metric: str
    workloads: tuple[str, ...]
    queries_by_workload: dict[str, tuple[str, ...]]
    width: str
    summary_key: str
    y_label: str
    query_workload_labels: bool = False


ROW_GROUP_RUNTIME_SYNTHETIC = FigureSpec(
    slug="row_group_runtime_synthetic",
    metric="runtime_ms",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_ORDERED_QUERIES},
    width="one-column",
    summary_key="row_group_pruning",
    y_label="Execution time (ms)",
)

ROW_GROUP_RUNTIME_EXTERNAL = FigureSpec(
    slug="row_group_runtime_external",
    metric="runtime_ms",
    workloads=("tpch-skew",),
    queries_by_workload={
        "tpch-skew": (
            "q02.sql",
            "q04.sql",
            "q07.sql",
            "q08.sql",
            "q09.sql",
            "q11.sql",
            "q16.sql",
            "q17.sql",
            "q19.sql",
        ),
    },
    width="two-column",
    summary_key="row_group_pruning",
    y_label="Execution time (ms)",
)

ROW_GROUP_ROWS_SYNTHETIC = FigureSpec(
    slug="row_group_rows_synthetic",
    metric="total_rows",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_ORDERED_QUERIES},
    width="one-column",
    summary_key="row_group_pruning",
    y_label="Cumulative rows scanned",
)

ROW_GROUP_ROWS_EXTERNAL = FigureSpec(
    slug="row_group_rows_external",
    metric="total_rows",
    workloads=("tpch-skew",),
    queries_by_workload=ROW_GROUP_RUNTIME_EXTERNAL.queries_by_workload,
    width="two-column",
    summary_key="row_group_pruning",
    y_label="Cumulative rows scanned",
)

ROW_WISE_RUNTIME_SYNTHETIC = FigureSpec(
    slug="row_wise_runtime_synthetic",
    metric="runtime_ms",
    workloads=("join-pruning-workload",),
    queries_by_workload={"join-pruning-workload": JPB_UNIFORM_QUERIES},
    width="one-column",
    summary_key="row_wise_filtering",
    y_label="Execution time (ms)",
)

ROW_WISE_RUNTIME_EXTERNAL = FigureSpec(
    slug="row_wise_runtime_external",
    metric="runtime_ms",
    workloads=("tpch-skew",),
    queries_by_workload={
        "tpch-skew": (
            "q03.sql",
            "q05.sql",
            "q10.sql",
            "q13.sql",
            "q14.sql",
            "q15.sql",
            "q18.sql",
            "q20.sql",
            "q22.sql",
        ),
    },
    width="two-column",
    summary_key="row_wise_filtering",
    y_label="Execution time (ms)",
)

END_TO_END_RUNTIME = FigureSpec(
    slug="end_to_end_runtime",
    metric="runtime_ms",
    workloads=("join-pruning-workload", "tpch-skew", "join-order-benchmark"),
    queries_by_workload={},
    width="two-column",
    summary_key="end_to_end",
    y_label="Execution time (ms)",
    query_workload_labels=True,
)

FIGURES = (
    ROW_GROUP_RUNTIME_SYNTHETIC,
    ROW_GROUP_RUNTIME_EXTERNAL,
    ROW_GROUP_ROWS_SYNTHETIC,
    ROW_GROUP_ROWS_EXTERNAL,
    ROW_WISE_RUNTIME_SYNTHETIC,
    ROW_WISE_RUNTIME_EXTERNAL,
    END_TO_END_RUNTIME,
)
