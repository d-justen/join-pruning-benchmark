import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from benchmarking import load_experiment_config
from benchmarking.experiment_runner import run_experiment
from paper.figure_specs import PARAMETER_SWEEP_PANELS, VARIANT_ORDER
from paper.figures.build_all import (
    build_figures,
    _draw_parameter_sweep_overview,
    _draw_relative_probe_cardinality_boxplot,
    _draw_runtime_probe_bars,
)
from paper.figures.data import (
    MetricRow,
    PlanMetricRow,
    ProbeScanRow,
    _extract_join_probe_metrics,
    _extract_probe_scan_rows,
    _prf_phase_metrics,
    _runtime_ms_with_profile_fallback,
    build_relative_probe_cardinality_values,
    build_runtime_probe_table,
    summarize_benchmark_metrics,
)


class PaperSsbTest(unittest.TestCase):
    def test_scaled_parameter_sweep_suite_runs_and_builds_common_figure(self) -> None:
        binary = shutil.which("duckdb")
        if binary is None:
            self.skipTest("DuckDB CLI is not installed")
        fixed = {
            "fact_rows": 4096,
            "dimension_cardinality": 128,
            "key_domain_size": "dimension_cardinality",
            "selectivity": 0.5,
            "join_count": 1,
            "build_key_multiplicity": 1,
            "probe_layout": "ordered",
            "row_group_size": 122880,
            "dummy_string_cols": 0,
            "seed": 7,
        }

        def workload(
            name,
            kind,
            values,
            *,
            query_types=None,
            fixed_overrides=None,
            sweep_extra=None,
        ):
            workload_fixed = {**fixed, **(fixed_overrides or {})}
            params = {
                "fixed": workload_fixed,
                "sweep": {"kind": kind, "values": values, **(sweep_extra or {})},
            }
            if query_types is not None:
                params["query_types"] = query_types
            return {"name": name, "kind": "cross_product", "params": params}

        shoulder_scattered_ordered = [
            {"kind": "shoulder", "probe_layouts": ["ordered"]},
            {"kind": "scattered", "probe_layouts": ["ordered"]},
        ]
        workloads = [
            workload(
                "jpb-sweep-joins",
                "join_count",
                [1, 2],
                query_types=[
                    {"kind": "shoulder", "probe_layouts": ["uniform"]},
                    {"kind": "scattered", "probe_layouts": ["uniform"]},
                ],
                fixed_overrides={"probe_layout": "uniform"},
            ),
            workload(
                "jpb-sweep-multiplicity",
                "build_key_multiplicity",
                [1, 2, 4],
                query_types=shoulder_scattered_ordered,
                fixed_overrides={"key_domain_size": "dimension_distinct_keys"},
                sweep_extra={"cardinality_mode": "fixed_dimension_cardinality"},
            ),
            workload(
                "jpb-sweep-selectivity",
                "selectivity",
                [0.25, 0.5],
                query_types=shoulder_scattered_ordered,
            ),
            workload(
                "jpb-sweep-density",
                "key_domain_size",
                [128, 256],
                query_types=shoulder_scattered_ordered,
            ),
            workload(
                "jpb-sweep-clusters",
                "cluster_count",
                [2, 4],
                query_types=None,
            ),
            workload(
                "jpb-sweep-row-groups",
                "row_group_size",
                [2048, 2097152],
                query_types=shoulder_scattered_ordered,
            ),
        ]
        systems = [
            {
                "name": "duckdb",
                "kind": "binary",
                "binary_path": binary,
                "variants": [
                    {"label": "no prefilter", "setup_sql": []}
                ],
            }
        ]
        repository_root = Path(__file__).resolve().parents[1]
        rpt_binary = repository_root / "dynamic-predicate-transfer/build/release/duckdb"
        if rpt_binary.is_file():
            systems.append(
                {
                    "name": "rpt-plus-smoke",
                    "kind": "binary",
                    "binary_path": str(rpt_binary),
                    "workload_names": [workload["name"] for workload in workloads],
                    "variants": [
                        {"label": "rpt none", "setup_sql": ["SET rpt_filter = 'none';"]},
                        {"label": "rpt cascade", "setup_sql": ["SET rpt_filter = 'cascade';"]},
                        {
                            "label": "rpt prf",
                            "setup_sql": [
                                "SET rpt_filter = 'prf';",
                                "SET rpt_prf_scan_pushdown = true;",
                            ],
                        },
                    ],
                }
            )
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "sweeps.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "scaled-paper-sweeps",
                        "output_root": str(root / "out"),
                        "overwrite_output": True,
                        "workloads": workloads,
                        "systems": systems,
                        "benchmark": {
                            "repetitions": 1,
                            "collect_profiles": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths = run_experiment(load_experiment_config(config_path))
            output_dir = build_figures(paths.root, None)
            figure = output_dir / "figures/parameter_sweep_overview.pdf"
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(figure.is_file())
            self.assertEqual(
                set(summary["sections"]["parameter_sweeps"]),
                {panel.workload for panel in PARAMETER_SWEEP_PANELS},
            )
            if rpt_binary.is_file():
                variants = {
                    point["variant_label"]
                    for panel_summary in summary["sections"]["parameter_sweeps"].values()
                    for point in panel_summary["points"]
                }
                self.assertTrue({"rpt none", "rpt cascade", "rpt prf"}.issubset(variants))

    def test_paper_sweep_config_materializes_all_six_definitions(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            repository_root / "paper/configs/local_paper_sweeps.json"
        )
        workloads = {workload.name: workload for workload in config.workloads}
        self.assertEqual(
            set(workloads),
            {panel.workload for panel in PARAMETER_SWEEP_PANELS},
        )
        multiplicity = workloads["jpb-sweep-multiplicity"].experiment_definition
        self.assertEqual(
            multiplicity.fixed.key_domain_size,
            "dimension_distinct_keys",
        )
        joins = workloads["jpb-sweep-joins"].experiment_definition
        self.assertEqual(joins.sweep.values, (1, 2, 3, 4, 5))
        self.assertEqual(joins.fixed.selectivity, 0.1)
        rpt = next(system for system in config.systems if system.name == "rpt-plus")
        self.assertEqual(set(rpt.workload_names), set(workloads))

    def test_parameter_sweep_overview_uses_one_shared_figure(self) -> None:
        rows = []
        for panel in PARAMETER_SWEEP_PANELS:
            query_types = ("seasonal",) if panel.sweep_kind == "cluster_count" else (
                "shoulder",
                "scattered",
            )
            x_values = panel.x_ticks
            for variant_index, variant in enumerate(VARIANT_ORDER):
                for query_type in query_types:
                    for point_index, x_value in enumerate(x_values):
                        rows.append(
                            MetricRow(
                                panel.workload,
                                f"q-{query_type}-{x_value}.sql",
                                variant,
                                1.0 + variant_index + point_index,
                                None,
                                None,
                                {
                                    "sweep_kind": panel.sweep_kind,
                                    "sweep_value": x_value,
                                    "query_type": query_type,
                                },
                            )
                        )
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "sweeps.pdf"
            verification, summary = _draw_parameter_sweep_overview(
                output_path,
                rows=rows,
            )
            self.assertTrue(output_path.is_file())
            self.assertEqual(set(summary), {panel.workload for panel in PARAMETER_SWEEP_PANELS})
            self.assertFalse(verification["tick_label_overlap"])
            self.assertFalse(verification["text_outside_figure"])

    def test_zero_runtime_uses_profile_latency(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "profile.json"
            profile_path.write_text('{"latency": 0.000125}', encoding="utf-8")

            runtime_ms = _runtime_ms_with_profile_fallback(0.0, str(profile_path))

            self.assertEqual(runtime_ms, 0.125)

    def test_prf_phase_totals_use_elapsed_profile_fields(self) -> None:
        profile = {
            "children": [
                {"prefix_range_filter": {
                    "build_phase_elapsed_seconds": 0.003,
                    "compression_phase_elapsed_seconds": 0.001,
                    "build_worker_seconds": 0.2,
                }},
                {"prefix_range_filter": {
                    "build_phase_elapsed_seconds": 0.004,
                    "compression_phase_elapsed_seconds": 0.002,
                    "build_worker_seconds": 0.3,
                }},
            ]
        }
        self.assertEqual(_prf_phase_metrics(profile), (7.0, 3.0, 2))
        self.assertEqual(
            _prf_phase_metrics({"children": [{"prefix_range_filter": {
                "build_phase_elapsed_seconds": 0.003,
            }}]}),
            (3.0, None, 1),
        )

        rows = [
            MetricRow("workload", "q1.sql", "prf with fallback", 10, 5, 5,
                      prf_build_phase_ms=7.0, prf_analysis_phase_ms=3.0, prf_filter_count=2),
            MetricRow("workload", "q2.sql", "prf with fallback", 20, 5, 5),
        ]
        plan_rows = [
            PlanMetricRow("workload", query, "prf with fallback", 5, 5, 1)
            for query in ("q1.sql", "q2.sql")
        ]
        summary = summarize_benchmark_metrics(
            rows, plan_rows, (("workload", "Workload", ("q1.sql", "q2.sql")),),
        )
        variant = summary["benchmarks"][0]["variants"]["prf with fallback"]
        self.assertEqual(variant["prf_build_phase_ms"], 7.0)
        self.assertEqual(variant["prf_analysis_phase_ms"], 3.0)
        self.assertEqual(variant["prf_build_profile_count"], 1)
        self.assertEqual(variant["prf_analysis_profile_count"], 1)

    def test_rpt_use_bf_metadata_is_attached_to_scan(self) -> None:
        profile = {
            "operator_type": "HASH_JOIN",
            "operator_cardinality": 10,
            "children": [
                {
                    "operator_type": "USE_BF",
                    "operator_cardinality": 12,
                    "extra_info": {
                        "Filter": "Prefix Range",
                        "Tuple Placement": "scan",
                    },
                    "children": [
                        {
                            "operator_type": "PROJECTION",
                            "operator_cardinality": 12,
                            "extra_info": {},
                            "children": [
                                {
                                    "operator_type": "TABLE_SCAN",
                                    "operator_rows_scanned": 100,
                                    "operator_cardinality": 80,
                                    "extra_info": {"Table": "fact"},
                                    "children": [],
                                }
                            ],
                        }
                    ],
                },
                {"operator_type": "TABLE_SCAN", "children": []},
            ],
        }

        rows = _extract_probe_scan_rows(
            profile,
            workload_name="workload",
            query_name="query.sql",
            variant_label="rpt prf",
            probe_tables=set(),
            table_file_counts={},
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("RPT Prefix Range (scan)", rows[0].dynamic_filters)
        self.assertEqual(rows[0].operator_cardinality, 80)
        self.assertEqual(rows[0].join_probe_cardinality, 12)

    def test_relative_cardinality_matches_qualified_logical_tables(self) -> None:
        variants = (
            "sip",
            "sip + min/max",
            "prf with fallback",
            "rpt cascade",
            "rpt prf",
        )
        cardinalities = (20, 15, 10, 40, 12)
        rows = []
        for index, (variant, cardinality) in enumerate(
            zip(variants, cardinalities)
        ):
            table_name = "benchmark.main.fact" if index < 3 else "fact"
            rows.append(
                ProbeScanRow(
                    "workload",
                    "query.sql",
                    variant,
                    f"{table_name}::{index}",
                    table_name,
                    100,
                    cardinality,
                    "runtime filter",
                )
            )

        values = build_relative_probe_cardinality_values(rows)

        self.assertEqual(values["prf with fallback"], [1.0])
        self.assertEqual(values["rpt prf"], [1.2])
        self.assertEqual(values["rpt cascade"], [4.0])

    def test_empty_relative_cardinality_plot_is_rendered(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "empty.pdf"

            verification = _draw_relative_probe_cardinality_boxplot(
                output_path,
                relative_values={},
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(
                verification,
                {"tick_label_overlap": False, "text_outside_figure": False},
            )

    def test_ducklake_http_paper_config_includes_motivation_workloads(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            repository_root / "paper/configs/local_paper_ducklake_http_10ms.json"
        )
        workloads = {workload.name: workload for workload in config.workloads}

        self.assertIn("tpch-skew", workloads)
        self.assertIn("ssb-ordered", workloads)
        self.assertEqual(workloads["tpch-skew-intro"].query_names, ("q02.sql",))
        self.assertEqual(workloads["ssb-ordered-intro"].query_names, ("q01.sql",))
        self.assertEqual(
            workloads["tpch-skew-intro"].setup_sql,
            ("SET enable_join_bloom_filter_row_group_pruning=false;",),
        )
        self.assertEqual(
            workloads["ssb-ordered-intro"].setup_sql,
            ("SET enable_join_bloom_filter_row_group_pruning=false;",),
        )

    def test_extract_join_probe_metrics_counts_distinct_left_subtree_scans(self) -> None:
        profile_data = {
            "operator_type": "HASH_JOIN",
            "children": [
                {
                    "operator_type": "HASH_JOIN",
                    "operator_cardinality": 30,
                    "children": [
                        {
                            "operator_type": "TABLE_SCAN",
                            "operator_cardinality": 10,
                            "children": [],
                        },
                        {
                            "operator_type": "TABLE_SCAN",
                            "operator_cardinality": 20,
                            "children": [],
                        },
                    ],
                },
                {
                    "operator_type": "HASH_JOIN",
                    "operator_cardinality": 50,
                    "children": [
                        {
                            "operator_type": "TABLE_SCAN",
                            "operator_cardinality": 30,
                            "children": [],
                        },
                        {
                            "operator_type": "TABLE_SCAN",
                            "operator_cardinality": 40,
                            "children": [],
                        },
                    ],
                },
            ],
        }

        join_probe_tuples, join_operator_count = _extract_join_probe_metrics(profile_data)

        self.assertEqual(join_probe_tuples, 70)
        self.assertEqual(join_operator_count, 3)

    def test_build_runtime_probe_table_selects_lineorder(self) -> None:
        metric_rows = [
            MetricRow("ssb-ordered", "q01.sql", label, 10.0, None, None)
            for label in VARIANT_ORDER
        ]
        scan_rows = [
            ProbeScanRow(
                "ssb-ordered",
                "q01.sql",
                label,
                f"main.lineorder::{index}",
                "main.lineorder",
                1000,
                250,
                "",
            )
            for index, label in enumerate(VARIANT_ORDER)
        ]
        scan_rows.append(
            ProbeScanRow(
                "ssb-ordered",
                "q01.sql",
                "sip",
                "main.date::0",
                "main.date",
                100,
                10,
                "",
            )
        )

        entries = build_runtime_probe_table(
            metric_rows,
            scan_rows,
            "ssb-ordered",
            ("q01.sql",),
            "lineorder",
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(
            [item["variant_label"] for item in entries[0]["series"]],
            list(VARIANT_ORDER),
        )
        self.assertEqual(entries[0]["series"][0]["probe_output_cardinality"], 250)
        self.assertEqual(entries[0]["series"][0]["probe_rows_filtered_out"], 750)
        self.assertEqual(entries[0]["series"][0]["join_probe_rows"], 250)
        self.assertEqual(entries[0]["series"][0]["probe_rows_skipped_at_scan"], 0)

    def test_runtime_probe_table_splits_scan_and_join_pruning(self) -> None:
        metrics = [MetricRow("workload", "query.sql", "rpt cascade", 1.0, None, None)]
        scans = [
            ProbeScanRow(
                "workload",
                "query.sql",
                "rpt cascade",
                "fact::0",
                "fact",
                20,
                18,
                "RPT Cascade (operator)",
                join_probe_cardinality=12,
            )
        ]

        entries = build_runtime_probe_table(
            metrics,
            scans,
            "workload",
            ("query.sql",),
            "fact",
            logical_probe_rows=100,
        )
        values = entries[0]["series"][0]

        self.assertEqual(values["join_probe_rows"], 12)
        self.assertEqual(values["probe_rows_filtered_after_scan"], 8)
        self.assertEqual(values["probe_rows_skipped_at_scan"], 80)

    def test_runtime_probe_table_rejects_impossible_scan_count(self) -> None:
        metrics = [MetricRow("workload", "query.sql", "rpt prf", 1.0, None, None)]
        scans = [
            ProbeScanRow(
                "workload",
                "query.sql",
                "rpt prf",
                "fact::0",
                "fact",
                800,
                10,
                "RPT Prefix Range (scan)",
                join_probe_cardinality=10,
            )
        ]

        with self.assertRaisesRegex(ValueError, "physical scan=800"):
            build_runtime_probe_table(
                metrics,
                scans,
                "workload",
                ("query.sql",),
                "fact",
                logical_probe_rows=100,
            )

    def test_single_query_plot_without_x_tick(self) -> None:
        entries = [
            {
                "workload_name": "ssb-ordered",
                "workload_label": "SSB Ordered",
                "query_name": "q01.sql",
                "query_label": "q01",
                "series": [
                    {
                        "variant_label": label,
                        "median_ms": 10.0 + index,
                        "join_probe_rows": 250,
                        "probe_rows_filtered_after_scan": 750,
                        "probe_rows_skipped_at_scan": 0,
                    }
                    for index, label in enumerate(VARIANT_ORDER)
                ],
            }
        ]
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "ssb.pdf"
            verification = _draw_runtime_probe_bars(
                output_path,
                entries=entries,
                width_key="one-column",
                placement_width_key="one-column",
                show_query_labels=False,
                annotate_runtime_values=False,
                runtime_log_axis=True,
                rows_log_axis=True,
            )
            self.assertTrue(output_path.exists())
            self.assertFalse(verification["tick_label_overlap"])
            self.assertFalse(verification["text_outside_figure"])


if __name__ == "__main__":
    unittest.main()
