import unittest
from pathlib import Path

from benchmarking import load_experiment_config
from paper.figure_specs import VARIANT_ORDER
from paper.figures.build_all import (
    COLORS,
    PROBE_CARDINALITY_VARIANTS,
    SWEEP_MARKERS,
    VARIANT_DISPLAY_LABELS,
    VARIANT_PANEL_LABELS,
)
from paper.report_tpch_ducklake_http import VARIANTS as TPCH_DUCKLAKE_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIGS = (
    "local_paper_experiments.json",
    "local_paper_sweeps.json",
    "local_paper_ducklake.json",
    "local_paper_ducklake_http_10ms.json",
    "tpch_skew_ducklake_http_10ms.json",
)


class PaperPrfVariantsTest(unittest.TestCase):
    def test_paper_join_pruning_workloads_use_500m_fact_rows(self) -> None:
        for config_name in PAPER_CONFIGS:
            with self.subTest(config=config_name):
                config = load_experiment_config(ROOT / "paper/configs" / config_name)
                for workload in config.workloads:
                    if workload.kind != "cross_product":
                        continue
                    with self.subTest(workload=workload.name):
                        self.assertEqual(
                            workload.experiment_definition.fixed.fact_rows,
                            500_000_000,
                        )

    def test_all_paper_configs_compare_compressed_and_uncompressed_prf(self) -> None:
        compression_statements = {
            "SET enable_prefix_range_filter_compression=false;",
            "SET enable_prefix_range_filter_compression=true;",
        }
        for config_name in PAPER_CONFIGS:
            with self.subTest(config=config_name):
                config = load_experiment_config(ROOT / "paper/configs" / config_name)
                duckdb_system = config.systems[0]
                variants = {variant.label: variant for variant in duckdb_system.variants}
                uncompressed = variants["prf uncompressed"].setup_sql
                compressed = variants["prf with fallback"].setup_sql

                self.assertIn("SET enable_prefix_range_filter=true;", uncompressed)
                self.assertIn(
                    "SET enable_prefix_range_filter_compression=false;",
                    uncompressed,
                )
                self.assertIn(
                    "SET enable_prefix_range_filter_compression=true;",
                    compressed,
                )
                self.assertEqual(
                    tuple(item for item in uncompressed if item not in compression_statements),
                    tuple(item for item in compressed if item not in compression_statements),
                )

    def test_uncompressed_prf_is_available_to_all_paper_outputs(self) -> None:
        self.assertLess(
            VARIANT_ORDER.index("prf uncompressed"),
            VARIANT_ORDER.index("prf with fallback"),
        )
        self.assertIn("prf uncompressed", COLORS)
        self.assertIn("prf uncompressed", SWEEP_MARKERS)
        self.assertIn("prf uncompressed", VARIANT_DISPLAY_LABELS)
        self.assertIn("prf uncompressed", VARIANT_PANEL_LABELS)
        self.assertIn("prf uncompressed", PROBE_CARDINALITY_VARIANTS)
        self.assertIn(
            ("prf uncompressed", "prf_uncompressed"),
            TPCH_DUCKLAKE_VARIANTS,
        )


if __name__ == "__main__":
    unittest.main()
