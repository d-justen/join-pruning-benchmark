from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paper.figure_specs import VARIANT_ORDER
from paper.figures.build_all import _draw_runtime_probe_bars
from paper.figures.data import (
    MetricRow,
    ProbeScanRow,
    _extract_join_probe_metrics,
    build_runtime_probe_table,
)


class PaperSsbTest(unittest.TestCase):
    def test_extract_join_probe_metrics_counts_distinct_left_subtree_scans(self) -> None:
        profile_data = {
            "operator_type": "HASH_JOIN",
            "children": [
                {
                    "operator_type": "HASH_JOIN",
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

        self.assertEqual(join_probe_tuples, 60)
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
                        "probe_output_cardinality": 250,
                        "probe_rows_filtered_out": 750,
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
