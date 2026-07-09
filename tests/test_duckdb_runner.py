from __future__ import annotations

import unittest

from benchmarking.duckdb_runner import _extract_profile_metrics


class DuckDBRunnerProfileMetricsTest(unittest.TestCase):
    def test_extracts_byte_and_file_metrics_from_profile(self) -> None:
        metrics = _extract_profile_metrics(
            {
                "cumulative_rows_scanned": 1000,
                "total_bytes_read": 4096,
                "total_bytes_written": 128,
                "children": [
                    {
                        "operator_type": "TABLE_SCAN",
                        "operator_rows_scanned": 700,
                        "extra_info": {"Table": "main.fact", "Total Files Read": "2"},
                        "children": [],
                    },
                    {
                        "operator_type": "TABLE_SCAN",
                        "operator_rows_scanned": 300,
                        "extra_info": {"Table": "main.dim", "Total Files Read": 1},
                        "children": [],
                    },
                ],
            }
        )

        self.assertEqual(metrics.total_rows_scanned, 1000)
        self.assertEqual(metrics.probe_rows_scanned, 700)
        self.assertEqual(metrics.build_rows_scanned, 300)
        self.assertEqual(metrics.total_bytes_read, 4096)
        self.assertEqual(metrics.total_bytes_written, 128)
        self.assertEqual(metrics.table_scan_file_count, 3)

    def test_leaves_optional_metrics_null_when_missing(self) -> None:
        metrics = _extract_profile_metrics({"children": []})

        self.assertIsNone(metrics.total_rows_scanned)
        self.assertIsNone(metrics.probe_rows_scanned)
        self.assertIsNone(metrics.build_rows_scanned)
        self.assertIsNone(metrics.total_bytes_read)
        self.assertIsNone(metrics.total_bytes_written)
        self.assertIsNone(metrics.table_scan_file_count)


if __name__ == "__main__":
    unittest.main()
