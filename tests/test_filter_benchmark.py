from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from paper.figures.filter_benchmark import (
    BenchmarkGenerationConfig,
    build_filter_benchmark_artifacts,
    build_build_time_series,
    build_false_positive_rate_series,
    build_throughput_series,
    load_filter_benchmark_rows,
    summarize_filter_benchmark,
)


class FilterBenchmarkTest(unittest.TestCase):
    def test_missing_measured_fpr_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.csv"
            path.write_text("filter,cluster_count,build_time_ns,post_processing_time_ns,summary_bytes,probe_throughput_per_sec,false_positives\nbloom,2,100,0,16,100,1\n")
            with self.assertRaisesRegex(ValueError, "actual_fpr"):
                load_filter_benchmark_rows(path)

    def test_throughput_aggregation_and_summary(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "filter_benchmark_result.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filter",
                        "cluster_count",
                        "build_time_ns",
                        "post_processing_time_ns",
                        "summary_bytes",
                        "probe_throughput_per_sec",
                        "false_positives",
                        "actual_fpr",
                        "actual_negatives",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "filter": "bloom",
                            "cluster_count": 2,
                            "build_time_ns": 4_000_000,
                            "post_processing_time_ns": 1_000_000,
                            "summary_bytes": 16,
                            "probe_throughput_per_sec": 100.0,
                            "false_positives": 10,
                            "actual_fpr": 10 / 900,
                            "actual_negatives": 900,
                        },
                        {
                            "filter": "bloom",
                            "cluster_count": 2,
                            "build_time_ns": 6_000_000,
                            "post_processing_time_ns": 1_000_000,
                            "summary_bytes": 16,
                            "probe_throughput_per_sec": 140.0,
                            "false_positives": 30,
                            "actual_fpr": 30 / 900,
                            "actual_negatives": 900,
                        },
                        {
                            "filter": "prf",
                            "cluster_count": 2,
                            "build_time_ns": 5_000_000,
                            "post_processing_time_ns": 3_000_000,
                            "summary_bytes": 32,
                            "probe_throughput_per_sec": 200.0,
                            "false_positives": 0,
                            "actual_fpr": 0,
                            "actual_negatives": 900,
                        },
                        {
                            "filter": "prf",
                            "cluster_count": 102,
                            "build_time_ns": 7_000_000,
                            "post_processing_time_ns": 5_000_000,
                            "summary_bytes": 48,
                            "probe_throughput_per_sec": 220.0,
                            "false_positives": 5,
                            "actual_fpr": 5 / 880,
                            "actual_negatives": 880,
                        },
                    ]
                )

            rows = load_filter_benchmark_rows(csv_path)
            series = build_throughput_series(rows, aggregate="median")
            build_time_series = build_build_time_series(rows, aggregate="median")
            false_positive_rate_series = build_false_positive_rate_series(
                rows,
                aggregate="median",
            )
            summary = summarize_filter_benchmark(
                rows,
            )

            self.assertEqual(series["bloom"], [(2, 120.0)])
            self.assertEqual(series["prf"], [(2, 200.0), (102, 220.0)])
            self.assertEqual(build_time_series["bloom"], [(2, 6.0)])
            self.assertEqual(build_time_series["prf"], [(2, 8.0), (102, 12.0)])
            self.assertAlmostEqual(false_positive_rate_series["bloom"][0][1], 20 / 900)
            self.assertEqual(false_positive_rate_series["prf"], [(2, 0.0), (102, 5 / 880)])
            bloom_summary = next(system for system in summary["systems"] if system["filter"] == "bloom")
            self.assertAlmostEqual(bloom_summary["build_time_ms"]["min"], 5.0)
            self.assertAlmostEqual(bloom_summary["build_time_ms"]["max"], 7.0)
            self.assertAlmostEqual(bloom_summary["build_time_ms"]["avg"], 6.0)
            self.assertAlmostEqual(bloom_summary["false_positive_rate"]["avg"], 20 / 900)
            self.assertAlmostEqual(bloom_summary["true_negatives"]["avg"], 880.0)

    def test_build_artifacts_writes_pdf_markdown_and_json(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "filter_benchmark_result.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filter",
                        "cluster_count",
                        "build_time_ns",
                        "post_processing_time_ns",
                        "summary_bytes",
                        "probe_throughput_per_sec",
                        "false_positives",
                        "actual_fpr",
                        "actual_negatives",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "filter": "bloom",
                            "cluster_count": 2,
                            "build_time_ns": 4_000_000,
                            "post_processing_time_ns": 1_000_000,
                            "summary_bytes": 16,
                            "probe_throughput_per_sec": 100.0,
                            "false_positives": 10,
                            "actual_fpr": 10 / 900,
                            "actual_negatives": 900,
                        },
                        {
                            "filter": "prf",
                            "cluster_count": 2,
                            "build_time_ns": 5_000_000,
                            "post_processing_time_ns": 3_000_000,
                            "summary_bytes": 32,
                            "probe_throughput_per_sec": 200.0,
                            "false_positives": 0,
                            "actual_fpr": 0,
                            "actual_negatives": 900,
                        },
                    ]
                )
            output_dir = Path(temporary_dir) / "out"
            payload = build_filter_benchmark_artifacts(
                csv_path=csv_path,
                output_dir=output_dir,
                generation_config=BenchmarkGenerationConfig(
                    domain_size=100,
                    total_keys=10,
                    probe_count=20,
                    seed=42,
                ),
                aggregate="median",
            )

            figure_path = Path(payload["figure_path"])
            markdown_path = Path(payload["summary_markdown_path"])
            json_path = output_dir / "filter_benchmark_summary.json"

            self.assertTrue(figure_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(json_path.exists())
            loaded_payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_payload["aggregate"], "median")
            self.assertEqual(loaded_payload["generation_config"]["probe_count"], 20)
            self.assertEqual(loaded_payload["verification"]["build_time_aggregate"], "median")
            self.assertEqual(loaded_payload["verification"]["false_positive_rate_aggregate"], "median")


if __name__ == "__main__":
    unittest.main()
