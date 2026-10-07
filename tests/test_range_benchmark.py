from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from paper.figures.range_benchmark import (
    build_latency_series,
    build_range_benchmark_artifacts,
    load_range_benchmark_rows,
)


class RangeBenchmarkTest(unittest.TestCase):
    def test_latency_series_separates_prf_modes_and_aggregates_repetitions(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "range_results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["summary", "mode", "cluster_count", "queries_per_sec", "query_kind"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"query_kind": "empty", "summary": "prf", "mode": "compressed", "cluster_count": 2, "queries_per_sec": 1_000_000},
                        {"query_kind": "empty", "summary": "prf", "mode": "compressed", "cluster_count": 2, "queries_per_sec": 333_333.3333333333},
                        {"query_kind": "empty", "summary": "prf", "mode": "uncompressed", "cluster_count": 2, "queries_per_sec": 125_000},
                        {"query_kind": "empty", "summary": "grafite", "mode": "na", "cluster_count": 2, "queries_per_sec": 250_000},
                        {"query_kind": "empty", "summary": "diva", "mode": "na", "cluster_count": 2, "queries_per_sec": 166_666.6666666667},
                    ]
                )

            series = build_latency_series(load_range_benchmark_rows(csv_path), aggregate="median")
            self.assertEqual(series["prf"], [(2, 2.0)])
            self.assertEqual(series["prf_uncompressed"], [(2, 8.0)])
            self.assertEqual(series["grafite"], [(2, 4.0)])

    def test_cluster_count_is_used_as_the_series_x_value(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "range_results.csv"
            csv_path.write_text(
                "query_kind,summary,mode,cluster_count,queries_per_sec\n"
                "empty,prf,compressed,128,2000000\n",
                encoding="utf-8",
            )

            series = build_latency_series(load_range_benchmark_rows(csv_path), aggregate="median")
            self.assertEqual(series["prf"], [(128, 0.5)])

    def test_build_artifacts_writes_pdf_and_json(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "range_results.csv"
            csv_path.write_text(
                "query_kind,summary,mode,cluster_count,queries_per_sec\n"
                "empty,prf,compressed,1,1000000\n"
                "empty,diva,na,1,500000\n",
                encoding="utf-8",
            )
            output_dir = Path(temporary_dir) / "out"
            payload = build_range_benchmark_artifacts(csv_path=csv_path, output_dir=output_dir)

            self.assertTrue(Path(payload["figure_path"]).exists())
            summary = json.loads((output_dir / "range_query_latency.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["verification"]["x_scale"], "log2")
            self.assertEqual(summary["verification"]["plotted_series"], ["diva", "prf"])


if __name__ == "__main__":
    unittest.main()
