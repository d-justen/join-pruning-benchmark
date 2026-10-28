from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from benchmarking.experiment_config import (
    CrossProductWorkloadConfig,
    load_experiment_config,
)
from benchmarking.manifests import load_manifest
from benchmarking.manifests import DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER
from benchmarking.workloads.cross_product import prepare_cross_product_workload
from paper.figures.data import _extract_probe_scan_rows


class CrossProductDuckLakeTest(unittest.TestCase):
    def test_config_parser_reads_ducklake_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "test",
                        "output_root": "out",
                        "overwrite_output": True,
                        "workloads": [
                            {
                                "name": "cross",
                                "kind": "cross_product",
                                "params": {
                                    "build_sizes": [100],
                                    "selectivities": [0.1],
                                    "predicate_types": ["range"],
                                    "probe_clusterings": ["ordered"],
                                    "fact_rows": 1000,
                                    "storage": {
                                        "kind": "ducklake",
                                        "parquet_file_size_mb": 32,
                                        "http_latency_ms": 10,
                                    },
                                },
                            }
                        ],
                        "systems": [
                            {
                                "name": "test",
                                "kind": "binary",
                                "binary_path": "/bin/true",
                                "variants": [{"label": "default", "setup_sql": []}],
                            }
                        ],
                        "benchmark": {},
                    }
                ),
                encoding="utf-8",
            )

            workload = load_experiment_config(config_path).workloads[0]

            self.assertIsInstance(workload, CrossProductWorkloadConfig)
            self.assertEqual(workload.storage_kind, "ducklake")
            self.assertEqual(workload.parquet_file_size_mb, 32)
            self.assertEqual(workload.ducklake_http_latency_ms, 10)

    def test_preparation_exports_parquet_and_ducklake_sql(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact_dir = Path(temporary_dir) / "workload"
            spec = CrossProductWorkloadConfig(
                kind="cross_product",
                name="cross-ducklake",
                build_sizes=(100, 1000),
                selectivities=(0.1,),
                predicate_types=("range", "shoulder"),
                probe_clusterings=("ordered", "uniform"),
                fact_rows=10_000,
                dummy_string_cols=1,
                dummy_string_len=8,
                seed=7,
                remote_http_parquet_latency_ms=None,
                storage_kind="ducklake",
                parquet_file_size_mb=1,
                ducklake_http_latency_ms=1,
            )

            manifest = prepare_cross_product_workload(spec, artifact_dir)
            loaded = load_manifest(artifact_dir / "manifest.json")

            self.assertEqual(manifest.storage_mode, "ducklake")
            self.assertEqual(loaded.storage_mode, "ducklake")
            self.assertEqual(loaded.ducklake_http_latency_ms, 1)
            self.assertFalse((artifact_dir / "benchmark.duckdb").exists())
            self.assertIsNotNone(manifest.parquet_catalog_path)
            self.assertIsNotNone(manifest.ducklake_bootstrap_sql_path)
            self.assertEqual(
                manifest.setup_sql_paths,
                ((artifact_dir / "ducklake_setup.sql").resolve(),),
            )
            catalog = json.loads(
                manifest.parquet_catalog_path.read_text(encoding="utf-8")
            )
            self.assertEqual(set(catalog), {"dim_100", "dim_1k", "fact"})
            self.assertTrue(catalog["fact"])
            for file_paths in catalog.values():
                for relative_path in file_paths:
                    parquet_path = artifact_dir / "parquet" / relative_path
                    row_groups = duckdb.sql(
                        "SELECT COUNT(*) FROM parquet_metadata(?)",
                        params=[str(parquet_path)],
                    ).fetchone()[0]
                    self.assertGreater(row_groups, 0)
            bootstrap_sql = manifest.ducklake_bootstrap_sql_path.read_text(
                encoding="utf-8"
            )
            setup_sql = manifest.setup_sql_paths[0].read_text(encoding="utf-8")
            self.assertIn("LOAD ducklake", bootstrap_sql)
            self.assertIn("LOAD httpfs", bootstrap_sql)
            self.assertIn("SET enable_external_file_cache=false", bootstrap_sql)
            self.assertIn(DUCKLAKE_HTTP_BASE_URL_PLACEHOLDER, bootstrap_sql)
            self.assertIn("CALL ducklake_add_data_files", bootstrap_sql)
            self.assertIn('CREATE OR REPLACE TEMP VIEW "fact"', setup_sql)

    def test_profile_scan_extracts_ducklake_file_metrics(self) -> None:
        rows = _extract_probe_scan_rows(
            {
                "children": [
                    {
                        "operator_type": "TABLE_SCAN",
                        "operator_rows_scanned": 1000,
                        "operator_cardinality": 100,
                        "extra_info": {
                            "Table": "fact",
                            "Total Files Read": "2",
                            "Total Files Skipped": 3,
                            "Dynamic Filters": "optional: k IN PRF(k)",
                        },
                        "children": [],
                    }
                ]
            },
            workload_name="cross-ducklake",
            query_name="q.sql",
            variant_label="prf with fallback",
            probe_tables={"fact"},
            table_file_counts={"fact": 5},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].total_files_read, 2)
        self.assertEqual(rows[0].total_files_skipped, 3)
        self.assertEqual(rows[0].table_file_count, 5)


if __name__ == "__main__":
    unittest.main()
