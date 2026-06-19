from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import duckdb

from benchmarking.experiment_config import (
    StringJoinsWorkloadConfig,
    load_experiment_config,
)
from benchmarking.workloads.string_joins import (
    UUID_MASK_COLUMN,
    prepare_string_joins_workload,
)


class StringJoinsWorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.airports_path = self.root / "airports.csv"
        self.airports_path.write_text(
            "ident,continent\nAAA,AN\nBBB,EU\nCCC,NA\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def spec(self) -> StringJoinsWorkloadConfig:
        return StringJoinsWorkloadConfig(
            kind="string_joins",
            name="string-joins",
            airport_csv_path=self.airports_path,
            uuid_dim_size=1_000,
            uuid_timestamp_start_ms=1_704_067_200_000,
            uuid_timestamp_step_ms=100_000_000,
            uuid_selected_ranges=((101, 125), (451, 475), (851, 875)),
            fact_rows=10_000,
            dummy_string_cols=1,
            dummy_string_len=8,
            seed=7,
            remote_http_parquet_latency_ms=None,
        )

    def test_config_parser_reads_uuid_ranges(self) -> None:
        config_path = self.root / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "experiment_name": "test",
                    "output_root": "out",
                    "overwrite_output": True,
                    "workloads": [
                        {
                            "name": "string-joins",
                            "kind": "string_joins",
                            "params": {
                                "airport_csv_path": "airports.csv",
                                "uuid_dim_size": 1000,
                                "uuid_timestamp_start_ms": 1704067200000,
                                "uuid_timestamp_step_ms": 100000000,
                                "uuid_selected_ranges": [
                                    [101, 125],
                                    [451, 475],
                                    [851, 875],
                                ],
                                "fact_rows": 10000,
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

        self.assertIsInstance(workload, StringJoinsWorkloadConfig)
        self.assertEqual(
            workload.uuid_selected_ranges,
            ((101, 125), (451, 475), (851, 875)),
        )

    def test_prepared_workload_has_ordered_uuidv7_keys_and_gapped_mask(self) -> None:
        artifact_dir = self.root / "workload"
        manifest = prepare_string_joins_workload(self.spec(), artifact_dir)
        metadata = json.loads(
            (artifact_dir / "queries" / "metadata.json").read_text(encoding="utf-8")
        )

        con = duckdb.connect(str(manifest.db_path), read_only=True)
        try:
            keys = con.execute(
                """
                SELECT d_key, uuid_key, mask_gapped
                FROM uuid_dim
                WHERE d_key IN (1, 101, 126, 451, 476, 851, 876, 1000)
                ORDER BY d_key
                """
            ).fetchall()
            selected_rows = con.execute(
                f"SELECT COUNT(*) FROM uuid_dim WHERE {UUID_MASK_COLUMN}"
            ).fetchone()[0]
            out_of_order = con.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT uuid_key, lag(uuid_key) OVER (ORDER BY d_key) AS previous_key
                    FROM uuid_dim
                )
                WHERE uuid_key <= previous_key
                """
            ).fetchone()[0]
        finally:
            con.close()

        self.assertEqual(selected_rows, 75)
        self.assertEqual(out_of_order, 0)
        self.assertEqual([row[2] for row in keys], [False, True, False, True, False, True, False, False])
        self.assertTrue(all(key[1][14] == "7" for key in keys))
        self.assertTrue(all(key[1][19] == "8" for key in keys))
        self.assertLess(
            metadata["uuid_prefix_coverage"],
            metadata["uuid_min_max_coverage"],
        )
        self.assertIn(
            "WHERE d.mask_gapped",
            (artifact_dir / "queries" / "uuid_v7_like_ordered.sql").read_text(
                encoding="utf-8"
            ),
        )

    def test_validation_rejects_small_or_overlapping_build_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than 50 rows"):
            prepare_string_joins_workload(
                replace(self.spec(), uuid_selected_ranges=((1, 50),)),
                self.root / "small",
            )

        with self.assertRaisesRegex(ValueError, "sorted and non-overlapping"):
            prepare_string_joins_workload(
                replace(
                    self.spec(),
                    uuid_selected_ranges=((101, 160), (150, 220)),
                ),
                self.root / "overlap",
            )

    def test_validation_rejects_prefix_invisible_gaps(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefix-visible gaps"):
            prepare_string_joins_workload(
                replace(
                    self.spec(),
                    uuid_timestamp_step_ms=1,
                    uuid_selected_ranges=((101, 125), (451, 475), (851, 875)),
                ),
                self.root / "collapsed-prefix",
            )


if __name__ == "__main__":
    unittest.main()
