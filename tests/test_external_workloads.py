from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarking.experiment_config import ExternalWorkloadConfig
from benchmarking.workloads.external import discover_external_layout, prepare_external_workload


class ExternalWorkloadLayoutTest(unittest.TestCase):
    def test_prefers_checked_in_benchmark_db_over_init_sql(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            external_dir = Path(temporary_dir)
            (external_dir / "queries").mkdir()
            (external_dir / "schema.sql").write_text("CREATE TABLE t(i INTEGER);\n", encoding="utf-8")
            (external_dir / "load.sql").write_text("INSERT INTO t VALUES (1);\n", encoding="utf-8")
            (external_dir / "benchmark.duckdb").write_bytes(b"placeholder")
            (external_dir / "source.json").write_text(
                json.dumps(
                    {
                        "benchmark_name": "job",
                        "queries_dir": "queries",
                        "init_sql_paths": ["schema.sql", "load.sql"],
                    }
                ),
                encoding="utf-8",
            )

            layout = discover_external_layout(external_dir)

            self.assertEqual(layout.benchmark_name, "job")
            self.assertEqual(layout.source_db_path, (external_dir / "benchmark.duckdb").resolve())
            self.assertEqual(layout.db_handling, "reuse")
            self.assertEqual(layout.init_sql_paths, ())

if __name__ == "__main__":
    unittest.main()
