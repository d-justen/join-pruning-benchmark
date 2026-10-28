from pathlib import Path
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

import duckdb


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "external_benchmarks" / "tpch-skew"


class TpchSkewGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("cmake") is None or shutil.which("cc") is None:
            raise unittest.SkipTest("CMake and a C compiler are required")
        cls.temporary = TemporaryDirectory(prefix="tpcdskew-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.build = cls.root / "build"
        for command in (
            ["cmake", "-S", str(ROOT / "TPCDSkew"), "-B", str(cls.build),
             "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(cls.build), "--parallel", "2"],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)

    def generate(self, destination, *arguments):
        destination.mkdir(exist_ok=True)
        return subprocess.run(
            [str(self.build / "dbgen"), *arguments],
            cwd=destination,
            env={**os.environ, "DSS_CONFIG": str(self.build), "DSS_PATH": str(destination)},
            input="",
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_fresh_uniform_and_skewed_data_load_and_run_queries(self):
        top_customer_counts = []
        for skew, lineitems in ((0, 60005), (2, 60004)):
            with self.subTest(skew=skew):
                destination = self.root / f"z{skew}"
                self.generate(destination, "-s", "0.01", "-z", str(skew))
                self.assertEqual(len(list(destination.glob("*.tbl"))), 8)
                con = duckdb.connect(":memory:")
                try:
                    con.execute((WORKLOAD / "schema.sql").read_text())
                    # Keep the actual loader's table/filename mapping without
                    # changing the test process's working directory.
                    load_sql = (WORKLOAD / "load.sql").read_text()
                    for table_file in destination.glob("*.tbl"):
                        escaped_path = str(table_file).replace("'", "''")
                        load_sql = load_sql.replace(
                            f"'{table_file.name}'", f"'{escaped_path}'"
                        )
                    con.execute(load_sql)
                    expected = {
                        "region": 5, "nation": 25, "supplier": 100, "customer": 1500,
                        "part": 2000, "partsupp": 8000, "orders": 15000,
                        "lineitem": lineitems,
                    }
                    actual = {
                        table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                        for table in expected
                    }
                    self.assertEqual(actual, expected)
                    for sql in (
                        "SELECT count(*) FROM orders o ANTI JOIN customer c "
                        "ON o.o_custkey = c.c_custkey",
                        "SELECT count(*) FROM lineitem l ANTI JOIN orders o "
                        "ON l.l_orderkey = o.o_orderkey",
                        "SELECT count(*) FROM lineitem l ANTI JOIN partsupp ps "
                        "ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey",
                        "SELECT count(*) - count(DISTINCT (l_orderkey, l_linenumber)) "
                        "FROM lineitem",
                    ):
                        self.assertEqual(con.execute(sql).fetchone()[0], 0)
                    queries = sorted((WORKLOAD / "queries").glob("*.sql"))
                    self.assertEqual(len(queries), 22)
                    for query in queries:
                        con.execute(query.read_text()).fetchall()
                    top_customer_counts.append(con.execute(
                        "SELECT max(n) FROM "
                        "(SELECT count(*) AS n FROM orders GROUP BY o_custkey)"
                    ).fetchone()[0])
                finally:
                    con.close()
        self.assertGreater(top_customer_counts[1], top_customer_counts[0])

    def test_existing_output_is_preserved_without_confirmation(self):
        destination = self.root / "existing"
        self.generate(destination, "-T", "r", "-s", "0.01", "-z", "2")
        original = (destination / "region.tbl").read_bytes()
        # Closed stdin must decline an overwrite without hanging.
        self.generate(destination, "-T", "r", "-s", "0.01", "-z", "0")
        self.assertEqual((destination / "region.tbl").read_bytes(), original)
        self.generate(destination, "-f", "-T", "r", "-s", "0.01", "-z", "0")
        self.assertNotEqual((destination / "region.tbl").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
