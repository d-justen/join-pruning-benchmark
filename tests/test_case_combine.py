from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb

from benchmarking.suite import execute
from benchmarking.suite_jobs import combine_dimension_cases
from benchmarking.suite_io import fingerprint, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class CaseCombineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = ROOT / "duckdb/build/paper-native-release/duckdb"
        if not binary.is_file():
            raise unittest.SkipTest("requires native PRF binary")
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        raw = json.loads((ROOT / "configs/smoke.json").read_text())
        raw.pop("microbenchmarks", None)
        raw.pop("jpb_microbenchmarks", None)
        raw["workloads"] = raw["workloads"][:1]
        workload = raw["workloads"][0]
        workload["params"]["fixed"].update(fact_rows=1000, dimension_cardinality=100,
                                            probe_layout="ordered", selectivity=.1)
        workload["params"]["sweep"]["values"] = [100, 200]
        workload["params"]["query_types"] = [{"kind": "range", "probe_layouts": ["ordered"]}]
        workload["estimated_peak_gib"] = .001
        raw["systems"] = raw["systems"][:1]
        raw["systems"][0]["binary_path"] = str(binary)
        raw["systems"][0]["variants"] = raw["systems"][0]["variants"][:1]
        raw["suite"].update(checkpoint_queries=True, paper_figures=False, required_figures=[],
                             required_workloads=[workload["name"]], disk_reserve_gib=.001)
        write_json(cls.root / "combined-config.json", raw)
        for name, value in (("first", 100), ("second", 200)):
            part = copy.deepcopy(raw)
            part["workloads"][0]["params"]["sweep"]["values"] = [value]
            path = cls.root / (name + ".json")
            write_json(path, part)
            run = cls.root / name
            if execute(path, "all", run, cls.root / (name + "-cache"), False):
                raise RuntimeError("fixture measurement failed")
            write_json(run / "cloud/attempt/launch.json", {"instance_type": "fixture", "ami": "fixture-ami"})

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.config = self.work / "config.json"
        shutil.copy2(self.root / "combined-config.json", self.config)
        self.sources = []
        for name in ("first", "second"):
            path = self.work / name
            shutil.copytree(self.root / name, path)
            self.sources.append(path)

    def test_complete_workloads_from_running_runs_combine_without_mutating_sources(self):
        first = self.sources[0]
        state = json.loads((first / "run.json").read_text())
        write_json(first / "run.json", {**state, "status": "running"})
        before = {p: sha256(p) for root in self.sources for p in root.rglob("*") if p.is_file()}
        output = (self.work / "combined").resolve()
        combine_dimension_cases(self.config, self.sources, output)
        self.assertEqual(json.loads((output / "coverage.json").read_text())["status"], "complete")
        catalog = json.loads((output / "workloads/join-pruning-workload/queries/query_catalog.json").read_text())
        self.assertEqual(len(catalog["queries"]), 2)
        with duckdb.connect(str(output / "results.duckdb"), read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT DISTINCT experiment_name FROM timing_runs").fetchall(), [("combined",)])
            for path, in con.execute("SELECT profile_path FROM profile_runs").fetchall():
                self.assertTrue(Path(path).is_relative_to(output))
                self.assertTrue(Path(path).is_file())
        self.assertEqual(before, {p: sha256(p) for p in before})

    def test_missing_and_overlapping_partitions_rejected(self):
        for sources, message in ((self.sources[:1], "missing dimension"),
                                 (self.sources + self.sources[:1], "overlapping")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                combine_dimension_cases(self.config, sources, self.work / "rejected")

    def test_changed_software_or_hardware_rejected(self):
        source = self.sources[1]
        for path, change in ((source / "run.json", lambda v: v.update(source="different-runner")),
                             (source / "cloud/attempt/launch.json", lambda v: v.update(ami="different-ami"))):
            original = path.read_text()
            data = json.loads(original)
            change(data)
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "runner, binaries, runtime or EC2 hardware"):
                combine_dimension_cases(self.config, self.sources, self.work / "rejected")
            path.write_text(original)

    def test_changed_benchmark_settings_rejected(self):
        source = self.sources[1]
        request = json.loads((source / "request.json").read_text())
        request["benchmark"]["threads"] += 1
        write_json(source / "request.json", request)
        with self.assertRaisesRegex(ValueError, "benchmark settings"):
            combine_dimension_cases(self.config, self.sources, self.work / "rejected")

    def test_corrupt_measurement_rejected(self):
        database = next((self.sources[1] / "units").glob("*/results.duckdb"))
        database.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "incomplete or corrupt"):
            combine_dimension_cases(self.config, self.sources, self.work / "rejected")

    def test_missing_rows_rejected_even_with_valid_file_checksums(self):
        database = next((self.sources[1] / "units").glob("*/results.duckdb"))
        with duckdb.connect(str(database)) as con:
            con.execute("DELETE FROM timing_runs")
        status = database.with_name("status.json")
        state = json.loads(status.read_text())
        state["sha256"] = sha256(database)
        write_json(status, state)
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            combine_dimension_cases(self.config, self.sources, self.work / "rejected")


if __name__ == "__main__":
    unittest.main()
