from __future__ import annotations

import copy
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarking.cloud import validate_resources
from benchmarking.suite import analyze, execute
from benchmarking.suite_io import sha256, write_json
from benchmarking.suite_plan import load_plan, uses_s3, validate_plan
import test_suite

ROOT = Path(__file__).resolve().parents[1]


class PaperPlanTests(unittest.TestCase):
    def test_paper_embeds_both_groups_and_retains_distinct_binaries_and_experiments(self):
        raw = load_plan(ROOT / "configs/paper.json")
        native, lake = [g["config"] for g in raw["groups"]]
        self.assertTrue(uses_s3(raw))
        self.assertIn("microbenchmarks", native)
        self.assertIn("jpb_microbenchmarks", native)
        self.assertIn("jpb_microbenchmarks", lake)
        self.assertEqual(sum(w["name"].startswith("jpb-sweep-") for w in native["workloads"]), 6)
        self.assertEqual(native["systems"][0]["binary_path"], "/opt/prf/duckdb")
        self.assertEqual(lake["systems"][0]["binary_path"], "/opt/prf/duckdb-extensions")
        self.assertNotEqual(native["workloads"][0], lake["workloads"][0])
        self.assertEqual(native["suite"]["timeout_seconds"], 86400)
        self.assertEqual(lake["suite"]["timeout_seconds"], 86400)
        self.assertEqual(json.loads((ROOT / "configs/aws.json").read_text())["max_runtime_seconds"], 86400)
        # Resolved launches survive a move without any referenced config files.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "launch.json"
            write_json(path, raw)
            self.assertEqual(load_plan(path), raw)

    def test_smokes_and_group_resource_validation(self):
        self.assertFalse(uses_s3(load_plan(ROOT / "configs/smoke-paper.json")))
        self.assertTrue(uses_s3(load_plan(ROOT / "configs/smoke-paper-s3.json")))
        raw = load_plan(ROOT / "configs/paper.json")
        instance = {"ProcessorInfo": {"SupportedArchitectures": ["arm64"]},
                    "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 32768},
                    "InstanceStorageInfo": {"TotalSizeInGB": 475}}
        validate_resources(raw, instance)
        raw["groups"][1]["config"]["workloads"][0]["estimated_peak_gib"] = 1000
        with self.assertRaisesRegex(ValueError, "storage"):
            validate_resources(raw, instance)
        raw["suite"]["container_memory"] = "6g"
        with self.assertRaisesRegex(ValueError, "memory"):
            validate_plan(raw)

    def test_invalid_groups_rejected_before_launch(self):
        raw = load_plan(ROOT / "configs/paper.json")
        raw["groups"][1]["name"] = raw["groups"][0]["name"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_plan(raw)
        raw["groups"][1]["name"] = "../escape"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_plan(raw)


class PaperSuiteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = ROOT / "duckdb/build/paper-native-release/duckdb"
        if not cls.binary.is_file():
            raise unittest.SkipTest("native PRF binary required")

    def plan(self):
        config = test_suite.SuiteIntegrationTests.configuration(self)
        return {"suite": {k: config["suite"][k] for k in ("container_memory", "container_cpus")},
                "groups": [{"name": n, "config": copy.deepcopy(config)} for n in ("first", "second")]}

    def test_group_resume_export_and_relocated_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, run, cache = root / "plan.json", root / "paper", root / "cache"
            write_json(config, self.plan())
            self.assertEqual(execute(config, "all", run, cache, False), 0)
            files = sorted((run / "groups").glob("*/units/*/results.duckdb"))
            self.assertEqual(len(files), 2)
            before = {str(p.relative_to(run)): sha256(p) for p in files}
            shutil.rmtree(cache)
            self.assertEqual(execute(config, "all", run, cache, True), 0)
            self.assertEqual({str(p.relative_to(run)): sha256(p) for p in files}, before)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(set(summary["groups"]), {"first", "second"})
            self.assertFalse(list((run / "groups").glob("*.tar.gz")))
            with tarfile.open(root / "paper.tar.gz") as archive:
                archive.extractall(root / "moved", filter="data")
            shutil.rmtree(run)
            shutil.rmtree(cache)
            analyze(root / "moved/paper", paper=True)
            self.assertEqual(json.loads((root / "moved/paper/summary.json").read_text())["status"], "complete")
            with patch.dict(os.environ, {"BENCH_IMAGE_METADATA": '{"id":"other-image"}'}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    execute(config, "all", root / "moved/paper", cache, True)

    def test_deadline_between_groups_exports_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, run, cache = root / "plan.json", root / "paper", root / "cache"
            write_json(config, self.plan())
            def expire_after_first_group():
                if (run / "groups/first/analysis.json").exists():
                    os.environ["BENCH_DEADLINE_EPOCH"] = "1"
            with patch.dict(os.environ), patch("benchmarking.suite.checkpoint", side_effect=expire_after_first_group):
                self.assertEqual(execute(config, "all", run, cache, False), 1)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["groups"]["first"]["status"], "complete")
            self.assertEqual(summary["groups"]["second"]["status"], "pending")
            self.assertTrue((root / "paper.tar.gz").exists())
            self.assertEqual(execute(config, "all", run, cache, True), 0)

    def test_failed_group_does_not_hide_later_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.plan()
            raw["groups"][0]["config"]["systems"][0]["variants"][0]["setup_sql"] = ["SET nonexistent_paper_option=true"]
            write_json(root / "plan.json", raw)
            self.assertEqual(execute(root / "plan.json", "all", root / "paper", root / "cache", False), 1)
            summary = json.loads((root / "paper/summary.json").read_text())
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["groups"]["second"]["status"], "complete")
            self.assertEqual(json.loads((root / "paper/run.json").read_text())["status"], "partial")


if __name__ == "__main__":
    unittest.main()
