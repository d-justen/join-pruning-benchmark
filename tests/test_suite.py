from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from benchmarking.datasets import download
from benchmarking.experiment_config import load_experiment_config
from benchmarking.resources import connect
from benchmarking.suite import aggregate, analyze, execute, export, run_unit
from benchmarking.suite_io import disk_preflight, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class SuiteIOTests(unittest.TestCase):
    def test_extension_smoke_selects_http_storage(self):
        config = load_experiment_config(ROOT / "configs/smoke-extensions.json")
        self.assertEqual(config.workloads[0].storage_kind, "ducklake")
        self.assertEqual(config.workloads[0].ducklake_http_latency_ms, 0)
        self.assertEqual(config.workloads[0].parquet_file_size_mb, 1)
        self.assertIn("q01.sql", config.workloads[0].query_names)
        self.assertTrue(json.loads((ROOT / "configs/smoke-extensions.json").read_text())["suite"]["verify_http_requests"])

    def test_download_repairs_corrupt_cache_and_rejects_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "source", root / "cache/file"
            source.write_bytes(b"pinned input\n")
            spec = {"url": source.as_uri(), "sha256": sha256(source), "bytes": source.stat().st_size}
            self.assertEqual(download(spec, target).read_bytes(), source.read_bytes())
            target.write_bytes(b"corrupt")
            self.assertEqual(download(spec, target).read_bytes(), source.read_bytes())
            with patch("benchmarking.datasets.time.sleep"), self.assertRaisesRegex(ValueError, "checksum"):
                download({**spec, "sha256": "0" * 64}, target)
            self.assertFalse(target.with_suffix(".part").exists())

    def test_resource_policy_overrides_preparation_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = {"threads": 1, "memory_limit": "64MiB", "temp_directory": tmp,
                      "max_temp_directory_size": "100MiB", "storage_compatibility_version": "v1.2.0"}
            with patch.dict(os.environ, {"BENCH_PREPARATION_CONFIG": json.dumps(policy)}):
                with connect(config={"threads": 8}) as con:
                    self.assertEqual(con.execute("SELECT current_setting('threads')").fetchone()[0], 1)
                    self.assertEqual(con.execute("SELECT current_setting('temp_directory')").fetchone()[0], tmp)
                    self.assertEqual(con.execute("SELECT current_setting('memory_limit')").fetchone()[0], "64.0 MiB")

    def test_disk_budget_fails_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "insufficient space"):
                disk_preflight(Path(tmp), 10**9, 1)

    def test_timeout_kills_child_group_and_persists_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"action": "measure", "unit_dir": str(root / "unit"), "spill": str(root / "spill"),
                       "config": {"suite": {"disk_reserve_gib": 0.00001}}}
            real_popen = subprocess.Popen
            def sleeper(*args, **kwargs):
                return real_popen([sys.executable, "-c", "import time;time.sleep(30)"], **kwargs)
            with patch("benchmarking.suite.subprocess.Popen", side_effect=sleeper):
                self.assertFalse(run_unit(payload, .1))
            status = json.loads((root / "unit/status.json").read_text())
            self.assertEqual(status["failure"], "timeout")
            self.assertEqual(status["status"], "failed")
            self.assertLess(status["elapsed_seconds"], 10)
            self.assertFalse((root / "spill").exists())


class SuiteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = ROOT / "duckdb/build/paper-native-release/duckdb"
        if not cls.binary.exists():
            raise unittest.SkipTest("native PRF binary required; Docker smoke covers this in Linux")

    def configuration(self):
        raw = json.loads((ROOT / "configs/smoke.json").read_text())
        raw["workloads"] = [raw["workloads"][0]]
        raw["workloads"][0]["params"]["fixed"]["fact_rows"] = 1000
        raw["workloads"][0]["params"]["sweep"]["values"] = [1000]
        raw["workloads"][0]["params"]["query_types"] = [{"kind": "range", "probe_layouts": ["uniform"]}]
        raw["workloads"][0]["estimated_peak_gib"] = .01
        raw["systems"] = [raw["systems"][0]]
        raw["systems"][0]["binary_path"] = str(self.binary)
        raw["systems"][0]["variants"] = [raw["systems"][0]["variants"][0]]
        raw["suite"].update(disk_reserve_gib=.001, evict_datasets=True)
        raw.pop("microbenchmarks")
        raw.pop("jpb_microbenchmarks")
        return raw

    def test_resume_export_relocation_and_changed_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, run, cache = root / "config.json", root / "run", root / "cache"
            raw = self.configuration()
            write_json(config, raw)
            self.assertEqual(execute(config, "all", run, cache, False), 0)
            units = list((run / "units").glob("*/results.duckdb"))
            self.assertEqual(len(units), 1)
            old_hash = sha256(units[0])
            self.assertEqual(execute(config, "all", run, cache, True), 0)
            self.assertEqual(sha256(units[0]), old_hash)
            with patch.dict(os.environ, {"BENCH_IMAGE_METADATA": '{"id":"different-image"}'}):
                with self.assertRaisesRegex(ValueError, "image or Python runtime"):
                    execute(config, "all", run, cache, True)
            with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 1)
                parameters = json.loads(con.execute("SELECT query_parameters FROM timing_runs").fetchone()[0])
                self.assertEqual(parameters["sweep_kind"], "dimension_cardinality")
                self.assertEqual(parameters["sweep_value"], 1000)
            # No original source data or run directory is needed for analysis.
            archive = export(run)
            moved = root / "relocated"
            moved.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(moved, filter="data")
            shutil.rmtree(run)
            shutil.rmtree(cache)
            analyze(moved / "run", paper=False)
            with duckdb.connect(str(moved / "run/results.duckdb"), read_only=True) as con:
                profile = con.execute("SELECT profile_path FROM profile_runs").fetchone()[0]
                self.assertTrue(Path(profile).is_file())
            raw["benchmark"]["repetitions"] = 2
            write_json(config, raw)
            with self.assertRaisesRegex(ValueError, "config or runner code changed"):
                execute(config, "all", moved / "run", root / "new-cache", True)

    def test_failed_variant_isolated_and_partial_results_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.configuration()
            raw["systems"][0]["variants"].insert(0, {"label": "bad", "setup_sql": ["SET nonexistent_suite_option=true"]})
            config, run = root / "config.json", root / "partial"
            write_json(config, raw)
            self.assertEqual(execute(config, "all", run, root / "cache", False), 1)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["status"], "partial")
            self.assertTrue((root / "partial.tar.gz").is_file())
            with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 1)
            # Aggregating repeatedly never imports failed/partial unit rows.
            aggregate(run)
            with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 1)

    def test_prepare_then_run_and_transient_failure_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.configuration()
            wrapper = root / "duckdb"
            failed_once = root / "failed-once"
            wrapper.write_text(
                f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
                f"marker=Path({str(failed_once)!r})\n"
                "if '--version' not in sys.argv and not marker.exists():\n"
                " marker.touch()\n sys.exit(13)\n"
                f"os.execv({str(self.binary)!r}, [{str(self.binary)!r}, *sys.argv[1:]])\n"
            )
            wrapper.chmod(0o755)
            raw["systems"][0]["binary_path"] = str(wrapper)
            config, run, cache = root / "config.json", root / "run", root / "cache"
            write_json(config, raw)
            self.assertEqual(execute(config, "prepare", run, cache, False), 0)
            self.assertFalse((run / "results.duckdb").exists())
            databases = list((cache / "workloads").rglob("*.duckdb"))
            original_data = {str(p): sha256(p) for p in databases}
            self.assertEqual(execute(config, "run", run, cache, True), 1)
            self.assertEqual(execute(config, "run", run, cache, True), 0)
            self.assertEqual(original_data, {str(p): sha256(p) for p in databases})
            statuses = [json.loads(p.read_text()) for p in (run / "units").glob("*/status.json")]
            measured = next(s for s in statuses if s["action"] == "measure")
            self.assertEqual(measured["attempt"], 2)
            self.assertEqual(measured["status"], "complete")
            with duckdb.connect(str(run / "results.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
