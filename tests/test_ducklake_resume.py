from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb

from benchmarking import suite
from benchmarking.cloud import Cloud
from benchmarking.suite_io import fingerprint, sha256, write_json
from infra.aws.checkpoint_resume import rebind_legacy_ducklake_units


class WorkerStarted(RuntimeError):
    pass


class CheckpointFixture(unittest.TestCase):
    def checkpoint(self, root, *, normalized=False):
        run = root / "run"
        unit = run / "units" / "case"
        unit.mkdir(parents=True)
        workload = {"name": "lake", "params": {"fixed": {"seed": 7, "fact_rows": 1000},
                    "storage": {"kind": "ducklake", "data_source": "s3"}}}
        payload = {"action": "measure", "workload": workload, "system": "duckdb-paper",
                   "variant": "prf with fallback", "query": "range.sql",
                   "config": {"benchmark": {"repetitions": 10}, "suite": {"disk_reserve_gib": .001}},
                   "run": str(run), "unit_dir": str(unit), "spill": str(root / "spill"),
                   "s3_base_url": "s3://bucket/scratch/run/old/lake"}
        database = unit / "results.duckdb"
        with duckdb.connect(str(database)) as con:
            con.execute("CREATE TABLE timing_runs AS SELECT 42 AS elapsed_ns")
        sql, profile = run / "query.sql", run / "profile.json"
        sql.write_text("SELECT 42;\n")
        profile.write_text('{"latency": 0.1}\n')
        identity = (suite.unit_identity(payload) if normalized else
                    fingerprint({k: v for k, v in payload.items() if k != "existing_only"}))
        state = {"action": "measure", "workload": "lake", "status": "complete", "attempt": 1,
                 "identity": identity, "sha256": sha256(database),
                 "artifacts": {str(p.relative_to(run)): sha256(p) for p in (sql, profile)}}
        write_json(unit / "payload.json", payload)
        write_json(unit / "status.json", state)
        return run, unit, payload, state

    def assert_worker_needed(self, payload):
        with patch("benchmarking.suite.subprocess.Popen", side_effect=WorkerStarted), self.assertRaises(WorkerStarted):
            suite.run_unit(payload, 1)


class DuckLakeResumeTests(CheckpointFixture):
    def test_new_attempt_skips_legacy_and_normalized_checkpoints_without_rewriting_evidence(self):
        for normalized in (False, True):
            with self.subTest(normalized=normalized), tempfile.TemporaryDirectory() as tmp:
                run, unit, payload, state = self.checkpoint(Path(tmp), normalized=normalized)
                before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                with patch("benchmarking.suite.subprocess.Popen", side_effect=AssertionError("reran completed case")):
                    for attempt in ("second", "third"):
                        self.assertTrue(suite.run_unit({**payload, "s3_base_url": f"s3://bucket/scratch/run/{attempt}/lake"}, 1))
                self.assertEqual(before, {p: p.read_bytes() for p in run.rglob("*") if p.is_file()})
                with duckdb.connect(str(unit / "results.duckdb"), read_only=True) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM timing_runs").fetchone()[0], 1)

    def test_changed_measurement_inputs_cannot_reuse_legacy_checkpoint(self):
        changes = [lambda p: p.update(query="shoulder.sql"),
                   lambda p: p.update(variant="sip"),
                   lambda p: p.update(system="other-binary"),
                   lambda p: p["config"]["benchmark"].update(repetitions=3),
                   lambda p: p["workload"]["params"]["fixed"].update(seed=8),
                   lambda p: p["workload"]["params"]["storage"].update(kind="native")]
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as tmp:
                run, unit, payload, state = self.checkpoint(Path(tmp))
                changed = copy.deepcopy(payload)
                changed["s3_base_url"] = "s3://bucket/scratch/run/new/lake"
                change(changed)
                self.assert_worker_needed(changed)

    def test_corrupt_results_profiles_and_unauthenticated_legacy_payload_are_retried(self):
        for corruption in ("database", "profile", "missing_profile", "payload", "missing_payload", "interrupted"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                run, unit, payload, state = self.checkpoint(Path(tmp))
                if corruption == "database":
                    (unit / "results.duckdb").write_bytes(b"corrupt")
                elif corruption == "profile":
                    (run / "profile.json").write_text("{}")
                elif corruption == "missing_profile":
                    (run / "profile.json").unlink()
                elif corruption == "payload":
                    write_json(unit / "payload.json", {**payload, "query": "different.sql"})
                elif corruption == "missing_payload":
                    (unit / "payload.json").unlink()
                else:
                    write_json(unit / "status.json", {**state, "status": "running"})
                self.assert_worker_needed({**payload, "s3_base_url": "s3://bucket/scratch/run/new/lake"})

    def test_non_s3_measurements_keep_strict_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, unit, payload, state = self.checkpoint(Path(tmp))
            payload["workload"]["params"]["storage"]["data_source"] = "http"
            write_json(unit / "payload.json", payload)
            write_json(unit / "status.json", {**state, "identity": suite.unit_identity(payload)})
            self.assert_worker_needed({**payload, "s3_base_url": "http://different-host"})


class LegacyHostResumeTests(CheckpointFixture):
    def test_rebind_preserves_measurements_and_original_metadata_across_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, unit, original_payload, original_state = self.checkpoint(Path(tmp))
            measurements = {p: p.read_bytes() for p in (unit / "results.duckdb", run / "query.sql", run / "profile.json")}
            for attempt in ("second", "third"):
                before_payload = json.loads((unit / "payload.json").read_text())
                before_state = json.loads((unit / "status.json").read_text())
                self.assertEqual(rebind_legacy_ducklake_units(run, "bucket", attempt), 1)
                rebound = json.loads((unit / "payload.json").read_text())
                state = json.loads((unit / "status.json").read_text())
                self.assertEqual(rebound, {**before_payload, "s3_base_url": f"s3://bucket/scratch/run/{attempt}/lake"})
                self.assertEqual(state["identity"], fingerprint(rebound))
                self.assertEqual({k: v for k, v in state.items() if k != "identity"},
                                 {k: v for k, v in original_state.items() if k != "identity"})
                evidence = run / "cloud" / attempt / "ducklake-resume" / "case"
                self.assertEqual(json.loads((evidence / "payload.json").read_text()), before_payload)
                self.assertEqual(json.loads((evidence / "status.json").read_text()), before_state)
                with patch("benchmarking.suite.subprocess.Popen", side_effect=AssertionError("reran completed case")):
                    self.assertTrue(suite.run_unit(rebound, 1))
                self.assertEqual(rebind_legacy_ducklake_units(run, "bucket", attempt), 0)
                self.assertEqual(json.loads((evidence / "payload.json").read_text()), before_payload)
            self.assertEqual(measurements, {p: p.read_bytes() for p in measurements})

    def test_rebind_leaves_invalid_or_unrelated_checkpoints_untouched(self):
        for condition in ("interrupted", "corrupt_database", "corrupt_profile", "changed_payload", "normalized",
                          "other_run", "other_workload", "other_bucket", "http", "path_escape"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                run, unit, payload, state = self.checkpoint(Path(tmp), normalized=condition == "normalized")
                if condition == "interrupted":
                    state["status"] = "failed"
                elif condition == "corrupt_database":
                    (unit / "results.duckdb").write_bytes(b"corrupt")
                elif condition == "corrupt_profile":
                    (run / "profile.json").write_text("{}")
                elif condition == "changed_payload":
                    payload["query"] = "changed.sql"
                elif condition in ("other_run", "other_workload", "other_bucket"):
                    payload["s3_base_url"] = {"other_run": "s3://bucket/scratch/other/old/lake",
                        "other_workload": "s3://bucket/scratch/run/old/other",
                        "other_bucket": "s3://other/scratch/run/old/lake"}[condition]
                    state["identity"] = fingerprint(payload)
                elif condition == "http":
                    payload["workload"]["params"]["storage"]["data_source"] = "http"
                    state["identity"] = fingerprint(payload)
                elif condition == "path_escape":
                    outside = Path(tmp) / "outside"
                    outside.write_text("outside")
                    state["artifacts"] = {"../outside": sha256(outside)}
                write_json(unit / "payload.json", payload)
                write_json(unit / "status.json", state)
                before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                self.assertEqual(rebind_legacy_ducklake_units(run, "bucket", "new"), 0)
                self.assertEqual(before, {p: p.read_bytes() for p in run.rglob("*") if p.is_file()})

    def test_worker_bundle_includes_resume_compatibility(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            cloud = Cloud.__new__(Cloud)
            cloud.local = Path(tmp)
            cloud.aws = MagicMock()
            cloud.bundle("bucket")
            with tarfile.open(Path(tmp) / "worker.tar.gz") as tar:
                self.assertIn("checkpoint_resume.py", tar.getnames())
                self.assertIn(b"rebind_legacy_ducklake_units", tar.extractfile("host.py").read())


if __name__ == "__main__":
    unittest.main()
