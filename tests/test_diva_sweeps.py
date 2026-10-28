from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from benchmarking.benchmark_binary import select_benchmark_binary
from benchmarking.duckdb_runner import ensure_results_schema
from benchmarking.resources import connect
from benchmarking.suite import _resources, prepare_one, validate, measurement_workloads
from benchmarking.suite_coverage import expected_parameters, expected_queries, result_coverage
from benchmarking.suite_jobs import job_configs, split_jobs, combine_jobs
from benchmarking.suite_io import write_json, fingerprint, sha256

ROOT = Path(__file__).resolve().parents[1]


class DivaSelectionTests(unittest.TestCase):
    def test_backend_comparison_rejects_answer_changes(self):
        from benchmarking.diva_validation import compare_results
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backend in ("portable", "sve2"):
                folder = root / backend / "microbenchmarks/filter-probes"
                folder.mkdir(parents=True)
                point = {"filter": "diva", "repetition": 0, "cluster_count": 64, "summary_bytes": 4096,
                         "false_positives": 2, "build_time_ns": 100, "post_processing_time_ns": 0,
                         "probe_throughput_per_sec": 100 if backend == "portable" else 150}
                ranges = {"summary": "diva", "repetition": 0, "cluster_count": 64, "range_count": 1,
                          "possibly_overlapping": 2, "total_query_time_ns": 100, "queries_per_sec": 100,
                          "post_processing_time_ns": 0}
                for name, row in (("point", point), ("range", ranges)):
                    with (folder / (name + ".csv")).open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
            report = compare_results(root)
            self.assertTrue(report["identical_inputs_memory_and_answers"])
            self.assertEqual(report["comparisons"][1]["sve2_over_portable"], 1.5)
            path = root / "sve2/microbenchmarks/filter-probes/point.csv"
            path.write_text(path.read_text().replace("4096", "4097"))
            with self.assertRaisesRegex(ValueError, "disagree"):
                compare_results(root)

    def test_dispatch_checks_hardware_and_binary_and_can_force_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "filter_benchmark"
            fast = base.with_name(base.name + "_sve2")
            for path in (base, fast):
                path.touch()
                path.chmod(0o755)
            with patch.dict(os.environ, {"BENCH_DIVA_BACKEND": "auto"}):
                with patch("benchmarking.benchmark_binary.supports_sve_bitperm", return_value=True):
                    self.assertEqual(select_benchmark_binary(base), fast.resolve())
                with patch("benchmarking.benchmark_binary.supports_sve_bitperm", return_value=False):
                    self.assertEqual(select_benchmark_binary(base), base.resolve())
            with patch.dict(os.environ, {"BENCH_DIVA_BACKEND": "sve2"}), patch(
                    "benchmarking.benchmark_binary.supports_sve_bitperm", return_value=False):
                with self.assertRaisesRegex(ValueError, "kernel"):
                    select_benchmark_binary(base)
            with patch.dict(os.environ, {"BENCH_DIVA_BACKEND": "portable"}), patch(
                    "benchmarking.benchmark_binary.supports_sve_bitperm", return_value=True):
                self.assertEqual(select_benchmark_binary(base), base.resolve())
            fast.unlink()
            with patch.dict(os.environ, {"BENCH_DIVA_BACKEND": "sve2"}), patch(
                    "benchmarking.benchmark_binary.supports_sve_bitperm", return_value=True):
                with self.assertRaisesRegex(ValueError, "optimized binary"):
                    select_benchmark_binary(base)


class SweepSuiteTests(unittest.TestCase):
    def raw(self, name="smoke-sweeps"):
        return json.loads((ROOT / "configs" / (name + ".json")).read_text())

    def test_full_sweeps_keep_paper_parameters_and_all_variants(self):
        raw = self.raw("graviton-sweeps")
        validate(raw)
        paper = json.loads((ROOT / "paper/configs/local_paper_sweeps.json").read_text())
        self.assertEqual([w["params"] for w in raw["workloads"]], [w["params"] for w in paper["workloads"]])
        self.assertEqual(sum(len(expected_queries(w, ROOT)) for w in raw["workloads"]), 89)
        self.assertEqual(sum(len(s["variants"]) for s in raw["systems"]), 8)
        missing = copy.deepcopy(raw)
        missing["workloads"].pop()
        with self.assertRaisesRegex(ValueError, "required workloads"):
            validate(missing)
        full = self.raw("graviton-full")
        validate(full)
        self.assertTrue(set(w["name"] for w in raw["workloads"]).issubset(w["name"] for w in full["workloads"]))
        for section in ("microbenchmarks", "jpb_microbenchmarks"):
            self.assertIn("diva", full[section]["required_filters"])
        self.assertEqual(measurement_workloads(self.raw("graviton-microbenchmarks")), [])

    def test_preparation_overrides_are_honored_and_capped(self):
        raw = self.raw("graviton-sweeps")
        density = raw["workloads"][3]
        with tempfile.TemporaryDirectory() as tmp:
            policy = _resources(raw, Path(tmp), density)
            self.assertEqual(policy["threads"], 4)
            self.assertEqual(policy["memory_limit"], "8GiB")
            self.assertEqual(policy["max_temp_directory_size"], "64GiB")
            with patch.dict(os.environ, {"BENCH_PREPARATION_CONFIG": json.dumps(policy)}):
                with connect(config={"threads": 32}) as con:
                    self.assertEqual(con.execute("SELECT current_setting('threads')").fetchone()[0], 4)
            density["preparation"] = {"threads": 100, "memory_limit": "100GiB"}
            policy = _resources(raw, Path(tmp), density)
            self.assertEqual(policy["threads"], 8)
            self.assertEqual(policy["memory_limit"], "20GiB")

    def test_suite_resumes_committed_density_preparation(self):
        from benchmarking.workloads import cross_product as module
        raw = self.raw()
        workload = raw["workloads"][3]
        workload["params"]["fixed"].update(fact_rows=245777, dimension_cardinality=1000, key_domain_size=1000, selectivity=.1)
        workload["params"]["sweep"]["values"] = [500, 1000, 3000]
        workload["params"]["preparation"]["batch_rows"] = 122880
        insert = module._insert_density_fact_batches
        def interrupted(con, sql, fact_rows, batch_rows, next_id):
            insert(con, sql, batch_rows, batch_rows, next_id)
            raise RuntimeError("simulated interruption after commit")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"config": raw, "workload": workload, "cache": str(root / "cache"),
                       "run": str(root / "run"), "unit_dir": str(root / "unit")}
            policy = _resources(raw, root / "spill", workload)
            with patch.dict(os.environ, {"BENCH_PREPARATION_CONFIG": json.dumps(policy)}):
                with patch.object(module, "_insert_density_fact_batches", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        prepare_one(payload)
                database = next((root / "cache/workloads").glob("*/benchmark.duckdb"))
                with duckdb.connect(str(database), read_only=True) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM fact").fetchone()[0], 122880)
                with patch.object(module, "_insert_density_fact_batches", wraps=insert) as resumed:
                    prepare_one(payload)
                self.assertEqual(resumed.call_args.args[-1], 122881)
                with duckdb.connect(str(database), read_only=True) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM fact").fetchone()[0], 245777)
                self.assertTrue(database.with_name("complete.json").is_file())

    def test_coverage_detects_absent_sweeps_and_duplicate_repetitions(self):
        raw = self.raw()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_results_schema(root / "results.duckdb")
            report = result_coverage(raw, root)
            self.assertEqual(report["status"], "partial")
            missing = report["timing_runs"]["missing"]
            with duckdb.connect(str(root / "results.duckdb")) as con:
                con.executemany("INSERT INTO timing_runs(workload_name,system_name,variant_label,query_name,repetition) VALUES (?,?,?,?,?)", missing)
                con.executemany("INSERT INTO profile_runs(workload_name,system_name,variant_label,query_name) VALUES (?,?,?,?)", report["profile_runs"]["missing"])
                for workload in raw["workloads"]:
                    for query, parameters in expected_parameters(workload).items():
                        for table in ("timing_runs", "profile_runs"):
                            con.execute(f"UPDATE {table} SET query_parameters=? WHERE workload_name=? AND query_name=?",
                                        [json.dumps(parameters), workload["name"], query])
            self.assertEqual(result_coverage(raw, root)["status"], "complete")
            with duckdb.connect(str(root / "results.duckdb")) as con:
                con.execute("INSERT INTO timing_runs SELECT * FROM timing_runs LIMIT 1")
            self.assertTrue(result_coverage(raw, root)["timing_runs"]["duplicate"])

    def test_split_preserves_definitions_and_combine_rejects_missing_or_mixed_jobs(self):
        raw = self.raw()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            write_json(config, raw)
            paths = split_jobs(config, root / "jobs")
            self.assertEqual(len(paths), 6)
            sources = []
            for i, job in enumerate(job_configs(raw)):
                self.assertEqual(job["workloads"][0], raw["workloads"][i])
                validate(job)
                source = root / f"job-{i}"
                write_json(source / "request.json", job)
                write_json(source / "run.json", {"status": "complete", "source": "code", "binaries": {},
                                                 "runtime": {"image_id": "image"}, "config_sha256": fingerprint(job)})
                write_json(source / "coverage.json", {"status": "complete"})
                sources.append(source)
            with self.assertRaisesRegex(ValueError, "missing jobs"):
                combine_jobs(config, sources[:-1], root / "missing")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                combine_jobs(config, sources + sources[:1], root / "duplicate")
            state = json.loads((sources[-1] / "run.json").read_text())
            state["runtime"]["image_id"] = "different"
            write_json(sources[-1] / "run.json", state)
            with self.assertRaisesRegex(ValueError, "runtime differ"):
                combine_jobs(config, sources, root / "mixed")


if __name__ == "__main__":
    unittest.main()
