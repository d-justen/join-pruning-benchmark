from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb

from benchmarking import load_experiment_config
from benchmarking.duckdb_build import ResolvedSystem
from benchmarking.experiment_config import parse_jpb_microbenchmark_config
from benchmarking.experiment_runner import run_experiment
from benchmarking.jpb_microbenchmarks import (
    ARTIFACT_SUBDIR, available_filters, load_jpb_manifest, resolve_jpb_cases, run_jpb_microbenchmarks, validate_results,
)
from benchmarking.microbenchmarks import read_csv
from benchmarking.workloads.cross_product import (
    _resolve_sweep_cases, _sweep_dimension_table_name, _sweep_mask_column, prepare_cross_product_workload,
)
from paper.figures.build_all import build_figures
from paper.figures.jpb_microbenchmark import aggregate_rows, build_jpb_artifacts

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("JPB_BENCHMARK_BINARY", ROOT / "duckdb/build/paper-native-release/tools/join_pruning_benchmark"))
SMOKE = ROOT / "paper/configs/local_paper_jpb_microbenchmarks_smoke.json"


def write_config(root: Path, *, full_filters=False):
    raw = json.loads(SMOKE.read_text())
    raw["output_root"] = str(root / "out")
    raw["systems"][0]["binary_path"] = str(ROOT / "duckdb/build/paper-native-release/duckdb")
    raw["workloads"][0]["params"]["fixed"].update(fact_rows=20003, selectivity=0.25)
    raw["workloads"][0]["params"]["sweep"]["values"] = [1000, 10000]
    raw["jpb_microbenchmarks"].update(repetitions=2, warmups=0)
    if full_filters:
        raw["jpb_microbenchmarks"]["required_filters"] += ["grafite", "diva"]
    path = root / "config.json"
    path.write_text(json.dumps(raw))
    return load_experiment_config(path)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_rows(config):
    rows = []
    for case in resolve_jpb_cases(config):
        groups = (case["probe_rows"] + case["row_group_size"] - 1) // case["row_group_size"]
        for name in ("grafite", "diva", "prf_uncompressed", "prf"):
            for rep in range(config.jpb_microbenchmarks.repetitions):
                row = {key: value for key, value in case.items() if key != "selected_sql"}
                row.update(filter=name, repetition=rep, build_time_ns=100 + rep * 100, finalization_time_ns=10,
                           range_time_ns=20, point_time_ns=300 - rep * 50, total_time_ns=420 + rep * 50,
                           range_probes=groups, groups_rejected=0, groups_accepted=0, groups_mixed=groups,
                           rows_rejected=0, rows_accepted=0, point_probes=case["probe_rows"], point_candidates=5100,
                           candidate_rows=5100, true_positives=5000, false_positives=100, false_negatives=0,
                           empty_ranges=0, range_false_positives=0, summary_bytes=1000, representation="fixture",
                           build_checksum=123, probe_checksum=456, measured_bpk=8)
                row.update({key: getattr(config.jpb_microbenchmarks, key) for key in ("grafite_bpk", "diva_bpk", "max_fpr")})
                rows.append({key: str(value) for key, value in row.items()})
    return rows


class JPBPipelineTests(unittest.TestCase):
    def test_config_validation_and_paper_cases(self):
        valid = dict(system_name="system", workload_name="workload")
        for change in ({"repetitions": 0}, {"warmups": -1}, {"order_seed": 2**32},
                       {"grafite_bpk": float("nan")}, {"max_fpr": 2}, {"unknown": 1},
                       {"required_filters": ["bloom"]}, {"required_filters": ["prf", "prf"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_jpb_microbenchmark_config({**valid, **change})
        config = load_experiment_config(ROOT / "paper/configs/local_paper_experiments.json")
        cases = resolve_jpb_cases(config)
        self.assertEqual(len(cases), 16)
        self.assertEqual({c["build_keys"] for c in cases}, {1000, 1000000})
        self.assertEqual({c["probe_rows"] for c in cases}, {500000000})
        for change in ({"join_count": 2}, {"build_key_multiplicity": 2}, {"key_domain_size": 1000000000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                workload = config.workloads[0]
                definition = workload.experiment_definition
                changed = replace(workload, experiment_definition=replace(definition, fixed=replace(definition.fixed, **change)))
                resolve_jpb_cases(replace(config, workloads=(changed,)))

    def test_selection_sql_matches_actual_generated_dimension_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp))
            workload = config.workloads[0]
            manifest = prepare_cross_product_workload(workload, Path(tmp) / "workload")
            cases = resolve_jpb_cases(config)
            with duckdb.connect(str(manifest.db_path), read_only=True) as con:
                resolved = _resolve_sweep_cases(workload.experiment_definition, storage_kind="duckdb")
                for case in resolved:
                    for query in workload.experiment_definition.query_types:
                        table = _sweep_dimension_table_name(case, 1, query)
                        mask = _sweep_mask_column(case, query)
                        expected = con.execute(f'SELECT d_key FROM "{table}" WHERE "{mask}" ORDER BY d_key').fetchall()
                        micro_case = next(c for c in cases if c["dimension_cardinality"] == case.dimension_cardinality
                                          and c["pattern"] == query.kind)
                        self.assertEqual(con.execute(micro_case["selected_sql"]).fetchall(), expected)

    def test_validator_rejects_incomplete_inconsistent_and_mismatched_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root, full_filters=True)
            cases = resolve_jpb_cases(config)
            rows = fixture_rows(config)
            names = ["prf", "prf_uncompressed", "grafite", "diva"]
            path = root / "measurements.csv"
            write_rows(path, rows)
            self.assertEqual(validate_results(path, config.jpb_microbenchmarks, cases, names)["measurement_rows"], 128)
            bad_rows = [rows[:-1], rows + [rows[0]]]
            for changes in ({"total_time_ns": "999"}, {"false_negatives": "1"}, {"probe_checksum": "999"},
                            {"groups_mixed": "0"}, {"point_probes": "0"}, {"max_fpr": "nan"},
                            {"candidate_rows": "1"}, {"build_time_ns": "-1"}, {"seed": "999"}):
                bad_rows.append([{**rows[0], **changes}, *rows[1:]])
            for bad in bad_rows:
                write_rows(path, bad)
                with self.assertRaises(ValueError):
                    validate_results(path, config.jpb_microbenchmarks, cases, names)

    def test_four_filter_chart_has_additive_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root)
            config = replace(config, jpb_microbenchmarks=replace(config.jpb_microbenchmarks, repetitions=3))
            rows = fixture_rows(config)
            # Medians of these phases would sum to 420 ns, but the median total
            # is 1020 ns. The additive means must instead sum to 820 ns.
            for row in rows:
                rep = int(row["repetition"])
                row["build_time_ns"] = str([100, 200, 900][rep])
                row["point_time_ns"] = str([900, 100, 200][rep])
                row["total_time_ns"] = str(int(row["build_time_ns"]) + int(row["point_time_ns"]) + 20)
            series = aggregate_rows(rows)
            self.assertEqual(len(series), 64)
            for row in series:
                self.assertAlmostEqual(row["total_ms"], row["build_ms"] + row["range_ms"] + row["point_ms"])
                self.assertAlmostEqual(row["total_ms"], 820 / 1e6)
                self.assertAlmostEqual(row["median_total_ms"], 1020 / 1e6)
                self.assertIsNone(row["mean_range_false_positive_rate"])
            write_rows(root / "measurements.csv", rows)
            artifacts = build_jpb_artifacts(root / "measurements.csv", root / "paper", root / "paper/figures", [])
            self.assertEqual(artifacts["verification"]["bar_count"], 64)
            self.assertEqual(artifacts["verification"]["panel_count"], 4)
            self.assertEqual(artifacts["verification"]["phase_order"], ["build", "range", "point"])
            self.assertGreater(Path(artifacts["figure_path"]).stat().st_size, 1000)

    def test_required_competitors_fail_before_execution_and_invalidate_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root, full_filters=True)
            binary = root / "binary"
            binary.write_text("fixture")
            with patch("benchmarking.jpb_microbenchmarks.resolve_binary", return_value=binary), \
                 patch("benchmarking.jpb_microbenchmarks.available_filters", return_value=["prf", "prf_uncompressed"]), \
                 patch("subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "lacks required filters: diva, grafite"):
                    run_jpb_microbenchmarks(config, SimpleNamespace(), config.experiment_root)
                run.assert_not_called()
            saved = json.loads((config.experiment_root / ARTIFACT_SUBDIR / "manifest.json").read_text())
            self.assertEqual(saved["status"], "failed")

    def test_micro_only_preserves_sql(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root)
            saved_root = config.experiment_root
            saved_root.mkdir(parents=True)
            # Adding the new microbenchmark to an existing experiment is allowed.
            raw = json.loads(config.config_path.read_text())
            old = {**raw, "jpb_microbenchmarks": None}
            (saved_root / "config.json").write_text(json.dumps(old))
            preserved = {"results.duckdb": b"SQL", "summary.json": b"summary", "workloads/data": b"data"}
            for name, data in preserved.items():
                path = saved_root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
            with patch("benchmarking.experiment_runner.resolve_system", return_value=SimpleNamespace(name="duckdb-paper")), \
                 patch("benchmarking.experiment_runner.run_jpb_microbenchmarks") as run, \
                 patch("benchmarking.experiment_runner.prepare_workloads") as prepare:
                run_experiment(config, only_microbenchmarks=True)
                run.assert_called_once()
                prepare.assert_not_called()
            for name, data in preserved.items():
                self.assertEqual((saved_root / name).read_bytes(), data)
            raw["workloads"][0]["params"]["fixed"]["fact_rows"] += 1
            config.config_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "SQL configuration differs"):
                run_experiment(load_experiment_config(config.config_path), only_microbenchmarks=True)

    def test_micro_only_resolves_both_suite_systems(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp))
            raw = json.loads(config.config_path.read_text())
            raw["systems"].append({**raw["systems"][0], "name": "primitive-system"})
            raw["microbenchmarks"] = {"system_name": "primitive-system"}
            config.config_path.write_text(json.dumps(raw))
            config = load_experiment_config(config.config_path)
            with patch("benchmarking.experiment_runner.resolve_system",
                       side_effect=lambda system, *args: SimpleNamespace(name=system.name)) as resolve, \
                 patch("benchmarking.experiment_runner.run_microbenchmarks") as primitive, \
                 patch("benchmarking.experiment_runner.run_jpb_microbenchmarks") as jpb, \
                 patch("benchmarking.experiment_runner.prepare_workloads") as prepare:
                run_experiment(config, only_microbenchmarks=True)
                self.assertEqual(resolve.call_count, 2)
                self.assertEqual(primitive.call_args.args[1].name, "primitive-system")
                self.assertEqual(jpb.call_args.args[1].name, "duckdb-paper")
                prepare.assert_not_called()


@unittest.skipUnless(BINARY.is_file(), "build join_pruning_benchmark or set JPB_BENCHMARK_BINARY")
class JPBNativeTests(unittest.TestCase):
    def test_always_passing_approximate_ranges_count_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keys.sql").write_text("SELECT unnest([1,32]::UBIGINT[])")
            (root / "cases.tsv").write_text("approx\tshoulder\tordered\t32\t32\t8\t7\t2\tkeys.sql\n")
            result = subprocess.run([str(BINARY), f"--cases={root / 'cases.tsv'}", f"--output={root / 'measurements.csv'}",
                                     "--warmups=0", "--repetitions=1", "--max-fpr=1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for row in read_csv(root / "measurements.csv"):
                if row["filter"].startswith("prf"):
                    self.assertEqual(int(row["groups_accepted"]), 2)
                    self.assertEqual(int(row["point_probes"]), 0)
                    self.assertEqual(int(row["point_time_ns"]), 0)
                    self.assertEqual(int(row["candidate_rows"]), 16)
                    self.assertEqual(int(row["false_positives"]), 14)
                    self.assertEqual(int(row["false_negatives"]), 0)

    def test_native_all_available_filters_and_saved_artifact_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_config(root)
            system = ResolvedSystem("binary", "duckdb-paper", config.systems[0].binary_path,
                                    "fixture-host", None, None, None, build_dir=BINARY.parent)
            with patch("benchmarking.jpb_microbenchmarks.resolve_binary", return_value=BINARY):
                directory = run_jpb_microbenchmarks(config, system, config.experiment_root)
            (config.experiment_root / "config.json").write_text(config.config_path.read_text())
            manifest = load_jpb_manifest(config.experiment_root, config)
            self.assertEqual(manifest["available_filters"], available_filters(BINARY))
            rows = read_csv(directory / "measurements.csv")
            self.assertTrue(all(int(row["false_negatives"]) == 0 for row in rows))
            self.assertTrue(any(int(row["groups_accepted"]) > 0 for row in rows))
            self.assertTrue(any(int(row["groups_rejected"]) > 0 for row in rows))
            self.assertTrue(any(int(row["groups_mixed"]) > 0 for row in rows))
            self.assertTrue(all(int(row["range_probes"]) == 10 for row in rows))
            # Independently check the ordered stream, including multiplicities,
            # instead of relying only on the harness's own membership oracle.
            with duckdb.connect() as con:
                for case in resolve_jpb_cases(config):
                    if case["layout"] != "ordered":
                        continue
                    keys = {row[0] for row in con.execute(case["selected_sql"]).fetchall()}
                    checksum, positives = 14695981039346656037, 0
                    for i in range(case["probe_rows"]):
                        key = 1 + i * case["dimension_cardinality"] // case["probe_rows"]
                        checksum = ((checksum ^ key) * 1099511628211) % 2**64
                        positives += key in keys
                    samples = [row for row in rows if row["query"] == case["query"]]
                    self.assertTrue(all(int(row["probe_checksum"]) == checksum for row in samples))
                    self.assertTrue(all(int(row["true_positives"]) == positives for row in samples))
            before = (directory / "manifest.json").read_bytes()
            with patch("benchmarking.jpb_microbenchmarks.resolve_binary", return_value=BINARY), \
                 patch("subprocess.run", side_effect=AssertionError("must reuse saved artifacts")):
                run_jpb_microbenchmarks(config, system, config.experiment_root, reuse_completed=True)
                output = build_figures(config.experiment_root, None)
            self.assertEqual((directory / "manifest.json").read_bytes(), before)
            self.assertTrue((output / "figures/jpb_simulated_filtering.pdf").is_file())
            changed = replace(config, jpb_microbenchmarks=replace(config.jpb_microbenchmarks, max_fpr=0.1))
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_jpb_manifest(config.experiment_root, changed)
            (directory / "inputs/0.sql").write_text("SELECT 1")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                load_jpb_manifest(config.experiment_root, config)
            with patch("benchmarking.jpb_microbenchmarks.resolve_binary", side_effect=FileNotFoundError("missing tool")):
                with self.assertRaises(FileNotFoundError):
                    run_jpb_microbenchmarks(config, system, config.experiment_root)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_jpb_manifest(config.experiment_root, config)

    def test_range_boundaries_shortcuts_and_zero_point_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definitions = [
                ("boundaries", 10, 10, 2, [1, 3, 4, 10]),
                ("empty", 4096, 1, 2048, [4095, 4096]),
                ("full", 2, 7, 2048, [1, 2]),
            ]
            lines = []
            for name, domain, count, group_size, keys in definitions:
                (root / f"{name}.sql").write_text("SELECT unnest([" + ",".join(map(str, keys)) + "]::UBIGINT[])")
                lines.append(f"{name}\trange\tordered\t{domain}\t{count}\t{group_size}\t7\t{len(keys)}\t{name}.sql\n")
            (root / "cases.tsv").write_text("".join(lines))
            result = subprocess.run([str(BINARY), f"--cases={root / 'cases.tsv'}", f"--output={root / 'measurements.csv'}",
                                    "--warmups=0", "--repetitions=1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = read_csv(root / "measurements.csv")
            self.assertEqual(len(rows), 3 * len(available_filters(BINARY)))
            for row in rows:
                self.assertEqual(int(row["false_negatives"]), 0)
                if not row["filter"].startswith("prf"):
                    continue
                if row["query"] == "boundaries":
                    self.assertEqual([int(row[key]) for key in ("groups_rejected", "groups_accepted", "groups_mixed")], [2, 1, 2])
                    self.assertEqual(int(row["point_probes"]), 4)
                    self.assertEqual(int(row["candidate_rows"]), 4)
                else:
                    self.assertEqual(int(row["point_probes"]), 0)
                    self.assertEqual(int(row["point_time_ns"]), 0)
                    self.assertEqual(int(row["candidate_rows"]), 7 if row["query"] == "full" else 0)


if __name__ == "__main__":
    unittest.main()
