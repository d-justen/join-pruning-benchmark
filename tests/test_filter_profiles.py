from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import duckdb

from benchmarking.duckdb_runner import ensure_results_schema, populate_profile_metrics
from benchmarking.experiment_config import load_experiment_config
from benchmarking.experiment_runner import run_experiment
from benchmarking.filter_profiles import JoinFilterMetrics, extract_join_filter_metrics
from benchmarking.results_analysis import _filter_phase_rows, analyze_experiment
from paper.figures.data import (
    MetricRow, PlanMetricRow, load_metric_rows, summarize_benchmark_metrics, write_filter_phase_report,
)
from paper.figures.build_all import (
    _draw_filter_phase_overlays, _write_benchmark_summary_markdown,
)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "duckdb/build/paper-native-release/duckdb"


def snapshot(build=.010, analysis=.002, compression=.003, prf=1, bloom=0, count=1):
    return {"join_finalize_elapsed_seconds": build + analysis + compression,
            "join_finalize_build_elapsed_seconds": build,
            "prf_analysis_elapsed_seconds": analysis,
            "prf_compression_elapsed_seconds": compression,
            "finalize_count": count, "prf_build_count": prf, "bloom_build_count": bloom,
            "perfect_hash_join_count": 0, "external_finalize_count": 0}


class FilterProfileTests(unittest.TestCase):
    def test_elapsed_phases_and_counts_are_not_worker_or_filter_sums(self):
        profile = {"join_filter": snapshot(prf=2, bloom=1, count=2),
                   "prefix_range_filter": {"build_worker_seconds": 999, "build_phase_elapsed_seconds": 100},
                   "children": [{"join_filter": snapshot(.020, 0, 0, prf=0, bloom=1)}]}
        phases = extract_join_filter_metrics(profile)
        self.assertAlmostEqual(phases.join_finalize_build_ms, 30)
        self.assertAlmostEqual(phases.join_finalize_ms, 35)
        self.assertEqual((phases.prf_analysis_ms, phases.prf_compression_ms), (2, 3))
        self.assertEqual((phases.join_finalize_count, phases.prf_build_count, phases.bloom_build_count), (3, 2, 2))

    def test_zero_is_observed_but_missing_and_invalid_are_unknown(self):
        self.assertEqual(extract_join_filter_metrics({}), JoinFilterMetrics())
        self.assertEqual(extract_join_filter_metrics({"prefix_range_filter": {
            "build_phase_elapsed_seconds": 1, "compression_phase_elapsed_seconds": 2,
        }}), JoinFilterMetrics())
        bloom = extract_join_filter_metrics({"join_filter": snapshot(.01, 0, 0, prf=0, bloom=1)})
        self.assertEqual((bloom.prf_analysis_ms, bloom.prf_compression_ms), (0, 0))
        for invalid in (None, True, "0.1", -1, float("nan"), float("inf")):
            item = snapshot()
            item["prf_analysis_elapsed_seconds"] = invalid
            phases = extract_join_filter_metrics({"join_filter": snapshot(), "children": [{"join_filter": item}]})
            self.assertIsNone(phases.prf_analysis_ms)
        self.assertEqual(extract_join_filter_metrics({"children": [
            {"join_filter": snapshot()}, {"join_filter": {}}]}), JoinFilterMetrics())

    def test_schema_migration_and_reimport_persist_comparable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "results.duckdb"
            ensure_results_schema(db)
            # Simulate an old result database and migrate it when importing profiles.
            with duckdb.connect(str(db)) as con:
                for field in JoinFilterMetrics().to_dict():
                    con.execute(f"ALTER TABLE profile_runs DROP COLUMN {field}")
                path = root / "profile.json"
                path.write_text(json.dumps({"join_filter": snapshot(.025, .004, 0, bloom=1)}))
                con.execute("INSERT INTO profile_runs(run_id,variant_label,query_name,profile_path) VALUES (?,?,?,?)",
                            ["run", "prf uncompressed", "q.sql", str(path)])
            populate_profile_metrics(db, "run")
            with duckdb.connect(str(db)) as con:
                self.assertEqual(con.execute("SELECT join_finalize_build_ms,prf_analysis_ms,prf_compression_ms,"
                                             "prf_build_count,bloom_build_count FROM profile_runs").fetchone(),
                                 (25, 4, 0, 1, 1))

    def test_reports_and_overlays_include_bloom_and_keep_missing_values(self):
        profiles = [extract_join_filter_metrics({"join_filter": snapshot(.010, 0, 0, prf=0, bloom=1)}),
                    extract_join_filter_metrics({"join_filter": snapshot(.010, .002, 0)}),
                    extract_join_filter_metrics({"join_filter": snapshot(.010, 0, .003)}),
                    JoinFilterMetrics()]
        labels = ["sip", "prf uncompressed", "prf with fallback", "rpt prf"]
        rows = [MetricRow("fixture", "q.sql", label, 50, 100, 200, filter_phases=phases)
                for label, phases in zip(labels, profiles)]
        plans = [PlanMetricRow("fixture", "q.sql", label, 200, 100, 1) for label in labels]
        summary = summarize_benchmark_metrics(rows, plans, (("fixture", "Fixture", None),))
        variants = summary["benchmarks"][0]["variants"]
        self.assertEqual(variants["sip"]["join_finalize_build_ms"], 10)
        self.assertEqual(variants["sip"]["prf_compression_ms"], 0)
        self.assertEqual(variants["prf uncompressed"]["prf_analysis_ms"], 2)
        self.assertIsNone(variants["rpt prf"]["join_finalize_build_ms"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_filter_phase_report(root / "phases.csv", rows)
            with (root / "phases.csv").open() as stream:
                data = list(csv.DictReader(stream))
            self.assertEqual(data[0]["prf_analysis_ms"], "0.0")
            self.assertEqual(data[-1]["prf_analysis_ms"], "")
            _write_benchmark_summary_markdown(root / "summary.md", experiment_root=root,
                                              system_name="fixture", benchmark_summary=summary)
            markdown = (root / "summary.md").read_text()
            self.assertIn("Join/filter build | PRF analysis | PRF compression", markdown)
            self.assertIn("0.00 ms (1/1)", markdown)
            self.assertIn("initial finalization only", markdown)
        fig, ax = plt.subplots()
        try:
            entries = [{"series": [{"variant_label": row.variant_label, "median_ms": row.median_ms,
                                    **row.filter_phases.to_dict()} for row in rows]}]
            verification = _draw_filter_phase_overlays(ax, entries, tuple(labels), [-.3, -.1, .1, .3], .2, .1)
            self.assertEqual(verification, {"filter_phase_bars": 3, "filter_phase_overflows": 0})
            # Two more patches than build-only: separate analysis and compression.
            self.assertEqual(len(ax.patches), 5)
            entries[0]["series"][0]["median_ms"] = 1
            self.assertEqual(_draw_filter_phase_overlays(ax, entries, tuple(labels), [-.3, -.1, .1, .3], .2, .1)
                             ["filter_phase_overflows"], 1)
        finally:
            plt.close(fig)

    def test_summary_excludes_non_join_queries_and_floors_only_relative_counts(self):
        rows, plans = [], []
        # One variant eliminates a join; another query has zero work in both.
        fixtures = {
            "scan_only.sql": ((100, 1000, 0, 0), (200, 2000, 0, 0)),
            "eliminated_join.sql": ((.25, 100, 10, 1), (.125, 0, 0, 0)),
            "zero_work.sql": ((.5, 0, 0, 1), (.25, 0, 0, 1)),
            "positive_work.sql": ((1, 9, 4, 1), (.5, 1, 1, 1)),
        }
        for query, variants in fixtures.items():
            for label, (runtime, scanned, probes, joins) in zip(("sip", "rpt prf"), variants):
                rows.append(MetricRow(
                    "fixture", query, label, runtime, None, scanned,
                    filter_phases=JoinFilterMetrics(
                        join_finalize_build_ms=999 if query == "scan_only.sql" else 2,
                        prf_build_count=999 if query == "scan_only.sql" else 1,
                    ),
                ))
                plans.append(PlanMetricRow("fixture", query, label, scanned, probes, joins))
        summary = summarize_benchmark_metrics(rows, plans, (("fixture", "Fixture", None),))
        benchmark = summary["benchmarks"][0]
        self.assertEqual(benchmark["query_count"], 3)
        self.assertEqual(benchmark["join_free_query_count"], 1)
        self.assertEqual(benchmark["join_free_query_names"], ["scan_only.sql"])
        self.assertEqual(benchmark["relative_query_counts"],
                         {"runtime_ms": 3, "rows_scanned": 3, "join_probe_tuples": 3})
        self.assertEqual(benchmark["relative_excluded_query_counts"],
                         {"runtime_ms": 0, "rows_scanned": 0, "join_probe_tuples": 0})
        sip, rpt = (benchmark["variants"][label] for label in ("sip", "rpt prf"))
        self.assertEqual((sip["total_runtime_ms"], rpt["total_runtime_ms"]), (1.75, .875))
        self.assertEqual((sip["total_rows_scanned"], rpt["total_rows_scanned"]), (109, 1))
        self.assertEqual((sip["total_join_probe_tuples"], rpt["total_join_probe_tuples"]), (14, 1))
        self.assertAlmostEqual(sip["relative_runtime_geomean"], 2)
        self.assertAlmostEqual(sip["relative_rows_scanned_geomean"], 900 ** (1 / 3))
        self.assertAlmostEqual(sip["relative_join_probe_geomean"], 40 ** (1 / 3))
        for metric in ("relative_runtime_geomean", "relative_rows_scanned_geomean",
                       "relative_join_probe_geomean"):
            self.assertAlmostEqual(rpt[metric], 1)
        for variant in (sip, rpt):
            self.assertEqual(variant["join_finalize_build_ms"], 6)
            self.assertEqual(variant["join_finalize_build_profile_count"], 3)
            self.assertEqual(variant["prf_build_count"], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_benchmark_summary_markdown(path, experiment_root=Path(directory),
                                              system_name="fixture", benchmark_summary=summary)
            markdown = path.read_text()
            self.assertIn("3 join queries; relative time uses 3 (excluded 0)", markdown)
            self.assertIn("1 non-join queries excluded (scan_only.sql)", markdown)
            self.assertIn("replacing zero counts with one before normalization", markdown)
            self.assertIn("absolute totals retain the recorded zeros", markdown)
            self.assertIn("6.00 ms (3/3)", markdown)

    def test_summary_with_only_non_join_queries_has_no_relative_metrics(self):
        rows = [MetricRow("fixture", "scan.sql", "sip", 10, None, 100)]
        plans = [PlanMetricRow("fixture", "scan.sql", "sip", 100, 0, 0)]
        summary = summarize_benchmark_metrics(rows, plans, (("fixture", "Fixture", None),))
        benchmark = summary["benchmarks"][0]
        self.assertEqual(benchmark["query_count"], 0)
        self.assertIsNone(benchmark["variants"]["sip"]["relative_runtime_geomean"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_benchmark_summary_markdown(path, experiment_root=Path(directory),
                                              system_name="fixture", benchmark_summary=summary)
            self.assertIn("0 join queries", path.read_text())

    def test_latest_profile_is_selected_as_a_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "results.duckdb"
            ensure_results_schema(db)
            with duckdb.connect(str(db)) as con:
                con.execute("INSERT INTO timing_runs(experiment_name,workload_name,system_name,variant_label,"
                            "query_name,elapsed_ns) VALUES ('e','w','s','sip','q.sql',50000000)")
                for filename, date, build in (("z-old.json", "2020-01-01", .1), ("a-new.json", "2020-01-02", .01)):
                    path = root / filename
                    path.write_text(json.dumps({"join_filter": snapshot(build, 0, 0, prf=0, bloom=1)}))
                    con.execute("INSERT INTO profile_runs(run_id,experiment_name,workload_name,system_name,"
                                "variant_label,query_name,profile_path,profiled_at) VALUES (?,'e','w','s','sip','q.sql',?,?)",
                                [filename, str(path), date])
            with patch("paper.figures.data.load_experiment_config", return_value=SimpleNamespace(experiment_name="e")):
                rows = load_metric_rows(root, ("s",))
            self.assertEqual(rows[0].filter_phases.join_finalize_build_ms, 10)
            self.assertEqual(_filter_phase_rows(db, "e", "w", "s")[0]["join_finalize_build_ms"], 10)

    @unittest.skipUnless(BINARY.exists(), "build the vendored DuckDB to run the smoke test")
    def test_small_runner_collects_and_reports_all_filter_variants(self):
        common = ["SET disabled_optimizers='compressed_materialization,join_order,build_side_probe_side';",
                  "SET enable_perfect_hash_join_filter_pushdown=false;", "SET dynamic_or_filter_threshold=0;"]
        variants = []
        for label, prf, compression, bloom in (("disabled", False, False, False), ("sip", False, False, True),
                                              ("prf uncompressed", True, False, True),
                                              ("prf with fallback", True, True, True)):
            variants.append({"label": label, "setup_sql": common + [
                f"SET enable_prefix_range_filter={str(prf).lower()};",
                f"SET enable_prefix_range_filter_compression={str(compression).lower()};",
                f"SET enable_join_bloom_filter_pushdown={str(bloom).lower()};"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"experiment_name": "profile-smoke", "output_root": str(root / "out"),
                      "systems": [{"kind": "binary", "name": "duckdb", "binary_path": str(BINARY), "variants": variants}],
                      "workloads": [{"kind": "cross_product", "name": "profile-fixture", "params": {
                          "fixed": {"fact_rows": 5000, "dimension_cardinality": 1000, "key_domain_size": 100000000,
                                    "selectivity": .1, "join_count": 1, "build_key_multiplicity": 1,
                                    "probe_layout": "uniform", "row_group_size": 122880, "seed": 7},
                          "sweep": {"kind": "selectivity", "values": [.1]},
                          "query_types": [{"kind": "shoulder", "probe_layouts": ["uniform"]}]}}],
                      "benchmark": {"repetitions": 1, "threads": 2, "prepare_duckdb_binary": str(BINARY),
                                    "collect_profiles": True}}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            result = run_experiment(load_experiment_config(path))
            rows = load_metric_rows(result.root, ("duckdb",))
            by_variant = {row.variant_label: row.filter_phases for row in rows}
            self.assertEqual(len(rows), 4)
            for phases in by_variant.values():
                self.assertGreater(phases.join_finalize_build_ms, 0)
                self.assertAlmostEqual(phases.join_finalize_ms, phases.join_finalize_build_ms
                                       + phases.prf_analysis_ms + phases.prf_compression_ms)
            self.assertEqual(by_variant["sip"].bloom_build_count, 1)
            self.assertEqual(by_variant["sip"].prf_build_count, 0)
            self.assertEqual(by_variant["sip"].prf_analysis_ms, 0)
            self.assertEqual(by_variant["prf uncompressed"].prf_compression_ms, 0)
            self.assertGreater(by_variant["prf uncompressed"].prf_analysis_ms, 0)
            self.assertGreater(by_variant["prf with fallback"].prf_compression_ms, 0)
            reported = _filter_phase_rows(result.root / "results.duckdb", "profile-smoke", "profile-fixture", "duckdb")
            self.assertEqual(len(reported), 4)
            self.assertTrue(all(row["join_finalize_count"] == 1 for row in reported))
            analysis = analyze_experiment(result.root, baseline_variant=None)
            summary = json.loads(analysis.summary_json.read_text())
            self.assertEqual(len(summary["workloads"][0]["systems"][0]["filter_phases"]), 4)

    @unittest.skipUnless(BINARY.exists(), "build the vendored DuckDB to run the fallback smoke test")
    def test_actual_bloom_fallback_is_included_once(self):
        with tempfile.TemporaryDirectory() as directory:
            for compressed in (False, True):
                path = Path(directory) / f"{compressed}.json"
                sql = f"""
                    SET threads=4;
                    SET disabled_optimizers='compressed_materialization,join_order,build_side_probe_side';
                    SET enable_perfect_hash_join_filter_pushdown=false;
                    SET enable_prefix_range_filter=true;
                    SET enable_prefix_range_filter_compression={str(compressed).lower()};
                    SET enable_join_bloom_filter_pushdown=true;
                    CREATE TABLE b AS SELECT (i*2000000000)::BIGINT k FROM range(100000) t(i);
                    CREATE TABLE p AS SELECT k FROM b UNION ALL SELECT k+1 FROM b;
                    SET profiling_mode='standard'; SET enable_profiling='json';
                    SET profiling_output='{str(path).replace(chr(39), chr(39)*2)}';
                    SELECT count(*) FROM p JOIN b ON p.k=b.k AND p.k%7>=b.k%7;
                """
                subprocess.run([str(BINARY), "-batch", "-bail"], input=sql, text=True,
                               capture_output=True, check=True, timeout=60)
                phases = extract_join_filter_metrics(json.loads(path.read_text()))
                self.assertEqual((phases.prf_build_count, phases.bloom_build_count), (1, 1))
                self.assertAlmostEqual(phases.join_finalize_ms, phases.join_finalize_build_ms
                                       + phases.prf_analysis_ms + phases.prf_compression_ms)
                self.assertGreater(phases.prf_compression_ms if compressed else phases.prf_analysis_ms, 0)
                self.assertEqual(phases.prf_analysis_ms if compressed else phases.prf_compression_ms, 0)


if __name__ == "__main__":
    unittest.main()
