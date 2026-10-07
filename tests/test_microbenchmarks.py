import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarking import load_experiment_config
from benchmarking.experiment_config import MicrobenchmarkConfig, parse_microbenchmark_config
from benchmarking.experiment_runner import run_experiment
from benchmarking.microbenchmarks import (
    ARTIFACT_SUBDIR, load_microbenchmark_manifest, run_microbenchmarks, sha256, validate_results,
)
from paper.figures.build_all import build_figures
from paper.figures.filter_benchmark import load_filter_benchmark_rows, build_false_positive_rate_series
from paper.figures.range_benchmark import load_range_benchmark_rows, build_range_throughput_series


CONFIG = MicrobenchmarkConfig(system_name="duckdb-paper", domain_size=1000, total_keys=100,
                              min_clusters=2, max_clusters=4, probe_count=100,
                              queries_per_cluster_count=1000, repetitions=2)
AVAILABLE = {"point": ["bloom", "prf_uncompressed", "prf"], "range": ["prf_uncompressed", "prf"]}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    directory = root / ARTIFACT_SUBDIR
    directory.mkdir(parents=True)
    points, ranges = [], []
    for cluster in (2, 4):
        for rep in (0, 1):
            for name in AVAILABLE["point"]:
                negatives = 90 - rep * 10
                points.append(dict(filter=name, cluster_count=cluster, repetition=rep, seed=42 + rep,
                                   probe_count=100, actual_negatives=negatives, false_positives=10,
                                   actual_fpr=10 / negatives, build_time_ns=10000,
                                   post_processing_time_ns=1000, summary_bytes=200,
                                   probe_throughput_per_sec=1000000))
            for mode in ("compressed", "uncompressed"):
                for kind in ("empty", "full", "mixed_bridge", "mixed_domain"):
                    ns = (1000000 if rep == 0 else 250000) if kind == "empty" else 10000
                    ranges.append(dict(summary="prf", mode=mode, query_kind=kind, cluster_count=cluster,
                                       repetition=rep, seed=42 + rep, query_count=1000,
                                       total_query_time_ns=ns, queries_per_sec=1000 * 1e9 / ns,
                                       possibly_overlapping=0 if kind == "empty" else 1000,
                                       max_false_positive_rate=0.001, correct_result="true",
                                       summary_bytes=200, post_processing_time_ns=1000,
                                       min_range_width=10, max_range_width=10))
    write_csv(directory / "point.csv", points)
    write_csv(directory / "range.csv", ranges)
    manifest = {"schema_version": 1, "status": "complete", "config": asdict(CONFIG),
                "available_filters": AVAILABLE, "unavailable_filters": {"point": ["grafite", "diva"],
                                                                         "range": ["grafite", "diva"]},
                "outputs": {kind: {"sha256": sha256(directory / f"{kind}.csv")} for kind in ("point", "range")}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, points, ranges


def experiment_json(root):
    return {"experiment_name": "experiment", "output_root": str(root), "overwrite_output": True,
            "workloads": [], "systems": [{"kind": "binary", "name": "duckdb-paper",
                "binary_path": "/unused/duckdb", "variants": [{"label": "prf with fallback", "setup_sql": []}]}],
            "microbenchmarks": asdict(CONFIG)}


class MicrobenchmarkTests(unittest.TestCase):
    def test_config_rejects_invalid_experiments(self):
        for changes in ({"max_clusters": 3}, {"total_keys": 1000}, {"max_clusters": 128},
                        {"repetitions": 0}, {"max_fpr": float("nan")}, {"seed": -1},
                        {"seed": 2**64 - 1}, {"required_filters": ["unknown"]},
                        {"range_query_kind": "full"}, {"aggregate": "minimum"}, {"typo": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_microbenchmark_config({**asdict(CONFIG), **changes})

    def test_measured_fpr_and_range_kind_are_preserved(self):
        with TemporaryDirectory() as tmp:
            directory, points, _ = fixture(Path(tmp))
            # The two repetitions have different measured denominators.
            series = build_false_positive_rate_series(load_filter_benchmark_rows(directory / "point.csv"),
                                                       aggregate="mean")
            self.assertAlmostEqual(series["prf"][0][1], (10 / 90 + 10 / 80) / 2)
            ranges = load_range_benchmark_rows(directory / "range.csv")
            throughput = build_range_throughput_series(ranges, aggregate="mean")
            self.assertEqual(throughput["prf"][0][1], 2500000)
            self.assertEqual(build_range_throughput_series(ranges, aggregate="mean", query_kind="full")
                             ["prf"][0][1], 100000000)
            # Legacy point CSVs still plot measured FPR without fabricating TN counts.
            for row in points:
                del row["actual_negatives"]
            write_csv(directory / "point.csv", points)
            self.assertIsNone(load_filter_benchmark_rows(directory / "point.csv")[0].actual_negatives)

    def test_incomplete_duplicate_or_incorrect_measurements_are_rejected(self):
        with TemporaryDirectory() as tmp:
            directory, points, ranges = fixture(Path(tmp))
            self.assertEqual(validate_results(directory, CONFIG, AVAILABLE), {"point_rows": 12, "range_rows": 32})
            for bad in (points[:-1], points + [points[0]], [{**points[0], "actual_fpr": 0.5}, *points[1:]],
                        [{**points[0], "probe_throughput_per_sec": "nan"}, *points[1:]]):
                write_csv(directory / "point.csv", bad)
                with self.assertRaises(ValueError):
                    validate_results(directory, CONFIG, AVAILABLE)
            write_csv(directory / "point.csv", points)
            bad_ranges = [dict(row) for row in ranges]
            bad_ranges[1]["possibly_overlapping"] = 999
            write_csv(directory / "range.csv", bad_ranges)
            with self.assertRaisesRegex(ValueError, "false negative"):
                validate_results(directory, CONFIG, AVAILABLE)

    def test_regeneration_uses_only_saved_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, _, _ = fixture(root)
            (root / "config.json").write_text(json.dumps(experiment_json(root.parent)))
            with patch("subprocess.run", side_effect=AssertionError("must not run a compiler or benchmark")):
                output = build_figures(root, None)
            self.assertTrue((output / "figures/filter_benchmark_overview.pdf").is_file())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["verification"]["filter_benchmark_overview"]["panel_order"],
                             ["point_throughput", "range_throughput", "summary_size", "build_time", "point_fpr"])
            with (output / "filter_benchmark_series.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["query_kind"] for row in rows if row["metric"] == "range_throughput"},
                             {"empty", "full", "mixed_bridge", "mixed_domain"})
            self.assertEqual({row["sample_count"] for row in rows}, {"2"})
            # A changed raw file cannot silently be treated as a completed run.
            with (directory / "point.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_microbenchmark_manifest(root)

    def test_microbenchmark_only_preserves_sql_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            raw = experiment_json(root)
            source.write_text(json.dumps(raw))
            config = load_experiment_config(source)
            config.experiment_root.mkdir()
            (config.experiment_root / "config.json").write_text(json.dumps(raw))
            preserved = {"results.duckdb": b"existing SQL measurements", "profiles/query.json": b"existing profile",
                         "summary.json": b"existing summary", "workloads/data": b"existing data"}
            for name, content in preserved.items():
                path = config.experiment_root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(content)
            with patch("benchmarking.experiment_runner.resolve_system", return_value=SimpleNamespace(name="duckdb-paper")), \
                 patch("benchmarking.experiment_runner.run_microbenchmarks") as run, \
                 patch("benchmarking.experiment_runner.prepare_workloads") as prepare:
                run_experiment(config, only_microbenchmarks=True)
                run.assert_called_once()
                prepare.assert_not_called()
            for name, content in preserved.items():
                self.assertEqual((config.experiment_root / name).read_bytes(), content)
            raw["benchmark"] = {"threads": 4}
            source.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "SQL configuration differs"):
                run_experiment(load_experiment_config(source), only_microbenchmarks=True)

    def test_required_competitors_fail_before_measurement(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "binary"
            binary.write_text("dummy")
            config = replace(CONFIG, required_filters=("diva",))
            with patch("benchmarking.microbenchmarks.resolve_microbenchmark_binaries",
                       return_value={"point": binary, "range": binary}), \
                 patch("benchmarking.microbenchmarks._source_metadata", return_value={}), \
                 patch("benchmarking.microbenchmarks._available_filters", side_effect=[AVAILABLE["point"], AVAILABLE["range"]]), \
                 patch("platform.processor", return_value="test CPU"), \
                 patch("platform.platform", return_value="test OS"), \
                 patch("subprocess.run") as process:
                with self.assertRaisesRegex(ValueError, "lacks required filters: diva"):
                    run_microbenchmarks(config, SimpleNamespace(), root)
                process.assert_not_called()
            manifest = json.loads((root / ARTIFACT_SUBDIR / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_missing_binary_invalidates_a_previous_complete_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with patch("benchmarking.microbenchmarks.resolve_microbenchmark_binaries",
                       side_effect=FileNotFoundError("missing tool")):
                with self.assertRaises(FileNotFoundError):
                    run_microbenchmarks(CONFIG, SimpleNamespace(), root)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_microbenchmark_manifest(root)


if __name__ == "__main__":
    unittest.main()
