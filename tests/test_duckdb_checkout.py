from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from benchmarking.duckdb_build import (
    DuckDBCheckoutOptions,
    ensure_duckdb_checkout,
    resolve_built_duckdb_binary,
    resolve_existing_system,
)
from benchmarking.experiment_config import GitSystemConfig, load_experiment_config


def _make_fake_duckdb_tree(root: Path) -> Path:
    (root / "src" / "include" / "duckdb").mkdir(parents=True)
    (root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    build_dir = root / "build" / "release"
    build_dir.mkdir(parents=True)
    binary_path = build_dir / "duckdb"
    binary_path.write_text("#!/bin/sh\necho 'v1.5.1-dev1234 deadbeefcafebabe'\n", encoding="utf-8")
    binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR)
    return root


class DuckDBCheckoutTest(unittest.TestCase):
    def test_preferred_build_directory_wins_over_release_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkout_dir = _make_fake_duckdb_tree(Path(temporary_dir) / "duckdb")
            native_build_dir = checkout_dir / "build" / "paper-native-release"
            native_build_dir.mkdir(parents=True)
            native_binary = native_build_dir / "duckdb"
            native_binary.write_text("native", encoding="utf-8")

            resolved = resolve_built_duckdb_binary(
                checkout_dir, native_build_dir
            )

            self.assertEqual(resolved, native_binary.resolve())

    def test_git_system_accepts_local_checkout_and_workload_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkout_dir = _make_fake_duckdb_tree(root / "rpt")
            external_dir = root / "external"
            (external_dir / "queries").mkdir(parents=True)
            (external_dir / "source.json").write_text(
                json.dumps({"benchmark_name": "external", "queries_dir": "queries"}),
                encoding="utf-8",
            )
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "test",
                        "output_root": "out",
                        "overwrite_output": True,
                        "workloads": [
                            {
                                "name": "external",
                                "kind": "external",
                                "source_dir": "external",
                            }
                        ],
                        "systems": [
                            {
                                "name": "rpt",
                                "kind": "git",
                                "checkout_dir": "rpt",
                                "workload_names": ["external"],
                                "variants": [{"label": "rpt prf", "setup_sql": []}],
                            }
                        ],
                        "benchmark": {},
                    }
                ),
                encoding="utf-8",
            )

            system = load_experiment_config(config_path).systems[0]

            self.assertEqual(system.checkout_dir, checkout_dir.resolve())
            self.assertEqual(system.workload_names, ("external",))

    def test_system_rejects_unknown_workload_subset_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            external_dir = root / "external"
            (external_dir / "queries").mkdir(parents=True)
            (external_dir / "source.json").write_text(
                json.dumps({"benchmark_name": "external", "queries_dir": "queries"}),
                encoding="utf-8",
            )
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "test",
                        "output_root": "out",
                        "overwrite_output": True,
                        "workloads": [
                            {
                                "name": "external",
                                "kind": "external",
                                "source_dir": "external",
                            }
                        ],
                        "systems": [
                            {
                                "name": "rpt",
                                "kind": "git",
                                "workload_names": ["missing"],
                                "variants": [{"label": "rpt prf", "setup_sql": []}],
                            }
                        ],
                        "benchmark": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown workloads: missing"):
                load_experiment_config(config_path)

    def test_config_parser_rejects_repo_and_ref_for_git_systems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "test",
                        "output_root": "out",
                        "overwrite_output": True,
                        "workloads": [
                            {
                                "name": "external",
                                "kind": "external",
                                "source_dir": "external",
                            }
                        ],
                        "systems": [
                            {
                                "name": "duckdb",
                                "kind": "git",
                                "repo": "https://github.com/example/duckdb.git",
                                "ref": "feature",
                                "variants": [{"label": "default", "setup_sql": []}],
                            }
                        ],
                        "benchmark": {},
                    }
                ),
                encoding="utf-8",
            )
            (root / "external").mkdir()
            (root / "external" / "source.json").write_text(
                json.dumps({"benchmark_name": "external", "queries_dir": "queries"}),
                encoding="utf-8",
            )
            (root / "external" / "queries").mkdir()

            with self.assertRaisesRegex(ValueError, "no longer accept repo/ref"):
                load_experiment_config(config_path)

    def test_existing_anonymized_checkout_is_used_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkout_dir = _make_fake_duckdb_tree(Path(temporary_dir) / "duckdb")

            resolved_dir = ensure_duckdb_checkout(
                DuckDBCheckoutOptions(checkout_dir=checkout_dir, github_branch=None)
            )

            self.assertEqual(resolved_dir, checkout_dir.resolve())

    def test_missing_checkout_requires_runtime_branch_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkout_dir = Path(temporary_dir) / "duckdb"

            with self.assertRaisesRegex(RuntimeError, "Populate it manually or pass --duckdb-github-branch"):
                ensure_duckdb_checkout(DuckDBCheckoutOptions(checkout_dir=checkout_dir))

    def test_resolved_git_system_redacts_commit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkout_dir = _make_fake_duckdb_tree(Path(temporary_dir) / "duckdb")
            config = GitSystemConfig(
                kind="git",
                name="duckdb",
                c_compiler=None,
                cxx_compiler=None,
                variants=(),
            )

            resolved = resolve_existing_system(
                config,
                Path(temporary_dir) / "unused-system-dir",
                DuckDBCheckoutOptions(checkout_dir=checkout_dir),
            )

            self.assertEqual(resolved.version, "v1.5.1-dev1234")
            self.assertIsNone(resolved.git_commit)
            self.assertIsNone(resolved.repo)
            self.assertIsNone(resolved.ref)


if __name__ == "__main__":
    unittest.main()
