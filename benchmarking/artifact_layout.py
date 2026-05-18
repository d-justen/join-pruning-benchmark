from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def workloads_dir(self) -> Path:
        return self.root / "workloads"

    @property
    def systems_dir(self) -> Path:
        return self.root / "systems"

    @property
    def sql_dir(self) -> Path:
        return self.root / "sql"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def results_db(self) -> Path:
        return self.root / "results.duckdb"

    @property
    def summary_json(self) -> Path:
        return self.root / "summary.json"

    @property
    def copied_config_path(self) -> Path:
        return self.root / "config.json"

    def workload_dir(self, workload_name: str) -> Path:
        return self.workloads_dir / workload_name

    def system_dir(self, system_name: str) -> Path:
        return self.systems_dir / system_name


def prepare_experiment_root(root: Path, *, overwrite: bool) -> ExperimentPaths:
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = ExperimentPaths(root=root.resolve())
    paths.workloads_dir.mkdir(parents=True, exist_ok=True)
    paths.systems_dir.mkdir(parents=True, exist_ok=True)
    paths.sql_dir.mkdir(parents=True, exist_ok=True)
    paths.profiles_dir.mkdir(parents=True, exist_ok=True)
    return paths
