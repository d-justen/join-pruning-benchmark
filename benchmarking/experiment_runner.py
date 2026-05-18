from __future__ import annotations

import shutil
from pathlib import Path

from benchmarking.artifact_layout import ExperimentPaths, prepare_experiment_root
from benchmarking.duckdb_build import resolve_system
from benchmarking.duckdb_runner import benchmark_workload, summarize_results, write_summary
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.manifests import PreparedWorkload
from benchmarking.workloads import (
    prepare_cross_product_workload,
    prepare_external_workload,
    prepare_single_join_clusters_workload,
)


def prepare_workloads(config: ExperimentConfig, paths: ExperimentPaths) -> list[PreparedWorkload]:
    prepared: list[PreparedWorkload] = []
    for workload in config.workloads:
        artifact_dir = paths.workload_dir(workload.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if workload.kind == "single_join_clusters":
            manifest = prepare_single_join_clusters_workload(workload, artifact_dir)
        elif workload.kind == "cross_product":
            manifest = prepare_cross_product_workload(workload, artifact_dir)
        else:
            manifest = prepare_external_workload(
                workload,
                artifact_dir,
                prepare_duckdb_binary=config.benchmark.prepare_duckdb_binary,
            )
        prepared.append(manifest)
    return prepared


def copy_config(config_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination)


def run_experiment(config: ExperimentConfig) -> ExperimentPaths:
    paths = prepare_experiment_root(
        config.experiment_root,
        overwrite=config.overwrite_output,
    )
    copy_config(config.config_path, paths.copied_config_path)
    workloads = prepare_workloads(config, paths)
    for system_config in config.systems:
        resolved_system = resolve_system(system_config, paths.system_dir(system_config.name))
        for workload in workloads:
            benchmark_workload(
                config.experiment_name,
                workload,
                resolved_system,
                system_config.variants,
                config.benchmark,
                paths.results_db,
                paths.sql_dir,
                paths.profiles_dir,
            )
    summary = summarize_results(paths.results_db, config.experiment_name)
    write_summary(paths.summary_json, summary)
    return paths
