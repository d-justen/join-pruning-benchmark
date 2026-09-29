from __future__ import annotations

import shutil
from pathlib import Path

from benchmarking.artifact_layout import ExperimentPaths, prepare_experiment_root
from benchmarking.duckdb_build import DuckDBCheckoutOptions, ResolvedSystem, resolve_system
from benchmarking.duckdb_runner import (
    benchmark_workload,
    bootstrap_ducklake_workload,
    start_remote_http_service,
    stop_remote_http_service,
    summarize_results,
    write_summary,
)
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.manifests import PreparedWorkload
from benchmarking.workloads import (
    prepare_cross_product_workload,
    prepare_external_workload,
    prepare_single_join_clusters_workload,
    prepare_string_joins_workload,
)


def prepare_workloads(
    config: ExperimentConfig,
    paths: ExperimentPaths,
) -> list[PreparedWorkload]:
    prepared: list[PreparedWorkload] = []
    for workload in config.workloads:
        artifact_dir = paths.workload_dir(workload.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if workload.kind == "single_join_clusters":
            manifest = prepare_single_join_clusters_workload(workload, artifact_dir)
        elif workload.kind == "cross_product":
            manifest = prepare_cross_product_workload(workload, artifact_dir)
        elif workload.kind == "string_joins":
            manifest = prepare_string_joins_workload(workload, artifact_dir)
        else:
            manifest = prepare_external_workload(workload, artifact_dir)
        prepared.append(manifest)
    return prepared


def copy_config(config_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination)


def run_experiment(config: ExperimentConfig) -> ExperimentPaths:
    return run_experiment_with_options(config, DuckDBCheckoutOptions())


def run_experiment_with_options(
    config: ExperimentConfig,
    checkout_options: DuckDBCheckoutOptions,
) -> ExperimentPaths:
    paths = prepare_experiment_root(
        config.experiment_root,
        overwrite=config.overwrite_output,
    )
    copy_config(config.config_path, paths.copied_config_path)
    def selected_workload_configs(system_config):
        if system_config.workload_names is None:
            return config.workloads
        selected = set(system_config.workload_names)
        return tuple(workload for workload in config.workloads if workload.name in selected)

    resolved_systems = []
    for system_config in config.systems:
        selected = selected_workload_configs(system_config)
        required_extensions: set[str] = set()
        if any(getattr(workload, "storage_kind", "duckdb") == "ducklake" for workload in selected):
            required_extensions.add("ducklake")
        if any(
            getattr(workload, "storage_kind", "duckdb") == "ducklake"
            and getattr(workload, "ducklake_http_latency_ms", None) is not None
            for workload in selected
        ):
            required_extensions.add("httpfs")
        resolved_systems.append(
            resolve_system(
                system_config,
                paths.system_dir(system_config.name),
                checkout_options,
                frozenset(required_extensions),
            )
        )
    workloads = prepare_workloads(
        config,
        paths,
    )
    remote_services = {}
    try:
        for workload in workloads:
            remote_services[workload.name] = start_remote_http_service(workload)
        for workload in workloads:
            bootstrap_ducklake_workload(
                resolved_systems[0].binary_path,
                workload,
                remote_services[workload.name],
            )
        for system_config, resolved_system in zip(config.systems, resolved_systems):
            selected_names = (
                None
                if system_config.workload_names is None
                else set(system_config.workload_names)
            )
            for workload in workloads:
                if selected_names is not None and workload.name not in selected_names:
                    continue
                benchmark_workload(
                    config.experiment_name,
                    workload,
                    resolved_system,
                    system_config.variants,
                    config.benchmark,
                    paths.results_db,
                    paths.sql_dir,
                    paths.profiles_dir,
                    remote_services[workload.name],
                )
    finally:
        for remote_service in remote_services.values():
            stop_remote_http_service(remote_service)
    summary = summarize_results(paths.results_db, config.experiment_name)
    write_summary(paths.summary_json, summary)
    return paths
