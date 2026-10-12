from __future__ import annotations

import json
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
from benchmarking.manifests import PreparedWorkload, load_manifest
from benchmarking.microbenchmarks import run_microbenchmarks
from benchmarking.jpb_microbenchmarks import run_jpb_microbenchmarks
from benchmarking.workloads import (
    prepare_cross_product_workload,
    prepare_external_workload,
    prepare_single_join_clusters_workload,
    prepare_string_joins_workload,
)


def prepare_workloads(
    config: ExperimentConfig,
    paths: ExperimentPaths,
    *,
    resume_preparation: bool = False,
) -> list[PreparedWorkload]:
    prepared: list[PreparedWorkload] = []
    for workload in config.workloads:
        artifact_dir = paths.workload_dir(workload.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "manifest.json"
        if resume_preparation and manifest_path.exists():
            manifest = load_manifest(manifest_path)
            data_path = manifest.db_path
            if manifest.storage_mode == "remote_http_parquet":
                data_path = manifest.remote_parquet_catalog_path
            elif manifest.storage_mode == "ducklake":
                data_path = manifest.parquet_catalog_path
            if data_path is None or not data_path.is_file() or not manifest.queries_dir.is_dir():
                raise ValueError(f"prepared workload is missing its data or queries: {workload.name}")
            prepared.append(manifest)
            continue
        if workload.kind == "single_join_clusters":
            manifest = prepare_single_join_clusters_workload(workload, artifact_dir)
        elif workload.kind == "cross_product":
            resume_density = (resume_preparation and workload.experiment_definition is not None
                              and workload.experiment_definition.sweep.cardinality_mode
                              == "fixed_selected_build_and_probe")
            manifest = prepare_cross_product_workload(workload, artifact_dir, resume_density=resume_density)
        elif workload.kind == "string_joins":
            manifest = prepare_string_joins_workload(workload, artifact_dir)
        else:
            manifest = prepare_external_workload(workload, artifact_dir)
        prepared.append(manifest)
    return prepared


def copy_config(config_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination)


def run_experiment(config: ExperimentConfig, *, resume_preparation: bool = False,
                   only_microbenchmarks: bool = False) -> ExperimentPaths:
    return run_experiment_with_options(config, DuckDBCheckoutOptions(), resume_preparation=resume_preparation,
                                       only_microbenchmarks=only_microbenchmarks)


def _validate_microbenchmark_only(config: ExperimentConfig) -> None:
    if config.microbenchmarks is None and config.jpb_microbenchmarks is None:
        raise ValueError("--only-microbenchmarks requires a microbenchmarks configuration")
    saved = ExperimentPaths(config.experiment_root).copied_config_path
    if not saved.exists():
        if config.experiment_root.exists() and any(config.experiment_root.iterdir()):
            raise ValueError("existing experiment directory has no config.json")
        return
    def sql_config(path: Path) -> dict:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("microbenchmarks", None)
        data.pop("jpb_microbenchmarks", None)
        data.pop("overwrite_output", None)
        return data
    if sql_config(saved) != sql_config(config.config_path):
        raise ValueError("SQL configuration differs from the saved experiment; use a new experiment_name")


def _validate_preparation_resume(config: ExperimentConfig) -> None:
    """Allow resource changes, but never mix definitions or repeat measured runs."""
    paths = ExperimentPaths(config.experiment_root)
    if not paths.copied_config_path.is_file():
        raise ValueError("--resume-preparation requires an existing experiment config.json")
    if paths.results_db.exists() or paths.summary_json.exists():
        raise ValueError("--resume-preparation is only supported before benchmarking has started")

    def generation_config(path: Path) -> dict:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("overwrite_output", None)
        for workload in data.get("workloads", []):
            workload.get("params", {}).pop("preparation", None)
        return data

    if generation_config(config.config_path) != generation_config(paths.copied_config_path):
        raise ValueError("saved experiment config differs; only preparation resources may change on resume")


def run_experiment_with_options(
    config: ExperimentConfig,
    checkout_options: DuckDBCheckoutOptions,
    *,
    resume_preparation: bool = False,
    only_microbenchmarks: bool = False,
) -> ExperimentPaths:
    if resume_preparation and only_microbenchmarks:
        raise ValueError("--resume-preparation and --only-microbenchmarks are mutually exclusive")
    if only_microbenchmarks:
        _validate_microbenchmark_only(config)
    elif resume_preparation:
        _validate_preparation_resume(config)
    elif config.experiment_root.exists() and not config.overwrite_output:
        raise FileExistsError(f"experiment already exists: {config.experiment_root}; "
                              "use --resume-preparation to continue failed preparation")
    paths = prepare_experiment_root(
        config.experiment_root,
        overwrite=config.overwrite_output and not resume_preparation and not only_microbenchmarks,
    )
    copy_config(config.config_path, paths.copied_config_path)
    def selected_workload_configs(system_config):
        if system_config.workload_names is None:
            return config.workloads
        selected = set(system_config.workload_names)
        return tuple(workload for workload in config.workloads if workload.name in selected)

    resolved_systems = []
    micro_system_names = {suite.system_name for suite in (config.microbenchmarks, config.jpb_microbenchmarks)
                          if suite is not None}
    for system_config in config.systems:
        if only_microbenchmarks and system_config.name not in micro_system_names:
            continue
        selected = selected_workload_configs(system_config)
        if only_microbenchmarks:
            selected = ()
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
    if config.microbenchmarks is not None:
        micro_system = next(system for system in resolved_systems if system.name == config.microbenchmarks.system_name)
        run_microbenchmarks(config.microbenchmarks, micro_system, paths.root, reuse_completed=resume_preparation)
    if config.jpb_microbenchmarks is not None:
        micro_system = next(system for system in resolved_systems if system.name == config.jpb_microbenchmarks.system_name)
        run_jpb_microbenchmarks(config, micro_system, paths.root, reuse_completed=resume_preparation)
    if only_microbenchmarks or not config.workloads:
        return paths
    workloads = prepare_workloads(
        config,
        paths,
        resume_preparation=resume_preparation,
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
