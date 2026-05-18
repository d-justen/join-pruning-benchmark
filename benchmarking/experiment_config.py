from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _require_string(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return raw


def _require_int(raw: object, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{label} must be an integer")
    return raw


def _require_bool(raw: object, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be a boolean")
    return raw


def _require_list(raw: object, label: str) -> list:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    return raw


def _require_dict(raw: object, label: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def _as_float(raw: object, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(raw)


@dataclass(frozen=True)
class VariantConfig:
    label: str
    setup_sql: tuple[str, ...]


@dataclass(frozen=True)
class BinarySystemConfig:
    kind: str
    name: str
    binary_path: Path
    variants: tuple[VariantConfig, ...]


@dataclass(frozen=True)
class GitSystemConfig:
    kind: str
    name: str
    repo: str
    ref: str
    variants: tuple[VariantConfig, ...]


SystemConfig = BinarySystemConfig | GitSystemConfig


@dataclass(frozen=True)
class SingleJoinClustersWorkloadConfig:
    kind: str
    name: str
    build_size: int
    domain_multiplier: int
    fact_rows: int
    cluster_counts: tuple[int, ...]
    dummy_string_cols: int
    dummy_string_len: int
    seed: int


@dataclass(frozen=True)
class CrossProductWorkloadConfig:
    kind: str
    name: str
    build_sizes: tuple[int, ...]
    selectivities: tuple[float, ...]
    predicate_types: tuple[str, ...]
    probe_clusterings: tuple[str, ...]
    fact_rows: int
    dummy_string_cols: int
    dummy_string_len: int
    seed: int


@dataclass(frozen=True)
class ExternalWorkloadConfig:
    kind: str
    name: str
    source_dir: Path


WorkloadConfig = (
    SingleJoinClustersWorkloadConfig | CrossProductWorkloadConfig | ExternalWorkloadConfig
)


@dataclass(frozen=True)
class BenchmarkOptions:
    repetitions: int
    threads: int | None
    prepare_duckdb_binary: str
    collect_profiles: bool


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    experiment_name: str
    output_root: Path
    overwrite_output: bool
    workloads: tuple[WorkloadConfig, ...]
    systems: tuple[SystemConfig, ...]
    benchmark: BenchmarkOptions

    @property
    def experiment_root(self) -> Path:
        return (self.output_root / self.experiment_name).resolve()


def _normalize_sql_statement(statement: str) -> str:
    normalized = statement.strip()
    if not normalized:
        raise ValueError("setup_sql statements cannot be empty")
    if not normalized.endswith(";"):
        normalized += ";"
    return normalized


def _parse_variants(raw: object, label: str) -> tuple[VariantConfig, ...]:
    variants_raw = _require_list(raw, label)
    if not variants_raw:
        raise ValueError(f"{label} must not be empty")
    variants: list[VariantConfig] = []
    seen_labels: set[str] = set()
    for index, variant_raw in enumerate(variants_raw, start=1):
        variant_obj = _require_dict(variant_raw, f"{label}[{index}]")
        variant_label = _require_string(variant_obj.get("label"), f"{label}[{index}].label")
        if variant_label in seen_labels:
            raise ValueError(f"duplicate variant label: {variant_label}")
        seen_labels.add(variant_label)
        setup_sql_raw = variant_obj.get("setup_sql", [])
        setup_sql_list = _require_list(setup_sql_raw, f"{label}[{index}].setup_sql")
        setup_sql = tuple(
            _normalize_sql_statement(_require_string(statement, "setup_sql entry"))
            for statement in setup_sql_list
        )
        variants.append(VariantConfig(label=variant_label, setup_sql=setup_sql))
    return tuple(variants)


def _parse_single_join_clusters(raw: dict, name: str) -> SingleJoinClustersWorkloadConfig:
    params = _require_dict(raw.get("params"), f"workloads[{name}].params")
    cluster_counts_raw = _require_list(params.get("cluster_counts"), "cluster_counts")
    cluster_counts = tuple(sorted({_require_int(value, "cluster_counts entry") for value in cluster_counts_raw}))
    if not cluster_counts:
        raise ValueError("cluster_counts must not be empty")
    return SingleJoinClustersWorkloadConfig(
        kind="single_join_clusters",
        name=name,
        build_size=_require_int(params.get("build_size"), "build_size"),
        domain_multiplier=_require_int(params.get("domain_multiplier"), "domain_multiplier"),
        fact_rows=_require_int(params.get("fact_rows"), "fact_rows"),
        cluster_counts=cluster_counts,
        dummy_string_cols=_require_int(params.get("dummy_string_cols", 2), "dummy_string_cols"),
        dummy_string_len=_require_int(params.get("dummy_string_len", 24), "dummy_string_len"),
        seed=_require_int(params.get("seed", 0), "seed"),
    )


def _parse_cross_product(raw: dict, name: str) -> CrossProductWorkloadConfig:
    params = _require_dict(raw.get("params"), f"workloads[{name}].params")
    build_sizes = tuple(_require_int(item, "build_sizes entry") for item in _require_list(params.get("build_sizes"), "build_sizes"))
    selectivities = tuple(_as_float(item, "selectivities entry") for item in _require_list(params.get("selectivities"), "selectivities"))
    predicate_types = tuple(_require_string(item, "predicate_types entry") for item in _require_list(params.get("predicate_types"), "predicate_types"))
    probe_clusterings = tuple(_require_string(item, "probe_clusterings entry") for item in _require_list(params.get("probe_clusterings"), "probe_clusterings"))
    return CrossProductWorkloadConfig(
        kind="cross_product",
        name=name,
        build_sizes=build_sizes,
        selectivities=selectivities,
        predicate_types=predicate_types,
        probe_clusterings=probe_clusterings,
        fact_rows=_require_int(params.get("fact_rows"), "fact_rows"),
        dummy_string_cols=_require_int(params.get("dummy_string_cols", 2), "dummy_string_cols"),
        dummy_string_len=_require_int(params.get("dummy_string_len", 24), "dummy_string_len"),
        seed=_require_int(params.get("seed", 0), "seed"),
    )


def _parse_external(raw: dict, name: str, base_dir: Path) -> ExternalWorkloadConfig:
    return ExternalWorkloadConfig(
        kind="external",
        name=name,
        source_dir=(base_dir / _require_string(raw.get("source_dir"), "source_dir")).resolve(),
    )


def _parse_workloads(raw: object, base_dir: Path) -> tuple[WorkloadConfig, ...]:
    workloads_raw = _require_list(raw, "workloads")
    if not workloads_raw:
        raise ValueError("workloads must not be empty")
    workloads: list[WorkloadConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(workloads_raw, start=1):
        workload_obj = _require_dict(item, f"workloads[{index}]")
        name = _require_string(workload_obj.get("name"), f"workloads[{index}].name")
        kind = _require_string(workload_obj.get("kind"), f"workloads[{index}].kind")
        if name in seen_names:
            raise ValueError(f"duplicate workload name: {name}")
        seen_names.add(name)
        if kind == "single_join_clusters":
            workloads.append(_parse_single_join_clusters(workload_obj, name))
        elif kind == "cross_product":
            workloads.append(_parse_cross_product(workload_obj, name))
        elif kind == "external":
            workloads.append(_parse_external(workload_obj, name, base_dir))
        else:
            raise ValueError(f"unsupported workload kind: {kind}")
    return tuple(workloads)


def _parse_systems(raw: object, base_dir: Path) -> tuple[SystemConfig, ...]:
    systems_raw = _require_list(raw, "systems")
    if not systems_raw:
        raise ValueError("systems must not be empty")
    systems: list[SystemConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(systems_raw, start=1):
        system_obj = _require_dict(item, f"systems[{index}]")
        name = _require_string(system_obj.get("name"), f"systems[{index}].name")
        kind = _require_string(system_obj.get("kind"), f"systems[{index}].kind")
        if name in seen_names:
            raise ValueError(f"duplicate system name: {name}")
        seen_names.add(name)
        variants = _parse_variants(system_obj.get("variants"), f"systems[{index}].variants")
        if kind == "binary":
            binary_raw = _require_string(system_obj.get("binary_path"), "binary_path")
            binary_path = Path(binary_raw)
            systems.append(
                BinarySystemConfig(
                    kind="binary",
                    name=name,
                    binary_path=(
                        binary_path.resolve()
                        if binary_path.is_absolute() or "/" in binary_raw
                        else binary_path
                    ),
                    variants=variants,
                )
            )
        elif kind == "git":
            systems.append(
                GitSystemConfig(
                    kind="git",
                    name=name,
                    repo=_require_string(system_obj.get("repo"), "repo"),
                    ref=_require_string(system_obj.get("ref"), "ref"),
                    variants=variants,
                )
            )
        else:
            raise ValueError(f"unsupported system kind: {kind}")
    return tuple(systems)


def _parse_benchmark(raw: object) -> BenchmarkOptions:
    benchmark_obj = _require_dict(raw, "benchmark")
    threads_raw = benchmark_obj.get("threads")
    threads = None if threads_raw is None else _require_int(threads_raw, "threads")
    return BenchmarkOptions(
        repetitions=_require_int(benchmark_obj.get("repetitions", 5), "repetitions"),
        threads=threads,
        prepare_duckdb_binary=_require_string(
            benchmark_obj.get("prepare_duckdb_binary", "duckdb"),
            "prepare_duckdb_binary",
        ),
        collect_profiles=_require_bool(
            benchmark_obj.get("collect_profiles", True),
            "collect_profiles",
        ),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = path.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("top-level config must be an object")
    base_dir = config_path.parent
    output_root_raw = _require_string(raw.get("output_root", "out/experiments"), "output_root")
    output_root = (
        (base_dir / output_root_raw).resolve()
        if not Path(output_root_raw).is_absolute()
        else Path(output_root_raw).resolve()
    )
    config = ExperimentConfig(
        config_path=config_path,
        experiment_name=_require_string(raw.get("experiment_name"), "experiment_name"),
        output_root=output_root,
        overwrite_output=_require_bool(raw.get("overwrite_output", False), "overwrite_output"),
        workloads=_parse_workloads(raw.get("workloads"), base_dir),
        systems=_parse_systems(raw.get("systems"), base_dir),
        benchmark=_parse_benchmark(raw.get("benchmark", {})),
    )
    if config.benchmark.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if config.benchmark.threads is not None and config.benchmark.threads <= 0:
        raise ValueError("threads must be positive when set")
    return config
