from __future__ import annotations

import json
import math
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
    workload_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GitSystemConfig:
    kind: str
    name: str
    c_compiler: str | None
    cxx_compiler: str | None
    variants: tuple[VariantConfig, ...]
    checkout_dir: Path | None = None
    workload_names: tuple[str, ...] | None = None


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
    remote_http_parquet_latency_ms: int | None


@dataclass(frozen=True)
class CrossProductQueryTypeConfig:
    kind: str
    cluster_count: int | None
    probe_layouts: tuple[str, ...]


@dataclass(frozen=True)
class CrossProductSweepConfig:
    kind: str
    values: tuple[int | float, ...]
    cardinality_mode: str | None = None


@dataclass(frozen=True)
class CrossProductFixedParameters:
    fact_rows: int
    dimension_cardinality: int
    key_domain_size: int | str | None
    selectivity: float
    join_count: int
    build_key_multiplicity: int
    probe_layout: str
    row_group_size: int
    dummy_string_cols: int
    dummy_string_len: int
    seed: int


@dataclass(frozen=True)
class CrossProductExperimentDefinition:
    fixed: CrossProductFixedParameters
    sweep: CrossProductSweepConfig
    query_types: tuple[CrossProductQueryTypeConfig, ...]


@dataclass(frozen=True)
class DensityPreparationOptions:
    # Eight native row groups; bounds fact-side ASOF sorts independently of N.
    batch_rows: int = 983_040
    threads: int = 4
    memory_limit: str | None = None
    max_temp_directory_size: str | None = None
    temp_directory: str | None = None


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
    remote_http_parquet_latency_ms: int | None
    storage_kind: str = "duckdb"
    parquet_file_size_mb: int = 200
    ducklake_http_latency_ms: int | None = None
    experiment_definition: CrossProductExperimentDefinition | None = None
    post_variant_setup_sql: tuple[str, ...] = ()
    preparation: DensityPreparationOptions = DensityPreparationOptions()


@dataclass(frozen=True)
class StringJoinsWorkloadConfig:
    kind: str
    name: str
    airport_csv_path: Path
    uuid_dim_size: int
    uuid_timestamp_start_ms: int
    uuid_timestamp_step_ms: int
    uuid_selected_ranges: tuple[tuple[int, int], ...]
    fact_rows: int
    dummy_string_cols: int
    dummy_string_len: int
    seed: int
    remote_http_parquet_latency_ms: int | None


@dataclass(frozen=True)
class ExternalWorkloadConfig:
    kind: str
    name: str
    source_dir: Path
    query_names: tuple[str, ...] | None = None
    setup_sql: tuple[str, ...] = ()
    storage_kind: str = "duckdb"
    parquet_file_size_mb: int = 200
    ducklake_http_latency_ms: int | None = None


WorkloadConfig = (
    SingleJoinClustersWorkloadConfig
    | CrossProductWorkloadConfig
    | StringJoinsWorkloadConfig
    | ExternalWorkloadConfig
)


@dataclass(frozen=True)
class BenchmarkOptions:
    repetitions: int
    threads: int | None
    prepare_duckdb_binary: str
    collect_profiles: bool


@dataclass(frozen=True)
class MicrobenchmarkConfig:
    system_name: str
    domain_size: int = 100_000_000
    total_keys: int = 1_000_000
    min_clusters: int = 2
    max_clusters: int = 524_288
    probe_count: int = 1_000_000
    queries_per_cluster_count: int = 100_000
    repetitions: int = 5
    seed: int = 42
    grafite_bpk: float = 12.0
    diva_bpk: float = 16.0
    max_fpr: float = 0.001
    required_filters: tuple[str, ...] = ("bloom", "prf_uncompressed", "prf")
    aggregate: str = "median"
    range_query_kind: str = "empty"


def parse_microbenchmark_config(raw: object) -> MicrobenchmarkConfig | None:
    if raw is None:
        return None
    obj = _require_dict(raw, "microbenchmarks")
    defaults = MicrobenchmarkConfig(system_name="")
    unknown = set(obj) - set(defaults.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown microbenchmarks options: {', '.join(sorted(unknown))}")
    values = {"system_name": _require_string(obj.get("system_name"), "microbenchmarks.system_name")}
    for key in ("domain_size", "total_keys", "min_clusters", "max_clusters", "probe_count",
                "queries_per_cluster_count", "repetitions", "seed"):
        values[key] = _require_int(obj.get(key, getattr(defaults, key)), f"microbenchmarks.{key}")
        if values[key] < (0 if key == "seed" else 1) or values[key] >= 2**64:
            raise ValueError(f"microbenchmarks.{key} is outside the supported unsigned integer range")
    for key in ("grafite_bpk", "diva_bpk", "max_fpr"):
        values[key] = _as_float(obj.get(key, getattr(defaults, key)), f"microbenchmarks.{key}")
        if not math.isfinite(values[key]):
            raise ValueError(f"microbenchmarks.{key} must be finite")
    values["required_filters"] = tuple(
        _require_string(item, "microbenchmarks.required_filters entry")
        for item in _require_list(obj.get("required_filters", list(defaults.required_filters)),
                                  "microbenchmarks.required_filters")
    )
    for key in ("aggregate", "range_query_kind"):
        values[key] = _require_string(obj.get(key, getattr(defaults, key)), f"microbenchmarks.{key}")
    config = MicrobenchmarkConfig(**values)
    if config.total_keys >= config.domain_size:
        raise ValueError("microbenchmarks.total_keys must be less than domain_size")
    if (config.min_clusters < 2 or config.max_clusters < config.min_clusters
            or any(n & (n - 1) for n in (config.min_clusters, config.max_clusters))
            or config.max_clusters > config.total_keys
            or config.max_clusters - 1 > config.domain_size - config.total_keys):
        raise ValueError("microbenchmarks cluster bounds must be feasible powers of two, starting at 2 or higher")
    if config.grafite_bpk <= 2 or config.diva_bpk <= 1 or not 0 <= config.max_fpr <= 1:
        raise ValueError("invalid microbenchmarks bits-per-key or max_fpr")
    if config.seed + config.repetitions - 1 >= 2**64:
        raise ValueError("microbenchmarks repetition seeds overflow uint64")
    if (len(set(config.required_filters)) != len(config.required_filters)
            or set(config.required_filters) - {"bloom", "grafite", "diva", "prf_uncompressed", "prf"}):
        raise ValueError("microbenchmarks.required_filters contains unknown or duplicate filters")
    if config.aggregate not in {"mean", "median"}:
        raise ValueError("microbenchmarks.aggregate must be mean or median")
    if config.range_query_kind != "empty":
        raise ValueError("the comparison panel requires range_query_kind='empty'; other kinds remain in the artifacts")
    return config


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    experiment_name: str
    output_root: Path
    overwrite_output: bool
    workloads: tuple[WorkloadConfig, ...]
    systems: tuple[SystemConfig, ...]
    benchmark: BenchmarkOptions
    microbenchmarks: MicrobenchmarkConfig | None = None

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
        remote_http_parquet_latency_ms=(
            None
            if params.get("remote_http_parquet_latency_ms") is None
            else _require_int(params.get("remote_http_parquet_latency_ms"), "remote_http_parquet_latency_ms")
        ),
    )


_CROSS_PRODUCT_SWEEP_KINDS = {
    "dimension_cardinality",
    "selectivity",
    "key_domain_size",
    "build_key_multiplicity",
    "join_count",
    "cluster_count",
    "row_group_size",
    "probe_row_group_size",
}

_CROSS_PRODUCT_QUERY_TYPES = {"range", "shoulder", "seasonal", "scattered"}
_CROSS_PRODUCT_PROBE_LAYOUTS = {"ordered", "uniform"}


def _parse_cross_product_query_types(
    raw: object,
    *,
    default_probe_layout: str,
    sweep_kind: str,
) -> tuple[CrossProductQueryTypeConfig, ...]:
    if sweep_kind == "cluster_count":
        if raw not in (None, []):
            raise ValueError("query_types cannot be set for a cluster_count sweep")
        return ()
    query_types_raw = _require_list(raw, "query_types")
    if not query_types_raw:
        raise ValueError("query_types must not be empty")
    result: list[CrossProductQueryTypeConfig] = []
    seen: set[tuple[str, int | None, tuple[str, ...]]] = set()
    for index, query_type_raw in enumerate(query_types_raw, start=1):
        query_type = _require_dict(query_type_raw, f"query_types[{index}]")
        kind = _require_string(query_type.get("kind"), f"query_types[{index}].kind")
        if kind not in _CROSS_PRODUCT_QUERY_TYPES:
            raise ValueError(f"unsupported cross-product query type: {kind}")
        cluster_count_raw = query_type.get("cluster_count")
        if kind == "seasonal":
            cluster_count = (
                10
                if cluster_count_raw is None
                else _require_int(cluster_count_raw, f"query_types[{index}].cluster_count")
            )
            if cluster_count < 2:
                raise ValueError("seasonal cluster_count must be at least 2")
        else:
            if cluster_count_raw is not None:
                raise ValueError(f"cluster_count is only valid for seasonal query types, not {kind}")
            cluster_count = None
        probe_layouts_raw = query_type.get("probe_layouts")
        if probe_layouts_raw is None:
            probe_layouts = (default_probe_layout,)
        else:
            probe_layouts = tuple(
                _require_string(value, f"query_types[{index}].probe_layouts entry")
                for value in _require_list(
                    probe_layouts_raw,
                    f"query_types[{index}].probe_layouts",
                )
            )
            if not probe_layouts:
                raise ValueError(f"query_types[{index}].probe_layouts must not be empty")
        if len(set(probe_layouts)) != len(probe_layouts):
            raise ValueError(f"query_types[{index}].probe_layouts must be distinct")
        unsupported_layouts = set(probe_layouts) - _CROSS_PRODUCT_PROBE_LAYOUTS
        if unsupported_layouts:
            raise ValueError(
                "unsupported probe layouts: " + ", ".join(sorted(unsupported_layouts))
            )
        signature = (kind, cluster_count, probe_layouts)
        if signature in seen:
            raise ValueError(f"duplicate query type definition at query_types[{index}]")
        seen.add(signature)
        result.append(
            CrossProductQueryTypeConfig(
                kind=kind,
                cluster_count=cluster_count,
                probe_layouts=probe_layouts,
            )
        )
    return tuple(result)


def _parse_cross_product_definition(params: dict) -> CrossProductExperimentDefinition:
    fixed_raw = _require_dict(params.get("fixed"), "fixed")
    sweep_raw = _require_dict(params.get("sweep"), "sweep")
    sweep_kind = _require_string(sweep_raw.get("kind"), "sweep.kind")
    if sweep_kind not in _CROSS_PRODUCT_SWEEP_KINDS:
        raise ValueError(f"unsupported cross-product sweep kind: {sweep_kind}")
    if sweep_kind == "probe_row_group_size":
        sweep_kind = "row_group_size"
    raw_values = _require_list(sweep_raw.get("values"), "sweep.values")
    if not raw_values:
        raise ValueError("sweep.values must not be empty")
    if sweep_kind == "selectivity":
        values: tuple[int | float, ...] = tuple(
            _as_float(value, "sweep.values entry") for value in raw_values
        )
    else:
        values = tuple(_require_int(value, "sweep.values entry") for value in raw_values)
    if len(set(values)) != len(values):
        raise ValueError("sweep.values must be distinct")
    cardinality_mode = (
        None
        if sweep_raw.get("cardinality_mode") is None
        else _require_string(sweep_raw.get("cardinality_mode"), "sweep.cardinality_mode")
    )
    if sweep_kind == "build_key_multiplicity":
        cardinality_mode = cardinality_mode or "fixed_dimension_cardinality"
        if cardinality_mode != "fixed_dimension_cardinality":
            raise ValueError(
                "build_key_multiplicity currently requires "
                "cardinality_mode='fixed_dimension_cardinality'"
            )
    elif sweep_kind == "key_domain_size":
        if cardinality_mode not in (None, "fixed_selected_build_and_probe"):
            raise ValueError("unsupported key_domain_size cardinality_mode")
    elif cardinality_mode is not None:
        raise ValueError("sweep.cardinality_mode is only valid for multiplicity or density")

    key_domain_size_raw = fixed_raw.get("key_domain_size")
    if key_domain_size_raw in (None, "dimension_cardinality"):
        key_domain_size = None
    elif key_domain_size_raw == "dimension_distinct_keys":
        key_domain_size = "dimension_distinct_keys"
    else:
        key_domain_size = _require_int(key_domain_size_raw, "fixed.key_domain_size")
    probe_layout = _require_string(
        fixed_raw.get("probe_layout", "uniform"),
        "fixed.probe_layout",
    )
    if probe_layout not in _CROSS_PRODUCT_PROBE_LAYOUTS:
        raise ValueError(f"unsupported probe layout: {probe_layout}")
    fixed = CrossProductFixedParameters(
        fact_rows=_require_int(fixed_raw.get("fact_rows"), "fixed.fact_rows"),
        dimension_cardinality=_require_int(
            fixed_raw.get("dimension_cardinality"),
            "fixed.dimension_cardinality",
        ),
        key_domain_size=key_domain_size,
        selectivity=_as_float(fixed_raw.get("selectivity"), "fixed.selectivity"),
        join_count=_require_int(fixed_raw.get("join_count", 1), "fixed.join_count"),
        build_key_multiplicity=_require_int(
            fixed_raw.get("build_key_multiplicity", 1),
            "fixed.build_key_multiplicity",
        ),
        probe_layout=probe_layout,
        row_group_size=_require_int(
            fixed_raw.get("row_group_size", 122_880),
            "fixed.row_group_size",
        ),
        dummy_string_cols=_require_int(
            fixed_raw.get("dummy_string_cols", 2),
            "fixed.dummy_string_cols",
        ),
        dummy_string_len=_require_int(
            fixed_raw.get("dummy_string_len", 24),
            "fixed.dummy_string_len",
        ),
        seed=_require_int(fixed_raw.get("seed", 0), "fixed.seed"),
    )
    query_types = _parse_cross_product_query_types(
        params.get("query_types"),
        default_probe_layout=fixed.probe_layout,
        sweep_kind=sweep_kind,
    )
    return CrossProductExperimentDefinition(
        fixed=fixed,
        sweep=CrossProductSweepConfig(
            kind=sweep_kind,
            values=values,
            cardinality_mode=cardinality_mode,
        ),
        query_types=query_types,
    )


def _parse_cross_product(raw: dict, name: str) -> CrossProductWorkloadConfig:
    params = _require_dict(raw.get("params"), f"workloads[{name}].params")
    storage_raw = params.get("storage", {})
    storage = _require_dict(storage_raw, "storage")
    storage_kind = _require_string(storage.get("kind", "duckdb"), "storage.kind")
    if storage_kind not in {"duckdb", "ducklake"}:
        raise ValueError(f"unsupported cross-product storage kind: {storage_kind}")
    parquet_file_size_mb = _require_int(
        storage.get("parquet_file_size_mb", 200),
        "storage.parquet_file_size_mb",
    )
    if parquet_file_size_mb <= 0:
        raise ValueError("storage.parquet_file_size_mb must be positive")
    ducklake_http_latency_ms = (
        None
        if storage.get("http_latency_ms") is None
        else _require_int(storage.get("http_latency_ms"), "storage.http_latency_ms")
    )
    if ducklake_http_latency_ms is not None:
        if storage_kind != "ducklake":
            raise ValueError("storage.http_latency_ms requires storage.kind='ducklake'")
        if ducklake_http_latency_ms < 0:
            raise ValueError("storage.http_latency_ms must be non-negative")
    remote_http_parquet_latency_ms = (
        None
        if params.get("remote_http_parquet_latency_ms") is None
        else _require_int(
            params.get("remote_http_parquet_latency_ms"),
            "remote_http_parquet_latency_ms",
        )
    )
    if storage_kind != "duckdb" and remote_http_parquet_latency_ms is not None:
        raise ValueError(
            "cross-product storage.kind cannot be combined with remote_http_parquet_latency_ms"
        )
    experiment_definition = (
        _parse_cross_product_definition(params)
        if "fixed" in params or "sweep" in params
        else None
    )
    preparation_raw = _require_dict(params.get("preparation", {}), "preparation")
    if preparation_raw and (
        experiment_definition is None
        or experiment_definition.sweep.cardinality_mode != "fixed_selected_build_and_probe"
    ):
        raise ValueError("preparation options currently require fixed_selected_build_and_probe density")
    unknown = set(preparation_raw) - set(DensityPreparationOptions.__dataclass_fields__)
    if unknown:
        raise ValueError("unknown preparation options: " + ", ".join(sorted(unknown)))
    batch_rows = _require_int(preparation_raw.get("batch_rows", 983_040), "preparation.batch_rows")
    threads = _require_int(preparation_raw.get("threads", 4), "preparation.threads")
    if batch_rows <= 0 or batch_rows % 122_880:
        raise ValueError("preparation.batch_rows must be a positive multiple of 122880")
    if threads <= 0:
        raise ValueError("preparation.threads must be positive")
    preparation = DensityPreparationOptions(
        batch_rows=batch_rows, threads=threads,
        **{key: _require_string(preparation_raw[key], f"preparation.{key}")
           for key in ("memory_limit", "max_temp_directory_size", "temp_directory")
           if key in preparation_raw},
    )
    if experiment_definition is None:
        build_sizes = tuple(_require_int(item, "build_sizes entry") for item in _require_list(params.get("build_sizes"), "build_sizes"))
        selectivities = tuple(_as_float(item, "selectivities entry") for item in _require_list(params.get("selectivities"), "selectivities"))
        predicate_types = tuple(_require_string(item, "predicate_types entry") for item in _require_list(params.get("predicate_types"), "predicate_types"))
        probe_clusterings = tuple(_require_string(item, "probe_clusterings entry") for item in _require_list(params.get("probe_clusterings"), "probe_clusterings"))
        fact_rows = _require_int(params.get("fact_rows"), "fact_rows")
        dummy_string_cols = _require_int(params.get("dummy_string_cols", 2), "dummy_string_cols")
        dummy_string_len = _require_int(params.get("dummy_string_len", 24), "dummy_string_len")
        seed = _require_int(params.get("seed", 0), "seed")
    else:
        build_sizes = ()
        selectivities = ()
        predicate_types = ()
        probe_clusterings = ()
        fact_rows = experiment_definition.fixed.fact_rows
        dummy_string_cols = experiment_definition.fixed.dummy_string_cols
        dummy_string_len = experiment_definition.fixed.dummy_string_len
        seed = experiment_definition.fixed.seed
    return CrossProductWorkloadConfig(
        kind="cross_product",
        name=name,
        build_sizes=build_sizes,
        selectivities=selectivities,
        predicate_types=predicate_types,
        probe_clusterings=probe_clusterings,
        fact_rows=fact_rows,
        dummy_string_cols=dummy_string_cols,
        dummy_string_len=dummy_string_len,
        seed=seed,
        remote_http_parquet_latency_ms=remote_http_parquet_latency_ms,
        storage_kind=storage_kind,
        parquet_file_size_mb=parquet_file_size_mb,
        ducklake_http_latency_ms=ducklake_http_latency_ms,
        experiment_definition=experiment_definition,
        preparation=preparation,
        post_variant_setup_sql=tuple(
            _normalize_sql_statement(_require_string(statement, "post_variant_setup_sql entry"))
            for statement in _require_list(
                raw.get("post_variant_setup_sql", []),
                f"workloads[{name}].post_variant_setup_sql",
            )
        ),
    )


def _parse_string_joins(raw: dict, name: str, base_dir: Path) -> StringJoinsWorkloadConfig:
    params = _require_dict(raw.get("params"), f"workloads[{name}].params")
    airport_csv_raw = _require_string(params.get("airport_csv_path"), "airport_csv_path")
    ranges_raw = _require_list(params.get("uuid_selected_ranges"), "uuid_selected_ranges")
    selected_ranges: list[tuple[int, int]] = []
    for index, range_raw in enumerate(ranges_raw, start=1):
        range_values = _require_list(range_raw, f"uuid_selected_ranges[{index}]")
        if len(range_values) != 2:
            raise ValueError(f"uuid_selected_ranges[{index}] must contain [start, end]")
        selected_ranges.append(
            (
                _require_int(range_values[0], f"uuid_selected_ranges[{index}][0]"),
                _require_int(range_values[1], f"uuid_selected_ranges[{index}][1]"),
            )
        )
    return StringJoinsWorkloadConfig(
        kind="string_joins",
        name=name,
        airport_csv_path=(base_dir / airport_csv_raw).resolve(),
        uuid_dim_size=_require_int(params.get("uuid_dim_size"), "uuid_dim_size"),
        uuid_timestamp_start_ms=_require_int(
            params.get("uuid_timestamp_start_ms"), "uuid_timestamp_start_ms"
        ),
        uuid_timestamp_step_ms=_require_int(
            params.get("uuid_timestamp_step_ms"), "uuid_timestamp_step_ms"
        ),
        uuid_selected_ranges=tuple(selected_ranges),
        fact_rows=_require_int(params.get("fact_rows"), "fact_rows"),
        dummy_string_cols=_require_int(params.get("dummy_string_cols", 2), "dummy_string_cols"),
        dummy_string_len=_require_int(params.get("dummy_string_len", 24), "dummy_string_len"),
        seed=_require_int(params.get("seed", 0), "seed"),
        remote_http_parquet_latency_ms=(
            None
            if params.get("remote_http_parquet_latency_ms") is None
            else _require_int(params.get("remote_http_parquet_latency_ms"), "remote_http_parquet_latency_ms")
        ),
    )


def _parse_external(raw: dict, name: str, base_dir: Path) -> ExternalWorkloadConfig:
    query_names_raw = raw.get("query_names")
    query_names = (
        None
        if query_names_raw is None
        else tuple(
            _require_string(item, "query_names entry")
            for item in _require_list(query_names_raw, "query_names")
        )
    )
    setup_sql = tuple(
        _normalize_sql_statement(_require_string(statement, "setup_sql entry"))
        for statement in _require_list(raw.get("setup_sql", []), "setup_sql")
    )
    storage = _require_dict(raw.get("storage", {}), f"workloads[{name}].storage")
    storage_kind = _require_string(
        storage.get("kind", "duckdb"), f"workloads[{name}].storage.kind"
    )
    if storage_kind not in {"duckdb", "ducklake"}:
        raise ValueError(f"unsupported external storage kind: {storage_kind}")
    parquet_file_size_mb = _require_int(
        storage.get("parquet_file_size_mb", 200),
        f"workloads[{name}].storage.parquet_file_size_mb",
    )
    if parquet_file_size_mb <= 0:
        raise ValueError("external storage.parquet_file_size_mb must be positive")
    ducklake_http_latency_ms = (
        None
        if storage.get("http_latency_ms") is None
        else _require_int(
            storage.get("http_latency_ms"),
            f"workloads[{name}].storage.http_latency_ms",
        )
    )
    if ducklake_http_latency_ms is not None:
        if storage_kind != "ducklake":
            raise ValueError("external storage.http_latency_ms requires kind='ducklake'")
        if ducklake_http_latency_ms < 0:
            raise ValueError("external storage.http_latency_ms must be non-negative")
    return ExternalWorkloadConfig(
        kind="external",
        name=name,
        source_dir=(base_dir / _require_string(raw.get("source_dir"), "source_dir")).resolve(),
        query_names=query_names,
        setup_sql=setup_sql,
        storage_kind=storage_kind,
        parquet_file_size_mb=parquet_file_size_mb,
        ducklake_http_latency_ms=ducklake_http_latency_ms,
    )


def _parse_workloads(raw: object, base_dir: Path) -> tuple[WorkloadConfig, ...]:
    workloads_raw = _require_list(raw, "workloads")
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
        elif kind == "string_joins":
            workloads.append(_parse_string_joins(workload_obj, name, base_dir))
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
        workload_names_raw = system_obj.get("workload_names")
        workload_names = (
            None
            if workload_names_raw is None
            else tuple(
                _require_string(value, f"systems[{index}].workload_names entry")
                for value in _require_list(
                    workload_names_raw, f"systems[{index}].workload_names"
                )
            )
        )
        if workload_names is not None and len(set(workload_names)) != len(workload_names):
            raise ValueError(f"systems[{index}].workload_names contains duplicates")
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
                    workload_names=workload_names,
                )
            )
        elif kind == "git":
            if "repo" in system_obj or "ref" in system_obj:
                raise ValueError(
                    "git systems no longer accept repo/ref in config; use the local duckdb checkout "
                    "and the runtime-only --duckdb-github-branch flag instead"
                )
            systems.append(
                GitSystemConfig(
                    kind="git",
                    name=name,
                    c_compiler=(
                        None
                        if system_obj.get("c_compiler") is None
                        else _require_string(system_obj.get("c_compiler"), "c_compiler")
                    ),
                    cxx_compiler=(
                        None
                        if system_obj.get("cxx_compiler") is None
                        else _require_string(system_obj.get("cxx_compiler"), "cxx_compiler")
                    ),
                    variants=variants,
                    checkout_dir=(
                        None
                        if system_obj.get("checkout_dir") is None
                        else (base_dir / _require_string(
                            system_obj.get("checkout_dir"), "checkout_dir"
                        )).resolve()
                    ),
                    workload_names=workload_names,
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
        microbenchmarks=parse_microbenchmark_config(raw.get("microbenchmarks")),
    )
    if not config.workloads and config.microbenchmarks is None:
        raise ValueError("workloads must not be empty unless microbenchmarks are configured")
    if config.microbenchmarks and config.microbenchmarks.system_name not in {s.name for s in config.systems}:
        raise ValueError("microbenchmarks.system_name must reference a configured system")
    if config.benchmark.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if config.benchmark.threads is not None and config.benchmark.threads <= 0:
        raise ValueError("threads must be positive when set")
    known_workloads = {workload.name for workload in config.workloads}
    for system in config.systems:
        if system.workload_names is None:
            continue
        unknown = set(system.workload_names) - known_workloads
        if unknown:
            raise ValueError(
                f"system {system.name!r} references unknown workloads: "
                + ", ".join(sorted(unknown))
            )
    return config
